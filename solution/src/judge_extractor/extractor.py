from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


HEADER_KEYWORDS = ("coram", "present", "before", "bench")
JUDGE_SUFFIX_PATTERN = r"(?:c\.?\s*j\.?|jj\.?|\bj\b\s*\.?|chief\s*justice|judge)"
AUTHOR_CUE_PATTERNS = [
    re.compile(r"\b(delivered|pronounced|authored|written)\s+by\b", re.IGNORECASE),
    re.compile(r"\bjudg(?:e)?ment\s+by\b", re.IGNORECASE),
    re.compile(r"\bper\s+", re.IGNORECASE),
]

NON_NAME_TOKENS = {
    "and",
    "or",
    "the",
    "of",
    "for",
    "to",
    "in",
    "on",
    "before",
    "present",
    "court",
    "respondent",
    "respondents",
    "petitioner",
    "petitioners",
    "justice",
    "judge",
    "chief",
    "counsel",
    "section",
    "article",
    "that",
    "this",
    "shall",
    "be",
    "was",
    "were",
    "appeared",
    "pc",
}


@dataclass(frozen=True)
class ExtractionResult:
    source_file: str
    bench: list[str]
    author_judge: list[str]

    def to_json_dict(self) -> dict[str, object]:
        return {
            "source_file": self.source_file,
            "bench": self.bench,
            "author_judge": self.author_judge,
        }


def _read_pdf_text(pdf_path: Path) -> tuple[str, list[str]]:
    reader = PdfReader(str(pdf_path))
    page_texts: list[str] = []
    for page in reader.pages:
        page_texts.append(page.extract_text() or "")
    full_text = "\n".join(page_texts)

    if len(full_text.strip()) < 80:
        ocr_text = _read_pdf_text_with_ocr(pdf_path)
        if ocr_text.strip():
            return ocr_text, [ocr_text]

    return full_text, page_texts


def _read_pdf_text_with_ocr(pdf_path: Path) -> str:
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    import fitz
    import numpy as np
    from rapidocr_onnxruntime import RapidOCR

    ocr_engine = RapidOCR()
    doc = fitz.open(str(pdf_path))
    lines: list[str] = []

    for page in doc:
        pix = page.get_pixmap(dpi=170)
        image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        result, _ = ocr_engine(image)
        if not result:
            continue
        for row in result:
            text = _normalize_line(row[1])
            if text:
                lines.append(text)

    return "\n".join(lines)


def _normalize_line(line: str) -> str:
    line = line.replace("\u00a0", " ")
    line = re.sub(r"\s+", " ", line).strip()
    return line


def _name_key(name: str) -> str:
    return re.sub(r"[^a-z]", "", name.lower())


def _clean_name(name: str) -> str:
    name = name.strip(" ,.;:-")
    name = re.sub(r"^and\s+", "", name, flags=re.IGNORECASE)
    name = re.sub(r"\b(hon'?ble|honourable|honorable|mr\.?|mrs\.?|ms\.?)\b", "", name, flags=re.IGNORECASE)
    name = re.sub(r"\b(chief\s*justice|justice)\b", "", name, flags=re.IGNORECASE)
    name = re.sub(rf"\s*,?\s*{JUDGE_SUFFIX_PATTERN}\s*$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"\s+", " ", name).strip(" ,.;:-")
    if not name:
        return ""

    parts = []
    for token in name.split(" "):
        if re.fullmatch(r"[A-Za-z]\.?", token):
            parts.append(token[0].upper() + ".")
        elif re.fullmatch(r"[A-Za-z](?:\.[A-Za-z])+\.?[A-Za-z]+", token):
            stripped = token.strip(".")
            segs = stripped.split(".")
            initials = ".".join(s.upper() for s in segs[:-1] if s)
            surname = segs[-1].capitalize()
            parts.append(f"{initials}.{surname}" if initials else surname)
        elif "." in token and token.replace(".", "").isalpha() and len(token) <= 8:
            parts.append(token.upper())
        else:
            parts.append(token.capitalize())
    return " ".join(parts)


def _is_likely_person_name(name: str) -> bool:
    tokens = [t for t in re.split(r"\s+", name.strip()) if t]
    if not (1 <= len(tokens) <= 6):
        return False
    if any(any(ch.isdigit() for ch in t) for t in tokens):
        return False

    lowered = [re.sub(r"[^a-z]", "", t.lower()) for t in tokens]
    if any(t in NON_NAME_TOKENS for t in lowered if t):
        return False

    valid_token_count = 0
    alpha_count = 0
    for token in tokens:
        raw = token.strip(".,")
        if re.fullmatch(r"[A-Za-z]", raw):
            valid_token_count += 1
            alpha_count += 1
            continue
        if re.fullmatch(r"[A-Za-z]{2,}", raw) or re.fullmatch(r"[A-Za-z]\.[A-Za-z]?\.?", raw):
            valid_token_count += 1
            alpha_count += 1
            continue
        if re.fullmatch(r"[A-Za-z](?:\.[A-Za-z])+\.?[A-Za-z]+", raw):
            valid_token_count += 1
            alpha_count += 1
            continue
        if re.fullmatch(r"[A-Za-z]+\.", raw):
            valid_token_count += 1
            alpha_count += 1
            continue
    if valid_token_count < max(1, len(tokens) - 1):
        return False

    if len(tokens) == 1:
        only = re.sub(r"[^A-Za-z]", "", tokens[0])
        if len(only) <= 1:
            return False

    return alpha_count >= 1


def _extract_names_from_line(line: str) -> list[str]:
    names: list[str] = []

    p1 = re.finditer(
        r"(?:hon'?ble\s+)?(?:mr\.?|mrs\.?|ms\.?)?\s*(?:justice|chief\s*justice)\s+([A-Za-z][A-Za-z .'-]{1,50})",
        line,
        re.IGNORECASE,
    )
    for match in p1:
        cleaned = _clean_name(match.group(1))
        if cleaned and _is_likely_person_name(cleaned):
            names.append(cleaned)

    p2 = re.finditer(
        rf"([A-Za-z][A-Za-z .'-]{{0,50}}?)\s*[,.]?\s*(?:p\.?c\.?\s*,\s*)?{JUDGE_SUFFIX_PATTERN}",
        line,
        re.IGNORECASE,
    )
    for match in p2:
        cleaned = _clean_name(match.group(1))
        if cleaned and _is_likely_person_name(cleaned):
            names.append(cleaned)

    seen: set[str] = set()
    deduped: list[str] = []
    for name in names:
        k = _name_key(name)
        if k and k not in seen:
            seen.add(k)
            deduped.append(name)
    return deduped


def _extract_plain_names_from_header_line(line: str) -> list[str]:
    if re.search(r"\b(counsel|argued|decided|written submissions)\b", line, re.IGNORECASE):
        return []
    if len(line) > 90:
        return []
    if re.search(r"\b(petitioner|respondent|application|article|section|court|vs\.?)\b", line, re.IGNORECASE):
        return []
    if re.search(r"\d", line):
        return []

    text = line.replace("&", " and ")
    parts = re.split(r",|\band\b", text, flags=re.IGNORECASE)
    names: list[str] = []
    for part in parts:
        candidate = _clean_name(part)
        if not candidate:
            continue
        token_count = len(candidate.split())
        if token_count == 0 or token_count > 4:
            continue
        if _is_likely_person_name(candidate):
            names.append(candidate)
    return names


def _collect_lines(full_text: str) -> list[str]:
    return [_normalize_line(line) for line in full_text.splitlines() if _normalize_line(line)]


def _extract_bench(lines: list[str]) -> list[str]:
    bench_names: list[str] = []
    seen: set[str] = set()

    scan_limit = min(len(lines), 260)

    for idx, line in enumerate(lines[:scan_limit]):
        line_lc = line.lower()
        is_header_anchor = bool(
            re.search(r"^\s*(before|coram|present|bench)\b", line_lc)
            or re.search(r"\b(before|coram|present|bench)\b\s*:", line_lc)
        )
        if is_header_anchor:
            backward_scan = 3 if re.search(r"^\s*(present|coram)\b", line_lc) else 0
            start = max(0, idx - backward_scan)
            window = lines[start : min(idx + 8, scan_limit)]
            for offset, candidate_line in enumerate(window):
                if re.search(
                    r"\b(counsel|councel|argued|decided|written submissions|petitioner|respondent)\b",
                    candidate_line,
                    re.IGNORECASE,
                ):
                    break
                for name in _extract_names_from_line(candidate_line):
                    key = _name_key(name)
                    if key not in seen:
                        seen.add(key)
                        bench_names.append(name)
                if offset <= 5:
                    for name in _extract_plain_names_from_header_line(candidate_line):
                        key = _name_key(name)
                        if key not in seen:
                            seen.add(key)
                            bench_names.append(name)

    top_window = lines[:40]
    for line in top_window:
        judge_marker_count = len(
            re.findall(JUDGE_SUFFIX_PATTERN, line, flags=re.IGNORECASE)
        )
        if judge_marker_count >= 2:
            for name in _extract_names_from_line(line):
                key = _name_key(name)
                if key not in seen:
                    seen.add(key)
                    bench_names.append(name)

    return bench_names


def _extract_author(lines: list[str], bench: list[str]) -> list[str]:
    if not bench:
        return []

    bench_map = {_name_key(name): name for name in bench}
    score: dict[str, int] = defaultdict(int)
    cue_evidence = False

    for idx, line in enumerate(lines):
        lower_line = line.lower()
        names_in_line = _extract_names_from_line(line)
        if not names_in_line:
            continue

        in_last_third = idx >= int(len(lines) * 0.66)

        cue_hit = any(p.search(line) for p in AUTHOR_CUE_PATTERNS)
        if cue_hit:
            cue_evidence = True
        signature_hit = bool(re.search(JUDGE_SUFFIX_PATTERN, line, re.IGNORECASE))
        judgment_heading_hit = "judgment" in lower_line and len(line) < 120
        header_hit = any(k in lower_line for k in HEADER_KEYWORDS)

        for name in names_in_line:
            key = _name_key(name)
            if key not in bench_map:
                continue
            if cue_hit:
                score[key] += 5
            if signature_hit and not header_hit and idx > 30:
                score[key] += 4
            if judgment_heading_hit:
                score[key] += 3
            if in_last_third:
                score[key] += 1

    for idx, line in enumerate(lines[:450]):
        if any(k in line.lower() for k in HEADER_KEYWORDS):
            continue
        if re.search(r"\b(counsel|petitioner|respondent)\b", line, re.IGNORECASE):
            continue
        names_in_line = _extract_names_from_line(line)
        if not names_in_line:
            continue
        if re.search(JUDGE_SUFFIX_PATTERN, line, re.IGNORECASE):
            for name in names_in_line:
                key = _name_key(name)
                if key in bench_map:
                    score[key] += 7 if idx < 300 else 3

    if score:
        max_score = max(score.values())
        selected = [bench_map[k] for k, v in score.items() if v == max_score and v > 0]
        if selected:
            if len(selected) > 1 and not cue_evidence:
                return bench[:1]
            return sorted(selected, key=lambda n: bench.index(n))

    if len(bench) == 1:
        return bench.copy()

    for line in reversed(lines):
        names = _extract_names_from_line(line)
        if not names:
            continue
        matched = [bench_map[_name_key(name)] for name in names if _name_key(name) in bench_map]
        if matched:
            unique = []
            seen = set()
            for name in matched:
                k = _name_key(name)
                if k not in seen:
                    seen.add(k)
                    unique.append(name)
            if unique:
                return unique

    return bench[:1]


def extract_from_pdf(pdf_path: Path) -> ExtractionResult:
    full_text, _ = _read_pdf_text(pdf_path)
    lines = _collect_lines(full_text)
    bench = _extract_bench(lines)
    author_judge = _extract_author(lines, bench)

    return ExtractionResult(
        source_file=pdf_path.name,
        bench=bench,
        author_judge=author_judge,
    )


def process_folder(data_dir: Path, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    for pdf_path in sorted(data_dir.glob("*.pdf")):
        result = extract_from_pdf(pdf_path)
        out_path = output_dir / f"{pdf_path.stem}.json"
        out_path.write_text(json.dumps(result.to_json_dict(), indent=2), encoding="utf-8")
        generated.append(out_path)

    return generated
