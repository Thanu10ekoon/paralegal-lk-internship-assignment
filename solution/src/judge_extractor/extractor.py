from __future__ import annotations

import json
import logging
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from pypdf import PdfReader


HEADER_KEYWORDS = ("coram", "present", "before", "bench")
JUDGE_SUFFIX_PATTERN = r"(?:c\.?\s*j\.?|jj\.?|\bj\b\s*\.?|chief\s*justice|judge)"
BENCH_SCAN_LIMIT = 260
BENCH_WINDOW_FORWARD = 8
BENCH_WINDOW_BACKWARD_PRESENT_CORAM = 3
TOP_WINDOW_SIZE = 40
HEADER_LINE_MAX_LEN = 90
AUTHOR_SCAN_LIMIT = 450
AUTHOR_CONFIDENCE_MIN_SCORE = 10

HEADER_ANCHOR_RE = re.compile(r"^\s*(before|coram|present|bench)\b", re.IGNORECASE)
HEADER_COLON_RE = re.compile(r"\b(before|coram|present|bench)\b\s*:", re.IGNORECASE)
PRESENT_CORAM_ANCHOR_RE = re.compile(r"^\s*(present|coram)\b", re.IGNORECASE)
STOP_BENCH_SCAN_RE = re.compile(
    r"\b(counsel|councel|argued|decided|written submissions|petitioner|respondent)\b",
    re.IGNORECASE,
)
HEADER_SKIP_RE = re.compile(r"\b(counsel|argued|decided|written submissions)\b", re.IGNORECASE)
NON_BENCH_CONTENT_RE = re.compile(
    r"\b(petitioner|respondent|application|article|section|court|vs\.?)\b",
    re.IGNORECASE,
)
DIGIT_RE = re.compile(r"\d")
JUDGE_SUFFIX_RE = re.compile(JUDGE_SUFFIX_PATTERN, re.IGNORECASE)
SUFFIX_NEAR_END_RE = re.compile(rf"{JUDGE_SUFFIX_PATTERN}\s*[.)]*\s*$", re.IGNORECASE)
CHIEF_JUSTICE_RE = re.compile(r"\b(c\.?\s*j\.?|chief\s*justice)\b", re.IGNORECASE)

AUTHOR_CUE_PATTERNS = [
    re.compile(r"\b(delivered|pronounced|authored|written)\s+by\b", re.IGNORECASE),
    re.compile(r"\bjudg(?:e)?ment\s+by\b", re.IGNORECASE),
    re.compile(r"\bper\s+", re.IGNORECASE),
]

logger = logging.getLogger(__name__)

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


@dataclass(frozen=True)
class LineRecord:
    page_index: int
    line_index: int
    text: str


def _read_pdf_text(pdf_path: Path) -> tuple[str, list[str]]:
    reader = PdfReader(str(pdf_path))
    page_texts: list[str] = []
    for page in reader.pages:
        page_texts.append(page.extract_text() or "")
    full_text = "\n".join(page_texts)

    if len(full_text.strip()) < 80:
        logger.info("OCR fallback triggered for %s", pdf_path.name)
        ocr_text = _read_pdf_text_with_ocr(pdf_path)
        if ocr_text.strip():
            return ocr_text, [ocr_text]
        logger.warning("OCR fallback returned no text for %s", pdf_path.name)

    return full_text, page_texts


@lru_cache(maxsize=1)
def _get_ocr_engine() -> object:
    from rapidocr_onnxruntime import RapidOCR

    return RapidOCR()


def _read_pdf_text_with_ocr(pdf_path: Path) -> str:
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    try:
        import fitz
        import numpy as np
    except ImportError as exc:
        logger.warning(
            "OCR dependencies are unavailable (%s). Continuing without OCR for %s.",
            exc,
            pdf_path.name,
        )
        return ""

    try:
        ocr_engine = _get_ocr_engine()
    except Exception as exc:  # pragma: no cover - defensive runtime guard
        logger.warning("Failed to initialize OCR engine for %s: %s", pdf_path.name, exc)
        return ""

    doc = fitz.open(str(pdf_path))
    lines: list[str] = []
    page_failure_logged = False

    for page in doc:
        pix = page.get_pixmap(dpi=120)
        image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        try:
            result, _ = ocr_engine(image)
        except Exception as exc:  # pragma: no cover - runtime/device-specific
            if not page_failure_logged:
                logger.warning("OCR failed on one or more pages for %s: %s", pdf_path.name, exc)
                page_failure_logged = True
            continue
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
    name = re.sub(r"^(judg(?:e)?ment|delivered|pronounced|authored|written)\s+by\s+", "", name, flags=re.IGNORECASE)
    name = re.sub(r"^per\s+", "", name, flags=re.IGNORECASE)
    name = re.sub(r"\b(hon'?ble|honourable|honorable|mr\.?|mrs\.?|ms\.?)\b", "", name, flags=re.IGNORECASE)
    name = re.sub(r"\b(chief\s*justice|justice)\b", "", name, flags=re.IGNORECASE)
    name = re.sub(rf"\s*,?\s*{JUDGE_SUFFIX_PATTERN}\s*$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"\s+", " ", name).strip(" ,.;:-")
    if not name:
        return ""

    parts: list[str] = []
    particles = {"de", "van", "von", "bin", "da", "dos", "del", "di", "la", "le"}

    def _title_with_apostrophe_hyphen(token: str) -> str:
        chunks = re.split(r"([-'])", token)
        out: list[str] = []
        for chunk in chunks:
            if chunk in {"-", "'"}:
                out.append(chunk)
                continue
            if not chunk:
                continue
            if len(chunk) > 2 and chunk.lower().startswith("mc"):
                out.append("Mc" + chunk[2:].capitalize())
            else:
                out.append(chunk.capitalize())
        return "".join(out)

    for token in name.split(" "):
        if token.lower() in particles:
            parts.append(token.lower())
            continue
        if token != token.lower() and token != token.upper():
            parts.append(token)
            continue
        if re.fullmatch(r"[A-Za-z]\.?", token):
            parts.append(token[0].upper() + ".")
        elif re.fullmatch(r"[A-Za-z](?:\.[A-Za-z])+\.?[A-Za-z]+", token):
            stripped = token.strip(".")
            segs = stripped.split(".")
            initials = ".".join(s.upper() for s in segs[:-1] if s)
            surname = _title_with_apostrophe_hyphen(segs[-1])
            parts.append(f"{initials}.{surname}" if initials else surname)
        elif "." in token and token.replace(".", "").isalpha() and len(token) <= 8:
            parts.append(token.upper())
        else:
            parts.append(_title_with_apostrophe_hyphen(token))
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

    if len(line) > 140:
        return []

    p1 = re.finditer(
        r"(?:hon'?ble\s+)?(?:mr\.?|mrs\.?|ms\.?)?\s*(?:justice|chief\s*justice)\s+([A-Za-z][A-Za-z .'-]{1,50})",
        line,
        re.IGNORECASE,
    )
    for match in p1:
        cleaned = _clean_name(match.group(1))
        if cleaned and _is_likely_person_name(cleaned):
            names.append(cleaned)

    if SUFFIX_NEAR_END_RE.search(line):
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
    if HEADER_SKIP_RE.search(line):
        return []
    if len(line) > HEADER_LINE_MAX_LEN:
        return []
    if NON_BENCH_CONTENT_RE.search(line):
        return []
    if DIGIT_RE.search(line):
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
    lines: list[str] = []
    for line in full_text.splitlines():
        normalized = _normalize_line(line)
        if normalized:
            lines.append(normalized)
    return lines


def _collect_line_records(page_texts: list[str]) -> list[LineRecord]:
    records: list[LineRecord] = []
    for page_index, page_text in enumerate(page_texts):
        line_index = 0
        for line in page_text.splitlines():
            normalized = _normalize_line(line)
            if not normalized:
                continue
            records.append(LineRecord(page_index=page_index, line_index=line_index, text=normalized))
            line_index += 1
    return records


def _extract_bench(records: list[LineRecord]) -> list[str]:
    bench_names: list[str] = []
    seen: set[str] = set()
    lines = [r.text for r in records]

    scan_limit = min(len(lines), BENCH_SCAN_LIMIT)

    for idx, line in enumerate(lines[:scan_limit]):
        is_header_anchor = bool(HEADER_ANCHOR_RE.search(line) or HEADER_COLON_RE.search(line))
        if is_header_anchor:
            backward_scan = BENCH_WINDOW_BACKWARD_PRESENT_CORAM if PRESENT_CORAM_ANCHOR_RE.search(line) else 0
            start = max(0, idx - backward_scan)
            window = lines[start : min(idx + BENCH_WINDOW_FORWARD, scan_limit)]
            for offset, candidate_line in enumerate(window):
                if STOP_BENCH_SCAN_RE.search(candidate_line):
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

    top_window = lines[:TOP_WINDOW_SIZE]
    for line in top_window:
        judge_marker_count = len(JUDGE_SUFFIX_RE.findall(line))
        if judge_marker_count >= 2:
            for name in _extract_names_from_line(line):
                key = _name_key(name)
                if key not in seen:
                    seen.add(key)
                    bench_names.append(name)

    return bench_names


def _extract_author(records: list[LineRecord], bench: list[str]) -> list[str]:
    if not bench:
        return []

    lines = [r.text for r in records]
    bench_map = {_name_key(name): name for name in bench}
    score: dict[str, int] = defaultdict(int)
    cue_evidence = False
    signature_evidence = False

    max_page = max((r.page_index for r in records), default=0)

    for idx, record in enumerate(records):
        line = record.text
        lower_line = line.lower()
        names_in_line = _extract_names_from_line(line)
        if not names_in_line:
            continue

        in_last_third = len(lines) >= 50 and idx >= int(len(lines) * 0.66)
        in_last_page = record.page_index >= max(0, max_page - 1)

        cue_hit = any(p.search(line) for p in AUTHOR_CUE_PATTERNS)
        if cue_hit:
            cue_evidence = True
        signature_hit = bool(JUDGE_SUFFIX_RE.search(line))
        if signature_hit:
            signature_evidence = True
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
            if in_last_page:
                score[key] += 1

    for idx, record in enumerate(records[:AUTHOR_SCAN_LIMIT]):
        line = record.text
        if any(k in line.lower() for k in HEADER_KEYWORDS):
            continue
        if STOP_BENCH_SCAN_RE.search(line):
            continue
        names_in_line = _extract_names_from_line(line)
        if not names_in_line:
            continue
        if JUDGE_SUFFIX_RE.search(line):
            for name in names_in_line:
                key = _name_key(name)
                if key in bench_map:
                    score[key] += 7 if idx < 300 else 3

    if score:
        max_score = max(score.values())
        low_confidence_no_cue = max_score < AUTHOR_CONFIDENCE_MIN_SCORE and not cue_evidence
        selected = [bench_map[k] for k, v in score.items() if v == max_score and v > 0]
        if selected:
            if low_confidence_no_cue:
                selected = []
            if len(selected) > 1 and not cue_evidence:
                selected = []
            if selected:
                return sorted(selected, key=lambda n: bench.index(n))

    # Structural fallback: prefer a single Chief Justice line when explicit cue scoring is absent.
    if not cue_evidence:
        for record in records[:AUTHOR_SCAN_LIMIT]:
            if not CHIEF_JUSTICE_RE.search(record.text):
                continue
            names = _extract_names_from_line(record.text)
            matched = [bench_map[_name_key(name)] for name in names if _name_key(name) in bench_map]
            if matched:
                return [matched[0]]

    # Conservative fallback: only return signature-matched judges from the final page region.
    for record in reversed(records):
        if not JUDGE_SUFFIX_RE.search(record.text):
            continue
        if record.page_index != max_page:
            continue
        names = _extract_names_from_line(record.text)
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

    if signature_evidence and len(bench) == 1:
        return bench.copy()

    return []


def extract_from_pdf(pdf_path: Path) -> ExtractionResult:
    full_text, page_texts = _read_pdf_text(pdf_path)
    line_records = _collect_line_records(page_texts) if page_texts else [
        LineRecord(page_index=0, line_index=i, text=t) for i, t in enumerate(_collect_lines(full_text))
    ]
    logger.info("Processing %s: %d normalized lines", pdf_path.name, len(line_records))
    bench = _extract_bench(line_records)
    logger.info("Bench candidates for %s: %d", pdf_path.name, len(bench))
    author_judge = _extract_author(line_records, bench)
    logger.info("Author candidates for %s: %d", pdf_path.name, len(author_judge))

    return ExtractionResult(
        source_file=pdf_path.name,
        bench=bench,
        author_judge=author_judge,
    )


def process_folder(data_dir: Path, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    for pdf_path in sorted(data_dir.glob("*.pdf")):
        logger.info("Processing file: %s", pdf_path.name)
        result = extract_from_pdf(pdf_path)
        out_path = output_dir / f"{pdf_path.stem}.json"
        out_path.write_text(
            json.dumps(result.to_json_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        generated.append(out_path)

    return generated
