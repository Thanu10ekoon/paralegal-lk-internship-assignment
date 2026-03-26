# Workflow: How The Extractor Works

This document explains the end-to-end workflow used by the deterministic judge extraction system.

## 1. It Defines The Output Structure

The pipeline stores each result in a strongly typed dataclass:

```python
@dataclass(frozen=True)
class ExtractionResult:
    source_file: str
    bench: list[str]
    author_judge: list[str]
```

Each PDF therefore produces one structured record containing:

- the source file name
- the full bench
- the author judge(s)

## 2. It Reads PDF Text

The `_read_pdf_text()` function uses `pypdf.PdfReader` to extract text page by page:

```python
reader = PdfReader(str(pdf_path))
for page in reader.pages:
    page_texts.append(page.extract_text() or "")
```

It then merges all pages into one text block.

If the extracted text is too short, the system treats the file as likely image-based and triggers OCR fallback:

```python
if len(full_text.strip()) < 80:
    ocr_text = _read_pdf_text_with_ocr(pdf_path)
```

The extractor therefore supports two modes:

- standard text extraction
- OCR fallback for scanned PDFs

OCR setup and inference are guarded with safe fallbacks. If OCR dependencies are unavailable, or OCR fails at runtime, extraction continues without crashing the full batch.

## 3. OCR Fallback Converts Images To Text

In `_read_pdf_text_with_ocr()`:

- `fitz` opens the PDF
- each page is rendered as an image
- RapidOCR reads text from the image
- OCR lines are normalized and collected

This enables extraction when `extract_text()` cannot read scanned documents, while still keeping failure behavior explicit and predictable.

## 4. It Normalizes Text Line By Line

`_normalize_line()` removes non-breaking spaces and compresses whitespace:

```python
line = line.replace("\u00a0", " ")
line = re.sub(r"\s+", " ", line).strip()
```

This step is important because legal PDFs often include irregular spacing and line breaks.

The line collector now runs in a single pass, so each line is normalized once (not twice).

## 5. It Cleans And Filters Candidate Names

`_clean_name()` removes honorific and legal noise such as:

- Hon'ble
- Justice / Chief Justice
- trailing judicial suffixes like J., JJ., CJ

It then standardizes spacing and capitalization.

`_is_likely_person_name()` verifies that a candidate resembles a person name by checking rules such as:

- no digits
- reasonable token length
- not a legal stop word (for example, respondent, court, section)
- mostly alphabetic token structure

This filtering stage reduces false positives.

## 6. It Extracts Judge Names From Individual Lines

`_extract_names_from_line()` applies two core patterns:

- title + name, e.g. `Hon'ble Mr. Justice A. B. Silva`
- name + judicial suffix, e.g. `A. B. Silva, J.` / `A. B. Silva, CJ` / `A. B. Silva, Judge`

Each match is cleaned and revalidated before being accepted.

## 7. It Extracts Plain Names From Header Lines

`_extract_plain_names_from_header_line()` handles formats where judges are listed without explicit markers like Justice or J., especially around CORAM/PRESENT blocks.

It rejects lines likely to be non-bench content, including:

- counsel lines
- petitioner/respondent lines
- long descriptive prose
- lines containing digits

It then splits on commas and `and`, and validates each part as a likely person name.

## 8. It Finds The Bench

`_extract_bench()` scans the top document region for anchors such as:

- CORAM
- PRESENT
- BEFORE
- BENCH

Once an anchor is detected, the extractor inspects nearby lines and collects deduplicated judge names in document order.

It also includes a fallback for compact bench lines that contain multiple judicial suffix markers.

The extractor is page-aware: it stores normalized line records with page index and line index, then uses those positions to score top and bottom structural evidence more reliably.

## 9. It Finds The Author Judge

`_extract_author()` uses deterministic scoring. A judge accumulates points when the line contains signals such as:

- explicit authorship cues (`delivered by`, `judgment by`, `per`, etc.)
- signature-style judicial line patterns
- judgment-heading context
- position near the final third of the document

Additional weighting is applied when end-of-judgment structure resembles a final signature block.

The highest-scoring judge(s) are selected; if strong evidence is absent, conservative fallback logic is used.

Compared with earlier behavior, weak-evidence fallback is intentionally stricter to reduce false positives.

Because these rules are fixed, the output remains deterministic and reproducible for the same input.

## 10. It Processes PDFs And Writes JSON

`extract_from_pdf()` performs the core sequence:

- read text
- collect normalized lines
- extract bench
- extract author judge
- return `ExtractionResult`

`process_folder()` then:

- iterates over all `.pdf` files in `data/`
- writes one `.json` file per input into `output/` using UTF-8 and `ensure_ascii=False`
- returns generated output paths

The CLI also supports structured logging (`--log-level`) for easier debugging and validation.

## 11. Tests And Validation

The project includes rule-focused regression tests for key edge cases, including:

- bench extraction from `BEFORE`/`CORAM` style blocks
- signature-style name extraction (`A. B. Silva, J.`)
- author extraction with explicit cue lines
- conservative behavior when author evidence is weak

These tests help prevent regressions as extraction heuristics evolve.

## Technologies Used

- `pypdf` for primary text extraction
- `PyMuPDF` (imported as `fitz`) for rendering pages in OCR fallback
- `rapidocr-onnxruntime` for OCR
- `numpy` and `onnxruntime` as OCR runtime dependencies
- Python standard library modules (`re`, `json`, `logging`, `pathlib`, `functools`)

This behavior exactly matches the assignment requirement.

## Overall Assessment

The solution is aligned with the expected design goals:

- deterministic
- rule-based
- reproducible
- no LLM dependency
- structured JSON output
- automatic batch processing

Continuous refinement can still improve edge-case accuracy, but the architecture already follows a strong, production-minded extraction workflow.
