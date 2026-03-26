# Judge Extraction Solution

This repository contains a deterministic, non-LLM extraction system as an Internship Assessment – Paralegal Pvt Ltd.
The system does identifying:

- `bench`: all judges listed in the bench/coram/present/before sections
- `author_judge`: judge(s) who authored/delivered the final judgment

The system processes every `.pdf` file in the `../data` folder and writes one JSON file per input into `../output`.

## Constraints Compliance

- No LLMs or generative AI are used.
- Extraction is fully automatic and reproducible.
- No manual post-editing is required.
- Dependencies are declared in `pyproject.toml` and installable with `uv`.

# Usage
## Setup

From the repository root:

```bash
cd solution
uv sync
```

## Run

From `solution`:

```bash
uv run extract-judges
```

Run with diagnostic logs:

```bash
uv run extract-judges --log-level INFO
```

Optional custom paths:

```bash
uv run extract-judges --data-dir ../data --output-dir ../output
```

## Output Format

For each PDF input, one JSON file is generated:

```json
{
  "source_file": "sample-judgment-1.pdf",
  "bench": ["Judge Name 1", "Judge Name 2"],
  "author_judge": ["Judge Name 1"]
}
```

# Approach

The extractor uses deterministic rule-based NLP over PDF text:

For a detailed step-by-step technical workflow, see [Workflow.md](Workflow.md).

1. Parse PDF text with `pypdf`.
2. Normalize lines to remove noisy whitespace and formatting artifacts.
3. Extract judge names using robust regex patterns that capture common legal styles:
   - `Justice <Name>`
   - `<Name>, J.` / `<Name>, JJ.` / `<Name>, Judge`
   - parenthesized signature forms near judgment end
4. Use OCR fallback for image-based PDFs (PyMuPDF + RapidOCR), with safe runtime guards so OCR failures do not crash the full batch.
5. Keep page-aware normalized line records to preserve positional context for top-of-document bench blocks and end-of-document author signals.
6. Build `bench` from:
   - windows around `CORAM`, `PRESENT`, `BEFORE`, `BENCH`
   - supporting top-of-document judicial mention lines
7. Infer `author_judge` via scoring heuristics:
   - strong cues (`delivered by`, `pronounced by`, `authored by`, `judgment by`, `per`)
   - end-of-document judicial signatures
   - judgment heading hints
   - recency in the final third of document
8. Use conservative deterministic fallback logic:
   - avoid forcing `author_judge` when evidence is weak
   - prefer structured Chief Justice or signature evidence when available
9. Write JSON with UTF-8 and `ensure_ascii=False` to preserve names cleanly.

## Assumptions

- Input files may be machine-readable or scanned; scanned files use OCR fallback.
- Judge names appear in one of the common legal formats captured by rules.
- `author_judge` may contain one or multiple names when evidence supports it.

## Technologies Used

- Python 3.11+
- `uv` for dependency/environment management
- `pypdf` for PDF text extraction
- `PyMuPDF` (imported as `fitz`) for PDF page rendering in OCR flow
- `rapidocr-onnxruntime` for OCR inference
- `numpy` and `onnxruntime` for OCR tensor/runtime processing
- Standard library modules:
   - `logging` for extraction diagnostics
  - `re` for regex parsing
  - `pathlib` for file handling
  - `json` for structured output

## Tests

From `solution`:

```bash
uv run python -m unittest discover -s tests -v
```

# Applicant Details

### T.M.T.A.B. Tennekoon,
### Computer Engineering Undergraduate,
### Dept. Electrical & Information Engineering,
### Faculty of Engineering,
### University of Ruhuna

0763253332
mailto:thanujayaabtennekoon@gmail.com
www.linkedin.com/in/thanujaya-tennekoon