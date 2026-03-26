from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from src.judge_extractor.extractor import (
    LineRecord,
    _collect_lines,
    _extract_author,
    _extract_bench,
    _extract_names_from_line,
)


class ExtractorTests(unittest.TestCase):
    def test_collect_lines_normalizes_once_behavior(self) -> None:
        full_text = "  A  B  \n\n\u00a0C\tD  "
        self.assertEqual(_collect_lines(full_text), ["A B", "C D"])

    def test_extract_names_from_signature_line(self) -> None:
        line = "A. B. Silva, J."
        names = _extract_names_from_line(line)
        self.assertIn("A. B. Silva", names)

    def test_extract_bench_from_before_block(self) -> None:
        records = [
            LineRecord(0, 0, "Before:"),
            LineRecord(0, 1, "J.A.N. de Silva, C.J."),
            LineRecord(0, 2, "Dr. Bandaranayake, J."),
            LineRecord(0, 3, "Counsel:"),
        ]
        bench = _extract_bench(records)
        self.assertEqual(bench, ["J.A.N. de Silva", "Dr. Bandaranayake"])

    def test_extract_author_with_cue(self) -> None:
        bench = ["A. B. Silva", "C. D. Fernando"]
        records = [
            LineRecord(0, 0, "Before:"),
            LineRecord(0, 1, "A. B. Silva, J."),
            LineRecord(0, 2, "C. D. Fernando, J."),
            LineRecord(2, 0, "Judgment by A. B. Silva, J."),
        ]
        author = _extract_author(records, bench)
        self.assertEqual(author, ["A. B. Silva"])

    def test_extract_author_returns_empty_when_weak(self) -> None:
        bench = ["A. B. Silva", "C. D. Fernando"]
        records = [
            LineRecord(0, 0, "Before:"),
            LineRecord(0, 1, "A. B. Silva, J."),
            LineRecord(0, 2, "C. D. Fernando, J."),
            LineRecord(1, 0, "This appeal concerns Article 13 rights."),
        ]
        author = _extract_author(records, bench)
        self.assertEqual(author, [])


if __name__ == "__main__":
    unittest.main()
