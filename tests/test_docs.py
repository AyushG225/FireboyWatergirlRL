"""Checks on README and report prose."""

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = [ROOT / "README.md", *sorted((ROOT / "reports").glob("*.md"))]
LINK = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)\)")


class DocsTests(unittest.TestCase):
    def test_no_em_or_en_dashes(self):
        for path in DOCS:
            for number, line in enumerate(path.read_text().splitlines(), start=1):
                with self.subTest(file=path.name, line=number):
                    self.assertNotRegex(line, "[–—]")

    def test_banned_word_is_absent(self):
        for path in DOCS:
            with self.subTest(file=path.name):
                self.assertNotRegex(path.read_text(), r"(?i)\brather\b")

    def test_relative_links_resolve(self):
        for path in DOCS:
            for target in LINK.findall(path.read_text()):
                if target.startswith(("http://", "https://", "#", "mailto:")):
                    continue
                with self.subTest(file=path.name, link=target):
                    resolved = (path.parent / target.split("#")[0]).resolve()
                    self.assertTrue(resolved.exists(), f"{target} does not exist")


if __name__ == "__main__":
    unittest.main()
