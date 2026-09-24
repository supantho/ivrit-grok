"""Canonical locations and pinned source versions."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
CANONICAL = DATA / "canonical"
DERIVED = DATA / "derived"
SPLITS = DATA / "splits"
REPORTS = DATA / "reports"
METADATA = DATA / "metadata"

# ---- pinned sources (change => new dataset version) ----
UNIMORPH_REPO_URL = "https://github.com/unimorph/heb"
UNIMORPH_COMMIT = "b2bff12338caa1922e6aeea128c617730f93d334"
UNIMORPH_DIR = RAW / "unimorph" / "heb"
UNIMORPH_FILES = {
    "heb_voc": "heb_voc",
    "heb": "heb",
    "heb.args": "heb.args",
    "heb_unvoc.args": "heb_unvoc.args",
    "from_UD_unvoc": "from_UD_unvoc",
}
UNIMORPH_LICENSE = "CC BY-SA 3.0"

WIKTIONARY_DUMP_DATE = "20260901"
WIKTIONARY_DUMP_URL = (f"https://dumps.wikimedia.org/hewiktionary/{WIKTIONARY_DUMP_DATE}/"
                       f"hewiktionary-{WIKTIONARY_DUMP_DATE}-pages-articles.xml.bz2")
WIKTIONARY_DUMP = (RAW / "wiktionary" / f"dump_{WIKTIONARY_DUMP_DATE}" /
                   f"hewiktionary-{WIKTIONARY_DUMP_DATE}-pages-articles.xml.bz2")
WIKTIONARY_DUMP_SHA1 = "eb96787cc3c45a938259aeede923ccc7cdc143a7"
WIKTIONARY_LICENSE = "CC BY-SA 4.0 (text); GFDL"
WIKTIONARY_PAGE_URL = "https://he.wiktionary.org/w/index.php?title={title}&oldid={rev}"


def wiktionary_url(title: str, rev_id: int) -> str:
    from urllib.parse import quote
    return WIKTIONARY_PAGE_URL.format(title=quote(title.replace(" ", "_")), rev=rev_id)


def unimorph_url(file: str, line: int | None = None) -> str:
    u = f"{UNIMORPH_REPO_URL}/blob/{UNIMORPH_COMMIT}/{file}"
    return u + (f"#L{line}" if line else "")
