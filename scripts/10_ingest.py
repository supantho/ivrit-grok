"""RAW -> INTERIM: parse UniMorph and Wiktionary into flat provenance tables."""
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hebmorph.ingest import ingest_unimorph, ingest_wiktionary  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
if __name__ == "__main__":
    s1 = ingest_unimorph()
    print(json.dumps({k: v for k, v in s1.items() if k != "combining_mark_inventory"}, ensure_ascii=False, indent=1))
    s2 = ingest_wiktionary()
    print(json.dumps(s2, ensure_ascii=False, indent=1))
