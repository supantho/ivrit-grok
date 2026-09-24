#!/usr/bin/env bash
# Acquire raw sources at pinned versions.  Raw files are never edited afterwards.
set -euo pipefail
cd "$(dirname "$0")/.."

UNIMORPH_COMMIT=b2bff12338caa1922e6aeea128c617730f93d334
DUMP_DATE=20260901
DUMP_SHA1=eb96787cc3c45a938259aeede923ccc7cdc143a7

mkdir -p data/raw/unimorph data/raw/wiktionary/dump_${DUMP_DATE} data/raw/validators

if [ ! -d data/raw/unimorph/heb/.git ]; then
  git clone https://github.com/unimorph/heb.git data/raw/unimorph/heb
fi
git -C data/raw/unimorph/heb checkout -q ${UNIMORPH_COMMIT}
test "$(git -C data/raw/unimorph/heb rev-parse HEAD)" = "${UNIMORPH_COMMIT}"

D=data/raw/wiktionary/dump_${DUMP_DATE}
F=hewiktionary-${DUMP_DATE}-pages-articles.xml.bz2
if [ ! -f "$D/$F" ]; then
  curl -sSf -o "$D/$F" "https://dumps.wikimedia.org/hewiktionary/${DUMP_DATE}/$F"
  curl -sSf -o "$D/hewiktionary-${DUMP_DATE}-sha1sums.txt" \
       "https://dumps.wikimedia.org/hewiktionary/${DUMP_DATE}/hewiktionary-${DUMP_DATE}-sha1sums.txt"
fi
echo "${DUMP_SHA1}  $D/$F" | sha1sum -c -

# Record retrieval metadata (first retrieval only; never overwritten).
M=data/raw/retrieval_log.json
if [ ! -f "$M" ]; then
  cat > "$M" <<EOF
{
  "unimorph_heb": {"url": "https://github.com/unimorph/heb", "commit": "${UNIMORPH_COMMIT}",
                   "retrieved": "$(date -u +%Y-%m-%d)"},
  "hewiktionary_dump": {"url": "https://dumps.wikimedia.org/hewiktionary/${DUMP_DATE}/$F",
                        "sha1": "${DUMP_SHA1}", "retrieved": "$(date -u +%Y-%m-%d)"}
}
EOF
fi
echo "sources OK"
