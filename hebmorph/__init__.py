"""hebmorph: provenance-aware Modern Hebrew verbal morphology database.

Pipeline: RAW SOURCES -> NORMALIZATION -> CANONICAL DB -> VALIDATION/QC ->
EXPERIMENTAL VIEWS -> SPLIT MANIFESTS.
"""

DATASET_NAME = "HEB_MORPH_GROK"
DATASET_VERSION = "v0.1"
SCHEMA_VERSION = "0.1.0"

# Bump a parser version whenever its output for the same input can change.
PARSER_VERSIONS = {
    "hebrew_normalizer": "0.1.0",
    "unimorph_parser": "0.1.0",
    "wiktionary_parser": "0.1.0",
    "matcher": "0.1.0",
    "canonical_builder": "0.1.0",
    "derived_views": "0.1.0",
    "splitter": "0.1.0",
    "synthetic_generator": "0.1.0",
}
