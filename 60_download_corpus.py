"""
60_download_corpus.py
═════════════════════
Downloads and cleans ~5M terms/phrases from public sources:
  - ConceptNet 5.7      (~800K terms ES+EN)
  - Wikipedia titles    (~1.5M ES+EN)
  - Wikidata labels     (~1.5M entities)
  - OpenSubtitles       (~1.2M short phrases)

Output: corpus/corpus_final.txt  (one entry per line, UTF-8, deduplicated)

Usage:
  python 60_download_corpus.py
  python 60_download_corpus.py --only conceptnet wikipedia   # only some sources
  python 60_download_corpus.py --max 1000000                 # limit per source
"""

import os
import re
import gzip
import json
import csv
import argparse
import urllib.request
import urllib.error
from pathlib import Path
from collections import OrderedDict

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
CORPUS_DIR   = Path("corpus")
RAW_DIR      = CORPUS_DIR / "raw"
OUTPUT_FILE  = CORPUS_DIR / "corpus_final.txt"

MAX_PER_SOURCE   = 2_000_000   # limit per source (prevents blowing up RAM)
MIN_CHARS        = 3
MAX_CHARS        = 120
TARGET_LANGUAGES = {"es", "en"}   # accepted languages where applicable

SOURCES = {
    "conceptnet": {
        "url": "https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz",
        "file": RAW_DIR / "conceptnet-assertions-5.7.0.csv.gz",
    },
    "wikipedia_en": {
        "url": "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-all-titles-in-ns0.gz",
        "file": RAW_DIR / "enwiki-titles.gz",
    },
    "wikipedia_es": {
        "url": "https://dumps.wikimedia.org/eswiki/latest/eswiki-latest-all-titles-in-ns0.gz",
        "file": RAW_DIR / "eswiki-titles.gz",
    },
    "wikidata": {
        # Simplified truthy dump with labels in ES+EN — JSON lines file
        "url": "https://dumps.wikimedia.org/wikidatawiki/entities/latest-lexemes.json.gz",
        "file": RAW_DIR / "wikidata-labels.json.gz",
        # More manageable alternative: use the SPARQL endpoint (see fetch_wikidata_sparql function)
        "use_sparql": True,
    },
    "opensubtitles": {
        # OpenSubtitles2018 ES, subset of short phrases
        "url": "https://opus.nlpl.eu/download.php?f=OpenSubtitles/v2018/mono/OpenSubtitles.raw.es.gz",
        "file": RAW_DIR / "opensubtitles_es.gz",
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ──────────────────────────────────────────────────────────────────────────────
def _mkdir():
    CORPUS_DIR.mkdir(exist_ok=True)
    RAW_DIR.mkdir(exist_ok=True)


def _download(url: str, dest: Path, desc: str = ""):
    if dest.exists():
        print(f"  ✓ Ya existe: {dest.name}")
        return
    print(f"  ↓ Descargando {desc or dest.name} …")
    try:
        def _progress(count, block, total):
            if total > 0:
                pct = min(100, count * block * 100 // total)
                print(f"\r    {pct:3d}%", end="", flush=True)

        urllib.request.urlretrieve(url, dest, reporthook=_progress)
        print(f"\r    ✅ {dest.stat().st_size / 1e6:.1f} MB")
    except urllib.error.URLError as e:
        print(f"\r    ❌ Error descargando {url}: {e}")
        if dest.exists():
            dest.unlink()


def _clean(text: str) -> str | None:
    """Cleans and filters a string. Returns None if it should be discarded."""
    # Remove Wikipedia underscores
    text = text.replace("_", " ").strip()
    # Remove disambiguation parentheses: "Banco (institución)" → "Banco"
    text = re.sub(r"\s*\(.*?\)\s*$", "", text).strip()
    # Remove control characters
    text = re.sub(r"[\x00-\x1f\x7f]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if len(text) < MIN_CHARS or len(text) > MAX_CHARS:
        return None
    # Discard if >40% are digits (IDs, dates, coordinates…)
    digits = sum(c.isdigit() for c in text)
    if digits / len(text) > 0.4:
        return None
    # Discard if it contains URLs
    if re.search(r"https?://|www\.", text, re.I):
        return None
    return text


# ──────────────────────────────────────────────────────────────────────────────
# SOURCES
# ──────────────────────────────────────────────────────────────────────────────
def extract_conceptnet(limit: int):
    """
    ConceptNet CSV: /a/[/r/IsA/,/c/en/dog/n/,/c/en/animal/n/] …
    Extracts the terms from the start and end columns for ES + EN.
    """
    path = SOURCES["conceptnet"]["file"]
    _download(SOURCES["conceptnet"]["url"], path, "ConceptNet 5.7")
    if not path.exists():
        return []

    terms = set()
    print("  📖 Procesando ConceptNet…")
    with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 3:
                continue
            for col in (2, 3):   # start_node, end_node
                node = row[col]
                # Format: /c/en/term  or  /c/es/term/pos/sense
                parts = node.split("/")
                if len(parts) >= 4 and parts[2] in TARGET_LANGUAGES:
                    term = parts[3].replace("_", " ")
                    cleaned = _clean(term)
                    if cleaned:
                        terms.add(cleaned)
            if len(terms) >= limit:
                break

    result = sorted(terms)
    print(f"  ✅ ConceptNet: {len(result):,} términos")
    return result


def extract_wikipedia(lang: str, limit: int):
    key  = f"wikipedia_{lang}"
    path = SOURCES[key]["file"]
    _download(SOURCES[key]["url"], path, f"Wikipedia {lang.upper()} titles")
    if not path.exists():
        return []

    terms = []
    print(f"  📖 Procesando Wikipedia {lang.upper()}…")
    with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            cleaned = _clean(line.strip())
            if cleaned:
                terms.append(cleaned)
            if len(terms) >= limit:
                break

    print(f"  ✅ Wikipedia {lang.upper()}: {len(terms):,} títulos")
    return terms


def fetch_wikidata_sparql(limit: int):
    """
    Uses Wikidata's public SPARQL endpoint to obtain labels.
    Downloads in batches of 10K so as not to saturate the endpoint.
    """
    import urllib.parse, time

    ENDPOINT = "https://query.wikidata.org/sparql"
    BATCH    = 10_000
    terms    = []
    offset   = 0

    print("  📖 Consultando Wikidata SPARQL…")
    headers  = {"User-Agent": "SemanticEmbeddingProject/1.0 (research)"}

    while len(terms) < limit:
        query = f"""
        SELECT ?label WHERE {{
          ?item wikibase:sitelinks ?links .
          FILTER(?links > 5)
          ?item rdfs:label ?label .
          FILTER(LANG(?label) IN ("es","en"))
        }}
        ORDER BY DESC(?links)
        LIMIT {BATCH}
        OFFSET {offset}
        """
        url = f"{ENDPOINT}?query={urllib.parse.quote(query)}&format=json"
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            batch = [b["label"]["value"] for b in data["results"]["bindings"]]
            if not batch:
                break
            for t in batch:
                cleaned = _clean(t)
                if cleaned:
                    terms.append(cleaned)
            offset += BATCH
            print(f"    … {len(terms):,} etiquetas", end="\r", flush=True)
            time.sleep(1.5)   # respect Wikidata's rate limit
        except Exception as e:
            print(f"\n    ⚠ Error en lote {offset}: {e}")
            break

    print(f"\n  ✅ Wikidata: {len(terms):,} etiquetas")
    return terms


def extract_opensubtitles(limit: int):
    path = SOURCES["opensubtitles"]["file"]
    _download(SOURCES["opensubtitles"]["url"], path, "OpenSubtitles ES")
    if not path.exists():
        return []

    phrases = []
    print("  📖 Procesando OpenSubtitles…")
    try:
        with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                # Only phrases of between 3 and 8 words (more natural, less noise)
                words = line.split()
                if 3 <= len(words) <= 8:
                    cleaned = _clean(line)
                    if cleaned:
                        phrases.append(cleaned)
                if len(phrases) >= limit:
                    break
    except Exception as e:
        print(f"  ⚠ Error leyendo OpenSubtitles: {e}")

    print(f"  ✅ OpenSubtitles: {len(phrases):,} frases")
    return phrases


# ──────────────────────────────────────────────────────────────────────────────
# DEDUPLICATION
# ──────────────────────────────────────────────────────────────────────────────
def deduplicate(entries: list[str]) -> list[str]:
    """Deduplicates preserving order, case-insensitive."""
    seen  = set()
    clean = []
    for e in entries:
        key = e.lower()
        if key not in seen:
            seen.add(key)
            clean.append(e)
    return clean


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Descarga corpus para tabla de embeddings humanizados")
    parser.add_argument("--only",  nargs="*",
                        choices=["conceptnet", "wikipedia", "wikidata", "opensubtitles"],
                        help="Fuentes a usar (por defecto todas)")
    parser.add_argument("--max",   type=int, default=MAX_PER_SOURCE,
                        help=f"Límite de entradas por fuente (default: {MAX_PER_SOURCE:,})")
    parser.add_argument("--out",   type=str, default=str(OUTPUT_FILE),
                        help="Archivo de salida")
    args = parser.parse_args()

    sources = set(args.only) if args.only else {"conceptnet", "wikipedia", "wikidata", "opensubtitles"}
    limit   = args.max
    outfile = Path(args.out)

    _mkdir()
    all_terms = []

    # ── ConceptNet ────────────────────────────────────────────────────────────
    if "conceptnet" in sources:
        print("\n[1/4] ConceptNet")
        all_terms.extend(extract_conceptnet(limit))

    # ── Wikipedia ────────────────────────────────────────────────────────────
    if "wikipedia" in sources:
        print("\n[2/4] Wikipedia")
        all_terms.extend(extract_wikipedia("en", limit))
        all_terms.extend(extract_wikipedia("es", limit))

    # ── Wikidata ─────────────────────────────────────────────────────────────
    if "wikidata" in sources:
        print("\n[3/4] Wikidata")
        all_terms.extend(fetch_wikidata_sparql(limit))

    # ── OpenSubtitles ─────────────────────────────────────────────────────────
    if "opensubtitles" in sources:
        print("\n[4/4] OpenSubtitles")
        all_terms.extend(extract_opensubtitles(limit))

    # ── Deduplicate and save ──────────────────────────────────────────────────
    print(f"\n🔄 Deduplicando {len(all_terms):,} entradas totales…")
    final = deduplicate(all_terms)
    print(f"✅ {len(final):,} entradas únicas tras deduplicación")

    outfile.parent.mkdir(exist_ok=True)
    with open(outfile, "w", encoding="utf-8") as f:
        f.write("\n".join(final))

    print(f"\n💾 Corpus guardado en: {outfile}")
    print(f"   Tamaño aprox.: {outfile.stat().st_size / 1e6:.1f} MB")
    print(f"\n➡  Siguiente paso:  python 70_build_table.py")


if __name__ == "__main__":
    main()
