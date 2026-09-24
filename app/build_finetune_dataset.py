"""Genera el dataset de pares (pregunta, texto-positivo) para el fine-tuning
contrastivo del modelo de embeddings (paraphrase-multilingual-MiniLM-L12-v2).

Combina tres fuentes:
  1. Plantillas de pregunta por titulo de Proyecto (usa el propio chunk
     inicial del Proyecto como positivo).
  2. Plantillas de pregunta por fase pedagogica dentro de cada Proyecto (usa
     los chunks tageados con esa fase como positivo -- ver el fix de
     app/build_index.py que corrigio el tageo de fase).
  3. Casos reales curados a mano, sacados de la sesion de debugging de hoy
     (lecturas nombradas como "El puñal"/"Yo, Julio Tarqui" que el embedding
     base ubicaba mas lejos de su propio Proyecto que de Proyectos ajenos).

Salida: JSONL con {"query": ..., "text": ...} por linea, separado en
train/eval (90/10, shuffle con seed fijo para reproducibilidad).
"""

import json
import random
import sqlite3
from pathlib import Path

DB_PATH = "book.db"
OUTPUT_DIR = Path("data")
TRAIN_PATH = OUTPUT_DIR / "embedder_finetune_train.jsonl"
EVAL_PATH = OUTPUT_DIR / "embedder_finetune_eval.jsonl"
EVAL_FRACTION = 0.1
SEED = 42
MAX_CHUNKS_PER_INTRO = 2  # cuantos chunks iniciales del Proyecto usar como positivo

PROYECTO_TEMPLATES = [
    '¿Qué es "{title}"?',
    '¿De qué trata el proyecto "{title}"?',
    'Muestra el proyecto "{title}"',
    'Háblame del proyecto "{title}"',
    '¿En qué consiste "{title}"?',
]

FASE_TEMPLATES = [
    '¿Qué plantea la fase {fase} del proyecto "{title}"?',
    'Muestra la fase {fase} de "{title}"',
    '¿Qué actividades hay en {fase} para "{title}"?',
    'En "{title}", ¿qué se hace en {fase}?',
]

# Casos reales de hoy donde el embedding base fallaba -- mezclan la forma en
# que un usuario realmente pregunta (comillas imperfectas, verbos variados)
# con el/los chunk(s) que SI son la respuesta correcta (ids de book.db).
CURATED_CASES = [
    {
        "queries": [
            "que es Juguemos con la Lengua?",
            "Juguemos con la lengua",
        ],
        "chunk_ids": [65, 66],
    },
    {
        "queries": [
            'muestra la lectura "El puñal"',
            '¿De qué trata "El puñal"?',
            '¿A qué proyecto pertenece "El puñal"?',
        ],
        "chunk_ids": [67],
    },
    {
        "queries": [
            'de que habla "Yo Julio tarqui"',
            'puedes desplegar "Yo Julio Tarqui"?',
            '¿Qué es "Yo, Julio Tarqui"?',
        ],
        "chunk_ids": [238, 239, 240],
    },
]


def fetch_proyectos(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute("SELECT id, title FROM proyecto ORDER BY id").fetchall()


def fetch_intro_text(conn: sqlite3.Connection, proyecto_id: int) -> str | None:
    rows = conn.execute(
        "SELECT text FROM chunk WHERE proyecto_id = ? ORDER BY id LIMIT ?",
        (proyecto_id, MAX_CHUNKS_PER_INTRO),
    ).fetchall()
    return " ".join(r["text"] for r in rows) if rows else None


def fetch_fase_groups(conn: sqlite3.Connection, proyecto_id: int) -> dict[str, str]:
    """Devuelve {fase: texto_concatenado} para cada fase presente en el
    Proyecto (agrupa todos sus chunks tageados con esa fase, en orden)."""
    rows = conn.execute(
        "SELECT fase, text FROM chunk WHERE proyecto_id = ? AND fase IS NOT NULL ORDER BY id",
        (proyecto_id,),
    ).fetchall()
    groups: dict[str, list[str]] = {}
    for row in rows:
        groups.setdefault(row["fase"], []).append(row["text"])
    return {fase: " ".join(texts) for fase, texts in groups.items()}


def fetch_chunk_text(conn: sqlite3.Connection, chunk_ids: list[int]) -> str:
    placeholders = ",".join("?" * len(chunk_ids))
    rows = conn.execute(
        f"SELECT text FROM chunk WHERE id IN ({placeholders}) ORDER BY id",
        chunk_ids,
    ).fetchall()
    return " ".join(r["text"] for r in rows)


def build_pairs(conn: sqlite3.Connection) -> list[dict]:
    pairs: list[dict] = []

    for proyecto in fetch_proyectos(conn):
        intro = fetch_intro_text(conn, proyecto["id"])
        if not intro:
            continue
        for template in PROYECTO_TEMPLATES:
            pairs.append({"query": template.format(title=proyecto["title"]), "text": intro})

        for fase, text in fetch_fase_groups(conn, proyecto["id"]).items():
            for template in FASE_TEMPLATES:
                pairs.append({
                    "query": template.format(title=proyecto["title"], fase=fase),
                    "text": text,
                })

    for case in CURATED_CASES:
        text = fetch_chunk_text(conn, case["chunk_ids"])
        for query in case["queries"]:
            pairs.append({"query": query, "text": text})

    return pairs


def main() -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    pairs = build_pairs(conn)
    print(f"Pares generados: {len(pairs)}")

    random.Random(SEED).shuffle(pairs)
    n_eval = max(1, int(len(pairs) * EVAL_FRACTION))
    eval_pairs, train_pairs = pairs[:n_eval], pairs[n_eval:]

    OUTPUT_DIR.mkdir(exist_ok=True)
    for path, subset in [(TRAIN_PATH, train_pairs), (EVAL_PATH, eval_pairs)]:
        with open(path, "w", encoding="utf-8") as f:
            for pair in subset:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")
        print(f"{path}: {len(subset)} pares")


if __name__ == "__main__":
    main()
