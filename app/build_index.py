"""Parsea el Indice impreso de resultado.md, trocea el contenido de cada
Proyecto en chunks embebibles y los guarda en una base SQLite (sqlite-vec)
con sus embeddings, listos para busqueda por similitud.
"""

import argparse
import re
import sqlite3
from dataclasses import dataclass, field

import sqlite_vec
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_TARGET_WORDS = 80

# Modelos E5 (intfloat/multilingual-e5-*) fueron entrenados exigiendo estos
# prefijos en el texto de entrada -- omitirlos degrada mucho la calidad del
# embedding. Los modelos "plain" (MiniLM, paraphrase-*) no los necesitan.
E5_PASSAGE_PREFIX = "passage: "

# El libro usa TRES esquemas de fases pedagogicas distintos segun el tipo de
# Proyecto (confirmado escaneando los headings reales de los 38 Proyectos, no
# solo los de "Lenguajes" con los que se origino esta lista):
#   - "Lenguajes": Identificacion..Consideracion y avances (el esquema
#     original, unico que esta lista cubria antes de este fix).
#   - "indagacion cientifica" (varios Proyectos de Saberes y pensamiento
#     cientifico y de De lo Humano y lo comunitario): Sensibilizacion..
#     Autorreflexion.
#   - "sociocritico" (Etica naturaleza y sociedad): Problematica..Valorando
#     mis pasos.
# Sin las ultimas dos, detect_fase() nunca matcheaba nada fuera de
# "Lenguajes" y chunk.fase quedaba NULL en 27/38 Proyectos (confirmado
# contando chunks tageados por proyecto en book.db).
#
# Cada fase mapea a una lista de variantes textuales -- el VLM transcribe la
# misma fase con pequeñas diferencias reales entre Proyectos (no solo ruido
# de OCR al azar): "construccion Y comprobacion" vs "construccion Y/O
# comprobacion" vs "construccion O comprobacion" son las tres formas reales
# que aparecen en el libro impreso, igual que "registro de experiencia" vs
# "registro de LA experiencia". No se persiguen erratas de un solo caso (ej.
# "Reconoczamos", "SENSABILIZACIÓN") -- esas son ruido puntual del VLM, no
# variantes sistematicas del esquema.
FASE_ALIASES: dict[str, list[str]] = {
    # Esquema "Lenguajes"
    "Identificación": ["identificación"],
    "Recuperación": ["recuperación"],
    "Planificación": ["planificación"],
    "Acercamiento": ["acercamiento"],
    "Comprensión y producción": ["comprensión y producción"],
    "Reconocimiento": ["reconocimiento"],
    "Integración": ["integración"],
    "Difusión": ["difusión"],
    "Consideración y avances": ["consideración y avances"],
    # Esquema "indagación científica"
    "Sensibilización": ["sensibilización"],
    "Diseño y desarrollo de la indagación": ["diseño y desarrollo de la indagación"],
    "Construcción y comprobación": [
        "construcción y comprobación",
        "construcción y/o comprobación",
        "construcción o comprobación",
    ],
    "Comunicación": ["comunicación"],
    "Autorreflexión": ["autorreflexión"],
    # Esquema "sociocrítico"
    "Problemática": ["problemática"],
    "Identificamos el problema": ["identificamos el problema", "identificamos la problemática"],
    "Encontramos el origen": ["encontramos el origen"],
    "Propuestas a seguir": ["propuestas a seguir"],
    "Organizamos los pasos": ["organizamos los pasos"],
    "Seguir el camino": ["seguir el camino"],
    "Registro de experiencia": ["registro de experiencia", "registro de la experiencia"],
    "Valorando mis pasos": ["valorando mis pasos", "valorando más pasos"],
}

# Margen de caracteres permitido entre el heading completo y la variante que
# matcheo -- suficiente para prefijos/puntuacion ("1. Comunicación") pero no
# tanto como para que fases con nombres cortos y genericos (ej.
# "Comunicación") matcheen dentro de titulos de Proyecto largos que
# casualmente contienen esa palabra (ej. el propio titulo del Proyecto
# "Colaboración, comunicación y satisfacción en el logro de metas comunes",
# que de otro modo se auto-tagearia como fase "Comunicación" antes de que
# empiece el contenido real).
FASE_MATCH_SLACK = 20

PAGE_MARKER_RE = re.compile(r"<!-- fuente: (\d+)\.jpg -->")
ENTRY_LINE_RE = re.compile(r"^(.*?)\s*\.{2,}\s*(\d{1,3})\s*$")
HEADING_RE = re.compile(r"^#{1,6}\s*(.+?)\s*$")
# Ademas de "# Heading"/"## Heading", el VLM a veces marca un heading de fase
# solo con negritas ("**Recuperación**", parrafo aislado, sin '#') -- sin
# esto detect_fase() lo ignora en silencio y la fase vigente se queda pegada
# en la anterior (confirmado: 31/38 Proyectos sin NINGUN chunk tageado
# "Recuperación", con el heading real presente en resultado.md pero nunca
# detectado). Exige que el parrafo sea EXACTAMENTE "**texto**" -- no basta
# con que empiece en negritas -- para no tocar parrafos como "**Nota:**
# cuidado con..." que traen mas contenido despues del cierre.
BOLD_HEADING_RE = re.compile(r"^\*\*\s*(.+?)\s*\*\*$")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class IndexEntry:
    kind: str  # "campo" o "proyecto"
    title: str
    page_start: int
    campo_formativo: str | None = None
    page_end: int | None = None


@dataclass
class Chunk:
    text: str
    page_start: int
    page_end: int
    fase: str | None


def load_pages(text: str) -> list[tuple[int, str]]:
    """Divide resultado.md en (numero_de_pagina, texto_de_pagina) usando los
    comentarios `<!-- fuente: NNN.jpg -->`, en el orden en que aparecen."""
    matches = list(PAGE_MARKER_RE.finditer(text))
    pages = []
    for i, m in enumerate(matches):
        page_num = int(m.group(1))
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        pages.append((page_num, text[start:end]))
    return pages


def parse_indice(text: str) -> list[IndexEntry]:
    heading_pos = text.index("# Índice")
    first_content_marker = re.search(r"<!-- fuente: 014\.jpg -->", text)
    block = text[heading_pos:first_content_marker.start()]

    entries: list[IndexEntry] = []
    current_campo: str | None = None
    for raw_line in block.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        m = ENTRY_LINE_RE.match(line)
        if not m:
            continue
        title_part, page = m.group(1).strip(), int(m.group(2))

        if title_part.startswith("-"):
            title = title_part.lstrip("-").strip()
            entries.append(IndexEntry("proyecto", title, page, campo_formativo=current_campo))
        elif title_part.startswith("**") or re.match(r"^#{1,3}\s*Campo formativo", title_part, re.I):
            title = re.sub(r"^\*+|\*+$", "", title_part).strip()
            title = re.sub(r"^#+\s*", "", title).strip()
            title = re.sub(r"(?i)^campo formativo\s*", "", title).strip()
            current_campo = title
            entries.append(IndexEntry("campo", title, page))
        else:
            entries.append(IndexEntry("proyecto", title_part, page, campo_formativo=None))

    entries.sort(key=lambda e: e.page_start)
    return entries


def assign_page_ranges(entries: list[IndexEntry], last_page: int, indice_page: int) -> list[IndexEntry]:
    """El page_end de cada entrada es el page_start de la siguiente menos 1
    (la ultima llega hasta la ultima pagina del libro). Ademas antepone un
    bloque de portada/presentacion (paginas 0 hasta el Indice) y separa el
    Indice impreso (paginas indice_page..primera entrada) como su propio
    bloque de kind="indice": su texto crudo repite literalmente los titulos
    de cada Proyecto, así que si se trocea junto con el resto contamina la
    busqueda semantica (cualquier pregunta parecida a un titulo de Proyecto
    matchea tambien su propia linea del Indice). Ya esta capturado en las
    tablas campo_formativo/proyecto, por lo que no se pierde nada al
    excluirlo del vector store (ver kind != "proyecto" en main())."""
    for i, entry in enumerate(entries):
        entry.page_end = entries[i + 1].page_start - 1 if i + 1 < len(entries) else last_page

    indice_block = IndexEntry(
        kind="indice",
        title="Índice",
        page_start=indice_page,
        campo_formativo=None,
        page_end=entries[0].page_start - 1,
    )
    front_matter = IndexEntry(
        kind="proyecto",
        title="Portada y presentación",
        page_start=0,
        campo_formativo=None,
        page_end=indice_page - 1,
    )
    return [front_matter, indice_block, *entries]


def split_paragraphs(page_text: str) -> list[str]:
    blocks = re.split(r"\n\s*\n", page_text)
    return [b.strip() for b in blocks if b.strip()]


def match_heading(paragraph: str) -> re.Match | None:
    return HEADING_RE.match(paragraph) or BOLD_HEADING_RE.match(paragraph)


def detect_fase(paragraph: str) -> str | None:
    m = match_heading(paragraph)
    if not m:
        return None
    heading_text = m.group(1).lower()
    for fase, aliases in FASE_ALIASES.items():
        for alias in aliases:
            if alias in heading_text and len(heading_text) <= len(alias) + FASE_MATCH_SLACK:
                return fase
    return None


def strip_heading_markup(paragraph: str) -> str:
    m = match_heading(paragraph)
    return m.group(1) if m else paragraph


def is_low_quality(text: str) -> bool:
    """Descarta ruido de OCR/VLM: parrafos vacios o dominados por palabras sin
    letras (p. ej. paginas alucinadas como '- \\n\\n- \\n\\n- ...'). Se mide por
    palabra, no por caracter, para no penalizar lineas de indice con lideres de
    puntos ("Titulo .................... 14")."""
    words = text.split()
    if not words:
        return True
    alphabetic_words = sum(any(ch.isalpha() for ch in w) for w in words)
    return alphabetic_words / len(words) < 0.5


def units_for_page(page_num: int, page_text: str, current_fase: str | None) -> tuple[list[tuple[str, int, str | None]], str | None]:
    """Devuelve unidades encadenables (texto, pagina, fase) para una pagina,
    actualizando la fase vigente cuando detecta un encabezado conocido."""
    units = []
    for paragraph in split_paragraphs(page_text):
        fase = detect_fase(paragraph)
        if fase:
            current_fase = fase
        content = strip_heading_markup(paragraph)
        if is_low_quality(content):
            continue
        if len(content.split()) > CHUNK_TARGET_WORDS:
            for sentence in SENTENCE_SPLIT_RE.split(content):
                sentence = sentence.strip()
                if sentence and not is_low_quality(sentence):
                    units.append((sentence, page_num, current_fase))
        else:
            units.append((content, page_num, current_fase))
    return units, current_fase


def chunk_entry(entry: IndexEntry, pages_by_num: dict[int, str]) -> list[Chunk]:
    current_fase: str | None = None
    all_units: list[tuple[str, int, str | None]] = []
    for page_num in range(entry.page_start, entry.page_end + 1):
        page_text = pages_by_num.get(page_num)
        if page_text is None:
            continue
        units, current_fase = units_for_page(page_num, page_text, current_fase)
        all_units.extend(units)

    chunks: list[Chunk] = []
    buffer: list[tuple[str, int, str | None]] = []

    def flush():
        if not buffer:
            return
        fase_weights: dict[str | None, int] = {}
        for text, _, fase in buffer:
            fase_weights[fase] = fase_weights.get(fase, 0) + len(text.split())
        dominant_fase = max(fase_weights, key=fase_weights.get)
        chunks.append(
            Chunk(
                text=" ".join(text for text, _, _ in buffer),
                page_start=min(page for _, page, _ in buffer),
                page_end=max(page for _, page, _ in buffer),
                fase=dominant_fase,
            )
        )

    word_count = 0
    for unit in all_units:
        text, _page_num, _fase = unit
        n_words = len(text.split())
        if word_count + n_words > CHUNK_TARGET_WORDS and buffer:
            flush()
            buffer, word_count = [], 0
        buffer.append(unit)
        word_count += n_words
    flush()
    return chunks


def build_database(
    db_path: str,
    entries: list[IndexEntry],
    entry_chunks: dict[int, list[Chunk]],
    embeddings,
    embedding_dim: int,
) -> None:
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    conn.executescript(
        """
        DROP TABLE IF EXISTS campo_formativo;
        DROP TABLE IF EXISTS proyecto;
        DROP TABLE IF EXISTS chunk;
        DROP TABLE IF EXISTS chunk_vec;
        DROP TABLE IF EXISTS chunk_fts;

        CREATE TABLE campo_formativo (
            id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            page_start INTEGER NOT NULL,
            page_end INTEGER NOT NULL
        );

        CREATE TABLE proyecto (
            id INTEGER PRIMARY KEY,
            campo_formativo_id INTEGER REFERENCES campo_formativo(id),
            title TEXT NOT NULL,
            page_start INTEGER NOT NULL,
            page_end INTEGER NOT NULL
        );

        CREATE TABLE chunk (
            id INTEGER PRIMARY KEY,
            proyecto_id INTEGER REFERENCES proyecto(id),
            fase TEXT,
            page_start INTEGER NOT NULL,
            page_end INTEGER NOT NULL,
            text TEXT NOT NULL
        );
        """
    )
    conn.execute(
        f"CREATE VIRTUAL TABLE chunk_vec USING vec0(embedding float[{embedding_dim}])"
    )
    conn.execute(
        "CREATE VIRTUAL TABLE chunk_fts USING fts5("
        "text, content='chunk', content_rowid='id', "
        "tokenize='unicode61 remove_diacritics 2')"
    )

    campo_ids: dict[str, int] = {}
    for entry in entries:
        if entry.kind == "campo":
            cur = conn.execute(
                "INSERT INTO campo_formativo (title, page_start, page_end) VALUES (?, ?, ?)",
                (entry.title, entry.page_start, entry.page_end),
            )
            campo_ids[entry.title] = cur.lastrowid

    embedding_offset = 0
    for entry in entries:
        if entry.kind != "proyecto":
            continue
        campo_id = campo_ids.get(entry.campo_formativo) if entry.campo_formativo else None
        cur = conn.execute(
            "INSERT INTO proyecto (campo_formativo_id, title, page_start, page_end) VALUES (?, ?, ?, ?)",
            (campo_id, entry.title, entry.page_start, entry.page_end),
        )
        proyecto_id = cur.lastrowid

        for chunk in entry_chunks.get(id(entry), []):
            embedding = embeddings[embedding_offset]
            embedding_offset += 1
            cur = conn.execute(
                "INSERT INTO chunk (proyecto_id, fase, page_start, page_end, text) VALUES (?, ?, ?, ?, ?)",
                (proyecto_id, chunk.fase, chunk.page_start, chunk.page_end, chunk.text),
            )
            conn.execute(
                "INSERT INTO chunk_vec (rowid, embedding) VALUES (?, ?)",
                (cur.lastrowid, embedding.astype("float32").tobytes()),
            )
            conn.execute(
                "INSERT INTO chunk_fts (rowid, text) VALUES (?, ?)",
                (cur.lastrowid, chunk.text),
            )

    conn.commit()
    conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="resultado.md", help="Markdown fuente")
    parser.add_argument("--output", default="book.db", help="Base SQLite de salida")
    parser.add_argument(
        "--model", default=EMBEDDING_MODEL_NAME,
        help="Modelo de sentence-transformers a usar para los embeddings",
    )
    parser.add_argument(
        "--e5", action="store_true",
        help="Antepone 'passage: ' a cada chunk antes de embeber (requerido por los modelos intfloat/multilingual-e5-*)",
    )
    args = parser.parse_args()

    text = open(args.input, encoding="utf-8").read()

    pages = load_pages(text)
    pages_by_num = dict(pages)
    last_page = max(pages_by_num)

    heading_pos = text.index("# Índice")
    markers_before = [m for m in PAGE_MARKER_RE.finditer(text) if m.start() < heading_pos]
    indice_page = int(markers_before[-1].group(1))

    entries = parse_indice(text)
    entries = assign_page_ranges(entries, last_page, indice_page)

    entry_chunks: dict[int, list[Chunk]] = {}
    for entry in entries:
        if entry.kind != "proyecto":
            continue  # paginas divisorias de Campo formativo y el Indice crudo son ruido/redundantes
        entry_chunks[id(entry)] = chunk_entry(entry, pages_by_num)

    flat_chunks = [c for chunks in entry_chunks.values() for c in chunks]
    print(f"Entradas del indice: {len(entries)}")
    print(f"Chunks generados: {len(flat_chunks)}")

    model = SentenceTransformer(args.model)
    texts = [c.text for c in flat_chunks]
    if args.e5:
        texts = [E5_PASSAGE_PREFIX + t for t in texts]
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    build_database(args.output, entries, entry_chunks, embeddings, model.get_embedding_dimension())
    print(f"Base de datos escrita en: {args.output} (modelo: {args.model}, dim: {model.get_embedding_dimension()})")


if __name__ == "__main__":
    main()
