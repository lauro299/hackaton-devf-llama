import argparse
import json
import re
import unicodedata
from difflib import SequenceMatcher
from os import environ as environment

import sqlite3

import sqlite_vec
import torch
from huggingface_hub import login
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging

model_name = "Qwen/Qwen3-8B"
# Embedder fine-tuneado publicado en Hugging Face Hub. Debe ser el MISMO
# modelo con el que se construyo la base (--db): los vectores de chunk_vec
# solo son comparables con embeddings de ese modelo.
DEFAULT_EMBEDDER = "lau299/ximhai-embedder-es"
DEFAULT_DB = "book.db"
TOP_K = 3
WINDOW_SIZE = 5
MAX_CONTEXT_CHUNKS = 50
FULL_FALLBACK_WINDOW = 10
MAX_TOTAL_CONTEXT_CHARS = 20000
DISTANCE_MARGIN = 0.08
MAX_ACCEPTABLE_DISTANCE = 0.93
LEXICAL_FALLBACK_LIMIT = 5
TITLE_MATCH_THRESHOLD = 0.6

# Palabras funcionales en espanol a ignorar al armar la consulta FTS5 -- sin
# esto, palabras como "como"/"las"/"debe" dominan el ranking BM25 y tapan
# las palabras de contenido reales (ej. "aulas", "suelo").
STOPWORDS = {
    "que", "como", "cual", "cuales", "donde", "cuando", "quien", "quienes",
    "por", "para", "con", "sin", "las", "los", "del", "una", "uno", "unos",
    "unas", "este", "esta", "estos", "estas", "ese", "esa", "esos", "esas",
    "debe", "deben", "hay", "son", "ser", "sobre", "entre", "desde", "hacia",
    "segun", "tras", "mas", "pero", "porque", "tiene", "tienen", "puede",
    "pueden", "puedes", "podrias", "podria", "podrian", "quieres", "quiero",
    "dime", "dame", "muestra", "mostrar", "despliega", "desplegar",
    "ensename", "muy", "todo", "toda", "todos", "todas",
}

logging.set_verbosity_error()

# Se inicializan en main() a partir de --db/--embedder.
embedding_model: SentenceTransformer | None = None
conn: sqlite3.Connection | None = None


def configure_database(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn

# NOTA: is_structure_query/STRUCTURE_QUERY_RE/CONTENT_FILTER_RE existieron
# aqui como el mecanismo de routing por regex -- reemplazados por
# route_intent (tool-calling, ver mas abajo) despues de dos intentos
# fallidos de generalizarlos via embeddings (ver el comentario junto a
# TOOLS/route_intent para el detalle completo de por que se abandono el
# enfoque por regex/similitud en general). find_by_title sigue existiendo
# (ahora siempre fuzzy) y la usan handle_mostrar_texto_citado y
# search_context.


def describe_structure() -> str:
    rows = conn.execute(
        """
        SELECT cf.title AS campo, p.title AS proyecto
        FROM proyecto p
        JOIN campo_formativo cf ON cf.id = p.campo_formativo_id
        ORDER BY cf.page_start, p.page_start
        """
    ).fetchall()

    lines = ["El libro se organiza en los siguientes Campos formativos y Proyectos:"]
    current_campo = None
    for row in rows:
        if row["campo"] != current_campo:
            current_campo = row["campo"]
            lines.append(f"\nCampo formativo: {current_campo}")
        lines.append(f"  - {row['proyecto']}")
    return "\n".join(lines)


QUOTE_RE = re.compile(r'["“”‘’\'«»]([^"“”‘’\'«»]{3,80})["“”‘’\'«»]')


def extract_quoted(ask: str) -> str | None:
    """Si la pregunta cita uno o mas nombres entre comillas (ej. 'muestra la
    lectura "El punal"', 'de que habla "Yo Julio Tarqui"'), devuelve la cita
    MAS LARGA. Es una senal mucho mas confiable que las palabras sueltas de
    la pregunta -- el verbo que envuelve la cita ("de que habla"/"puedes
    desplegar"/"muestra") resulto ser fragil para el FTS (ej. "puedes"
    colandose como palabra de contenido y cambiando el ranking entre dos
    frases que piden lo mismo).

    Cuando hay VARIAS citas en la misma pregunta (ej. 'muestra la fase
    "recuperacion" del proyecto "Reconocer, valorar y proteger las lenguas
    indigenas..."'), tomar la primera es un error -- "recuperacion" es el
    nombre de una fase generica que se repite en casi todos los Proyectos,
    y le gana al titulo real si se toma por orden de aparicion. El titulo o
    lectura real casi siempre es la cita mas larga, asi que se prioriza por
    longitud en vez de por posicion."""
    matches = QUOTE_RE.findall(ask)
    if not matches:
        return None
    return max((m.strip() for m in matches), key=len)


FTS_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def fts_query(ask: str) -> str:
    """Arma una consulta FTS5 con las palabras de contenido de la pregunta
    (ignora stopwords y palabras muy cortas), unidas por OR."""
    tokens = FTS_TOKEN_RE.findall(ask.lower())
    tokens = [t for t in tokens if len(t) > 2 and t not in STOPWORDS]
    return " OR ".join(f'"{t}"' for t in tokens)


def lexical_matches(ask: str, limit: int = LEXICAL_FALLBACK_LIMIT) -> list[sqlite3.Row]:
    """Red de seguridad por palabra clave (FTS5/BM25): rescata Proyectos que
    comparten palabras literales con la pregunta (ej. "aulas") cuando la
    busqueda vectorial no encuentra nada confiable. No reemplaza al vector
    -- solo se usa cuando este ya fallo (ver find_matches)."""
    query = fts_query(ask)
    if not query:
        return []
    cursor = conn.cursor()
    rows = cursor.execute(
        """
        SELECT c.id, c.proyecto_id, p.title AS proyecto_title, cf.title AS campo_title, bm25(chunk_fts) AS score
        FROM chunk_fts
        JOIN chunk c ON c.id = chunk_fts.rowid
        JOIN proyecto p ON p.id = c.proyecto_id
        LEFT JOIN campo_formativo cf ON cf.id = p.campo_formativo_id
        WHERE chunk_fts MATCH ?
        ORDER BY score
        LIMIT ?
        """,
        (query, limit),
    ).fetchall()

    seen_proyectos = set()
    distinct_matches = []
    for row in rows:
        if row["proyecto_id"] in seen_proyectos:
            continue
        seen_proyectos.add(row["proyecto_id"])
        distinct_matches.append(row)
    return distinct_matches


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^\w\s]", "", text.lower()).strip()


def find_by_title(ask: str) -> dict | None:
    """Busca si la pregunta (o la entidad extraida por route_intent) nombra
    directamente el titulo de un Proyecto real (ej. "que es Juguemos con la
    lengua?"). La busqueda vectorial falla en este caso: midiendo distancias
    reales, el propio contenido del Proyecto 2 queda mas lejos (l2~0.87) del
    embedding de esa pregunta que chunks de otros Proyectos sin relacion
    (que ganaron con ~0.81) -- el titulo no se repite palabra por palabra en
    el cuerpo del texto, asi que el match por string contra proyecto.title
    es mas confiable aqui que el embedding."""
    ask_norm = normalize(ask)
    if not ask_norm:
        return None

    rows = conn.execute(
        """
        SELECT p.id AS proyecto_id, p.title AS proyecto_title, cf.title AS campo_title
        FROM proyecto p
        LEFT JOIN campo_formativo cf ON cf.id = p.campo_formativo_id
        """
    ).fetchall()

    best = None
    best_score = 0.0
    for row in rows:
        title_norm = normalize(row["proyecto_title"])
        if not title_norm:
            continue
        if title_norm in ask_norm:
            best = row
            best_score = 1.0
            break
        score = SequenceMatcher(None, ask_norm, title_norm).ratio()
        if score > best_score:
            best_score = score
            best = row

    if best is None or best_score < TITLE_MATCH_THRESHOLD:
        return None

    first_chunk = conn.execute(
        "SELECT id FROM chunk WHERE proyecto_id = ? ORDER BY id LIMIT 1",
        (best["proyecto_id"],),
    ).fetchone()
    if first_chunk is None:
        return None

    return {
        "id": first_chunk["id"],
        "proyecto_id": best["proyecto_id"],
        "proyecto_title": best["proyecto_title"],
        "campo_title": best["campo_title"],
    }


def find_best_chunk_in_proyecto(ask: str, proyecto_id: int) -> sqlite3.Row | None:
    """Busca el chunk mas cercano a `ask` por embedding, pero FILTRADO a un
    solo Proyecto ya identificado (via find_by_title o el texto citado).
    Sin este filtro, centrar la ventana en "el primer chunk del proyecto"
    (find_by_title) o en "lo que gano el BM25 en todo el libro"
    (lexical_matches) no necesariamente da el chunk mas relevante a la
    pregunta dentro de ESE proyecto -- y el BM25 sin filtro a veces trae un
    Proyecto vecino de ruido (comparte un par de palabras) en vez de
    quedarse en el que ya sabemos que es el correcto."""
    embedding_ask = embedding_model.encode([ask], normalize_embeddings=True)[0]
    query_blob = embedding_ask.astype("float32").tobytes()
    return conn.execute(
        """
        SELECT c.id, c.proyecto_id, p.title AS proyecto_title, cf.title AS campo_title,
               vec_distance_l2(cv.embedding, ?) AS distance
        FROM chunk c
        JOIN chunk_vec cv ON cv.rowid = c.id
        JOIN proyecto p ON p.id = c.proyecto_id
        LEFT JOIN campo_formativo cf ON cf.id = p.campo_formativo_id
        WHERE c.proyecto_id = ?
        ORDER BY distance
        LIMIT 1
        """,
        (query_blob, proyecto_id),
    ).fetchone()


def find_matches(ask: str, k: int = TOP_K) -> list[sqlite3.Row]:
    """Busca los k chunks mas cercanos y devuelve un match por Proyecto
    distinto (el mas cercano de cada uno), junto con su Campo formativo.
    Un proyecto secundario solo se incluye si su distancia no se aleja mas
    de DISTANCE_MARGIN respecto al mejor match; si no, es "ruido" que solo
    quedo en el top-k por no haber nada mejor, no un tema realmente
    relacionado con la pregunta."""
    embedding_ask = embedding_model.encode([ask], normalize_embeddings=True)[0]
    query_blob = embedding_ask.astype("float32").tobytes()
    cursor = conn.cursor()
    matches = cursor.execute(
        """
        SELECT c.id, c.proyecto_id, p.title AS proyecto_title, cf.title AS campo_title, distance
        FROM chunk_vec
        JOIN chunk c ON c.id = chunk_vec.rowid
        JOIN proyecto p ON p.id = c.proyecto_id
        LEFT JOIN campo_formativo cf ON cf.id = p.campo_formativo_id
        WHERE embedding MATCH ? AND k = ?
        ORDER BY distance
        """,
        (query_blob, k),
    ).fetchall()

    if not matches:
        return []

    best_distance = matches[0]["distance"]
    print(f"marches: {[m['proyecto_title'] for m in matches]}")
    print(f"Best distance: {best_distance:.4f}, max allowed: {best_distance * (1 + DISTANCE_MARGIN):.4f}")

    if best_distance > MAX_ACCEPTABLE_DISTANCE:
        # el vector no encontro nada confiable -- antes de rendirse, probar
        # coincidencia lexica literal (ver lexical_matches). Rescata casos
        # como "convivir"/"aulas" que el embedding no relaciona bien con
        # "convivencia" pero que comparten palabras con el proyecto real.
        lexical = lexical_matches(ask)
        if lexical:
            print(f"(vector no confiable, usando match lexico: {[m['proyecto_title'] for m in lexical]})")
        return lexical

    max_distance = best_distance * (1 + DISTANCE_MARGIN)

    seen_proyectos = set()
    distinct_matches = []
    for match in matches:
        if match["distance"] > max_distance:
            continue
        if match["proyecto_id"] in seen_proyectos:
            continue
        seen_proyectos.add(match["proyecto_id"])
        distinct_matches.append(match)
    return distinct_matches


def find_quote_location(quoted: str) -> sqlite3.Row | None:
    """Encuentra el chunk (con su pagina y Proyecto) donde aparece
    literalmente `quoted`. Se usa para preguntas de metadatos (pagina,
    proyecto al que pertenece una lectura) -- esas NO se pueden responder
    con el LLM porque build_prompt nunca le pasa page_start/page_end, asi
    que hay que resolverlas directo contra la tabla, igual que
    describe_structure() resuelve la estructura del libro sin pasar por
    embeddings ni LLM."""
    quoted_norm = normalize(quoted)
    rows = conn.execute(
        """
        SELECT c.page_start, c.page_end, p.title AS proyecto_title, cf.title AS campo_title, c.text
        FROM chunk c
        JOIN proyecto p ON p.id = c.proyecto_id
        LEFT JOIN campo_formativo cf ON cf.id = p.campo_formativo_id
        ORDER BY c.id
        """
    ).fetchall()
    match = None
    for row in rows:
        if quoted_norm in normalize(row["text"]):
            match = row
    return match


def describe_quote_location(quoted: str) -> str | None:
    match = find_quote_location(quoted)
    if match is None:
        return None
    if match["page_start"] == match["page_end"]:
        pages = f"la página {match['page_start']}"
    else:
        pages = f"las páginas {match['page_start']}-{match['page_end']}"
    campo = f" (Campo formativo: {match['campo_title']})" if match["campo_title"] else ""
    return f'"{quoted}" pertenece al Proyecto "{match["proyecto_title"]}"{campo}, en {pages}.'


def fetch_all_chunks(proyecto_id: int) -> list[sqlite3.Row]:
    cursor = conn.cursor()
    return cursor.execute(
        "SELECT id, text FROM chunk WHERE proyecto_id = ? ORDER BY id",
        (proyecto_id,),
    ).fetchall()


def fetch_window(proyecto_id: int, matched_chunk_id: int, window: int = WINDOW_SIZE) -> str:
    """Trae solo una ventana chica de chunks alrededor del que hizo match.
    Es el contexto por defecto: precisa y barata en tokens, ideal cuando la
    respuesta esta cerca de donde matcheo el vector."""
    rows = fetch_all_chunks(proyecto_id)
    ids = [row["id"] for row in rows]
    center = ids.index(matched_chunk_id)
    rows = rows[max(0, center - window):center + window]
    return "\n\n".join(row["text"] for row in rows)


# Las 22 fases de los tres esquemas pedagogicos del libro (ver FASE_ALIASES
# en build_index.py, que es donde chunk.fase se etiqueta con estos mismos
# nombres canonicos). Antes solo listaba las 9 de "Lenguajes" -- esta funcion
# nunca podia reconocer una fase pedida en un Proyecto de otro esquema (ej.
# "Sensibilización" o "Problemática").
FASE_NAMES = [
    # Esquema "Lenguajes"
    "Identificación", "Recuperación", "Planificación", "Acercamiento",
    "Comprensión y producción", "Reconocimiento", "Integración",
    "Difusión", "Consideración y avances",
    # Esquema "indagación científica"
    "Sensibilización", "Diseño y desarrollo de la indagación",
    "Construcción y comprobación", "Comunicación", "Autorreflexión",
    # Esquema "sociocrítico"
    "Problemática", "Identificamos el problema", "Encontramos el origen",
    "Propuestas a seguir", "Organizamos los pasos", "Seguir el camino",
    "Registro de experiencia", "Valorando mis pasos",
]
FASE_LOOKUP = {normalize(f): f for f in FASE_NAMES}


def extract_fase(ask: str) -> str | None:
    """Detecta si la pregunta nombra una de las 9 fases pedagogicas del
    libro (ver chunk.fase). Es mucho mas preciso que anclar por texto citado
    o por vector cuando lo que se pide es literalmente una fase -- ese dato
    ya esta estructurado y tageado con 100% de fidelidad dentro de cada
    Proyecto, no hay que adivinarlo (ver el caso "muestra recuperacion" que
    fetch_around_quote fallaba: sin comillas alrededor de "recuperacion",
    terminaba anclando en el titulo del proyecto en vez de la fase)."""
    ask_norm = normalize(ask)
    for fase_norm, fase in FASE_LOOKUP.items():
        if fase_norm in ask_norm:
            return fase
    return None


def fetch_fase(proyecto_id: int, fase: str) -> str | None:
    rows = conn.execute(
        "SELECT text FROM chunk WHERE proyecto_id = ? AND fase = ? ORDER BY id",
        (proyecto_id, fase),
    ).fetchall()
    if not rows:
        return None
    return "\n\n".join(row["text"] for row in rows)


def describe_proyecto_fases(proyecto_id: int) -> str | None:
    """Lista, en el orden en que aparecen, las fases presentes en un
    Proyecto junto con un fragmento inicial de cada una a modo de
    descripcion. No todos los Proyectos usan las 9 fases (ver el fix del
    tagger en build_index.py) -- esto solo lista las que SI aparecen para
    este Proyecto especifico."""
    rows = conn.execute(
        "SELECT fase, text FROM chunk WHERE proyecto_id = ? AND fase IS NOT NULL ORDER BY id",
        (proyecto_id,),
    ).fetchall()
    if not rows:
        return None
    seen: dict[str, str] = {}
    for row in rows:
        seen.setdefault(row["fase"], row["text"])
    lines = []
    for fase, text in seen.items():
        snippet = text[:180].rsplit(" ", 1)[0]
        lines.append(f"- {fase}: {snippet}...")
    return "\n".join(lines)


QUOTE_CONTEXT_BEFORE = 2
QUOTE_CONTEXT_AFTER = 2


def fetch_around_quote(proyecto_id: int, quoted: str) -> str | None:
    """Extrae directamente los chunks alrededor de la ULTIMA aparicion
    literal de `quoted` dentro del Proyecto, sin pasar por el LLM. Se usa la
    ultima aparicion porque en el libro el titulo de una lectura casi
    siempre queda pegado justo antes de su contenido (ej. "...¿Para que
    sirven los pies? El punal Tu primera mirada..."), no al principio.
    Devuelve None si el texto citado no aparece literal en ningun chunk del
    Proyecto -- en ese caso hay que recurrir al flujo normal via LLM."""
    rows = fetch_all_chunks(proyecto_id)
    quoted_norm = normalize(quoted)
    match_idx = None
    for i, row in enumerate(rows):
        if quoted_norm in normalize(row["text"]):
            match_idx = i
    if match_idx is None:
        return None
    window = rows[max(0, match_idx - QUOTE_CONTEXT_BEFORE):match_idx + QUOTE_CONTEXT_AFTER + 1]
    return "\n\n".join(row["text"] for row in window)


def fetch_full(proyecto_id: int, matched_chunk_id: int) -> str:
    """Trae todo el contenido de un Proyecto (o una ventana mas amplia
    alrededor del chunk que hizo match, si el Proyecto es demasiado
    grande). Fallback quando la ventana chica no alcanzo para responder."""
    rows = fetch_all_chunks(proyecto_id)
    if len(rows) > MAX_CONTEXT_CHUNKS:
        ids = [row["id"] for row in rows]
        center = ids.index(matched_chunk_id)
        rows = rows[max(0, center - FULL_FALLBACK_WINDOW):center + FULL_FALLBACK_WINDOW]
    return "\n\n".join(row["text"] for row in rows)


def search_context(ask: str, k: int = TOP_K, full: bool = False) -> tuple[str, list]:
    """Devuelve ("ok", [(proyecto, texto), ...]) si el contexto reunido cabe
    comodo, o ("too_broad", [(proyecto, campo_formativo), ...]) si la
    pregunta toca demasiados temas/caracteres, para pedirle al usuario que
    la acote en vez de generar una respuesta con contexto disperso.

    Por defecto (full=False) trae solo una ventana chica de chunks por
    match, para no gastar contexto de mas. Con full=True trae el Proyecto
    completo -- se usa como segunda pasada cuando la ventana chica no
    alcanzo para responder."""
    matches = None
    quoted = extract_quoted(ask)

    # Si la pregunta nombra un Proyecto real (citado o no), usarlo como
    # delimitador: filtrar la busqueda vectorial a SOLO ese Proyecto en vez
    # de buscar en todo el libro. Esto tambien evita el ruido de BM25 que
    # a veces trae un Proyecto vecino ademas del correcto.
    target_proyecto = find_by_title(ask)
    if not target_proyecto and quoted:
        lexical = lexical_matches(quoted)
        if lexical:
            target_proyecto = lexical[0]

    if target_proyecto:
        anchor = find_best_chunk_in_proyecto(ask, target_proyecto["proyecto_id"])
        if anchor:
            print(
                f'(proyecto delimitado: "{target_proyecto["proyecto_title"]}" '
                f"-- busqueda vectorial filtrada a ese proyecto, distance={anchor['distance']:.4f})"
            )
            matches = [anchor]
        else:
            matches = [target_proyecto]

    if not matches:
        matches = find_matches(ask, k)

    if not matches:
        return "not_found", []

    topics = [(m["proyecto_title"], m["campo_title"]) for m in matches]

    fetch = fetch_full if full else fetch_window
    contexts = [
        (m["proyecto_title"], fetch(m["proyecto_id"], m["id"]))
        for m in matches
    ]
    total_chars = sum(len(text) for _, text in contexts)

    if total_chars > MAX_TOTAL_CONTEXT_CHARS:
        return "too_broad", topics
    return "ok", contexts


NO_ANSWER_PHRASE = "No encontré esa información en el libro."

NO_ANSWER_RE = re.compile(
    r"no\s+encontr[eé]\s+esa\s+informacion|no\s+(lo\s+)?s[eé]\b|"
    r"no\s+tengo\s+(esa\s+)?informacion|no\s+cuento\s+con",
    re.IGNORECASE,
)


def build_prompt(ask: str, contexts: list[tuple[str, str]]) -> str:
    joined_context = "\n\n".join(
        f"### Proyecto: {title}\n{text}" for title, text in contexts
    )
    return (
        "Responde en espanol, usando UNICAMENTE la informacion de los fragmentos. "
        "Si la 'Pregunta' es en realidad un tema o titulo (no una pregunta directa), "
        "resume lo que dicen los fragmentos sobre ese tema. "
        "Si los fragmentos no responden la pregunta ni se relacionan con el tema, "
        f'responde EXACTAMENTE esto y nada mas: "{NO_ANSWER_PHRASE}". '
        "No inventes ni completes con conocimiento propio, y no mezcles otros "
        "idiomas en tu respuesta.\n\n"
        f"Fragmentos:\n{joined_context}\n\nPregunta: {ask}"
    )


# Routing por tool-calling en vez de regex de palabras disparadoras.
#
# Se probaron tres enfoques para decidir que accion tomar segun la
# pregunta: (1) regex de palabras clave -- funcionaba, pero cada frase no
# anticipada rompia algo nuevo (faltaba "puedes", "muestrame" sin el "me",
# acentos correctos como "qué"/"está" nunca matcheaban contra "que"/"esta"
# sin tilde); (2) similitud de embeddings contra frases de ejemplo, sin
# entrenar -- fallo (margenes de 0.002-0.06, 10/15); (3) fine-tuning
# dedicado de un clasificador (BatchAllTripletLoss) -- tambien fallo, por
# overfitting con tan pocos ejemplos (10/16, ver la seccion "Experiments"
# del README).
#
# La causa comun: todos intentaban ANTICIPAR cada forma posible de
# preguntar. Tool calling se lo delega al LLM, que ya entiende lenguaje
# natural sin que tengamos que enumerar variantes -- Qwen3 lo soporta
# nativo via el chat template (tools=...), sin fine-tuning ni regex.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "listar_proyectos",
            "description": (
                "Lista todos los Proyectos y Campos formativos del libro. "
                "Usar cuando piden la ESTRUCTURA completa del libro (ej. "
                "'cuales son los proyectos', 'muestra el indice'), NO cuando "
                "piden contenido filtrado por tema (ej. 'que proyectos "
                "tratan el bullying' -- eso es responder_contenido)."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mostrar_texto_citado",
            "description": (
                "Muestra el texto literal de una lectura, poema, fase o "
                "fragmento del libro TAL CUAL aparece, sin resumir ni "
                "parafrasear. Usar cuando piden 'muestra', 'cita', "
                "'despliega', 'enseñame', 'copia' un texto o fase especifica."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "entidad": {
                        "type": "string",
                        "description": "Nombre de la lectura, poema o Proyecto citado en la pregunta.",
                    },
                    "fase": {
                        "type": "string",
                        "description": (
                            "Nombre de la fase pedagogica si se pide una fase "
                            "especifica, tal como la nombra la pregunta (ej. "
                            "Identificacion, Recuperacion, Sensibilizacion, "
                            "Problematica -- el libro usa distintos nombres de "
                            "fase segun el Proyecto). Omitir si no aplica."
                        ),
                    },
                },
                "required": ["entidad"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "buscar_metadato",
            "description": (
                "Busca en que pagina o a que Proyecto pertenece una lectura "
                "o texto especifico del libro. Usar para 'en que pagina "
                "esta X' o 'a que proyecto pertenece X'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "entidad": {
                        "type": "string",
                        "description": "Nombre de la lectura o texto por el que se pregunta.",
                    },
                },
                "required": ["entidad"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "listar_fases_proyecto",
            "description": (
                "Lista TODAS las fases pedagogicas de un Proyecto (con una "
                "breve descripcion de cada una) -- no una fase en "
                "particular. Usar para 'que fases tiene X', 'muestra las "
                "fases de X (con descripcion)', 'cuales son las fases de "
                "X'. Si piden UNA sola fase especifica con su contenido "
                "completo, usar mostrar_texto_citado en su lugar."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "proyecto": {
                        "type": "string",
                        "description": "Nombre del Proyecto por el que se pregunta.",
                    },
                },
                "required": ["proyecto"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "responder_contenido",
            "description": (
                "Responde una pregunta de CONTENIDO sobre el libro (que es "
                "X, de que trata X, que plantea UNA fase especifica de X, "
                "que proyectos tratan sobre Z). Es la opcion por defecto "
                "cuando ninguna de las otras aplica."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def route_intent(tokenizer, model, ask: str) -> tuple[str, dict]:
    """Le pide al LLM que elija la herramienta correcta para responder
    `ask`, en vez de intentar cubrir cada frase posible con regex a mano.
    Devuelve (nombre_tool, argumentos); si el modelo no genera un tool_call
    valido, cae a "responder_contenido" (la ruta general, la mas segura por
    default)."""
    messages = [{
        "role": "user",
        "content": f"Elige la herramienta correcta para esta pregunta sobre un libro escolar:\n\n{ask}",
    }]
    inputs = tokenizer.apply_chat_template(
        messages,
        tools=TOOLS,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
        return_dict=True,
    ).to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        repetition_penalty=1.1,
    )
    new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True)

    match = TOOL_CALL_RE.search(raw)
    if not match:
        return "responder_contenido", {}
    try:
        call = json.loads(match.group(1))
    except json.JSONDecodeError:
        return "responder_contenido", {}
    return call.get("name", "responder_contenido"), call.get("arguments") or {}


def generate_answer(tokenizer, model, ask: str, contexts: list[tuple[str, str]]) -> str:
    input_text = build_prompt(ask, contexts)
    messages = [{"role": "user", "content": input_text}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
        return_dict=True,
    ).to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        repetition_penalty=1.1,
    )
    new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# Handlers -- uno por tool, todos reusan las funciones deterministas ya
# validadas (describe_structure, fetch_fase, fetch_around_quote,
# describe_quote_location, search_context+generate_answer). Lo unico que
# cambio es QUIEN decide cual usar (route_intent, no regex).

def handle_listar_proyectos(tokenizer, model, ask: str, args: dict) -> None:
    print(describe_structure())


def handle_buscar_metadato(tokenizer, model, ask: str, args: dict) -> None:
    entidad = (args.get("entidad") or "").strip()
    if not entidad:
        print(NO_ANSWER_PHRASE)
        return
    info = describe_quote_location(entidad)
    print(info if info else NO_ANSWER_PHRASE)


def handle_mostrar_texto_citado(tokenizer, model, ask: str, args: dict) -> None:
    entidad = (args.get("entidad") or "").strip()
    if not entidad:
        print(NO_ANSWER_PHRASE)
        return

    target = find_by_title(entidad)
    if not target:
        matches = lexical_matches(entidad)
        target = matches[0] if matches else None
    if not target:
        print(NO_ANSWER_PHRASE)
        return

    extracted = None
    fase_arg = (args.get("fase") or "").strip()
    if fase_arg:
        fase = extract_fase(fase_arg)
        if fase:
            extracted = fetch_fase(target["proyecto_id"], fase)
    if not extracted:
        extracted = fetch_around_quote(target["proyecto_id"], entidad)

    if extracted:
        print("Respuesta (texto citado tal cual del libro):")
        print(extracted)
    else:
        print(NO_ANSWER_PHRASE)


def handle_listar_fases_proyecto(tokenizer, model, ask: str, args: dict) -> None:
    proyecto_arg = (args.get("proyecto") or "").strip()
    if not proyecto_arg:
        print(NO_ANSWER_PHRASE)
        return

    target = find_by_title(proyecto_arg)
    if not target:
        matches = lexical_matches(proyecto_arg)
        target = matches[0] if matches else None
    if not target:
        print(NO_ANSWER_PHRASE)
        return

    info = describe_proyecto_fases(target["proyecto_id"])
    if info:
        print(f'Fases del proyecto "{target["proyecto_title"]}":')
        print(info)
    else:
        print(NO_ANSWER_PHRASE)


def handle_responder_contenido(tokenizer, model, ask: str, args: dict) -> None:
    status, payload = search_context(ask)

    if status == "not_found":
        print(NO_ANSWER_PHRASE)
        return

    if status == "too_broad":
        print("Tu pregunta toca varios temas del libro. Prueba acotarla a uno de estos:")
        for proyecto_title, campo_title in payload:
            campo_str = f" (Campo formativo: {campo_title})" if campo_title else ""
            print(f"  - {proyecto_title}{campo_str}")
        return

    contexts = payload
    print("Proyectos encontrados (ventana chica):", [title for title, _ in contexts])
    answer = generate_answer(tokenizer, model, ask, contexts)

    if NO_ANSWER_RE.search(answer):
        print("(La ventana chica no alcanzo, ampliando al Proyecto completo...)")
        status, payload = search_context(ask, full=True)
        if status == "ok":
            contexts = payload
            answer = generate_answer(tokenizer, model, ask, contexts)

    print("Respuesta:", answer)


TOOL_HANDLERS = {
    "listar_proyectos": handle_listar_proyectos,
    "buscar_metadato": handle_buscar_metadato,
    "mostrar_texto_citado": handle_mostrar_texto_citado,
    "listar_fases_proyecto": handle_listar_fases_proyecto,
    "responder_contenido": handle_responder_contenido,
}


def main():
    global conn, embedding_model

    parser = argparse.ArgumentParser(description="CLI de RAG sobre el libro indexado")
    parser.add_argument("--db", default=DEFAULT_DB, help="Base SQLite generada por build_index.py")
    parser.add_argument(
        "--embedder", default=DEFAULT_EMBEDDER,
        help="Modelo de embeddings (id de Hugging Face o ruta local); debe ser el mismo que construyo --db",
    )
    args = parser.parse_args()

    # Qwen3-8B no es un modelo restringido: el token solo hace falta para
    # evitar limites de descarga anonimos, asi que el login es opcional.
    token = environment.get("HUGGINGFACE_TOKEN")
    if token:
        login(token=token)

    conn = configure_database(args.db)
    embedding_model = SentenceTransformer(args.embedder)

    print(f"GPU disponible: {torch.cuda.is_available()}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16).to("cuda")

    while True:
        ask = input("Pregunta: ").strip()
        if not ask:
            continue

        tool_name, args = route_intent(tokenizer, model, ask)
        print(f"(tool: {tool_name}({args}))")
        handler = TOOL_HANDLERS.get(tool_name, handle_responder_contenido)
        handler(tokenizer, model, ask, args)


if __name__ == "__main__":
    main()
