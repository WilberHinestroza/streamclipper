"""Clasificación del texto transcrito: ¿esta parte del stream es del JUEGO o
es CHARLA CON EL chat?

Sirve para dos cosas:

1. Evitar clips donde el streamer grita pero está peleando con el chat en vez
   de estar jugando: si alrededor del pico solo hay interacción con el chat y
   nada de vocabulario de juego, el clip se penaliza (ver
   `scoring.apply_chat_penalty`).
2. Detectar si el VOD entero parece ser de la categoría "Just Chatting" (el
   streamer NO está jugando): ahí esa penalización se apaga, porque en ese
   modo la charla con el chat SÍ es el contenido y un grito peleando con el
   chat es tan clippeable como uno durante una jugada.

Es heurístico por diseño: un archivo de video local no trae metadatos de la
categoría del stream, así que se infiere del vocabulario de lo que se dice.
Por eso las dos direcciones del error están balanceadas hacia "no penalizar
si hay duda": solo se penaliza cuando hay evidencia de charla con el chat y
CERO evidencia de juego dentro del clip.

Los mismos patrones se reutilizan para clasificar segmentos sueltos
(`classify_text` / `classify_segments`) y para medir evidencia en una ventana
de tiempo (`window_evidence`).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


def normalize_text(text: str) -> str:
    """minúsculas + sin tildes, para no perder matches por acentos
    (ej. 'murió' vs 'murio')."""
    text = text.lower()
    return "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )


# Marcadores de vocabulario de JUEGO (español e inglés). Se usa como evidencia
# POSITIVA ("esto es de juego, no penalizar"), así que conviene ser generoso:
# un falso positivo acá solo significa que un clip dudoso se conserva. Por eso
# están incluidas palabras algo genéricas como "vida" o "ganar".
GAME_MARKERS = (
    # Español — términos genéricos de juego
    "ronda", "rondas", "jugada", "jugadas", "partida", "partidas",
    "mapa", "mapas", "nivel", "niveles", "victoria", "derrota",
    "ganamos", "perdimos", "ganar", "perder", "muerte", "muerto", "muerta",
    "mato", "mate", "enemigo", "enemigos", "aliado", "aliados",
    "arma", "armas", "escudo", "municion", "balas", "recarga", "granada",
    "pistola", "escopeta", "sniper", "rifle", "dano", "vida", "pocion",
    "cofre", "inventario", "loot", "daño", "craftear", "crafting",
    # Español — juegos de equipo / shooters
    "clutch", "ace", "headshot", "kill", "kills", "ulti", "ultimate",
    "combo", "skill", "rush", "push", "eco", "aim", "spray", "peek",
    # Español — deportes (FIFA, etc.)
    "gol", "goool", "penal", "falta", "tarjeta", "arquero", "remate",
    "offside", "partido",
    # Minecraft
    "creeper", "nether", "bloques", "diamante", "aldeano", "enderman",
    "mazmorra", "raid", "dungeon",
    # Inglés
    "round", "rounds", "won", "win", "lost", "lose", "defeat", "victory",
    "death", "died", "killed", "enemy", "enemies", "teammate", "teammates",
    "squad", "match", "boss", "gun", "sword", "shield", "health", "heal",
    "healing", "chest", "mob", "spawn", "respawn", "revive", "potion",
    "damage", "armor",
)

# Marcadores de INTERACCIÓN CON EL CHAT / discusión con los espectadores.
# Acá SÍ conviene ser específico: son evidencia negativa (puede hacer que un
# clip se penalice), así que solo entran frases que difícilmente se dicen
# durante una jugada.
CHAT_MARKERS = (
    # Español — dirigirse al chat
    "chat", "el chat", "en el chat", "dice el chat", "según el chat",
    "ustedes", "ustedes dicen", "ustedes quieren", "ustedes dicen que",
    "chicos", "gente", "suscribete", "suscriptor", "suscriptores",
    "suscripcion", "suscripciones", "prime", "donacion", "donaciones",
    "donaste", "donador", "gracias por", "gracias a todos",
    "buenas noches", "buenos dias", "buenas tardes", "como estan",
    "buenas a todos", "que tal",
    # Español — preguntas / respuesta al chat
    "pregunta", "preguntas", "pregunten", "respondo", "responderte",
    "comenten", "comentarios", "lee el chat", "mira el chat",
    # Español — discusión / pelea con el chat
    "peleando con", "discutiendo", "discusion", "insult", "trolea", "troll",
    "haters", "me reclaman", "me critican", "me insultan", "segun ustedes",
    "porque me dicen", "por que me dicen",
    # Inglés
    "chat", "guys", "folks", "subscribers", "thank you for", "thanks for",
    "appreciate", "donation", "donated", "follow", "followers",
    "question", "questions", "ask me", "chat says", "why do you",
    "trolling", "trolls", "haters",
)


def _compile_patterns(markers: tuple[str, ...]) -> list[re.Pattern[str]]:
    """Precompila los marcadores como regex con límites de palabra.

    Los límites de palabra importan acá tanto como en transcribe.py: sin ellos
    "ace" haría match dentro de "hace" y "gol" dentro de "goloso", lo que
    contaminaría la evidencia con palabras de conversación normal.
    """
    patterns: list[re.Pattern[str]] = []
    seen: set[str] = set()
    for marker in markers:
        norm = normalize_text(marker).strip()
        if not norm or norm in seen:
            continue
        seen.add(norm)
        patterns.append(re.compile(r"(?<!\w)" + re.escape(norm) + r"(?!\w)"))
    return patterns


GAME_PATTERNS = _compile_patterns(GAME_MARKERS)
CHAT_PATTERNS = _compile_patterns(CHAT_MARKERS)


def classify_text(text: str) -> tuple[int, int]:
    """Devuelve `(game_hits, chat_hits)`: cuántos marcadores de cada tipo hay
    en el texto (palabras distintas, no ocurrencias)."""
    norm = normalize_text(text)
    if not norm:
        return (0, 0)
    game_hits = sum(1 for p in GAME_PATTERNS if p.search(norm))
    chat_hits = sum(1 for p in CHAT_PATTERNS if p.search(norm))
    return (game_hits, chat_hits)


@dataclass(frozen=True)
class SegmentLabel:
    """Un segmento de transcripción con sus conteos de evidencia."""

    start: float
    end: float
    game_hits: int
    chat_hits: int

    @property
    def label(self) -> str:
        if self.game_hits and not self.chat_hits:
            return "game"
        if self.chat_hits and not self.game_hits:
            return "chat"
        if self.game_hits and self.chat_hits:
            return "mixed"
        return "neutral"


def classify_segments(segments) -> list[SegmentLabel]:
    """Clasifica cada segmento de la transcripción (duck-typing sobre
    objetos con `.start`, `.end` y `.text`, ej. TranscriptSegment)."""
    labels: list[SegmentLabel] = []
    for seg in segments:
        game_hits, chat_hits = classify_text(seg.text)
        labels.append(
            SegmentLabel(
                start=float(seg.start),
                end=float(seg.end),
                game_hits=game_hits,
                chat_hits=chat_hits,
            )
        )
    return labels


@dataclass(frozen=True)
class WindowEvidence:
    """Evidencia de juego/charla dentro de una ventana de tiempo.

    `*_seconds` es cuánto tiempo (dentro de la ventana) se habló de cada cosa,
    y es lo que se usa como umbral: una sola palabra suelta pesa menos que un
    párrafo entero de discusión con el chat.
    """

    game_hits: int
    chat_hits: int
    game_seconds: float
    chat_seconds: float


def window_evidence(labels: list[SegmentLabel], start: float, end: float) -> WindowEvidence:
    """Suma la evidencia de los segmentos que se solapan con [start, end]."""
    game_hits = chat_hits = 0
    game_seconds = chat_seconds = 0.0
    for seg in labels:
        overlap = min(end, seg.end) - max(start, seg.start)
        if overlap <= 0:
            continue
        if seg.game_hits:
            game_hits += seg.game_hits
            game_seconds += overlap
        if seg.chat_hits:
            chat_hits += seg.chat_hits
            chat_seconds += overlap
    return WindowEvidence(
        game_hits=game_hits,
        chat_hits=chat_hits,
        game_seconds=game_seconds,
        chat_seconds=chat_seconds,
    )


def detect_stream_kind(labels: list[SegmentLabel]) -> str:
    """Infiere si el VOD es "gameplay" o "justchatting" a partir del
    vocabulario de TODA la transcripción.

    La regla es deliberadamente conservadora: solo se declara "justchatting"
    si el vocabulario de juego aparece en menos del 5% de los segmentos con
    evidencia y el del chat en al menos el 25%. Si hay duda (pocos segmentos
    etiquetados en general, o vocabulario mixto), se asume "gameplay" — que es
    el modo en el que el filtro de chat está activo y es el caso de uso
    principal de la herramienta.

    Se puede forzar a mano con `--stream-kind` (CLI) o el desplegable
    "Tipo de stream" (GUI) si la detección no coincide con tu VOD.
    """
    labeled = [s for s in labels if s.game_hits or s.chat_hits]
    if len(labeled) < 10:
        return "gameplay"
    game_n = sum(1 for s in labeled if s.game_hits)
    chat_n = sum(1 for s in labeled if s.chat_hits)
    total = len(labeled)
    if game_n * 20 <= total and chat_n * 4 >= total:
        return "justchatting"
    return "gameplay"
