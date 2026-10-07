"""Tests unitarios del pipeline de detección/recorte (sin ffmpeg ni video).

Corre de dos formas:

    python tests/test_units.py     # sin dependencias extra
    pytest tests/test_units.py     # si tenés pytest instalado

Cubre los dos problemas reportados sobre clips mal recortados:
  1. el recorte alrededor de keywords ("ace", "win", "derrota") debe mostrar
     la JUGADA, que pasa ANTES de que se diga la palabra, y
  2. los gritos que son peleas con el chat (y no del juego) deben quedar
     penalizados — salvo en streams Just Chatting.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from streamclipper.audio import EnergyProfile
from streamclipper.content import (
    SegmentLabel,
    classify_text,
    detect_stream_kind,
)
from streamclipper.scoring import (
    Candidate,
    apply_chat_penalty,
    candidates_from_anchors,
    grow_candidates,
    merge_overlapping,
    select_best,
)
from streamclipper.transcribe import (
    TranscriptSegment,
    find_keyword_hits,
    find_keyword_timestamps,
)

HOP = 0.5


def make_profile(duration: float = 120.0) -> tuple[EnergyProfile, np.ndarray]:
    times = np.arange(0.0, duration, HOP) + HOP / 2
    scores = np.full(len(times), 0.1)
    return EnergyProfile(times=times, rms=scores.copy(), window_seconds=1.0), scores


def mask(times: np.ndarray, t0: float, t1: float) -> np.ndarray:
    return (times >= t0) & (times <= t1)


# ------------------------------------------------------------------ keywords

def test_keyword_hit_time_is_inside_the_segment():
    # El bug original usaba seg.start: si la keyword estaba al final de un
    # segmento de 6 segundos, el recorte quedaba anclado ~6s antes de la
    # palabra. Ahora el tiempo cae dentro del segmento.
    segs = [TranscriptSegment(10.0, 16.0, "creo que fue un ace impresionante")]
    hits = find_keyword_hits(segs, keywords=["ace"])
    assert len(hits) == 1
    assert 10.0 < hits[0].time < 16.0
    assert hits[0].keyword == "ace"


def test_keyword_respects_word_boundaries():
    # "ace" no debe matchear dentro de "hace" (bug real histórico).
    segs = [TranscriptSegment(0.0, 4.0, "así hace calor hoy")]
    assert find_keyword_hits(segs, keywords=["ace"]) == []
    # compat: la función vieja sigue devolviendo floats
    assert find_keyword_timestamps(segs, keywords=["ace"]) == []


def test_anchor_candidate_covers_the_play_before_the_keyword():
    # La jugada pasa entre 40s y 52s; el streamer dice "ace" recién en 54s.
    profile, scores = make_profile()
    scores[mask(profile.times, 40.0, 52.0)] = 0.9

    cands = candidates_from_anchors(
        profile,
        scores,
        [(54.0, "keyword 'ace'")],
        min_score=0.5,
        pre_roll=15.0,
        post_roll=10.0,
    )
    assert len(cands) == 1
    cand = cands[0]
    # Lo esencial: la ventana arranca ANTES de que empiece la jugada.
    assert cand.start <= 40.0 + HOP
    assert cand.end >= 54.0 + 10.0 - HOP
    assert cand.score >= 0.85
    assert cand.reasons == ["keyword 'ace'"]


def test_anchor_candidate_is_rejected_when_nothing_happened():
    # Keyword dicha en calma, sin ningún pico de energía en la ventana:
    # el score (solo el boost, ~0.5) no llega a un min_score exigente.
    profile, scores = make_profile()
    kw_idx = int(np.argmin(np.abs(profile.times - 54.0)))
    scores[kw_idx] += 0.4  # el boost de keyword aplicado sobre base calmada

    cands = candidates_from_anchors(
        profile, scores, [(54.0, "keyword 'gg'")], min_score=0.7, pre_roll=15.0, post_roll=10.0
    )
    assert cands == []


def test_nearby_keywords_cluster_into_one_candidate():
    profile, scores = make_profile()
    scores[mask(profile.times, 40.0, 60.0)] = 0.8
    cands = candidates_from_anchors(
        profile,
        scores,
        [(54.0, "keyword 'ace'"), (57.0, "keyword 'vamos'")],
        min_score=0.5,
        pre_roll=15.0,
        post_roll=10.0,
    )
    assert len(cands) == 1
    assert set(cands[0].reasons) == {"keyword 'ace'", "keyword 'vamos'"}


# -------------------------------------------------------------------- crecer

def test_grow_extends_while_the_action_is_active():
    # Ventana inicial que arranca a mitad de la jugada: al estar la acción
    # todavía "caliente" (score alto sostenido), el borde debe retroceder
    # hasta el inicio real de la acción, y frenar ahí donde vuelve la calma.
    profile, scores = make_profile()
    scores[mask(profile.times, 30.0, 45.0)] = 0.6

    cand = Candidate(peak_time=44.0, score=0.9, start=35.0, end=45.0, reasons=[])
    grow_candidates([cand], profile.times, scores, video_duration=120.0)

    assert cand.start <= 30.0 + HOP  # cubre el arranque de la acción
    assert cand.start > 25.0         # pero no se pasa del piso de score
    assert cand.end <= 45.0 + HOP    # después de la acción no crece


def test_grow_is_bounded_by_max_window():
    profile, scores = make_profile()
    scores[:] = 0.8  # score alto sostenido por todo el video

    cand = Candidate(peak_time=60.0, score=0.9, start=55.0, end=65.0, reasons=[])
    grow_candidates([cand], profile.times, scores, max_window=60.0, video_duration=120.0)

    assert cand.end - cand.start <= 60.0 + HOP


# ------------------------------------------------------- juego vs. el chat

def test_classify_text_separates_game_from_chat():
    game_hits, chat_hits = classify_text("vamos, qué ronda más loca")
    assert game_hits > 0 and chat_hits == 0

    game_hits, chat_hits = classify_text("suscríbete si te gustó, gente del chat")
    assert chat_hits > 0 and game_hits == 0

    game_hits, chat_hits = classify_text("hola, ¿cómo andan?")
    assert (game_hits, chat_hits) == (0, 0)

    # sin límites de palabra, "ace" haría match en "hace"
    assert classify_text("así hace calor hoy") == (0, 0)


def test_detect_stream_kind():
    chat_only = [
        SegmentLabel(start=i * 5.0, end=i * 5.0 + 4.0, game_hits=0, chat_hits=2)
        for i in range(40)
    ]
    assert detect_stream_kind(chat_only) == "justchatting"

    mixed = [
        SegmentLabel(start=i * 5.0, end=i * 5.0 + 4.0, game_hits=2, chat_hits=0)
        for i in range(20)
    ] + [
        SegmentLabel(start=100.0 + i * 5.0, end=100.0 + i * 5.0 + 4.0, game_hits=0, chat_hits=2)
        for i in range(20)
    ]
    assert detect_stream_kind(mixed) == "gameplay"

    # Muy pocos datos = duda -> se asume gameplay (modo estricto)
    assert detect_stream_kind(chat_only[:3]) == "gameplay"


def test_chat_penalty_drops_a_clip_that_is_just_arguing():
    cand = Candidate(peak_time=50.0, score=0.95, start=42.0, end=66.0, reasons=["pico de energía"])
    labels = [SegmentLabel(start=48.0, end=54.0, game_hits=0, chat_hits=3)]

    penalized = apply_chat_penalty([cand], labels, [], "gameplay", penalty=0.6)
    assert penalized == 1
    assert cand.score == 0.35  # queda por debajo de un min_score típico (0.5)
    assert any("chat" in r for r in cand.reasons)


def test_chat_penalty_keeps_clips_with_game_evidence():
    game_labels = [SegmentLabel(start=48.0, end=54.0, game_hits=1, chat_hits=3)]
    chat_labels = [SegmentLabel(start=48.0, end=54.0, game_hits=0, chat_hits=3)]

    # Evidencia de juego dentro del clip: no se toca, aunque hable del chat.
    with_game = Candidate(peak_time=50.0, score=0.95, start=42.0, end=66.0, reasons=[])
    n = apply_chat_penalty([with_game], game_labels, [], "gameplay", penalty=0.6)
    assert n == 0
    assert with_game.score == 0.95

    # Texto en pantalla (OCR) dentro del clip también cuenta como juego.
    with_visual = Candidate(peak_time=50.0, score=0.95, start=42.0, end=66.0, reasons=[])
    n = apply_chat_penalty([with_visual], chat_labels, [51.0], "gameplay", penalty=0.6)
    assert n == 0
    assert with_visual.score == 0.95

    # Sin transcripción en la ventana: ante la duda no se penaliza.
    no_evidence = Candidate(peak_time=50.0, score=0.95, start=42.0, end=66.0, reasons=[])
    n = apply_chat_penalty([no_evidence], [], [], "gameplay", penalty=0.6)
    assert n == 0
    assert no_evidence.score == 0.95


def test_chat_penalty_is_disabled_in_justchatting():
    cand = Candidate(peak_time=50.0, score=0.95, start=42.0, end=66.0, reasons=[])
    labels = [SegmentLabel(start=48.0, end=54.0, game_hits=0, chat_hits=3)]

    n = apply_chat_penalty([cand], labels, [], "justchatting", penalty=0.6)
    assert n == 0
    assert cand.score == 0.95


# --------------------------------------------------------------- selección

def test_select_best_prefers_higher_score_and_respects_gap():
    low = Candidate(peak_time=10.0, score=0.6, start=0.0, end=26.0, reasons=[])
    high = Candidate(peak_time=12.0, score=0.9, start=2.0, end=30.0, reasons=[])
    far = Candidate(peak_time=90.0, score=0.7, start=80.0, end=106.0, reasons=[])

    chosen = select_best([low, high, far], max_clips=5, min_score=0.5, min_gap=25.0)
    assert [c.score for c in chosen] == [0.9, 0.7]  # low se pisa con high

    penalized = Candidate(peak_time=50.0, score=0.35, start=40.0, end=60.0, reasons=[])
    chosen = select_best([penalized, far], max_clips=5, min_score=0.5, min_gap=25.0)
    assert chosen == [far]


def test_merge_respects_max_span():
    # Tres ventanas encadenadas no deben convertirse en un clip eterno.
    a = Candidate(peak_time=10.0, score=0.6, start=0.0, end=40.0, reasons=[])
    b = Candidate(peak_time=50.0, score=0.7, start=35.0, end=75.0, reasons=[])
    merged = merge_overlapping([a, b], max_span=60.0)
    assert len(merged) == 2  # la unión (75s) supera max_span


def test_merge_requires_a_real_overlap():
    # Solapes chicos = momentos distintos: mejor dos clips que uno largo.
    a = Candidate(peak_time=10.0, score=0.6, start=0.0, end=30.0, reasons=[])
    b = Candidate(peak_time=40.0, score=0.7, start=27.0, end=57.0, reasons=[])
    assert len(merge_overlapping([a, b])) == 1  # por defecto fusiona
    # OJO: merge_overlapping muta los candidatos, por eso se recrean.
    a = Candidate(peak_time=10.0, score=0.6, start=0.0, end=30.0, reasons=[])
    b = Candidate(peak_time=40.0, score=0.7, start=27.0, end=57.0, reasons=[])
    assert len(merge_overlapping([a, b], min_overlap=5.0)) == 2  # solape de 3s: separados


def test_full_flow_recorta_la_jugada_completa():
    """Punta a punta: jugada -> grito -> keyword, más un pico que es solo
    charla con el chat, aplicando el flujo del CLI."""
    profile, scores = make_profile()
    scores[mask(profile.times, 40.0, 52.0)] = 0.9     # la jugada + grito
    kw_idx = int(np.argmin(np.abs(profile.times - 54.0)))
    scores[kw_idx] += 0.4                              # boost de "ace"

    peaks = [  # candidato "pico" típico: centrado en el grito, corta la jugada
        Candidate(peak_time=53.0, score=0.9, start=45.0, end=71.0, reasons=["pico de energía"])
    ]
    anchors = candidates_from_anchors(
        profile, scores, [(54.0, "keyword 'ace'")],
        min_score=0.5, pre_roll=15.0, post_roll=10.0,
    )
    candidates = merge_overlapping(peaks + anchors, max_span=75.0)
    grow_candidates(candidates, profile.times, scores, video_duration=120.0)
    candidates = merge_overlapping(candidates, max_span=75.0)
    apply_chat_penalty(
        candidates,
        [SegmentLabel(start=54.0, end=58.0, game_hits=1, chat_hits=0)],
        [],
        "gameplay",
        penalty=0.6,
    )
    chosen = select_best(candidates, max_clips=5, min_score=0.5, min_gap=25.0)

    assert len(chosen) == 1
    best = chosen[0]
    assert best.start <= 40.0 + HOP   # empieza con la jugada
    assert best.end >= 64.0           # incluye la reacción posterior
    assert best.score >= 0.85


def _run_all():
    tests = [
        (name, fn)
        for name, fn in sorted(globals().items())
        if name.startswith("test_") and callable(fn)
    ]
    for name, fn in tests:
        fn()
        print(f"OK  {name}")
    print(f"\n{len(tests)} tests OK")


if __name__ == "__main__":
    _run_all()
