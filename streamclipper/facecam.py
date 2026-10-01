"""Detección OPCIONAL de la posición de la cámara (facecam) en streams
horizontales, para armar un layout vertical con la cámara arriba y el
gameplay abajo (`--vertical-layout cam-top`).

Es una capa aparte, igual que la detección visual por OCR (`visual.py`):
usa `opencv-python-headless` (importado de forma perezosa) solo para
detectar caras con un clasificador Haar Cascade que YA viene incluido
dentro de OpenCV — no descarga nada de internet ni necesita un modelo
aparte. Si no está instalado, el programa avisa y sigue con el recorte
vertical centrado normal, o el usuario puede indicar la región de la
cámara a mano con `--cam-region` (sin necesidad de esta detección).
"""

from __future__ import annotations

Region = tuple[float, float, float, float]


def _get_face_cascade():
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "opencv-python-headless no está instalado. Instálalo con "
            "`pip install opencv-python-headless` para detectar tu cámara "
            "automáticamente, o indica la región a mano con --cam-region."
        ) from exc

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    return cv2, cascade


def detect_camera_region(
    video_path: str,
    sample_count: int = 12,
    padding: float = 1.0,
    min_hits: int = 3,
    cluster_tolerance: float = 0.12,
) -> Region | None:
    """Intenta encontrar la región de la cámara (facecam) muestreando varios
    frames a lo largo del video y buscando caras.

    Se asume que la cámara está en una posición FIJA durante todo el video
    (lo normal en un stream) — por eso se agrupan las detecciones por
    posición en pantalla en vez de quedarse con la primera cara que
    aparezca, que podría ser un falso positivo dentro del propio juego (un
    personaje, un póster, etc.). El grupo de detecciones que se repite de
    forma más consistente en el mismo lugar es, con bastante confianza, la
    cámara real.

    Devuelve `(x, y, w, h)` como fracción [0,1] del frame, con un padding
    extra alrededor de la cara detectada (la ventana de cámara real suele
    mostrar más que solo la cara: hombros, parte del fondo del cuarto,
    etc.), o `None` si no se encuentra nada suficientemente consistente —
    en ese caso, quien llama debería avisar y usar el recorte centrado
    normal, o pedirle al usuario que indique la región a mano.
    """
    cv2, cascade = _get_face_cascade()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return None

    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration = (frame_count / fps) if frame_count > 0 else 0.0
    if duration <= 0:
        cap.release()
        return None

    detections: list[tuple[float, float, float, float]] = []
    margin = duration * 0.05  # evita los primeros/últimos segundos (intros/outros)
    span = max(0.0, duration - 2 * margin)
    for i in range(sample_count):
        t = margin + span * (i / max(1, sample_count - 1))
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        for (x, y, fw, fh) in faces:
            detections.append((x / w, y / h, fw / w, fh / h))

    cap.release()

    if len(detections) < min_hits:
        return None

    # Agrupa detecciones por cercanía del centro: la cámara real produce
    # detecciones repetidas casi en el mismo lugar a través de varios
    # frames, mientras que un falso positivo dentro del juego tiende a
    # aparecer en posiciones distintas cada vez (o no repetirse).
    clusters: list[list[tuple[float, float, float, float]]] = []
    for det in detections:
        cx, cy = det[0] + det[2] / 2, det[1] + det[3] / 2
        placed = False
        for cluster in clusters:
            ref = cluster[0]
            rcx, rcy = ref[0] + ref[2] / 2, ref[1] + ref[3] / 2
            if abs(cx - rcx) < cluster_tolerance and abs(cy - rcy) < cluster_tolerance:
                cluster.append(det)
                placed = True
                break
        if not placed:
            clusters.append([det])

    best = max(clusters, key=len)
    if len(best) < min_hits:
        return None

    x = sum(d[0] for d in best) / len(best)
    y = sum(d[1] for d in best) / len(best)
    w = sum(d[2] for d in best) / len(best)
    h = sum(d[3] for d in best) / len(best)

    # La ventana de cámara real casi siempre muestra bastante más que solo
    # la cara (hombros, torso, parte del fondo del cuarto) — la cara típica
    # ocupa solo una parte de esa ventana. Se agranda la caja detectada para
    # compensar, con más margen hacia abajo (hombros/torso) que hacia arriba
    # (apenas un poco de frente/pelo).
    pad_w = w * padding
    top_pad = h * padding * 0.4
    bottom_pad = h * padding * 1.3
    bottom = y + h + bottom_pad
    x = max(0.0, x - pad_w / 2)
    y = max(0.0, y - top_pad)
    w = min(1.0 - x, w + pad_w)
    h = min(1.0 - y, bottom - y)

    return (round(x, 4), round(y, 4), round(w, 4), round(h, 4))
