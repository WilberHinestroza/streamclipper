# StreamClipper

Herramienta para sacar clips destacados automáticamente de tus VODs (Twitch,
Kick, YouTube, TikTok o grabaciones locales). Pensada para streams de
videojuegos variados (Minecraft, Valorant, FIFA, etc.), porque detecta
momentos por **energía de audio** (gritos, risas, reacciones fuertes) en vez
de depender de reglas específicas de un juego.

## Cómo funciona

1. Extrae el audio del video con `ffmpeg` y calcula una curva de energía
   (RMS) a lo largo de todo el VOD.
2. Convierte esa energía en un score de "qué tan excepcional es este momento
   comparado con su propio contexto cercano" (los ~90 segundos alrededor),
   no comparado contra todo el video de una sola vez. Esto es importante:
   el audio de fondo de un juego (música, disparos, ambiente) puede sonar
   fuerte y parejo durante horas, así que comparar contra un solo nivel
   global hacía que la sensibilidad casi no cambiara nada. Comparando cada
   momento contra su entorno inmediato, un grito real destaca aunque el
   juego ya esté sonando fuerte de fondo.
3. Encuentra los picos de ese score con un espaciado mínimo entre ellos para
   no generar clips pegados.
4. (Opcional) Transcribe el audio con `faster-whisper` para reforzar el
   score cuando se detectan frases típicas de hype ("no way", "vamos",
   "gg", "insano", etc.), **anclar el recorte del clip a esos momentos**
   (ver la sección de keywords abajo) y generar subtítulos quemados en el
   clip.
5. (Opcional) Clasifica lo transcrito en "juego" vs "charla con el chat" para
   separar gritos por una jugada de gritos porque el streamer está peleando
   con el chat (ver sección de abajo).
6. Corta cada momento en un clip de video independiente con `ffmpeg`,
   opcionalmente en formato vertical 9:16 (TikTok/Reels/Shorts) y con
   subtítulos.
7. Genera un `manifest.json` con el detalle de cada clip (inicio, fin,
   score, y el motivo por el que fue elegido) para que puedas revisar o
   automatizar el resto del flujo (publicar, renombrar, etc.).

## Instalación

Requiere Python 3.10+ y `ffmpeg`/`ffprobe` instalados en el sistema
(en Windows: https://ffmpeg.org/download.html, o `winget install ffmpeg`).

```bash
pip install -r requirements.txt

# Opcional, solo si quieres subtítulos y detección de keywords:
pip install faster-whisper

# Opcional, solo si quieres detección visual (texto en pantalla, ej. "You Died"):
pip install rapidocr-onnxruntime pillow

# Opcional, solo si quieres que detecte tu cámara (facecam) automáticamente
# para el vertical "cámara arriba + gameplay abajo":
pip install opencv-python-headless
```

> Nota sobre `faster-whisper`: la primera vez que lo uses (botón "Transcribir"
> en la GUI, o `--transcribe` en la terminal) va a descargar el modelo de
> internet (entre ~75MB para `tiny` y varios GB para `large-v3`). Necesitas
> conexión a internet en ese primer uso; después queda guardado localmente y
> las siguientes veces es instantáneo. Si la descarga falla (sin internet,
> firewall, etc.), el programa avisa y sigue generando los clips solo con la
> detección por audio, no se cae.

## Uso con interfaz gráfica (recomendado si no te gusta la terminal)

Hay una interfaz gráfica simple (hecha con tkinter, ya viene con Python, no
requiere instalar nada extra) donde eliges el video, la carpeta de salida,
el número de clips, la sensibilidad, formato vertical, subtítulos, etc.

En Windows, la forma más fácil es hacer **doble clic en `StreamClipper.bat`**
(está en la raíz del proyecto). También puedes abrirla desde la terminal:

```bash
python -m streamclipper.gui
```

La ventana muestra el progreso en tiempo real (los mismos pasos `[1/6]...[6/6]`
que ves en la terminal, con barra de progreso real durante el escaneo visual
y la exportación de clips) y al terminar te deja abrir la carpeta de clips
con un botón.

## Uso por línea de comandos

```bash
python -m streamclipper.cli mi_vod.mp4 --out-dir clips --num-clips 8
```

Esto genera hasta 8 clips en la carpeta `clips/`, ordenados por score
(el más "hype" primero), más un `manifest.json` con los timestamps.

> Nota: si vuelves a correr el programa apuntando a la misma carpeta de
> salida con un número de clips menor que la vez anterior, los clips viejos
> que ya no correspondan (ej. `clip_03`, `clip_04`) no se borran solos —
> bórralos a mano o usa una carpeta de salida distinta por corrida si eso
> te genera confusión.

## Opciones útiles

| Flag | Qué hace |
|---|---|
| `--num-clips N` | Máximo de clips a generar (default 8) |
| `--min-score X` | Umbral de sensibilidad [0-1]. Más bajo = más clips, algunos menos intensos (default 0.5) |
| `--fast` | Corte rápido sin recodificar. Más veloz pero el clip puede arrancar en el keyframe más cercano (a veces se ve entrecortado al inicio). Por defecto siempre recodifica para evitar ese problema |
| `--min-gap S` | Segundos mínimos entre dos clips distintos (default 25) |
| `--pre-roll S` / `--post-roll S` | Cuánto contexto incluir antes/después del pico de energía (default 8s / 18s) |
| `--keyword-pre-roll S` / `--keyword-post-roll S` | Lo mismo, pero para los clips anclados a una palabra clave de la transcripción (default 15s antes / 10s después). El default es asimétrico a propósito: "ace", "win" o "derrota" se dicen DESPUÉS de la jugada, así que lo que hay que mostrar está antes de la palabra. Requiere `--transcribe` |
| `--stream-kind auto\|gameplay\|justchatting` | Si el VOD es de juego o de la categoría Just Chatting. `auto` (default) lo infiere de la transcripción. En Just Chatting los gritos de charla con el chat SÍ son clippeables |
| `--chat-penalty X` | Cuánto restarle al score de un clip donde el grito es peleando/charlando con el chat y no hay nada de juego en el transcript (default 0.6, que suele dejarlo por debajo de `--min-score`). `0` desactiva el filtro. Necesita `--transcribe` y no aplica en `justchatting` |
| `--vertical` | Exporta en 1080x1920 para TikTok/Shorts/Reels |
| `--vertical-fit blur\|crop` | Cómo se adapta el video 16:9 al lienzo 9:16. `blur` (default) = el gameplay entra **completo**, con una copia de fondo ampliada y desenfocada llenando el resto (sin barras negras, sin perder juego). `crop` = el recorte centrado de siempre |
| `--transcribe` | Activa transcripción con whisper (más lento, necesita `faster-whisper`) |
| `--captions` | Quema subtítulos en el clip (requiere `--transcribe`) |
| `--whisper-model` | tiny/base/small/medium/large-v3 (default base; más grande = más preciso y más lento) |
| `--no-timestamp` | No crear la subcarpeta con fecha/hora de la corrida dentro de `--out-dir` (por defecto sí se crea, ej. `clips/20260829_0911`) |
| `--no-normalize` | No normalizar el volumen (loudness) de los clips (por defecto sí se normaliza a -14 LUFS) |
| `--no-thumbnails` | No generar una miniatura `.jpg` por cada clip (por defecto sí se genera) |
| `--visual-detect` | Activa la detección visual por OCR (ver sección de abajo). Necesita `rapidocr-onnxruntime` |
| `--visual-keywords` | Palabras a buscar en pantalla, separadas por coma (ej. `"you died,has muerto"`) |
| `--visual-interval` | Segundos entre cada frame analizado por OCR (default 3.0). Más bajo = más preciso y más lento |
| `--visual-region` | Región de pantalla a mirar, `"x,y,w,h"` en fracciones 0-1 (ej. `"0.25,0.25,0.5,0.5"` = el centro). Vacío = pantalla completa |
| `--hype-keywords` | Palabras/frases propias para el boost por audio, separadas por coma (ej. `"insano,que locura,vamos"`). Vacío = usar la lista por defecto en español/inglés. Requiere `--transcribe` |
| `--vertical-layout` | Diseño del vertical (requiere `--vertical`). `crop` (default) = recorte centrado. `cam-top` = tu cámara en una mitad y el gameplay completo en la otra |
| `--cam-region` | Región de tu cámara como `"x,y,w,h"` en fracciones 0-1 (ej. `"0.02,0.02,0.28,0.35"` = esquina superior izquierda). Vacío = detectarla automáticamente buscando tu cara. Solo aplica con `--vertical-layout cam-top` |
| `--cam-position` | `top` (default) o `bottom`: en qué mitad del vertical va la cámara con `--vertical-layout cam-top` |

Cada corrida deja sus clips en su propia subcarpeta con fecha y hora (ej.
`clips/20260829_0911/`), así que puedes correr el programa varias veces
sobre el mismo VOD (probando distinta sensibilidad, por ejemplo) sin que una
corrida sobrescriba o se mezcle con la anterior. Además de los `.mp4`, cada
clip trae su miniatura `.jpg` y, si activaste `--transcribe`, un
`suggested_title` en el `manifest.json` con un título sugerido armado a
partir de la transcripción de ese momento (sin usar ninguna IA externa,
solo el texto que ya transcribió whisper).

### `log.txt` — registro completo de cada corrida

Cada vez que corres el programa (desde la terminal o desde la interfaz
gráfica), se guarda un archivo **`log.txt`** dentro de esa misma subcarpeta
de salida (ej. `clips/20260829_0911/log.txt`) con una copia exacta de todo
lo que se imprimió durante esa corrida: fecha y hora, el comando ejecutado,
cada paso `[1/6]...[6/6]`, avisos (por ejemplo si falta `faster-whisper` o
si no se encontraron subtítulos para un clip), y — si el programa llegó a
fallar con un error — el traceback completo del error, no solo el mensaje
corto que ves en la consola.

Esto sirve para diagnosticar problemas sin tener que copiar y pegar a mano
lo que aparece en la terminal o en el cuadro de texto de la GUI: si algo no
sale como esperabas (por ejemplo subtítulos que no aparecen, o pocos clips
generados), simplemente abre o comparte el `log.txt` de esa corrida.

Ejemplo completo, para Shorts con subtítulos:

```bash
python -m streamclipper.cli mi_vod.mp4 --out-dir clips --vertical --transcribe --captions --whisper-model small
```

## Probar sin tener un VOD a mano

Hay un generador de video sintético para validar que todo funciona en tu
máquina antes de usar un VOD real de varias horas:

```bash
python tests/make_test_video.py test_video.mp4
python -m streamclipper.cli test_video.mp4 --out-dir clips_test --min-score 0.3
```

Debería detectar 4 "momentos hype" cerca de los segundos 15, 45, 90 y 140.

Hay además dos suites de tests que no necesitan ffmpeg ni un VOD real (la
segunda sí exporta clips del video sintético, así que tarda un poco más):

```bash
python tests/test_units.py        # recorte por keywords, crecimiento de ventanas, filtro de chat
python tests/test_cli_stubbed.py  # el CLI completo con transcripción simulada
```

## Detección visual (OCR) — experimental

Además del audio, el programa puede "leer" texto que aparece en pantalla
dentro del juego — por ejemplo la pantalla de **"You Died"** en Minecraft —
y usar eso como una señal extra y bastante fuerte de que ahí hay un momento
para clippear. Funciona muestreando frames del video (uno cada X segundos,
configurable) y corriendo reconocimiento de texto (OCR) sobre cada uno.

**Qué tan bien funciona hoy, en criollo:**
- Para eventos que muestran texto fijo y legible en pantalla (la pantalla de
  muerte de Minecraft es el caso que probé y funciona bien) es una señal
  confiable.
- Para detectar **tus kills específicamente en un shooter** (distinguir que
  fuiste TÚ el que mató, y contar múltiples kills en una ronda) todavía no
  está implementado de verdad — el kill-feed de la mayoría de los shooters
  muestra nombres de jugadores en vez de una palabra fija, así que hace
  falta calibrarlo con capturas reales de tu juego (¿cómo se ve tu kill-feed
  exactamente? ¿en qué esquina? ¿qué fuente?). Si quieres que sigamos con
  eso, la mejor forma de avanzar es que compartas un clip corto o una
  captura de pantalla mostrando cómo se ve una kill/muerte en tu juego.
- Las palabras a buscar dependen del idioma y la versión de tu juego — las
  que vienen por defecto en cada perfil son un punto de partida, no una
  lista completa. Si notas que no detecta algo, prueba agregando la frase
  exacta que aparece en tu pantalla al campo de "Palabras a buscar".

**Es lento:** a diferencia del análisis de audio (que tarda segundos), esto
analiza el video frame por frame — en mis pruebas, del orden de 2-3 frames
por segundo de procesamiento. Para un VOD de 2 horas con un frame cada 3
segundos, son ~2400 frames a analizar, lo que puede tomar bastantes minutos.
Por eso viene **desactivado por defecto** — actívalo solo cuando quieras
más precisión y estés dispuesto a esperar.

## Bugs corregidos (a partir de un log.txt real)

Analizando un `log.txt` real de una corrida larga (VOD de más de 3 horas,
con transcripción y detección visual activadas) aparecieron dos problemas
más, además del de las keywords de audio (ver más abajo):

**1. El programa se cortaba a mitad de exportar los clips en Windows.**
Al llegar al clip 7 de 8, el programa reventaba con
`UnicodeEncodeError: 'charmap' codec can't encode characters...`. La causa:
la consola de Windows en español suele usar un code page antiguo (cp1252)
que no puede representar todos los caracteres — y algún texto salido de
whisper o del OCR (comillas curvas, símbolos, etc.) caía fuera de ese
repertorio. Como se imprimía tanto en la consola como en el `log.txt`, un
solo carácter raro en la consola tiraba abajo TODO el programa, perdiendo
los clips que faltaban por exportar. Ahora la salida de consola se fuerza
a UTF-8 (reemplazando cualquier carácter no representable en vez de
reventar), y además cada escritura está protegida individualmente — así
que esto no debería volver a cortar una corrida a mitad de camino.

**2. La detección visual (OCR) generaba cientos de "hits" con overlays de
HUD permanentes** (por ejemplo, un panel de estadísticas de TikTok Live
que muestra "Muertes: ¡Has muerto!" como texto fijo en pantalla, no como
aviso puntual). Antes, cada frame muestreado donde ese texto seguía visible
contaba como un hit nuevo — en un stream con ese tipo de HUD, eso generó
173 "coincidencias" y un boost de score casi constante, en vez de marcar
momentos puntuales. Ahora solo cuenta como hit el momento en que la
keyword **aparece** en pantalla (pasa de no estar a estar, comparado con
el frame muestreado anterior), no cada frame en que sigue visible. Un
overlay permanente ahora genera como mucho un hit; una pantalla de muerte
real que aparece y desaparece cada vez sigue generando un hit por cada
muerte, que es el comportamiento que se busca.

## Bug corregido: clips de charla normal marcados con score=1.00

Analizando un `log.txt` real (streams largos y muy hablados en español)
encontramos dos bugs que hacían que, con transcripción activada, salieran
clips de conversación normal en vez de los momentos realmente más intensos
— y encima todos con `score=1.00`, por eso la sensibilidad parecía no
importar en esos casos:

1. **La detección de keywords buscaba "substring", no palabra completa.**
   Una keyword corta como `"ace"` hacía match dentro de palabras comunes en
   español como **"hace"** o **"nace"** — palabras que aparecen todo el
   tiempo en cualquier conversación, sin tener nada que ver con hype. En un
   VOD de 3+ horas muy hablado esto disparaba cientos de "keywords de hype"
   falsas. Ahora se usa coincidencia de palabra completa (con límites de
   palabra) y sin distinguir tildes.
2. **El boost de keywords se sumaba sin límite.** Si el jugador hablaba
   rápido y dos o tres keywords (aunque fueran reales) caían cerca en el
   tiempo, sus boosts se sumaban entre sí (`0.4 + 0.4 + 0.4...`) y saturaban
   el score a 1.0 sin importar si el audio en ese momento era intenso o no.
   Ahora se toma el máximo de esos boosts, no la suma, así que una keyword
   nunca puede aportar más que su valor base a un punto del score.

Con estos dos fixes, `--min-score` vuelve a discriminar de verdad incluso
en streams con mucha transcripción. Si vienes de una corrida donde casi
todos tus clips salían con `score=1.00` y no te convencían, vale la pena
volver a correrlo — probablemente ahora te dé una selección bastante
distinta (y probablemente necesites bajar el `--min-score` que usabas
antes, ya que ya no está inflado artificialmente).

También se sacaron de la lista de keywords por defecto un par de palabras
demasiado genéricas (`"eso es"`, `"bro"`) que eran muy comunes como
muletillas normales de conversación y no solo en momentos de hype. Si te
sirven para tu forma de hablar, puedes agregarlas de vuelta con
`--hype-keywords` (ver tabla de opciones arriba) — ese campo también está
disponible en la interfaz gráfica, en la sección de transcripción.

## Recortes alrededor de keywords y gritos que no son del juego

Dos problemas reportados con el generador de clips, y cómo se resolvieron:

### 1. Los clips con "ace", "win", "derrota" no mostraban la jugada

El recorte viejo dependía SOLO de los picos de energía: `pico - pre_roll ..
pico + post_roll` (8s antes / 18s después). El problema es que una palabra
así se dice **después** de la jugada (el streamer reacciona al resultado), y
encima `find_peaks` se queda con un único máximo local por zona — si el grito
caía a varios segundos de la palabra, la ventana quedaba anclada al grito y
la jugada previa quedaba cortada o directamente afuera. Resultado típico:
"se ven unos pocos segundos antes y el resto después, sin la jugada".

Ahora hay tres cambios que trabajan juntos:

1. **El timestamp de la keyword es el de la palabra**, no el inicio del
   segmento de whisper (que puede ser de varios segundos y poner el ancla
   mucho antes de donde se dijo realmente).
2. **Se arma un candidato DIRECTAMENTE anclado a la keyword** (y a cada texto
   en pantalla detectado por OCR), con ventana asimétrica: 15s antes / 10s
   después por defecto (`--keyword-pre-roll` / `--keyword-post-roll`). Su
   score es el máximo de la curva (con el boost ya aplicado) dentro de esa
   ventana, así que una keyword dicha en calma no genera clip por sí sola.
3. **La ventana crece mientras dure la acción** (`grow_candidates`): desde el
   recorte base, cada borde se extiende mientras el score se mantenga alto
   (con tope de 20s más hacia atrás, 12s hacia adelante y 60s de duración
   total). Así una ronda que empieza 25s antes de que digas "derrota" queda
   entera, y el clip se corta en cuanto vuelve la calma — en vez de depender
   de un número fijo de segundos.

Además, dos ventanas que se solapan poco ya no se fusionan en un clip eterno
(hay un `max_span` de 75s y un solape mínimo de 5s para fusionar): dos
momentos distintos a pocos segundos dan dos clips, no uno largo con los dos
adentro. Cada clip del `manifest.json` ahora trae un campo `reasons` con el
motivo por el que fue elegido ("pico de energía", "keyword 'ace'", "texto en
pantalla", etc.), y esos motivos también se imprimen en el log.

### 2. Gritos que en realidad son peleas con el chat

La energía de audio no distingue un grito por una jugada de un grito porque
el streamer está discutiendo con el chat: son el mismo volumen. La
transcripción sí lo distingue, así que ahora cada candidato se contrasta con
lo que se dijo en su ventana:

- Si hay **evidencia de juego** (vocabulario de juego en el transcript, o
  texto en pantalla detectado por OCR) → se conserva.
- Si hay **charla con el chat alrededor del pico y CERO evidencia de
  juego** → se le resta `--chat-penalty` (0.6 por defecto) al score, lo que
  normalmente lo manda por debajo de `--min-score` y queda afuera. Ese
  "alegando con el chat" no sobrevive; un "CHAT NOOO" gritado durante una
  jugada, en cambio, suele venir con vocabulario de juego en la ventana y se
  queda.
- Si no hay transcripción en el rango, **no se penaliza** (ante la duda, no
  se descarta un clip bueno).

Y la excepción que pediste: **en streams Just Chatting no se penaliza nada**,
porque ahí la charla con el chat ES el contenido. Como un archivo local no
trae la categoría del stream, `--stream-kind auto` (el default) la infiere del
vocabulario de toda la transcripción — solo se declara "justchatting" si el
vocabulario de juego aparece en menos del 5% de los segmentos con evidencia y
el del chat en al menos el 25%; con dudas se asume gameplay. Si se equivoca
en tu VOD, se fuerza con `--stream-kind justchatting` (en la GUI: el
desplegable "Tipo de stream", junto a la casilla "Evitar clips donde el grito
es peleando/charlando con el chat").

Ojo: este filtro (como la detección de keywords y los subtítulos) necesita
`--transcribe` activado. Sin transcripción no hay forma de saber de qué se
habla en el clip, y el programa lo avisa en el log.

## Vertical 9:16 sin perder gameplay (`--vertical-fit blur`)

Antes, activar el vertical (checkbox "Formato vertical 9:16" en la GUI, o
`--vertical` en la terminal) aplicaba un **recorte centrado** al centro del
frame. Para pasar de 16:9 a 9:16 eso recorta a una columna angosta: en un
gameplay 1920x1080 el clip final conserva el 100% del alto pero apenas el
**32% del ancho** del juego — se perdían los bordes izquierdo y derecho, que
en la mayoría de los juegos es justo donde está el HUD, el inventario y los
jugadores enemigos. Además el recorte se upscaleaba a 1080 de ancho, así que
además de perder contenido se veía borroso.

Ahora el default es `blur`: el gameplay entra **completo** (escalado a caber
con `force_original_aspect_ratio=decrease`) y el espacio sobrante del 9:16 se
llena con una copia del mismo frame ampliada y **desenfocada**, en lugar de
recortarse o dejar barras negras. Es el mismo recurso que usan las apps
verticales para adaptar contenido horizontal.

Detalles de la implementación:

- **El fondo sale barato.** En vez de aplicar un desenfoque fuerte a un frame
  de 1080x1920 (que es de lo más caro en CPU), primero se reduce a 1/4 de
  resolución y después se vuelve a escalar al tamaño final: ese reescalado
  *es* el desenfoque, y sale ~16x más barato.
- **Los subtítulos se reposicionan solos.** Con el fondo desenfocado, los
  subtítulos quemados por defecto caerían sobre el fondo borroso de abajo, no
  sobre el juego. El programa lee las dimensiones del video con `ffprobe` y
  calcula el `MarginV` necesario para apoyarlos sobre el gameplay. No se usa
  un margen fijo porque el hueco depende del aspecto de origen: en un 16:9 es
  de ~650px y en un video ya vertical es de ~0 (en ese caso no se mueve nada,
  para no sacar el texto de cuadro).
- Si el video de origen ya es 9:16, el juego llena el lienzo entero y el
  fondo desenfocado no llega a verse: el resultado es un recorte limpio.

Para volver al comportamiento anterior:

```bash
python -m streamclipper.cli mi_vod.mp4 --vertical --vertical-fit crop
```

En la GUI es la casilla "Fondo desenfocado: el gameplay entra COMPLETO en el
9:16", dentro de la sección de formato vertical.

## Vertical con cámara arriba y gameplay abajo (`--vertical-layout cam-top`)

Si grabas horizontal con tu cámara (facecam) como una ventanita superpuesta,
esta opción arma el vertical 9:16 con tu cámara ocupando una mitad
(1080x960) y el gameplay completo ocupando la otra mitad, en vez del
recorte centrado normal que puede dejar tu cara fuera de cuadro.

La mitad del gameplay usa el mismo tratamiento de "entra completo sobre fondo
desenfocado" que el vertical normal, pero contra un lienzo de 1080x960: el
juego se ve entero en su mitad, en vez de recortarse al centro (que dentro de
una tan baja también empujaba parte del juego fuera de cuadro).

```bash
# Detecta tu cámara sola (necesita opencv-python-headless instalado):
python -m streamclipper.cli mi_vod.mp4 --vertical --vertical-layout cam-top

# O indicas la región a mano si prefieres no depender de la detección:
python -m streamclipper.cli mi_vod.mp4 --vertical --vertical-layout cam-top \
  --cam-region "0.02,0.02,0.28,0.35" --cam-position top
```

**Cómo funciona la detección automática:** muestrea varios frames a lo
largo del video buscando caras con un detector clásico (Haar Cascade, viene
incluido en OpenCV, no descarga nada de internet ni necesita GPU). Como se
asume que tu cámara está en una posición fija durante todo el stream (lo
normal), agrupa las detecciones por posición en pantalla y se queda con la
que se repite de forma más consistente — así evita confundirse con una
cara que aparezca una sola vez dentro del propio juego.

**Limitaciones, en criollo:**
- La caja que devuelve la detección automática es una **estimación
  centrada en tu cara**, agrandada con margen para incluir hombros/torso —
  pero no sabe dónde termina realmente la ventanita de tu cámara (eso
  depende de cómo la configuraste en OBS/Streamlabs/etc.). Si el resultado
  te queda muy ajustado o con margen de más, usa `--cam-region` con la
  medida exacta.
- Para medir tu región a mano: abre un frame del video en cualquier
  visor/editor de imágenes, fíjate en qué píxel empieza y termina tu
  ventanita de cámara, y divide esas coordenadas por el ancho/alto total
  del video (ej. cámara en `(40,30)` a `(360,330)` en un video de
  `1280x720` → `--cam-region "0.031,0.042,0.25,0.417"`).
- Si tu cámara es circular, muy pequeña, o cambia de posición durante el
  stream, la detección automática probablemente no la encuentre bien —
  usa `--cam-region` en esos casos.
- Como con la detección visual (OCR), esto se probó con un video sintético
  (una foto real superpuesta sobre gameplay de prueba) para validar que el
  mecanismo funciona de punta a punta, pero no con facecams reales de
  distintos streamers/setups. Si la posición te queda mal calibrada con tu
  cámara real, cuéntame cómo se ve (o compárteme un frame) para seguir
  ajustando la detección.
- El gameplay de la otra mitad se muestra completo (no se recorta la
  cámara de esa mitad), así que tu ventanita de cámara puede seguir
  viéndose chiquita en su esquina original dentro de esa mitad — es a
  propósito, más simple y robusto que intentar "borrarla" del gameplay.

## Ideas para siguientes pasos (no implementadas aún)

- Detección de kills propias en shooters (matching por tu nombre de usuario
  en el kill-feed, contando múltiples kills por ronda) — necesita capturas
  reales de tu juego para calibrar la región de pantalla y el formato.
- Integrar el chat de Twitch/Kick (mensajes por segundo) como señal extra
  de score, sumado a la energía de audio y la detección visual.
- Generar automáticamente una miniatura con texto/thumbnail llamativo por
  clip (más allá del frame simple que se genera hoy).
- Recorte inteligente para vertical (seguir la cara/acción en vez de
  recorte centrado fijo).

## Notas de rendimiento

- El análisis de energía es rápido (segundos, incluso para streams largos).
- `--transcribe` es el paso más lento de audio: con CPU, un modelo `base`
  procesa aproximadamente en tiempo real o algo más rápido; `small`/`medium`
  son más precisos pero más lentos. Si tienes GPU NVIDIA, instala
  `faster-whisper` con soporte CUDA para acelerarlo bastante.
- `--visual-detect` es, en general, el paso más lento de todos (ver sección
  de arriba) — puede tomar varios minutos en VODs largos.
- Los VODs de varias horas generan archivos de audio grandes en memoria
  (el pipeline carga el audio completo como PCM). Para streams muy largos
  (6h+) puede convenir dividir el VOD por partes primero.
