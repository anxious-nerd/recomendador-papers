# Recomendador de papers (AW)

Interfaz web para el modelo de recuperación entrenado en `AW_experimento_main.ipynb`.
Escribes una consulta en lenguaje natural, presionas **Enter** y obtienes:

1. Los **top-k papers recomendados**, ordenados por similitud coseno con la consulta.
2. El **tópico estimado de la consulta**, con sus probabilidades.
3. Un **grafo ponderado** que representa la matriz de adyacencia entre los k papers, con una vista alternativa como **matriz/mapa de calor**.

## Inicio rápido

Todos los comandos se ejecutan **desde la carpeta del proyecto** (donde está `app.py`), no desde la carpeta del entorno.

1. Entra a la carpeta del proyecto:

```bash
cd ruta/a/recomendador-papers-aw
```

2. Crea el entorno virtual (solo la primera vez):

```bash
python -m venv .venv
```

3. Actívalo. En Windows (PowerShell):

```bash
.venv\Scripts\Activate.ps1
```

En Windows (cmd):

```bash
.venv\Scripts\activate.bat
```

En Linux/macOS:

```bash
source .venv/bin/activate
```

4. Instala las dependencias (solo la primera vez):

```bash
pip install -r requirements.txt
```

5. Coloca los 3 `.pkl` en `modelos/` y el CSV en `data/dataset.csv` (ver [Datos y modelos](#datos-y-modelos)).

6. Levanta la app:

```bash
python app.py
```

7. Abre **http://127.0.0.1:8000**, escribe una consulta y presiona Enter. Para detener el servidor usa `Ctrl+C`.

Si ya tienes un entorno creado en otro lugar (por ejemplo con conda), puedes saltarte los pasos 2–4 y llamar directamente a su Python desde la carpeta del proyecto:

```bash
ruta/al/entorno/python.exe app.py
```

## Cómo funciona

| Paso | Componente | Detalle |
|---|---|---|
| Preprocesamiento | `limpiar()` | Minúsculas, sin puntuación, sin *stopwords* ES/EN (NLTK). Idéntico al notebook. |
| Recuperación | **D2VsgA** (`mejor_extractor_rec_topK_D2VsgA.pkl`) | Promedio de vectores Word2Vec Skip-gram (64 dims), similitud coseno contra los abstracts del corpus. Fue el ganador en nDCG@20 de validación (0.6047). |
| Matriz de adyacencia | coseno entre los top-k | `A[i,j] = cos(v_i, v_j)`, diagonal = 0. Es la misma matriz que la celda "Experimento 1" del notebook. |
| Tópico de la consulta | **VLADcbow + LR** (`mejor_extractor_clas_VLADcbow.pkl`, `mejor_clasificador_LR_con_VLADcbow.pkl`) | VLAD de 32 clusters × 64 dims sobre Word2Vec CBOW, con StandardScaler + Regresión Logística. Accuracy test: 0.8243. |

**Cómo leer el grafo**
- Cada nodo es un paper (P1…Pk).
- El color indica el tópico del paper.
- El tamaño del nodo es proporcional a su similitud con la consulta.
- El grosor y la cercanía de las aristas son proporcionales al peso (coseno). Como los pesos suelen estar en un rango estrecho (~0.88–0.98), se normalizan entre el mínimo y el máximo solo para el dibujo. Los valores reales se ven al pasar el mouse.
- El deslizador muestra el top-X% de aristas más fuertes. Además, cada nodo conserva su arista más fuerte para que ninguno quede aislado.

## Estructura

```
.
├── app.py              # Backend Flask + API (/api/info, /api/buscar)
├── static/index.html   # Interfaz (D3.js v7 desde cdnjs)
├── requirements.txt
├── iniciar.bat         # Arranque rápido en Windows
├── iniciar.sh          # Arranque rápido en Linux/macOS
├── modelos/            # Aquí van los 3 .pkl (no versionados por tamaño)
├── data/               # Aquí va dataset.csv (no versionado por tamaño)
└── cache/              # Se genera solo: vectores del corpus precalculados
```

## Datos y modelos

Por su tamaño, los modelos y el dataset **no se suben al repositorio** (ver `.gitignore`). Colócalos así:

```
modelos/mejor_extractor_rec_topK_D2VsgA.pkl
modelos/mejor_extractor_clas_VLADcbow.pkl
modelos/mejor_clasificador_LR_con_VLADcbow.pkl
data/dataset.csv
```

- Los `.pkl` se exportan en la última celda de exportación del notebook (`pickle.dump(...)`).
- `data/dataset.csv` es el mismo CSV que usa el notebook (`DATA_PATH`). Debe tener las columnas `paper_id, title, abstract, authors, url, year, source, fuente, topic`.
- También se acepta `data/dataset_final.csv`. Cualquier ruta puede pasarse por argumento o por variable de entorno (ver *Opciones*).

> Los modelos (~84 MB c/u) sí entran en el límite de 100 MB de GitHub. Para versionarlos conviene usar
> [Git LFS](https://git-lfs.com) (`git lfs track "modelos/*.pkl"`) o adjuntarlos en un *Release*.
> El dataset (~265 MB) supera ese límite y debe distribuirse aparte (Drive, Release, etc.).

## Instalación

Requiere **Python 3.10–3.12**. Las versiones de `numpy>=2` y `scikit-learn==1.6.1` son necesarias porque los modelos se serializaron en Colab con esas versiones. Con NumPy 1.x, los `.pkl` de Word2Vec fallan con `MT19937 is not a known BitGenerator`.

**Con venv**

```bash
python -m venv .venv
```

Activar el entorno en Windows (PowerShell):

```bash
.venv\Scripts\Activate.ps1
```

Activar el entorno en Linux/macOS:

```bash
source .venv/bin/activate
```

Instalar dependencias:

```bash
pip install -r requirements.txt
```

**Con conda**

```bash
conda create -y -n aw python=3.12 pip
```

```bash
conda activate aw
```

```bash
pip install -r requirements.txt
```

## Ejecución

```bash
python app.py
```

Luego abre **http://127.0.0.1:8000**.

En Windows también puedes hacer doble clic en `iniciar.bat`; en Linux/macOS usa `./iniciar.sh`. Ambos usan `.venv` si existe y abren el navegador.

- **Primer arranque:** tarda unos minutos (~5 min para ~145k papers) porque limpia todos los abstracts y calcula sus vectores D2VsgA. El resultado se guarda en `cache/`.
- **Siguientes arranques:** toman segundos. La caché se invalida sola si cambia el CSV, el modelo o el modo de corpus.
- La primera vez también se descargan las *stopwords* de NLTK si no están instaladas.
- La interfaz carga D3 desde cdnjs, así que el navegador necesita internet.

### Opciones

| Argumento | Variable de entorno | Por defecto | Descripción |
|---|---|---|---|
| `--data` | `AW_DATA` | `data/dataset.csv` | CSV del corpus |
| `--rec` | `AW_REC` | `modelos/mejor_extractor_rec_topK_D2VsgA.pkl` | Extractor de recuperación |
| `--ext-clas` | `AW_EXT_CLAS` | `modelos/mejor_extractor_clas_VLADcbow.pkl` | Extractor de clasificación |
| `--clf` | `AW_CLF` | `modelos/mejor_clasificador_LR_con_VLADcbow.pkl` | Clasificador |
| `--corpus` | `AW_CORPUS` | `todo` | `todo`: todo el dataset filtrado. `train`: reproduce el `train_df` del notebook (StratifiedGroupKFold, `SEED=2026`), útil para comparar con los resultados del experimento. |
| `--port` | `AW_PORT` | `8000` | Puerto |

Ejemplo:

```bash
python app.py --corpus train --port 8080
```

### API

- `GET /api/info`: tamaño del corpus, modelos y lista de tópicos.
- `GET /api/buscar?q=<consulta>&k=<2..60>`: devuelve `papers` (pos, score, title, abstract, authors, year, url, fuente, topic), `adyacencia` (matriz k×k), `topico_consulta` (top-3 con probabilidad) y `en_vocab` (términos reconocidos por el modelo).

## Limitaciones

- El vocabulario Word2Vec es mayormente en español. Si ninguna palabra de la consulta está en el vocabulario, la API responde con un aviso en lugar de resultados.
- La relevancia se evaluó con *proxy* de tópico (nDCG@20 ≈ 0.60) y con una evaluación LLM pequeña. Conviene que un experto valide las recomendaciones antes de usarlas en conclusiones.
- Servidor de desarrollo de Flask: pensado para uso local o demostración, no para producción.

---
*Interfaz desarrollada con asistencia de IA generativa (Claude). Los modelos y el pipeline provienen de `AW_experimento_main.ipynb`.*
