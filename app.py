"""
Recomendador de papers (AW) — interfaz web.

Reutiliza los artefactos exportados por AW_experimento_main.ipynb:
  * mejor_extractor_rec_topK_D2VsgA.pkl   -> {"word2vec": Word2Vec(sg=1), "kmeans": KMeans}
  * mejor_extractor_clas_VLADcbow.pkl     -> {"word2vec": Word2Vec(sg=0), "kmeans": KMeans}
  * mejor_clasificador_LR_con_VLADcbow.pkl -> Pipeline(StandardScaler, LogisticRegression)

Flujo por consulta (igual que `recuperar_top_k` del notebook):
  limpiar(consulta) -> promedio Word2Vec-SG (D2VsgA) -> coseno contra el corpus -> top-k
  matriz de adyacencia = coseno entre los vectores de los top-k, diagonal = 0
Adicionalmente se estima el tópico de la consulta con VLADcbow + LR.

Uso (ver README.md):
  python app.py                       # corpus = dataset completo filtrado
  python app.py --corpus train        # corpus = mismo train_df del notebook (split reproducido)
  python app.py --data ruta.csv --port 8080
"""
import argparse
import hashlib
import os
import pickle
import re
import string
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

warnings.filterwarnings("ignore", category=FutureWarning)

BASE = Path(__file__).resolve().parent
DOWNLOADS = Path.home() / "Downloads"

# --- Configuración (mismos valores que la sección 1 del notebook) ---------------------------
SEED = 2026
TEST_SIZE = 0.20
VALID_SIZE = 0.20
VECTOR_SIZE = 64
N_CLUSTERS = 32
K_DEFECTO = 20
K_MAX = 60

def ruta_defecto(env, *candidatas):
    """Variable de entorno > primera candidata existente > primera candidata."""
    if os.getenv(env):
        return os.getenv(env)
    for c in candidatas:
        if Path(c).exists():
            return str(c)
    return str(candidatas[0])


MOD, DAT = BASE / "modelos", BASE / "data"
parser = argparse.ArgumentParser()
parser.add_argument("--data", default=ruta_defecto("AW_DATA", DAT / "dataset.csv", DAT / "dataset_final.csv", DOWNLOADS / "dataset_final.csv"))
parser.add_argument("--rec", default=ruta_defecto("AW_REC", MOD / "mejor_extractor_rec_topK_D2VsgA.pkl", DOWNLOADS / "mejor_extractor_rec_topK_D2VsgA.pkl"))
parser.add_argument("--ext-clas", default=ruta_defecto("AW_EXT_CLAS", MOD / "mejor_extractor_clas_VLADcbow.pkl", DOWNLOADS / "mejor_extractor_clas_VLADcbow.pkl"))
parser.add_argument("--clf", default=ruta_defecto("AW_CLF", MOD / "mejor_clasificador_LR_con_VLADcbow.pkl", DOWNLOADS / "mejor_clasificador_LR_con_VLADcbow.pkl"))
parser.add_argument("--corpus", choices=["todo", "train"], default=os.getenv("AW_CORPUS", "todo"))
parser.add_argument("--port", type=int, default=int(os.getenv("AW_PORT", "8000")))
args, _ = parser.parse_known_args()

faltan = [p for p in [args.data, args.rec, args.ext_clas, args.clf] if not Path(p).exists()]
if faltan:
    raise SystemExit("Faltan archivos (ver README, sección 'Datos y modelos'):\n  " + "\n  ".join(faltan))

# --- Preprocesamiento (idéntico a la celda `limpiar`) ----------------------------------------
import nltk

try:
    from nltk.corpus import stopwords
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords", quiet=True)
    from nltk.corpus import stopwords
stop_words = set(stopwords.words("english") + stopwords.words("spanish"))
_TABLA = str.maketrans("", "", string.punctuation + "¡¿«»“”‘’…–—")


def limpiar(texto):
    texto = str(texto).lower().translate(_TABLA)
    texto = re.sub(r"\s+", " ", texto).strip()
    return " ".join(p for p in texto.split() if p not in stop_words)


# --- Extractores (idénticos a FV_D2VsgA y FV_VLADcbow) ---------------------------------------
def promedio_w2v(textos, modelo):
    salida = []
    for texto in textos:
        vectores = [modelo.wv[p] for p in texto.split() if p in modelo.wv]
        salida.append(np.mean(vectores, axis=0) if vectores else np.zeros(VECTOR_SIZE))
    return np.asarray(salida, dtype=np.float32)


def vlad(textos, modelo, kmeans):
    salida = []
    for texto in textos:
        vectores = [modelo.wv[p] for p in texto.split() if p in modelo.wv]
        FV = np.zeros((N_CLUSTERS, VECTOR_SIZE), dtype=np.float32)
        if vectores:
            V = np.asarray(vectores)
            grupos = kmeans.predict(V)
            for c in np.unique(grupos):
                FV[c] = (V[grupos == c] - kmeans.cluster_centers_[c]).sum(axis=0)
        FV = FV.ravel()
        norma = np.linalg.norm(FV)
        salida.append(FV / norma if norma else FV)
    return np.asarray(salida, dtype=np.float32)


# --- Carga de modelos ------------------------------------------------------------------------
print("Cargando modelos…")
with open(args.rec, "rb") as f:
    w2v_sg = pickle.load(f)["word2vec"]
with open(args.ext_clas, "rb") as f:
    paquete_clas = pickle.load(f)
w2v_cbow, kmeans_cbow = paquete_clas["word2vec"], paquete_clas["kmeans"]
with open(args.clf, "rb") as f:
    clasificador = pickle.load(f)
print(f"  D2VsgA: sg={w2v_sg.sg}, vocab={len(w2v_sg.wv):,} | VLADcbow: sg={w2v_cbow.sg}, k={kmeans_cbow.n_clusters}")


# --- Corpus (secciones 2-4 del notebook) -----------------------------------------------------
def cargar_corpus():
    df = pd.read_csv(args.data).dropna(subset=["title", "abstract", "topic"]).copy()
    df = df[df["topic"] != "Otros / Multidisciplinario"]
    for c in ["title", "abstract", "topic"]:
        df[c] = df[c].astype(str).str.strip()
    df = df[(df["title"] != "") & (df["abstract"] != "") & (df["topic"] != "")].reset_index(drop=True)
    df["abstracts"] = df["abstract"].apply(limpiar)
    df = df[df["abstracts"].str.len() > 0].copy()
    if args.corpus == "train":
        from sklearn.model_selection import StratifiedGroupKFold
        dataset = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
        sep_test = StratifiedGroupKFold(n_splits=int(round(1 / TEST_SIZE)), shuffle=True, random_state=SEED)
        idx_tv, _ = next(sep_test.split(dataset, dataset.topic, groups=dataset.abstracts))
        tv = dataset.iloc[idx_tv]
        sep_val = StratifiedGroupKFold(n_splits=int(round((1 - TEST_SIZE) / VALID_SIZE)), shuffle=True, random_state=SEED)
        idx_tr, _ = next(sep_val.split(tv, tv.topic, groups=tv.abstracts))
        df = tv.iloc[idx_tr]
    return df.reset_index(drop=True)


def huella():
    h = hashlib.md5()
    for p in [args.data, args.rec]:
        st = os.stat(p)
        h.update(f"{p}|{st.st_size}|{st.st_mtime}".encode())
    h.update(args.corpus.encode())
    return h.hexdigest()[:10]


CACHE = BASE / "cache"
CACHE.mkdir(exist_ok=True)
clave = huella()
f_meta, f_fv = CACHE / f"meta_{args.corpus}_{clave}.pkl", CACHE / f"FV_D2VsgA_{args.corpus}_{clave}.npy"
if f_meta.exists() and f_fv.exists():
    print("Cargando corpus y matriz FV desde caché…")
    meta = pd.read_pickle(f_meta)
    M = np.load(f_fv)
else:
    print("Preparando corpus (solo la primera vez, puede tardar unos minutos)…")
    corpus = cargar_corpus()
    M = promedio_w2v(corpus["abstracts"].tolist(), w2v_sg)
    cols = [c for c in ["paper_id", "title", "abstract", "authors", "url", "year", "fuente", "topic"] if c in corpus]
    meta = corpus[cols].copy()
    meta.to_pickle(f_meta)
    np.save(f_fv, M)
normas = np.linalg.norm(M, axis=1)
M_norm = M / np.where(normas == 0, 1, normas)[:, None]
print(f"Corpus listo: {len(meta):,} papers ({args.corpus}) | matriz {M.shape}")


def _txt(v):
    return "" if pd.isna(v) else str(v)


# --- API -------------------------------------------------------------------------------------
app = Flask(__name__, static_folder=str(BASE / "static"))


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/info")
def info():
    return jsonify({"n_papers": int(len(meta)), "corpus": args.corpus, "extractor": "D2VsgA",
                    "clasificador": "VLADcbow + LR", "k_defecto": K_DEFECTO, "k_max": K_MAX,
                    "topicos": sorted(meta["topic"].unique().tolist())})


@app.get("/api/buscar")
def buscar():
    consulta = request.args.get("q", "").strip()
    k = max(2, min(int(request.args.get("k", K_DEFECTO)), K_MAX))
    if not consulta:
        return jsonify({"error": "Escribe una consulta."}), 400
    limpia = limpiar(consulta)
    tokens = limpia.split()
    en_vocab = [t for t in tokens if t in w2v_sg.wv]
    if not en_vocab:
        return jsonify({"error": "Ninguna palabra de la consulta está en el vocabulario del modelo. Prueba con otros términos (el corpus es mayormente en español)."}), 422

    q = promedio_w2v([limpia], w2v_sg)[0]
    q = q / (np.linalg.norm(q) or 1)
    scores = M_norm @ q
    top = np.argpartition(-scores, k - 1)[:k]
    top = top[np.argsort(-scores[top])]

    V = M_norm[top]
    A = V @ V.T  # coseno entre los top-k (igual a cosine_similarity)
    np.fill_diagonal(A, 0)

    proba = clasificador.predict_proba(vlad([limpia], w2v_cbow, kmeans_cbow))[0]
    orden = np.argsort(-proba)[:3]

    papers = []
    for pos, i in enumerate(top, 1):
        fila = meta.iloc[int(i)]
        papers.append({"pos": pos, "id": f"P{pos}", "score": float(scores[i]), "title": _txt(fila.get("title")),
                       "abstract": _txt(fila.get("abstract")), "authors": _txt(fila.get("authors")),
                       "year": int(fila["year"]) if "year" in fila and not pd.isna(fila["year"]) else None,
                       "url": _txt(fila.get("url")), "fuente": _txt(fila.get("fuente")), "topic": _txt(fila.get("topic"))})
    return jsonify({"consulta": consulta, "tokens": tokens, "en_vocab": en_vocab, "k": k, "papers": papers,
                    "adyacencia": np.round(A, 4).tolist(),
                    "topico_consulta": [{"topic": str(clasificador.classes_[j]), "p": float(proba[j])} for j in orden]})


if __name__ == "__main__":
    print(f"\n>> Abre http://127.0.0.1:{args.port}\n")
    app.run(host="127.0.0.1", port=args.port, debug=False)
