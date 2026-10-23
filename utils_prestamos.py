"""Funciones de la Parte A (balanceo + reducción de dimensiones). Usadas por app.py y por el notebook."""
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             balanced_accuracy_score, confusion_matrix, roc_curve)
from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek

TECNICAS = ["Sin balanceo", "class_weight", "Under-sampling", "Over-sampling", "SMOTE", "ADASYN", "SMOTE+Tomek"]


def cargar(ruta_o_archivo, objetivo="class"):
    """Lee CSV/CSV.GZ/XLSX, quita duplicados y separa X, y."""
    nombre = getattr(ruta_o_archivo, "name", str(ruta_o_archivo)).lower()
    df = pd.read_excel(ruta_o_archivo) if nombre.endswith(("xlsx", "xls")) else pd.read_csv(ruta_o_archivo)
    n0 = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    return df.drop(columns=objetivo), df[objetivo].astype(int), n0 - len(df)


def particionar(X, y, test_size=0.30, seed=42):
    """Partición estratificada + estandarización ajustada SOLO con train (evita fuga de datos)."""
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    sc = StandardScaler().fit(Xtr)
    return sc.transform(Xtr), sc.transform(Xte), ytr.values, yte.values


def _modelo(nombre, cw, seed, n_arboles):
    if nombre == "Regresión logística":
        return LogisticRegression(max_iter=3000, class_weight=cw)
    return RandomForestClassifier(n_estimators=n_arboles, min_samples_leaf=2, class_weight=cw, random_state=seed, n_jobs=-1)


def _muestreador(tecnica, seed):
    return {"Under-sampling": RandomUnderSampler(random_state=seed), "Over-sampling": RandomOverSampler(random_state=seed),
            "SMOTE": SMOTE(random_state=seed), "ADASYN": ADASYN(random_state=seed),
            "SMOTE+Tomek": SMOTETomek(random_state=seed)}.get(tecnica)


def gmean(y, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return float(np.sqrt((tp / max(tp + fn, 1)) * (tn / max(tn + fp, 1))))


def evaluar(modelo, tecnica, Xtr, ytr, Xte, yte, seed=42, n_arboles=150, umbral=0.5):
    """Entrena con una técnica de balanceo (solo sobre TRAIN) y evalúa en TEST sin balancear."""
    t0 = time.time()
    cw = "balanced" if tecnica == "class_weight" else None
    Xa, ya = Xtr, ytr
    s = _muestreador(tecnica, seed)
    if s is not None:
        Xa, ya = s.fit_resample(Xtr, ytr)
    m = _modelo(modelo, cw, seed, n_arboles).fit(Xa, ya)
    p = m.predict_proba(Xte)[:, 1]
    pred = (p >= umbral).astype(int)
    tn, fp, fn, tp = confusion_matrix(yte, pred, labels=[0, 1]).ravel()
    res = {"Técnica": tecnica, "Accuracy": accuracy_score(yte, pred), "Precision": precision_score(yte, pred, zero_division=0),
           "Recall": recall_score(yte, pred), "F1": f1_score(yte, pred), "AUC": roc_auc_score(yte, p),
           "Balanced Acc": balanced_accuracy_score(yte, pred), "G-mean": gmean(yte, pred),
           "VP": int(tp), "FP": int(fp), "FN": int(fn), "VN": int(tn), "Filas de entrenamiento": len(ya),
           "Tiempo (s)": time.time() - t0}
    return res, p


def comparar_tecnicas(modelo, tecnicas, Xtr, ytr, Xte, yte, seed=42, n_arboles=150):
    filas, probs = [], {}
    for t in tecnicas:
        r, p = evaluar(modelo, t, Xtr, ytr, Xte, yte, seed, n_arboles)
        filas.append(r)
        probs[t] = p
    return pd.DataFrame(filas), probs


def costo_total(df, c_fn, c_fp):
    """Costo esperado en test: FN = quiebra/impago no detectado; FP = cliente bueno rechazado."""
    return df["FN"] * c_fn + df["FP"] * c_fp


def curva_pca(Xtr):
    pca = PCA().fit(Xtr)
    return pca.explained_variance_ratio_, np.cumsum(pca.explained_variance_ratio_)


def comparar_pca(modelo, tecnica, Xtr, ytr, Xte, yte, varianzas=(0.80, 0.90, 0.95, 0.99), seed=42, n_arboles=150):
    """Compara el modelo original (todas las variables) vs PCA (retener X % de varianza). PCA se ajusta en train."""
    _, cum = curva_pca(Xtr)
    filas = []
    r, _ = evaluar(modelo, tecnica, Xtr, ytr, Xte, yte, seed, n_arboles)
    r.update({"Escenario": "Original (sin PCA)", "Varianza retenida": 1.0, "Nº variables": Xtr.shape[1]})
    filas.append(r)
    for v in varianzas:
        k = int(np.searchsorted(cum, v) + 1)
        P = PCA(n_components=k, random_state=seed).fit(Xtr)
        r, _ = evaluar(modelo, tecnica, P.transform(Xtr), ytr, P.transform(Xte), yte, seed, n_arboles)
        r.update({"Escenario": f"PCA {int(v * 100)} %", "Varianza retenida": float(cum[k - 1]), "Nº variables": k})
        filas.append(r)
    return pd.DataFrame(filas)
