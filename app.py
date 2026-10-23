"""Parte A · Clasificación con datos desbalanceados y reducción de dimensiones (Streamlit).
Ejecutar local:  streamlit run app.py
"""
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import roc_curve

import utils_prestamos as u

st.set_page_config(page_title="Préstamos · Desbalanceo y PCA", page_icon="🏦", layout="wide")
st.title("🏦 Riesgo de préstamos: datos desbalanceados y reducción de dimensiones")
st.caption("EC2 · Criterio 3 · Parte A — Modelamiento Predictivo de Datos")

# ---------------- Barra lateral ----------------
with st.sidebar:
    st.header("Datos")
    f = st.file_uploader("datasetPrestamos (.xlsx / .csv / .csv.gz)", type=["xlsx", "csv", "gz"])
    st.header("Modelo")
    modelo = st.selectbox("Algoritmo", ["Random Forest", "Regresión logística"])
    n_arb = st.slider("Árboles (Random Forest)", 50, 300, 100, 50, disabled=modelo != "Random Forest")
    test_size = st.slider("% de prueba", 0.2, 0.4, 0.3, 0.05)
    seed = st.number_input("Semilla", 0, 9999, 42)
    st.header("Costos del negocio")
    c_fn = st.number_input("Costo de NO detectar un cliente de riesgo (FN)", 1.0, 1000.0, 20.0, 1.0)
    c_fp = st.number_input("Costo de rechazar a un buen cliente (FP)", 0.1, 1000.0, 1.0, 0.1)
    st.caption("Ejemplo: perder un préstamo vale 20 veces más que rechazar a un buen cliente.")


@st.cache_data(show_spinner="Leyendo datos…")
def leer(f_bytes, nombre):
    import io
    if f_bytes is None:
        ruta = "datasetPrestamos.csv.gz" if os.path.exists("datasetPrestamos.csv.gz") else "datasetPrestamos.xlsx"
        return u.cargar(ruta)
    buf = io.BytesIO(f_bytes); buf.name = nombre
    return u.cargar(buf)


try:
    X, y, n_dup = leer(f.getvalue() if f else None, f.name if f else None)
except Exception as e:
    st.error(f"No pude leer los datos: {e}. Sube datasetPrestamos.xlsx en la barra lateral.")
    st.stop()

Xtr, Xte, ytr, yte = u.particionar(X, y, test_size, int(seed))

tab0, tab1, tab2, tab3 = st.tabs(["📊 Datos", "⚖️ a) Balanceo", "📉 b) Reducción de dimensiones", "🧾 Conclusión administrativa"])

# ---------------- Datos ----------------
with tab0:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Registros (sin duplicados)", f"{len(X):,}")
    c2.metric("Variables", X.shape[1])
    c3.metric("Duplicados eliminados", n_dup)
    c4.metric("% clase 1 (riesgo)", f"{y.mean():.2%}")
    cnt = y.value_counts().rename({0: "Clase 0 (buen cliente)", 1: "Clase 1 (riesgo)"}).reset_index()
    cnt.columns = ["clase", "n"]
    st.plotly_chart(px.bar(cnt, x="clase", y="n", text="n", title="Distribución de la variable objetivo"), width="stretch")
    st.info(f"Razón de desbalance: **{(y == 0).sum() / (y == 1).sum():.1f} : 1**. "
            "Un modelo que prediga siempre 'buen cliente' tendría accuracy de "
            f"**{(y == 0).mean():.1%}** pero detectaría **0 %** de los clientes de riesgo.")
    st.dataframe(X.head(), width="stretch")

# ---------------- a) Balanceo ----------------
with tab1:
    st.subheader("a) ¿Qué técnica de balanceo es más adecuada?")
    tecs = st.multiselect("Técnicas a comparar", u.TECNICAS, default=u.TECNICAS)
    if st.button("▶ Ejecutar comparación", type="primary") or "res_bal" in st.session_state:
        if st.session_state.get("firma_bal") != (modelo, tuple(tecs), n_arb, test_size, int(seed)):
            with st.spinner("Entrenando modelos… (1–2 min con Random Forest)"):
                res, probs = u.comparar_tecnicas(modelo, tecs, Xtr, ytr, Xte, yte, int(seed), n_arb)
            st.session_state.update(res_bal=res, probs_bal=probs, firma_bal=(modelo, tuple(tecs), n_arb, test_size, int(seed)))
        res, probs = st.session_state["res_bal"], st.session_state["probs_bal"]
        res = res.copy()
        res["Costo total (FN·c_fn + FP·c_fp)"] = u.costo_total(res, c_fn, c_fp)

        crit = st.radio("Criterio de selección", ["G-mean", "Balanced Acc", "Recall", "F1", "AUC", "Costo total (FN·c_fn + FP·c_fp)"], horizontal=True,
                        help="G-mean y Balanced Acc castigan por igual errores en ambas clases; Recall prioriza detectar riesgo; Costo usa tus costos de la barra lateral.")
        asc = crit.startswith("Costo")
        mejor = res.sort_values(crit, ascending=asc).iloc[0]
        cols = ["Técnica", "Accuracy", "Precision", "Recall", "F1", "AUC", "Balanced Acc", "G-mean", "VP", "FP", "FN", "Costo total (FN·c_fn + FP·c_fp)", "Tiempo (s)"]
        st.dataframe(res[cols].sort_values(crit, ascending=asc).style.format(
            {c: "{:.3f}" for c in ["Accuracy", "Precision", "Recall", "F1", "AUC", "Balanced Acc", "G-mean"]} | {"Costo total (FN·c_fn + FP·c_fp)": "{:,.0f}", "Tiempo (s)": "{:.1f}"}),
            width="stretch")
        st.success(f"**Técnica más adecuada según {crit}: {mejor['Técnica']}** — detecta {mejor['Recall']:.1%} de los clientes de riesgo "
                   f"(recall), con precisión {mejor['Precision']:.1%}, AUC {mejor['AUC']:.3f} y G-mean {mejor['G-mean']:.3f}.")
        m_long = res.melt(id_vars="Técnica", value_vars=["Recall", "Precision", "F1", "Balanced Acc", "G-mean", "AUC"], var_name="Métrica", value_name="valor")
        st.plotly_chart(px.bar(m_long, x="Técnica", y="valor", color="Métrica", barmode="group", title="Métricas por técnica (test)"), width="stretch")
        fig = go.Figure()
        for t, p in probs.items():
            fpr, tpr, _ = roc_curve(yte, p); fig.add_scatter(x=fpr, y=tpr, name=t, mode="lines")
        fig.add_scatter(x=[0, 1], y=[0, 1], name="azar", line=dict(dash="dash", color="gray"))
        fig.update_layout(title="Curvas ROC", xaxis_title="FPR", yaxis_title="TPR")
        st.plotly_chart(fig, width="stretch")
        st.session_state["mejor_tecnica"] = mejor["Técnica"]
        st.session_state["res_final"] = res
        st.markdown("""
**Cómo justificar:** *Accuracy engaña* con clases desbalanceadas. Se elige con métricas sensibles a la clase minoritaria
(**recall, F1, G-mean, balanced accuracy, AUC**) y con el **costo** de equivocarse. **Under-sampling** detecta más riesgo pero descarta
datos y genera muchas falsas alarmas; **SMOTE/ADASYN/SMOTE+Tomek** conservan toda la información y equilibran precisión y recall;
**class_weight** no altera los datos y suele maximizar F1. Se aplica **solo sobre train** para no contaminar el test.""")
    else:
        st.write("Pulsa **Ejecutar comparación**.")

# ---------------- b) PCA ----------------
with tab2:
    st.subheader("b) Reducción de dimensiones (PCA)")
    var_ind, var_cum = u.curva_pca(Xtr)
    st.plotly_chart(px.line(x=np.arange(1, len(var_cum) + 1), y=var_cum, markers=True, labels={"x": "Nº de componentes", "y": "Varianza acumulada"},
                            title="Varianza explicada acumulada").add_hline(y=0.90, line_dash="dash").add_hline(y=0.95, line_dash="dash"), width="stretch")
    tec_pca = st.selectbox("Técnica de balanceo a usar en la comparación", u.TECNICAS, index=u.TECNICAS.index(st.session_state.get("mejor_tecnica", "Under-sampling")))
    if st.button("▶ Comparar original vs PCA", type="primary") or "res_pca" in st.session_state:
        firma = (modelo, tec_pca, n_arb, test_size, int(seed))
        if st.session_state.get("firma_pca") != firma:
            with st.spinner("Entrenando…"):
                rp = u.comparar_pca(modelo, tec_pca, Xtr, ytr, Xte, yte, seed=int(seed), n_arboles=n_arb)
            st.session_state.update(res_pca=rp, firma_pca=firma)
        rp = st.session_state["res_pca"].copy()
        rp["Costo total"] = u.costo_total(rp, c_fn, c_fp)
        base = rp.iloc[0]
        rp["Δ variables"] = (rp["Nº variables"] / base["Nº variables"] - 1)
        rp["Δ AUC"] = rp["AUC"] - base["AUC"]; rp["Δ Recall"] = rp["Recall"] - base["Recall"]; rp["Δ G-mean"] = rp["G-mean"] - base["G-mean"]
        rp["Δ tiempo"] = rp["Tiempo (s)"] / base["Tiempo (s)"] - 1
        st.dataframe(rp[["Escenario", "Nº variables", "Varianza retenida", "Recall", "Precision", "F1", "AUC", "G-mean", "Costo total", "Tiempo (s)",
                         "Δ variables", "Δ AUC", "Δ Recall", "Δ G-mean", "Δ tiempo"]].style.format(
            {"Varianza retenida": "{:.1%}", "Recall": "{:.3f}", "Precision": "{:.3f}", "F1": "{:.3f}", "AUC": "{:.3f}", "G-mean": "{:.3f}", "Costo total": "{:,.0f}",
             "Tiempo (s)": "{:.2f}", "Δ variables": "{:+.0%}", "Δ AUC": "{:+.3f}", "Δ Recall": "{:+.3f}", "Δ G-mean": "{:+.3f}", "Δ tiempo": "{:+.0%}"}), width="stretch")
        pca90 = rp[rp["Escenario"] == "PCA 90 %"].iloc[0]
        st.info(f"Con **{int(pca90['Nº variables'])} componentes** (−{-pca90['Δ variables']:.0%} de variables) se retiene {pca90['Varianza retenida']:.1%} de la varianza, "
                f"pero el AUC cambia **{pca90['Δ AUC']:+.3f}**, el recall **{pca90['Δ Recall']:+.3f}** y el tiempo **{pca90['Δ tiempo']:+.0%}**.")
        st.plotly_chart(px.line(rp, x="Nº variables", y=["AUC", "Recall", "G-mean"], markers=True, title="Desempeño vs. nº de variables"), width="stretch")
    else:
        st.write("Pulsa **Comparar original vs PCA**.")

# ---------------- Conclusión ----------------
with tab3:
    st.subheader("Conclusión administrativa (se actualiza con tus resultados)")
    if "res_final" in st.session_state:
        r = st.session_state["res_final"]; mej = st.session_state["mejor_tecnica"]
        fila = r[r["Técnica"] == mej].iloc[0]; sin = r[r["Técnica"] == "Sin balanceo"]
        txt = (f"- **Problema:** solo {y.mean():.1%} de los registros pertenece a la clase de riesgo; sin balanceo el modelo parece excelente "
               f"(accuracy ≈ {(y == 0).mean():.0%}) pero casi no detecta riesgo.\n"
               f"- **Técnica recomendada:** **{mej}**, que detecta **{fila['Recall']:.1%}** de los clientes de riesgo "
               f"({int(fila['VP'])} de {int(fila['VP'] + fila['FN'])} en test)"
               + (f" frente a {sin.iloc[0]['Recall']:.1%} sin balanceo" if len(sin) else "") + ".\n"
               f"- **Costo del cambio:** {int(fila['FP'])} clientes buenos marcados como riesgo (falsas alarmas); con costos FN={c_fn:g} y FP={c_fp:g} el costo total de test es "
               f"{fila['Costo total (FN·c_fn + FP·c_fp)']:,.0f}.\n")
        st.markdown(txt)
    if "res_pca" in st.session_state:
        rp = st.session_state["res_pca"]; b = rp.iloc[0]; p9 = rp[rp["Escenario"] == "PCA 90 %"].iloc[0]
        st.markdown(f"- **Reducción de dimensiones:** PCA al 90 % reduce de {int(b['Nº variables'])} a {int(p9['Nº variables'])} variables "
                    f"({1 - p9['Nº variables'] / b['Nº variables']:.0%} menos), pero el AUC pasa de {b['AUC']:.3f} a {p9['AUC']:.3f} y el recall de {b['Recall']:.3f} a {p9['Recall']:.3f}. "
                    + ("**No se recomienda** usar PCA para el modelo final: el ahorro de cómputo no compensa la pérdida de capacidad predictiva."
                       if p9["AUC"] < b["AUC"] - 0.02 else "PCA conserva el desempeño y simplifica el modelo: **se recomienda** usarlo."))
    if "res_final" not in st.session_state and "res_pca" not in st.session_state:
        st.write("Ejecuta las pestañas a) y b) para generar la conclusión.")
