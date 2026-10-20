import streamlit as st
import pandas as pd 
import plotly.express as px
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

st.set_page_config(
    page_title="Ventas Valentina Café",
    page_icon="☕️",
    layout="wide"
)

st.title("☕️ Ventas Valentina Café")

st.write(
    """
    Esta aplicación analizará información de la venta de productos de Valentina Café,
    para apoyar la toma de decisiones de decisiones sobre que productos vender y cuales retirar.
    """
)

st.markdown(
    "**Pregunta guía:** ¿Qué productos tienen mayor probabilidad de tener un mes de venta alta "
    "y qué acción comercial deberíamos tomar con cada tipo de producto?"
)

# ---------- Cargar datos ----------
df = pd.read_csv("data/ventas_cafeteria.csv", parse_dates=["fecha"])

# ---------- Pestañas ----------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["1. Entender", "2. Limpiar", "3. Estadística descriptiva", "4. Estadística predictiva", "5. Estadística prescriptiva", "6. Toma de decisiones"]
)

# ================= PESTAÑA 1: ENTENDER =================
with tab1:
    st.header("¿Qué datos tenemos?")

    # Números generales
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Filas", f"{df.shape[0]:,}")
    col2.metric("Variables", df.shape[1])
    col3.metric("Productos distintos", df["producto"].nunique())
    col4.metric("Periodo", f"{df['anio'].min()} – {df['anio'].max()}")

    st.markdown(
        "**Cada fila es un producto en un mes.** "
        "Por ejemplo: «Latte, Grande» en marzo de 2023."
    )

    # Variables clave
    st.subheader("Variables clave")
    variables = pd.DataFrame(
        [
            ["venta_alta", "Resultado (Sí/No)", "Sí, si el producto vendió 50 unidades o más en el mes. Es lo que queremos predecir."],
            ["unidades", "Numérica", "Unidades vendidas en el mes."],
            ["precio_unitario", "Numérica", "Ingreso con IVA ÷ unidades."],
            ["meses_en_catalogo", "Numérica", "Cuántos meses lleva vendiéndose el producto."],
            ["categoria", "Categórica", "Tipo de producto: café, repostería, frappé, etc."],
            ["tamano", "Categórica", "Chico, Regular, Grande o No aplica."],
            ["temporada", "Categórica", "Calor (abr-sep) o Fresco (oct-mar)."],
            ["leche_vegetal / jarabe", "Categórica (Sí/No)", "Si la bebida lleva ese extra."],
        ],
        columns=["Variable", "Tipo", "Qué significa"],
    )
    st.dataframe(variables, hide_index=True)

    # ¿Qué tan seguido hay venta alta?
    st.subheader("¿Qué tan común es una venta alta?")
    st.dataframe(df["venta_alta"].value_counts().rename("filas"))

    # Dudas sobre la calidad
    st.subheader("Dudas sobre la calidad de los datos")
    negativos = (df["unidades"] <= 0).sum()
    ingreso_cero = (df["ingreso_con_iva"] == 0).sum()
    ajustes = (df["categoria"] == "Ajuste de caja").sum()

    st.markdown(
        f"""
1. **¿Todas las filas son ventas reales?** Hay **{ajustes}** filas de *ajustes de caja*
   (sobrantes, faltantes) que no son productos.
2. **¿Por qué hay ventas de \\$0?** Hay **{ingreso_cero}** filas con ingreso \\$0, sobre todo
   movimientos de *Almacén* (consumo interno).
3. **¿Unidades negativas o en cero?** Hay **{negativos}** filas así. Se consideran errores de captura de la información.
4. **Duda de significado:** si un producto no se vendió en un mes, **no aparece** en el reporte
   (no hay fila con 0). Por eso no sé si dejó de venderse o simplemente no estaba en el menú.
5. **El umbral de 50 unidades:** esta regla fue establecida por el programador de esta app (Jose Guerra), puede variar en caso de que se elija otro umbral. Esto puede afectar en qué cuenta como «venta alta».
        """
    )

    # Vista previa
    st.subheader("Vista previa de los datos")
    st.dataframe(df.head(10), hide_index=True)

# ================= LAS DEMÁS PESTAÑAS (las llenamos después) =================
# ================= PESTAÑA 2: LIMPIAR =================
with tab2:
    st.header("¿Qué tan confiables son los datos?")

    # ---- 1. Contar nulos ----
    st.subheader("1. Valores nulos (vacíos)")
    nulos = df.isna().sum()
    nulos = nulos[nulos > 0].rename("nulos").to_frame()
    if nulos.empty:
        st.success("No hay valores nulos.")
    else:
        st.dataframe(nulos)
        st.markdown(
            "**Decisión:** los nulos están en `precio_unitario` porque esas filas tienen 0 unidades "
            "(no se puede dividir entre 0). Son ajustes de caja, así que **se eliminan** junto con ellos."
        )

    # ---- 2. Contar duplicados ----
    st.subheader("2. Duplicados")
    dup_exactos = df.duplicated().sum()
    dup_producto_mes = df.duplicated(subset=["fecha", "producto"]).sum()
    c1, c2 = st.columns(2)
    c1.metric("Filas idénticas", dup_exactos)
    c2.metric("Mismo producto repetido en un mes", dup_producto_mes)
    st.markdown(
        "**Decisión:** los repetidos son *Sobrante* e *Importe* (ajustes de caja). "
        "Como **se eliminan** con los ajustes, no hace falta sumarlos ni corregirlos."
    )

    # ---- 3. Limpieza paso a paso ----
    st.subheader("3. Pasos de limpieza")
    pasos = []
    df_limpio = df.copy()

    def quitar(condicion, paso, decision):
        global df_limpio
        antes = len(df_limpio)
        df_limpio = df_limpio[~condicion(df_limpio)]
        pasos.append({"Paso": paso, "Filas eliminadas": antes - len(df_limpio),
                      "Filas que quedan": len(df_limpio), "Decisión": decision})

    quitar(lambda d: d["categoria"] == "Ajuste de caja", "Ajustes de caja",
           "Eliminar: sobrantes y faltantes no son productos.")
    quitar(lambda d: d["categoria"] == "Insumo interno", "Insumos de almacén",
           "Eliminar: son consumo interno con ingreso $0.")
    quitar(lambda d: d["unidades"] <= 0, "Unidades ≤ 0",
           "Eliminar: son correcciones de captura.")
    quitar(lambda d: d["ingreso_con_iva"] <= 0, "Ingreso $0",
           "Eliminar: cortesías, no reflejan demanda.")
    quitar(lambda d: d.duplicated(subset=["fecha", "producto"]), "Duplicados",
           "Eliminar si quedara alguno.")

    st.dataframe(pd.DataFrame(pasos), hide_index=True)

    # ---- 4. Resultado ----
    st.subheader("4. Resultado")
    c1, c2, c3 = st.columns(3)
    c1.metric("Filas antes", f"{len(df):,}")
    c2.metric("Filas después", f"{len(df_limpio):,}")
    c3.metric("Nulos después", int(df_limpio.isna().sum().sum()))
    st.success(f"Conservamos el {len(df_limpio) / len(df):.0%} de las filas: los datos son confiables "
               "una vez que quitamos lo que no son ventas reales.")
# ================= PESTAÑA 3: DESCRIBIR =================
with tab3:
    st.header("¿Qué está pasando en las ventas?")

    # ---- KPIs ----
    st.subheader("Indicadores clave (KPIs)")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Ingreso total (con IVA)", f"${df_limpio['ingreso_con_iva'].sum():,.0f}")
    k2.metric("Unidades vendidas", f"{df_limpio['unidades'].sum():,}")
    k3.metric("Meses con venta alta", f"{(df_limpio['venta_alta'] == 'Sí').mean():.0%}")
    k4.metric("Precio mediano", f"${df_limpio['precio_unitario'].median():,.0f}")

    # ---- Gráfica por segmento ----
    st.subheader("Comparar por segmento")
    opciones = {
        "Categoría": "categoria",
        "Tamaño": "tamano",
        "Temporada": "temporada",
        "Leche vegetal": "leche_vegetal",
        "Jarabe": "jarabe",
    }
    etiqueta = st.selectbox(
        "Elige cómo agrupar los productos:",
        list(opciones),
        index=None,
        placeholder="Selecciona una opción...",
    )

    if etiqueta is None:
        st.info("👆 Elige una opción para ver las gráficas.")
    else:
        segmento = opciones[etiqueta]

        resumen = (
            df_limpio.groupby(segmento)
            .agg(
                ingreso=("ingreso_con_iva", "sum"),
                pct_venta_alta=("venta_alta", lambda s: (s == "Sí").mean()),
                productos=("producto", "nunique"),
            )
            .reset_index()
            .sort_values("pct_venta_alta")
        )

        col_a, col_b = st.columns(2)

        fig1 = px.bar(resumen, x="pct_venta_alta", y=segmento, orientation="h",
                      text_auto=".0%", title=f"% de meses con venta alta por {etiqueta}",
                      color_discrete_sequence=["#2a78d6"])
        fig1.update_layout(xaxis_tickformat=".0%", xaxis_title=None, yaxis_title=None)
        col_a.plotly_chart(fig1)

        fig2 = px.bar(resumen.sort_values("ingreso"), x="ingreso", y=segmento, orientation="h",
                      title=f"Ingreso total por {etiqueta}",
                      color_discrete_sequence=["#2a78d6"])
        fig2.update_layout(xaxis_title="Pesos con IVA", yaxis_title=None)
        col_b.plotly_chart(fig2)

        st.dataframe(resumen, hide_index=True)

        # ---- Tendencia en el tiempo, filtrada por el segmento ----
        st.subheader(f"Unidades vendidas por mes según {etiqueta}")
        valores = ["Todas"] + sorted(df_limpio[segmento].unique())
        valor = st.selectbox(f"¿Qué {etiqueta.lower()} quieres ver?", valores)

        if valor == "Todas":
            datos_linea = df_limpio
        else:
            datos_linea = df_limpio[df_limpio[segmento] == valor]

        mensual = datos_linea.groupby("fecha", as_index=False)["unidades"].sum()
        fig3 = px.line(mensual, x="fecha", y="unidades", markers=True,
                       title=f"Unidades por mes: {valor}",
                       color_discrete_sequence=["#2a78d6"])
        fig3.update_layout(xaxis_title=None, yaxis_title="Unidades")
        st.plotly_chart(fig3)
# ================= PESTAÑA 4: PREDECIR =================
with tab4:
    st.header("¿Qué tienen en común los meses de venta alta y podemos predecirlos?")

    # ---- 1. Características: venta alta vs. no ----
    st.subheader("1. ¿En qué se parecen los meses de venta alta?")
    comparacion = df_limpio.groupby("venta_alta").agg(
        filas=("producto", "size"),
        precio_promedio=("precio_unitario", "mean"),
        meses_en_catalogo=("meses_en_catalogo", "mean"),
        pct_reposteria=("categoria", lambda s: (s == "Repostería").mean()),
        pct_tamano_grande=("tamano", lambda s: (s == "Grande").mean()),
    ).round(2)
    st.dataframe(comparacion)
    st.caption("Compara el promedio de cada característica entre los meses con venta alta (Sí) y sin ella (No).")

    # ---- 2. Entrenar el modelo ----
    st.subheader("2. Modelo: regresión logística")
    categoricas = ["categoria", "tamano", "temporada", "leche_vegetal", "jarabe"]
    numericas = ["precio_unitario", "meses_en_catalogo"]
    columnas_x = categoricas + numericas

    entrenamiento = df_limpio[df_limpio["anio"] <= 2023]
    prueba = df_limpio[df_limpio["anio"] == 2024].copy()

    preparador = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore",
                              feature_name_combiner=lambda col, valor: f"{col} = {valor}"), categoricas),
        ("num", Pipeline([("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                          ("escala", StandardScaler())]), numericas),
    ])
    modelo = Pipeline([
        ("preparar", preparador),
        ("logistica", LogisticRegression(max_iter=1000)),
    ])
    modelo.fit(entrenamiento[columnas_x], entrenamiento["venta_alta"] == "Sí")

    prueba["probabilidad"] = modelo.predict_proba(prueba[columnas_x])[:, 1]

    st.markdown(
        "- **Entrena** con 2022 y 2023, y se **prueba** con 2024 (un año que el modelo no vio).\n"
        "- **No usa** unidades ni ingreso: con ellos se calcula el resultado, sería hacer trampa."
    )

    # ---- 3. ¿Qué tan bueno es? ----
    real = prueba["venta_alta"] == "Sí"
    predicho = prueba["probabilidad"] >= 0.5
    m1, m2, m3 = st.columns(3)
    m1.metric("Exactitud del modelo", f"{accuracy_score(real, predicho):.0%}")
    m2.metric("Línea base (decir siempre «No»)", f"{1 - real.mean():.0%}")
    m3.metric("AUC", f"{roc_auc_score(real, prueba['probabilidad']):.2f}",
              help="0.5 = adivinar al azar · 1.0 = perfecto")

    # ---- 4. Variables que más pesan ----
    st.subheader("3. Variables que más pesan")
    nombres = modelo.named_steps["preparar"].get_feature_names_out()
    pesos = pd.DataFrame({
        "variable": [n.replace("cat__", "").replace("num__", "") for n in nombres],
        "peso": modelo.named_steps["logistica"].coef_[0],
    })
    pesos["efecto"] = np.where(pesos["peso"] > 0, "Sube la probabilidad", "Baja la probabilidad")
    top = pesos.reindex(pesos["peso"].abs().sort_values(ascending=False).index).head(10).sort_values("peso")

    fig4 = px.bar(top, x="peso", y="variable", orientation="h", color="efecto",
                  color_discrete_map={"Sube la probabilidad": "#2a78d6", "Baja la probabilidad": "#eb6834"},
                  title="Las 10 variables con más peso")
    fig4.update_layout(xaxis_title="Peso en el modelo", yaxis_title=None, legend_title=None)
    st.plotly_chart(fig4)

# ================= PESTAÑA 5: PRESCRIBIR =================
with tab5:
    st.header("¿Qué acción deberíamos tomar con cada producto?")

    # ---- 1. Probabilidad promedio por producto (2024) ----
    productos = prueba.groupby(["producto", "categoria"], as_index=False).agg(
        probabilidad=("probabilidad", "mean"),
        precio=("precio_unitario", "mean"),
        unidades_2024=("unidades", "sum"),
        ingreso_2024=("ingreso_con_iva", "sum"),
    )

    # ---- 2. Nivel de probabilidad ----
    def nivel(p):
        if p >= 0.5:
            return "Alta"
        elif p >= 0.25:
            return "Media"
        else:
            return "Baja"

    productos["nivel"] = productos["probabilidad"].apply(nivel)

    acciones = {
        "Alta": "Producto estrella: asegurar que siempre haya en inventario, destacarlo en el menú y ofrecer el tamaño grande (en caso de que aplique).",
        "Media": "Producto con potencial: combinarlo con repostería o darlo a probar con una promoción de un día.",
        "Baja": "Producto a revisar: rotarlo como producto de temporada o retirarlo del menú por completo para evitar merma y perdidas.",
    }
    productos["accion"] = productos["nivel"].map(acciones)

    # ---- 3. Tabla de acción por nivel ----
    st.subheader("Acción recomendada por nivel de probabilidad")
    st.markdown("**Alta:** 50 % o más · **Media:** de 25 % a 50 % · **Baja:** menos de 25 %")
    resumen_niveles = (
        productos.groupby("nivel")
        .agg(productos=("producto", "size"), ingreso_2024=("ingreso_2024", "sum"))
        .reindex(["Alta", "Media", "Baja"])
        .reset_index()
    )
    resumen_niveles["% del ingreso"] = (resumen_niveles["ingreso_2024"] / resumen_niveles["ingreso_2024"].sum()).map("{:.0%}".format)
    resumen_niveles["ingreso_2024"] = resumen_niveles["ingreso_2024"].map("${:,.0f}".format)
    resumen_niveles["acción"] = resumen_niveles["nivel"].map(acciones)
    st.dataframe(resumen_niveles, hide_index=True)

        # ---- 4. Detalle por producto, filtrado por categoría ----
    st.subheader("Detalle por producto")
    categorias = ["Todas"] + sorted(productos["categoria"].unique())
    categoria_elegida = st.selectbox(
        "Elige una categoría:",
        categorias,
        index=None,
        placeholder="Selecciona una categoría...",
    )

    if categoria_elegida is None:
        st.info("👆 Elige una categoría para ver sus productos.")
    else:
        if categoria_elegida == "Todas":
            tabla = productos
        else:
            tabla = productos[productos["categoria"] == categoria_elegida]

        st.caption(f"{len(tabla)} productos")
        st.dataframe(
            tabla.sort_values("probabilidad", ascending=False)[
                ["producto", "categoria", "nivel", "probabilidad", "precio", "ingreso_2024"]
            ],
            hide_index=True,
            column_config={
                "probabilidad": st.column_config.ProgressColumn("Probabilidad", format="percent", min_value=0, max_value=1),
                "precio": st.column_config.NumberColumn("Precio", format="$%.0f"),
                                "ingreso_2024": st.column_config.NumberColumn("Ingreso 2024", format="dollar"),
            },
        )

# ================= PESTAÑA 6: DECIDIR =================
with tab6:
    st.header("¿Qué mensaje le mostramos a cada tipo de producto?")

    mensajes = {
        "Alta": "«¿Ya es tu favorito? Pídelo en tamaño grande y llévate un macaroon de regalo (válido hasta agotar existencias).» → Para clientes frecuentes, en el menú y en redes.",
        "Media": "«Esta semana: descuento del 5 % con tu bebida.» → Para clientes que ya nos visitan, en mostrador y en historias de Instagram (cuando enseñen la historia en mostrador).",
        "Baja": "«Últimos días: edición de temporada.» → Mensaje interno para el equipo: vender lo que queda y decidir si sale del menú por completo para evitar merma y perdidas.",
    }

    elegido = st.selectbox("Elige un nivel de probabilidad:", ["Alta", "Media", "Baja"])
    grupo = productos[productos["nivel"] == elegido]
    ejemplos = grupo.nlargest(3, "ingreso_2024")["producto"].tolist()
    pct_ingreso = grupo["ingreso_2024"].sum() / productos["ingreso_2024"].sum()

    st.subheader("🎯 Acción")
    st.info(acciones[elegido])

    st.subheader("💬 Mensaje")
    st.success(mensajes[elegido])

    st.subheader("📊 Justificación")
    j1, j2, j3 = st.columns(3)
    j1.metric("Productos en este nivel", len(grupo))
    j2.metric("Probabilidad promedio", f"{grupo['probabilidad'].mean():.0%}")
    j3.metric("% del ingreso 2024", f"{pct_ingreso:.0%}")
    st.markdown(f"**Ejemplos con más ingreso:** {', '.join('«' + e + '»' for e in ejemplos)}")

    with st.expander("⚠️ Alcance del modelo"):
        st.markdown(
            "- Predice **meses de venta alta por producto**, no el comportamiento de clientes.\n"
            f"- Acierta el {accuracy_score(real, predicho):.0%}, solo un poco más que decir siempre «No» ({1 - real.mean():.0%}): "
            "sirve para **ordenar** productos, no para dar cifras exactas.\n"
            "- No conoce promociones, clima ni cambios de menú, y los meses sin ventas no aparecen en los datos."
        )