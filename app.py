import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.special as sc
from dataclasses import dataclass, asdict
from typing import List, Tuple
import math

# ---------- CONFIGURACIÓN GENERAL ----------
st.set_page_config(
    page_title="Operador de Acumulación Híbrida",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- ESTILOS CSS ----------
st.markdown("""
    <style>
    .main { background-color: #0E1117; }
    h1, h2, h3 { font-family: 'Helvetica Neue', sans-serif; font-weight: 600; }
    div[data-testid="metric-container"] {
        background-color: #1F2937;
        border: 1px solid #374151;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    div[data-testid="stMetricValue"] { color: #F3F4F6 !important; font-size: 1.6rem !important; }
    [data-testid="stSidebar"] { background-color: #111827; border-right: 1px solid #374151; }
    .math-box { background-color: #262730; padding: 15px; border-radius: 5px; border-left: 3px solid #60A5FA; }
    </style>
    """, unsafe_allow_html=True)

# ---------- MODELO ----------

@dataclass
class Shock:
    id: str
    nombre: str
    tiempo: float
    magnitud: float
    activo: bool = True
    descripcion: str = ""

class SolarisModel:
    def __init__(self, r: float, T: float, A: float, param_B_or_g: float, mode: str = 'lineal'):
        self.r = float(r)
        self.T = float(T)
        self.A = float(A)
        self.param_2 = float(param_B_or_g) # Puede ser B (pendiente) o g (growth)
        self.mode = mode
        self.shocks: List[Shock] = []

    def add_shock(self, shock: Shock):
        self.shocks.append(shock)

    def integral_kernel(self, a: float, b: float) -> float:
        # Esta es la integral de e^(-rt) desde el inicio del shock (a) hasta el final (b)
        if b <= a: return 0.0
        if abs(self.r) < 1e-9: return b - a
        return (math.exp(-self.r * a) - math.exp(-self.r * b)) / self.r

    def calcular_vpn_base(self) -> float:
        r = self.r
        T = self.T
        A = self.A
        p2 = self.param_2
        
        if self.mode == 'lineal':
            # f(t) = A + B*t
            if abs(r) < 1e-9:
                return A * T + 0.5 * p2 * T**2
            term_A = A * (1 - math.exp(-r * T)) / r
            term_B = p2 * ((1 - math.exp(-r * T)) / r**2 - T * math.exp(-r * T) / r)
            return term_A + term_B
            
        elif self.mode == 'exponencial':
            # f(t) = A * e^(g*t)
            g = p2
            net_rate = r - g
            if abs(net_rate) < 1e-9:
                return A * T
            return A * (1 - math.exp(-net_rate * T)) / net_rate
        
        return 0.0

    def calcular_impacto_shock(self, shock: Shock) -> float:
        if not shock.activo or shock.tiempo >= self.T:
            return 0.0
        # El valor es Magnitud * Factor de Descuento Integral
        factor_descuento = self.integral_kernel(shock.tiempo, self.T)
        return float(shock.magnitud * factor_descuento)

    def generar_trayectorias(self, pasos: int = 500):
        t = np.linspace(0, self.T, pasos)
        if self.mode == 'lineal':
            y_base = self.A + self.param_2 * t
        else:
            y_base = self.A * np.exp(self.param_2 * t)
            
        y_total = y_base.copy()
        for s in self.shocks:
            if s.activo and s.tiempo < self.T:
                y_total += np.where(t >= s.tiempo, s.magnitud, 0)
        return t, y_base, y_total
    
    # --- MÉTODO PARA CÁLCULO TRADICIONAL (AÑADIDO PARA LA COMPARATIVA FINAL) ---
    def calcular_tradicional_dcf(self) -> Tuple[float, pd.DataFrame]:
        """
        Calcula el VPN usando el método tradicional discreto:
        Sumatoria de flujos al final de cada año (t=1, 2, ..., T).
        """
        years = np.arange(1, int(self.T) + 1)
        dcf_val = 0.0
        details = []
        
        for yr in years:
            # 1. Flujo Base (Muestreo discreto en t=yr)
            if self.mode == 'lineal':
                cf_base = self.A + self.param_2 * yr
            else:
                cf_base = self.A * np.exp(self.param_2 * yr)
            
            # 2. Shocks (Se asignan al año si ocurren durante ese periodo)
            cf_shocks = 0.0
            for s in self.shocks:
                # Si el shock ocurre entre el año anterior y este (ej: t=1.5 entra en año 2)
                if (yr - 1) < s.tiempo <= yr and s.activo:
                    cf_shocks += s.magnitud 
            
            total_cf = cf_base + cf_shocks
            # Descuento discreto tradicional: 1 / (1+r)^t
            disc_factor = 1 / ((1 + self.r) ** yr)
            pv = total_cf * disc_factor
            
            dcf_val += pv
            details.append({
                "Año (t)": yr, 
                "Flujo Base": cf_base, 
                "Flujo Eventos": cf_shocks, 
                "Flujo Total": total_cf, 
                "Factor (1+r)^-t": disc_factor,
                "VP (Descontado)": pv
            })
            
        return dcf_val, pd.DataFrame(details)

# ---------- MAIN ----------

def main():
    col_title, col_logo = st.columns([4, 1])
    with col_title:
        st.title("💠 Operador de Acumulación Híbrida")
        st.markdown("Calculadora de Valoración Dinámica")
    
    # Sidebar
    st.sidebar.header("⚙️ Configuración")
    with st.sidebar.expander("1. Parámetros Globales", expanded=True):
        r_input = st.slider("Tasa de Descuento (r)", 0.0, 0.25, 0.08, 0.005, format="%.1f%%")
        T_input = st.number_input("Horizonte (T años)", 1.0, 50.0, 10.0, 0.5)
    
    with st.sidebar.expander("2. Función Base f(t)", expanded=True):
        tipo_funcion = st.selectbox("Modelo de Flujo", ["Lineal (A + Bt)", "Exponencial (A * e^gt)"])
        
        col_a, col_b = st.columns(2)
        A_input = col_a.number_input("Inicial (A)", value=500.0, step=50.0)
        
        if "Lineal" in tipo_funcion:
            mode_sel = 'lineal'
            B_input = col_b.number_input("Pendiente (B)", value=50.0, step=10.0, help="Cambio lineal por año")
        else:
            mode_sel = 'exponencial'
            B_input = col_b.number_input("Crecimiento (g)", value=0.05, step=0.01, format="%.2f", help="Tasa de crecimiento compuesto")

    modelo = SolarisModel(r_input, T_input, A_input, B_input, mode=mode_sel)

    # Shocks
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚡ Eventos Discretos")
    defaults = [
        {"id": "S1", "nombre": "Subsidio Verde", "tiempo": 2.0, "magnitud": 100.0},
        {"id": "S2", "nombre": "Fallo Inversores", "tiempo": 4.5, "magnitud": -80.0},
        {"id": "S3", "nombre": "Expansión Red", "tiempo": 7.0, "magnitud": 150.0},
    ]
    if 'custom_shocks' not in st.session_state:
        st.session_state.custom_shocks = defaults

    shocks_to_process = []
    
    for idx, shock in enumerate(st.session_state.custom_shocks):
        with st.sidebar.container():
            c1, c2 = st.columns([0.8, 0.2])
            is_active = c1.checkbox(f"{shock['nombre']} (t={shock['tiempo']})", value=True, key=f"chk_{idx}")
            if c2.button("✖", key=f"del_{idx}"):
                st.session_state.custom_shocks.pop(idx)
                st.rerun()
            if is_active:
                obj_shock = Shock(id=shock['id'], nombre=shock['nombre'], tiempo=shock['tiempo'], magnitud=shock['magnitud'])
                modelo.add_shock(obj_shock)
                shocks_to_process.append(obj_shock)

    with st.sidebar.expander("➕ Agregar Evento", expanded=False):
        with st.form("add_shock"):
            n_name = st.text_input("Nombre", "Nuevo Evento")
            c_t, c_m = st.columns(2)
            n_time = c_t.number_input("Año", 0.0, T_input, 1.0)
            n_mag = c_m.number_input("Impacto ($)", value=50.0)
            if st.form_submit_button("Añadir"):
                st.session_state.custom_shocks.append({"id": f"C{len(st.session_state.custom_shocks)}", "nombre": n_name, "tiempo": n_time, "magnitud": n_mag})
                st.rerun()

    # Cálculos Principales
    vpn_base = modelo.calcular_vpn_base()
    impactos = []
    explicacion_shocks = [] 

    for s in shocks_to_process:
        val = modelo.calcular_impacto_shock(s)
        # Factor de descuento integral
        factor = modelo.integral_kernel(s.tiempo, modelo.T)
        
        impactos.append({"nombre": s.nombre, "valor": val, "tiempo": s.tiempo, "magnitud": s.magnitud})
        explicacion_shocks.append({
            "Evento": s.nombre,
            "Inicio (t)": s.tiempo,
            "Horizonte (T)": T_input,
            "Magnitud (Δ)": s.magnitud,
            "Factor Descuento ∫": factor,
            "Valor Presente": val
        })
    
    vpn_shocks = sum(x['valor'] for x in impactos)
    vpn_total = vpn_base + vpn_shocks
    variacion_pct = (vpn_shocks / vpn_base) * 100 if vpn_base != 0 else 0

    # --- DASHBOARD ---
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("VPN Total", f"${vpn_total:,.0f}", delta=f"{vpn_shocks:+,.0f} eventos")
    k2.metric("Variación vs Base", f"{variacion_pct:+.1f}%", delta="Impacto Relativo", delta_color="off")
    k3.metric("Valor Base", f"${vpn_base:,.0f}", delta="Estructural", delta_color="off")
    k4.metric("Tasa Efectiva", f"{r_input*100:.1f}%", f"T={T_input}")

    st.markdown("---")

    tab_chart, tab_waterfall, tab_data = st.tabs(["📈 Trayectoria", "🧱 Descomposición", "📋 Datos"])

    with tab_chart:
        t, y_base, y_total = modelo.generar_trayectorias()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=y_base, mode='lines', name='Base', line=dict(color='#4B5563', width=1, dash='dash')))
        fig.add_trace(go.Scatter(x=t, y=y_total, mode='lines', name='Híbrido', line=dict(color='#60A5FA', width=3), fill='tonexty', fillcolor='rgba(96, 165, 250, 0.1)'))
        for s in shocks_to_process:
            color = '#34D399' if s.magnitud > 0 else '#F87171'
            fig.add_vline(x=s.tiempo, line_dash="dot", line_color=color)
            fig.add_annotation(x=s.tiempo, y=max(y_total)*1.05, text=s.nombre, showarrow=False, font=dict(color=color, size=10))
        fig.update_layout(title="Dinámica del Flujo", template="plotly_dark", height=450, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

    with tab_waterfall:
        wf_labels = ["Base"] + [x['nombre'] for x in impactos] + ["Total"]
        wf_y = [vpn_base] + [x['valor'] for x in impactos] + [0]
        measure_types = ["absolute"] + ["relative"] * len(impactos) + ["total"]
        fig_wf = go.Figure(go.Waterfall(
            name="20", orientation="v", measure=measure_types, x=wf_labels, y=wf_y,
            text=[f"${v/1000:.1f}k" for v in wf_y[:-1]] + [f"${vpn_total/1000:.1f}k"],
            connector={"line": {"color": "rgb(63, 63, 63)"}}, decreasing={"marker": {"color": "#F87171"}}, increasing={"marker": {"color": "#34D399"}}, totals={"marker": {"color": "#60A5FA"}}
        ))
        fig_wf.update_layout(title="Descomposición del Valor", template="plotly_dark", height=500)
        st.plotly_chart(fig_wf, use_container_width=True)

    with tab_data:
        st.dataframe(pd.DataFrame(impactos))

    # ==============================================================================
    # SECCIÓN: FÓRMULA DE ACUMULACIÓN (Híbrida)
    # ==============================================================================
    
    st.markdown("---")
    st.subheader("🧮 Estructura del Operador Híbrido")
    st.markdown("El valor total ($\mathcal{H}_K$) se compone de la **integral del flujo base continuo** más la **acumulación discreta** de los eventos.")

    if mode_sel == 'lineal':
        ft_latex = rf"({A_input:,.0f} + {B_input:,.0f}t)"
    else:
        ft_latex = rf"({A_input:,.0f} \cdot e^{{{B_input}t}})"

    # 1. Formulación
    st.markdown("#### 1. Formulación Matemática")
    st.latex(rf"""
    \mathcal{{H}}_K = \underbrace{{ \int_{{0}}^{{T}} f(t) \cdot e^{{-rt}} dt }}_{{\text{{Base Continua}}}} + \sum_{{i=1}}^{{n}} \text{{Impacto}}(E_i)
    """)

    # 2. Desglose
    st.markdown("#### 2. Desglose de Componentes")
    integral_visual = rf"\left[ \int_{{0}}^{{{T_input}}} {ft_latex} e^{{-{r_input}t}} dt \right]"
    latex_formula_names = rf"\mathcal{{H}}_K = {integral_visual}"
    for s in impactos:
        s_clean = s['nombre'].replace(" ", "\\;")
        latex_formula_names += rf" + (\text{{{s_clean}}})"
    st.latex(latex_formula_names)

    # 3. Instanciación
    st.markdown("#### 3. Instanciación Numérica")
    str_vals = f"{vpn_base:,.0f}"
    for s in impactos:
        val = s['valor']
        str_vals += rf" + ({val:,.0f})"
    st.latex(rf"\mathcal{{H}}_K = {str_vals} = \mathbf{{ {vpn_total:,.0f} }}")

    # ==============================================================================
    # SECCIÓN: DESGLOSE ANALÍTICO (Expander)
    # ==============================================================================
    
    with st.expander("🔎 Desglose: ¿Cómo se calcula el valor de cada Choque?", expanded=False):
        st.markdown(r"""
        El valor presente de un choque de magnitud $\Delta$ en $t$ se calcula con:
        $$ \text{Valor}(E_i) = \Delta \times \int_{t}^{T} e^{-r \tau} d\tau = \Delta \times \left[ \frac{e^{-r t} - e^{-r T}}{r} \right] $$
        """)
        
        if explicacion_shocks:
            st.dataframe(pd.DataFrame(explicacion_shocks).style.format({
                    "Inicio (t)": "{:.1f}",
                    "Horizonte (T)": "{:.1f}",
                    "Magnitud (Δ)": "${:,.0f}",
                    "Factor Descuento ∫": "{:.4f}",
                    "Valor Presente": "${:,.2f}"
                }), use_container_width=True)
        else:
            st.info("No hay eventos activos.")

    # ==============================================================================
    # NUEVA SECCIÓN: COMPARATIVA CON MÉTODO TRADICIONAL
    # ==============================================================================

    st.markdown("---")
    st.subheader("🆚 Benchmarking: Híbrido ($\mathcal{H}_K$) vs. Tradicional (DCF)")
    st.markdown("""
    Comparativa en tiempo real contra el método tradicional de Flujos de Caja Descontados (DCF), 
    que asume flujos discretos al final de cada año ($t=1, 2, ..., T$).
    """)

    # Calcular tradicional
    vpn_trad, df_trad = modelo.calcular_tradicional_dcf()
    
    # Calcular diferencia
    diff_val = vpn_total - vpn_trad
    diff_pct = (diff_val / vpn_trad * 100) if vpn_trad != 0 else 0

    c_bench_1, c_bench_2 = st.columns([1, 1])

    with c_bench_1:
        st.markdown("**Resultados Comparativos**")
        st.metric("Método Híbrido (Continuo)", f"${vpn_total:,.0f}")
        st.metric("Método Tradicional (Discreto)", f"${vpn_trad:,.0f}")
        
    with c_bench_2:
        st.markdown("**Diferencia (Precisión Ganada)**")
        st.metric("Delta Valor", f"${diff_val:,.0f}", delta=f"{diff_pct:+.2f}% vs Tradicional")
        st.info("La diferencia surge porque el Método Híbrido captura el valor del dinero exacto en el tiempo continuo, mientras que el tradicional 'redondea' al final del año.")

    # Visualización de la fórmula tradicional para contraste
    st.markdown("#### Procedimiento Tradicional")
    st.latex(r"\text{VPN}_{trad} = \sum_{t=1}^{T} \frac{\text{Flujo}_t}{(1+r)^t}")
    
    with st.expander("Ver tabla de cálculo Tradicional (Año a Año)", expanded=False):
        st.dataframe(df_trad.style.format({
            "Flujo Base": "${:,.2f}",
            "Flujo Eventos": "${:,.2f}",
            "Flujo Total": "${:,.2f}",
            "Factor (1+r)^-t": "{:.4f}",
            "VP (Descontado)": "${:,.2f}"
        }), use_container_width=True)

    # ==============================================================================
    # SECCIÓN: NOTA TÉCNICA Y ALCANCE (NUEVO AGREGADO)
    # ==============================================================================
    st.markdown("---")
    with st.expander("ℹ️ Nota Técnica: Alcance y Aplicabilidad del Modelo", expanded=False):
        st.markdown("""
        ### 🎯 ¿Cuándo es ideal usar el Operador Híbrido ($\mathcal{H}_K$)?
        Este modelo utiliza matemática de **tiempo continuo**, asumiendo que el valor se genera constantemente (como el agua fluyendo) y no en bloques anuales.

        * **✅ Casos de Uso Óptimos (Economía Moderna):**
            * **SaaS & Suscripciones:** Ingresos recurrentes diarios/mensuales (ej. Netflix, Spotify).
            * **Utilities & Energía:** Generación eléctrica constante, peajes, telecomunicaciones.
            * **Finanzas:** Valoración de derivados o activos de alta liquidez.

        * **⚠️ Cuándo preferir el Método Tradicional:**
            * **Flujos "Lumpy" (Agrupados):** Agricultura (una cosecha al año), Construcción (pagos contra entrega de hitos), Rentas inmobiliarias anuales.
            * *Razón:* En estos casos, usar tiempo continuo podría "adelantar" valor teóricamente que en la práctica está bloqueado hasta fin de año.

        **Conclusión:** La diferencia de valor mostrada arriba (**Delta Valor**) representa la **Captura de Valor por Continuidad**. El $\mathcal{H}_K$ elimina la pérdida de eficiencia que asume el método tradicional al esperar al final del periodo para contabilizar los flujos.
        """)

def asdict(shock: Shock):
    return {"id": shock.id, "nombre": shock.nombre, "tiempo": shock.tiempo, "magnitud": shock.magnitud, "activo": shock.activo, "descripcion": shock.descripcion}

if __name__ == "__main__":
    main()
