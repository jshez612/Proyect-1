import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dataclasses import dataclass, asdict
from typing import List, Tuple  # <--- AQUÍ ESTÁ LA CORRECCIÓN IMPRESCINDIBLE
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
        self.param_2 = float(param_B_or_g) 
        self.mode = mode
        self.shocks: List[Shock] = []

    def add_shock(self, shock: Shock):
        self.shocks.append(shock)

    def integral_kernel(self, a: float, b: float) -> float:
        if b <= a: return 0.0
        if abs(self.r) < 1e-9: return b - a
        return (math.exp(-self.r * a) - math.exp(-self.r * b)) / self.r

    def calcular_vpn_base(self) -> float:
        r = self.r
        T = self.T
        A = self.A
        p2 = self.param_2
        
        if self.mode == 'lineal':
            if abs(r) < 1e-9:
                return A * T + 0.5 * p2 * T**2
            term_A = A * (1 - math.exp(-r * T)) / r
            term_B = p2 * ((1 - math.exp(-r * T)) / r**2 - T * math.exp(-r * T) / r)
            return term_A + term_B
            
        elif self.mode == 'exponencial':
            g = p2
            net_rate = r - g
            if abs(net_rate) < 1e-9:
                return A * T
            return A * (1 - math.exp(-net_rate * T)) / net_rate
        return 0.0

    def calcular_impacto_shock(self, shock: Shock) -> float:
        if not shock.activo or shock.tiempo >= self.T:
            return 0.0
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
    
    # --- FUNCIÓN PARA CALCULAR EL MÉTODO TRADICIONAL (DCF) ---
    def calcular_tradicional_dcf(self) -> Tuple[float, pd.DataFrame]:
        """
        Simula el cálculo tradicional: flujos discretos al final de cada año.
        Ignora la continuidad y asume que los shocks ocurren en el año entero.
        """
        years = np.arange(1, int(self.T) + 1)
        dcf_val = 0.0
        details = []
        
        for yr in years:
            # 1. Flujo Base al final del año t
            if self.mode == 'lineal':
                cf_base = self.A + self.param_2 * yr
            else:
                cf_base = self.A * np.exp(self.param_2 * yr)
            
            # 2. Shocks (Simplificación tradicional: Sumar al flujo del año si ocurre en ese año)
            cf_shocks = 0.0
            for s in self.shocks:
                # Si el shock ocurre durante este año (ej: entre año 1 y 2, se cobra en t=2)
                if (yr - 1) < s.tiempo <= yr and s.activo:
                    cf_shocks += s.magnitud 
            
            total_cf = cf_base + cf_shocks
            disc_factor = 1 / ((1 + self.r) ** yr)
            pv = total_cf * disc_factor
            
            dcf_val += pv
            details.append({
                "Año": yr, 
                "CF Base": cf_base, 
                "CF Shocks": cf_shocks, 
                "Total CF": total_cf, 
                "PV (Descontado)": pv
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
            B_input = col_b.number_input("Pendiente (B)", value=50.0, step=10.0)
        else:
            mode_sel = 'exponencial'
            B_input = col_b.number_input("Crecimiento (g)", value=0.05, step=0.01, format="%.2f")

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

    # Cálculos Híbridos
    vpn_base = modelo.calcular_vpn_base()
    impactos = []
    for s in shocks_to_process:
        val = modelo.calcular_impacto_shock(s)
        impactos.append({"nombre": s.nombre, "valor": val, "tiempo": s.tiempo, "magnitud": s.magnitud})
    
    vpn_shocks = sum(x['valor'] for x in impactos)
    vpn_total = vpn_base + vpn_shocks
    variacion_pct = (vpn_shocks / vpn_base) * 100 if vpn_base != 0 else 0

    # Cálculo Tradicional (Para comparación)
    vpn_tradicional, df_tradicional = modelo.calcular_tradicional_dcf()

    # --- DASHBOARD ---
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("VPN Híbrido (Propio)", f"${vpn_total:,.0f}", delta=f"{vpn_shocks:+,.0f} eventos")
    k2.metric("Variación vs Base", f"{variacion_pct:+.1f}%", delta="Impacto", delta_color="off")
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
    # SECCIÓN 1: FÓRMULA DE ACUMULACIÓN (MÉTODO PROPIO)
    # ==============================================================================
    
    st.markdown("---")
    st.subheader("🧮 Método Propio: Operador de Acumulación Híbrida")
    st.markdown("Cálculo exacto utilizando integración continua y acumulación de eventos en tiempo real.")

    if mode_sel == 'lineal':
        ft_latex = rf"({A_input:,.0f} + {B_input:,.0f}t)"
    else:
        ft_latex = rf"({A_input:,.0f} \cdot e^{{{B_input}t}})"

    # Formulación
    st.markdown("#### 1. Formulación Matemática (Continua)")
    st.latex(rf"""
    \mathcal{{H}}_K = \underbrace{{ \int_{{0}}^{{T}} f(t) \cdot e^{{-rt}} dt }}_{{\text{{Exacto}}}} + \sum_{{i=1}}^{{n}} \text{{Impacto}}(E_i)
    """)

    # Instanciación
    st.markdown("#### 2. Instanciación Numérica")
    str_vals = f"{vpn_base:,.0f}"
    for s in impactos:
        val = s['valor']
        str_vals += rf" + ({val:,.0f})"
    st.latex(rf"\mathcal{{H}}_K = {str_vals} = \mathbf{{ {vpn_total:,.0f} }}")

    # ==============================================================================
    # SECCIÓN 2: BENCHMARKING (MÉTODO TRADICIONAL)
    # ==============================================================================
    
    st.markdown("---")
    st.subheader("🆚 Benchmarking: Comparativa con Método Tradicional (DCF Discreto)")
    
    col_trad_1, col_trad_2 = st.columns([1, 1])
    
    with col_trad_1:
        st.markdown("**Método Tradicional (Discreto)**")
        st.markdown("Suma de flujos al final de cada año. Ignora el valor del dinero *intra-anual* y la continuidad.")
        st.latex(r"\text{VPN}_{trad} = \sum_{t=1}^{T} \frac{CF_t}{(1+r)^t}")
        
        delta_val = vpn_total - vpn_tradicional
        delta_pct = (delta_val / vpn_tradicional) * 100 if vpn_tradicional != 0 else 0
        
        st.metric("Resultado Tradicional", f"${vpn_tradicional:,.0f}")
        
    with col_trad_2:
        st.markdown("**Diferencia (Error de Discretización)**")
        st.info(f"""
        El Método Híbrido captura **${delta_val:,.0f}** adicionales ({delta_pct:+.2f}%) que el método tradicional pierde por aproximación.
        """)
        st.progress(min(100, max(0, int(50 + delta_pct*5)))) # Visual bar centered
    
    with st.expander("Ver tabla de cálculo Tradicional (Año a Año)", expanded=False):
        st.dataframe(df_tradicional.style.format("${:,.2f}"))

def asdict(shock: Shock):
    return {"id": shock.id, "nombre": shock.nombre, "tiempo": shock.tiempo, "magnitud": shock.magnitud, "activo": shock.activo, "descripcion": shock.descripcion}

if __name__ == "__main__":
    main()
