import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.special as sc
from dataclasses import dataclass, asdict
from typing import List, Tuple  # Importación necesaria
import math

# ---------- CONFIGURACIÓN GENERAL ----------
st.set_page_config(
    page_title="Operador de Acumulación Híbrida",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- ESTILOS CSS PERSONALIZADOS ----------
st.markdown("""
    <style>
    /* Fondo general y fuentes */
    .main { background-color: #0E1117; }
    h1, h2, h3 { font-family: 'Helvetica Neue', sans-serif; font-weight: 600; }
    
    /* Tarjetas de métricas personalizadas */
    div[data-testid="metric-container"] {
        background-color: #1F2937;
        border: 1px solid #374151;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        border-color: #60A5FA;
    }
    label[data-testid="stMetricLabel"] { color: #9CA3AF !important; }
    div[data-testid="stMetricValue"] { color: #F3F4F6 !important; font-size: 1.8rem !important; }
    
    /* Ajustes de Sidebar */
    [data-testid="stSidebar"] { background-color: #111827; border-right: 1px solid #374151; }
    </style>
    """, unsafe_allow_html=True)

# ---------- LÓGICA DEL MODELO ----------

@dataclass
class Shock:
    id: str
    nombre: str
    tiempo: float
    magnitud: float
    activo: bool = True
    descripcion: str = ""

class SolarisModel:
    """
    Motor de valoración analítica híbrida.
    Calcula integrales exactas para evitar errores de discretización.
    """
    def __init__(self, r: float, T: float, A: float, B: float):
        self.r = float(r)
        self.T = float(T)
        self.A = float(A)
        self.B = float(B)
        self.shocks: List[Shock] = []

    def add_shock(self, shock: Shock):
        self.shocks.append(shock)

    def integral_kernel(self, a: float, b: float) -> float:
        """Integral de e^(-rt) dt entre a y b"""
        if b <= a: return 0.0
        if abs(self.r) < 1e-9: return b - a
        return (math.exp(-self.r * a) - math.exp(-self.r * b)) / self.r

    def calcular_vpn_base(self) -> float:
        """VPN de la parte continua f(t) = A + B*t"""
        r, T, A, B = self.r, self.T, self.A, self.B
        if abs(r) < 1e-9:
            return A * T + 0.5 * B * T**2
        term_A = A * (1 - math.exp(-r * T)) / r
        term_B = B * ((1 - math.exp(-r * T)) / r**2 - T * math.exp(-r * T) / r)
        return term_A + term_B

    def calcular_impacto_shock(self, shock: Shock) -> float:
        """Impacto en VPN de un salto discreto en el tiempo t"""
        if not shock.activo or shock.tiempo >= self.T:
            return 0.0
        factor_descuento = self.integral_kernel(shock.tiempo, self.T)
        return float(shock.magnitud * factor_descuento)

    def generar_trayectorias(self, pasos: int = 500):
        t = np.linspace(0, self.T, pasos)
        y_base = self.A + self.B * t
        y_total = y_base.copy()
        for s in self.shocks:
            if s.activo and s.tiempo < self.T:
                y_total += np.where(t >= s.tiempo, s.magnitud, 0)
        return t, y_base, y_total

    def obtener_datos_discretos(self, n_points: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Extrae 'n' puntos representativos para el análisis de Sistemas Grises"""
        t_discrete = np.linspace(max(0, self.T - n_points + 1), self.T, n_points)
        y_vals = []
        for ti in t_discrete:
            val = self.A + self.B * ti
            for s in self.shocks:
                if s.activo and s.tiempo <= ti:
                    val += s.magnitud
            y_vals.append(val)
        return t_discrete, np.array(y_vals)

# Caché para cálculos pesados
@st.cache_data
def calcular_matriz_sensibilidad(A, B, r_start, r_end, t_start, t_end, shocks_data):
    r_range = np.linspace(r_start, r_end, 20)
    t_range = np.linspace(t_start, t_end, 20)
    z_values = []
    
    for r_val in r_range:
        row = []
        for t_val in t_range:
            m = SolarisModel(r_val, t_val, A, B)
            for s in shocks_data:
                m.add_shock(Shock(**s))
            vpn = m.calcular_vpn_base() + sum(m.calcular_impacto_shock(s) for s in m.shocks)
            row.append(vpn)
        z_values.append(row)
    return r_range, t_range, np.array(z_values)

def calcular_acumulacion_hibrida(series: np.ndarray, r_order: float, lambda_param: float):
    """
    Implementación matemática del Operador H_K.
    """
    n = len(series)
    weights = np.array([sc.gamma(r_order + n - i) / (sc.gamma(n - i + 1) * sc.gamma(r_order)) for i in range(1, n + 1)])
    trend_component = np.sum(series * weights)
    jump_component = series[-1]
    hk_value = (1 - lambda_param) * trend_component + lambda_param * jump_component
    return hk_value, trend_component, jump_component, weights

# ---------- INTERFAZ DE USUARIO ----------

def main():
    col_title, col_logo = st.columns([4, 1])
    with col_title:
        st.title("💠 Operador de Acumulación Híbrida")
        st.markdown("Calculadora de Valoración Dinámica con **Kernel Exponencial**")
    
    # --- SIDEBAR ---
    st.sidebar.header("⚙️ Configuración")
    
    with st.sidebar.expander("1. Parámetros Globales", expanded=True):
        r_input = st.slider("Tasa de Descuento (r)", 0.0, 0.25, 0.08, 0.005, format="%.1f%%")
        T_input = st.number_input("Horizonte (T años)", 1.0, 50.0, 10.0, 0.5)
    
    with st.sidebar.expander("2. Flujo Base (Continuo)", expanded=False):
        col_a, col_b = st.columns(2)
        A_input = col_a.number_input("Inicial (A)", value=500.0, step=50.0)
        B_input = col_b.number_input("Crecimiento (B)", value=50.0, step=10.0)

    modelo = SolarisModel(r_input, T_input, A_input, B_input)

    # Gestión de Shocks
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
    shocks_data_for_cache = [] 

    for idx, shock in enumerate(st.session_state.custom_shocks):
        with st.sidebar.container():
            c1, c2 = st.columns([0.8, 0.2])
            is_active = c1.checkbox(
                f"{shock['nombre']} (t={shock['tiempo']} | Δ={shock['magnitud']})", 
                value=True, key=f"chk_{idx}"
            )
            if c2.button("✖", key=f"del_{idx}"):
                st.session_state.custom_shocks.pop(idx)
                st.rerun()
            
            if is_active:
                obj_shock = Shock(id=shock['id'], nombre=shock['nombre'], tiempo=shock['tiempo'], magnitud=shock['magnitud'])
                modelo.add_shock(obj_shock)
                shocks_to_process.append(obj_shock)
                shocks_data_for_cache.append(asdict(obj_shock))

    with st.sidebar.expander("➕ Agregar Evento", expanded=False):
        with st.form("add_shock"):
            n_name = st.text_input("Nombre", "Nuevo Evento")
            c_t, c_m = st.columns(2)
            n_time = c_t.number_input("Año", 0.0, T_input, 1.0)
            n_mag = c_m.number_input("Impacto ($)", value=50.0)
            if st.form_submit_button("Añadir"):
                st.session_state.custom_shocks.append(
                    {"id": f"C{len(st.session_state.custom_shocks)}", "nombre": n_name, "tiempo": n_time, "magnitud": n_mag}
                )
                st.rerun()

    # --- CÁLCULOS ---
    vpn_base = modelo.calcular_vpn_base()
    impactos = []
    for s in shocks_to_process:
        val = modelo.calcular_impacto_shock(s)
        impactos.append({"nombre": s.nombre, "valor": val, "tiempo": s.tiempo, "magnitud": s.magnitud})
    
    vpn_shocks = sum(x['valor'] for x in impactos)
    vpn_total = vpn_base + vpn_shocks

    # --- DASHBOARD ---
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("VPN Total", f"${vpn_total:,.0f}", delta=f"{vpn_shocks:,.0f} eventos")
    k2.metric("Valor Base", f"${vpn_base:,.0f}", delta="Estructural")
    k3.metric("Impacto Eventos", f"${vpn_shocks:,.0f}", delta_color="off")
    k4.metric("Tasa Efectiva", f"{r_input*100:.1f}%", f"T={T_input}")

    st.markdown("---")

    tab_chart, tab_waterfall, tab_sens, tab_data = st.tabs([
        "📈 Trayectoria", "🧱 Descomposición", "🎯 Sensibilidad", "📋 Datos"
    ])

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

    with tab_sens:
        r_vals, t_vals, z_vals = calcular_matriz_sensibilidad(
            A_input, B_input, max(0.01, r_input - 0.05), r_input + 0.05, max(1.0, T_input - 5), T_input + 5, shocks_data_for_cache
        )
        fig_hm = go.Figure(data=go.Heatmap(z=z_vals, x=t_vals, y=r_vals, colorscale='Viridis'))
        fig_hm.add_trace(go.Scatter(x=[T_input], y=[r_input], mode='markers', marker=dict(color='red', symbol='x', size=12)))
        fig_hm.update_layout(template="plotly_dark", height=500, title="Sensibilidad VPN (r vs T)")
        st.plotly_chart(fig_hm, use_container_width=True)

    with tab_data:
        df = pd.DataFrame(impactos)
        st.dataframe(df)
        st.download_button("⬇️ CSV", df.to_csv().encode('utf-8'), "data.csv")

    # ==============================================================================
    # APLICACIÓN MATEMÁTICA CORREGIDA
    # ==============================================================================
    
    st.markdown("---")
    st.subheader("🧮 Aplicación: Operador de Acumulación Híbrida ($\mathcal{H}_K$)")
    
    n_sample = 5
    t_disc, y_disc = modelo.obtener_datos_discretos(n_points=n_sample)
    
    col_math_ctrl, col_math_viz = st.columns([1, 2])
    
    with col_math_ctrl:
        lambda_h = st.slider("Prioridad Nueva Información ($\lambda$)", 0.0, 1.0, 0.6, 0.05)
        r_ago = st.slider("Orden Fraccionario ($r$)", 0.1, 1.5, 0.5, 0.1)
        hk_val, comp_trend, comp_jump, weights = calcular_acumulacion_hibrida(y_disc, r_ago, lambda_h)
        st.metric("Valor $\mathcal{H}_K$", f"${hk_val:,.0f}")

    with col_math_viz:
        st.markdown("#### Instanciación Numérica")
        
        str_trend = f"({1-lambda_h:.2f}) \\times [{comp_trend:,.0f}]"
        str_jump = f"({lambda_h:.2f}) \\times [{comp_jump:,.0f}]"
        
        # CORRECCIÓN AQUÍ: Usamos {{H}} para escapar las llaves en f-string
        st.latex(rf"""
        \mathcal{{H}}_K = {str_trend} + {str_jump} = \mathbf{{ {hk_val:,.2f} }}
        """)
        
        st.markdown("**Datos Recientes**")
        st.dataframe(pd.DataFrame({"t": t_disc, "x(t)": y_disc}).T)

def asdict(shock: Shock):
    return {"id": shock.id, "nombre": shock.nombre, "tiempo": shock.tiempo, "magnitud": shock.magnitud, "activo": shock.activo, "descripcion": shock.descripcion}

if __name__ == "__main__":
    main()
