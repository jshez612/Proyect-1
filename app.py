%%writefile .streamlit/config.toml
[theme]
primaryColor="#1f77b4"
backgroundColor="#0E1117"
secondaryBackgroundColor="#262730"
textColor="#FAFAFA"
font="sans serif"

# 3. Escribir la aplicación
%%writefile app.py
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.integrate import quad
from dataclasses import dataclass
from typing import List

# --- CONFIGURACIÓN VISUAL PRO ---
st.set_page_config(
    page_title="Solaris: Valuation Engine",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados (Reforzados para la Sidebar)
st.markdown("""
    <style>
    /* Forzar visibilidad de inputs en la sidebar */
    [data-testid="stSidebar"] input {
        color: white !important;
    }
    /* Estilo para las métricas (KPIs) */
    [data-testid="stMetric"] {
        background-color: #262730; /* Fondo oscuro para coincidir con el tema */
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    [data-testid="stMetricLabel"] { color: #FAFAFA !important; }
    [data-testid="stMetricValue"] { color: #1f77b4 !important; }
    
    /* Ajuste de checkboxes */
    [data-testid="stCheckbox"] {
        padding-top: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 1. LÓGICA MATEMÁTICA ---

@dataclass
class Shock:
    id: str
    nombre: str
    tiempo: float
    magnitud: float
    activo: bool = True
    descripcion: str = ""

class SolarisModel:
    def __init__(self, r: float, T: float):
        self.r = r
        self.T = T
        self.shocks: List[Shock] = []

    def flujo_base(self, t):
        return 500 + 50 * t

    def kernel(self, t):
        return np.exp(-self.r * t)

    def add_shock(self, shock: Shock):
        self.shocks.append(shock)

    def calcular_vpn_base(self):
        integrando = lambda t: self.flujo_base(t) * self.kernel(t)
        res, _ = quad(integrando, 0, self.T)
        return res

    def calcular_potencial_shock(self, shock: Shock):
        if not shock.activo or shock.tiempo >= self.T:
            return 0.0
        integral_k, _ = quad(self.kernel, shock.tiempo, self.T)
        return shock.magnitud * integral_k

    def obtener_datos_grafica(self, pasos=1000):
        t = np.linspace(0, self.T, pasos)
        y_base = self.flujo_base(t)
        y_total = y_base.copy()
        for s in self.shocks:
            if s.activo:
                y_total += np.where(t >= s.tiempo, s.magnitud, 0)
        return t, y_base, y_total

# --- 2. INTERFAZ DE USUARIO ---

def main():
    col_header1, col_header2 = st.columns([3, 1])
    with col_header1:
        st.title("☀️ Solaris: Valuation Engine")
        st.markdown(r"**Operador de Acumulación Híbrida ($\mathcal{H}_K$)** | *Real-time Sensitivity Analysis*")
    
    # --- SIDEBAR ---
    st.sidebar.header("🎛️ Panel de Control")
    
    with st.sidebar.container():
        st.subheader("Parámetros Globales")
        r_input = st.sidebar.slider("Tasa de Descuento (r)", 0.0, 0.20, 0.08, 0.005, format="%.3f")
        T_input = 10.0

    modelo = SolarisModel(r_input, T_input)

    if 'custom_shocks' not in st.session_state:
        st.session_state.custom_shocks = []

    st.sidebar.markdown("---")
    st.sidebar.subheader("⚡ Gestión de Eventos")

    base_shocks = [
        ("A", "Subsidio Verde", 2.0, 100.0, "Política Gubernamental"),
        ("B", "Fallo Inversores", 4.5, -80.0, "Fin de garantía"),
        ("C", "Expansión Red", 7.0, 150.0, "Nueva infraestructura"),
        ("D", "Riesgo Regulatorio", 1.5, -40.0, "Posible impuesto")
    ]

    for sid, name, t, mag, desc in base_shocks:
        # Aumentamos el ancho de la columna del checkbox para que no se corte
        col_check, col_info = st.sidebar.columns([0.2, 0.8])
        act = col_check.checkbox("", value=True, key=f"check_{sid}")
        col_info.markdown(f"**{name}** <br><span style='font-size:0.8em; color:gray'>t={t} | Δ={mag}</span>", unsafe_allow_html=True)
        if act:
            modelo.add_shock(Shock(sid, name, t, mag, True, desc))

    if st.session_state.custom_shocks:
        st.sidebar.markdown("---")
        st.sidebar.caption("Eventos Personalizados")
        for i, cs in enumerate(st.session_state.custom_shocks):
            col_c1, col_c2, col_c3 = st.sidebar.columns([0.2, 0.65, 0.15])
            act_c = col_c1.checkbox("", value=True, key=f"custom_check_{i}")
            col_c2.markdown(f"**{cs['n']}** <br><span style='font-size:0.8em'>t={cs['t']} | Δ={cs['m']}</span>", unsafe_allow_html=True)
            if col_c3.button("🗑️", key=f"del_{i}"):
                st.session_state.custom_shocks.pop(i)
                st.rerun()
            if act_c:
                modelo.add_shock(Shock(f"C{i}", cs['n'], cs['t'], cs['m'], True))

    with st.sidebar.expander("➕ Agregar Nuevo Evento", expanded=False):
        with st.form("new_shock_form"):
            new_name = st.text_input("Nombre", "Crisis X")
            c1, c2 = st.columns(2)
            new_time = c1.number_input("Año (t)", 0.0, 10.0, 5.0)
            new_mag = c2.number_input("Impacto ($k)", value=-50.0)
            submitted = st.form_submit_button("Añadir", use_container_width=True)
            if submitted:
                st.session_state.custom_shocks.append({"n": new_name, "t": new_time, "m": new_mag})
                st.rerun()

    # --- CÁLCULOS ---
    vpn_base = modelo.calcular_vpn_base()
    shocks_calc = []
    vpn_shocks = 0
    for s in modelo.shocks:
        val = modelo.calcular_potencial_shock(s)
        shocks_calc.append({'nombre': s.nombre, 'valor': val})
        vpn_shocks += val
    
    vpn_total = vpn_base + vpn_shocks

    # --- VISUALIZACIÓN ---
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("VPN Total (Híbrido)", f"${vpn_total:,.2f}", delta=f"{vpn_shocks:,.2f} vs Base")
    kpi2.metric("VPN Base (Continuo)", f"${vpn_base:,.2f}")
    kpi3.metric("Impacto Neto Eventos", f"${vpn_shocks:,.2f}", delta="Positivo" if vpn_shocks > 0 else "Negativo")

    st.markdown("---")

    tab1, tab2 = st.tabs(["📈 Trayectoria Dinámica", "📊 Descomposición del Valor"])

    with tab1:
        t, y_base, y_total = modelo.obtener_datos_grafica()
        fig_traj = go.Figure()
        
        # Trayectoria Base (Línea punteada)
        fig_traj.add_trace(go.Scatter(
            x=t, y=y_base, mode='lines', name='Trayectoria Base',
            line=dict(color='gray', width=2, dash='dash'),
            hovertemplate='Base: $%{y:.0f}<extra></extra>'
        ))
        
        # Trayectoria Híbrida (Línea sólida)
        fig_traj.add_trace(go.Scatter(
            x=t, y=y_total, mode='lines', name='Trayectoria Híbrida',
            fill='tonexty', fillcolor='rgba(31, 119, 180, 0.1)',
            line=dict(color='#1f77b4', width=3),
            hovertemplate='Total: $%{y:.0f}<extra></extra>'
        ))
        
        # Marcadores de Saltos
        for s in modelo.shocks:
            color = '#2ecc71' if s.magnitud > 0 else '#e74c3c'
            fig_traj.add_vline(x=s.tiempo, line_width=1, line_dash="dot", line_color=color)
            # Anotación estilo Desmos
            fig_traj.add_annotation(
                x=s.tiempo, y=min(y_base) if s.magnitud < 0 else max(y_total),
                text=s.nombre, showarrow=False, yshift=10,
                font=dict(size=10, color=color)
            )
            
        fig_traj.update_layout(
            title="Dinámica del Flujo de Caja", 
            xaxis_title="Tiempo (Años)",
            yaxis_title="Flujo de Caja ($k/año)", 
            hovermode="x unified",
            template="plotly_dark", # Usamos tema oscuro para coincidir con la config
            height=500,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_traj, use_container_width=True)

    with tab2:
        labels = ["Base"] + [s['nombre'] for s in shocks_calc]
        values = [vpn_base] + [s['valor'] for s in shocks_calc]
        colors = ['#3498db'] + ['#2ecc71' if v >= 0 else '#e74c3c' for v in values[1:]]
        text_labels = [f"${v:,.0f}" for v in values]

        # Gráfica de Barras corregida
        bar_obj = go.Bar(
            x=labels, 
            y=values,
            marker_color=colors,
            text=text_labels,
            textposition='auto',
            hovertemplate='%{x}: $%{y:,.2f}<extra></extra>'
        )

        fig_bar = go.Figure(data=[bar_obj])

        fig_bar.update_layout(
            title="Descomposición Modular del VPN",
            yaxis_title="Valor Presente ($k)",
            template="plotly_dark",
            height=500
        )
        st.plotly_chart(fig_bar, use_container_width=True)

if __name__ == "__main__":
    main()
