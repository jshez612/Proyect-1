import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dataclasses import dataclass
from typing import List, Tuple
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

# ---------- LÓGICA DEL MODELO (Optimizado) ----------

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
        # Valor Presente del Shock = Magnitud * Integral(kernel) desde t hasta T
        # Interpretación: El shock afecta el nivel de flujo desde t hasta el final T
        factor_descuento = self.integral_kernel(shock.tiempo, self.T)
        return float(shock.magnitud * factor_descuento)

    def generar_trayectorias(self, pasos: int = 500):
        t = np.linspace(0, self.T, pasos)
        # Flujo Base: A + Bt
        y_base = self.A + self.B * t
        # Flujo Total: Base + Shocks (función escalón Heaviside)
        y_total = y_base.copy()
        for s in self.shocks:
            if s.activo and s.tiempo < self.T:
                y_total += np.where(t >= s.tiempo, s.magnitud, 0)
        return t, y_base, y_total

# Caché para cálculos pesados (Sensibilidad)
@st.cache_data
def calcular_matriz_sensibilidad(A, B, r_start, r_end, t_start, t_end, shocks_data):
    """Genera una matriz de VPN variando r y T"""
    r_range = np.linspace(r_start, r_end, 20)
    t_range = np.linspace(t_start, t_end, 20)
    z_values = []
    
    for r_val in r_range:
        row = []
        for t_val in t_range:
            m = SolarisModel(r_val, t_val, A, B)
            # Reconstruir shocks temporales para el cálculo
            for s in shocks_data:
                m.add_shock(Shock(**s))
            
            vpn = m.calcular_vpn_base() + sum(m.calcular_impacto_shock(s) for s in m.shocks)
            row.append(vpn)
        z_values.append(row)
    
    return r_range, t_range, np.array(z_values)

# ---------- INTERFAZ DE USUARIO ----------

def main():
    # Encabezado
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
        st.markdown("Define $f(t) = A + B \cdot t$")
        col_a, col_b = st.columns(2)
        A_input = col_a.number_input("Inicial (A)", value=500.0, step=50.0)
        B_input = col_b.number_input("Crecimiento (B)", value=50.0, step=10.0)

    # Modelo inicial
    modelo = SolarisModel(r_input, T_input, A_input, B_input)

    # Gestión de Shocks (Eventos Discretos)
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚡ Eventos Discretos")
    
    # Shocks Default
    defaults = [
        {"id": "S1", "nombre": "Subsidio Verde", "tiempo": 2.0, "magnitud": 100.0},
        {"id": "S2", "nombre": "Fallo Inversores", "tiempo": 4.5, "magnitud": -80.0},
        {"id": "S3", "nombre": "Expansión Red", "tiempo": 7.0, "magnitud": 150.0},
    ]
    
    # Estado de sesión para shocks custom
    if 'custom_shocks' not in st.session_state:
        st.session_state.custom_shocks = defaults

    # Renderizar lista de shocks
    shocks_to_process = []
    shocks_data_for_cache = [] # Diccionarios simples para pasar a la función cacheada

    for idx, shock in enumerate(st.session_state.custom_shocks):
        with st.sidebar.container():
            c1, c2 = st.columns([0.8, 0.2])
            # Checkbox estilizado visualmente con nombre y datos
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
                # Para el cache, necesitamos dicts serializables
                shocks_data_for_cache.append(asdict(obj_shock))

    # Agregar nuevo shock
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

    # --- DASHBOARD PRINCIPAL ---
    
    # 1. KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("VPN Total", f"${vpn_total:,.0f}", delta=f"{vpn_shocks:,.0f} por eventos")
    k2.metric("Valor Base (Continuo)", f"${vpn_base:,.0f}", delta="Estructural")
    k3.metric("Impacto Eventos", f"${vpn_shocks:,.0f}", delta_color="off")
    k4.metric("Tasa Efectiva", f"{r_input*100:.1f}%", f"T={T_input} años")

    st.markdown("---")

    # 2. Pestañas de Análisis
    tab_chart, tab_waterfall, tab_sens, tab_data = st.tabs([
        "📈 Trayectoria Dinámica", 
        "🧱 Descomposición (Waterfall)", 
        "🎯 Mapa de Sensibilidad", 
        "📋 Datos"
    ])

    # --- GRAFICO DE TRAYECTORIA ---
    with tab_chart:
        t, y_base, y_total = modelo.generar_trayectorias()
        
        fig = go.Figure()
        # Area Base
        fig.add_trace(go.Scatter(
            x=t, y=y_base, mode='lines', name='Flujo Base',
            line=dict(color='#4B5563', width=1, dash='dash'),
            fill=None
        ))
        # Area Total con Gradiente (simulado visualmente con fill tonexty)
        fig.add_trace(go.Scatter(
            x=t, y=y_total, mode='lines', name='Flujo Híbrido',
            line=dict(color='#60A5FA', width=3),
            fill='tonexty', # Rellena hasta la traza anterior (Base)
            fillcolor='rgba(96, 165, 250, 0.1)'
        ))

        # Marcadores de eventos
        for s in shocks_to_process:
            color = '#34D399' if s.magnitud > 0 else '#F87171'
            fig.add_vline(x=s.tiempo, line_dash="dot", line_color=color, opacity=0.6)
            fig.add_annotation(
                x=s.tiempo, y=max(y_total)*1.05,
                text=s.nombre, showarrow=False,
                font=dict(color=color, size=10),
                yshift=10
            )

        fig.update_layout(
            title="Dinámica del Flujo de Caja Instantáneo",
            xaxis_title="Tiempo (años)",
            yaxis_title="Flujo ($/año)",
            template="plotly_dark",
            hovermode="x unified",
            height=450,
            margin=dict(l=20, r=20, t=60, b=20),
            legend=dict(orientation="h", y=1.02, yanchor="bottom", x=1, xanchor="right")
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- WATERFALL CHART ---
    with tab_waterfall:
        # Preparar datos para Waterfall
        wf_labels = ["Base Estructural"] + [x['nombre'] for x in impactos] + ["VPN Total"]
        wf_values = [vpn_base] + [x['valor'] for x in impactos] + [0] # El último se calcula auto
        
        # El tipo 'waterfall' calcula los totales automáticamente si usamos 'relative' y 'total'
        measure_types = ["absolute"] + ["relative"] * len(impactos) + ["total"]
        wf_y = [vpn_base] + [x['valor'] for x in impactos] + [0] # El valor final es dummy para 'total'

        fig_wf = go.Figure(go.Waterfall(
            name="Composición", orientation="v",
            measure=measure_types,
            x=wf_labels,
            y=wf_y,
            textposition="outside",
            text=[f"${v/1000:.1f}k" for v in wf_y[:-1]] + [f"${vpn_total/1000:.1f}k"],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            decreasing={"marker": {"color": "#F87171"}},
            increasing={"marker": {"color": "#34D399"}},
            totals={"marker": {"color": "#60A5FA"}}
        ))

        fig_wf.update_layout(
            title="Descomposición del Valor (Acumulación)",
            template="plotly_dark",
            yaxis_title="Valor Presente Net ($)",
            height=500
        )
        st.plotly_chart(fig_wf, use_container_width=True)

    # --- SENSIBILIDAD (HEATMAP) ---
    with tab_sens:
        st.markdown("#### Impacto de Tasa (r) vs Tiempo (T)")
        st.caption("Calculando matriz de VPN en tiempo real...")
        
        # Usamos la función cacheada para rapidez
        r_vals, t_vals, z_vals = calcular_matriz_sensibilidad(
            A_input, B_input, 
            max(0.01, r_input - 0.05), r_input + 0.05,
            max(1.0, T_input - 5), T_input + 5,
            shocks_data_for_cache
        )

        fig_hm = go.Figure(data=go.Heatmap(
            z=z_vals, x=t_vals, y=r_vals,
            colorscale='Viridis',
            colorbar=dict(title="VPN ($)"),
            hovertemplate='T: %{x:.1f} años<br>r: %{y:.1%}<br>VPN: $%{z:,.0f}<extra></extra>'
        ))
        
        # Marcar el punto actual
        fig_hm.add_trace(go.Scatter(
            x=[T_input], y=[r_input], mode='markers',
            marker=dict(color='red', symbol='x', size=12, line=dict(width=2, color='white')),
            name='Escenario Actual'
        ))

        fig_hm.update_layout(
            template="plotly_dark",
            xaxis_title="Horizonte (T)",
            yaxis_title="Tasa de Descuento (r)",
            height=500
        )
        st.plotly_chart(fig_hm, use_container_width=True)

    # --- TABLA DE DATOS ---
    with tab_data:
        # Crear DataFrame limpio
        data_rows = [{"Componente": "Flujo Base", "Tiempo (t)": "-", "Magnitud (Flux)": f"{A_input} + {B_input}t", "VPN Impacto": vpn_base}]
        for item in impactos:
            data_rows.append({
                "Componente": item['nombre'],
                "Tiempo (t)": f"{item['tiempo']:.1f}",
                "Magnitud (Flux)": f"{item['magnitud']:.2f}",
                "VPN Impacto": item['valor']
            })
        
        df = pd.DataFrame(data_rows)
        
        # Mostrar DataFrame estilizado
        st.dataframe(
            df.style.format({"VPN Impacto": "${:,.2f}"}),
            use_container_width=True
        )
        
        # Botón CSV
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "⬇️ Descargar Reporte CSV",
            data=csv,
            file_name="acumulacion_hibrida_reporte.csv",
            mime="text/csv"
        )

def asdict(shock: Shock):
    """Helper para convertir dataclass a dict para cacheo"""
    return {"id": shock.id, "nombre": shock.nombre, "tiempo": shock.tiempo, "magnitud": shock.magnitud, "activo": shock.activo, "descripcion": shock.descripcion}

if __name__ == "__main__":
    main()
