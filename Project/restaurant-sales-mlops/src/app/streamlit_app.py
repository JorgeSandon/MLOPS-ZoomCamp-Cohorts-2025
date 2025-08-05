import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="Restaurant Sales Predictor",
    page_icon="🍔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths
MODEL_PATH = "data/outputs/best_model.pkl"
MODEL_INFO_PATH = "data/outputs/model_info.json"
DATA_PATH = "data/processed/sales_clean.csv"
PROCESSED_DIR = "data/processed"

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(45deg, #f0f2f6, #ffffff);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .prediction-result {
        background: linear-gradient(45deg, #e8f5e8, #f0fff0);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
        border: 2px solid #28a745;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .sidebar-info {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .error-box {
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Carga los datos procesados"""
    # Buscar archivo de datos
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    elif os.path.exists(PROCESSED_DIR):
        csv_files = [f for f in os.listdir(PROCESSED_DIR) if f.endswith('.csv')]
        if csv_files:
            return pd.read_csv(os.path.join(PROCESSED_DIR, csv_files[0]))
    return None

@st.cache_resource
def load_model():
    """Carga el modelo entrenado"""
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

@st.cache_data
def load_model_info():
    """Carga información del modelo"""
    if os.path.exists(MODEL_INFO_PATH):
        with open(MODEL_INFO_PATH, 'r') as f:
            return json.load(f)
    return None

def create_visualization(data):
    """Crea visualizaciones de los datos"""
    
    if data is None or data.empty:
        st.warning("No hay datos disponibles para visualizar")
        return
    
    # Crear subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Distribución de Ventas', 'Precio vs Ventas', 
                       'Cantidad vs Ventas', 'Correlación'),
        specs=[[{"type": "histogram"}, {"type": "scatter"}],
               [{"type": "scatter"}, {"type": "heatmap"}]]
    )
    
    # Identificar columnas numéricas
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) >= 3:
        # Asumir que las primeras columnas son Price, Quantity, Total_Sales
        price_col = numeric_cols[0]
        quantity_col = numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0]
        sales_col = numeric_cols[-1]  # Última columna como ventas
        
        # Histograma de ventas
        fig.add_trace(
            go.Histogram(x=data[sales_col], name="Ventas", nbinsx=30),
            row=1, col=1
        )
        
        # Scatter plots
        fig.add_trace(
            go.Scatter(x=data[price_col], y=data[sales_col], 
                      mode='markers', name="Precio vs Ventas",
                      marker=dict(color='blue', size=6, opacity=0.6)),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(x=data[quantity_col], y=data[sales_col], 
                      mode='markers', name="Cantidad vs Ventas",
                      marker=dict(color='red', size=6, opacity=0.6)),
            row=2, col=1
        )
        
        # Matriz de correlación
        corr_matrix = data[numeric_cols].corr()
        fig.add_trace(
            go.Heatmap(z=corr_matrix.values, 
                      x=corr_matrix.columns, 
                      y=corr_matrix.columns,
                      colorscale='RdBu', 
                      text=np.round(corr_matrix.values, 2),
                      texttemplate="%{text}",
                      textfont={"size": 10}),
            row=2, col=2
        )
    
    fig.update_layout(height=800, showlegend=False, 
                     title_text="Análisis Exploratorio de Datos")
    
    return fig

def main():
    # Header principal
    st.markdown('<h1 class="main-header">🍔 Restaurant Sales Predictor</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📊 Información del Sistema")
        
        # Cargar información del modelo
        model_info = load_model_info()
        model = load_model()
        
        if model_info:
            st.markdown(f"""
            <div class="sidebar-info">
                <b>🤖 Modelo:</b> {model_info.get('model_name', 'N/A')}<br>
                <b>📈 R² Score:</b> {model_info.get('r2_score', 0):.3f}<br>
                <b>🎯 Features:</b> {len(model_info.get('features', []))}<br>
                <b>⏰ Estado:</b> {'✅ Listo' if model else '❌ No disponible'}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="sidebar-info">
                <b>⚠️ Información del modelo no disponible</b><br>
                Ejecuta el pipeline de entrenamiento primero.
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### ⚙️ Configuración")
        
        # Configuraciones de predicción
        confidence_interval = st.checkbox("Mostrar intervalo de confianza", value=False)
        show_charts = st.checkbox("Mostrar gráficos", value=True)
        
        st.markdown("---")
        st.markdown("### 📋 Instrucciones")
        st.markdown("""
        1. Ingresa el precio del producto
        2. Ingresa la cantidad vendida
        3. Haz clic en 'Predecir Ventas'
        4. Revisa los resultados y análisis
        """)
    
    # Contenido principal
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 🎯 Realizar Predicción")
        
        if model is None:
            st.markdown("""
            <div class="error-box">
                <h4>❌ Modelo no disponible</h4>
                <p>No se pudo cargar el modelo entrenado. Por favor:</p>
                <ol>
                    <li>Ejecuta el pipeline de entrenamiento</li>
                    <li>Verifica que existe: <code>data/outputs/best_model.pkl</code></li>
                    <li>Recarga la aplicación</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Inputs para predicción
            col_price, col_quantity = st.columns(2)
            
            with col_price:
                price = st.number_input(
                    "💰 Precio del producto ($)",
                    min_value=0.01,
                    max_value=1000.0,
                    value=15.50,
                    step=0.50,
                    help="Ingresa el precio unitario del producto"
                )
            
            with col_quantity:
                quantity = st.number_input(
                    "📦 Cantidad vendida",
                    min_value=1,
                    max_value=10000,
                    value=100,
                    step=1,
                    help="Ingresa la cantidad de productos vendidos"
                )
            
            # Botón de predicción
            if st.button("🚀 Predecir Ventas Totales", use_container_width=True):
                try:
                    # Realizar predicción
                    prediction = model.predict([[price, quantity]])[0]
                    
                    # Calcular métricas adicionales
                    revenue_per_unit = prediction / quantity if quantity > 0 else 0
                    profit_margin = ((prediction - (price * quantity)) / prediction * 100) if prediction > 0 else 0
                    
                    # Mostrar resultado principal
                    st.markdown(f"""
                    <div class="prediction-result">
                        <h2>💰 Ventas Estimadas</h2>
                        <h1 style="color: #28a745; margin: 1rem 0;">${prediction:,.2f}</h1>
                        <p style="font-size: 1.1rem;">Basado en {price} × {quantity} unidades</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Métricas adicionales
                    col_m1, col_m2, col_m3 = st.columns(3)
                    
                    with col_m1:
                        st.metric(
                            "💵 Ingreso por Unidad",
                            f"${revenue_per_unit:.2f}",
                            delta=f"{((revenue_per_unit - price) / price * 100):+.1f}%" if price > 0 else None
                        )
                    
                    with col_m2:
                        st.metric(
                            "📊 Margen Estimado",
                            f"{profit_margin:.1f}%",
                            delta="Estimado" if profit_margin > 0 else "Revisar"
                        )
                    
                    with col_m3:
                        st.metric(
                            "🎯 Precio Promedio",
                            f"${price:.2f}",
                            delta=f"×{quantity} unidades"
                        )
                    
                    # Análisis de sensibilidad
                    if confidence_interval:
                        st.markdown("### 📈 Análisis de Sensibilidad")
                        
                        # Crear rangos de variación
                        price_range = np.linspace(price * 0.8, price * 1.2, 10)
                        quantity_range = np.linspace(quantity * 0.8, quantity * 1.2, 10)
                        
                        predictions_price = [model.predict([[p, quantity]])[0] for p in price_range]
                        predictions_quantity = [model.predict([[price, q]])[0] for q in quantity_range]
                        
                        col_sens1, col_sens2 = st.columns(2)
                        
                        with col_sens1:
                            fig_price = px.line(
                                x=price_range, y=predictions_price,
                                title="Sensibilidad al Precio",
                                labels={'x': 'Precio ($)', 'y': 'Ventas Predichas ($)'}
                            )
                            fig_price.add_vline(x=price, line_dash="dash", line_color="red")
                            st.plotly_chart(fig_price, use_container_width=True)
                        
                        with col_sens2:
                            fig_quantity = px.line(
                                x=quantity_range, y=predictions_quantity,
                                title="Sensibilidad a la Cantidad",
                                labels={'x': 'Cantidad', 'y': 'Ventas Predichas ($)'}
                            )
                            fig_quantity.add_vline(x=quantity, line_dash="dash", line_color="red")
                            st.plotly_chart(fig_quantity, use_container_width=True)
                
                except Exception as e:
                    st.error(f"Error al realizar predicción: {str(e)}")
    
    with col2:
        st.markdown("### 📊 Datos y Análisis")
        
        # Cargar y mostrar datos
        data = load_data()
        
        if data is not None:
            # Estadísticas del dataset
            st.markdown("#### 📈 Estadísticas del Dataset")
            
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            
            with col_stat1:
                st.metric("📋 Total Registros", f"{len(data):,}")
            
            with col_stat2:
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    avg_sales = data[numeric_cols[-1]].mean()
                    st.metric("💰 Venta Promedio", f"${avg_sales:,.2f}")
            
            with col_stat3:
                st.metric("📊 Columnas", f"{len(data.columns)}")
            
            # Vista previa de datos
            with st.expander("🔍 Vista Previa de Datos", expanded=False):
                st.dataframe(data.head(10), use_container_width=True)
            
            # Gráficos
            if show_charts:
                st.markdown("#### 📊 Visualizaciones")
                try:
                    fig = create_visualization(data)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"No se pudieron generar gráficos: {str(e)}")
        else:
            st.warning("No se pudieron cargar los datos para análisis")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>🍔 Restaurant Sales Predictor | Powered by MLOps Pipeline</p>
        <p>Última actualización: {}</p>
    </div>
    """.format(datetime.now().strftime("%Y-%m-%d %H:%M")), unsafe_allow_html=True)

if __name__ == "__main__":
    main()