import streamlit as st
import streamlit_authenticator as stauth
import pandas as pd
import joblib
import shap
import yaml
from yaml.loader import SafeLoader
from azure.storage.blob import BlobServiceClient
import io  # Para manejar archivos en memoria
import matplotlib.pyplot as plt # Necesario para el gráfico SHAP
from sklearn.preprocessing import StandardScaler # Para escalar
import numpy as np # Necesario para SHAP

# --- 1. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="OncoEarly - Cáncer Gástrico", layout="centered")

# --- 2. CARGA DE USUARIOS (Desde config.yaml) ---
try:
    with open('config.yaml') as file:
        config = yaml.load(file, Loader=SafeLoader)
except FileNotFoundError:
    st.error("Error: Archivo 'config.yaml' no encontrado. Asegúrate de que exista en la carpeta.")
    st.stop() # Detiene la ejecución si no hay config.yaml

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

# --- 3. PÁGINA DE LOGIN (Página 1) ---
try:
    st.image("oncoearly-sinfondo.png", width=300) 
except FileNotFoundError:
    st.error("No se encontró el logo. Asegúrate de que 'oncoearly-sinfondo.png' esté en la carpeta.")

# CSS para ocultar el botón "Registrate" y el footer
st.markdown("""
<style>
[data-testid="stForm"] + div a { display: none; }
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

name, authentication_status, username = authenticator.login()


# --- 4. FUNCIÓN DE CARGA DEL MODELO (Desde Azure) ---
@st.cache_resource # Cachea el modelo descargado
def load_model_from_azure():
    """
    Se conecta a Azure Blob Storage, descarga el modelo y lo carga en memoria.
    """
    try:
        connection_string = st.secrets["azure_storage"]["connection_string"]
        container_name = "modelos-ml"
        blob_name = "modelo_rf_entrenado-v2.joblib" # ¡Nombre de tu modelo!
        
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
        
        with st.spinner("Descargando y cargando modelo de IA... 🧠"):
            downloader = blob_client.download_blob()
            blob_bytes = downloader.readall()
            model = joblib.load(io.BytesIO(blob_bytes))
        
        st.success("Modelo cargado exitosamente. ✅")
        return model
    except Exception as e:
        st.error(f"Error crítico al cargar el modelo desde Azure: {e}")
        return None

# --- 5. FUNCIÓN DE EXPLICABILIDAD (SHAP) ---
# Cacheamos el explainer para eficiencia
@st.cache_resource
def get_shap_explainer(_model, _background_data):
    """Crea el objeto explicador de SHAP usando datos de fondo."""
    background_int = _background_data.astype(int)
    explainer = shap.TreeExplainer(_model, background_int)
    return explainer

def plot_shap_force_plot(explainer, input_data):
    """Genera y muestra el gráfico SHAP force plot."""
    st.subheader("Factores clave para ESTE paciente (SHAP):")
    try:
        input_data_int = input_data.astype(int)
        shap_values = explainer.shap_values(input_data_int)
        
        # Asumiendo que 1 es la clase 'Alto Riesgo'
        fig = shap.force_plot(explainer.expected_value[1], shap_values[1], input_data_int, matplotlib=True, show=False)
        st.pyplot(fig, bbox_inches='tight')
        st.caption("📈 Características en rojo aumentan el riesgo; las de azul lo disminuyen.")
    except Exception as e:
        st.warning(f"No se pudo generar el gráfico SHAP: {e}")

# --- 6. FUNCIÓN DE PROCESAMIENTO DE DATOS ---
# Basada en tu notebook 'CancerGastricoModelo_v4'
def procesar_datos_para_modelo(data_dict, scaler, training_columns_after_dummies, numerical_cols_to_scale):
    """
    Convierte datos del formulario, aplica get_dummies y escalado,
    y reindexa para que coincida con el formato de entrenamiento.
    """
    input_df = pd.DataFrame([data_dict])

    # 1. Definir categorías EXACTAS del entrenamiento
    categories = {
        'gender': ['Female', 'Male'],
        'dietary_habits': ['High_Salt', 'Low_Salt'],
        'existing_conditions': ['Chronic Gastritis', 'Diabetes', 'None'],
        'endoscopic_images': ['Normal', 'Abnormal', 'No result'],
        'biopsy_results': ['Positive', 'Negative', 'No result'],
        'ct_scan': ['Positive', 'Negative', 'No result']
    }
    categorical_cols_to_encode = list(categories.keys())

    for col in categorical_cols_to_encode:
         if col in input_df.columns:
              input_df[col] = pd.Categorical(input_df[col], categories=categories[col])

    # 2. Aplicar One-Hot Encoding (drop_first=True como en el notebook)
    input_df_encoded = pd.get_dummies(input_df, drop_first=True)

    # 3. Reindexar para asegurar el orden y columnas correctas
    input_reindexed = input_df_encoded.reindex(columns=training_columns_after_dummies, fill_value=0)

    # 4. Aplicar Escalado
    if scaler is not None:
        cols_present = [col for col in numerical_cols_to_scale if col in input_reindexed.columns]
        if cols_present:
            input_reindexed[cols_present] = input_reindexed[cols_present].astype(float)
            try:
                 input_reindexed[cols_present] = scaler.transform(input_reindexed[cols_present])
            except ValueError as e:
                 st.error(f"Error al escalar datos: {e}.")
                 return None 
    return input_reindexed

# --- 7. FUNCIÓN DE MAPEO DE RIESGO (NUEVA) ---
def mapear_riesgo(prob_positive):
    """Convierte la probabilidad (0-1) a la escala de 5 riesgos."""
    if prob_positive <= 0.2:
        return "Muy bajo"
    elif prob_positive <= 0.4:
        return "Bajo"
    elif prob_positive <= 0.6:
        return "Medio"
    elif prob_positive <= 0.8:
        return "Alto"
    else:
        return "Muy alto"

# --- 8. LÓGICA PRINCIPAL DE LA APLICACIÓN ---
if authentication_status:
    # --- Si el login es exitoso, TODOS ven esto (Página 2 y 3) ---
    
    model = load_model_from_azure()

    # --- Reajustar el Scaler ---
    # (Como se discutió, esto es necesario porque el scaler no se guardó.
    # Usamos una muestra de datos representativa del entrenamiento)
    scaler = None
    try:
        sample_data_for_scaler = pd.DataFrame({
             'age': [43, 86, 68, 57, 33],
             'family_history': [1, 1, 0, 0, 0],
             'smoking_habits': [0, 0, 1, 0, 1],
             'alcohol_consumption': [0, 0, 1, 0, 1],
             'helicobacter_pylori_infection': [0, 1, 0, 1, 0]
        })
        numerical_cols_to_scale = list(sample_data_for_scaler.columns)
        scaler = StandardScaler().fit(sample_data_for_scaler)
    except Exception as e:
        st.warning(f"No se pudo reajustar el scaler: {e}. Las predicciones pueden no ser precisas.")


    # --- Columnas esperadas por el modelo DESPUÉS de get_dummies ---
    training_columns_after_dummies = [
        'age', 'family_history', 'smoking_habits', 'alcohol_consumption', 
        'helicobacter_pylori_infection', 'gender_Male', 'dietary_habits_Low_Salt', 
        'existing_conditions_Diabetes', 'existing_conditions_None', 
        'endoscopic_images_Normal', 'endoscopic_images_No result', 
        'biopsy_results_Positive', 'biopsy_results_No result', 
        'ct_scan_Positive', 'ct_scan_No result' 
    ]

    # --- Barra Lateral (Formulario de Ingreso) ---
    st.sidebar.image("oncoearly-sinfondo.png", width=150)
    st.sidebar.title(f"Bienvenido Dr. {name} 🩺")
    
    with st.sidebar.form("prediction_form"):
        st.header("Ingreso de datos clínicos 📋")
        
        age_input = st.number_input("Edad", min_value=0, max_value=120, value=50, step=1)
        gender = st.selectbox("Género", options=["Female", "Male"], index=None, placeholder="Seleccione...")
        family_history = st.selectbox("Antecedente familiar", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0], index=None, placeholder="Seleccione...")
        smoking_habits = st.selectbox("Hábito de fumar", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0], index=None, placeholder="Seleccione...")
        alcohol_consumption = st.selectbox("Consumo de alcohol", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0], index=None, placeholder="Seleccione...")
        helicobacter_pylori_infection = st.selectbox("Infección por Helicobacter pylori", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0], index=None, placeholder="Seleccione...")
        dietary_habits = st.selectbox("Hábitos alimenticios", options=["High_Salt", "Low_Salt"], index=None, placeholder="Seleccione...")
        existing_conditions = st.selectbox("Condiciones existentes", options=["Chronic Gastritis", "Diabetes", "None"], index=None, placeholder="Seleccione...")
        endoscopic_images = st.selectbox("Imágenes endoscópicas", options=["Normal", "Abnormal", "No result"], index=None, placeholder="Seleccione...")
        biopsy_results = st.selectbox("Resultados de biopsia", options=["Positive", "Negative", "No result"], index=None, placeholder="Seleccione...")
        ct_scan = st.selectbox("Tomografía computarizada", options=["Positive", "Negative", "No result"], index=None, placeholder="Seleccione...")

        submitted = st.form_submit_button("Predecir 🔍")
        
        if st.form_submit_button("Cerrar sesión 🚪", key="logout_button_sidebar"):
            if 'page' in st.session_state: del st.session_state.page
            if 'form_data' in st.session_state: del st.session_state.form_data
            authenticator.logout("Cerrar sesión", "main")
    
    # --- GESTIÓN DE PÁGINAS (Usando st.session_state) ---
    
    if 'page' not in st.session_state:
        st.session_state.page = 'input'
        
    if submitted:
        form_data_values = [gender, family_history, smoking_habits, alcohol_consumption,
                            helicobacter_pylori_infection, dietary_habits, existing_conditions,
                            endoscopic_images, biopsy_results, ct_scan]
        
        if None in form_data_values:
             st.sidebar.error("⚠️ Por favor, completa todos los campos.")
             st.session_state.page = 'input'
        else:
             st.session_state.page = 'results'
             st.session_state.form_data = {
                  'age': age_input,
                  'gender': gender,
                  'family_history': family_history[1], 
                  'smoking_habits': smoking_habits[1],
                  'alcohol_consumption': alcohol_consumption[1],
                  'helicobacter_pylori_infection': helicobacter_pylori_infection[1],
                  'dietary_habits': dietary_habits,
                  'existing_conditions': existing_conditions,
                  'endoscopic_images': endoscopic_images,
                  'biopsy_results': biopsy_results,
                  'ct_scan': ct_scan
             }
             st.experimental_rerun()


    # --- RENDERIZADO DEL ÁREA PRINCIPAL ---
    
    # PÁGINA 2: INGRESO DE DATOS (Guía de ayuda)
    if st.session_state.page == 'input':
        st.title("Dudas sobre qué significa cada opción? Consulta esta guía rápida. 💡")
        with st.expander("Ver descripciones", expanded=True):
             st.markdown(r"""
             - **Edad:** Edad del paciente al momento de la evaluación.
             - **Género:** Género biológico del paciente.
             - **Antecedente familiar:** Si existen casos de cáncer gástrico en familiares directos.
             - **Hábito de fumar:** Si el paciente fuma actualmente o lo ha hecho.
             - **Consumo de alcohol:** Frecuencia y cantidad de consumo de alcohol.
             - **Infección por Helicobacter pylori:** Resultado de prueba para H. pylori (Sí/No).
             - **Hábitos alimenticios:** Descripción general de la dieta (ej. alta en sal/grasas, balanceada).
             - **Condiciones existentes:** Presencia de otras condiciones médicas relevantes (ej. gastritis crónica, diabetes).
             - **Imágenes endoscópicas:** Hallazgos visuales de la endoscopia (Normal/Anormal/No realizado).
             - **Resultados de biopsia:** Resultado histopatológico (Positivo/Negativo/No realizado).
             - **Tomografía computarizada:** Hallazgos en la TC abdominal (Posible tumor/Sin hallazgos/No realizado).
             """)

    # PÁGINA 3: RESULTADOS
    elif st.session_state.page == 'results' and model and scaler:
        st.title("Resultados de la Predicción 📊")
        
        if 'form_data' in st.session_state:
            input_data = procesar_datos_para_modelo(st.session_state.form_data, scaler, training_columns_after_dummies, numerical_cols_to_scale)
            
            if input_data is not None:
                 try:
                      prediction_proba = model.predict_proba(input_data)[0]
                      prob_positive = prediction_proba[1] # Probabilidad de "Alto Riesgo"

                      # --- LÓGICA DE ESCALA DE RIESGO (NUEVA) ---
                      riesgo_texto = mapear_riesgo(prob_positive)

                      st.subheader("Resultado:")
                      # Mostrar solo la escala de riesgo
                      if prob_positive >= 0.6: # Umbral para Alto o Muy Alto
                           st.error(f"**Riesgo de predicción de cáncer gástrico:**\n# {riesgo_texto.upper()}")
                      else: # Medio, Bajo, Muy Bajo
                           st.success(f"**Riesgo de predicción de cáncer gástrico:**\n# {riesgo_texto.upper()}")
                      
                      # --- Como comentario: así mostrarías el porcentaje también ---
                      # if prob_positive >= 0.6:
                      #     st.error(f"**Riesgo:** {riesgo_texto.upper()} ({prob_positive:.2%})")
                      # else:
                      #     st.success(f"**Riesgo:** {riesgo_texto.upper()} ({prob_positive:.2%})")
                      
                      # --- Mostrar el gráfico SHAP ---
                      # Usamos una muestra de datos de fondo (del ajuste del scaler)
                      explainer = get_shap_explainer(model, sample_data_for_scaler)
                      plot_shap_force_plot(explainer, input_data)

                 except Exception as e:
                      st.error(f"Ocurrió un error durante la predicción: {e}")
            else:
                 st.error("Error al procesar los datos de entrada.")

            # --- CAMBIO DE BOTÓN ---
            if st.button("⬅️ Volver a predecir"):
                st.session_state.page = 'input' 
                del st.session_state.form_data 
                st.experimental_rerun()
        else:
             st.warning("No hay datos de paciente. Ingrese datos en la barra lateral.")
             if st.button("⬅️ Ir al ingreso de datos"):
                  st.session_state.page = 'input'
                  st.experimental_rerun()

# --- 9. MENSAJES DE ERROR/INFO DE LOGIN ---
elif authentication_status == False:
    st.error('❌ Usuario/contraseña incorrecto')
elif authentication_status == None:
    st.warning('Por favor, ingrese su usuario y contraseña.')
    st.caption("“Cada dato clínico es una oportunidad para anticipar el riesgo.”")