import streamlit as st
import streamlit_authenticator as stauth
import pandas as pd
import joblib
import yaml
from yaml.loader import SafeLoader
from azure.storage.blob import BlobServiceClient
import io  # Para manejar archivos en memoria
import matplotlib.pyplot as plt # Necesario para el gráfico SHAP
import lime
import lime.lime_tabular
from sklearn.preprocessing import StandardScaler # Para escalar
import numpy as np

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

# CSS para ocultar "Registrate", el footer Y aplicar estilos a la página de inicio
st.markdown("""
<style>
    /* Oculta "Registrate" y el footer (código existente) */
    [data-testid="stForm"] + div a { display: none; }
    footer {visibility: hidden;}

    /* Oculta el título 'Login' de dentro del formulario */
    [data-testid="stForm"] h1 { display: none; }

    /* --- CAMBIO 1: Centrar y agrandar el logo --- */
    /* Selecciona la imagen (img) dentro de su contenedor (stImage) */
    div[data-testid="stImage"] > img {
        width: 350px;  /* "un poco más grande" que los 300px originales */
        display: block;
        margin-left: auto;
        margin-right: auto;
    }

    /* Botón Predecir (Verde) */
    [data-testid="stSidebar"] [data-testid="stFormSubmitButton"] button {
        background-color: #28a745; /* Verde */
        color: white;
        font-size: 30px; /* Letra más grande */
        font-weight: bold;
        width: 100%; /* Ocupa todo el ancho */
    }
    [data-testid="stSidebar"] [data-testid="stFormSubmitButton"] button:hover {
        background-color: #218838; /* Verde más oscuro al pasar el mouse */
    }

    /* Botón Cerrar Sesión (Rojo) */
    /* Target el botón normal (stButton) en la barra lateral */
    [data-testid="stSidebar"] [data-testid="stButton"] button {
        background-color: #dc3545; /* Rojo */
        color: white;
        font-size: 30px; /* Letra más grande */
        font-weight: bold;
        width: 100%; /* Ocupa todo el ancho */
    }
    [data-testid="stSidebar"] [data-testid="stButton"] button:hover {
        background-color: #c82333; /* Rojo más oscuro al pasar el mouse */
    }

    /* --- CAMBIO 2: Centrar y agrandar el slogan --- */
    /* Selecciona el texto (p) dentro del contenedor del caption (stCaptionContainer) */
    [data-testid="stCaptionContainer"] > p {
        text-align: center;
        font-size: 1.1em; /* "un poco más grande" (10% más) */
        font-weight: 500; /* Opcional: un poco más grueso */
    }

    /* Agrandar el logo en la BARRA LATERAL (Sidebar) */
    [data-testid="stSidebar"] [data-testid="stImage"] > img {
        width: 200px; /* Ajusta este valor (ej. 200px) como quieras */
    }
</style>
""", unsafe_allow_html=True)

# Define los nuevos nombres para los campos del formulario
custom_fields = {
    'Username': 'Usuario',
    'Password': 'Contraseña',
    'Login': 'Iniciar Sesión'
}

# Pasa el diccionario a la función login()
authenticator.login(fields=custom_fields) # Esta línea dibuja el formulario de "Login"

# Ahora, obtenemos los valores de forma segura desde st.session_state
name = st.session_state.get('name')
authentication_status = st.session_state.get('authentication_status')
username = st.session_state.get('username')


# --- 4. FUNCIÓN DE CARGA DEL MODELO (Desde Azure) ---
@st.cache_resource # Cachea el modelo descargado
def load_model_from_azure():
    """
    Se conecta a Azure Blob Storage, descarga el modelo y lo carga en memoria.
    """
    try:
        connection_string = st.secrets["azure_storage"]["connection_string"]
        container_name = "modelos-ml"
        blob_name = "rf_entrenado-v1.joblib" # ¡Nombre de tu modelo!
        
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
        
        with st.spinner("Descargando y cargando modelo de IA... 🧠"):
            downloader = blob_client.download_blob()
            blob_bytes = downloader.readall()
            model = joblib.load(io.BytesIO(blob_bytes))
        
        st.toast("Modelo cargado exitosamente. ✅")
        return model
    except Exception as e:
        st.error(f"Error crítico al cargar el modelo desde Azure: {e}")
        return None

# --- 5. FUNCIÓN DE EXPLICABILIDAD (LIME) ---
@st.cache_resource
def get_lime_explainer(_background_data_processed, _feature_names):
    """
    Crea el objeto explicador de LIME usando datos de fondo procesados.
    _background_data_processed debe ser un array NumPy.
    """
    st.toast("Inicializando explicador LIME...")
    try:
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=_background_data_processed, # LIME necesita un array NumPy
            feature_names=_feature_names,
            class_names=['Bajo Riesgo', 'Alto Riesgo'], # Nombres de tus clases
            mode='classification',
            discretize_continuous=False,
            random_state=42
        )
        st.toast("Explicador LIME listo.")
        return explainer
    except Exception as e:
        st.error(f"Error al inicializar LIME Explainer: {e}")
        st.exception(e)
        return None

def plot_lime_explanation(explainer, model, input_data_processed, raw_form_data, friendly_names_dict):
    """
    Genera y muestra una explicación LIME legible, consolidada y limpia.
    """
    st.subheader("Factores clave para ESTE paciente (LIME):")
    if explainer is None:
        st.warning("No se puede generar LIME (Explainer no inicializado).")
        return
    
    try:
        # 1. Obtener la explicación de LIME
        input_data_np_1d = input_data_processed.iloc[0].values.astype(float)
        
        explanation = explainer.explain_instance(
            data_row=input_data_np_1d, 
            predict_fn=model.predict_proba,
            num_features=len(input_data_processed.columns),
            labels=[1] # Enfócate solo en la clase "Alto Riesgo"
        )
        
        exp_list = explanation.as_list(label=1) 
        
        # --- INICIO LÓGICA DE CONSOLIDACIÓN Y LIMPIEZA ---
        
        consolidated_exp = {} # Diccionario para sumar pesos (consolida 'Condición' en una sola barra)

        for feature_string, weight in exp_list:
            # 1. Identificar la "raíz" de la variable interna
            root_name = feature_string
            if 'existing_conditions_' in feature_string:
                root_name = 'existing_conditions'
            elif 'endoscopic_images_' in feature_string:
                root_name = 'endoscopic_images'
            elif 'biopsy_results_' in feature_string:
                root_name = 'biopsy_results'
            elif 'ct_scan_' in feature_string:
                root_name = 'ct_scan'
            elif 'dietary_habits_' in feature_string:
                root_name = 'dietary_habits'
            elif 'gender_' in feature_string:
                root_name = 'gender'
            
            # 2. Traducir la raíz (usa el nombre amigable: "Biopsia", "Condición", etc.)
            friendly_name = friendly_names_dict.get(root_name, feature_string)

            # 3. Sumar los pesos
            current_weight = consolidated_exp.get(friendly_name, 0)
            consolidated_exp[friendly_name] = current_weight + weight

        # 4. Preparar datos para el gráfico final
        sorted_exp = sorted(consolidated_exp.items(), key=lambda item: item[1])

        labels = [item[0] for item in sorted_exp]
        values = [item[1] for item in sorted_exp]
        # --- FIN LÓGICA DE CONSOLIDACIÓN ---

        # 5. Crear el gráfico de barras horizontal
        fig, ax = plt.subplots(figsize=(8, 6)) # Un poco más grande para mejor visualización
        colors = ['#dc3545' if v > 0 else '#28a745' for v in values]
        ax.barh(labels, values, color=colors)
        
        # Centrar el eje x en cero
        max_abs = max(abs(min(values)), abs(max(values)))
        ax.set_xlim(-max_abs * 1.1, max_abs * 1.1)

        ax.set_title("Impacto de cada factor en la predicción")
        ax.set_xlabel("Impacto (Rojo = Sube Riesgo, Verde = Baja Riesgo)")
        fig.tight_layout()
        st.pyplot(fig)
        st.caption("Gráfico LIME: Muestra la contribución neta de cada variable a la predicción de 'Alto Riesgo'.")

    except Exception as e:
        st.error("Ocurrió un error al generar el gráfico LIME:")
        st.exception(e)

# --- 6. FUNCIÓN DE PROCESAMIENTO DE DATOS ---
# Basada en tu notebook 'CancerGastricoModelo_v4'
def procesar_datos_para_modelo(data_dict, _scaler, training_columns_after_dummies, numerical_cols_to_scale):
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
            
    # Forzar TODO el DataFrame final a tipo numérico (float)
    try:
        input_final = input_reindexed.astype(float)
    except Exception as e:
        st.error(f"Error al convertir datos a numérico: {e}")
        return None

    return input_final # <-- Devuelve input_final, no input_reindexed

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
    numerical_cols_to_scale = ['age', 'family_history', 'smoking_habits', 'alcohol_consumption', 'helicobacter_pylori_infection']
    try:
        sample_data_for_scaler = pd.DataFrame({
             'age': [43, 86, 68, 57, 33],
             'family_history': [1, 1, 0, 0, 0],
             'smoking_habits': [0, 0, 1, 0, 1],
             'alcohol_consumption': [0, 0, 1, 0, 1],
             'helicobacter_pylori_infection': [0, 1, 0, 1, 0]
        })
        scaler = StandardScaler().fit(sample_data_for_scaler)
    except Exception as e:
        st.warning(f"No se pudo reajustar el scaler: {e}. Las predicciones pueden no ser precisas.")


# --- Columnas esperadas por el modelo DESPUÉS de get_dummies (VERSIÓN DE 12 COLUMNAS) ---
    training_columns_after_dummies = [
        'age', 'family_history', 'smoking_habits', 'alcohol_consumption', 
        'helicobacter_pylori_infection', 
        'gender_Male', 
        'dietary_habits_Low_Salt', 
        'existing_conditions_Diabetes', 
        'existing_conditions_None', 
        'endoscopic_images_Normal', 
        'biopsy_results_Positive', 
        'ct_scan_Positive'
    ]

    # --- Barra Lateral (Formulario de Ingreso) ---
    st.sidebar.image("oncoearly-sinfondo.png", width=150)
    st.sidebar.title(f"Bienvenido Dr. {username} 🩺")
    
    with st.sidebar.form("prediction_form"):
        st.header("Ingreso de datos clínicos 📋")
        
        age_input = st.number_input("Edad",
                                    min_value=0,
                                    max_value=120,
                                    value=50,
                                    step=1)

        gender = st.selectbox("Género", 
                              options=[("Femenino", "Female"), ("Masculino", "Male")], 
                              format_func=lambda x: x[0], 
                              index=None, placeholder="Seleccione...")
        
        family_history = st.selectbox("Antecedente familiar", 
                                      options=[("No", 0), ("Sí", 1)], 
                                      format_func=lambda x: x[0], 
                                      index=None, placeholder="Seleccione...")
        
        smoking_habits = st.selectbox("Hábito de fumar", 
                                      options=[("No", 0), ("Sí", 1)], 
                                      format_func=lambda x: x[0], 
                                      index=None, placeholder="Seleccione...")
        
        alcohol_consumption = st.selectbox("Consumo de alcohol", 
                                           options=[("No", 0), ("Sí", 1)], 
                                           format_func=lambda x: x[0], 
                                           index=None, placeholder="Seleccione...")
        
        helicobacter_pylori_infection = st.selectbox("Infección por Helicobacter pylori", 
                                                     options=[("No", 0), ("Sí", 1)], 
                                                     format_func=lambda x: x[0], 
                                                     index=None, placeholder="Seleccione...")
        
        dietary_habits = st.selectbox("Hábitos alimenticios", 
                                      options=[("Alto en sal", "High_Salt"), ("Bajo en sal", "Low_Salt")], 
                                      format_func=lambda x: x[0], 
                                      index=None, placeholder="Seleccione...")
        
        existing_conditions = st.selectbox("Condiciones existentes", 
                                           options=[("Gastritis Crónica", "Chronic Gastritis"), ("Diabetes", "Diabetes"), ("Ninguna", "None")], 
                                           format_func=lambda x: x[0], 
                                           index=None, placeholder="Seleccione...")
        
        endoscopic_images = st.selectbox("Imágenes endoscópicas", 
                                         options=[("Normal", "Normal"), ("Anormal", "Abnormal"), ("Sin resultado", "No result")], 
                                         format_func=lambda x: x[0], 
                                         index=None, placeholder="Seleccione...")
        
        biopsy_results = st.selectbox("Resultados de biopsia", 
                                      options=[("Positivo", "Positive"), ("Negativo", "Negative"), ("Sin resultado", "No result")], 
                                      format_func=lambda x: x[0], 
                                      index=None, placeholder="Seleccione...")
        
        ct_scan = st.selectbox("Tomografía computarizada", 
                               options=[("Positivo", "Positive"), ("Negativo", "Negative"), ("Sin resultado", "No result")], 
                               format_func=lambda x: x[0], 
                               index=None, placeholder="Seleccione...")

        submitted = st.form_submit_button("Predecir 🔍")
        
    authenticator.logout("Cerrar sesión 🚪", location='sidebar')
    
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
                  'gender': gender[1], # <-- CAMBIO: [1] para obtener "Female" o "Male"
                  'family_history': family_history[1],
                  'smoking_habits': smoking_habits[1],
                  'alcohol_consumption': alcohol_consumption[1],
                  'helicobacter_pylori_infection': helicobacter_pylori_infection[1],
                  'dietary_habits': dietary_habits[1], # <-- CAMBIO: [1] para "High_Salt" o "Low_Salt"
                  'existing_conditions': existing_conditions[1], # <-- CAMBIO: [1]
                  'endoscopic_images': endoscopic_images[1], # <-- CAMBIO: [1]
                  'biopsy_results': biopsy_results[1], # <-- CAMBIO: [1]
                  'ct_scan': ct_scan[1] # <-- CAMBIO: [1]
             }
             st.rerun()


    # --- RENDERIZADO DEL ÁREA PRINCIPAL ---
    
    # PÁGINA 2: INGRESO DE DATOS (Guía de ayuda)
    if st.session_state.page == 'input':
        st.title("Dudas? Consulta esta guía💡")
        with st.expander("Ver descripciones", expanded=True):
             st.markdown(r"""
             - **Edad:** Edad del paciente al momento de la evaluación.
             - **Género:** Género biológico del paciente.
             - **Antecedente familiar:** Si existen casos de cáncer gástrico en familiares directos.
             - **Hábito de fumar:** Si el paciente fuma actualmente o lo ha hecho.
             - **Consumo de alcohol:** Alta frecuencia y cantidad de consumo de alcohol.
             - **Infección por Helicobacter pylori:** Resultado de prueba para H. pylori (Sí/No).
             - **Hábitos alimenticios:** Descripción general de la dieta (alta en sal, bajo en sal).
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
                      prediction = model.predict(input_data)[0]
                      prediction_proba = model.predict_proba(input_data)[0]
                      prob_positive = prediction_proba[1] # Probabilidad de "Alto Riesgo"

                      # --- LÓGICA DE ESCALA DE RIESGO (NUEVA) ---
                      riesgo_texto = mapear_riesgo(prob_positive)

                      st.subheader("Resultado:")
                      # --- CÓDIGO ACTUALIZADO: MUESTRA ESCALA Y PORCENTAJE ---
                      if prob_positive >= 0.6: # Umbral para Alto o Muy Alto
                           st.error(f"**Riesgo de predicción de cáncer gástrico:**\n# {riesgo_texto.upper()} ({prob_positive:.2%})")
                      else: # Medio, Bajo, Muy Bajo
                           st.success(f"**Riesgo de predicción de cáncer gástrico:**\n# {riesgo_texto.upper()} ({prob_positive:.2%})")

                    # --- INICIO: SECCIÓN LIME (Traducida y Consolidada) ---
                      
                      # 1. Crear los datos de fondo (igual que antes)
                      @st.cache_resource
                      def create_explainer_background(_scaler):
                          # ... (código de create_explainer_background sin cambios) ...
                          # ... (asegúrate que devuelva pd.concat(processed_list).values) ...
                          background_data_raw = {
                              'age': [30, 50, 70], 'gender': ['Male', 'Female', 'Male'],
                              'family_history': [0, 1, 0], 'smoking_habits': [1, 0, 1],
                              'alcohol_consumption': [0, 1, 0], 'helicobacter_pylori_infection': [1, 0, 0],
                              'dietary_habits': ['High_Salt', 'Low_Salt', 'High_Salt'],
                              'existing_conditions': ['None', 'Diabetes', 'Chronic Gastritis'],
                              'endoscopic_images': ['Normal', 'Abnormal', 'No result'],
                              'biopsy_results': ['Negative', 'Positive', 'No result'],
                              'ct_scan': ['Negative', 'Positive', 'No result']
                          }
                          background_df = pd.DataFrame(background_data_raw)
                          processed_list = []
                          for i in range(len(background_df)):
                              processed_row = procesar_datos_para_modelo(
                                  background_df.iloc[i].to_dict(), _scaler, 
                                  training_columns_after_dummies, numerical_cols_to_scale
                              )
                              if processed_row is not None:
                                  processed_list.append(processed_row)
                          if processed_list:
                              return pd.concat(processed_list).values
                          else:
                              return None
                      
                      background_data_np = create_explainer_background(scaler)
                      
                      if background_data_np is not None:
                          
                          # --- ¡CAMBIO AQUÍ! ---
                          # 2. Crear DICCIONARIO DE TRADUCCIÓN
                          # (Mapea los 12 nombres internos a nombres amigables)
                          friendly_names_dict = {
                              # Raíces CONSOLIDADAS
                              'age': 'Edad',
                              'family_history': 'Antecedente Familiar',
                              'smoking_habits': 'Hábito de Fumar',
                              'alcohol_consumption': 'Consumo de Alcohol', 
                              'helicobacter_pylori_infection': 'Infección H. Pylori',
                              'gender': 'Género', # <-- Raíz
                              'dietary_habits': 'Dieta', # <-- Raíz
                              'existing_conditions': 'Condición', # <-- Raíz
                              'endoscopic_images': 'Im. Endoscópicas', # <-- Raíz
                              'biopsy_results': 'Biopsia', # <-- Raíz
                              'ct_scan': 'Tomografía', # <-- Raíz
                              
                              # Las Dummies NO CONSOLIDADAS también deben estar por si acaso
                              'gender_Male': 'Género: Masculino',
                              'dietary_habits_Low_Salt': 'Dieta: Baja en Sal',
                              'existing_conditions_Diabetes': 'Condición: Diabetes', 
                              'existing_conditions_None': 'Condición: Ninguna', 
                              'endoscopic_images_Normal': 'Endoscopía: Normal',
                              'biopsy_results_Positive': 'Biopsia: Positiva',
                              'ct_scan_Positive': 'Tomografía: Positiva'
                          }
                          
                          # 3. Crear el explainer (pasando los 12 nombres internos)
                          lime_explainer = get_lime_explainer(
                              background_data_np, 
                              training_columns_after_dummies 
                          )
                          
                          # 4. Llamar a la función de ploteo
                          if lime_explainer:
                              plot_lime_explanation(
                                  lime_explainer, 
                                  model, 
                                  input_data, # Los datos PROCESADOS
                                  st.session_state.form_data, # Los datos RAW (para traducir)
                                  friendly_names_dict # <--- ¡AÑADIDO!
                              )
                          else:
                              st.warning("No se pudo inicializar el Explainer de LIME.")
                      else:
                          st.warning("No se pudo generar la explicación LIME (sin datos de fondo).")
                      # --- FIN SECCIÓN LIME ---

                 except Exception as e:
                      st.error(f"Ocurrió un error durante la predicción: {e}")
            else:
                 st.error("Error al procesar los datos de entrada.")

            # --- CAMBIO DE BOTÓN ---
            if st.button("⬅️ Volver a predecir"):
                st.session_state.page = 'input' 
                del st.session_state.form_data 
                st.rerun()
        else:
             st.warning("No hay datos de paciente. Ingrese datos en la barra lateral.")
             if st.button("⬅️ Ir al ingreso de datos"):
                  st.session_state.page = 'input'
                  st.rerun()

# --- 9. MENSAJES DE ERROR/INFO DE LOGIN ---
elif authentication_status == False:
    # --- Caso 1: El login falló (contraseña incorrecta) ---
    st.error('❌ Usuario/contraseña incorrecto')
    st.caption("“Cada dato clínico es una oportunidad para anticipar el riesgo.”")

elif authentication_status == None:
    # --- Caso 2: El login está pendiente ---
    # (Puede ser la carga inicial O un envío vacío)

    if 'authentication_status' not in st.session_state:
        # --- Carga Inicial ---
        # (La variable 'authentication_status' aún no existe en la sesión)
        # Solo mostramos el caption, sin advertencia.
        st.caption("“Cada dato clínico es una oportunidad para anticipar el riesgo.”")
    else:
        # --- Envío Vacío ---
        # (El usuario hizo clic en Login con campos vacíos)
        # Ahora sí mostramos la advertencia.
        st.warning('⚠️ Por favor, ingrese su usuario y contraseña.')
        st.caption("“Cada dato clínico es una oportunidad para anticipar el riesgo.”")