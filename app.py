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
import os # NUEVO: Para manejar archivos
from datetime import datetime # NUEVO: Para registrar la hora de la predicción

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
    [data-testid="stForm"] + div a { display: none; }
    footer {visibility: hidden;}
    [data-testid="stForm"] h1 { display: none; }
    div[data-testid="stImage"] > img {
        width: 350px;
        display: block;
        margin-left: auto;
        margin-right: auto;
    }
    [data-testid="stSidebar"] [data-testid="stFormSubmitButton"] button {
        background-color: #28a745;
        color: white;
        font-size: 30px;
        font-weight: bold;
        width: 100%;
    }
    [data-testid="stSidebar"] [data-testid="stFormSubmitButton"] button:hover {
        background-color: #218838;
    }
    [data-testid="stSidebar"] [data-testid="stButton"] button {
        background-color: #dc3545;
        color: white;
        font-size: 30px;
        font-weight: bold;
        width: 100%;
    }
    [data-testid="stSidebar"] [data-testid="stButton"] button:hover {
        background-color: #c82333;
    }
    [data-testid="stCaptionContainer"] > p {
        text-align: center;
        font-size: 1.1em;
        font-weight: 500;
    }
    [data-testid="stSidebar"] [data-testid="stImage"] > img {
        width: 200px;
    }
</style>
""", unsafe_allow_html=True)

custom_fields = {
    'Username': 'Usuario',
    'Password': 'Contraseña',
    'Login': 'Iniciar Sesión'
}

authenticator.login(fields=custom_fields) 

name = st.session_state.get('name')
authentication_status = st.session_state.get('authentication_status')
username = st.session_state.get('username')

# --- 4. FUNCIÓN DE CARGA DEL MODELO ---
@st.cache_resource
def load_model_from_azure():
    try:
        connection_string = st.secrets["azure_storage"]["connection_string"]
        container_name = "modelos-ml"
        blob_name = "rf_entrenado-v1.joblib"
        
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

# --- 5. FUNCIÓN DE EXPLICABILIDAD (LIME) ---
@st.cache_resource
def get_lime_explainer(_background_data_processed, _feature_names):
    try:
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=_background_data_processed,
            feature_names=_feature_names,
            class_names=['Bajo Riesgo', 'Alto Riesgo'],
            mode='classification',
            discretize_continuous=False,
            random_state=42
        )
        return explainer
    except Exception as e:
        st.error(f"Error al inicializar LIME Explainer: {e}")
        return None

def plot_lime_explanation(explainer, model, input_data_processed, raw_form_data, friendly_names_dict):
    st.subheader("Impacto de Factores en el Riesgo de Cáncer Gástrico:")
    if explainer is None:
        st.warning("No se puede generar LIME (Explainer no inicializado).")
        return
    
    try:
        input_data_np_1d = input_data_processed.iloc[0].values.astype(float)
        explanation = explainer.explain_instance(
            data_row=input_data_np_1d, 
            predict_fn=model.predict_proba,
            num_features=len(input_data_processed.columns),
            labels=[1]
        )
        
        exp_list = explanation.as_list(label=1) 
        consolidated_exp = {}

        for feature_string, weight in exp_list:
            root_name = feature_string
            if 'existing_conditions_' in feature_string: root_name = 'existing_conditions'
            elif 'endoscopic_images_' in feature_string: root_name = 'endoscopic_images'
            elif 'biopsy_results_' in feature_string: root_name = 'biopsy_results'
            elif 'ct_scan_' in feature_string: root_name = 'ct_scan'
            elif 'dietary_habits_' in feature_string: root_name = 'dietary_habits'
            elif 'gender_' in feature_string: root_name = 'gender'
            
            friendly_name = friendly_names_dict.get(root_name, feature_string) 
            current_weight_magnitude = consolidated_exp.get(friendly_name, 0)
            consolidated_exp[friendly_name] = current_weight_magnitude + abs(weight)

        sorted_exp = sorted(consolidated_exp.items(), key=lambda item: item[1]) 
        labels = [item[0] for item in sorted_exp]
        values = [item[1] for item in sorted_exp]

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(labels, values, color='#007bff')
        ax.set_xlabel("Magnitud del Impacto en la Predicción")
        fig.tight_layout()
        st.pyplot(fig)

    except Exception as e:
        st.error("Ocurrió un error al generar el gráfico LIME:")
        st.exception(e)

# --- 6. FUNCIÓN DE PROCESAMIENTO DE DATOS ---
def procesar_datos_para_modelo(data_dict, _scaler, training_columns_after_dummies, numerical_cols_to_scale):
    input_df = pd.DataFrame([data_dict])

    # NUEVO: Descartamos el nombre del paciente para que no confunda al modelo
    if 'patient_name' in input_df.columns:
        input_df = input_df.drop(columns=['patient_name'])

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

    input_df_encoded = pd.get_dummies(input_df, drop_first=True)
    input_reindexed = input_df_encoded.reindex(columns=training_columns_after_dummies, fill_value=0)

    if _scaler is not None:
        cols_present = [col for col in numerical_cols_to_scale if col in input_reindexed.columns]
        if cols_present:
            input_reindexed[cols_present] = input_reindexed[cols_present].astype(float)
            try:
                 input_reindexed[cols_present] = _scaler.transform(input_reindexed[cols_present])
            except ValueError as e:
                 st.error(f"Error al escalar datos: {e}.")
                 return None 
            
    try:
        input_final = input_reindexed.astype(float)
    except Exception as e:
        st.error(f"Error al convertir datos a numérico: {e}")
        return None

    return input_final

# --- 7. FUNCIÓN DE MAPEO Y GUARDADO DE RIESGO ---
def mapear_riesgo(prob_positive):
    if prob_positive <= 0.2: return "Muy bajo"
    elif prob_positive <= 0.4: return "Bajo"
    elif prob_positive <= 0.6: return "Medio"
    elif prob_positive <= 0.8: return "Alto"
    else: return "Muy alto"

# NUEVO: Función para guardar el historial
def guardar_historial(datos_paciente, riesgo, probabilidad):
    archivo_csv = "historial_predicciones.csv"
    
    # Preparar el registro
    nuevo_registro = {
        "Fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Doctor": name,
        "Paciente": datos_paciente.get("patient_name", "Desconocido"),
        "Riesgo": riesgo,
        "Probabilidad_Alto_Riesgo": f"{probabilidad:.2%}",
        "Edad": datos_paciente.get("age"),
        "Genero": datos_paciente.get("gender"),
        "H. Pylori": datos_paciente.get("helicobacter_pylori_infection"),
        "Condiciones": datos_paciente.get("existing_conditions"),
        "Biopsia": datos_paciente.get("biopsy_results")
    }
    
    df_nuevo = pd.DataFrame([nuevo_registro])
    
    if os.path.exists(archivo_csv):
        df_historial = pd.read_csv(archivo_csv)
        df_historial = pd.concat([df_historial, df_nuevo], ignore_index=True)
    else:
        df_historial = df_nuevo
        
    df_historial.to_csv(archivo_csv, index=False)


# --- 8. LÓGICA PRINCIPAL DE LA APLICACIÓN ---
if authentication_status:
    model = load_model_from_azure()

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
        st.warning(f"No se pudo reajustar el scaler: {e}.")

    training_columns_after_dummies = [
        'age', 'family_history', 'smoking_habits', 'alcohol_consumption', 
        'helicobacter_pylori_infection', 'gender_Male', 'dietary_habits_Low_Salt', 
        'existing_conditions_Diabetes', 'existing_conditions_None', 
        'endoscopic_images_Normal', 'biopsy_results_Positive', 'ct_scan_Positive'
    ]

    # --- Barra Lateral ---
    st.sidebar.image("oncoearly-sinfondo.png", width=150)
    st.sidebar.title(f"Bienvenido Dr. {name} 🩺")
    
    # NUEVO: Botón para acceder al historial de pacientes
    if st.sidebar.button("📂 Ver Historial de Pacientes"):
        st.session_state.page = 'history'
        st.rerun()

    with st.sidebar.form("prediction_form"):
        st.header("Ingresar datos clínicos:")
        
        # NUEVO: Input de nombre de paciente
        patient_name = st.text_input("Nombre del Paciente", placeholder="Ej. Juan Pérez")
        
        age_input = st.number_input("Edad", min_value=0, max_value=120, value=50, step=1)
        gender = st.selectbox("Género", options=[("Femenino", "Female"), ("Masculino", "Male")], format_func=lambda x: x[0], index=None, placeholder="Seleccione...")
        family_history = st.selectbox("Antecedente familiar", options=[("No", 0), ("Sí", 1)], format_func=lambda x: x[0], index=None, placeholder="Seleccione...")
        smoking_habits = st.selectbox("Hábito de fumar", options=[("No", 0), ("Sí", 1)], format_func=lambda x: x[0], index=None, placeholder="Seleccione...")
        alcohol_consumption = st.selectbox("Consumo de alcohol", options=[("No", 0), ("Sí", 1)], format_func=lambda x: x[0], index=None, placeholder="Seleccione...")
        helicobacter_pylori_infection = st.selectbox("Infección por Helicobacter pylori", options=[("No", 0), ("Sí", 1)], format_func=lambda x: x[0], index=None, placeholder="Seleccione...")
        dietary_habits = st.selectbox("Hábitos alimenticios", options=[("Alto en sal", "High_Salt"), ("Bajo en sal", "Low_Salt")], format_func=lambda x: x[0], index=None, placeholder="Seleccione...")
        existing_conditions = st.selectbox("Condiciones existentes", options=[("Gastritis Crónica", "Chronic Gastritis"), ("Diabetes", "Diabetes"), ("Ninguna", "None")], format_func=lambda x: x[0], index=None, placeholder="Seleccione...")
        endoscopic_images = st.selectbox("Imágenes endoscópicas", options=[("Normal", "Normal"), ("Anormal", "Abnormal"), ("Sin resultado", "No result")], format_func=lambda x: x[0], index=None, placeholder="Seleccione...")
        biopsy_results = st.selectbox("Resultados de biopsia", options=[("Positivo", "Positive"), ("Negativo", "Negative"), ("Sin resultado", "No result")], format_func=lambda x: x[0], index=None, placeholder="Seleccione...")
        ct_scan = st.selectbox("Tomografía computarizada", options=[("Positivo", "Positive"), ("Negativo", "Negative"), ("Sin resultado", "No result")], format_func=lambda x: x[0], index=None, placeholder="Seleccione...")

        submitted = st.form_submit_button("Predecir 🔍")
        
    # NUEVO: Lógica de restablecer contraseña (Expander en la barra lateral)
    with st.sidebar.expander("⚙️ Opciones de Cuenta"):
        try:
            if authenticator.reset_password(username, 'Cambiar mi contraseña'):
                st.success('Contraseña modificada correctamente')
                with open('config.yaml', 'w') as file:
                    yaml.dump(config, file, default_flow_style=False)
        except Exception as e:
            st.error(e)

    authenticator.logout("Cerrar sesión 🚪", location='sidebar')
    
    # --- GESTIÓN DE PÁGINAS ---
    if 'page' not in st.session_state:
        st.session_state.page = 'input'
        
    if submitted:
        # Se agrega patient_name a la validación
        form_data_values = [patient_name, gender, family_history, smoking_habits, alcohol_consumption,
                            helicobacter_pylori_infection, dietary_habits, existing_conditions,
                            endoscopic_images, biopsy_results, ct_scan]
        
        # Validamos que no esté vacío el string y no haya Nones
        if None in form_data_values or "" in form_data_values:
             st.sidebar.error("⚠️ Por favor, completa todos los campos (incluyendo el nombre del paciente).")
             st.session_state.page = 'input'
        else:
             st.session_state.page = 'results'
             st.session_state.prediction_saved = False # Flag para evitar guardar doble en base de datos
             st.session_state.form_data = {
                  'patient_name': patient_name, # <-- NUEVO: guardamos el nombre
                  'age': age_input,
                  'gender': gender[1],
                  'family_history': family_history[1],
                  'smoking_habits': smoking_habits[1],
                  'alcohol_consumption': alcohol_consumption[1],
                  'helicobacter_pylori_infection': helicobacter_pylori_infection[1],
                  'dietary_habits': dietary_habits[1],
                  'existing_conditions': existing_conditions[1],
                  'endoscopic_images': endoscopic_images[1],
                  'biopsy_results': biopsy_results[1],
                  'ct_scan': ct_scan[1]
             }
             st.rerun()

    # --- PÁGINA 2: INGRESO DE DATOS ---
    if st.session_state.page == 'input':
        st.title("Guía Rápida de Opciones 💡") 
        st.info("**Importante:** Para realizar la predicción, es necesario completar todos los campos del formulario.")

        with st.expander("Ver descripciones de cada campo", expanded=True):
             st.markdown(r"""
             - **Nombre del paciente:** Identificador para guardar en el historial.
             - **Edad:** Edad del paciente al momento de la evaluación (ej. `50`).
             - **Género:** Género biológico del paciente.
             - **Antecedente familiar:** `Sí` si existen casos de cáncer gástrico en familiares directos.
             - **Hábito de fumar:** `Sí` si el paciente fuma actualmente o lo ha hecho.
             - **Consumo de alcohol:** `Sí` si el paciente tiene un historial de consumo de alcohol.
             - **Infección por Helicobacter pylori:** `Sí` si la prueba para H. pylori fue positiva.
             - **Hábitos alimenticios:** Dieta predominante del paciente.
             - **Condiciones existentes:** Presencia de otras condiciones médicas relevantes.
             - **Imágenes endoscópicas:** Hallazgos visuales de la endoscopia.
             - **Resultados de biopsia:** Resultado histopatológico de la muestra.
             - **Tomografía computarizada:** Hallazgos en la TC abdominal.
             """)

    # --- PÁGINA 3: RESULTADOS ---
    elif st.session_state.page == 'results' and model and scaler:
        st.title(f"Resultado para: {st.session_state.form_data.get('patient_name')}")
        st.subheader("Riesgo de Cáncer Gástrico:")
        
        if 'form_data' in st.session_state:
            input_data = procesar_datos_para_modelo(st.session_state.form_data, scaler, training_columns_after_dummies, numerical_cols_to_scale)
            
            if input_data is not None:
                 try:
                      prediction = model.predict(input_data)[0]
                      prediction_proba = model.predict_proba(input_data)[0]
                      prob_positive = prediction_proba[1] 

                      riesgo_texto = mapear_riesgo(prob_positive)

                      if prob_positive >= 0.6: 
                           st.error(f"# {riesgo_texto.upper()} ({prob_positive:.2%})")
                      else: 
                           st.success(f"# {riesgo_texto.upper()} ({prob_positive:.2%})")

                      # NUEVO: Guardar en el historial automáticamente la primera vez que vemos el resultado
                      if not st.session_state.get('prediction_saved', True):
                          guardar_historial(st.session_state.form_data, riesgo_texto, prob_positive)
                          st.session_state.prediction_saved = True # Asegura que no guarde múltiples veces si el usuario interactúa con la página
                      
                      @st.cache_resource
                      def create_explainer_background(_scaler):
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
                          if processed_list: return pd.concat(processed_list).values
                          else: return None
                      
                      background_data_np = create_explainer_background(scaler)
                      
                      if background_data_np is not None:
                          friendly_names_dict = {
                              'gender': 'Género', 'dietary_habits': 'Dieta', 'existing_conditions': 'Condición',
                              'endoscopic_images': 'Im. Endoscópicas', 'biopsy_results': 'Biopsia', 'ct_scan': 'Tomografía',
                              'age': 'Edad', 'family_history': 'Antecedente Familiar', 'smoking_habits': 'Hábito de Fumar',
                              'alcohol_consumption': 'Consumo de Alcohol', 'helicobacter_pylori_infection': 'Infección H. Pylori',
                              'gender_Male': 'Género: Masculino', 'dietary_habits_Low_Salt': 'Dieta: Baja en Sal',
                              'existing_conditions_Diabetes': 'Condición: Diabetes', 'existing_conditions_None': 'Condición: Ninguna', 
                              'endoscopic_images_Normal': 'Endoscopía: Normal', 'biopsy_results_Positive': 'Biopsia: Positiva',
                              'ct_scan_Positive': 'Tomografía: Positiva'
                          }
                          
                          lime_explainer = get_lime_explainer(background_data_np, training_columns_after_dummies)
                          
                          if lime_explainer:
                              plot_lime_explanation(lime_explainer, model, input_data, st.session_state.form_data, friendly_names_dict)
                          else:
                              st.warning("No se pudo inicializar el Explainer de LIME.")
                      else:
                          st.warning("No se pudo generar la explicación LIME (sin datos de fondo).")

                 except Exception as e:
                      st.error(f"Ocurrió un error durante la predicción: {e}")
            else:
                 st.error("Error al procesar los datos de entrada.")

            if st.button("⬅️ Volver a predecir"):
                st.session_state.page = 'input' 
                del st.session_state.form_data 
                st.rerun()
        else:
             st.warning("No hay datos de paciente. Ingrese datos en la barra lateral.")
             if st.button("⬅️ Ir al ingreso de datos"):
                  st.session_state.page = 'input'
                  st.rerun()

    # --- NUEVO: PÁGINA 4: HISTORIAL DE PREDICCIONES ---
    elif st.session_state.page == 'history':
        st.title("📂 Historial de Predicciones")
        
        if os.path.exists("historial_predicciones.csv"):
            df_historial = pd.read_csv("historial_predicciones.csv")
            
            # Mostramos la tabla en la app
            st.dataframe(df_historial, use_container_width=True)
            
            # Opcional: Botón para descargar el CSV
            csv_data = df_historial.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Descargar Historial completo en CSV",
                data=csv_data,
                file_name="historial_pacientes_oncoearly.csv",
                mime="text/csv"
            )
        else:
            st.info("Aún no hay predicciones guardadas. Realiza tu primera predicción para empezar a registrar el historial.")
            
        if st.button("⬅️ Volver a Inicio"):
            st.session_state.page = 'input'
            st.rerun()

# --- 9. MENSAJES DE ERROR/INFO DE LOGIN ---
elif authentication_status == False:
    st.error('❌ Usuario/contraseña incorrecto')
    st.caption("“Cada dato clínico es una oportunidad para anticipar el riesgo.”")

elif authentication_status == None:
    if 'authentication_status' not in st.session_state:
        st.caption("“Cada dato clínico es una oportunidad para anticipar el riesgo.”")
    else:
        st.warning('⚠️ Por favor, ingrese su usuario y contraseña.')
        st.caption("“Cada dato clínico es una oportunidad para anticipar el riesgo.”")