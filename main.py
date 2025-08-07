import streamlit as st
import pandas as pd
from docxtpl import DocxTemplate
import google.generativeai as genai
import openai
import os
import re
import time
import zipfile
from io import BytesIO

# --- CONFIGURACIÓN DE LA PÁGINA DE STREAMLIT ---
st.set_page_config(
    page_title="Ensamblador de Fichas con Auditoría IA",
    page_icon="🤖✅",
    layout="wide"
)

# --- FUNCIONES DE LÓGICA ---

def limpiar_html(texto_html):
    """Limpia etiquetas HTML de un texto."""
    if not isinstance(texto_html, str):
        return texto_html
    cleanr = re.compile('<.*?>')
    texto_limpio = re.sub(cleanr, '', texto_html)
    return texto_limpio

# --- Funciones de Configuración de Modelos ---
# Función para configurar el modelo Gemini
def setup_gemini_model(api_key):
    try:
        genai.configure(api_key=api_key)
        generation_config = {
            "temperature": 0.6, "top_p": 1, "top_k": 1, "max_output_tokens": 8192
        }
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]
        model = genai.GenerativeModel(
            model_name="gemini-1.5-pro-latest",
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        return model
    except Exception as e:
        st.error(f"Error al configurar la API de Google: {e}")
        return None


def setup_openai_model(api_key):
    """Configura y retorna un cliente para los modelos de OpenAI."""
    try:
        client = openai.OpenAI(api_key=api_key)
        client.models.list()  # Verifica que la clave funciona
        return client
    except openai.AuthenticationError:
        st.error("Clave API de OpenAI inválida o incorrecta.")
        return None
    except Exception as e:
        st.error(f"Error al configurar la API de OpenAI: {e}")
        return None

# --- Función Unificada de Generación ---
def generar_contenido(modelo_cliente, nombre_modelo_api, prompt):
    """Genera contenido usando el cliente y nombre de modelo apropiado."""
    try:
        if isinstance(modelo_cliente, genai.GenerativeModel):
            response = modelo_cliente.generate_content(prompt)
            # Asegurarse de que la respuesta tenga texto antes de devolverla
            return response.text.strip() if hasattr(response, 'text') and response.text else ""
        elif isinstance(modelo_cliente, openai.OpenAI):
            chat_completion = modelo_cliente.chat.completions.create(
                messages=[
                    {"role": "system", "content": "Eres un asistente experto en pedagogía y evaluación educativa."},
                    {"role": "user", "content": prompt},
                ],
                model=nombre_modelo_api,
                temperature=0.6,
            )
            return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        st.warning(f"Error durante la generación de contenido: {e}")
        time.sleep(3) # Pausa por si es un error de rate limit
        return f"ERROR API: {str(e)}"

# --- Funciones de Construcción de Prompts ---

def construir_prompt_analisis(fila):
    """Construye el prompt para el análisis del ítem."""
    fila = fila.fillna('')
    descripcion_item = (
        f"Enunciado: {fila.get('Enunciado', '')}\n"
        f"A. {fila.get('OpcionA', '')}\n"
        f"B. {fila.get('OpcionB', '')}\n"
        f"C. {fila.get('OpcionC', '')}\n"
        f"D. {fila.get('OpcionD', '')}\n"
        f"Respuesta correcta: {fila.get('AlternativaClave', '')}"
    )
    return f"""
🎯 ROL DEL SISTEMA
Eres un experto psicómetra y pedagogo, especializado en la deconstrucción cognitiva de ítems de evaluación de lectura. Tu misión es realizar un análisis tripartito y riguroso de un ítem, explicando qué evalúa, cómo se resuelve correctamente y por qué las opciones incorrectas (distractores) son atractivas para un estudiante que comete un error específico de razonamiento.

🧠 INSUMOS DE ENTRADA
- Texto/Fragmento: {fila.get('ItemContexto', 'No aplica')}
- Descripción del Ítem: {descripcion_item}
- Componente: {fila.get('ComponenteNombre', 'No aplica')}
- Competencia: {fila.get('CompetenciaNombre', '')}
- Aprendizaje Priorizado: {fila.get('AfirmacionNombre', '')}
- Evidencia de Aprendizaje: {fila.get('EvidenciaNombre', '')}
- Grado Escolar: {fila.get('ItemGradoId', '')}
- Respuesta correcta: {fila.get('AlternativaClave', 'No aplica')}
- Opción A: {fila.get('OpcionA', 'No aplica')}
- Opción B: {fila.get('OpcionB', 'No aplica')}
- Opción C: {fila.get('OpcionC', 'No aplica')}
- Opción D: {fila.get('OpcionD', 'No aplica')}

📝 INSTRUCCIONES PARA EL ANÁLISIS DEL ÍTEM
Genera el análisis del ítem siguiendo estas reglas y en el orden exacto solicitado:

### 1. Qué Evalúa
**Regla de Oro:** La descripción debe ser una síntesis directa y precisa de la taxonomía del ítem teniendo en cuenta lo que el ítem evalúa.
- Redacta una única frase (máximo 2 renglones) que comience obligatoriamente con "Este ítem evalúa la capacidad del estudiante para...".
- La frase debe construirse usando la **Evidencia de Aprendizaje** como núcleo de la habilidad y la **Competencia** como el marco general.
- **Prohibido** referirse al contenido o a los personajes del texto. El foco es 100% en el proceso cognitivo definido por la taxonomía.

### 2. Ruta Cognitiva Correcta
- Describe, en un párrafo continuo y de forma impersonal, el procedimiento mental que un estudiante debe ejecutar para llegar a la respuesta correcta.
- Debes articular la ruta usando **verbos que representen procesos cognitivos** (ej: identificar, relacionar, inferir, comparar, evaluar) para mostrar la secuencia lógica de pensamiento de manera explícita.
- El último paso de la ruta debe ser la justificación final de por qué la alternativa clave es la única respuesta válida, conectando el razonamiento con la selección de esa opción.
- Los verbos de los procesos cognitivos deben tener relación con la compentecia, evidencia de aprendizaje, y aprendizaje priorizado.

### 3. Análisis de Opciones No Válidas (Distractores)
Para cada una de las TRES opciones incorrectas, realiza un análisis del error.
- Primero, identifica la **naturaleza del error** (ej: es una lectura literal cuando se pide inferir, una sobregeneralización, una interpretación de un detalle irrelevante pero llamativo, una opinión personal no sustentada en el texto, etc.).
- Luego, explica el posible razonamiento que lleva al estudiante a cometer ese error.
- Finalmente, clarifica por qué esa opción es incorrecta en el contexto de la tarea evaluativa.

✍️ FORMATO DE SALIDA DEL ANÁLISIS
**REGLA CRÍTICA:** Responde únicamente con el texto solicitado y en la estructura definida a continuación. Es crucial que los tres títulos aparezcan en la respuesta, en el orden correcto y sin texto introductorio, de cierre o conclusiones.

Qué Evalúa:
Este ítem evalúa la capacidad del estudiante para [síntesis de la taxonomía, centrada en la Evidencia de Aprendizaje, aprendizaje priorizado y lo que hace para responder el ítem].

Ruta Cognitiva Correcta:
Para resolver correctamente este ítem, el estudiante primero debe [verbo cognitivo 1]... Luego, necesita [verbo cognitivo 2]... Este proceso le permite [verbo cognitivo 3]..., lo que finalmente lo lleva a concluir que la opción [letra de la respuesta correcta] es la correcta porque [justificación final].

Análisis de Opciones No Válidas:
- **Opción [Letra del distractor]:** El estudiante podría escoger esta opción si comete un error de [naturaleza de la confusión], lo que lo lleva a pensar que [razonamiento erróneo]. Sin embargo, esto es incorrecto porque [razón clara y concisa].
"""

def construir_prompt_recomendaciones(fila):
    """Construye el prompt para las recomendaciones pedagógicas."""
    fila = fila.fillna('')
    return f"""
🎯 ROL DEL SISTEMA
Eres un diseñador instruccional experto en evaluación, especializado en crear material pedagógico de alto valor. Tu misión es generar dos recomendaciones **novedosas, creativas e inspiradoras** (Fortalecer y Avanzar) que cumplan de manera inviolable las siguientes directrices:
1.  **FIDELIDAD A LA TAXONOMÍA:** Toda recomendación debe originarse y alinearse estrictamente con la jerarquía cognitiva definida por la **Competencia, el Aprendizaje priorizado y la Evidencia**.
2.  **CERO PRODUCCIÓN ESCRITA:** Existe una prohibición total de actividades que impliquen escritura y producción. Debe estar centrado en procesos de lectura*.
3.  **GENERALIDAD DEL CONTENIDO:** Las actividades deben ser aplicables a textos generales y NO deben basarse en el contenido específico del ítem de entrada.
4.  **REDACCIÓN IMPERSONAL:** Todo el texto generado debe mantener un tono profesional e impersonal.

🧠 INSUMOS DE ENTRADA
- Texto/Fragmento: {fila.get('ItemContexto', 'No aplica')}
- Descripción del Ítem: {fila.get('ItemEnunciado', 'No aplica')}
- Componente: {fila.get('ComponenteNombre', 'No aplica')}
- Competencia: {fila.get('CompetenciaNombre', '')}
- Aprendizaje Priorizado: {fila.get('AfirmacionNombre', '')}
- Evidencia de Aprendizaje: {fila.get('EvidenciaNombre', '')}
- Tipología Textual (Solo para Lectura Crítica): {fila.get('Tipologia Textual', 'No aplica')}
- Grado Escolar: {fila.get('ItemGradoId', '')}
- Respuesta correcta: {fila.get('AlternativaClave', 'No aplica')}
- Opción A: {fila.get('OpcionA', 'No aplica')}
- Opción B: {fila.get('OpcionB', 'No aplica')}
- Opción C: {fila.get('OpcionC', 'No aplica')}
- Opción D: {fila.get('OpcionD', 'No aplica')}

📝 INSTRUCCIONES PARA GENERAR LAS RECOMENDACIONES
Genera las dos recomendaciones adhiriéndote estrictamente a lo siguiente:

### 1. Recomendación para FORTALECER 💪
- **Identificación de procesos** Identifica los procesos cognitivos (verbos) mas básicos necesarios para poder responder con el ítem de la taxonomia dada.
- **Objetivo Central:** Andamiar el proceso cognitivo exacto descrito en la competencia, aprendizaje priorizado y evidencia de aprendizaje.
- **Contexto Pedagógico:** La actividad debe ser un microcosmos de dicha evidencia, pero simplificada. Debes **descomponer el proceso cognitivo en pasos manejables**.
- **Actividad Propuesta:** Diseña una actividad de lectura, selección u organización oral que sea **novedosa, creativa y lúdica**. **Evita explícitamente ejercicios típicos** como cuestionarios, llenar espacios en blanco o buscar ideas principales de forma tradicional. La actividad debe sentirse como un juego o un pequeño acertijo.
- **Preguntas Orientadoras:** Formula preguntas que funcionen como un **"paso a paso" del razonamiento**, guiando al estudiante a través del proceso de forma sutil.

### 2. Recomendación para AVANZAR 🚀
- **Identificación de procesos** Identifica los procesos cognitivos (verbos) mas avanzados que permiten que un estudiante avance mas allá sin salir del proceso cognitivo general dado por la competencia.
- **Objetivo Central:** Asegurar una **progresión cognitiva clara y directa** comparada con la actividad planteada en Fortalecer.
- **Contexto Pedagógico:** La actividad para Avanzar debe ser la **evolución natural y más compleja de la habilidad trabajada en Fortalecer**. La conexión entre ambas debe ser explícita y lógica.
- **Actividad Propuesta:** Diseña un desafío intelectual de lectura, análisis comparativo u oral que sea **estimulante y poco convencional**. La actividad debe promover el pensamiento crítico y la transferencia de habilidades de una manera que no sea habitual en el aula.
- **Preguntas Orientadoras:** Formula preguntas abiertas que exijan **evaluación, síntesis, aplicación o metacognición**, demostrando un salto cualitativo respecto a las preguntas de Fortalecer.

✍️ FORMATO DE SALIDA DE LAS RECOMENDACIONES
**IMPORTANTE: Responde de forma directa, usando obligatoriamente la siguiente estructura. No añadas texto adicional.**
- **Redacción Impersonal:** Utiliza siempre una redacción profesional e impersonal (ej. "se sugiere", "la tarea consiste en", "se entregan tarjetas").
- **Sin Conclusiones:** Termina directamente con la lista de preguntas.

RECOMENDACIÓN PARA FORTALECER EL APRENDIZAJE EVALUADO EN EL ÍTEM
Para fortalecer la habilidad de [verbo clave extraído de la Evidencia de Aprendizaje], se sugiere [descripción de la estrategia de andamiaje para ese proceso exacto].
Una actividad que se puede hacer es: [Descripción detallada de la actividad novedosa y creativa, que no implica escritura].
Las preguntas orientadoras para esta actividad, entre otras, pueden ser:
- [Pregunta 1: Que guíe el primer paso del proceso cognitivo]
- [Pregunta 2: Que ayude a analizar un componente clave del proceso]
- [Pregunta 3: Que conduzca a la conclusión del proceso base]

RECOMENDACIÓN PARA AVANZAR EN EL APRENDIZAJE EVALUADO EN EL ÍTEM
Para avanzar desde [proceso cognitivo de Fortalecer] hacia la habilidad de [verbo clave del proceso cognitivo superior], se sugiere [descripción de la estrategia de complejización].
Una actividad que se puede hacer es: [Descripción detallada de la actividad estimulante y poco convencional, que no implique escritura].
Las preguntas orientadoras para esta actividad, entre otras, pueden ser:
- [Pregunta 1: De análisis o evaluación que requiera un razonamiento más profundo]
- [Pregunta 2: De aplicación, comparación o transferencia a un nuevo contexto]
- [Pregunta 3: De metacognición o pensamiento crítico sobre el proceso completo]
"""

def construir_prompt_auditoria(prompt_original, texto_generado):
    """Construye el prompt para el modelo auditor."""
    return f"""
🎯 ROL DEL SISTEMA
Eres un auditor de calidad experto en evaluación pedagógica. Tu misión es revisar de forma crítica un texto generado por otra IA para asegurar que cumple al 100% con las instrucciones del prompt original. Eres estricto, preciso y tu objetivo es la excelencia.

📝 INSTRUCCIONES DEL PROMPT ORIGINAL (PARA TU REFERENCIA):
---
{prompt_original}
---

📄 TEXTO GENERADO QUE DEBES AUDITAR:
---
{texto_generado}
---

✅ TU TAREA DE AUDITORÍA:
Revisa el "TEXTO GENERADO" y compáralo contra las "INSTRUCCIONES DEL PROMPT ORIGINAL". Responde únicamente con la siguiente estructura. No añadas texto fuera de esta estructura.

VEREDICTO: [APROBADO o REQUIERE REVISIÓN]

ANÁLISIS DE CUMPLIMIENTO:
- Fidelidad a la Taxonomía: [Indica si cumple y por qué, o dónde falla]
- Cero Producción Escrita: [Confirma si se cumple la regla de no escritura]
- Novedad y Creatividad: [Evalúa si la actividad es novedosa o un ejercicio típico]
- Redacción Impersonal: [Verifica si el tono es consistentemente impersonal]
- Progresión Lógica (si aplica): [Analiza si la progresión Fortalecer/Avanzar es lógica]

SUGERENCIAS DE MEJORA:
[Si el VEREDICTO es "REQUIERE REVISIÓN", proporciona sugerencias claras y accionables para que la otra IA corrija el texto. Si es "APROBADO", escribe "Ninguna".]
"""

# --- INTERFAZ PRINCIPAL DE STREAMLIT ---
st.title("🤖 Ensamblador de Fichas con Auditoría Cruzada de IA")
st.markdown("Genera, audita y refina contenido pedagógico para asegurar la máxima calidad.")

# Inicializar session_state para todo
if 'df_enriquecido' not in st.session_state:
    st.session_state.df_enriquecido = None
if 'zip_buffer' not in st.session_state:
    st.session_state.zip_buffer = None

# --- PASO 0: Configuración de APIs ---
st.sidebar.header("🔑 Configuración de APIs")
gemini_api_key = st.sidebar.text_input("Ingresa tu Clave API de Google AI (Gemini)", type="password", help="Necesaria si seleccionas Gemini como generador o auditor.")
openai_api_key = st.sidebar.text_input("Ingresa tu Clave API de OpenAI (GPT)", type="password", help="Necesaria si seleccionas un modelo GPT como generador o auditor.")

# --- PASO 1: Carga de Archivos ---
st.header("Paso 1: Carga tus Archivos")
col1, col2 = st.columns(2)
with col1:
    archivo_excel = st.file_uploader("Sube tu Excel", type=["xlsx"])
with col2:
    archivo_plantilla = st.file_uploader("Sube tu Plantilla de Word", type=["docx"])

# --- PASO 2: Selección de Modelos y Generación ---
st.header("Paso 2: Configura la Generación y Auditoría")

col_gen, col_aud = st.columns(2)
with col_gen:
    modelo_generador_nombre = st.selectbox(
        "🤖 Elige el Modelo GENERADOR",
        ("Gemini 1.5 Pro", "GPT-4o", "GPT-3.5-Turbo"),
        help="Este modelo creará el contenido inicial."
    )
with col_aud:
    auditoria_activada = st.checkbox("Activar Auditoría Cruzada", value=True)
    modelo_auditor_nombre = st.selectbox(
        "✅ Elige el Modelo AUDITOR",
        ("GPT-4o", "Gemini 1.5 Pro", "GPT-3.5-Turbo"),
        index=0,  # GPT-4o por defecto como auditor
        help="Este modelo revisará el trabajo del generador.",
        disabled=not auditoria_activada
    )

if st.button("🚀 Iniciar Proceso de Generación y Auditoría", type="primary"):
    is_gemini_needed = "Gemini" in [modelo_generador_nombre, modelo_auditor_nombre]
    is_gpt_needed = any("GPT" in name for name in [modelo_generador_nombre, modelo_auditor_nombre])

    if not archivo_excel or (is_gemini_needed and not gemini_api_key) or (is_gpt_needed and not openai_api_key):
        st.error("Por favor, sube un archivo Excel y asegúrate de ingresar las claves API necesarias para los modelos seleccionados.")
    else:
        gemini_model = setup_gemini_model(gemini_api_key) if is_gemini_needed else True
        openai_client = setup_openai_model(openai_api_key) if is_gpt_needed else True

        if (gemini_model is None) or (openai_client is None):
            st.error("Fallo en la configuración de uno de los modelos. Revisa las claves API.")
        else:
            modelos = {
                "Gemini 1.5 Pro": (gemini_model, "gemini-1.5-pro-latest"),
                "GPT-4o": (openai_client, "gpt-4o"),
                "GPT-3.5-Turbo": (openai_client, "gpt-3.5-turbo")
            }
            cliente_generador, nombre_gen_api = modelos[modelo_generador_nombre]
            cliente_auditor, nombre_aud_api = modelos[modelo_auditor_nombre]

            df = pd.read_excel(archivo_excel)
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].apply(limpiar_html)

            columnas_nuevas = ["Que_Evalua", "Justificacion_Correcta", "Analisis_Distractores", "Recomendacion_Fortalecer", "Recomendacion_Avanzar"]
            for col in columnas_nuevas:
                if col not in df.columns:
                    df[col] = ""

            progress_bar_main = st.progress(0, text="Iniciando Proceso...")
            total_filas = len(df)

            for i, fila in df.iterrows():
                item_id = fila.get('ItemId', i + 1)
                st.markdown(f"--- \n ### Procesando Ítem: **{item_id}**")
                progress_bar_main.progress(i / total_filas, text=f"Procesando ítem {i+1}/{total_filas}")

                # --- Generación y Auditoría de ANÁLISIS ---
                with st.container(border=True):
                    st.write("#### 1. Generando y Auditando: Análisis del Ítem")
                    prompt_actual = construir_prompt_analisis(fila)
                    texto_actual = generar_contenido(cliente_generador, nombre_gen_api, prompt_actual)

                    if auditoria_activada and isinstance(texto_actual, str) and "ERROR API" not in texto_actual:
                        for audit_pass in range(3):
                            feedback = generar_contenido(cliente_auditor, nombre_aud_api, construir_prompt_auditoria(prompt_actual, texto_actual))
                            
                            with st.expander(f"Ver Auditoría de Análisis #{audit_pass + 1}"):
                                if isinstance(feedback, str):
                                    st.markdown(f"```\n{feedback}\n```")
                                else:
                                    st.error("La auditoría no devolvió una respuesta de texto válida.")

                            if isinstance(feedback, str) and "APROBADO" in feedback:
                                st.success(f"Análisis Aprobado en el intento #{audit_pass + 1}")
                                break
                            
                            if audit_pass < 2:
                                st.warning(f"Análisis requiere revisión. Refinando... (Intento {audit_pass + 2})")
                                refine_prompt = f"{prompt_actual}\n\nEl texto anterior que generaste fue auditado. Aquí están las sugerencias para corregirlo: {feedback}\n\nGenera una nueva versión corregida que cumpla con el formato de salida requerido."
                                texto_actual = generar_contenido(cliente_generador, nombre_gen_api, refine_prompt)
                                time.sleep(1)
                            else:
                                st.error(f"El análisis para el ítem {item_id} no fue aprobado después de 3 intentos.")
                    
                    if isinstance(texto_actual, str) and texto_actual:
                        header_que_evalua = "Qué Evalúa:"
                        header_correcta = "Ruta Cognitiva Correcta:"
                        header_distractores = "Análisis de Opciones No Válidas:"
                        idx_correcta = texto_actual.find(header_correcta)
                        idx_distractores = texto_actual.find(header_distractores)
                        
                        if idx_correcta != -1 and idx_distractores != -1 and texto_actual.startswith(header_que_evalua):
                            df.loc[i, "Que_Evalua"] = texto_actual[len(header_que_evalua):idx_correcta].strip()
                            df.loc[i, "Justificacion_Correcta"] = texto_actual[idx_correcta + len(header_correcta):idx_distractores].strip()
                            df.loc[i, "Analisis_Distractores"] = texto_actual[idx_distractores + len(header_distractores):].strip()
                        else:
                            df.loc[i, "Que_Evalua"] = "ERROR DE PARSEO"
                            df.loc[i, "Justificacion_Correcta"] = texto_actual
                            df.loc[i, "Analisis_Distractores"] = ""
                    else:
                        st.warning(f"La API no devolvió un texto válido para el análisis del ítem {item_id}.")
                        df.loc[i, "Que_Evalua"] = "ERROR API: Respuesta no válida"
                        df.loc[i, "Justificacion_Correcta"] = "ERROR API: Respuesta no válida"
                        df.loc[i, "Analisis_Distractores"] = "ERROR API: Respuesta no válida"

                # --- Generación y Auditoría de RECOMENDACIONES ---
                with st.container(border=True):
                    st.write("#### 2. Generando y Auditando: Recomendaciones Pedagógicas")
                    prompt_actual = construir_prompt_recomendaciones(fila)
                    texto_actual = generar_contenido(cliente_generador, nombre_gen_api, prompt_actual)
                    
                    if auditoria_activada and isinstance(texto_actual, str) and "ERROR API" not in texto_actual:
                        for audit_pass in range(3):
                            feedback = generar_contenido(cliente_auditor, nombre_aud_api, construir_prompt_auditoria(prompt_actual, texto_actual))
                            
                            with st.expander(f"Ver Auditoría de Recomendaciones #{audit_pass + 1}"):
                                if isinstance(feedback, str):
                                    st.markdown(f"```\n{feedback}\n```")
                                else:
                                    st.error("La auditoría no devolvió una respuesta de texto válida.")
                            
                            if isinstance(feedback, str) and "APROBADO" in feedback:
                                st.success(f"Recomendaciones Aprobadas en el intento #{audit_pass + 1}")
                                break

                            if audit_pass < 2:
                                st.warning(f"Recomendaciones requieren revisión. Refinando... (Intento {audit_pass + 2})")
                                refine_prompt = f"{prompt_actual}\n\nEl texto anterior que generaste fue auditado. Aquí están las sugerencias para corregirlo: {feedback}\n\nGenera una nueva versión corregida que cumpla con el formato de salida requerido (Fortalecer y Avanzar)."
                                texto_actual = generar_contenido(cliente_generador, nombre_gen_api, refine_prompt)
                                time.sleep(1)
                            else:
                                st.error(f"Las recomendaciones para el ítem {item_id} no fueron aprobadas después de 3 intentos.")

                    if isinstance(texto_actual, str) and texto_actual:
                        titulo_avanzar = "RECOMENDACIÓN PARA AVANZAR"
                        idx_avanzar = texto_actual.upper().find(titulo_avanzar)
                        if idx_avanzar != -1:
                            df.loc[i, "Recomendacion_Fortalecer"] = texto_actual[:idx_avanzar].strip()
                            df.loc[i, "Recomendacion_Avanzar"] = texto_actual[idx_avanzar:].strip()
                        else:
                            df.loc[i, "Recomendacion_Fortalecer"] = texto_actual
                            df.loc[i, "Recomendacion_Avanzar"] = "ERROR DE PARSEO: No se encontró 'AVANZAR'"
                    else:
                        st.warning(f"La API no devolvió un texto válido para las recomendaciones del ítem {item_id}.")
                        df.loc[i, "Recomendacion_Fortalecer"] = "ERROR API: Respuesta no válida"
                        df.loc[i, "Recomendacion_Avanzar"] = "ERROR API: Respuesta no válida"

            progress_bar_main.progress(1.0, text="¡Proceso completado!")
            st.session_state.df_enriquecido = df
            st.balloons()


# --- PASO 3: Vista Previa y Verificación ---
if st.session_state.df_enriquecido is not None:
    st.header("Paso 3: Verifica los Datos Enriquecidos")
    st.dataframe(st.session_state.df_enriquecido.head())
    
    output_excel = BytesIO()
    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        st.session_state.df_enriquecido.to_excel(writer, index=False, sheet_name='Datos Enriquecidos')
    output_excel.seek(0)
    
    st.download_button(
        label="📥 Descargar Excel Enriquecido",
        data=output_excel,
        file_name="excel_enriquecido_con_ia.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# --- PASO 4: Ensamblaje de Fichas ---
if st.session_state.df_enriquecido is not None and archivo_plantilla is not None:
    st.header("Paso 4: Ensambla las Fichas Técnicas")
    
    columna_nombre_archivo = st.text_input(
        "Escribe el nombre de la columna para nombrar los archivos (ej. ItemId)",
        value="ItemId"
    )
    
    if st.button("📄 Ensamblar Fichas Técnicas", type="primary"):
        df_final = st.session_state.df_enriquecido
        if columna_nombre_archivo not in df_final.columns:
            st.error(f"La columna '{columna_nombre_archivo}' no existe en el Excel. Por favor, elige una de: {', '.join(df_final.columns)}")
        else:
            with st.spinner("Ensamblando todas las fichas en un archivo .zip..."):
                plantilla_bytes = BytesIO(archivo_plantilla.getvalue())
                zip_buffer = BytesIO()
                
                with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
                    total_docs = len(df_final)
                    progress_bar_zip = st.progress(0, text="Iniciando ensamblaje...")
                    for i, fila in df_final.iterrows():
                        doc = DocxTemplate(plantilla_bytes)
                        contexto = fila.to_dict()
                        contexto_limpio = {k: (v if pd.notna(v) else "") for k, v in contexto.items()}
                        doc.render(contexto_limpio)
                        
                        doc_buffer = BytesIO()
                        doc.save(doc_buffer)
                        doc_buffer.seek(0)
                        
                        nombre_base = str(fila.get(columna_nombre_archivo, f"ficha_{i+1}")).replace('/', '_').replace('\\', '_')
                        nombre_archivo_salida = f"{nombre_base}.docx"
                        
                        zip_file.writestr(nombre_archivo_salida, doc_buffer.getvalue())
                        progress_bar_zip.progress((i + 1) / total_docs, text=f"Añadiendo ficha {i+1}/{total_docs} al .zip")
                
                st.session_state.zip_buffer = zip_buffer
                st.success("¡Ensamblaje completado!")

# --- PASO 5: Descarga Final ---
if st.session_state.zip_buffer:
    st.header("Paso 5: Descarga el Resultado Final")
    st.download_button(
        label="📥 Descargar TODAS las fichas (.zip)",
        data=st.session_state.zip_buffer.getvalue(),
        file_name="fichas_tecnicas_generadas.zip",
        mime="application/zip"
    )
