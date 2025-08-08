# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
from docxtpl import DocxTemplate
import google.generativeai as genai
import os
import re
import time
import zipfile
from io import BytesIO

# --- CONFIGURACIÓN DE LA PÁGINA DE STREAMLIT ---
st.set_page_config(
    page_title="Ensamblador de Fichas Técnicas con IA",
    page_icon="🤖",
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

def setup_model(api_key):
    """Configura y retorna el cliente para el modelo Gemini."""
    try:
        genai.configure(api_key=api_key)
        generation_config = {
            "temperature": 0.6, "top_p": 1, "top_k": 1, "max_output_tokens": 8192
        }
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"},
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

# --- EJEMPLOS DE ALTA CALIDAD (FEW-SHOT PROMPTING) ---

EJEMPLOS_ANALISIS_PREMIUM = """
A continuación, te muestro ejemplos de análisis de la más alta calidad. Tu respuesta debe seguir este mismo estilo, tono y nivel de detalle.

### EJEMPLO 1: LECTURA LITERAL (TEXTO NARRATIVO) ###
**INSUMOS:**
- Competencia: Comprensión de textos
- Componente: Lectura literal
- Evidencia: Reconoce información específica en el texto.
- Enunciado: Los personajes del cuento son:
- Opciones: A: "Un hombre, un hombrecito y alguien que sostiene unas pinzas.", B: "Un narrador, un hombre y un hombrecito.", C: Un hombrecito y alguien que sostiene unas pinzas., D: Un hombre y el narrador.

**RESULTADO ESPERADO:**
Ruta Cognitiva Correcta:
Para responder el ítem, el estudiante debe leer el cuento prestando atención a las entidades que realizan acciones o a quienes les suceden eventos en el texto. En el tercer párrafo, se menciona a "un hombre" que armó el barquito y a un "hombrecito diminuto" dentro de la botella. En el último párrafo, se describe que un "ojo enorme lo atisbaba desde fuera" al primer hombre y que "unas enormes pinzas que avanzaban hacia él". Este "ojo enorme" y las "enormes pinzas" implican la existencia de un tercer personaje, un ser que se encuentra mirando al primer personaje. El estudiante debe identificar a todos estos personajes que interactúan o son afectados por la trama.

Análisis de Opciones No Válidas:
- **Opción B:** No es correcta porque, en este cuento, el "narrador" es la voz que cuenta la historia, no un personaje que participe en los eventos del cuento. El relato está escrito en tercera persona y el narrador se mantiene fuera de la acción.
- **Opción C:** No es correcta porque omite al primer personaje introducido y central en la trama: "un hombre" que construye el barquito y observa al "hombrecito". Sin este personaje, la secuencia de eventos no se establece.
- **Opción D:** No es correcta porque, al igual que la opción B, incluye al "narrador" como personaje, lo cual es incorrecto. Además, omite al "hombrecito" y al ser con "unas pinzas", reduciendo el número de personajes activos en la historia.

### EJEMPLO 2: LECTURA INFERENCIAL (TEXTO NARRATIVO-INFORMATIVO) ###
**INSUMOS:**
- Competencia: Comprensión de textos
- Componente: Lectura inferencial
- Evidencia: Integra y compara diferentes partes del texto y analiza la estructura para hacer inferencias.
- Enunciado: Lee el siguiente fragmento del texto: “Los manglares están muriendo, por lo que el desequilibrio es cada vez mayor. La carretera lo cambió todo. Para construirla arrasaron veinte mil hectáreas de manglar...”. ¿Qué función cumple la parte subrayada dentro del fragmento?
- Opciones: A: Señalar la causa de un problema medioambiental., B: Establecer una comparación entre dos acciones de un proceso., C: Mostrar la consecuencia del daño medioambiental., D: Explicar el motivo por el que se decidió realizar una acción.

**RESULTADO ESPERADO:**
Ruta Cognitiva Correcta:
El estudiante debe comprender el contenido del fragmento y la estructura global del texto, para luego identificar cuál es la función que cumple dentro de esta. En este caso específico, el estudiante debe comprender que el fragmento señala la principal causa que ha llevado al desequilibrio del ecosistema de los manglares en la zona, y que este fragmento del texto justamente cumple con la función de señalar esa causa.

Análisis de Opciones No Válidas:
- **Opción B:** Es incorrecta porque la pregunta busca la causa del problema, no la comparación de acciones.
- **Opción C:** Es incorrecta porque el estudiante confunde la causa con la consecuencia del problema medioambiental. Identifica un efecto del problema, pero no su origen.
- **Opción D:** Es incorrecta porque se centra en la motivación detrás de una acción, en lugar de la causa del problema en sí mismo. La pregunta busca el origen del problema medioambiental.

### EJEMPLO 3: LECTURA CRÍTICA (TEXTO NARRATIVO-INFORMATIVO) ###
**INSUMOS:**
- Competencia: Comprensión de textos
- Componente: Lectura crítica
- Evidencia: Evalúa la credibilidad, confiabilidad y objetividad del texto, emitiendo juicios críticos sobre la información.
- Enunciado: ¿Por qué el autor cita el testimonio de Jesús Suárez en el texto?
- Opciones: A: Porque es el vocero que la comunidad palafítica ha designado., B: Porque es causante de la situación que ocurre en la población., C: Porque al ser experto en ecosistemas acuáticos su opinión es confiable., D: Porque al ser investigador puede verificar lo dicho por otro testigo de los hechos.

**RESULTADO ESPERADO:**
Ruta Cognitiva Correcta:
El estudiante analiza las opciones presentadas considerando la relación entre la justificación dada y la confiabilidad de la fuente. Evalúa la opción C y reconoce que la experticia en ecosistemas acuáticos otorga mayor credibilidad a la opinión de un individuo sobre una situación relacionada con este tema. Justifica la selección de la opción C al contrastarla con las demás opciones, considerando la relevancia de la experticia para la situación planteada.

Análisis de Opciones No Válidas:
- **Opción A:** Es incorrecta porque ser vocero no implica necesariamente tener el conocimiento experto para opinar sobre situaciones específicas.
- **Opción B:** Es incorrecta porque ser causante de un problema no implica tener el conocimiento o la imparcialidad para analizarlo y ofrecer una opinión confiable.
- **Opción D:** Es incorrecta porque la verificación de un testimonio en este contexto requiere una experticia específica en el tema, que en este caso es ecosistemas acuáticos.
"""

EJEMPLOS_RECOMENDACIONES_PREMIUM = """
A continuación, te muestro ejemplos de recomendaciones pedagógicas de la más alta calidad. Tu respuesta debe seguir este mismo estilo, estructura y enfoque creativo.

### EJEMPLO 1 DE RECOMENDACIONES PERFECTAS (TEXTO DISCONTINUO) ###
**INSUMOS:**
- Qué Evalúa el Ítem: El ítem evalúa la habilidad del estudiante para relacionar diferentes elementos del contenido e identificar nueva información en textos no literarios.
- Evidencia: Relaciona diferentes partes del texto para hacer inferencias sobre significados o sobre el propósito general.

**RESULTADO ESPERADO:**
RECOMENDACIÓN PARA FORTALECER EL APRENDIZAJE EVALUADO EN EL ÍTEM
Para reforzar la habilidad de vincular diferentes elementos del contenido y descubrir nuevas ideas, se sugiere la realización de actividades que impliquen el análisis de textos no literarios de carácter discontinuo como infografías. Los estudiantes podrían empezar por leer estas fuentes y marcar los datos que consideren relevantes. Posteriormente, en un esfuerzo colectivo, podrían construir un mapa conceptual que refleje la relación entre los diferentes datos resaltados. Finalmente, podrían trabajar en la identificación de las ideas principales y secundarias que emergen de este mapa, lo que les permitirá tener una comprensión más profunda del texto.

RECOMENDACIÓN PARA AVANZAR EN EL APRENDIZAJE EVALUADO EN EL ÍTEM
Para consolidar la capacidad de identificar las funciones de los diferentes elementos que componen un texto no literario de carácter discontinuo, se sugiere fomentar la práctica de reorganizar textos desordenados. Los estudiantes pueden recibir fragmentos de una infografía que deben arreglar en el orden correcto, identificando la introducción, el desarrollo y la conclusión. Durante esta actividad, se pueden formular preguntas como: ¿Cuál fragmento introduce el tema? ¿Qué información proporciona esta imagen o gráfico? ¿Cómo se relaciona con el texto?

### EJEMPLO 2 DE RECOMENDACIONES PERFECTAS (TEXTO INFORMATIVO) ###
**INSUMOS:**
- Qué Evalúa el Ítem: Este ítem evalúa la capacidad del estudiante para hacer una inferencia integrando información implícita presente en una parte del texto.
- Evidencia: Integra y compara diferentes partes del texto y analiza la estructura para hacer inferencias.

**RESULTADO ESPERADO:**
RECOMENDACIÓN PARA FORTALECER EL APRENDIZAJE EVALUADO EN EL ÍTEM
Para fortalecer la habilidad de hacer inferencias a partir de un segmento de un texto informativo, se sugiere implementar una dinámica de "lectura de pistas". Esta estrategia se enfoca en que los estudiantes identifiquen información implícita en fragmentos textuales cortos para inferir contextos o emociones que no se mencionan directamente. El docente puede presentar al grupo tres o cuatro fragmentos muy breves y evocadores (de noticias o crónicas) que insinúen una situación sin describirla por completo. Por ejemplo: "El teléfono sonó por décima vez. Al otro lado de la línea, solo se oía una respiración agitada. Afuera, la sirena de una ambulancia se acercaba". Los estudiantes, en parejas, leen el fragmento y discuten qué pueden deducir de la escena. Las preguntas orientadoras pueden ser: ¿Qué pistas te da el texto sobre el estado de ánimo de la persona?, ¿Qué crees que pasó justo antes de la escena descrita?

RECOMENDACIÓN PARA AVANZAR EN EL APRENDIZAJE EVALUADO EN EL ÍTEM
Para avanzar en la habilidad de hacer inferencias complejas a partir de la comparación de diferentes partes de un texto, se sugiere proponer un análisis de perspectivas múltiples dentro de una misma crónica o texto informativo. El objetivo es que los estudiantes superen la inferencia local y aprendan a contrastar voces, datos o argumentos presentados en un mismo relato. El docente puede seleccionar una crónica periodística sobre un tema urbano actual que incluya las voces de distintos actores sociales (un vendedor, un residente, un funcionario). Los estudiantes deben leer el texto e identificar y comparar las diferentes posturas frente al mismo hecho. Las preguntas orientadoras pueden ser: ¿Qué similitudes y diferencias encuentras entre las perspectivas?, ¿Qué visión del problema se formaría un lector si el texto solo hubiera incluido una de estas voces?
"""

# --- FUNCIONES DE PROMPTS SECUENCIALES ---

def construir_prompt_paso1_analisis_central(fila):
    """Paso 1: Genera la Ruta Cognitiva y el Análisis de Distractores, guiado por ejemplos."""
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
Eres un experto psicómetra y pedagogo. Tu misión es deconstruir un ítem de evaluación siguiendo el estilo y la calidad de los ejemplos proporcionados.

{EJEMPLOS_ANALISIS_PREMIUM}

🧠 INSUMOS DE ENTRADA (Para el nuevo ítem que debes analizar):
- Texto/Fragmento: {fila.get('ItemContexto', 'No aplica')}
- Descripción del Ítem: {fila.get('ItemEnunciado', 'No aplica')}
- Componente: {fila.get('ComponenteNombre', 'No aplica')}
- Competencia: {fila.get('CompetenciaNombre', '')}
- Aprendizaje Priorizado: {fila.get('AfirmacionNombre', '')}
- Evidencia de Aprendizaje: {fila.get('EvidenciaNombre', '')}
- Tipología Textual (Solo para Lectura Crítica): {fila.get('Tipologia Textual', 'No aplica')}
- Grado Escolar: {fila.get('ItemGradoId', '')}
- Análisis de Errores Comunes: {fila.get('Analisis_Errores', 'No aplica')}
- Respuesta correcta: {fila.get('AlternativaClave', 'No aplica')}
- Opción A: {fila.get('OpcionA', 'No aplica')}
- Opción B: {fila.get('OpcionB', 'No aplica')}
- Opción C: {fila.get('OpcionC', 'No aplica')}
- Opción D: {fila.get('OpcionD', 'No aplica')}


📝 INSTRUCCIONES
Basándote en los ejemplos de alta calidad y los nuevos insumos, realiza el siguiente proceso en dos fases:

FASE 1: RUTA COGNITIVA
Describe, en un párrafo continuo y de forma impersonal, el procedimiento mental que un estudiante debe ejecutar para llegar a la respuesta correcta.
1.  **Genera la Ruta Cognitiva:** Describe el paso a paso mental y lógico que un estudiante debe seguir para llegar a la respuesta correcta. Usa verbos que representen procesos cognitivos.
2.  **Auto-Verificación:** Revisa que la ruta se alinee con la Competencia ('{fila.get('CompetenciaNombre', '')}') y la Evidencia ('{fila.get('EvidenciaNombre', '')}').
3.  **Justificación Final:** El último paso debe justificar la elección de la respuesta correcta.

FASE 2: ANÁLISIS DE OPCIONES NO VÁLIDAS
- Para cada opción incorrecta, identifica la naturaleza del error y explica el razonamiento fallido.
- Luego, explica el posible razonamiento que lleva al estudiante a cometer ese error.
- Finalmente, clarifica por qué esa opción es incorrecta en el contexto de la tarea evaluativa.

✍️ FORMATO DE SALIDA
**REGLA CRÍTICA:** Responde únicamente con los dos títulos siguientes, en este orden y sin añadir texto adicional.

Ruta Cognitiva Correcta:
[Párrafo continuo y detallado.] Debe describir como es la secuencia de procesos cognitivos. Ejemplo: Para resolver correctamente este ítem, el estudiante primero debe [verbo cognitivo 1]... Luego, necesita [verbo cognitivo 2]... Este proceso le permite [verbo cognitivo 3]..., lo que finalmente lo lleva a concluir que la opción [letra de la respuesta correcta] es la correcta porque [justificación final].

Análisis de Opciones No Válidas:
- **Opción [Letra del distractor]:** El estudiante podría escoger esta opción si comete un error de [naturaleza de la confusión u error], lo que lo lleva a pensar que [razonamiento erróneo]. Sin embargo, esto es incorrecto porque [razón clara y concisa].
"""

def construir_prompt_paso2_sintesis_que_evalua(analisis_central_generado, fila):
    """Paso 2: Sintetiza el "Qué Evalúa" a partir del análisis central."""
    fila = fila.fillna('')
    try:
        header_distractores = "Análisis de Opciones No Válidas:"
        idx_distractores = analisis_central_generado.find(header_distractores)
        ruta_cognitiva_texto = analisis_central_generado[:idx_distractores].strip() if idx_distractores != -1 else analisis_central_generado
    except:
        ruta_cognitiva_texto = analisis_central_generado

    return f"""
🎯 ROL DEL SISTEMA
Eres un experto en evaluación que sintetiza análisis complejos en una sola frase concisa.

🧠 INSUMOS DE ENTRADA
A continuación, te proporciono un análisis detallado de la ruta cognitiva necesaria para resolver un ítem.

ANÁLISIS DE LA RUTA COGNITIVA:
---
{ruta_cognitiva_texto}
---

TAXONOMÍA DE REFERENCIA:
- Competencia: {fila.get('CompetenciaNombre', '')}
- Aprendizaje Priorizado: {fila.get('AfirmacionNombre', '')}
- Evidencia de Aprendizaje: {fila.get('EvidenciaNombre', '')}

📝 INSTRUCCIONES
Basándote **exclusivamente** en el ANÁLISIS DE LA RUTA COGNITIVA, redacta una única frase (máximo 2 renglones) que resuma la habilidad principal que se está evaluando.
- **Regla 1:** La frase debe comenzar obligatoriamente con "Este ítem evalúa la capacidad del estudiante para...".
- **Regla 2:** La frase debe describir los **procesos cognitivos**, no debe contener especificamene ninguno de los elementos del texto o del ítem, busca en cambio palabras/expresiones genéricas en reemplazo de elementos del item/texto cuando es necesario.
- **Regla 3:** Utiliza la TAXONOMÍA DE REFERENCIA para asegurar que el lenguaje sea preciso y alineado.

✍️ FORMATO DE SALIDA
Responde únicamente con la frase solicitada, sin el título "Qué Evalúa".
"""

def construir_prompt_paso3_recomendaciones(que_evalua_sintetizado, analisis_central_generado, fila):
    """Paso 3: Genera las recomendaciones, guiado por ejemplos."""
    fila = fila.fillna('')
    return f"""
🎯 ROL DEL SISTEMA
Eres un diseñador instruccional experto, especializado en crear actividades de lectura novedosas, siguiendo el estándar de los ejemplos provistos.

{EJEMPLOS_RECOMENDACIONES_PREMIUM}

🧠 INSUMOS DE ENTRADA (Para el nuevo ítem):
- Qué Evalúa el Ítem: {que_evalua_sintetizado}
- Análisis Detallado del Ítem: {analisis_central_generado}
- Texto/Fragmento: {fila.get('ItemContexto', 'No aplica')}
- Descripción del Ítem: {fila.get('ItemEnunciado', 'No aplica')}
- Componente: {fila.get('ComponenteNombre', 'No aplica')}
- Competencia: {fila.get('CompetenciaNombre', '')}
- Aprendizaje Priorizado: {fila.get('AfirmacionNombre', '')}
- Evidencia de Aprendizaje: {fila.get('EvidenciaNombre', '')}
- Tipología Textual (Solo para Lectura Crítica): {fila.get('Tipologia Textual', 'No aplica')}
- Grado Escolar: {fila.get('ItemGradoId', '')}
- Análisis de Errores Comunes: {fila.get('Analisis_Errores', 'No aplica')}
- Respuesta correcta: {fila.get('AlternativaClave', 'No aplica')}

📝 INSTRUCCIONES PARA GENERAR LAS RECOMENDACIONES
Basándote en los ejemplos de alta calidad y los nuevos insumos, genera dos recomendaciones (Fortalecer y Avanzar) que cumplan con estas reglas inviolables:
1.  **FIDELIDAD A LA TAXONOMÍA:** Las actividades deben alinearse con el 'Qué Evalúa el Ítem'.
2.  **CERO PRODUCCIÓN ESCRITA:** Deben ser actividades exclusivas de lectura, selección u organización oral.
3.  **GENERALIDAD Y CREATIVIDAD:** Las actividades deben ser novedosas, lúdicas, no típicas, y aplicables a textos generales.
4.  **REDACCIÓN IMPERSONAL.**

### 1. Recomendación para FORTALECER 💪
- **Objetivo:** Descomponer el proceso cognitivo descrito en el 'Qué Evalúa' en pasos manejables.
- **Actividad:** Diseña una actividad que sirva de andamio para la habilidad central. No le pongas ningun nombre a la actividad.
- **Preguntas:** Formula preguntas que guíen el razonamiento paso a paso.
- **Contexto Pedagógico:** La actividad debe ser un microcosmos de dicha evidencia, pero simplificada. Debes **descomponer el proceso cognitivo en pasos manejables**.
- **Actividad Propuesta:** Diseña una actividad de lectura que sea **novedosa, creativa y lúdica**. **Evita explícitamente ejercicios típicos** como cuestionarios, llenar espacios en blanco o buscar ideas principales de forma tradicional. La actividad debe ser útil para los profesores.
- **Preguntas Orientadoras:** Formula preguntas que funcionen como un **"paso a paso" del razonamiento**, guiando al estudiante a través del proceso de forma sutil.


### 2. Recomendación para AVANZAR 🚀
- **Objetivo:** Crear una progresión cognitiva clara desde Fortalecer, dentro de la misma Competencia.
- **Objetivo Central:** Asegurar una **progresión cognitiva clara y directa en la que el estudiante avanza** cuando se compara con la actividad de Fortalecer.
- **Contexto Pedagógico:** La actividad para Avanzar debe ser la **evolución natural y más compleja de la habilidad trabajada en Fortalecer**. La conexión entre ambas debe ser explícita y lógica.  No le pongas ningun nombre a la actividad.
- **Actividad Propuesta:** Diseña un desafío intelectual de lectura o análisis comparativo que sea **estimulante y poco convencional**. La actividad debe promover el pensamiento crítico y la transferencia de habilidades de una manera que no sea habitual en el aula.
- **Preguntas Orientadoras:** Formula preguntas abiertas que exijan **evaluación, síntesis, aplicación o metacognición**, demostrando un salto cualitativo respecto a las preguntas de Fortalecer.

✍️ FORMATO DE SALIDA DE LAS RECOMENDACIONES
**IMPORTANTE: Responde de forma directa, usando obligatoriamente la siguiente estructura. No añadas texto adicional.**
- **Redacción Impersonal:** Utiliza siempre una redacción profesional e impersonal (ej. "se sugiere (sin mencionar el docente)", "la tarea consiste en", "se entregan tarjetas").
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

# --- INTERFAZ PRINCIPAL DE STREAMLIT ---

st.title("🤖 Ensamblador de Fichas Técnicas con IA")
st.markdown("Una aplicación para enriquecer datos pedagógicos y generar fichas personalizadas.")

if 'df_enriquecido' not in st.session_state:
    st.session_state.df_enriquecido = None
if 'zip_buffer' not in st.session_state:
    st.session_state.zip_buffer = None

# --- PASO 0: Clave API ---
st.sidebar.header("🔑 Configuración Obligatoria")
api_key = st.sidebar.text_input("Ingresa tu Clave API de Google AI (Gemini)", type="password")

# --- PASO 1: Carga de Archivos ---
st.header("Paso 1: Carga tus Archivos")
col1, col2 = st.columns(2)
with col1:
    archivo_excel = st.file_uploader("Sube tu Excel con los datos base", type=["xlsx"])
with col2:
    archivo_plantilla = st.file_uploader("Sube tu Plantilla de Word", type=["docx"])

# --- PASO 2: Enriquecimiento con IA ---
st.header("Paso 2: Enriquece tus Datos con IA")
if st.button("🤖 Iniciar Análisis y Generación", disabled=(not api_key or not archivo_excel)):
    if not api_key:
        st.error("Por favor, ingresa tu clave API en la barra lateral izquierda.")
    elif not archivo_excel:
        st.warning("Por favor, sube un archivo Excel para continuar.")
    else:
        model = setup_model(api_key)
        if model:
            with st.spinner("Procesando archivo Excel y preparando datos..."):
                df = pd.read_excel(archivo_excel)
                for col in df.columns:
                    if df[col].dtype == 'object':
                        df[col] = df[col].apply(limpiar_html)

                columnas_nuevas = ["Que_Evalua", "Justificacion_Correcta", "Analisis_Distractores", "Recomendacion_Fortalecer", "Recomendacion_Avanzar"]
                for col in columnas_nuevas:
                    if col not in df.columns:
                        df[col] = ""
                st.success("Datos limpios y listos.")

            progress_bar_main = st.progress(0, text="Iniciando Proceso...")
            total_filas = len(df)

            for i, fila in df.iterrows():
                item_id = fila.get('ItemId', i + 1)
                st.markdown(f"--- \n ### Procesando Ítem: **{item_id}**")
                progress_bar_main.progress(i / total_filas, text=f"Procesando ítem {i+1}/{total_filas}")

                with st.container(border=True):
                    try:
                        # --- LLAMADA 1: ANÁLISIS CENTRAL (RUTA COGNITIVA Y DISTRACTORES) ---
                        st.write(f"**Paso 1/3:** Realizando análisis central del ítem...")
                        prompt_paso1 = construir_prompt_paso1_analisis_central(fila)
                        response_paso1 = model.generate_content(prompt_paso1)
                        analisis_central = response_paso1.text.strip()
                        time.sleep(1)

                        header_correcta = "Ruta Cognitiva Correcta:"
                        header_distractores = "Análisis de Opciones No Válidas:"
                        idx_distractores = analisis_central.find(header_distractores)
                        
                        if idx_distractores == -1:
                            raise ValueError("No se encontró el separador 'Análisis de Opciones No Válidas' en la respuesta del paso 1.")

                        ruta_cognitiva = analisis_central[len(header_correcta):idx_distractores].strip()
                        analisis_distractores = analisis_central[idx_distractores:].strip()

                        # --- LLAMADA 2: SÍNTESIS DEL "QUÉ EVALÚA" ---
                        st.write(f"**Paso 2/3:** Sintetizando 'Qué Evalúa'...")
                        prompt_paso2 = construir_prompt_paso2_sintesis_que_evalua(analisis_central, fila)
                        response_paso2 = model.generate_content(prompt_paso2)
                        que_evalua = response_paso2.text.strip()
                        time.sleep(1)
                        
                        # --- LLAMADA 3: GENERACIÓN DE RECOMENDACIONES ---
                        st.write(f"**Paso 3/3:** Generando recomendaciones pedagógicas...")
                        prompt_paso3 = construir_prompt_paso3_recomendaciones(que_evalua, analisis_central, fila)
                        response_paso3 = model.generate_content(prompt_paso3)
                        recomendaciones = response_paso3.text.strip()
                        
                        titulo_avanzar = "RECOMENDACIÓN PARA AVANZAR"
                        idx_avanzar = recomendaciones.upper().find(titulo_avanzar)
                        
                        if idx_avanzar == -1:
                           raise ValueError("No se encontró el separador 'RECOMENDACIÓN PARA AVANZAR' en la respuesta del paso 3.")

                        fortalecer = recomendaciones[:idx_avanzar].strip()
                        avanzar = recomendaciones[idx_avanzar:].strip()

                        # --- GUARDAR TODO EN EL DATAFRAME ---
                        df.loc[i, "Que_Evalua"] = que_evalua
                        df.loc[i, "Justificacion_Correcta"] = ruta_cognitiva
                        df.loc[i, "Analisis_Distractores"] = analisis_distractores
                        df.loc[i, "Recomendacion_Fortalecer"] = fortalecer
                        df.loc[i, "Recomendacion_Avanzar"] = avanzar
                        st.success(f"Ítem {item_id} procesado con éxito.")

                    except Exception as e:
                        st.error(f"Ocurrió un error procesando el ítem {item_id}: {e}")
                        df.loc[i, "Que_Evalua"] = "ERROR EN PROCESAMIENTO"
                        df.loc[i, "Justificacion_Correcta"] = f"Error: {e}"
                        df.loc[i, "Analisis_Distractores"] = "ERROR EN PROCESAMIENTO"
                        df.loc[i, "Recomendacion_Fortalecer"] = "ERROR EN PROCESAMIENTO"
                        df.loc[i, "Recomendacion_Avanzar"] = "ERROR EN PROCESAMIENTO"
            
            progress_bar_main.progress(1.0, text="¡Proceso completado!")
            st.session_state.df_enriquecido = df
            st.balloons()
        else:
            st.error("No se pudo inicializar el modelo de IA. Verifica tu clave API.")


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
