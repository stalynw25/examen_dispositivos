import os  # Para manejo de variables de entorno
import hashlib  # Para generar hash único del archivo PDF
from pathlib import Path  # Para verificar si existen archivos locales (hashes.txt)
import fitz  # Librería PyMuPDF: lectura y extracción de texto desde archivos PDF
from fastapi import FastAPI, UploadFile, File  # Para crear la API RESTful
from dotenv import load_dotenv  # Para cargar las variables de entorno desde el archivo .env

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings  # Modelo Gemini y embeddings de Google
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Para dividir el texto del PDF en fragmentos pequeños
from langchain_pinecone import PineconeVectorStore  # Conexión con el índice de vectores en Pinecone
from langchain.chains import RetrievalQA  # Cadena para preguntas aisladas con recuperación de contexto
from langchain_community.callbacks.manager import get_openai_callback  # Para medir tokens utilizados (funciona con modelos compatibles)
from pinecone import Pinecone, ServerlessSpec  # Cliente de Pinecone y configuración de región para índices vectoriales

# === Configuración inicial ===
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "asistente"
AI_MODEL = "gemini-2.5-flash"

# Inicializar Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
if INDEX_NAME not in [idx["name"] for idx in pc.list_indexes()]:
    pc.create_index(
        name=INDEX_NAME,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# Inicializar modelo y embeddings
llm = ChatGoogleGenerativeAI(model=AI_MODEL, convert_system_message_to_human=True)
embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embedding)

# === Funciones utilitarias ===
def hash_archivo(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()

def ya_existente(file_hash: str) -> bool:
    if not Path("hashes.txt").exists():
        return False
    return file_hash in Path("hashes.txt").read_text().splitlines()

def guardar_hash(file_hash: str):
    with open("hashes.txt", "a") as f:
        f.write(file_hash + "\n")

def leer_pdf_bytes(content: bytes) -> str:
    texto = ""
    with fitz.open(stream=content, filetype="pdf") as doc:
        for page in doc:
            texto += page.get_text()
    return texto

def leer_html_bytes(content: bytes) -> str:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(content, "html.parser")
    return soup.get_text(separator="\n")

def fragmentar(texto: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=400)
    return splitter.create_documents([texto])

def vectorizar_documento(texto: str):
    docs = fragmentar(texto)
    PineconeVectorStore.from_documents(docs, index_name=INDEX_NAME, embedding=embedding)

def crear_chain_qa():
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )

# === Función Código César +3 ===
def caesar_cipher(text: str, shift: int = 3) -> str:
    """Función para encriptar el texto con el código César +3."""
    result = []
    
    # Itera por cada carácter del texto
    for char in text:
        # Para las letras mayúsculas
        if char.isupper():
            result.append(chr((ord(char) - 65 + shift) % 26 + 65))
        # Para las letras minúsculas
        elif char.islower():
            result.append(chr((ord(char) - 97 + shift) % 26 + 97))
        # Los caracteres no alfabéticos (como espacio, puntuación, números) no se modifican
        else:
            result.append(char)
    
    return ''.join(result)

# Inicializar FastAPI
app = FastAPI()

@app.post("/uploadfile/")
async def upload_file(file: UploadFile = File(...)):
    content = await file.read()
    file_hash = hash_archivo(content)

    if ya_existente(file_hash):
        return {"message": "Este archivo ya fue vectorizado anteriormente."}
    else:
        if file.content_type == "application/pdf":
            texto = leer_pdf_bytes(content)
        elif file.content_type == "text/html":
            texto = leer_html_bytes(content)
        else:
            return {"message": "Formato de archivo no soportado"}

        if texto.strip():
            vectorizar_documento(texto)
            guardar_hash(file_hash)
            return {"message": "Documento vectorizado y almacenado correctamente."}
        else:
            return {"message": "No se pudo extraer texto del archivo."}

@app.get("/preguntar/")
async def ask_question(question: str):
    try:
        # Verifica si la cadena de preguntas ya está en la sesión
        if not hasattr(app.state, "qa_chain"):
            app.state.qa_chain = crear_chain_qa()

        # Imprime la pregunta que se está procesando
        print(f"Recibiendo la pregunta: {question}")

        # Realiza la consulta utilizando la cadena de preguntas
        with get_openai_callback() as cb:
            respuesta = app.state.qa_chain.run(question)

            # Encriptar la respuesta con el código César +3
            respuesta_encriptada = caesar_cipher(respuesta, shift=3)

            # Imprime la respuesta generada para ver el resultado en la consola del backend
            print(f"Respuesta generada (encriptada): {respuesta_encriptada}")

            # Retorna la respuesta encriptada y los tokens usados
            return {"respuesta": respuesta_encriptada, "tokens_usados": cb.total_tokens}
    
    except Exception as e:
        print(f"Error en la generación de respuesta: {str(e)}")  # Esto te ayudará a detectar el error
        return {"error": str(e)}
