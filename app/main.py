"""
API web simple usando FastAPI
Endpoints para procesar facturas y chatear con el agente.
"""
import os
import tempfile
from enum import Enum
from typing import Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

# Importamos los dos componentes principales de nuestra lógica
from app.core.rag_pipeline import invoice_processor
from app.core.graph import agent_graph
from app.utils.logger import get_logger

# Cargar variables de entorno del archivo .env
load_dotenv()

logger = get_logger(__name__)

# Crear instancia de la aplicación FastAPI
app = FastAPI(
    title="API de Agente Contable",
    description="API para procesar facturas y responder preguntas sobre ellas.",
    version="2.0.0"
)

# La clase InvoiceType se elimina porque ya no la necesitamos.

# Definimos las opciones posibles para los campos del formulario
class PartnerName(str, Enum):
    hernan = "HERNAN"
    joni = "JONI"
    maxi = "MAXI"
    leo = "LEO"

@app.get("/")
async def root():
    """Endpoint raíz para verificar que la API está funcionando"""
    return {"message": "API de Agente Contable está funcionando"}

@app.post("/process-invoice/")
async def process_invoice(
    # El parámetro 'invoice_type' se ha eliminado de aquí.
    file: UploadFile = File(...),
    partner_name: PartnerName = Form(...)
) -> JSONResponse:
    """
    Endpoint para procesar y almacenar una nueva factura.
    """
    temp_file_path = None
    try:
        logger.info(f"📄 Recibiendo archivo: {file.filename}")
        file_extension = os.path.splitext(file.filename.lower())[1]
        temp_fd, temp_file_path = tempfile.mkstemp(suffix=file_extension)
        
        with os.fdopen(temp_fd, 'wb') as temp_file:
            content = await file.read()
            temp_file.write(content)

        # La llamada a la función ya no pasa 'invoice_type'.
        result = await invoice_processor.process_and_upload_invoice(
            file_path=temp_file_path,
            partner_name=partner_name.value
        )
        response_data = { "success": result.get("success", False), "message": "Procesamiento de factura completado", "filename": file.filename, "processing_result": result }
        return JSONResponse(content=response_data, status_code=200)

    except Exception as e:
        logger.error(f"❌ Error en el endpoint /process-invoice/: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {str(e)}")
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

# --- ENDPOINT DE CHAT DEFINITIVO ---
@app.post("/chat/")
async def chat_with_agent(question: str = Form(...)):
    """
    Recibe una pregunta en lenguaje natural, la procesa y devuelve una respuesta.
    """
    try:
        logger.info(f"💬 Nueva pregunta para el agente: {question}")
        # Usamos el método run() de nuestro grafo, que ejecuta el flujo completo
        final_state = await agent_graph.run(question)
        
        return JSONResponse(content={
            "success": True,
            "question": question,
            "answer": final_state.get("final_answer"),
            "trace": final_state # Opcional: devuelve el estado completo para depuración
        }, status_code=200)

    except Exception as e:
        logger.error(f"❌ Error en el endpoint /chat/: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {str(e)}")