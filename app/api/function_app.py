"""
Azure Function App para mfn-mvp
Endpoint de chat usando FastAPI y Azure Functions con RAG
"""

import azure.functions as func
import json
import logging
import asyncio
from typing import Dict, Any
from pydantic import BaseModel, Field

from app.utils.logger import get_logger
from app.core.conversation_agent import ConversationAgent

logger = get_logger(__name__)

# Modelo Pydantic para la entrada de datos
class ChatRequest(BaseModel):
    """
    Modelo de entrada para las peticiones de chat
    """
    question: str = Field(..., description="Pregunta del usuario", min_length=1, max_length=1000)

class ChatResponse(BaseModel):
    """
    Modelo de respuesta para las peticiones de chat
    """
    answer: str = Field(..., description="Respuesta del agente")
    success: bool = Field(..., description="Indica si la operación fue exitosa")
    processing_time_seconds: float = Field(..., description="Tiempo de procesamiento en segundos")
    documents_retrieved: int = Field(..., description="Número de documentos recuperados")

# Instancia global del agente conversacional
# Esta es una buena práctica para optimizar el rendimiento y costos en entornos serverless:
# 1. Evita la inicialización repetida del agente en cada petición
# 2. Reduce el tiempo de respuesta (cold start)
# 3. Minimiza el uso de memoria y CPU
# 4. Permite reutilizar conexiones a servicios externos (Azure OpenAI, Azure Search)
# 5. Reduce costos al mantener el estado entre peticiones
conversation_agent = ConversationAgent()

def chat(req: func.HttpRequest) -> func.HttpResponse:
    """
    HTTP Trigger para el endpoint de chat
    Acepta peticiones POST en la ruta '/api/chat'
    """
    try:
        # Verificar que sea una petición POST
        if req.method != "POST":
            return func.HttpResponse(
                json.dumps({"error": "Método no permitido. Solo se aceptan peticiones POST"}),
                status_code=405,
                mimetype="application/json"
            )
        
        # Obtener el cuerpo de la petición
        body = req.get_body().decode('utf-8')
        
        # Logging de la petición entrante
        logger.info(f"📨 Petición de chat recibida: {body[:200]}...")
        
        # Parsear y validar el JSON usando Pydantic
        try:
            request_data = json.loads(body)
            chat_request = ChatRequest(**request_data)
        except json.JSONDecodeError as e:
            logger.error(f"❌ Error parseando JSON: {str(e)}")
            return func.HttpResponse(
                json.dumps({"error": "JSON inválido en el cuerpo de la petición"}),
                status_code=400,
                mimetype="application/json"
            )
        except Exception as e:
            logger.error(f"❌ Error validando datos: {str(e)}")
            return func.HttpResponse(
                json.dumps({"error": f"Datos de entrada inválidos: {str(e)}"}),
                status_code=400,
                mimetype="application/json"
            )
        
        # Llamar al método ask() del agente conversacional
        logger.info(f"🤔 Procesando pregunta: {chat_request.question}")
        
        # Ejecutar la pregunta de forma asíncrona
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            agent_response = loop.run_until_complete(conversation_agent.ask(chat_request.question))
        finally:
            loop.close()
        
        # Logging de la respuesta del agente
        logger.info(f"✅ Respuesta del agente generada en {agent_response.get('processing_time_seconds', 0):.2f}s")
        
        # Construir la respuesta usando el modelo Pydantic
        if agent_response.get("success", False):
            chat_response = ChatResponse(
                answer=agent_response.get("answer", "No se pudo generar una respuesta"),
                success=True,
                processing_time_seconds=agent_response.get("processing_time_seconds", 0.0),
                documents_retrieved=agent_response.get("documents_retrieved", 0)
            )
            
            # Logging de respuesta exitosa
            logger.info(f"📤 Respuesta enviada: {len(chat_response.answer)} caracteres")
            
            return func.HttpResponse(
                json.dumps(chat_response.dict(), indent=2, ensure_ascii=False),
                status_code=200,
                mimetype="application/json"
            )
        else:
            # Manejar error del agente
            error_message = agent_response.get("error", "Error desconocido en el agente")
            logger.error(f"❌ Error del agente: {error_message}")
            
            return func.HttpResponse(
                json.dumps({
                    "error": error_message,
                    "answer": "Lo siento, no pude procesar tu pregunta en este momento.",
                    "success": False,
                    "processing_time_seconds": agent_response.get("processing_time_seconds", 0.0),
                    "documents_retrieved": 0
                }),
                status_code=500,
                mimetype="application/json"
            )
            
    except Exception as e:
        # Manejar errores inesperados
        logger.error(f"❌ Error inesperado en el endpoint de chat: {str(e)}")
        return func.HttpResponse(
            json.dumps({
                "error": "Error interno del servidor",
                "answer": "Lo siento, ocurrió un error inesperado.",
                "success": False,
                "processing_time_seconds": 0.0,
                "documents_retrieved": 0
            }),
            status_code=500,
            mimetype="application/json"
        )

# Función principal para compatibilidad con Azure Functions
def main(req: func.HttpRequest) -> func.HttpResponse:
    """
    Función principal que redirige a los endpoints específicos
    """
    try:
        # Obtener la ruta de la petición
        route = req.route_params.get("route", "")
        
        # Redirigir al endpoint de chat
        if route == "chat" or req.url.endswith("/api/chat"):
            return chat(req)
        
        # Endpoint por defecto con información de la API
        else:
            response_data = {
                "status": "success",
                "message": "API de IA mfn-mvp funcionando correctamente",
                "endpoints": {
                    "chat": "POST /api/chat con body JSON: {\"question\": \"<pregunta>\"}",
                    "health": "GET /?action=health",
                    "status": "GET /?action=status"
                },
                "example_request": {
                    "question": "¿Qué es la inteligencia artificial?"
                },
                "example_response": {
                    "answer": "La inteligencia artificial es...",
                    "success": True,
                    "processing_time_seconds": 1.5,
                    "documents_retrieved": 3
                }
            }
            return func.HttpResponse(
                json.dumps(response_data, indent=2, ensure_ascii=False),
                status_code=200,
                mimetype="application/json"
            )
            
    except Exception as e:
        logger.error(f"Error en función principal: {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=500,
            mimetype="application/json"
        )

# Endpoints adicionales para compatibilidad
def health_check() -> func.HttpResponse:
    """
    Endpoint de verificación de salud
    """
    try:
        response_data = {
            "status": "healthy",
            "service": "mfn-mvp",
            "version": "1.0.0",
            "endpoint": "/api/chat",
            "timestamp": "2024-01-01T00:00:00Z"
        }
        return func.HttpResponse(
            json.dumps(response_data),
            status_code=200,
            mimetype="application/json"
        )
    except Exception as e:
        logger.error(f"Error en health check: {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=500,
            mimetype="application/json"
        )

def system_status() -> func.HttpResponse:
    """
    Endpoint de estado del sistema
    """
    try:
        # Obtener información del agente
        agent_info = conversation_agent.get_agent_info()
        
        response_data = {
            "status": "operational",
            "service": "mfn-mvp",
            "agent_info": agent_info,
            "endpoints": {
                "chat": "/api/chat",
                "health": "/?action=health"
            }
        }
        
        return func.HttpResponse(
            json.dumps(response_data, indent=2),
            status_code=200,
            mimetype="application/json"
        )
    except Exception as e:
        logger.error(f"Error obteniendo estado del sistema: {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=500,
            mimetype="application/json"
        )
