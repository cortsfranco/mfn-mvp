"""
Agente de Inteligencia Artificial principal para mfn-mvp
Actualizado para usar el patrón RAG con Azure AI
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from app.config.settings import settings
from app.utils.azure_clients import get_openai_client, get_openai_embeddings_client
from app.utils.azure_helpers import search_helper
from app.utils.logger import get_logger

logger = get_logger(__name__)

class AIAgent:
    """Agente principal de IA que maneja la lógica de procesamiento con RAG"""
    
    def __init__(self):
        """Inicializar el agente de IA"""
        self.openai_client = get_openai_client()
        self.embeddings_client = get_openai_embeddings_client()
        
        logger.info("Agente de IA inicializado con RAG")
    
    async def process_request(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Procesar una petición del usuario usando IA con RAG
        
        Args:
            user_input: Texto de entrada del usuario
            context: Contexto adicional para la petición
            
        Returns:
            Respuesta procesada por la IA con RAG
        """
        try:
            start_time = datetime.utcnow()
            
            # 1. Generar embeddings de la consulta
            query_embeddings = await self.embeddings_client.aembed_query(user_input)
            
            # 2. Buscar documentos relevantes
            relevant_documents = await self._retrieve_relevant_documents(user_input)
            
            # 3. Construir contexto con documentos encontrados
            context_text = self._build_context_from_documents(relevant_documents)
            
            # 4. Generar respuesta usando RAG
            response = await self._generate_rag_response(user_input, context_text, context)
            
            end_time = datetime.utcnow()
            processing_time = (end_time - start_time).total_seconds()
            
            return {
                "success": True,
                "response": response,
                "documents_retrieved": len(relevant_documents),
                "processing_time_seconds": processing_time,
                "model": settings.GENERATION_MODEL,
                "embedding_model": settings.EMBEDDING_MODEL,
                "context_length": len(context_text)
            }
            
        except Exception as e:
            logger.error(f"Error procesando petición con RAG: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _retrieve_relevant_documents(self, query: str) -> List[Dict[str, Any]]:
        """
        Recuperar documentos relevantes usando Azure Search
        
        Args:
            query: Consulta del usuario
            
        Returns:
            Lista de documentos relevantes
        """
        try:
            # Buscar documentos en Azure Search
            documents = await search_helper.search_documents(
                query=query,
                top=settings.TOP_K_DOCUMENTS
            )
            
            logger.info(f"📚 Documentos recuperados: {len(documents)}")
            return documents
            
        except Exception as e:
            logger.error(f"Error recuperando documentos: {str(e)}")
            return []
    
    def _build_context_from_documents(self, documents: List[Dict[str, Any]]) -> str:
        """
        Construir contexto a partir de documentos recuperados
        
        Args:
            documents: Lista de documentos recuperados
            
        Returns:
            Texto de contexto formateado
        """
        if not documents:
            return ""
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            content = doc.get("content", "")
            source = doc.get("source", "Documento desconocido")
            
            context_parts.append(f"Documento {i} (Fuente: {source}):\n{content}\n")
        
        return "\n".join(context_parts)
    
    async def _generate_rag_response(self, user_input: str, context: str, additional_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Generar respuesta usando RAG
        
        Args:
            user_input: Consulta del usuario
            context: Contexto de documentos recuperados
            additional_context: Contexto adicional
            
        Returns:
            Respuesta generada
        """
        try:
            # Construir prompt para RAG
            prompt = self._build_rag_prompt(user_input, context, additional_context)
            
            # Generar respuesta
            response = await self.openai_client.ainvoke(prompt)
            
            return response.content
            
        except Exception as e:
            logger.error(f"Error generando respuesta RAG: {str(e)}")
            raise
    
    def _build_rag_prompt(self, user_input: str, context: str, additional_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Construir prompt para RAG
        
        Args:
            user_input: Consulta del usuario
            context: Contexto de documentos
            additional_context: Contexto adicional
            
        Returns:
            Prompt formateado para RAG
        """
        base_prompt = f"""
        Eres un asistente de IA inteligente para la aplicación mfn-mvp.
        
        Basándote ÚNICAMENTE en el siguiente contexto, responde la pregunta del usuario.
        Si la información no está en el contexto, indica que no tienes esa información.
        
        Contexto:
        {context}
        
        Pregunta del usuario: {user_input}
        
        Respuesta:
        """
        
        if additional_context:
            context_str = "\n".join([f"{k}: {v}" for k, v in additional_context.items()])
            base_prompt += f"\n\nInformación adicional:\n{context_str}"
        
        return base_prompt
    
    async def process_document_upload(self, file_path: str) -> Dict[str, Any]:
        """
        Procesar la subida de un documento usando el pipeline RAG
        
        Args:
            file_path: Ruta al archivo a procesar
            
        Returns:
            Resultado del procesamiento
        """
        try:
            from app.core.rag_pipeline import rag_pipeline
            
            logger.info(f"📄 Procesando documento: {file_path}")
            
            # Usar el pipeline RAG para procesar el documento
            result = await rag_pipeline.process_and_index_document(file_path)
            
            return result
            
        except Exception as e:
            logger.error(f"Error procesando documento: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "file_path": file_path
            }
    
    async def search_documents(self, query: str, top: int = 5) -> Dict[str, Any]:
        """
        Buscar documentos en el índice
        
        Args:
            query: Consulta de búsqueda
            top: Número máximo de resultados
            
        Returns:
            Resultados de la búsqueda
        """
        try:
            documents = await search_helper.search_documents(query=query, top=top)
            
            return {
                "success": True,
                "query": query,
                "documents": documents,
                "total_found": len(documents)
            }
            
        except Exception as e:
            logger.error(f"Error buscando documentos: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "query": query
            }
    
    async def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analizar el sentimiento de un texto
        
        Args:
            text: Texto a analizar
            
        Returns:
            Análisis de sentimiento
        """
        try:
            prompt = f"""
            Analiza el sentimiento del siguiente texto. 
            Responde únicamente con: POSITIVO, NEGATIVO, o NEUTRAL.
            
            Texto: {text}
            
            Sentimiento:
            """
            
            response = await self.openai_client.ainvoke(prompt)
            sentiment = response.content.strip().upper()
            
            return {
                "success": True,
                "sentiment": sentiment,
                "text": text,
                "confidence": "high" if sentiment in ["POSITIVO", "NEGATIVO", "NEUTRAL"] else "low"
            }
            
        except Exception as e:
            logger.error(f"Error analizando sentimiento: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "text": text
            }
    
    async def get_system_status(self) -> Dict[str, Any]:
        """
        Obtener el estado del sistema
        
        Returns:
            Estado del sistema
        """
        try:
            # Verificar conectividad con servicios
            from app.utils.azure_clients import validate_all_clients
            
            clients_valid = validate_all_clients()
            
            return {
                "success": True,
                "status": "healthy" if clients_valid else "unhealthy",
                "timestamp": datetime.utcnow().isoformat(),
                "model": settings.GENERATION_MODEL,
                "embedding_model": settings.EMBEDDING_MODEL,
                "clients_valid": clients_valid
            }
            
        except Exception as e:
            logger.error(f"Error obteniendo estado del sistema: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "status": "error"
            }

# Instancia global del agente
ai_agent = AIAgent()
