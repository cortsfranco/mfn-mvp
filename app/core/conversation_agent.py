"""
Agente Conversacional con RAG usando LangChain
Implementa un agente que combina Azure AI Search con Azure OpenAI para respuestas inteligentes
"""

import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime

from langchain.schema import Document
from langchain.retrievers import AzureCognitiveSearchRetriever
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_openai import AzureChatOpenAI

from app.config.settings import settings
from app.utils.azure_clients import get_openai_client, get_search_client
from app.utils.logger import get_logger

logger = get_logger(__name__)

class ConversationAgent:
    """
    Agente conversacional que combina Azure AI Search con Azure OpenAI
    para proporcionar respuestas inteligentes basadas en documentos indexados
    """
    
    def __init__(self):
        """Inicializar el agente conversacional"""
        try:
            # Configurar cliente de Azure OpenAI
            self.openai_client = get_openai_client()
            
            # Configurar retriever de Azure AI Search
            self.retriever = self._setup_retriever()
            
            # Configurar prompt template
            self.prompt_template = self._setup_prompt_template()
            
            # Configurar cadena de RAG
            self.qa_chain = self._setup_qa_chain()
            
            logger.info("Agente conversacional inicializado correctamente")
            
        except Exception as e:
            logger.error(f"Error inicializando agente conversacional: {str(e)}")
            raise
    
    def _setup_retriever(self) -> AzureCognitiveSearchRetriever:
        """
        Configurar el retriever de Azure AI Search
        
        Returns:
            AzureCognitiveSearchRetriever configurado
        """
        try:
            # Configurar retriever
            retriever = AzureCognitiveSearchRetriever(
                service_name=settings.AZURE_SEARCH_ENDPOINT.split('.')[0],
                index_name=settings.AZURE_SEARCH_INDEX_NAME,
                api_key=settings.AZURE_SEARCH_API_KEY,
                content_key="content",
                top_k=settings.TOP_K_DOCUMENTS,
                semantic_configuration_name="default" if hasattr(settings, 'AZURE_SEARCH_SEMANTIC_CONFIG') else None
            )
            
            logger.info("Retriever de Azure AI Search configurado correctamente")
            return retriever
            
        except Exception as e:
            logger.error(f"Error configurando retriever: {str(e)}")
            raise
    
    def _setup_prompt_template(self) -> PromptTemplate:
        """
        Configurar el template de prompt para RAG
        
        Returns:
            PromptTemplate configurado
        """
        try:
            template = """
            Eres un asistente de IA inteligente y útil para la aplicación mfn-mvp.
            
            Basándote ÚNICAMENTE en el siguiente contexto, responde la pregunta del usuario.
            Si la información no está en el contexto, indica claramente que no tienes esa información.
            
            Contexto:
            {context}
            
            Pregunta: {question}
            
            Respuesta:
            """
            
            prompt_template = PromptTemplate(
                template=template,
                input_variables=["context", "question"]
            )
            
            logger.info("Template de prompt configurado correctamente")
            return prompt_template
            
        except Exception as e:
            logger.error(f"Error configurando prompt template: {str(e)}")
            raise
    
    def _setup_qa_chain(self) -> RetrievalQA:
        """
        Configurar la cadena de RAG
        
        Returns:
            RetrievalQA configurado
        """
        try:
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.openai_client,
                chain_type="stuff",
                retriever=self.retriever,
                chain_type_kwargs={
                    "prompt": self.prompt_template,
                    "verbose": settings.DEBUG
                },
                return_source_documents=True
            )
            
            logger.info("Cadena de RAG configurada correctamente")
            return qa_chain
            
        except Exception as e:
            logger.error(f"Error configurando cadena de RAG: {str(e)}")
            raise
    
    async def ask(self, question: str) -> Dict[str, Any]:
        """
        Procesar una pregunta del usuario usando RAG
        
        Args:
            question: Pregunta del usuario
            
        Returns:
            Respuesta con información detallada del proceso RAG
        """
        try:
            start_time = datetime.utcnow()
            
            logger.info(f"🤔 Procesando pregunta: {question}")
            
            # 1. Recuperar documentos relevantes
            relevant_docs = await self._retrieve_documents(question)
            
            # 2. Generar respuesta usando la cadena de RAG
            response = await self._generate_response(question, relevant_docs)
            
            end_time = datetime.utcnow()
            processing_time = (end_time - start_time).total_seconds()
            
            # 3. Construir respuesta final
            result = {
                "success": True,
                "question": question,
                "answer": response["answer"],
                "source_documents": self._format_source_documents(relevant_docs),
                "processing_time_seconds": processing_time,
                "documents_retrieved": len(relevant_docs),
                "model": settings.GENERATION_MODEL,
                "retrieval_strategy": "Azure AI Search + RAG"
            }
            
            logger.info(f"✅ Respuesta generada en {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error procesando pregunta: {str(e)}")
            return {
                "success": False,
                "question": question,
                "error": str(e),
                "answer": "Lo siento, no pude procesar tu pregunta en este momento."
            }
    
    async def _retrieve_documents(self, question: str) -> List[Document]:
        """
        Recuperar documentos relevantes usando el retriever
        
        Args:
            question: Pregunta del usuario
            
        Returns:
            Lista de documentos relevantes
        """
        try:
            # Usar el retriever para obtener documentos relevantes
            documents = await asyncio.to_thread(
                self.retriever.get_relevant_documents,
                question
            )
            
            logger.info(f"📚 Documentos recuperados: {len(documents)}")
            
            # Logging de documentos recuperados
            for i, doc in enumerate(documents[:3]):  # Mostrar solo los primeros 3
                source = doc.metadata.get("source", "Desconocido")
                logger.info(f"   Documento {i+1}: {source} ({len(doc.page_content)} caracteres)")
            
            return documents
            
        except Exception as e:
            logger.error(f"Error recuperando documentos: {str(e)}")
            return []
    
    async def _generate_response(self, question: str, documents: List[Document]) -> Dict[str, Any]:
        """
        Generar respuesta usando la cadena de RAG
        
        Args:
            question: Pregunta del usuario
            documents: Documentos relevantes
            
        Returns:
            Respuesta generada
        """
        try:
            # Usar la cadena de RAG para generar respuesta
            response = await asyncio.to_thread(
                self.qa_chain,
                {"query": question}
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error generando respuesta: {str(e)}")
            raise
    
    def _format_source_documents(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """
        Formatear documentos fuente para la respuesta
        
        Args:
            documents: Lista de documentos de LangChain
            
        Returns:
            Lista de documentos formateados
        """
        formatted_docs = []
        
        for i, doc in enumerate(documents):
            formatted_doc = {
                "id": i + 1,
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "source": doc.metadata.get("source", "Desconocido"),
                "chunk_id": doc.metadata.get("chunk_id", "N/A"),
                "confidence": doc.metadata.get("confidence", "N/A"),
                "full_content_length": len(doc.page_content)
            }
            formatted_docs.append(formatted_doc)
        
        return formatted_docs
    
    async def ask_with_custom_context(self, question: str, additional_context: str = "") -> Dict[str, Any]:
        """
        Procesar una pregunta con contexto adicional
        
        Args:
            question: Pregunta del usuario
            additional_context: Contexto adicional para incluir en la respuesta
            
        Returns:
            Respuesta con contexto adicional
        """
        try:
            start_time = datetime.utcnow()
            
            logger.info(f"🤔 Procesando pregunta con contexto adicional: {question}")
            
            # 1. Recuperar documentos relevantes
            relevant_docs = await self._retrieve_documents(question)
            
            # 2. Construir prompt con contexto adicional
            enhanced_prompt = self._build_enhanced_prompt(question, relevant_docs, additional_context)
            
            # 3. Generar respuesta
            response = await self.openai_client.ainvoke(enhanced_prompt)
            
            end_time = datetime.utcnow()
            processing_time = (end_time - start_time).total_seconds()
            
            result = {
                "success": True,
                "question": question,
                "answer": response.content,
                "source_documents": self._format_source_documents(relevant_docs),
                "additional_context": additional_context,
                "processing_time_seconds": processing_time,
                "documents_retrieved": len(relevant_docs),
                "model": settings.GENERATION_MODEL,
                "retrieval_strategy": "Azure AI Search + RAG + Contexto Adicional"
            }
            
            logger.info(f"✅ Respuesta con contexto adicional generada en {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error procesando pregunta con contexto: {str(e)}")
            return {
                "success": False,
                "question": question,
                "error": str(e),
                "answer": "Lo siento, no pude procesar tu pregunta en este momento."
            }
    
    def _build_enhanced_prompt(self, question: str, documents: List[Document], additional_context: str) -> str:
        """
        Construir prompt mejorado con contexto adicional
        
        Args:
            question: Pregunta del usuario
            documents: Documentos relevantes
            additional_context: Contexto adicional
            
        Returns:
            Prompt mejorado
        """
        # Construir contexto de documentos
        context_parts = []
        for i, doc in enumerate(documents, 1):
            context_parts.append(f"Documento {i} (Fuente: {doc.metadata.get('source', 'Desconocido')}):\n{doc.page_content}\n")
        
        context = "\n".join(context_parts)
        
        # Construir prompt completo
        prompt = f"""
        Eres un asistente de IA inteligente y útil para la aplicación mfn-mvp.
        
        Basándote en el siguiente contexto y la información adicional proporcionada, 
        responde la pregunta del usuario de manera completa y precisa.
        
        Contexto de documentos:
        {context}
        
        Información adicional:
        {additional_context}
        
        Pregunta: {question}
        
        Respuesta:
        """
        
        return prompt
    
    async def get_conversation_history(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Obtener historial de conversación (placeholder para implementación futura)
        
        Args:
            session_id: ID de la sesión de conversación
            
        Returns:
            Historial de conversación
        """
        # TODO: Implementar almacenamiento de historial de conversación
        logger.info(f"📝 Obteniendo historial para sesión: {session_id}")
        
        return [
            {
                "session_id": session_id,
                "message": "Función de historial no implementada aún",
                "timestamp": datetime.utcnow().isoformat()
            }
        ]
    
    async def validate_agent(self) -> bool:
        """
        Validar que el agente está funcionando correctamente
        
        Returns:
            True si el agente está funcionando, False en caso contrario
        """
        try:
            logger.info("🔍 Validando agente conversacional...")
            
            # Validar cliente de OpenAI
            if not self.openai_client:
                raise ValueError("Cliente de OpenAI no disponible")
            
            # Validar retriever
            if not self.retriever:
                raise ValueError("Retriever no disponible")
            
            # Validar cadena de RAG
            if not self.qa_chain:
                raise ValueError("Cadena de RAG no disponible")
            
            # Probar con una pregunta simple
            test_question = "¿Estás funcionando correctamente?"
            test_result = await self.ask(test_question)
            
            if test_result["success"]:
                logger.info("✅ Agente conversacional validado correctamente")
                return True
            else:
                logger.error("❌ Agente conversacional falló en la validación")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error validando agente: {str(e)}")
            return False
    
    def get_agent_info(self) -> Dict[str, Any]:
        """
        Obtener información del agente
        
        Returns:
            Información del agente
        """
        return {
            "agent_type": "ConversationAgent",
            "model": settings.GENERATION_MODEL,
            "retriever_type": "AzureCognitiveSearchRetriever",
            "search_index": settings.AZURE_SEARCH_INDEX_NAME,
            "top_k_documents": settings.TOP_K_DOCUMENTS,
            "chunk_size": settings.CHUNK_SIZE,
            "chunk_overlap": settings.CHUNK_OVERLAP,
            "temperature": settings.TEMPERATURE,
            "max_tokens": settings.MAX_TOKENS,
            "created_at": datetime.utcnow().isoformat()
        }

# Instancia global del agente conversacional
conversation_agent = ConversationAgent()
