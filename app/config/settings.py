"""
Configuraciones y variables de entorno para mfn-mvp
Configuración específica para servicios de Azure AI con patrón RAG
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Cargar variables de entorno desde .env
load_dotenv()

class Settings:
    """
    Clase para manejar todas las configuraciones de la aplicación
    Incluye configuraciones para servicios de Azure AI necesarios para RAG
    """
    
    # ============================================================================
    # CONFIGURACIONES DE AZURE OPENAI
    # ============================================================================
    AZURE_OPENAI_ENDPOINT: Optional[str] = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_API_KEY: Optional[str] = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_DEPLOYMENT_NAME: Optional[str] = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    
    # ============================================================================
    # CONFIGURACIONES DE AZURE COGNITIVE SEARCH
    # ============================================================================
    AZURE_SEARCH_ENDPOINT: Optional[str] = os.getenv("AZURE_SEARCH_ENDPOINT")
    AZURE_SEARCH_API_KEY: Optional[str] = os.getenv("AZURE_SEARCH_API_KEY")
    AZURE_SEARCH_INDEX_NAME: Optional[str] = os.getenv("AZURE_SEARCH_INDEX_NAME")
    
    # ============================================================================
    # CONFIGURACIONES DE AZURE DOCUMENT INTELLIGENCE
    # ============================================================================
    AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT: Optional[str] = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
    AZURE_DOCUMENT_INTELLIGENCE_KEY: Optional[str] = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")
    
    # ============================================================================
    # CONFIGURACIONES DE AZURE STORAGE
    # ============================================================================
    AZURE_STORAGE_CONNECTION_STRING: Optional[str] = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    AZURE_STORAGE_CONTAINER_NAME: Optional[str] = os.getenv("AZURE_STORAGE_CONTAINER_NAME")
    
    # ============================================================================
    # CONFIGURACIONES DE LA APLICACIÓN
    # ============================================================================
    APP_NAME: str = "mfn-mvp"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    # ============================================================================
    # CONFIGURACIONES DE IA Y MODELOS
    # ============================================================================
    # Modelo de embeddings para vectorización de documentos
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
    
    # Modelo de generación para respuestas
    GENERATION_MODEL: str = os.getenv("GENERATION_MODEL", "gpt-4")
    
    # Configuraciones de tokens
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "1000"))
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.7"))
    
    # ============================================================================
    # CONFIGURACIONES DE RAG
    # ============================================================================
    # Número de documentos a recuperar para el contexto
    TOP_K_DOCUMENTS: int = int(os.getenv("TOP_K_DOCUMENTS", "5"))
    
    # Umbral de similitud para filtrar documentos
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))
    
    # Tamaño del chunk para dividir documentos
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    # ============================================================================
    # CONFIGURACIONES DE LOGGING
    # ============================================================================
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    @classmethod
    def validate(cls) -> bool:
        """
        Validar que todas las configuraciones requeridas estén presentes
        
        Returns:
            bool: True si todas las configuraciones están presentes
            
        Raises:
            ValueError: Si faltan configuraciones requeridas
        """
        # Configuraciones críticas para Azure OpenAI
        azure_openai_required = [
            "AZURE_OPENAI_ENDPOINT",
            "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_DEPLOYMENT_NAME"
        ]
        
        # Configuraciones críticas para Azure Search
        azure_search_required = [
            "AZURE_SEARCH_ENDPOINT",
            "AZURE_SEARCH_API_KEY",
            "AZURE_SEARCH_INDEX_NAME"
        ]
        
        # Configuraciones críticas para Azure Storage
        azure_storage_required = [
            "AZURE_STORAGE_CONNECTION_STRING",
            "AZURE_STORAGE_CONTAINER_NAME"
        ]
        
        # Configuraciones críticas para Document Intelligence
        document_intelligence_required = [
            "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT",
            "AZURE_DOCUMENT_INTELLIGENCE_KEY"
        ]
        
        # Combinar todas las configuraciones requeridas
        all_required = (
            azure_openai_required + 
            azure_search_required + 
            azure_storage_required + 
            document_intelligence_required
        )
        
        missing_vars = []
        for var in all_required:
            if not getattr(cls, var):
                missing_vars.append(var)
        
        if missing_vars:
            raise ValueError(
                f"Variables de entorno faltantes para servicios de Azure AI: "
                f"{', '.join(missing_vars)}\n"
                f"Por favor, configura estas variables en tu archivo .env"
            )
        
        return True
    
    @classmethod
    def get_azure_openai_config(cls) -> dict:
        """
        Obtener configuración para Azure OpenAI
        
        Returns:
            dict: Configuración de Azure OpenAI
        """
        return {
            "endpoint": cls.AZURE_OPENAI_ENDPOINT,
            "api_key": cls.AZURE_OPENAI_API_KEY,
            "deployment_name": cls.AZURE_OPENAI_DEPLOYMENT_NAME,
            "api_version": "2024-02-15-preview"
        }
    
    @classmethod
    def get_azure_search_config(cls) -> dict:
        """
        Obtener configuración para Azure Cognitive Search
        
        Returns:
            dict: Configuración de Azure Search
        """
        return {
            "endpoint": cls.AZURE_SEARCH_ENDPOINT,
            "api_key": cls.AZURE_SEARCH_API_KEY,
            "index_name": cls.AZURE_SEARCH_INDEX_NAME
        }
    
    @classmethod
    def get_document_intelligence_config(cls) -> dict:
        """
        Obtener configuración para Azure Document Intelligence
        
        Returns:
            dict: Configuración de Document Intelligence
        """
        return {
            "endpoint": cls.AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT,
            "api_key": cls.AZURE_DOCUMENT_INTELLIGENCE_KEY
        }
    
    @classmethod
    def get_storage_config(cls) -> dict:
        """
        Obtener configuración para Azure Storage
        
        Returns:
            dict: Configuración de Azure Storage
        """
        return {
            "connection_string": cls.AZURE_STORAGE_CONNECTION_STRING,
            "container_name": cls.AZURE_STORAGE_CONTAINER_NAME
        }
    
    @classmethod
    def get_rag_config(cls) -> dict:
        """
        Obtener configuración para el patrón RAG
        
        Returns:
            dict: Configuración de RAG
        """
        return {
            "top_k": cls.TOP_K_DOCUMENTS,
            "similarity_threshold": cls.SIMILARITY_THRESHOLD,
            "chunk_size": cls.CHUNK_SIZE,
            "chunk_overlap": cls.CHUNK_OVERLAP,
            "embedding_model": cls.EMBEDDING_MODEL,
            "generation_model": cls.GENERATION_MODEL,
            "max_tokens": cls.MAX_TOKENS,
            "temperature": cls.TEMPERATURE
        }

# Instancia global de configuraciones
settings = Settings()
