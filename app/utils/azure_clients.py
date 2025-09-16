import os
from typing import Optional
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.search.documents import SearchClient
from langchain_openai import AzureChatOpenAI

from app.utils.logger import get_logger

logger = get_logger(__name__)

# Cache para los clientes (singleton pattern)
_openai_client: Optional[AzureChatOpenAI] = None
_search_client: Optional[SearchClient] = None
_doc_intelligence_client: Optional[DocumentAnalysisClient] = None


def get_openai_client() -> AzureChatOpenAI:
    """
    Obtener cliente de Azure OpenAI.
    Lee las credenciales del entorno.
    """
    global _openai_client
    if _openai_client is None:
        try:
            # Validamos que todas las variables necesarias existan
            endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            api_key = os.getenv("AZURE_OPENAI_API_KEY")
            deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
            api_version = os.getenv("OPENAI_API_VERSION")

            if not all([endpoint, api_key, deployment, api_version]):
                raise ValueError("Faltan variables de entorno para Azure OpenAI.")

            _openai_client = AzureChatOpenAI(
                azure_endpoint=endpoint,
                api_key=api_key,
                azure_deployment=deployment,
                api_version=api_version # <-- CORREGIDO: Se lee del .env
            )
            logger.info("Cliente de Azure OpenAI inicializado correctamente")
        except Exception as e:
            logger.error(f"Error inicializando cliente de Azure OpenAI: {str(e)}")
            raise
    return _openai_client


def get_search_client() -> SearchClient:
    """
    Obtener cliente de Azure Cognitive Search
    """
    global _search_client
    if _search_client is None:
        try:
            endpoint = os.getenv("AZURE_SEARCH_ENDPOINT") # <-- CORREGIDO: Nombre simplificado
            key = os.getenv("AZURE_SEARCH_ADMIN_KEY")
            index_name = os.getenv("AZURE_SEARCH_INDEX_NAME") # <-- CORREGIDO: Se valida que exista

            if not all([endpoint, key, index_name]):
                error_msg = "Revisa tus variables de entorno. Faltan valores para Azure Search (ENDPOINT, ADMIN_KEY, INDEX_NAME)."
                logger.error({"event": error_msg})
                raise ValueError(error_msg)

            credential = AzureKeyCredential(key)
            _search_client = SearchClient(
                endpoint=endpoint,
                index_name=index_name,
                credential=credential
            )
            logger.info("Cliente de Azure Cognitive Search inicializado correctamente")
        except Exception as e:
            logger.error(f"Error inicializando cliente de Azure Cognitive Search: {str(e)}")
            raise
    return _search_client


def get_doc_intelligence_client() -> DocumentAnalysisClient:
    """
    Obtener cliente de Azure Document Intelligence
    """
    global _doc_intelligence_client
    if _doc_intelligence_client is None:
        try:
            endpoint = os.getenv("AZURE_DOC_INTELLIGENCE_ENDPOINT")
            key = os.getenv("AZURE_DOC_INTELLIGENCE_KEY")

            if not all([endpoint, key]):
                raise ValueError("Configuraciones de Azure Document Intelligence incompletas.")

            credential = AzureKeyCredential(key)
            _doc_intelligence_client = DocumentAnalysisClient(
                endpoint=endpoint,
                credential=credential
            )
            logger.info("Cliente de Azure Document Intelligence inicializado correctamente")
        except Exception as e:
            logger.error(f"Error inicializando cliente de Azure Document Intelligence: {str(e)}")
            raise
    return _doc_intelligence_client