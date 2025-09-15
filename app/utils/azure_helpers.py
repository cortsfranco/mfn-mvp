"""
Utilidades para trabajar con servicios de Azure
Actualizado para usar los nuevos clientes de Azure AI
"""

import logging
from typing import Optional, Dict, Any, List
from azure.storage.blob import BlobServiceClient, ContainerClient
from azure.ai.search import SearchClient
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential

from app.config.settings import settings
from app.utils.logger import get_logger
from app.utils.azure_clients import (
    get_blob_service_client,
    get_search_client,
    get_doc_intelligence_client,
    get_blob_container_client
)

logger = get_logger(__name__)

class AzureStorageHelper:
    """Helper para trabajar con Azure Storage usando los nuevos clientes"""
    
    def __init__(self):
        """Inicializar el helper de Azure Storage"""
        try:
            self.client = get_blob_service_client()
            self.container_name = settings.AZURE_STORAGE_CONTAINER_NAME
            logger.info("Azure Storage Helper inicializado correctamente")
        except Exception as e:
            self.client = None
            logger.error(f"Error inicializando Azure Storage Helper: {str(e)}")
    
    async def upload_blob(self, blob_name: str, data: bytes, container_name: Optional[str] = None) -> bool:
        """
        Subir un blob a Azure Storage
        
        Args:
            blob_name: Nombre del blob
            data: Datos a subir
            container_name: Nombre del contenedor (opcional, usa el de configuración por defecto)
            
        Returns:
            True si se subió correctamente, False en caso contrario
        """
        try:
            if not self.client:
                return False
            
            container_name = container_name or self.container_name
            container_client = self.client.get_container_client(container_name)
            blob_client = container_client.get_blob_client(blob_name)
            
            blob_client.upload_blob(data, overwrite=True)
            logger.info(f"Blob {blob_name} subido exitosamente al contenedor {container_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error subiendo blob {blob_name}: {str(e)}")
            return False
    
    async def download_blob(self, blob_name: str, container_name: Optional[str] = None) -> Optional[bytes]:
        """
        Descargar un blob de Azure Storage
        
        Args:
            blob_name: Nombre del blob
            container_name: Nombre del contenedor (opcional, usa el de configuración por defecto)
            
        Returns:
            Datos del blob o None si hay error
        """
        try:
            if not self.client:
                return None
            
            container_name = container_name or self.container_name
            container_client = self.client.get_container_client(container_name)
            blob_client = container_client.get_blob_client(blob_name)
            
            download_stream = blob_client.download_blob()
            return download_stream.readall()
            
        except Exception as e:
            logger.error(f"Error descargando blob {blob_name}: {str(e)}")
            return None
    
    async def list_blobs(self, container_name: Optional[str] = None, prefix: Optional[str] = None) -> List[str]:
        """
        Listar blobs en un contenedor
        
        Args:
            container_name: Nombre del contenedor (opcional, usa el de configuración por defecto)
            prefix: Prefijo para filtrar blobs
            
        Returns:
            Lista de nombres de blobs
        """
        try:
            if not self.client:
                return []
            
            container_name = container_name or self.container_name
            container_client = self.client.get_container_client(container_name)
            
            blobs = []
            for blob in container_client.list_blobs(name_starts_with=prefix):
                blobs.append(blob.name)
            
            return blobs
            
        except Exception as e:
            logger.error(f"Error listando blobs: {str(e)}")
            return []

class AzureSearchHelper:
    """Helper para trabajar con Azure Cognitive Search"""
    
    def __init__(self):
        """Inicializar el helper de Azure Search"""
        try:
            self.client = get_search_client()
            logger.info("Azure Search Helper inicializado correctamente")
        except Exception as e:
            self.client = None
            logger.error(f"Error inicializando Azure Search Helper: {str(e)}")
    
    async def search_documents(self, query: str, top: int = 5, filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Buscar documentos en Azure Cognitive Search
        
        Args:
            query: Consulta de búsqueda
            top: Número máximo de resultados
            filter: Filtro opcional para la búsqueda
            
        Returns:
            Lista de documentos encontrados
        """
        try:
            if not self.client:
                return []
            
            results = self.client.search(
                search_text=query,
                top=top,
                filter=filter
            )
            
            documents = []
            for result in results:
                documents.append(dict(result))
            
            logger.info(f"Búsqueda completada: {len(documents)} documentos encontrados")
            return documents
            
        except Exception as e:
            logger.error(f"Error en búsqueda de documentos: {str(e)}")
            return []
    
    async def upload_document(self, document: Dict[str, Any]) -> bool:
        """
        Subir un documento al índice de búsqueda
        
        Args:
            document: Documento a subir
            
        Returns:
            True si se subió correctamente, False en caso contrario
        """
        try:
            if not self.client:
                return False
            
            self.client.upload_documents([document])
            logger.info("Documento subido exitosamente al índice de búsqueda")
            return True
            
        except Exception as e:
            logger.error(f"Error subiendo documento al índice: {str(e)}")
            return False

class AzureDocumentIntelligenceHelper:
    """Helper para trabajar con Azure Document Intelligence"""
    
    def __init__(self):
        """Inicializar el helper de Document Intelligence"""
        try:
            self.client = get_doc_intelligence_client()
            logger.info("Azure Document Intelligence Helper inicializado correctamente")
        except Exception as e:
            self.client = None
            logger.error(f"Error inicializando Azure Document Intelligence Helper: {str(e)}")
    
    async def analyze_document(self, document_url: str, model: str = "prebuilt-document") -> Optional[Dict[str, Any]]:
        """
        Analizar un documento usando Document Intelligence
        
        Args:
            document_url: URL del documento a analizar
            model: Modelo de análisis a usar
            
        Returns:
            Resultado del análisis o None si hay error
        """
        try:
            if not self.client:
                return None
            
            poller = self.client.begin_analyze_document_from_url(model, document_url)
            result = poller.result()
            
            # Extraer texto del documento
            extracted_text = ""
            for page in result.pages:
                for line in page.lines:
                    extracted_text += line.content + "\n"
            
            analysis_result = {
                "text": extracted_text,
                "pages": len(result.pages),
                "confidence": result.confidence,
                "model": model
            }
            
            logger.info(f"Documento analizado exitosamente: {len(extracted_text)} caracteres extraídos")
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error analizando documento: {str(e)}")
            return None
    
    async def analyze_document_bytes(self, document_bytes: bytes, model: str = "prebuilt-document") -> Optional[Dict[str, Any]]:
        """
        Analizar un documento desde bytes usando Document Intelligence
        
        Args:
            document_bytes: Bytes del documento a analizar
            model: Modelo de análisis a usar
            
        Returns:
            Resultado del análisis o None si hay error
        """
        try:
            if not self.client:
                return None
            
            poller = self.client.begin_analyze_document(model, document_bytes)
            result = poller.result()
            
            # Extraer texto del documento
            extracted_text = ""
            for page in result.pages:
                for line in page.lines:
                    extracted_text += line.content + "\n"
            
            analysis_result = {
                "text": extracted_text,
                "pages": len(result.pages),
                "confidence": result.confidence,
                "model": model
            }
            
            logger.info(f"Documento analizado exitosamente: {len(extracted_text)} caracteres extraídos")
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error analizando documento: {str(e)}")
            return None

# Instancias globales de los helpers
storage_helper = AzureStorageHelper()
search_helper = AzureSearchHelper()
doc_intelligence_helper = AzureDocumentIntelligenceHelper()
