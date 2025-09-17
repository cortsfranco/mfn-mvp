"""
Script especializado para procesar facturas
Utiliza Azure Document Intelligence para extraer información, previene duplicados
usando un hash SHA-256, y Azure Search para indexar los datos extraídos.
"""

import uuid
import json
import hashlib
from datetime import datetime, timezone
from typing import Dict, Any, Union, Optional

from azure.core.exceptions import HttpResponseError
from app.utils.azure_clients import get_doc_intelligence_client, get_search_client
from app.utils.logger import get_logger

logger = get_logger(__name__)

# --- Nombres de los modelos ---
MODELO_EMITIDAS = "opendoors-emitidas-custom"
MODELO_RECIBIDAS = "opendoors-recibidas-custom"

class InvoiceProcessor:
    """
    Procesador especializado para facturas que incluye lógica anti-duplicados.
    """

    def __init__(self):
        """Inicializar el procesador de facturas"""
        self.doc_intelligence_client = get_doc_intelligence_client()
        self.search_client = get_search_client()
        logger.info("Procesador de facturas inicializado correctamente")

    def _calculate_file_hash(self, file_bytes: bytes) -> str:
        """Calcula el hash SHA-256 de un archivo para usarlo como huella digital."""
        sha256_hash = hashlib.sha256()
        sha256_hash.update(file_bytes)
        return sha256_hash.hexdigest()

    async def _is_duplicate(self, file_hash: str) -> bool:
        """Verifica si ya existe una factura con el mismo hash en el índice."""
        try:
            filter_query = f"file_hash eq '{file_hash}'"
            logger.info(f"🔎 Verificando duplicados con el filtro: {filter_query}")
            search_results = self.search_client.search(search_text="*", filter=filter_query, include_total_count=True)
            count = search_results.get_count()
            logger.info(f"Se encontraron {count} facturas con el mismo hash.")
            return count > 0
        except Exception as e:
            logger.error(f"❌ Error verificando duplicados: {str(e)}", exc_info=True)
            return True

    async def _analyze_with_model(self, model_id: str, document_bytes: bytes) -> Optional[Any]:
        """
        Función auxiliar para analizar un documento con un modelo específico.
        Ahora incluye una verificación estricta de docType y confianza.
        """
        try:
            logger.info(f"🔍 Analizando con modelo: {model_id}...")
            poller = self.doc_intelligence_client.begin_analyze_document(model_id, document_bytes)
            result = poller.result()

            if result and result.documents:
                document = result.documents[0]
                doc_type = document.doc_type
                confidence = document.confidence
                
                logger.info(f"Modelo {model_id} detectó docType: '{doc_type}' con confianza: {confidence:.2%}")

                expected_doc_type = model_id
                
                if doc_type == expected_doc_type and confidence > 0.95: # <-- 1. Umbral de confianza aumentado
                    logger.info(f"✅ Verificación exitosa para el modelo {model_id}")
                    return result
                else:
                    logger.warning(f"⚠️ Verificación fallida para {model_id}. docType o confianza no cumplen el umbral.")
                    return None
            else:
                logger.warning(f"⚠️ El modelo {model_id} se ejecutó pero no encontró documentos en el archivo.")
                return None
        except HttpResponseError as e:
            logger.warning(f"El modelo {model_id} no pudo procesar el documento. Error: {e.message}")
            return None
        except Exception as e:
            logger.error(f"Error inesperado durante el análisis con {model_id}: {str(e)}")
            raise

    async def process_and_upload_invoice(self, file_path: str, partner_name: str) -> Dict[str, Any]:
        """
        Procesa una factura, previene duplicados, determina el tipo (INGRESO/EGRESO)
        y la sube al índice de búsqueda.
        """
        try:
            logger.info(f"📄 Procesando nueva factura: {file_path}")
            with open(file_path, 'rb') as file:
                document_bytes = file.read()

            file_hash = self._calculate_file_hash(document_bytes)
            logger.info(f"🔑 Huella digital (hash) del archivo: {file_hash}")
            if await self._is_duplicate(file_hash):
                logger.warning("🚫 Factura duplicada detectada. Proceso cancelado.")
                return { "success": False, "error": "duplicate", "message": "Esta factura ya fue cargada anteriormente." }
            
            analysis_result = None
            invoice_type = None

            analysis_result = await self._analyze_with_model(MODELO_EMITIDAS, document_bytes)
            if analysis_result:
                invoice_type = "ingreso"  # <-- 2. Estandarizado a minúscula
                logger.info("📊 Factura clasificada como INGRESO.")
            
            if not analysis_result:
                logger.info("Intentando con modelo de facturas recibidas...")
                analysis_result = await self._analyze_with_model(MODELO_RECIBIDAS, document_bytes)
                if analysis_result:
                    invoice_type = "egreso"   # <-- 2. Estandarizado a minúscula
                    logger.info("📊 Factura clasificada como EGRESO.")
            
            if not analysis_result or not invoice_type:
                raise ValueError("No se pudo analizar la factura con ninguno de los modelos disponibles.")

            invoice_data = self._extract_invoice_fields(analysis_result)
            structured_document = self._create_structured_document(invoice_data, file_path, invoice_type, partner_name, file_hash)

            logger.info("📤 Subiendo factura al índice de búsqueda...")
            upload_result = self.search_client.upload_documents([structured_document])
            
            upload_success = bool(upload_result and len(upload_result) > 0 and upload_result[0].succeeded)
            if upload_success:
                logger.info("✅ Factura subida al índice exitosamente")
            else:
                error_message = upload_result[0].error_message if upload_result and upload_result[0].error_message else "Error desconocido"
                logger.error(f"❌ Error subiendo factura: {error_message}")

            return { "success": upload_success, "invoice_data": invoice_data, "invoice_type": invoice_type }

        except Exception as e:
            logger.error(f"Error procesando factura: {str(e)}", exc_info=True)
            return { "success": False, "error": str(e) }

    def _extract_invoice_fields(self, analysis_result) -> Dict[str, Any]:
        """Extrae y limpia campos de la factura del resultado de un MODELO PERSONALIZADO."""
        try:
            if not analysis_result.documents: return {}
            document = analysis_result.documents[0]
            fields = document.fields
            invoice_data = {}
            def clean_currency(value: Any) -> float:
                if value is None: return 0.0
                try:
                    return float(str(value).replace("$", "").strip().replace(".", "").replace(",", "."))
                except (ValueError, TypeError): return 0.0
            def get_field_value(field_name: str) -> Union[str, float, None]:
                field = fields.get(field_name)
                return field.content if field else None
            invoice_data["VendorName"] = get_field_value("VendorName") or "N/A"
            invoice_data["InvoiceDate"] = get_field_value("InvoiceDate") or "N/A"
            invoice_data["InvoiceTotal"] = clean_currency(get_field_value("InvoiceTotal"))
            invoice_data["TotalTax"] = clean_currency(get_field_value("TotalTax"))
            logger.info(f"📋 Campos extraídos y limpios: {invoice_data}")
            return invoice_data
        except Exception as e:
            logger.error(f"Error extrayendo campos: {str(e)}", exc_info=True)
            return { "VendorName": "N/A", "InvoiceDate": "N/A", "InvoiceTotal": 0.0, "TotalTax": 0.0 }

    def _create_structured_document(self, invoice_data: Dict[str, Any], file_path: str, invoice_type: str, partner_name: str, file_hash: str) -> Dict[str, Any]:
        """Crear un diccionario estructurado para el índice de búsqueda, incluyendo el hash."""
        document_id = f"invoice_{uuid.uuid4().hex}"
        content_str = json.dumps(invoice_data)
        structured_document = {
            "id": document_id, 
            "content": content_str, 
            "VendorName": invoice_data.get("VendorName", "N/A"),
            "InvoiceDate": invoice_data.get("InvoiceDate", "N/A"), 
            "InvoiceTotal": invoice_data.get("InvoiceTotal", 0.0),
            "TotalTax": invoice_data.get("TotalTax", 0.0), 
            "source_file": file_path, 
            "document_type": "invoice",
            "processed_at": datetime.now(timezone.utc).isoformat(), 
            "InvoiceType": invoice_type, 
            "PartnerName": partner_name,
            "file_hash": file_hash
        }
        logger.info(f"📝 Documento estructurado creado con ID: {document_id} y Hash: {file_hash}")
        return structured_document

    async def query_invoices(self, filter_query: str) -> list[Dict[str, Any]]:
        """Realiza una consulta filtrada en el índice de Azure AI Search."""
        try:
            logger.info(f"🔎 Realizando búsqueda con filtro: {filter_query}")
            search_results = self.search_client.search(search_text="*", filter=filter_query, include_total_count=True)
            results_list = [dict(result) for result in search_results]
            logger.info(f"✅ Búsqueda completada. Se encontraron {search_results.get_count()} resultados.")
            return results_list
        except Exception as e:
            logger.error(f"❌ Error durante la búsqueda: {str(e)}", exc_info=True)
            return []

invoice_processor = InvoiceProcessor()