"""
Ejemplo de uso de los clientes de Azure para el patrón RAG
Este archivo muestra cómo usar todos los clientes y helpers de Azure
"""

import asyncio
from typing import Dict, Any, List
from app.config.settings import settings
from app.utils.azure_clients import (
    get_openai_client,
    get_openai_embeddings_client,
    get_search_client,
    get_doc_intelligence_client,
    get_blob_service_client,
    validate_all_clients
)
from app.utils.azure_helpers import (
    storage_helper,
    search_helper,
    doc_intelligence_helper
)
from app.utils.logger import get_logger

logger = get_logger(__name__)

async def example_validate_configuration():
    """Ejemplo: Validar que todas las configuraciones estén correctas"""
    try:
        logger.info("🔍 Validando configuración de Azure...")
        
        # Validar configuraciones
        settings.validate()
        
        # Validar clientes
        validate_all_clients()
        
        logger.info("✅ Configuración validada correctamente")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error en validación: {str(e)}")
        return False

async def example_document_processing():
    """Ejemplo: Procesar un documento completo con RAG"""
    try:
        logger.info("📄 Procesando documento con RAG...")
        
        # 1. Simular documento (en producción vendría de una petición)
        document_content = """
        La inteligencia artificial (IA) es una rama de la informática que busca crear 
        sistemas capaces de realizar tareas que normalmente requieren inteligencia humana. 
        Estas tareas incluyen el aprendizaje, el razonamiento, la percepción y la resolución de problemas.
        
        El machine learning es un subconjunto de la IA que se enfoca en el desarrollo de 
        algoritmos que pueden aprender y hacer predicciones basándose en datos.
        """
        
        # 2. Subir documento a Azure Storage
        document_name = "ai_document.txt"
        success = await storage_helper.upload_blob(
            blob_name=document_name,
            data=document_content.encode('utf-8')
        )
        
        if not success:
            raise Exception("Error subiendo documento a Storage")
        
        logger.info(f"📤 Documento subido: {document_name}")
        
        # 3. Extraer texto con Document Intelligence (simulado)
        # En producción, esto procesaría un PDF o imagen real
        extracted_text = document_content  # Simulado
        
        # 4. Crear embeddings del texto
        embeddings_client = get_openai_embeddings_client()
        embeddings = await embeddings_client.aembed_query(extracted_text)
        
        logger.info(f"🔢 Embeddings generados: {len(embeddings)} dimensiones")
        
        # 5. Crear documento para el índice de búsqueda
        search_document = {
            "id": document_name,
            "content": extracted_text,
            "contentVector": embeddings,
            "metadata": {
                "source": "example",
                "type": "text",
                "uploaded_at": "2024-01-01T00:00:00Z"
            }
        }
        
        # 6. Subir al índice de búsqueda
        success = await search_helper.upload_document(search_document)
        
        if not success:
            raise Exception("Error subiendo documento al índice")
        
        logger.info("📊 Documento indexado en Azure Search")
        
        return {
            "document_name": document_name,
            "embeddings_dimensions": len(embeddings),
            "content_length": len(extracted_text)
        }
        
    except Exception as e:
        logger.error(f"❌ Error procesando documento: {str(e)}")
        raise

async def example_rag_query():
    """Ejemplo: Realizar una consulta RAG"""
    try:
        logger.info("🔍 Realizando consulta RAG...")
        
        # 1. Consulta del usuario
        user_query = "¿Qué es la inteligencia artificial?"
        
        # 2. Generar embeddings de la consulta
        embeddings_client = get_openai_embeddings_client()
        query_embeddings = await embeddings_client.aembed_query(user_query)
        
        # 3. Buscar documentos similares
        # En producción, usarías búsqueda vectorial
        search_results = await search_helper.search_documents(
            query=user_query,
            top=3
        )
        
        logger.info(f"📚 Documentos encontrados: {len(search_results)}")
        
        # 4. Construir contexto con los documentos encontrados
        context = ""
        for i, doc in enumerate(search_results, 1):
            context += f"Documento {i}:\n{doc.get('content', '')}\n\n"
        
        # 5. Generar respuesta usando OpenAI
        openai_client = get_openai_client()
        
        prompt = f"""
        Basándote en el siguiente contexto, responde la pregunta del usuario.
        
        Contexto:
        {context}
        
        Pregunta: {user_query}
        
        Respuesta:
        """
        
        response = await openai_client.ainvoke(prompt)
        
        logger.info("🤖 Respuesta generada con RAG")
        
        return {
            "query": user_query,
            "documents_found": len(search_results),
            "response": response.content,
            "context_length": len(context)
        }
        
    except Exception as e:
        logger.error(f"❌ Error en consulta RAG: {str(e)}")
        raise

async def example_list_storage_contents():
    """Ejemplo: Listar contenido del storage"""
    try:
        logger.info("📁 Listando contenido del storage...")
        
        blobs = await storage_helper.list_blobs()
        
        logger.info(f"📋 Encontrados {len(blobs)} blobs en el storage")
        
        for blob in blobs:
            logger.info(f"  - {blob}")
        
        return blobs
        
    except Exception as e:
        logger.error(f"❌ Error listando storage: {str(e)}")
        raise

async def main():
    """Función principal que ejecuta todos los ejemplos"""
    try:
        logger.info("🚀 Iniciando ejemplos de uso de Azure AI...")
        
        # 1. Validar configuración
        if not await example_validate_configuration():
            logger.error("❌ Configuración inválida, abortando...")
            return
        
        # 2. Procesar documento
        doc_result = await example_document_processing()
        logger.info(f"✅ Documento procesado: {doc_result}")
        
        # 3. Realizar consulta RAG
        rag_result = await example_rag_query()
        logger.info(f"✅ Consulta RAG completada: {rag_result}")
        
        # 4. Listar contenido del storage
        storage_contents = await example_list_storage_contents()
        logger.info(f"✅ Storage listado: {len(storage_contents)} elementos")
        
        logger.info("🎉 Todos los ejemplos completados exitosamente!")
        
    except Exception as e:
        logger.error(f"❌ Error en ejemplos: {str(e)}")

if __name__ == "__main__":
    # Ejecutar ejemplos
    asyncio.run(main())
