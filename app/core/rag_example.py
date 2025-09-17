"""
Ejemplo de uso del pipeline RAG
Muestra cómo procesar documentos y crear un índice vectorial
"""

import asyncio
import os
from pathlib import Path
from typing import List

from app.core.rag_pipeline import rag_pipeline
from app.utils.logger import get_logger

logger = get_logger(__name__)

async def example_process_single_document():
    """Ejemplo: Procesar un solo documento"""
    try:
        logger.info("📄 Ejemplo: Procesando un solo documento")
        
        # Crear un archivo de ejemplo (en producción usarías un archivo real)
        example_content = """
        La inteligencia artificial (IA) es una rama de la informática que busca crear 
        sistemas capaces de realizar tareas que normalmente requieren inteligencia humana.
        
        Estas tareas incluyen:
        - Aprendizaje automático
        - Razonamiento lógico
        - Percepción visual
        - Resolución de problemas complejos
        
        El machine learning es un subconjunto de la IA que se enfoca en el desarrollo de 
        algoritmos que pueden aprender y hacer predicciones basándose en datos.
        
        Los modelos de deep learning utilizan redes neuronales artificiales para procesar 
        información de manera similar al cerebro humano.
        """
        
        # Crear archivo temporal
        temp_file = "example_document.txt"
        with open(temp_file, "w", encoding="utf-8") as f:
            f.write(example_content)
        
        try:
            # Procesar documento completo
            result = await rag_pipeline.process_and_index_document(temp_file)
            
            if result["success"]:
                logger.info(f"✅ Documento procesado exitosamente:")
                logger.info(f"   - Archivo: {result['file_path']}")
                logger.info(f"   - Documentos procesados: {result['documents_processed']}")
                logger.info(f"   - Tiempo de procesamiento: {result['processing_time_seconds']:.2f}s")
                logger.info(f"   - Tamaño del archivo: {result['file_size_bytes']} bytes")
                logger.info(f"   - Modelo de embeddings: {result['embedding_model']}")
            else:
                logger.error(f"❌ Error procesando documento: {result.get('error', 'Error desconocido')}")
            
            return result
            
        finally:
            # Limpiar archivo temporal
            if os.path.exists(temp_file):
                os.remove(temp_file)
                
    except Exception as e:
        logger.error(f"❌ Error en ejemplo de documento único: {str(e)}")
        raise

async def example_process_multiple_documents():
    """Ejemplo: Procesar múltiples documentos"""
    try:
        logger.info("📚 Ejemplo: Procesando múltiples documentos")
        
        # Crear múltiples archivos de ejemplo
        documents = [
            {
                "name": "ai_basics.txt",
                "content": """
                Conceptos básicos de Inteligencia Artificial:
                
                La IA se puede clasificar en:
                1. IA débil (narrow AI): Diseñada para tareas específicas
                2. IA fuerte (general AI): Capaz de realizar cualquier tarea intelectual humana
                
                Aplicaciones comunes:
                - Reconocimiento de voz
                - Procesamiento de lenguaje natural
                - Visión por computadora
                - Sistemas de recomendación
                """
            },
            {
                "name": "machine_learning.txt",
                "content": """
                Machine Learning Fundamentals:
                
                Tipos de aprendizaje:
                - Supervisado: Con datos etiquetados
                - No supervisado: Sin datos etiquetados
                - Por refuerzo: Aprendizaje basado en recompensas
                
                Algoritmos populares:
                - Regresión lineal
                - Árboles de decisión
                - Redes neuronales
                - Support Vector Machines
                """
            },
            {
                "name": "deep_learning.txt",
                "content": """
                Deep Learning Overview:
                
                Las redes neuronales profundas consisten en:
                - Capas de entrada
                - Capas ocultas
                - Capa de salida
                
                Arquitecturas populares:
                - CNN (Convolutional Neural Networks)
                - RNN (Recurrent Neural Networks)
                - Transformer
                - GAN (Generative Adversarial Networks)
                """
            }
        ]
        
        # Crear archivos temporales
        temp_files = []
        for doc in documents:
            file_path = doc["name"]
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(doc["content"])
            temp_files.append(file_path)
        
        try:
            results = []
            
            # Procesar cada documento
            for file_path in temp_files:
                logger.info(f"📄 Procesando: {file_path}")
                result = await rag_pipeline.process_and_index_document(file_path)
                results.append(result)
                
                if result["success"]:
                    logger.info(f"✅ {file_path}: {result['documents_processed']} chunks procesados")
                else:
                    logger.error(f"❌ {file_path}: {result.get('error', 'Error desconocido')}")
            
            # Resumen
            successful = sum(1 for r in results if r["success"])
            total_documents = len(results)
            total_chunks = sum(r.get("documents_processed", 0) for r in results if r["success"])
            
            logger.info(f"📊 Resumen del procesamiento:")
            logger.info(f"   - Documentos exitosos: {successful}/{total_documents}")
            logger.info(f"   - Total de chunks procesados: {total_chunks}")
            
            return results
            
        finally:
            # Limpiar archivos temporales
            for file_path in temp_files:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    
    except Exception as e:
        logger.error(f"❌ Error en ejemplo de múltiples documentos: {str(e)}")
        raise

async def example_step_by_step_processing():
    """Ejemplo: Procesamiento paso a paso"""
    try:
        logger.info("🔧 Ejemplo: Procesamiento paso a paso")
        
        # Crear archivo de ejemplo
        example_content = """
        Procesamiento paso a paso de documentos:
        
        Este documento será procesado en etapas separadas para demostrar
        el funcionamiento del pipeline RAG.
        
        Etapas del proceso:
        1. Extracción de texto
        2. División en chunks
        3. Generación de embeddings
        4. Indexación en Azure Search
        """
        
        temp_file = "step_by_step_example.txt"
        with open(temp_file, "w", encoding="utf-8") as f:
            f.write(example_content)
        
        try:
            # Paso 1: Procesar documento (extraer texto, chunking, embeddings)
            logger.info("🔄 Paso 1: Procesando documento...")
            documents = await rag_pipeline.process_document(temp_file)
            
            logger.info(f"✅ Documento procesado: {len(documents)} chunks creados")
            
            # Mostrar información de los chunks
            for i, doc in enumerate(documents[:3]):  # Mostrar solo los primeros 3
                logger.info(f"   Chunk {i+1}: {doc['chunk_size']} caracteres")
            
            # Paso 2: Agregar al índice de búsqueda
            logger.info("🔄 Paso 2: Agregando al índice de búsqueda...")
            index_success = await rag_pipeline.add_documents_to_search(documents)
            
            if index_success:
                logger.info(f"✅ Documentos agregados al índice exitosamente")
            else:
                logger.error(f"❌ Error agregando documentos al índice")
            
            return {
                "documents_processed": len(documents),
                "index_success": index_success
            }
            
        finally:
            # Limpiar archivo temporal
            if os.path.exists(temp_file):
                os.remove(temp_file)
                
    except Exception as e:
        logger.error(f"❌ Error en ejemplo paso a paso: {str(e)}")
        raise

async def example_validate_pipeline():
    """Ejemplo: Validar el pipeline"""
    try:
        logger.info("🔍 Ejemplo: Validando pipeline RAG")
        
        # Validar pipeline
        is_valid = await rag_pipeline.validate_pipeline()
        
        if is_valid:
            logger.info("✅ Pipeline RAG validado correctamente")
        else:
            logger.error("❌ Pipeline RAG tiene problemas")
        
        return is_valid
        
    except Exception as e:
        logger.error(f"❌ Error validando pipeline: {str(e)}")
        raise

async def main():
    """Función principal que ejecuta todos los ejemplos"""
    try:
        logger.info("🚀 Iniciando ejemplos del pipeline RAG...")
        
        # 1. Validar pipeline
        logger.info("=" * 50)
        await example_validate_pipeline()
        
        # 2. Procesar documento único
        logger.info("=" * 50)
        await example_process_single_document()
        
        # 3. Procesar múltiples documentos
        logger.info("=" * 50)
        await example_process_multiple_documents()
        
        # 4. Procesamiento paso a paso
        logger.info("=" * 50)
        await example_step_by_step_processing()
        
        logger.info("🎉 Todos los ejemplos del pipeline RAG completados exitosamente!")
        
    except Exception as e:
        logger.error(f"❌ Error en ejemplos del pipeline: {str(e)}")

if __name__ == "__main__":
    # Ejecutar ejemplos
    asyncio.run(main())
