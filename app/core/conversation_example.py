"""
Ejemplo de uso del agente conversacional
Muestra cómo usar el ConversationAgent para responder preguntas con RAG
"""

import asyncio
from typing import List

from app.core.conversation_agent import conversation_agent
from app.utils.logger import get_logger

logger = get_logger(__name__)

async def example_basic_question():
    """Ejemplo: Pregunta básica"""
    try:
        logger.info("🤔 Ejemplo: Pregunta básica")
        
        question = "¿Qué es la inteligencia artificial?"
        
        # Procesar pregunta
        result = await conversation_agent.ask(question)
        
        if result["success"]:
            logger.info(f"✅ Respuesta generada:")
            logger.info(f"   Pregunta: {result['question']}")
            logger.info(f"   Respuesta: {result['answer'][:200]}...")
            logger.info(f"   Documentos recuperados: {result['documents_retrieved']}")
            logger.info(f"   Tiempo de procesamiento: {result['processing_time_seconds']:.2f}s")
            
            # Mostrar documentos fuente
            if result['source_documents']:
                logger.info("   Documentos fuente:")
                for doc in result['source_documents'][:2]:  # Mostrar solo los primeros 2
                    logger.info(f"     - {doc['source']}: {doc['content'][:100]}...")
        else:
            logger.error(f"❌ Error: {result.get('error', 'Error desconocido')}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Error en ejemplo básico: {str(e)}")
        raise

async def example_technical_question():
    """Ejemplo: Pregunta técnica"""
    try:
        logger.info("🔧 Ejemplo: Pregunta técnica")
        
        question = "¿Cuáles son los diferentes tipos de machine learning?"
        
        # Procesar pregunta
        result = await conversation_agent.ask(question)
        
        if result["success"]:
            logger.info(f"✅ Respuesta técnica generada:")
            logger.info(f"   Pregunta: {result['question']}")
            logger.info(f"   Respuesta: {result['answer'][:300]}...")
            logger.info(f"   Estrategia: {result['retrieval_strategy']}")
            logger.info(f"   Modelo usado: {result['model']}")
        else:
            logger.error(f"❌ Error: {result.get('error', 'Error desconocido')}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Error en ejemplo técnico: {str(e)}")
        raise

async def example_question_with_context():
    """Ejemplo: Pregunta con contexto adicional"""
    try:
        logger.info("📝 Ejemplo: Pregunta con contexto adicional")
        
        question = "¿Cómo se relaciona con el deep learning?"
        additional_context = """
        El deep learning es una subcategoría del machine learning que utiliza 
        redes neuronales artificiales con múltiples capas para procesar datos 
        y aprender patrones complejos.
        """
        
        # Procesar pregunta con contexto
        result = await conversation_agent.ask_with_custom_context(question, additional_context)
        
        if result["success"]:
            logger.info(f"✅ Respuesta con contexto generada:")
            logger.info(f"   Pregunta: {result['question']}")
            logger.info(f"   Contexto adicional: {result['additional_context'][:100]}...")
            logger.info(f"   Respuesta: {result['answer'][:250]}...")
            logger.info(f"   Estrategia: {result['retrieval_strategy']}")
        else:
            logger.error(f"❌ Error: {result.get('error', 'Error desconocido')}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Error en ejemplo con contexto: {str(e)}")
        raise

async def example_multiple_questions():
    """Ejemplo: Múltiples preguntas"""
    try:
        logger.info("📚 Ejemplo: Múltiples preguntas")
        
        questions = [
            "¿Qué es el procesamiento de lenguaje natural?",
            "¿Cuáles son las aplicaciones de la visión por computadora?",
            "¿Cómo funciona el aprendizaje por refuerzo?",
            "¿Qué son las redes neuronales convolucionales?"
        ]
        
        results = []
        
        for i, question in enumerate(questions, 1):
            logger.info(f"   Procesando pregunta {i}/{len(questions)}: {question}")
            
            result = await conversation_agent.ask(question)
            results.append(result)
            
            if result["success"]:
                logger.info(f"   ✅ Respuesta {i}: {result['answer'][:100]}...")
            else:
                logger.error(f"   ❌ Error en pregunta {i}: {result.get('error', 'Error desconocido')}")
        
        # Resumen
        successful = sum(1 for r in results if r["success"])
        total_time = sum(r.get("processing_time_seconds", 0) for r in results if r["success"])
        
        logger.info(f"📊 Resumen de preguntas:")
        logger.info(f"   - Preguntas exitosas: {successful}/{len(questions)}")
        logger.info(f"   - Tiempo total: {total_time:.2f}s")
        logger.info(f"   - Tiempo promedio: {total_time/successful:.2f}s por pregunta" if successful > 0 else "   - No hay preguntas exitosas")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Error en ejemplo múltiple: {str(e)}")
        raise

async def example_agent_validation():
    """Ejemplo: Validar el agente"""
    try:
        logger.info("🔍 Ejemplo: Validando agente conversacional")
        
        # Validar agente
        is_valid = await conversation_agent.validate_agent()
        
        if is_valid:
            logger.info("✅ Agente conversacional validado correctamente")
            
            # Obtener información del agente
            agent_info = conversation_agent.get_agent_info()
            logger.info("📋 Información del agente:")
            logger.info(f"   - Tipo: {agent_info['agent_type']}")
            logger.info(f"   - Modelo: {agent_info['model']}")
            logger.info(f"   - Retriever: {agent_info['retriever_type']}")
            logger.info(f"   - Índice de búsqueda: {agent_info['search_index']}")
            logger.info(f"   - Documentos top-k: {agent_info['top_k_documents']}")
        else:
            logger.error("❌ Agente conversacional tiene problemas")
        
        return is_valid
        
    except Exception as e:
        logger.error(f"❌ Error validando agente: {str(e)}")
        raise

async def example_conversation_flow():
    """Ejemplo: Flujo de conversación"""
    try:
        logger.info("💬 Ejemplo: Flujo de conversación")
        
        # Simular una conversación
        conversation = [
            "¿Qué es la inteligencia artificial?",
            "¿Y cómo se relaciona con el machine learning?",
            "¿Puedes darme ejemplos de aplicaciones prácticas?",
            "¿Cuáles son los desafíos actuales de la IA?"
        ]
        
        conversation_history = []
        
        for i, question in enumerate(conversation, 1):
            logger.info(f"   Turno {i}: {question}")
            
            # Procesar pregunta
            result = await conversation_agent.ask(question)
            conversation_history.append({
                "turn": i,
                "question": question,
                "answer": result.get("answer", "Error"),
                "success": result.get("success", False)
            })
            
            if result["success"]:
                logger.info(f"   Respuesta: {result['answer'][:150]}...")
            else:
                logger.error(f"   Error: {result.get('error', 'Error desconocido')}")
            
            # Pausa entre preguntas
            await asyncio.sleep(1)
        
        # Resumen de la conversación
        successful_turns = sum(1 for turn in conversation_history if turn["success"])
        
        logger.info(f"📝 Resumen de la conversación:")
        logger.info(f"   - Turnos exitosos: {successful_turns}/{len(conversation)}")
        logger.info(f"   - Conversación completada: {'Sí' if successful_turns == len(conversation) else 'Parcialmente'}")
        
        return conversation_history
        
    except Exception as e:
        logger.error(f"❌ Error en flujo de conversación: {str(e)}")
        raise

async def main():
    """Función principal que ejecuta todos los ejemplos"""
    try:
        logger.info("🚀 Iniciando ejemplos del agente conversacional...")
        
        # 1. Validar agente
        logger.info("=" * 60)
        await example_agent_validation()
        
        # 2. Pregunta básica
        logger.info("=" * 60)
        await example_basic_question()
        
        # 3. Pregunta técnica
        logger.info("=" * 60)
        await example_technical_question()
        
        # 4. Pregunta con contexto
        logger.info("=" * 60)
        await example_question_with_context()
        
        # 5. Múltiples preguntas
        logger.info("=" * 60)
        await example_multiple_questions()
        
        # 6. Flujo de conversación
        logger.info("=" * 60)
        await example_conversation_flow()
        
        logger.info("🎉 Todos los ejemplos del agente conversacional completados exitosamente!")
        
    except Exception as e:
        logger.error(f"❌ Error en ejemplos del agente conversacional: {str(e)}")

if __name__ == "__main__":
    # Ejecutar ejemplos
    asyncio.run(main())
