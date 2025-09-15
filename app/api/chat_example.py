"""
Ejemplo de uso del endpoint de chat
Muestra cómo hacer peticiones al endpoint /api/chat
"""

import requests
import json
import time
from typing import Dict, Any

def test_chat_endpoint(base_url: str = "http://localhost:7071") -> None:
    """
    Probar el endpoint de chat con diferentes preguntas
    """
    print("🚀 Probando endpoint de chat...")
    print(f"URL base: {base_url}")
    print("=" * 60)
    
    # Preguntas de prueba
    test_questions = [
        "¿Qué es la inteligencia artificial?",
        "¿Cuáles son los diferentes tipos de machine learning?",
        "¿Cómo funciona el deep learning?",
        "¿Qué es el procesamiento de lenguaje natural?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n📝 Pregunta {i}: {question}")
        
        try:
            # Preparar la petición
            url = f"{base_url}/api/chat"
            payload = {
                "question": question
            }
            headers = {
                "Content-Type": "application/json"
            }
            
            # Hacer la petición
            start_time = time.time()
            response = requests.post(url, json=payload, headers=headers)
            end_time = time.time()
            
            # Procesar la respuesta
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Respuesta exitosa:")
                print(f"   Respuesta: {result.get('answer', '')[:200]}...")
                print(f"   Tiempo de procesamiento: {result.get('processing_time_seconds', 0):.2f}s")
                print(f"   Documentos recuperados: {result.get('documents_retrieved', 0)}")
                print(f"   Tiempo total (incluyendo red): {end_time - start_time:.2f}s")
            else:
                print(f"❌ Error HTTP {response.status_code}:")
                print(f"   {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Error de conexión: {str(e)}")
        except Exception as e:
            print(f"❌ Error inesperado: {str(e)}")
        
        print("-" * 40)

def test_invalid_requests(base_url: str = "http://localhost:7071") -> None:
    """
    Probar peticiones inválidas para verificar el manejo de errores
    """
    print("\n🔍 Probando manejo de errores...")
    print("=" * 60)
    
    # Petición con JSON inválido
    print("\n📝 Prueba 1: JSON inválido")
    try:
        url = f"{base_url}/api/chat"
        response = requests.post(url, data="invalid json", headers={"Content-Type": "application/json"})
        print(f"   Status: {response.status_code}")
        print(f"   Respuesta: {response.text[:100]}...")
    except Exception as e:
        print(f"   Error: {str(e)}")
    
    # Petición sin campo 'question'
    print("\n📝 Prueba 2: Sin campo 'question'")
    try:
        url = f"{base_url}/api/chat"
        payload = {"other_field": "test"}
        response = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
        print(f"   Status: {response.status_code}")
        print(f"   Respuesta: {response.text[:100]}...")
    except Exception as e:
        print(f"   Error: {str(e)}")
    
    # Petición GET (método no permitido)
    print("\n📝 Prueba 3: Método GET (no permitido)")
    try:
        url = f"{base_url}/api/chat"
        response = requests.get(url)
        print(f"   Status: {response.status_code}")
        print(f"   Respuesta: {response.text[:100]}...")
    except Exception as e:
        print(f"   Error: {str(e)}")

def test_health_endpoint(base_url: str = "http://localhost:7071") -> None:
    """
    Probar el endpoint de salud
    """
    print("\n🏥 Probando endpoint de salud...")
    print("=" * 60)
    
    try:
        url = f"{base_url}/?action=health"
        response = requests.get(url)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Servicio saludable:")
            print(f"   Status: {result.get('status')}")
            print(f"   Service: {result.get('service')}")
            print(f"   Version: {result.get('version')}")
            print(f"   Endpoint: {result.get('endpoint')}")
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"   {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")

def test_system_status(base_url: str = "http://localhost:7071") -> None:
    """
    Probar el endpoint de estado del sistema
    """
    print("\n📊 Probando estado del sistema...")
    print("=" * 60)
    
    try:
        url = f"{base_url}/?action=status"
        response = requests.get(url)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Estado del sistema:")
            print(f"   Status: {result.get('status')}")
            print(f"   Service: {result.get('service')}")
            
            agent_info = result.get('agent_info', {})
            print(f"   Modelo: {agent_info.get('model')}")
            print(f"   Retriever: {agent_info.get('retriever_type')}")
            print(f"   Índice: {agent_info.get('search_index')}")
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"   {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")

def main():
    """
    Función principal que ejecuta todas las pruebas
    """
    print("🎯 Ejemplos de uso del endpoint de chat")
    print("=" * 60)
    
    # URL base (cambiar según tu configuración)
    base_url = "http://localhost:7071"  # Para desarrollo local
    # base_url = "https://your-function-app.azurewebsites.net"  # Para Azure
    
    # Ejecutar pruebas
    test_health_endpoint(base_url)
    test_system_status(base_url)
    test_chat_endpoint(base_url)
    test_invalid_requests(base_url)
    
    print("\n🎉 Pruebas completadas!")

if __name__ == "__main__":
    main()
