# MFN-MVP

Aplicación de Inteligencia Artificial con patrón RAG (Retrieval-Augmented Generation) desplegada en Azure Functions.

## Estructura del Proyecto

```
mfn-mvp/
├── app/
│   ├── api/           # Código para conexiones a internet
│   ├── core/          # Lógica principal del agente de IA
│   ├── config/        # Configuraciones y llaves secretas
│   └── utils/         # Herramientas y funciones de ayuda
├── requirements.txt   # Dependencias del proyecto
├── env.example        # Plantilla de variables de entorno
└── README.md         # Este archivo
```

## Características

- **Patrón RAG**: Recuperación aumentada por generación usando Azure Cognitive Search
- **Azure OpenAI**: Integración con modelos de Azure OpenAI para embeddings y generación
- **Document Intelligence**: Extracción de texto de documentos con Azure Form Recognizer
- **Azure Storage**: Almacenamiento de documentos en Azure Blob Storage
- **LangChain**: Framework para construir cadenas de procesamiento de IA

## Instalación

1. Clona el repositorio
2. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

## Configuración

### 1. Variables de Entorno

Copia el archivo `env.example` a `.env` y configura las variables:

```bash
cp env.example .env
```

### 2. Servicios de Azure Requeridos

Necesitas configurar los siguientes servicios en Azure:

- **Azure OpenAI Service**: Para embeddings y generación de texto
- **Azure Cognitive Search**: Para búsqueda vectorial y semántica
- **Azure Document Intelligence**: Para extracción de texto de documentos
- **Azure Storage**: Para almacenamiento de documentos

### 3. Variables Críticas

Las siguientes variables son **obligatorias**:

```bash
# Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_DEPLOYMENT_NAME=your-deployment

# Azure Cognitive Search
AZURE_SEARCH_ENDPOINT=https://your-search-service.search.windows.net
AZURE_SEARCH_API_KEY=your-search-key
AZURE_SEARCH_INDEX_NAME=your-index-name

# Azure Document Intelligence
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=https://your-resource.cognitiveservices.azure.com/
AZURE_DOCUMENT_INTELLIGENCE_KEY=your-key

# Azure Storage
AZURE_STORAGE_CONNECTION_STRING=your-connection-string
AZURE_STORAGE_CONTAINER_NAME=documents
```

## Desarrollo Local

Para ejecutar la aplicación localmente:

```bash
# Instalar Azure Functions Core Tools
npm install -g azure-functions-core-tools

# Ejecutar localmente
func start
```

## Despliegue en Azure

Este proyecto está configurado para desplegarse en Azure Functions.

### Configuración en Azure

1. Crea una Function App en Azure
2. Configura las variables de entorno en la configuración de la aplicación
3. Despliega el código usando Azure CLI o GitHub Actions

## Validación de Configuración

El sistema validará automáticamente que todas las variables requeridas estén configuradas al iniciar:

```python
from app.config.settings import settings

# Validar configuración
settings.validate()
```
