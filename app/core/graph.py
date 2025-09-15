"""
Define el flujo del agente de procesamiento y consulta de facturas
utilizando LangGraph, con una lógica de enrutamiento para manejar
preguntas simples y complejas (multi-paso).
"""
import json
from typing import TypedDict, List, Literal

from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate

from app.core.rag_pipeline import invoice_processor
from app.utils.azure_clients import get_openai_client
from app.utils.logger import get_logger

logger = get_logger(__name__)

# --- 1. Definimos el Estado del Agente (Ahora más completo) ---
class AgentState(TypedDict):
    question: str
    task_type: str | None
    filter_query: str | None
    search_results: List[dict]
    income_total: float
    expense_total: float
    final_answer: str

# --- 2. Creamos la Clase del Grafo y sus Nodos ---
class AgentGraph:
    def __init__(self):
        self.llm = get_openai_client()
        self.graph = self._build_graph()
        logger.info("✅ Grafo de LangGraph con router construido y compilado")

    # --- Herramienta auxiliar para sumar totales ---
    def _sum_totals(self, search_results: List[dict]) -> float:
        total = 0.0
        for result in search_results:
            # El campo 'content' es un string JSON, necesitamos parsearlo
            try:
                content_data = json.loads(result.get("content", "{}"))
                total += content_data.get("InvoiceTotal", 0.0)
            except (json.JSONDecodeError, TypeError):
                continue
        return total

    # --- NODO 1 (NUEVO): El Router Estratégico ---
    async def route_question_node(self, state: AgentState) -> dict:
        logger.info("🧠 Nodo 1 (Router): Clasificando la pregunta...")
        question = state["question"]

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Tu tarea es clasificar la pregunta del usuario en una de las siguientes categorías para determinar el plan de acción: "
             "`busqueda_simple`, `calculo_balance`, `resumen_general`. "
             "- `busqueda_simple`: Preguntas sobre ingresos o egresos de una persona específica (Joni, Hernan, etc.), o listas de facturas. Ejemplos: 'cuánto gastó joni?', 'muéstrame los ingresos de hernan', 'lista las facturas de egreso'. "
             "- `calculo_balance`: Preguntas que piden un balance total, comparando ingresos y egresos. Ejemplos: 'cuál es el balance general?', 'dame el resultado final'. "
             "- `resumen_general`: Preguntas muy abiertas que piden un resumen de todo. Ejemplo: 'dame un resumen de la situación'. "
             "Responde SIEMPRE Y ÚNICAMENTE con una de las categorías."
             ),
            ("user", "{question}")
        ])
        
        chain = prompt | self.llm
        response = await chain.ainvoke({"question": question})
        task_type = response.content.strip()
        
        logger.info(f"  B - Tarea clasificada como: {task_type}")
        return {"task_type": task_type}

    # --- NODO 2a: Generar Filtro (para búsquedas simples) ---
    async def generate_filter_node(self, state: AgentState) -> dict:
        # (Esta función es idéntica a la anterior)
        logger.info("🧠 Nodo 2a: Generando filtro para búsqueda simple...")
        question = state["question"]
        prompt = ChatPromptTemplate.from_messages([
             ("system",
              "Eres un experto programador que convierte preguntas a filtros OData para Azure AI Search. "
              "Campos disponibles: `PartnerName`, `InvoiceType`. "
              "Reglas: `PartnerName` puede ser 'JONI', 'HERNAN', 'MAXI', 'LEO'. `InvoiceType` puede ser 'ingreso' o 'egreso'. "
              "Usa 'eq' para strings y 'and' para combinar. Si no se necesita filtro, responde 'NO_FILTER'. "
              "Responde SIEMPRE Y ÚNICAMENTE con el filtro o 'NO_FILTER'."
             ),
            ("user", "Pregunta: {question}")
        ])
        chain = prompt | self.llm
        response = await chain.ainvoke({"question": question})
        filter_query = response.content.strip()
        logger.info(f"   - Filtro generado: {filter_query}")
        return {"filter_query": filter_query}
    
    # --- NODO 3a: Ejecutar Búsqueda Simple ---
    async def execute_search_node(self, state: AgentState) -> dict:
        # (Esta función es idéntica a la anterior)
        logger.info("🔎 Nodo 3a: Ejecutando búsqueda simple...")
        filter_query = state["filter_query"]
        if filter_query and filter_query != "NO_FILTER":
            search_results = await invoice_processor.query_invoices(filter_query)
            logger.info(f"   - Se encontraron {len(search_results)} resultados.")
            return {"search_results": search_results}
        else:
            logger.info("   - No se generó un filtro. Saltando búsqueda.")
            return {"search_results": []}

    # --- NODO 2b: Buscar Ingresos (para cálculo de balance) ---
    async def search_income_node(self, state: AgentState) -> dict:
        logger.info("💰 Nodo 2b: Buscando todos los ingresos...")
        search_results = await invoice_processor.query_invoices("InvoiceType eq 'ingreso'")
        income_total = self._sum_totals(search_results)
        logger.info(f"   - Total de ingresos encontrado: {income_total}")
        return {"income_total": income_total}

    # --- NODO 3b: Buscar Egresos (para cálculo de balance) ---
    async def search_expense_node(self, state: AgentState) -> dict:
        logger.info("💸 Nodo 3b: Buscando todos los egresos...")
        search_results = await invoice_processor.query_invoices("InvoiceType eq 'egreso'")
        expense_total = self._sum_totals(search_results)
        logger.info(f"   - Total de egresos encontrado: {expense_total}")
        return {"expense_total": expense_total}

    # --- NODO 4: Generar Respuesta (para todos los flujos) ---
    async def generate_answer_node(self, state: AgentState) -> dict:
        logger.info("✍️ Nodo 4: Generando respuesta final...")
        question = state["question"]
        task_type = state["task_type"]

        # Si el flujo fue de cálculo de balance, usamos esos datos
        if task_type in ["calculo_balance", "resumen_general"]:
            income = state.get("income_total", 0.0)
            expense = state.get("expense_total", 0.0)
            balance = income - expense
            final_answer = (
                f"He calculado el balance general basado en todas las facturas:\n"
                f"- Total de Ingresos: ${income:,.2f}\n"
                f"- Total de Egresos: ${expense:,.2f}\n"
                f"-----------------------------------\n"
                f"**Balance General: ${balance:,.2f}**"
            )
            logger.info("   - Respuesta de balance generada.")
            return {"final_answer": final_answer}
        
        # Si el flujo fue de búsqueda simple, usamos los resultados
        search_results = state.get("search_results", [])
        if not search_results:
            final_answer = "No encontré información relevante para tu pregunta. Intenta ser más específico."
            logger.info("   - No hay resultados, generando respuesta por defecto.")
            return {"final_answer": final_answer}

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Eres un asistente contable amigable y directo. Tu tarea es responder la pregunta del usuario basándote únicamente en los datos de las facturas que se te proporcionan. "
             "Resume la información de forma clara y, si hay montos, súmalos para dar un total. Responde en español."
            ),
            ("user", "Pregunta del usuario: {question}\n\n"
                     "Estos son los datos de las facturas encontradas:\n{search_results}")
        ])
        chain = prompt | self.llm
        response = await chain.ainvoke({"question": question, "search_results": str(search_results)})
        final_answer = response.content
        logger.info(f"   - Respuesta de búsqueda simple generada.")
        return {"final_answer": final_answer}

    # --- NODO DE "NO SÉ QUÉ HACER" ---
    def unsupported_node(self, state: AgentState) -> dict:
        final_answer = "No estoy seguro de cómo procesar esa pregunta. Por favor, intenta preguntarme sobre gastos, ingresos o un balance general."
        logger.warning(f"   - Tarea no soportada: {state['task_type']}")
        return {"final_answer": final_answer}

    # --- Definimos la Lógica Condicional del Grafo ---
    def decide_path(self, state: AgentState) -> Literal["simple", "balance", "unsupported"]:
        task_type = state.get("task_type", "")
        if task_type == "busqueda_simple":
            return "simple"
        if task_type in ["calculo_balance", "resumen_general"]:
            return "balance"
        return "unsupported"

    # --- Ensamblamos el Grafo con el Router ---
    def _build_graph(self):
        workflow = StateGraph(AgentState)

        # Nodos
        workflow.add_node("router", self.route_question_node)
        workflow.add_node("generate_filter", self.generate_filter_node)
        workflow.add_node("execute_search", self.execute_search_node)
        workflow.add_node("search_income", self.search_income_node)
        workflow.add_node("search_expense", self.search_expense_node)
        workflow.add_node("generate_answer", self.generate_answer_node)
        workflow.add_node("unsupported", self.unsupported_node)

        # Flujo
        workflow.set_entry_point("router")
        workflow.add_conditional_edges(
            "router",
            self.decide_path,
            {
                "simple": "generate_filter",
                "balance": "search_income",
                "unsupported": "unsupported"
            }
        )
        
        # Rama de Búsqueda Simple
        workflow.add_edge("generate_filter", "execute_search")
        workflow.add_edge("execute_search", "generate_answer")

        # Rama de Cálculo de Balance
        workflow.add_edge("search_income", "search_expense")
        workflow.add_edge("search_expense", "generate_answer")

        # Puntos finales
        workflow.add_edge("generate_answer", END)
        workflow.add_edge("unsupported", END)
        
        return workflow.compile()

    # --- Método principal para ejecutar el grafo ---
    async def run(self, question: str) -> dict:
        initial_state = {
            "question": question, 
            "income_total": 0.0, 
            "expense_total": 0.0, 
            "search_results": []
        }
        final_state = await self.graph.ainvoke(initial_state)
        return final_state

# Instancia global del agente
agent_graph = AgentGraph()
