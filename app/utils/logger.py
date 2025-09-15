"""
Configuración de logging para mfn-mvp
"""

import logging
import structlog
from typing import Optional
from app.config.settings import settings

def setup_logging(log_level: Optional[str] = None) -> None:
    """
    Configurar el sistema de logging de la aplicación
    
    Args:
        log_level: Nivel de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    if log_level is None:
        log_level = settings.LOG_LEVEL
    
    # Configurar structlog para logging estructurado
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configurar logging estándar
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=getattr(logging, log_level.upper()),
        handlers=[
            logging.StreamHandler(),
        ]
    )

def get_logger(name: str) -> structlog.BoundLogger:
    """
    Obtener un logger configurado para un módulo específico
    
    Args:
        name: Nombre del módulo (generalmente __name__)
        
    Returns:
        Logger configurado
    """
    return structlog.get_logger(name)

# Configurar logging al importar el módulo
setup_logging()
