"""
Monitoring module for health checks and system status.

This module provides:
- FastAPI health check endpoints
- System status monitoring
- Component health verification

Based on ReadyTensor Week 2 monitoring best practices.
"""

try:
    from .health_endpoint import (
        HealthCheckEndpoint,
        HealthStatus,
        ComponentHealth,
        create_health_app
    )
    HEALTH_CHECK_AVAILABLE = True
except ImportError:
    HEALTH_CHECK_AVAILABLE = False
    HealthCheckEndpoint = None
    HealthStatus = None
    ComponentHealth = None
    create_health_app = None

__all__ = [
    'HealthCheckEndpoint',
    'HealthStatus',
    'ComponentHealth',
    'create_health_app',
    'HEALTH_CHECK_AVAILABLE',
]
