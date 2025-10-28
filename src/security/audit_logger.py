"""
Audit Logger for RAG System
Comprehensive logging for security auditing and compliance
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path

# Try to import python-json-logger, fall back to standard logging
try:
    from pythonjsonlogger import jsonlogger
    HAS_JSON_LOGGER = True
except ImportError:
    HAS_JSON_LOGGER = False
    logging.warning("python-json-logger not installed, using standard logging")


class EventType(Enum):
    """Types of audit events"""
    QUERY = "query"
    RESPONSE = "response"
    ERROR = "error"
    SECURITY_VIOLATION = "security_violation"
    RATE_LIMIT = "rate_limit"
    ACCESS = "access"
    CONFIG_CHANGE = "config_change"
    SYSTEM_EVENT = "system_event"


class SeverityLevel(Enum):
    """Severity levels for audit events"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AuditEvent:
    """Audit event data structure"""
    timestamp: str
    event_type: str
    severity: str
    user_id: Optional[str]
    ip_address: Optional[str]
    action: str
    resource: Optional[str]
    status: str
    message: str
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class AuditConfig:
    """Configuration for audit logging"""
    log_dir: str = "logs"
    log_file: str = "audit.log"
    json_format: bool = True
    console_output: bool = True
    file_output: bool = True
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    log_queries: bool = True
    log_responses: bool = True
    log_errors: bool = True
    log_security_events: bool = True


class AuditLogger:
    """
    Comprehensive audit logger for security and compliance

    Features:
    - Structured logging (JSON)
    - Multiple output destinations
    - Event categorization
    - Severity levels
    - User and IP tracking
    - Rotation support
    """

    def __init__(self, config: Optional[AuditConfig] = None):
        """
        Initialize audit logger

        Args:
            config: Audit configuration (uses defaults if None)
        """
        self.config = config or AuditConfig()

        # Create log directory
        os.makedirs(self.config.log_dir, exist_ok=True)

        # Setup logger
        self.logger = logging.getLogger('audit')
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers = []  # Clear existing handlers

        # File handler
        if self.config.file_output:
            log_file_path = os.path.join(self.config.log_dir, self.config.log_file)

            if HAS_JSON_LOGGER and self.config.json_format:
                # JSON formatter
                file_handler = logging.handlers.RotatingFileHandler(
                    log_file_path,
                    maxBytes=self.config.max_file_size,
                    backupCount=self.config.backup_count
                )
                formatter = jsonlogger.JsonFormatter(
                    '%(timestamp)s %(level)s %(name)s %(message)s'
                )
                file_handler.setFormatter(formatter)
            else:
                # Standard formatter
                file_handler = logging.handlers.RotatingFileHandler(
                    log_file_path,
                    maxBytes=self.config.max_file_size,
                    backupCount=self.config.backup_count
                )
                formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
                file_handler.setFormatter(formatter)

            self.logger.addHandler(file_handler)

        # Console handler
        if self.config.console_output:
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)

        # Statistics
        self.event_counts = {event_type.value: 0 for event_type in EventType}
        self.total_events = 0

    def log_event(self,
                  event_type: EventType,
                  severity: SeverityLevel,
                  action: str,
                  message: str,
                  user_id: Optional[str] = None,
                  ip_address: Optional[str] = None,
                  resource: Optional[str] = None,
                  status: str = "success",
                  metadata: Optional[Dict[str, Any]] = None):
        """
        Log an audit event

        Args:
            event_type: Type of event
            severity: Severity level
            action: Action performed
            message: Human-readable message
            user_id: User identifier
            ip_address: IP address
            resource: Resource accessed
            status: Status of the action
            metadata: Additional metadata
        """
        # Create audit event
        event = AuditEvent(
            timestamp=datetime.now().isoformat(),
            event_type=event_type.value,
            severity=severity.value,
            user_id=user_id,
            ip_address=ip_address,
            action=action,
            resource=resource,
            status=status,
            message=message,
            metadata=metadata
        )

        # Convert to log message
        if self.config.json_format and HAS_JSON_LOGGER:
            log_msg = event.to_dict()
        else:
            log_msg = (
                f"[{event.event_type}] {event.action} - {event.message} "
                f"(user: {event.user_id}, ip: {event.ip_address}, status: {event.status})"
            )

        # Log with appropriate level
        level_map = {
            SeverityLevel.DEBUG: logging.DEBUG,
            SeverityLevel.INFO: logging.INFO,
            SeverityLevel.WARNING: logging.WARNING,
            SeverityLevel.ERROR: logging.ERROR,
            SeverityLevel.CRITICAL: logging.CRITICAL
        }

        log_level = level_map.get(severity, logging.INFO)
        self.logger.log(log_level, log_msg)

        # Update statistics
        self.event_counts[event_type.value] += 1
        self.total_events += 1

    def log_query(self, query: str, user_id: Optional[str] = None,
                  ip_address: Optional[str] = None, metadata: Optional[Dict] = None):
        """Log a user query"""
        if not self.config.log_queries:
            return

        self.log_event(
            event_type=EventType.QUERY,
            severity=SeverityLevel.INFO,
            action="user_query",
            message=f"Query received: {query[:100]}...",
            user_id=user_id,
            ip_address=ip_address,
            resource="rag_system",
            status="received",
            metadata={
                'query': query,
                'query_length': len(query),
                **(metadata or {})
            }
        )

    def log_response(self, query: str, response: str, user_id: Optional[str] = None,
                    ip_address: Optional[str] = None, metadata: Optional[Dict] = None):
        """Log a system response"""
        if not self.config.log_responses:
            return

        self.log_event(
            event_type=EventType.RESPONSE,
            severity=SeverityLevel.INFO,
            action="system_response",
            message=f"Response generated for query: {query[:50]}...",
            user_id=user_id,
            ip_address=ip_address,
            resource="rag_system",
            status="success",
            metadata={
                'query': query[:200],
                'response_length': len(response),
                **(metadata or {})
            }
        )

    def log_error(self, error: Exception, context: str, user_id: Optional[str] = None,
                 ip_address: Optional[str] = None, metadata: Optional[Dict] = None):
        """Log an error"""
        if not self.config.log_errors:
            return

        self.log_event(
            event_type=EventType.ERROR,
            severity=SeverityLevel.ERROR,
            action="error_occurred",
            message=f"Error in {context}: {str(error)}",
            user_id=user_id,
            ip_address=ip_address,
            resource=context,
            status="error",
            metadata={
                'error_type': type(error).__name__,
                'error_message': str(error),
                **(metadata or {})
            }
        )

    def log_security_violation(self, violation_type: str, details: str,
                              user_id: Optional[str] = None,
                              ip_address: Optional[str] = None,
                              metadata: Optional[Dict] = None):
        """Log a security violation"""
        if not self.config.log_security_events:
            return

        self.log_event(
            event_type=EventType.SECURITY_VIOLATION,
            severity=SeverityLevel.WARNING,
            action="security_violation",
            message=f"{violation_type}: {details}",
            user_id=user_id,
            ip_address=ip_address,
            resource="security",
            status="blocked",
            metadata={
                'violation_type': violation_type,
                'details': details,
                **(metadata or {})
            }
        )

    def log_rate_limit(self, identifier: str, identifier_type: str,
                      limit_type: str, metadata: Optional[Dict] = None):
        """Log a rate limit event"""
        self.log_event(
            event_type=EventType.RATE_LIMIT,
            severity=SeverityLevel.WARNING,
            action="rate_limit_exceeded",
            message=f"Rate limit exceeded for {identifier_type}: {identifier}",
            user_id=identifier if identifier_type == 'user' else None,
            ip_address=identifier if identifier_type == 'ip' else None,
            resource="rate_limiter",
            status="blocked",
            metadata={
                'identifier': identifier,
                'identifier_type': identifier_type,
                'limit_type': limit_type,
                **(metadata or {})
            }
        )

    def log_access(self, user_id: str, resource: str, action: str,
                   status: str = "success", ip_address: Optional[str] = None,
                   metadata: Optional[Dict] = None):
        """Log resource access"""
        self.log_event(
            event_type=EventType.ACCESS,
            severity=SeverityLevel.INFO,
            action=action,
            message=f"User {user_id} {action} {resource}",
            user_id=user_id,
            ip_address=ip_address,
            resource=resource,
            status=status,
            metadata=metadata
        )

    def get_stats(self) -> Dict[str, Any]:
        """
        Get audit logger statistics

        Returns:
            Dictionary with logger stats
        """
        return {
            'total_events': self.total_events,
            'events_by_type': self.event_counts,
            'log_file': os.path.join(self.config.log_dir, self.config.log_file),
            'json_format': self.config.json_format,
            'has_json_logger': HAS_JSON_LOGGER
        }

    def reset_stats(self):
        """Reset statistics counters"""
        self.event_counts = {event_type.value: 0 for event_type in EventType}
        self.total_events = 0


# Global audit logger instance
_global_audit_logger = None


def get_audit_logger(config: Optional[AuditConfig] = None) -> AuditLogger:
    """
    Get or create global audit logger instance

    Args:
        config: Audit configuration (only used on first call)

    Returns:
        Global AuditLogger instance
    """
    global _global_audit_logger
    if _global_audit_logger is None:
        _global_audit_logger = AuditLogger(config)
    return _global_audit_logger


if __name__ == "__main__":
    # Example usage
    audit_logger = AuditLogger()

    # Log various events
    audit_logger.log_query(
        query="What is machine learning?",
        user_id="user123",
        ip_address="192.168.1.1"
    )

    audit_logger.log_response(
        query="What is machine learning?",
        response="Machine learning is a subset of AI...",
        user_id="user123",
        ip_address="192.168.1.1",
        metadata={'model': 'gemini-1.5-flash', 'tokens': 150}
    )

    audit_logger.log_security_violation(
        violation_type="prompt_injection",
        details="Detected instruction override attempt",
        user_id="user456",
        ip_address="192.168.1.2"
    )

    audit_logger.log_rate_limit(
        identifier="user789",
        identifier_type="user",
        limit_type="requests_per_minute"
    )

    # Print statistics
    print("Audit Logger Statistics:", json.dumps(audit_logger.get_stats(), indent=2))
