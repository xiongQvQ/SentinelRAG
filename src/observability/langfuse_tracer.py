"""
Langfuse integration for LLM tracing and observability.

Automatically captures:
- All LLM calls (Gemini)
- Prompts and completions
- Token usage
- Latency
- Cost estimates
- User context

Based on ReadyTensor Week 11 Lesson 1c best practices.
"""

import os
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

try:
    from langfuse import Langfuse
    from langfuse.callback import CallbackHandler
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    Langfuse = None
    CallbackHandler = None

logger = logging.getLogger(__name__)


class LangfuseTracer:
    """
    Wrapper for Langfuse LLM tracing.

    Provides automatic tracing of LangChain operations with:
    - Trace hierarchy (parent-child relationships)
    - Token usage tracking
    - Cost estimation
    - User and session management
    - Custom metadata

    Example:
        tracer = LangfuseTracer()
        callback = tracer.get_callback_handler(
            user_id="user123",
            session_id="session456"
        )

        response = qa_chain(
            {"query": question},
            callbacks=[callback]
        )

        # Flush to ensure data is sent
        tracer.flush()
    """

    def __init__(
        self,
        public_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        host: Optional[str] = None,
        enabled: bool = True
    ):
        """
        Initialize Langfuse tracer.

        Args:
            public_key: Langfuse public API key (or from LANGFUSE_PUBLIC_KEY env)
            secret_key: Langfuse secret API key (or from LANGFUSE_SECRET_KEY env)
            host: Langfuse host URL (or from LANGFUSE_HOST env, defaults to cloud)
            enabled: Whether to enable tracing
        """
        self.enabled = enabled and LANGFUSE_AVAILABLE
        self.client: Optional[Langfuse] = None

        if not self.enabled:
            if not LANGFUSE_AVAILABLE:
                logger.warning("Langfuse not installed. Tracing disabled.")
            else:
                logger.info("Langfuse tracing disabled by configuration.")
            return

        # Get configuration from environment or parameters
        self.public_key = public_key or os.getenv("LANGFUSE_PUBLIC_KEY")
        self.secret_key = secret_key or os.getenv("LANGFUSE_SECRET_KEY")
        self.host = host or os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

        # Initialize Langfuse client
        try:
            self.client = Langfuse(
                public_key=self.public_key,
                secret_key=self.secret_key,
                host=self.host
            )
            logger.info(f"Langfuse tracer initialized (host: {self.host})")
        except Exception as e:
            logger.error(f"Failed to initialize Langfuse: {e}")
            self.enabled = False
            self.client = None

    def get_callback_handler(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        trace_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> Optional[CallbackHandler]:
        """
        Get a Langfuse callback handler for LangChain.

        Args:
            user_id: User identifier for tracking
            session_id: Session identifier for grouping traces
            trace_name: Custom name for this trace
            metadata: Additional metadata to attach
            tags: Tags for filtering/organization

        Returns:
            CallbackHandler instance or None if tracing disabled
        """
        if not self.enabled or not self.client:
            return None

        try:
            # Create callback handler with context
            callback = CallbackHandler(
                public_key=self.public_key,
                secret_key=self.secret_key,
                host=self.host,
                user_id=user_id,
                session_id=session_id,
                trace_name=trace_name or "rag_query",
                metadata=metadata or {},
                tags=tags or []
            )
            return callback
        except Exception as e:
            logger.error(f"Failed to create Langfuse callback: {e}")
            return None

    def create_trace(
        self,
        name: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> Optional[Any]:
        """
        Manually create a trace for custom instrumentation.

        Args:
            name: Trace name
            user_id: User identifier
            session_id: Session identifier
            metadata: Custom metadata
            tags: Tags for organization

        Returns:
            Trace object or None if tracing disabled
        """
        if not self.enabled or not self.client:
            return None

        try:
            trace = self.client.trace(
                name=name,
                user_id=user_id,
                session_id=session_id,
                metadata=metadata or {},
                tags=tags or []
            )
            return trace
        except Exception as e:
            logger.error(f"Failed to create trace: {e}")
            return None

    def flush(self) -> None:
        """
        Flush all pending traces to Langfuse.

        Call this at the end of request processing to ensure
        all trace data is sent to Langfuse immediately.
        """
        if not self.enabled or not self.client:
            return

        try:
            self.client.flush()
            logger.debug("Langfuse traces flushed")
        except Exception as e:
            logger.error(f"Failed to flush Langfuse traces: {e}")

    def shutdown(self) -> None:
        """
        Gracefully shutdown Langfuse client.

        Flushes all pending data and closes connections.
        """
        if not self.enabled or not self.client:
            return

        try:
            self.client.flush()
            logger.info("Langfuse tracer shutdown complete")
        except Exception as e:
            logger.error(f"Error during Langfuse shutdown: {e}")
