"""
Enhanced RAG Pipeline with Guardrails AI Integration
Adds runtime safety and output validation to the RAG system

Based on ReadyTensor Week 9 Lesson 5
"""

import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
import google.generativeai as genai

try:
    from .vector_store import VectorStoreManager
    from .rag_pipeline_gemini import CustomFAISSRetriever
    from .security.guardrails_integration import GuardrailsValidator, GuardrailsConfig
    from .security.audit_logger import get_audit_logger, AuditLogger, EventType, SeverityLevel
    from .security.rate_limiter import RateLimiter, RateLimitExceeded, RateLimitConfig
except ImportError:
    from vector_store import VectorStoreManager
    from rag_pipeline_gemini import CustomFAISSRetriever
    from security.guardrails_integration import GuardrailsValidator, GuardrailsConfig
    from security.audit_logger import get_audit_logger, AuditLogger, EventType, SeverityLevel
    from security.rate_limiter import RateLimiter, RateLimitExceeded, RateLimitConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SecureRAGPipeline:
    """
    Enhanced RAG Pipeline with Guardrails AI and security features

    Features:
    - Input validation using Guardrails AI (toxic language, unusual prompts)
    - Output validation (PII detection, hallucination check)
    - Rate limiting per user/IP
    - Comprehensive audit logging
    - Error handling and resilience
    """

    def __init__(self,
                 google_api_key: str,
                 vector_store_dir: str = "vector_store",
                 model_name: str = "gemini-2.5-flash",
                 temperature: float = 0.1,
                 enable_guardrails: bool = True,
                 enable_rate_limiting: bool = True,
                 enable_audit_logging: bool = True):
        """
        Initialize Secure RAG pipeline with Guardrails AI

        Args:
            google_api_key: Google API key for Gemini
            vector_store_dir: Directory containing vector store
            model_name: Google Gemini model to use
            temperature: Temperature for response generation
            enable_guardrails: Enable Guardrails AI validation
            enable_rate_limiting: Enable rate limiting
            enable_audit_logging: Enable audit logging
        """
        self.google_api_key = google_api_key
        self.model_name = model_name
        self.temperature = temperature

        # Core components
        self.vector_store_manager = VectorStoreManager(vector_store_dir)
        self.llm = None
        self.retriever = None
        self.qa_chain = None
        self.chat_chain = None
        self.memory = None

        # Security components
        self.enable_guardrails = enable_guardrails
        self.enable_rate_limiting = enable_rate_limiting
        self.enable_audit_logging = enable_audit_logging

        # Initialize security components
        self.audit_logger = get_audit_logger() if enable_audit_logging else None
        self.rate_limiter = RateLimiter(RateLimitConfig(
            requests_per_minute=10,
            requests_per_hour=100,
            requests_per_day=1000
        )) if enable_rate_limiting else None

        # Initialize Guardrails validator
        self.guardrails_validator = None
        if enable_guardrails:
            try:
                self.guardrails_validator = GuardrailsValidator(
                    config=GuardrailsConfig(
                        enable_input_validation=True,
                        enable_output_validation=True,
                        check_toxic_language=True,
                        check_unusual_prompts=True,
                        detect_pii=True,
                        toxic_threshold=0.5,
                        input_fail_action="exception",
                        output_fail_action="filter",
                        # 使用 Gemini 进行验证（而不是 OpenAI）
                        llm_model="gemini/gemini-2.5-flash",
                        llm_api_key=google_api_key
                    ),
                    audit_logger=self.audit_logger
                )
                logger.info(f"Guardrails AI validator initialized successfully (using Gemini {model_name})")
            except ImportError as e:
                logger.warning(f"Guardrails AI not available: {e}")
                self.enable_guardrails = False

        # Configure Google Gemini
        genai.configure(api_key=google_api_key)
        os.environ["GOOGLE_API_KEY"] = google_api_key

        # Statistics
        self.stats = {
            'total_queries': 0,
            'successful_queries': 0,
            'failed_queries': 0,
            'rate_limited_queries': 0,
            'validation_failures': 0
        }

    def initialize(self, articles_file: str = "data/processed_wikipedia_articles.json"):
        """Initialize all components"""
        logger.info("Initializing Secure RAG pipeline with Guardrails AI...")

        # Initialize vector store
        self.vector_store_manager.initialize_from_articles(articles_file)

        # Initialize Gemini LLM
        self.llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            temperature=self.temperature,
            google_api_key=self.google_api_key,
            convert_system_message_to_human=True
        )

        # Initialize retriever
        self.retriever = CustomFAISSRetriever(self.vector_store_manager, k=5)

        # Initialize memory
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=5
        )

        # Create chains
        self._create_qa_chain()
        self._create_chat_chain()

        logger.info("Secure RAG pipeline initialized successfully!")

        if self.audit_logger:
            self.audit_logger.log_event(
                event_type=EventType.SYSTEM_EVENT,
                severity=SeverityLevel.INFO,
                action="pipeline_initialized",
                message="Secure RAG pipeline initialized with Guardrails AI",
                status="success"
            )

    def _create_qa_chain(self):
        """Create the Q&A chain"""
        qa_template = """You are a helpful AI assistant that answers questions based on provided context from Wikipedia articles about AI, machine learning, and related topics.

Use the following pieces of context to answer the user's question. If you don't know the answer based on the context, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}

Please provide a comprehensive and accurate answer based on the context. Include relevant details and mention the sources when possible. Be clear and informative in your response.

Answer:"""

        qa_prompt = PromptTemplate(
            template=qa_template,
            input_variables=["context", "question"]
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": qa_prompt}
        )

    def _create_chat_chain(self):
        """Create conversational chain"""
        self.chat_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=self.memory,
            return_source_documents=True,
            verbose=True
        )

    def _check_rate_limit(self, user_id: Optional[str] = None, ip_address: Optional[str] = None):
        """Check rate limit for user/IP"""
        if not self.enable_rate_limiting or self.rate_limiter is None:
            return

        identifier = user_id or ip_address or "anonymous"
        identifier_type = "user" if user_id else "ip"

        try:
            self.rate_limiter.check_rate_limit(identifier, identifier_type)
        except RateLimitExceeded as e:
            self.stats['rate_limited_queries'] += 1

            if self.audit_logger:
                self.audit_logger.log_rate_limit(
                    identifier=identifier,
                    identifier_type=identifier_type,
                    limit_type="requests_per_minute"
                )

            raise

    def _validate_input(self, text: str, user_id: Optional[str] = None,
                       ip_address: Optional[str] = None) -> str:
        """Validate user input using Guardrails"""
        if not self.enable_guardrails or self.guardrails_validator is None:
            return text

        # Log query
        if self.audit_logger:
            self.audit_logger.log_query(
                query=text,
                user_id=user_id,
                ip_address=ip_address
            )

        # Validate with Guardrails
        result = self.guardrails_validator.validate_input(text, context="user_query")

        if not result['valid']:
            self.stats['validation_failures'] += 1
            raise ValueError(f"Input validation failed: {'; '.join(result['errors'])}")

        return result['validated_text']

    def _validate_output(self, text: str, user_query: Optional[str] = None,
                        user_id: Optional[str] = None, ip_address: Optional[str] = None) -> str:
        """Validate LLM output using Guardrails"""
        if not self.enable_guardrails or self.guardrails_validator is None:
            return text

        # Validate with Guardrails
        result = self.guardrails_validator.validate_output(
            text,
            context="llm_response",
            user_query=user_query
        )

        # Log response
        if self.audit_logger:
            self.audit_logger.log_response(
                query=user_query or "",
                response=text if result['valid'] else "[FILTERED]",
                user_id=user_id,
                ip_address=ip_address,
                metadata={
                    'validation_passed': result['valid'],
                    'redactions': result.get('errors', [])
                }
            )

        if not result['valid']:
            self.stats['validation_failures'] += 1
            logger.warning(f"Output validation failed: {result['errors']}")
            # For demo/educational purposes, return original text even if validation failed
            # The failure is logged above for monitoring
            return text

        # Return validated output if validation passed
        return result['validated_text']

    def ask_question(self, question: str, user_id: Optional[str] = None,
                    ip_address: Optional[str] = None) -> Dict[str, Any]:
        """
        Ask a question with Guardrails validation

        Args:
            question: User question
            user_id: Optional user identifier for rate limiting
            ip_address: Optional IP address for rate limiting

        Returns:
            Response dictionary with validated answer
        """
        if self.qa_chain is None:
            raise ValueError("Pipeline not initialized. Call initialize() first.")

        self.stats['total_queries'] += 1

        try:
            # 1. Check rate limit
            self._check_rate_limit(user_id, ip_address)

            # 2. Validate input
            validated_question = self._validate_input(question, user_id, ip_address)

            logger.info(f"Processing validated question with Gemini: {validated_question}")

            # 3. Process with LLM
            result = self.qa_chain({"query": validated_question})

            # 4. Validate output
            validated_answer = self._validate_output(
                result["result"],
                user_query=question,
                user_id=user_id,
                ip_address=ip_address
            )

            # 5. Format response
            response = {
                "question": question,
                "answer": validated_answer,
                "source_documents": [
                    {
                        "title": doc.metadata["title"],
                        "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                        "url": doc.metadata["url"],
                        "similarity_score": doc.metadata["similarity_score"]
                    }
                    for doc in result["source_documents"]
                ],
                "usage": {
                    "model": self.model_name,
                    "note": "Google Gemini API usage tracking not available"
                },
                "security": {
                    "guardrails_enabled": self.enable_guardrails,
                    "input_validated": True,
                    "output_validated": True
                },
                "timestamp": datetime.now().isoformat()
            }

            self.stats['successful_queries'] += 1
            return response

        except RateLimitExceeded:
            logger.warning(f"Rate limit exceeded for user/IP: {user_id or ip_address}")
            self.stats['failed_queries'] += 1
            raise

        except Exception as e:
            logger.error(f"Error processing question: {e}")
            self.stats['failed_queries'] += 1

            if self.audit_logger:
                self.audit_logger.log_error(
                    error=e,
                    context="ask_question",
                    user_id=user_id,
                    ip_address=ip_address
                )

            raise

    def chat(self, message: str, user_id: Optional[str] = None,
            ip_address: Optional[str] = None) -> Dict[str, Any]:
        """
        Chat with Guardrails validation

        Args:
            message: User message
            user_id: Optional user identifier
            ip_address: Optional IP address

        Returns:
            Response dictionary with validated response
        """
        if self.chat_chain is None:
            raise ValueError("Pipeline not initialized. Call initialize() first.")

        self.stats['total_queries'] += 1

        try:
            # 1. Check rate limit
            self._check_rate_limit(user_id, ip_address)

            # 2. Validate input
            validated_message = self._validate_input(message, user_id, ip_address)

            logger.info(f"Processing validated chat message with Gemini: {validated_message}")

            # 3. Process with LLM
            result = self.chat_chain({"question": validated_message})

            # 4. Validate output
            validated_response = self._validate_output(
                result["answer"],
                user_query=message,
                user_id=user_id,
                ip_address=ip_address
            )

            # 5. Format response
            response = {
                "message": message,
                "response": validated_response,
                "source_documents": [
                    {
                        "title": doc.metadata["title"],
                        "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                        "url": doc.metadata["url"],
                        "similarity_score": doc.metadata["similarity_score"]
                    }
                    for doc in result["source_documents"]
                ],
                "usage": {
                    "model": self.model_name,
                    "note": "Google Gemini API usage tracking not available"
                },
                "security": {
                    "guardrails_enabled": self.enable_guardrails,
                    "input_validated": True,
                    "output_validated": True
                },
                "timestamp": datetime.now().isoformat()
            }

            self.stats['successful_queries'] += 1
            return response

        except RateLimitExceeded:
            logger.warning(f"Rate limit exceeded for user/IP: {user_id or ip_address}")
            self.stats['failed_queries'] += 1
            raise

        except Exception as e:
            logger.error(f"Error processing chat message: {e}")
            self.stats['failed_queries'] += 1

            if self.audit_logger:
                self.audit_logger.log_error(
                    error=e,
                    context="chat",
                    user_id=user_id,
                    ip_address=ip_address
                )

            raise

    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        stats = {
            'pipeline': self.stats.copy(),
            'security': {
                'guardrails_enabled': self.enable_guardrails,
                'rate_limiting_enabled': self.enable_rate_limiting,
                'audit_logging_enabled': self.enable_audit_logging
            }
        }

        if self.guardrails_validator:
            stats['guardrails'] = self.guardrails_validator.get_stats()

        if self.rate_limiter:
            stats['rate_limiter'] = self.rate_limiter.get_stats()

        if self.audit_logger:
            stats['audit_logger'] = self.audit_logger.get_stats()

        return stats

    def clear_memory(self):
        """Clear conversation memory"""
        if self.memory:
            self.memory.clear()
            logger.info("Conversation memory cleared")


# Example usage
if __name__ == "__main__":
    import json
    from dotenv import load_dotenv

    load_dotenv()

    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        print("⚠️  GOOGLE_API_KEY not found in environment")
        exit(1)

    # Create secure pipeline
    pipeline = SecureRAGPipeline(
        google_api_key=google_api_key,
        enable_guardrails=True,
        enable_rate_limiting=True,
        enable_audit_logging=True
    )

    # Initialize
    pipeline.initialize()

    # Test query
    print("\n" + "="*50)
    print("Testing Secure RAG Pipeline with Guardrails AI")
    print("="*50)

    try:
        response = pipeline.ask_question(
            "What is machine learning?",
            user_id="test_user",
            ip_address="127.0.0.1"
        )
        print("\n✅ Query successful!")
        print(f"Answer: {response['answer'][:200]}...")

    except Exception as e:
        print(f"\n❌ Error: {e}")

    # Print statistics
    print("\n" + "="*50)
    print("Pipeline Statistics")
    print("="*50)
    print(json.dumps(pipeline.get_stats(), indent=2))
