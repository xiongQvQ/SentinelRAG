# Claude Code Project Configuration

## Environment Setup

### Virtual Environment Activation

**Standard Method** (Recommended):
```bash
source venv/bin/activate
```

On Windows:
```bash
venv\Scripts\activate
```

### Python Version
- Python 3.10.18+

### Project Path
- Working Directory: `/Users/xiongbojian/work/readytensor/readytensor-agentic-ai-demo`

---

## Project Structure

```
readytensor-agentic-ai-demo/
├── src/
│   ├── observability/          # Monitoring & tracing
│   │   ├── __init__.py
│   │   ├── langfuse_tracer.py
│   │   ├── opentelemetry_config.py
│   │   ├── metrics_registry.py
│   │   ├── hallucination_detector.py
│   │   ├── latency_tracker.py
│   │   └── cost_calculator.py
│   ├── resilience/             # Reliability features
│   │   ├── __init__.py
│   │   ├── retry_policy.py
│   │   └── timeout_manager.py
│   ├── monitoring/             # Health checks
│   │   ├── __init__.py
│   │   └── health_endpoint.py
│   ├── security/               # Security features
│   │   ├── __init__.py
│   │   ├── rate_limiter.py
│   │   ├── audit_logger.py
│   │   └── guardrails_integration.py
│   ├── data_collector.py
│   ├── vector_store.py
│   ├── rag_pipeline.py
│   └── rag_pipeline_gemini.py
├── config/
│   ├── prometheus/
│   │   └── prometheus.yml
│   └── grafana/
│       └── dashboards/
│           └── rag_overview.json
├── tests/
├── app_gemini.py               # Basic Streamlit app
├── app_gemini_secure.py        # Secure version (with Guardrails)
├── test_week2_modules.py       # Test suite
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## Common Commands

### Activate Environment
```bash
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
```

### Install Dependencies
```bash
# All dependencies
pip install -r requirements.txt

# Verify installation
pip list | grep -E "langfuse|opentelemetry|prometheus|guardrails"
```

### Run Tests

⚠️ **Important**: Tests must be run in the virtual environment!

```bash
# Activate environment first
source venv/bin/activate

# Run all tests
python test_week2_modules.py

# Import test
python -c "from src.observability import HallucinationDetector; print('✅ OK')"
```

### Start Services

#### Streamlit Application
```bash
# Basic version
streamlit run app_gemini.py --server.headless=true --server.port=8501

# Secure version (with Guardrails AI)
streamlit run app_gemini_secure.py --server.headless=true --server.port=8502
```

#### Health Check Service
```bash
python health_server.py
# Access: http://localhost:8080/health
```

#### Prometheus
```bash
cd prometheus-*
./prometheus --config.file=../config/prometheus/prometheus.yml
# Access: http://localhost:9090
```

#### Grafana
```bash
brew services start grafana
# Access: http://localhost:3000 (admin/admin)
```

---

## Key Features

### Security Layer
- ✅ Guardrails AI integration for LLM safety
- ✅ Rate limiting (Token Bucket algorithm)
- ✅ Audit logging (structured JSON logs)
- ✅ Input/Output validation

### Observability Layer
- ✅ Hallucination detection (Faithfulness Score: 0.87)
- ✅ Latency tracking (component-level breakdown)
- ✅ Cost calculation ($0.000333/query)
- ✅ Prometheus metrics (4591 bytes export)
- ✅ OpenTelemetry (Span tracing)
- ✅ Langfuse (LLM call tracing with graceful degradation)

### Resilience Layer
- ✅ Retry policies (exponential backoff)
- ✅ Timeout management (context-based)

### Monitoring Layer
- ✅ Health check endpoints (FastAPI)
- ✅ Grafana dashboards (12 panels)

---

## Testing

### Test Status
**Last Run**: 2025-10-20
**Status**: ✅ 18/18 passed (100%)
**Success Rate**: 100%

### Key Validations
- ✅ Hallucination Detection: Faithfulness Score 0.87
- ✅ Latency Tracking: Component-level breakdown normal
- ✅ Cost Calculation: $0.000333/query
- ✅ Prometheus Metrics: 4591 bytes exported
- ✅ Retry Mechanism: Exponential backoff normal
- ✅ Timeout Management: Context management normal
- ✅ Health Check: FastAPI endpoints normal
- ✅ OpenTelemetry: Span tracing normal
- ✅ Langfuse: Graceful degradation normal

---

## Troubleshooting

### Issue: Test failures with "No module named 'faiss'"

**Cause**: Virtual environment not activated

**Solution**:
```bash
source venv/bin/activate
python test_week2_modules.py
```

### Issue: Langfuse warnings

This is **normal**! Langfuse is optional. To enable:

```bash
export LANGFUSE_PUBLIC_KEY="pk_..."
export LANGFUSE_SECRET_KEY="sk_..."
export LANGFUSE_HOST="https://cloud.langfuse.com"
```

### Issue: Dependency conflicts

```bash
# Reinstall all dependencies
pip install --upgrade -r requirements.txt
```

---

## Dependencies

### Core Dependencies
- `langchain==0.1.0`
- `faiss-cpu==1.7.4`
- `sentence-transformers==2.3.1`
- `streamlit==1.29.0`

### Security (Guardrails AI)
- `guardrails-ai==0.6.7`
- `slowapi==0.1.9`
- `python-json-logger==2.0.7`

### Observability
- `langfuse==2.12.0` (LLM tracing)
- `opentelemetry-api==1.22.0` (distributed tracing)
- `prometheus-client==0.19.0` (metrics)
- `scikit-learn==1.3.2` (hallucination detection)

### Resilience
- `tenacity==8.2.3` (retry logic)

### Monitoring
- `fastapi==0.109.0` (health check endpoints)
- `uvicorn==0.27.0` (ASGI server)

---

## Performance Metrics

- **Query Response**: 2-5 seconds
- **Observability Overhead**: ~70-130ms (~7-13%)
- **Cost per Query**: ~$0.00033 (Gemini 1.5 Flash)
- **Monthly Cost** (1000 queries/day): ~$10

---

## Notes

- ✅ Core RAG functionality implemented and tested
- ✅ Security layer with Guardrails AI
- ✅ Observability with hallucination detection
- ✅ Resilience mechanisms (retry + timeout)
- ✅ Monitoring with Prometheus + Grafana
- ✅ All 18 tests passing (100%)

**Last Updated**: 2025-10-20
**Version**: 2.0
