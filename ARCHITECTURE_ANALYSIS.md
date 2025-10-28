# RAG Pipeline 架构分析与重构建议

## 🚨 当前问题

### 问题1: 模型不一致
- **基础版本** (`app_gemini.py`): 使用 `gemini-2.5-flash`
- **安全版本** (`app_gemini_secure.py`):
  - 主LLM使用 `gemini-1.5-flash` (已过时)
  - Guardrails AI使用 `gpt-3.5-turbo` (需要OpenAI API key)

### 问题2: 依赖冲突
```
GuardrailsAI → 需要OpenAI API key → gpt-3.5-turbo
用户只有Google API key → gemini-2.5-flash
```

### 问题3: 接口设计问题
- **重复代码**: `rag_pipeline_gemini.py` 和 `rag_pipeline_with_guardrails.py` 有大量重复
- **耦合度高**: 安全功能与核心RAG逻辑耦合
- **配置混乱**: 模型配置分散在多个地方

---

## ✅ 重构方案

### 核心设计原则
1. **单一职责**: RAG pipeline只负责检索和生成
2. **依赖注入**: 安全组件作为可选依赖注入
3. **配置统一**: 所有模型配置在一个地方
4. **复用优先**: 避免重复代码

### 新架构设计

```
┌─────────────────────────────────────────────────┐
│           RAG Pipeline (Core)                    │
│  - model_name: gemini-2.5-flash                 │
│  - 统一的LLM接口                                 │
│  - 向量检索                                      │
│  - 生成答案                                      │
└────────────┬────────────────────────────────────┘
             │
             │ 依赖注入
             ▼
┌─────────────────────────────────────────────────┐
│         Security Layer (Optional)                │
│  ┌────────────────────────────────────────┐    │
│  │  Rate Limiter                          │    │
│  │  - Token bucket algorithm              │    │
│  └────────────────────────────────────────┘    │
│  ┌────────────────────────────────────────┐    │
│  │  Input Validator                       │    │
│  │  - Length check                        │    │
│  │  - SQL injection detection             │    │
│  │  - Prompt injection detection          │    │
│  └────────────────────────────────────────┘    │
│  ┌────────────────────────────────────────┐    │
│  │  Audit Logger                          │    │
│  │  - Structured logging                  │    │
│  │  - Event tracking                      │    │
│  └────────────────────────────────────────┘    │
│  ┌────────────────────────────────────────┐    │
│  │  Guardrails AI (Optional)              │    │
│  │  - 需要OpenAI API key                  │    │
│  │  - 默认禁用                             │    │
│  └────────────────────────────────────────┘    │
└─────────────────────────────────────────────────┘
```

---

## 📝 重构后的接口设计

### 1. 统一配置类

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    """统一的模型配置"""
    # 主LLM模型
    model_name: str = "gemini-2.5-flash"
    temperature: float = 0.1
    max_tokens: Optional[int] = None

    # 可选：Guardrails验证模型（需要OpenAI key）
    guardrails_model: Optional[str] = None  # 默认禁用

    @classmethod
    def gemini_2_5_flash(cls):
        """预设：Gemini 2.5 Flash"""
        return cls(model_name="gemini-2.5-flash")

    @classmethod
    def gemini_2_5_pro(cls):
        """预设：Gemini 2.5 Pro"""
        return cls(model_name="gemini-2.5-pro", temperature=0.2)


@dataclass
class SecurityConfig:
    """安全配置"""
    enable_rate_limiting: bool = True
    enable_input_validation: bool = True
    enable_audit_logging: bool = True
    enable_guardrails: bool = False  # 默认禁用（需要额外配置）

    # Rate limiting
    requests_per_minute: int = 10
    requests_per_hour: int = 100

    # Input validation
    max_query_length: int = 1000
    block_sql_injection: bool = True
    block_prompt_injection: bool = True
```

### 2. 核心RAG Pipeline（可复用）

```python
class RAGPipeline:
    """
    核心RAG Pipeline - 只负责检索和生成

    设计原则:
    - 单一职责：专注于RAG核心功能
    - 配置驱动：通过ModelConfig统一配置
    - 安全无关：不包含安全逻辑
    """

    def __init__(
        self,
        google_api_key: str,
        config: ModelConfig,
        vector_store_dir: str = "vector_store"
    ):
        self.config = config
        self.google_api_key = google_api_key

        # 初始化LLM（统一使用Gemini）
        self.llm = ChatGoogleGenerativeAI(
            model=config.model_name,
            temperature=config.temperature,
            google_api_key=google_api_key
        )

        # 初始化向量存储
        self.vector_store_manager = VectorStoreManager(vector_store_dir)
        self.retriever = None

    def query(self, question: str) -> Dict[str, Any]:
        """
        执行RAG查询

        Returns:
            {
                'answer': str,
                'sources': List[Dict],
                'metadata': Dict  # 包含延迟、tokens等
            }
        """
        # 核心RAG逻辑
        pass
```

### 3. 安全包装器（可选组合）

```python
class SecureRAGPipeline:
    """
    安全包装器 - 在RAG基础上添加安全层

    设计原则:
    - 装饰器模式：包装核心pipeline
    - 可选组合：每个安全组件可独立启用/禁用
    - 透明代理：不改变核心接口
    """

    def __init__(
        self,
        pipeline: RAGPipeline,  # 依赖注入核心pipeline
        security_config: SecurityConfig,
        openai_api_key: Optional[str] = None
    ):
        self.pipeline = pipeline
        self.config = security_config

        # 初始化安全组件（按需启用）
        self.rate_limiter = RateLimiter() if security_config.enable_rate_limiting else None
        self.input_validator = InputValidator() if security_config.enable_input_validation else None
        self.audit_logger = AuditLogger() if security_config.enable_audit_logging else None

        # Guardrails（需要OpenAI key）
        self.guardrails = None
        if security_config.enable_guardrails:
            if not openai_api_key:
                logger.warning("Guardrails需要OpenAI API key，已禁用")
            else:
                self.guardrails = GuardrailsValidator(openai_api_key)

    def query(self, question: str, user_id: str = None) -> Dict[str, Any]:
        """
        带安全检查的查询

        执行顺序:
        1. Rate limiting check
        2. Input validation
        3. Guardrails input check (optional)
        4. 调用核心pipeline
        5. Guardrails output check (optional)
        6. Audit logging
        """
        # 1. Rate limiting
        if self.rate_limiter:
            if not self.rate_limiter.check_rate_limit(user_id):
                raise RateLimitExceeded()

        # 2. Input validation
        if self.input_validator:
            validation_result = self.input_validator.validate(question)
            if not validation_result['valid']:
                raise ValueError(f"Input validation failed: {validation_result['errors']}")

        # 3. Guardrails input check (可选)
        if self.guardrails:
            guard_result = self.guardrails.validate_input(question)
            if not guard_result['valid']:
                raise ValueError(f"Guardrails check failed: {guard_result['errors']}")

        # 4. 核心查询
        result = self.pipeline.query(question)

        # 5. Guardrails output check (可选)
        if self.guardrails:
            result['answer'] = self.guardrails.validate_output(result['answer'])

        # 6. Audit logging
        if self.audit_logger:
            self.audit_logger.log_query(user_id, question, result)

        return result
```

---

## 🎯 使用示例

### 基础版本（app_gemini.py）

```python
# 简单直接
config = ModelConfig.gemini_2_5_flash()

pipeline = RAGPipeline(
    google_api_key=api_key,
    config=config
)

result = pipeline.query("What is machine learning?")
```

### 安全版本（app_gemini_secure.py）

```python
# 1. 创建核心pipeline
model_config = ModelConfig.gemini_2_5_flash()
pipeline = RAGPipeline(
    google_api_key=google_api_key,
    config=model_config
)

# 2. 配置安全选项
security_config = SecurityConfig(
    enable_rate_limiting=True,
    enable_input_validation=True,
    enable_audit_logging=True,
    enable_guardrails=False  # 默认禁用，除非提供OpenAI key
)

# 3. 包装为安全版本
secure_pipeline = SecureRAGPipeline(
    pipeline=pipeline,
    security_config=security_config,
    openai_api_key=None  # 可选
)

result = secure_pipeline.query("What is machine learning?", user_id="user123")
```

### 完全安全版本（带Guardrails）

```python
# 只有当用户同时有Google和OpenAI key时
security_config = SecurityConfig(
    enable_guardrails=True  # 启用Guardrails
)

secure_pipeline = SecureRAGPipeline(
    pipeline=pipeline,
    security_config=security_config,
    openai_api_key=openai_api_key  # 提供OpenAI key
)
```

---

## 📊 对比：重构前 vs 重构后

| 方面 | 重构前 | 重构后 |
|------|--------|--------|
| **代码重复** | 两个pipeline实现 | 一个核心+装饰器 |
| **模型配置** | 分散在多处 | 统一ModelConfig |
| **依赖关系** | 强耦合 | 依赖注入 |
| **扩展性** | 难以添加新安全组件 | 易于组合新组件 |
| **测试性** | 难以单独测试 | 组件独立测试 |
| **Guardrails问题** | 强制依赖OpenAI | 可选，默认禁用 |

---

## 🔧 立即修复建议

### 短期修复（快速解决当前问题）

1. **禁用Guardrails AI**（最快）
   ```python
   # app_gemini_secure.py
   enable_guardrails=False  # 改为False
   ```

2. **更新模型名称**
   ```python
   model_name="gemini-2.5-flash"  # 统一使用最新模型
   ```

### 长期重构（推荐）

1. **重构核心pipeline**
   - 提取公共基类
   - 统一配置接口
   - 分离安全逻辑

2. **实现装饰器模式**
   - SecureRAGPipeline包装RAGPipeline
   - 每个安全组件可独立启用

3. **文档和示例**
   - 清晰的使用文档
   - 配置示例
   - 迁移指南

---

## 💡 API设计的不足与改进

### 当前不足

1. **❌ 配置分散**
   - 模型配置在__init__参数中
   - 安全配置混杂其中
   - 难以维护和扩展

2. **❌ 强耦合**
   - 安全功能与核心RAG耦合
   - 无法独立测试
   - 难以替换组件

3. **❌ 隐式依赖**
   - Guardrails需要OpenAI key但不明显
   - 用户困惑为什么需要两个API key

4. **❌ 缺乏灵活性**
   - 无法动态启用/禁用安全功能
   - 无法自定义安全策略
   - 无法添加新的安全组件

### 改进方向

1. **✅ 配置对象**
   - ModelConfig: 模型相关配置
   - SecurityConfig: 安全相关配置
   - 类型安全，易于验证

2. **✅ 依赖注入**
   - SecureRAGPipeline依赖RAGPipeline
   - 每个安全组件独立注入
   - 易于测试和替换

3. **✅ 显式依赖**
   - openai_api_key作为可选参数
   - 清晰的文档说明
   - 优雅的降级处理

4. **✅ 插件化架构**
   - 安全组件为插件
   - 支持自定义插件
   - 统一的插件接口

---

## 📚 参考资料

### 设计模式
- **装饰器模式**: 动态添加功能
- **依赖注入**: 解耦组件
- **策略模式**: 可替换的算法

### 最佳实践
- SOLID原则
- 接口隔离
- 配置优于编码

---

**创建时间**: 2025-10-26
**作者**: Claude Code
**版本**: 1.0
**状态**: 建议文档
