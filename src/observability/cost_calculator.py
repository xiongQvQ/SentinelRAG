"""
Cost calculation for LLM and embedding operations.

Tracks and estimates costs based on:
- Token usage (input/output)
- Model pricing
- Embedding operations

Based on ReadyTensor Week 11 cost monitoring best practices.
"""

import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


# Gemini API Pricing (as of 2024)
# Source: https://ai.google.dev/pricing
GEMINI_PRICING = {
    'gemini-1.5-flash': {
        'input_tokens_per_million': 0.35,   # $0.35 per 1M input tokens
        'output_tokens_per_million': 1.05,  # $1.05 per 1M output tokens
    },
    'gemini-1.5-pro': {
        'input_tokens_per_million': 3.50,   # $3.50 per 1M input tokens
        'output_tokens_per_million': 10.50, # $10.50 per 1M output tokens
    },
}


@dataclass
class CostBreakdown:
    """
    Cost breakdown for a single query.

    All costs in USD.
    """
    # Token counts
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    # Costs
    input_cost: float = 0.0
    output_cost: float = 0.0
    total_cost: float = 0.0

    # Metadata
    model_name: str = ""
    query_id: Optional[str] = None
    timestamp: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            'input_tokens': self.input_tokens,
            'output_tokens': self.output_tokens,
            'total_tokens': self.total_tokens,
            'input_cost_usd': self.input_cost,
            'output_cost_usd': self.output_cost,
            'total_cost_usd': self.total_cost,
            'model_name': self.model_name,
            'query_id': self.query_id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            # Derived metrics
            'cost_per_token': (self.total_cost / self.total_tokens) if self.total_tokens > 0 else 0,
            'input_token_percentage': (self.input_tokens / self.total_tokens * 100) if self.total_tokens > 0 else 0,
            'output_token_percentage': (self.output_tokens / self.total_tokens * 100) if self.total_tokens > 0 else 0,
        }


class CostCalculator:
    """
    Calculates and tracks costs for LLM operations.

    Provides:
    - Per-query cost calculation
    - Token usage tracking
    - Cumulative cost monitoring
    - Cost projections

    Example:
        calculator = CostCalculator(model_name="gemini-1.5-flash")

        # Calculate cost for a query
        cost = calculator.calculate_query_cost(
            input_tokens=500,
            output_tokens=150,
            query_id="q123"
        )

        print(f"Query cost: ${cost.total_cost:.4f}")
        print(f"Total spent today: ${calculator.get_total_cost():.4f}")
    """

    def __init__(
        self,
        model_name: str = "gemini-1.5-flash",
        enabled: bool = True
    ):
        """
        Initialize cost calculator.

        Args:
            model_name: LLM model name for pricing lookup
            enabled: Whether to enable cost tracking
        """
        self.enabled = enabled
        self.model_name = model_name

        # Cumulative tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost_usd = 0.0
        self.query_count = 0

        # Get pricing for model
        self.pricing = GEMINI_PRICING.get(model_name, {})
        if not self.pricing:
            logger.warning(f"Unknown model '{model_name}'. Cost calculation may be inaccurate.")
            # Use default Gemini Flash pricing
            self.pricing = GEMINI_PRICING['gemini-1.5-flash']

        if enabled:
            logger.info(f"Cost calculator initialized (model: {model_name})")

    def calculate_query_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        query_id: Optional[str] = None
    ) -> CostBreakdown:
        """
        Calculate cost for a single query.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            query_id: Query identifier for tracking

        Returns:
            CostBreakdown with detailed cost information
        """
        if not self.enabled:
            return CostBreakdown(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                model_name=self.model_name,
                query_id=query_id
            )

        # Calculate costs
        input_cost = (input_tokens / 1_000_000) * self.pricing['input_tokens_per_million']
        output_cost = (output_tokens / 1_000_000) * self.pricing['output_tokens_per_million']
        total_cost = input_cost + output_cost

        # Update cumulative totals
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost_usd += total_cost
        self.query_count += 1

        # Create breakdown
        breakdown = CostBreakdown(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost,
            model_name=self.model_name,
            query_id=query_id,
            timestamp=datetime.utcnow()
        )

        logger.debug(
            f"Query cost: ${total_cost:.4f} "
            f"({input_tokens} in + {output_tokens} out tokens)"
        )

        return breakdown

    def get_total_cost(self) -> float:
        """
        Get cumulative total cost.

        Returns:
            Total cost in USD
        """
        return self.total_cost_usd

    def get_total_tokens(self) -> int:
        """
        Get cumulative total tokens.

        Returns:
            Total token count
        """
        return self.total_input_tokens + self.total_output_tokens

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive cost statistics.

        Returns:
            Dict with cumulative statistics
        """
        total_tokens = self.get_total_tokens()

        return {
            'total_cost_usd': self.total_cost_usd,
            'total_input_tokens': self.total_input_tokens,
            'total_output_tokens': self.total_output_tokens,
            'total_tokens': total_tokens,
            'query_count': self.query_count,
            'avg_cost_per_query': self.total_cost_usd / self.query_count if self.query_count > 0 else 0,
            'avg_tokens_per_query': total_tokens / self.query_count if self.query_count > 0 else 0,
            'cost_per_1k_tokens': (self.total_cost_usd / total_tokens * 1000) if total_tokens > 0 else 0,
            'model_name': self.model_name,
        }

    def project_cost(
        self,
        queries_per_day: int,
        avg_input_tokens: int = 500,
        avg_output_tokens: int = 150
    ) -> Dict[str, float]:
        """
        Project costs for given usage.

        Args:
            queries_per_day: Estimated daily query volume
            avg_input_tokens: Average input tokens per query
            avg_output_tokens: Average output tokens per query

        Returns:
            Dict with cost projections (daily, monthly, yearly)
        """
        # Calculate cost per query
        breakdown = self.calculate_query_cost(
            input_tokens=avg_input_tokens,
            output_tokens=avg_output_tokens
        )
        cost_per_query = breakdown.total_cost

        # Undo the cumulative update (this was just for projection)
        self.total_input_tokens -= avg_input_tokens
        self.total_output_tokens -= avg_output_tokens
        self.total_cost_usd -= cost_per_query
        self.query_count -= 1

        # Calculate projections
        daily_cost = cost_per_query * queries_per_day
        monthly_cost = daily_cost * 30
        yearly_cost = daily_cost * 365

        return {
            'cost_per_query': cost_per_query,
            'daily_cost': daily_cost,
            'monthly_cost': monthly_cost,
            'yearly_cost': yearly_cost,
            'queries_per_day': queries_per_day,
            'avg_input_tokens': avg_input_tokens,
            'avg_output_tokens': avg_output_tokens,
        }

    def reset_statistics(self) -> None:
        """Reset cumulative statistics."""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost_usd = 0.0
        self.query_count = 0
        logger.info("Cost statistics reset")
