#!/usr/bin/env python3
"""
OpenRouter API client for the Memora eval pipeline.

All four LLMs (Table 3) and all three multi-judge models route through
OpenRouter — the same provider used for data generation
(conversation_generation/prompt_manager.py).
"""

import os
import time
from abc import ABC, abstractmethod


class BaseAPIClient(ABC):
    """Base class for API clients"""
    
    def __init__(self, model: str):
        self.model = model
    
    @abstractmethod
    def generate_response(
        self, 
        system_prompt: str, 
        user_prompt: str, 
        temperature: float = 0.7,
        max_retries: int = 3
    ) -> str:
        """Generate a response from the model"""
        pass


class OpenRouterClient(BaseAPIClient):
    """OpenRouter API client using OpenAI library"""
    
    def __init__(self, model: str):
        super().__init__(model)
        
        try:
            import openai
            self.openai = openai
        except ImportError:
            raise ImportError(
                "openai package not installed. Install with: pip install openai"
            )
        
        # Check both OPENROUTER_API_KEY and OPEN_ROUTER_API_KEY (with underscore)
        self.api_key = os.getenv('OPENROUTER_API_KEY') or os.getenv('OPEN_ROUTER_API_KEY')
        if not self.api_key:
            raise ValueError(
                "OPENROUTER_API_KEY (or OPEN_ROUTER_API_KEY) environment variable not set. "
                "Please set one of these in your .env file."
            )
        
        # Use OpenAI client with OpenRouter base URL (like prompt_manager.py)
        self.client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key
        )
    
    def generate_response(
        self, 
        system_prompt: str, 
        user_prompt: str, 
        temperature: float = 0.7,
        max_retries: int = 3,
        reasoning: dict = None,
        show_reasoning: bool = True
    ) -> str:
        """
        Generate a response using OpenRouter API via OpenAI client
        
        Args:
            system_prompt: System prompt for the model
            user_prompt: User prompt for the model
            temperature: Temperature for response generation
            max_retries: Number of retries on failure
            reasoning: Optional reasoning config for models that support it
                       e.g., {"effort": "high"} or {"max_tokens": 2000}
                       See: https://openrouter.ai/docs/guides/best-practices/reasoning-tokens
            show_reasoning: Whether to print reasoning tokens (default: True)
        """
        
        for attempt in range(max_retries):
            try:
                # Build extra_body for reasoning if specified
                extra_body = {}
                if reasoning:
                    extra_body["reasoning"] = reasoning
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=temperature,
                    extra_body=extra_body if extra_body else None
                )
                
                message = response.choices[0].message
                choice = response.choices[0]
                
                # Get finish_reason for debugging
                finish_reason = getattr(choice, 'finish_reason', None) or getattr(message, 'finish_reason', None) or 'unknown'
                
                # Check for reasoning tokens and display them
                reasoning_text = getattr(message, 'reasoning', None)
                reasoning_details = getattr(message, 'reasoning_details', None)
                
                if show_reasoning and reasoning:
                    if reasoning_text:
                        print(f"\n💭 REASONING TOKENS ({len(reasoning_text)} chars):")
                        # Show first 500 chars of reasoning
                        preview = reasoning_text[:500] + "..." if len(reasoning_text) > 500 else reasoning_text
                        print(f"   {preview}")
                    elif reasoning_details:
                        print(f"\n💭 REASONING DETAILS: {len(reasoning_details)} blocks")
                        for i, detail in enumerate(reasoning_details[:2]):  # Show first 2 blocks
                            detail_type = detail.get('type', 'unknown') if isinstance(detail, dict) else str(type(detail))
                            print(f"   Block {i+1}: {detail_type}")
                    elif reasoning:
                        print(f"\n💭 Reasoning mode enabled but no reasoning tokens returned (model may not support it)")
                
                # Check if content is None or empty
                content = message.content
                
                # Debug: Print response details if content is empty
                if content is None or (isinstance(content, str) and len(content.strip()) == 0):
                    # Get more details about the response
                    response_id = getattr(response, 'id', 'unknown')
                    model_used = getattr(response, 'model', 'unknown')
                    usage_info = getattr(response, 'usage', None)
                    
                    error_details = {
                        'finish_reason': finish_reason,
                        'response_id': response_id,
                        'model': model_used,
                        'content_type': type(content).__name__,
                        'content_value': repr(content),
                    }
                    
                    if usage_info:
                        error_details['usage'] = {
                            'prompt_tokens': getattr(usage_info, 'prompt_tokens', None),
                            'completion_tokens': getattr(usage_info, 'completion_tokens', None),
                            'total_tokens': getattr(usage_info, 'total_tokens', None),
                        }
                    
                    # Check for common finish_reason values that indicate issues
                    finish_reason_lower = str(finish_reason).lower()
                    if 'length' in finish_reason_lower:
                        error_msg = (
                            f"API response was truncated due to length limit (finish_reason: {finish_reason}). "
                            f"This may indicate the response was cut off. Details: {error_details}"
                        )
                    elif 'content_filter' in finish_reason_lower or 'filter' in finish_reason_lower:
                        error_msg = (
                            f"API response was filtered (finish_reason: {finish_reason}). "
                            f"Content may have been blocked. Details: {error_details}"
                        )
                    elif 'stop' in finish_reason_lower:
                        error_msg = (
                            f"API returned empty response with finish_reason: {finish_reason}. "
                            f"This is unusual. Details: {error_details}"
                        )
                    else:
                        error_msg = (
                            f"API returned empty response. finish_reason: {finish_reason}. "
                            f"This may indicate a context size issue or API error. Details: {error_details}"
                        )
                    
                    print(f"⚠ DEBUG: Full response object keys: {dir(response)}")
                    print(f"⚠ DEBUG: Choice object keys: {dir(choice)}")
                    print(f"⚠ DEBUG: Message object keys: {dir(message)}")
                    
                    raise ValueError(error_msg)
                
                # Log finish_reason if it's not 'stop' (normal completion)
                if finish_reason and finish_reason != 'stop':
                    print(f"⚠ Note: finish_reason = {finish_reason}")
                
                return content
                
            except Exception as e:
                print(f"Request error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise
        
        raise Exception("Failed to generate response after all retries")


def get_api_client(model: str) -> BaseAPIClient:
    """Return an OpenRouterClient for the given model id.

    All four reported LLMs and the three multi-judge models route through
    OpenRouter — same provider used for data generation in
    conversation_generation/prompt_manager.py.
    """
    return OpenRouterClient(model=model)
