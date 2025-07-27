"""Standardized interface for different LLM providers."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
import asyncio
import logging
from datetime import datetime
import time

try:
    import openai
except ImportError:
    openai = None

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    import groq
except ImportError:
    groq = None

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import torch
except ImportError:
    AutoTokenizer = AutoModelForCausalLM = pipeline = torch = None


logger = logging.getLogger(__name__)


class ModelInterface(ABC):
    """Abstract interface for LLM models."""
    
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.request_count = 0
        self.last_request_time = 0
        self.rate_limit_delay = kwargs.get('rate_limit_delay', 0.1)
    
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate response from model."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get model metadata."""
        pass
    
    async def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate responses for multiple prompts with rate limiting."""
        semaphore = asyncio.Semaphore(kwargs.get('max_concurrent', 5))
        
        async def generate_with_semaphore(prompt: str) -> str:
            async with semaphore:
                return await self.generate(prompt, **kwargs)
        
        tasks = [generate_with_semaphore(prompt) for prompt in prompts]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _rate_limit(self):
        """Apply rate limiting between requests."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - time_since_last)
        self.last_request_time = time.time()
        self.request_count += 1


class OpenAIModel(ModelInterface):
    """OpenAI model interface."""
    
    def __init__(self, model_name: str, api_key: str, **kwargs):
        super().__init__(model_name, **kwargs)
        if openai is None:
            raise ImportError("openai package not installed. Install with: pip install openai")
        
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.max_tokens = kwargs.get('max_tokens', 1000)
        self.temperature = kwargs.get('temperature', 0.7)
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate response using OpenAI API."""
        await self._rate_limit()
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=kwargs.get('max_tokens', self.max_tokens),
                temperature=kwargs.get('temperature', self.temperature),
                timeout=kwargs.get('timeout', 30)
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get OpenAI model metadata."""
        return {
            "provider": "openai",
            "model_name": self.model_name,
            "type": "proprietary",
            "api_based": True,
            "supports_streaming": True,
            "context_window": self._get_context_window()
        }
    
    def _get_context_window(self) -> int:
        """Get context window size for OpenAI models."""
        context_windows = {
            "gpt-4": 8192,
            "gpt-4-32k": 32768,
            "gpt-4-turbo": 128000,
            "gpt-3.5-turbo": 4096,
            "gpt-3.5-turbo-16k": 16384
        }
        return context_windows.get(self.model_name, 8192)


class AnthropicModel(ModelInterface):
    """Anthropic model interface."""
    
    def __init__(self, model_name: str, api_key: str, **kwargs):
        super().__init__(model_name, **kwargs)
        if anthropic is None:
            raise ImportError("anthropic package not installed. Install with: pip install anthropic")
        
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self.max_tokens = kwargs.get('max_tokens', 1000)
        self.temperature = kwargs.get('temperature', 0.7)
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate response using Anthropic API."""
        await self._rate_limit()
        
        try:
            response = await self.client.messages.create(
                model=self.model_name,
                max_tokens=kwargs.get('max_tokens', self.max_tokens),
                temperature=kwargs.get('temperature', self.temperature),
                messages=[{"role": "user", "content": prompt}],
                timeout=kwargs.get('timeout', 30)
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Anthropic generation failed: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get Anthropic model metadata."""
        return {
            "provider": "anthropic",
            "model_name": self.model_name,
            "type": "proprietary",
            "api_based": True,
            "supports_streaming": True,
            "context_window": self._get_context_window()
        }
    
    def _get_context_window(self) -> int:
        """Get context window size for Anthropic models."""
        context_windows = {
            "claude-3-opus-20240229": 200000,
            "claude-3-sonnet-20240229": 200000,
            "claude-3-haiku-20240307": 200000,
            "claude-2.1": 200000,
            "claude-2.0": 100000
        }
        return context_windows.get(self.model_name, 100000)


class GroqModel(ModelInterface):
    """Groq model interface for open source models."""
    
    def __init__(self, model_name: str, api_key: str, **kwargs):
        super().__init__(model_name, **kwargs)
        if groq is None:
            raise ImportError("groq package not installed. Install with: pip install groq")
        
        self.client = groq.Groq(api_key=api_key)
        self.max_tokens = kwargs.get('max_tokens', 1000)
        self.temperature = kwargs.get('temperature', 0.7)
        
        # Groq-specific rate limiting (more aggressive)
        self.rate_limit_delay = kwargs.get('rate_limit_delay', 0.5)
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate response using Groq API for open source models."""
        await self._rate_limit()
        
        try:
            # Groq doesn't have async client, so we run in executor
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=kwargs.get('max_tokens', self.max_tokens),
                    temperature=kwargs.get('temperature', self.temperature),
                    timeout=kwargs.get('timeout', 30)
                )
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Groq generation failed: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get Groq model metadata."""
        return {
            "provider": "groq",
            "model_name": self.model_name,
            "type": "open_source",
            "api_based": True,
            "supports_streaming": False,
            "context_window": self._get_context_window(),
            "inference_speed": "high"
        }
    
    def _get_context_window(self) -> int:
        """Get context window size for Groq-hosted models."""
        context_windows = {
            "llama-3.2-90b-text-preview": 131072,
            "llama-3.2-11b-text-preview": 131072,
            "llama-3.1-70b-versatile": 131072,
            "llama-3.1-8b-instant": 131072,
            "llama-4-scout-17b-16e-instruct": 32768,
            "mixtral-8x7b-32768": 32768,
            "gemma-7b-it": 8192
        }
        return context_windows.get(self.model_name, 8192)


class LocalModel(ModelInterface):
    """Local transformers model interface."""
    
    def __init__(self, model_name: str, device: str = "auto", **kwargs):
        super().__init__(model_name, **kwargs)
        if AutoTokenizer is None or AutoModelForCausalLM is None:
            raise ImportError("transformers package not installed. Install with: pip install transformers torch")
        
        self.device = device
        self.max_tokens = kwargs.get('max_tokens', 1000)
        self.temperature = kwargs.get('temperature', 0.7)
        
        # Load model and tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                device_map=device,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
        except Exception as e:
            logger.error(f"Failed to load local model {model_name}: {e}")
            raise
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate response using local transformers model."""
        try:
            # Run generation in executor to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, self._generate_sync, prompt, kwargs)
            return response
        except Exception as e:
            logger.error(f"Local generation failed: {e}")
            raise
    
    def _generate_sync(self, prompt: str, kwargs: Dict[str, Any]) -> str:
        """Synchronous generation for executor."""
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        
        if torch.cuda.is_available() and self.device != "cpu":
            inputs = inputs.cuda()
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=kwargs.get('max_tokens', self.max_tokens),
                temperature=kwargs.get('temperature', self.temperature),
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the input prompt from response
        response = response[len(prompt):].strip()
        return response
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get local model metadata."""
        return {
            "provider": "local",
            "model_name": self.model_name,
            "type": "open_source",
            "api_based": False,
            "supports_streaming": False,
            "device": self.device,
            "inference_speed": "variable"
        }


def create_model(model_identifier: str, **kwargs) -> ModelInterface:
    """Factory function to create model instances from identifiers."""
    if model_identifier.startswith('openai/'):
        model_name = model_identifier.replace('openai/', '')
        api_key = kwargs.get('openai_api_key') or kwargs.get('api_key')
        if not api_key:
            raise ValueError("OpenAI API key required")
        return OpenAIModel(model_name, api_key, **kwargs)
    
    elif model_identifier.startswith('anthropic/'):
        model_name = model_identifier.replace('anthropic/', '')
        api_key = kwargs.get('anthropic_api_key') or kwargs.get('api_key')
        if not api_key:
            raise ValueError("Anthropic API key required")
        return AnthropicModel(model_name, api_key, **kwargs)
    
    elif model_identifier.startswith('groq/'):
        model_name = model_identifier.replace('groq/', '')
        api_key = kwargs.get('groq_api_key') or kwargs.get('api_key')
        if not api_key:
            raise ValueError("Groq API key required")
        return GroqModel(model_name, api_key, **kwargs)
    
    elif model_identifier.startswith('local/'):
        model_name = model_identifier.replace('local/', '')
        device = kwargs.get('device', 'auto')
        return LocalModel(model_name, device, **kwargs)
    
    else:
        # Default to local if no prefix
        return LocalModel(model_identifier, kwargs.get('device', 'auto'), **kwargs)