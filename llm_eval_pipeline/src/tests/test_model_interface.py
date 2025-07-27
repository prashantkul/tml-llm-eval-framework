"""Tests for model interface system."""

import pytest
from unittest.mock import AsyncMock, Mock, patch

from llm_eval.core.model_interface import ModelInterface, create_model


class MockModel(ModelInterface):
    """Mock model for testing."""
    
    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name, **kwargs)
        self.responses = kwargs.get('responses', ["Test response"])
        self.call_count = 0
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Mock generation."""
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return response
    
    def get_model_info(self) -> dict:
        """Mock model info."""
        return {
            "provider": "mock",
            "model_name": self.model_name,
            "type": "test"
        }


class TestModelInterface:
    """Test cases for ModelInterface."""
    
    @pytest.mark.asyncio
    async def test_basic_generation(self):
        """Test basic text generation."""
        model = MockModel("test-model")
        
        response = await model.generate("Test prompt")
        assert response == "Test response"
        assert model.call_count == 1
    
    @pytest.mark.asyncio
    async def test_batch_generation(self):
        """Test batch text generation."""
        model = MockModel("test-model", responses=["Response 1", "Response 2", "Response 3"])
        
        prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
        responses = await model.batch_generate(prompts)
        
        assert len(responses) == 3
        assert responses[0] == "Response 1"
        assert responses[1] == "Response 2"
        assert responses[2] == "Response 3"
    
    def test_model_info(self):
        """Test model info retrieval."""
        model = MockModel("test-model")
        info = model.get_model_info()
        
        assert info["provider"] == "mock"
        assert info["model_name"] == "test-model"
        assert info["type"] == "test"
    
    def test_rate_limiting(self):
        """Test rate limiting initialization."""
        model = MockModel("test-model", rate_limit_delay=0.5)
        
        assert model.rate_limit_delay == 0.5
        assert model.request_count == 0
        assert model.last_request_time == 0


class TestModelFactory:
    """Test cases for model factory function."""
    
    def test_groq_model_creation(self):
        """Test Groq model creation."""
        with patch('llm_eval.core.model_interface.groq') as mock_groq:
            mock_groq.Groq.return_value = Mock()
            
            model = create_model('groq/test-model', groq_api_key='test-key')
            
            assert model.model_name == 'test-model'
            assert hasattr(model, 'client')
    
    def test_openai_model_creation(self):
        """Test OpenAI model creation."""
        with patch('llm_eval.core.model_interface.openai') as mock_openai:
            mock_openai.AsyncOpenAI.return_value = Mock()
            
            model = create_model('openai/gpt-4', openai_api_key='test-key')
            
            assert model.model_name == 'gpt-4'
            assert hasattr(model, 'client')
    
    def test_anthropic_model_creation(self):
        """Test Anthropic model creation."""
        with patch('llm_eval.core.model_interface.anthropic') as mock_anthropic:
            mock_anthropic.AsyncAnthropic.return_value = Mock()
            
            model = create_model('anthropic/claude-3-sonnet-20240229', anthropic_api_key='test-key')
            
            assert model.model_name == 'claude-3-sonnet-20240229'
            assert hasattr(model, 'client')
    
    def test_local_model_creation(self):
        """Test local model creation."""
        with patch('llm_eval.core.model_interface.AutoTokenizer') as mock_tokenizer, \
             patch('llm_eval.core.model_interface.AutoModelForCausalLM') as mock_model:
            
            mock_tokenizer.from_pretrained.return_value = Mock()
            mock_model.from_pretrained.return_value = Mock()
            
            model = create_model('local/test-model')
            
            assert model.model_name == 'test-model'
            assert hasattr(model, 'tokenizer')
            assert hasattr(model, 'model')
    
    def test_invalid_model_identifier(self):
        """Test invalid model identifier handling."""
        # This should default to local model
        with patch('llm_eval.core.model_interface.AutoTokenizer') as mock_tokenizer, \
             patch('llm_eval.core.model_interface.AutoModelForCausalLM') as mock_model:
            
            mock_tokenizer.from_pretrained.return_value = Mock()
            mock_model.from_pretrained.return_value = Mock()
            
            model = create_model('invalid-format-model')
            
            assert model.model_name == 'invalid-format-model'
    
    def test_missing_api_key(self):
        """Test missing API key error."""
        with pytest.raises(ValueError, match="API key required"):
            create_model('groq/test-model')  # No API key provided