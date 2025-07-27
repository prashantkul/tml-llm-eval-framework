"""AutoEvoEval integration for robustness evaluation."""

import asyncio
import logging
import random
import copy
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

from ...core.model_interface import ModelInterface

logger = logging.getLogger(__name__)


class AutoEvoEvalEvaluator:
    """Evaluator for AutoEvoEval robustness assessment using 22 atomic evolution operations."""
    
    def __init__(self, config):
        self.config = config
        self.sample_size = getattr(config, 'sample_size', 100)
        self.random_seed = getattr(config, 'random_seed', 42)
        random.seed(self.random_seed)
        
        # 22 atomic evolution operations from AutoEvoEval
        self.evolution_operations = {
            # Character-level operations
            "char_insertion": self._char_insertion,
            "char_deletion": self._char_deletion,
            "char_substitution": self._char_substitution,
            "char_swapping": self._char_swapping,
            
            # Word-level operations  
            "word_insertion": self._word_insertion,
            "word_deletion": self._word_deletion,
            "word_substitution": self._word_substitution,
            "word_swapping": self._word_swapping,
            
            # Sentence-level operations
            "sentence_insertion": self._sentence_insertion,
            "sentence_deletion": self._sentence_deletion,
            "sentence_substitution": self._sentence_substitution,
            "sentence_swapping": self._sentence_swapping,
            
            # Semantic operations
            "synonym_substitution": self._synonym_substitution,
            "antonym_substitution": self._antonym_substitution,
            "paraphrase": self._paraphrase,
            "negation_addition": self._negation_addition,
            
            # Style operations
            "formality_change": self._formality_change,
            "tense_change": self._tense_change,
            "voice_change": self._voice_change,
            
            # Structure operations
            "format_change": self._format_change,
            "instruction_reorder": self._instruction_reorder,
            "context_addition": self._context_addition
        }
        
        # Load base prompts for evolution
        self.base_prompts = self._load_base_prompts()
        
        logger.info(f"AutoEvoEval evaluator initialized with {len(self.evolution_operations)} operations and {len(self.base_prompts)} base prompts")
    
    def _load_base_prompts(self) -> List[Dict[str, Any]]:
        """Load base prompts for evolution testing."""
        prompts = [
            {
                "prompt": "What is the capital of France?",
                "category": "factual_knowledge",
                "expected_answer": "Paris",
                "difficulty": "easy"
            },
            {
                "prompt": "Explain the process of photosynthesis in plants.",
                "category": "scientific_explanation", 
                "expected_answer": "chlorophyll converts sunlight to energy",
                "difficulty": "medium"
            },
            {
                "prompt": "Solve this math problem: If a train travels 60 mph for 2.5 hours, how far does it go?",
                "category": "mathematical_reasoning",
                "expected_answer": "150 miles",
                "difficulty": "medium"
            },
            {
                "prompt": "Write a short poem about autumn leaves.",
                "category": "creative_writing",
                "expected_answer": "creative poem about autumn",
                "difficulty": "easy"
            },
            {
                "prompt": "Compare and contrast democracy and autocracy.",
                "category": "analytical_thinking",
                "expected_answer": "democracy allows participation, autocracy concentrates power",
                "difficulty": "hard"
            },
            {
                "prompt": "What are the main causes of climate change?",
                "category": "factual_knowledge",
                "expected_answer": "greenhouse gases, human activities",
                "difficulty": "medium"
            },
            {
                "prompt": "Translate this to Spanish: 'The weather is beautiful today.'",
                "category": "language_translation",
                "expected_answer": "El tiempo está hermoso hoy",
                "difficulty": "easy"
            },
            {
                "prompt": "Analyze the themes in Shakespeare's Romeo and Juliet.",
                "category": "literary_analysis",
                "expected_answer": "love, fate, family conflict",
                "difficulty": "hard"
            },
            {
                "prompt": "Describe the steps to bake a chocolate cake.",
                "category": "procedural_knowledge", 
                "expected_answer": "mix ingredients, bake in oven",
                "difficulty": "easy"
            },
            {
                "prompt": "What is the significance of the discovery of DNA structure?",
                "category": "scientific_explanation",
                "expected_answer": "understanding genetics and heredity",
                "difficulty": "hard"
            }
        ]
        
        return prompts
    
    async def evaluate(self, model: ModelInterface, datasets: Optional[List[str]] = None) -> Dict[str, float]:
        """Run AutoEvoEval robustness evaluation."""
        logger.info(f"Starting AutoEvoEval evaluation with {len(self.evolution_operations)} operations")
        
        # Generate evolved prompts using all operations
        evolved_prompts = await self._generate_evolved_prompts()
        
        # Get baseline responses
        baseline_responses = await self._get_baseline_responses(model)
        
        # Get evolved responses
        evolved_responses = await self._get_evolved_responses(model, evolved_prompts)
        
        # Calculate robustness metrics
        metrics = await self._calculate_robustness_metrics(baseline_responses, evolved_responses)
        
        logger.info(f"AutoEvoEval completed. Evolution robustness PDR: {metrics.get('evolution_robustness_pdr', 0):.3f}")
        
        return metrics
    
    async def _generate_evolved_prompts(self) -> List[Dict[str, Any]]:
        """Generate evolved prompts using all 22 operations."""
        logger.info("Generating evolved prompts...")
        
        evolved_prompts = []
        operations_per_prompt = max(1, len(self.evolution_operations) // len(self.base_prompts))
        
        for base_prompt in self.base_prompts:
            for operation_name, operation_func in self.evolution_operations.items():
                try:
                    evolved_prompt = operation_func(base_prompt['prompt'])
                    evolved_prompts.append({
                        'original_prompt': base_prompt['prompt'],
                        'evolved_prompt': evolved_prompt,
                        'operation': operation_name,
                        'category': base_prompt['category'],
                        'expected_answer': base_prompt['expected_answer'],
                        'difficulty': base_prompt['difficulty']
                    })
                except Exception as e:
                    logger.warning(f"Failed to apply operation {operation_name}: {e}")
        
        # Limit to sample size
        if len(evolved_prompts) > self.sample_size:
            evolved_prompts = random.sample(evolved_prompts, self.sample_size)
        
        logger.info(f"Generated {len(evolved_prompts)} evolved prompts")
        return evolved_prompts
    
    # Character-level operations
    def _char_insertion(self, prompt: str) -> str:
        """Insert random characters."""
        if len(prompt) < 2:
            return prompt
        pos = random.randint(0, len(prompt))
        char = random.choice('abcdefghijklmnopqrstuvwxyz ')
        return prompt[:pos] + char + prompt[pos:]
    
    def _char_deletion(self, prompt: str) -> str:
        """Delete random characters."""
        if len(prompt) < 2:
            return prompt
        pos = random.randint(0, len(prompt) - 1)
        return prompt[:pos] + prompt[pos + 1:]
    
    def _char_substitution(self, prompt: str) -> str:
        """Substitute random characters."""
        if len(prompt) < 1:
            return prompt
        pos = random.randint(0, len(prompt) - 1)
        char = random.choice('abcdefghijklmnopqrstuvwxyz ')
        return prompt[:pos] + char + prompt[pos + 1:]
    
    def _char_swapping(self, prompt: str) -> str:
        """Swap adjacent characters."""
        if len(prompt) < 2:
            return prompt
        pos = random.randint(0, len(prompt) - 2)
        chars = list(prompt)
        chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
        return ''.join(chars)
    
    # Word-level operations
    def _word_insertion(self, prompt: str) -> str:
        """Insert random words."""
        words = prompt.split()
        if not words:
            return prompt
        pos = random.randint(0, len(words))
        new_word = random.choice(['really', 'actually', 'definitely', 'perhaps', 'maybe'])
        words.insert(pos, new_word)
        return ' '.join(words)
    
    def _word_deletion(self, prompt: str) -> str:
        """Delete random words."""
        words = prompt.split()
        if len(words) < 2:
            return prompt
        pos = random.randint(0, len(words) - 1)
        words.pop(pos)
        return ' '.join(words)
    
    def _word_substitution(self, prompt: str) -> str:
        """Substitute words with synonyms."""
        substitutions = {
            'explain': 'describe', 'what': 'which', 'how': 'in what way',
            'analyze': 'examine', 'compare': 'contrast', 'write': 'compose',
            'solve': 'calculate', 'find': 'determine', 'describe': 'explain'
        }
        words = prompt.split()
        for i, word in enumerate(words):
            clean_word = word.lower().strip('.,!?')
            if clean_word in substitutions:
                words[i] = word.replace(clean_word, substitutions[clean_word])
                break
        return ' '.join(words)
    
    def _word_swapping(self, prompt: str) -> str:
        """Swap adjacent words."""
        words = prompt.split()
        if len(words) < 2:
            return prompt
        pos = random.randint(0, len(words) - 2)
        words[pos], words[pos + 1] = words[pos + 1], words[pos]
        return ' '.join(words)
    
    # Sentence-level operations
    def _sentence_insertion(self, prompt: str) -> str:
        """Insert additional sentences."""
        additions = [
            "Please be thorough in your response.",
            "Take your time to think about this.",
            "Consider all aspects of the question.",
            "Provide a detailed explanation."
        ]
        return prompt + " " + random.choice(additions)
    
    def _sentence_deletion(self, prompt: str) -> str:
        """Delete sentences."""
        sentences = prompt.split('.')
        if len(sentences) > 1:
            sentences.pop(random.randint(0, len(sentences) - 2))
            return '.'.join(sentences)
        return prompt
    
    def _sentence_substitution(self, prompt: str) -> str:
        """Substitute sentence parts."""
        if '?' in prompt:
            return prompt.replace('?', ', please?')
        return prompt + " Can you help with this?"
    
    def _sentence_swapping(self, prompt: str) -> str:
        """Swap sentence order."""
        sentences = [s.strip() for s in prompt.split('.') if s.strip()]
        if len(sentences) > 1:
            random.shuffle(sentences)
            return '. '.join(sentences) + '.'
        return prompt
    
    # Semantic operations
    def _synonym_substitution(self, prompt: str) -> str:
        """Replace with synonyms."""
        return self._word_substitution(prompt)  # Reuse word substitution
    
    def _antonym_substitution(self, prompt: str) -> str:
        """Replace with antonyms where appropriate."""
        antonyms = {
            'explain': 'confuse', 'simple': 'complex', 'easy': 'difficult',
            'good': 'bad', 'positive': 'negative', 'increase': 'decrease'
        }
        words = prompt.split()
        for i, word in enumerate(words):
            clean_word = word.lower().strip('.,!?')
            if clean_word in antonyms:
                words[i] = word.replace(clean_word, antonyms[clean_word])
                break
        return ' '.join(words)
    
    def _paraphrase(self, prompt: str) -> str:
        """Paraphrase the prompt."""
        paraphrases = {
            "What is": "Can you tell me what",
            "Explain": "Please explain",
            "Describe": "Can you describe",
            "How": "In what way",
            "Why": "For what reason"
        }
        for original, paraphrase in paraphrases.items():
            if prompt.startswith(original):
                return prompt.replace(original, paraphrase, 1)
        return "Could you help me understand: " + prompt.lower()
    
    def _negation_addition(self, prompt: str) -> str:
        """Add negations."""
        if "not" not in prompt.lower():
            words = prompt.split()
            if len(words) > 2:
                # Insert "not" after auxiliary verbs or "is"
                for i, word in enumerate(words):
                    if word.lower() in ['is', 'are', 'was', 'were', 'can', 'could', 'would']:
                        words.insert(i + 1, 'not')
                        break
                return ' '.join(words)
        return prompt
    
    # Style operations
    def _formality_change(self, prompt: str) -> str:
        """Change formality level."""
        formal_replacements = {
            "what's": "what is",
            "can't": "cannot", 
            "won't": "will not",
            "don't": "do not"
        }
        informal_replacements = {
            "what is": "what's",
            "cannot": "can't",
            "will not": "won't", 
            "do not": "don't"
        }
        
        # Randomly choose formal or informal
        replacements = random.choice([formal_replacements, informal_replacements])
        result = prompt
        for original, replacement in replacements.items():
            result = result.replace(original, replacement)
        return result
    
    def _tense_change(self, prompt: str) -> str:
        """Change verb tenses."""
        tense_changes = {
            "is": "was", "are": "were", "was": "is", "were": "are",
            "will": "would", "would": "will", "can": "could", "could": "can"
        }
        words = prompt.split()
        for i, word in enumerate(words):
            if word.lower() in tense_changes:
                words[i] = tense_changes[word.lower()]
                break
        return ' '.join(words)
    
    def _voice_change(self, prompt: str) -> str:
        """Change active/passive voice."""
        if "by" in prompt.lower():
            return prompt  # Already passive
        return "Please have this explained: " + prompt
    
    # Structure operations
    def _format_change(self, prompt: str) -> str:
        """Change prompt format."""
        formats = [
            f"Question: {prompt}",
            f"Task: {prompt}",
            f"Request: {prompt}",
            f"Query: {prompt}"
        ]
        return random.choice(formats)
    
    def _instruction_reorder(self, prompt: str) -> str:
        """Reorder instruction components."""
        if "," in prompt:
            parts = [p.strip() for p in prompt.split(",")]
            random.shuffle(parts)
            return ", ".join(parts)
        return prompt
    
    def _context_addition(self, prompt: str) -> str:
        """Add context to prompt."""
        contexts = [
            "In the context of education, ",
            "For research purposes, ",
            "As part of a study, ",
            "In general terms, "
        ]
        return random.choice(contexts) + prompt.lower()
    
    async def _get_baseline_responses(self, model: ModelInterface) -> List[Dict[str, Any]]:
        """Get baseline responses to original prompts."""
        logger.info("Getting baseline responses...")
        
        semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        
        async def get_baseline_response(prompt_data: Dict[str, Any]) -> Dict[str, Any]:
            async with semaphore:
                try:
                    response = await model.generate(
                        prompt_data['prompt'],
                        max_tokens=300,
                        temperature=0.7
                    )
                    return {
                        'prompt': prompt_data['prompt'],
                        'response': response,
                        'category': prompt_data['category'],
                        'expected_answer': prompt_data['expected_answer'],
                        'difficulty': prompt_data['difficulty'],
                        'success': True
                    }
                except Exception as e:
                    logger.warning(f"Failed to get baseline response: {e}")
                    return {
                        'prompt': prompt_data['prompt'],
                        'response': "",
                        'category': prompt_data['category'],
                        'expected_answer': prompt_data['expected_answer'],
                        'difficulty': prompt_data['difficulty'],
                        'success': False
                    }
        
        tasks = [get_baseline_response(prompt_data) for prompt_data in self.base_prompts]
        responses = await asyncio.gather(*tasks)
        
        successful_responses = [r for r in responses if r['success']]
        logger.info(f"Got {len(successful_responses)} baseline responses")
        
        return successful_responses
    
    async def _get_evolved_responses(self, model: ModelInterface, evolved_prompts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get responses to evolved prompts."""
        logger.info("Getting evolved responses...")
        
        semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        
        async def get_evolved_response(evolved_data: Dict[str, Any]) -> Dict[str, Any]:
            async with semaphore:
                try:
                    response = await model.generate(
                        evolved_data['evolved_prompt'],
                        max_tokens=300,
                        temperature=0.7
                    )
                    return {
                        'original_prompt': evolved_data['original_prompt'],
                        'evolved_prompt': evolved_data['evolved_prompt'],
                        'response': response,
                        'operation': evolved_data['operation'],
                        'category': evolved_data['category'],
                        'expected_answer': evolved_data['expected_answer'],
                        'difficulty': evolved_data['difficulty'],
                        'success': True
                    }
                except Exception as e:
                    logger.warning(f"Failed to get evolved response: {e}")
                    return {
                        'original_prompt': evolved_data['original_prompt'],
                        'evolved_prompt': evolved_data['evolved_prompt'],
                        'response': "",
                        'operation': evolved_data['operation'],
                        'category': evolved_data['category'],
                        'expected_answer': evolved_data['expected_answer'],
                        'difficulty': evolved_data['difficulty'],
                        'success': False
                    }
        
        tasks = [get_evolved_response(evolved_data) for evolved_data in evolved_prompts]
        responses = await asyncio.gather(*tasks)
        
        successful_responses = [r for r in responses if r['success']]
        logger.info(f"Got {len(successful_responses)} evolved responses")
        
        return successful_responses
    
    async def _calculate_robustness_metrics(self, baseline_responses: List[Dict[str, Any]], evolved_responses: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate robustness metrics."""
        logger.info("Calculating robustness metrics...")
        
        if not baseline_responses or not evolved_responses:
            return {
                "evolution_robustness_pdr": 0.0,
                "recall_performance_rop": 0.0,
                "multi_round_degradation": 0.0,
                "operation_breakdown": {},
                "category_breakdown": {}
            }
        
        # Create baseline lookup
        baseline_lookup = {resp['prompt']: resp for resp in baseline_responses}
        
        # Calculate performance degradation for each evolved response
        degradations = []
        operation_degradations = {}
        category_degradations = {}
        
        for evolved_resp in evolved_responses:
            original_prompt = evolved_resp['original_prompt']
            if original_prompt in baseline_lookup:
                baseline_resp = baseline_lookup[original_prompt]
                
                # Calculate semantic similarity/quality degradation
                degradation = self._calculate_response_degradation(
                    baseline_resp['response'],
                    evolved_resp['response'],
                    evolved_resp['expected_answer']
                )
                
                degradations.append(degradation)
                
                # Track by operation
                operation = evolved_resp['operation']
                if operation not in operation_degradations:
                    operation_degradations[operation] = []
                operation_degradations[operation].append(degradation)
                
                # Track by category
                category = evolved_resp['category']
                if category not in category_degradations:
                    category_degradations[category] = []
                category_degradations[category].append(degradation)
        
        # Calculate metrics
        avg_degradation = sum(degradations) / len(degradations) if degradations else 0.0
        evolution_robustness_pdr = avg_degradation  # Performance Degradation Rate
        
        # Calculate operation-wise breakdown
        operation_breakdown = {}
        for operation, degs in operation_degradations.items():
            operation_breakdown[operation] = sum(degs) / len(degs)
        
        # Calculate category-wise breakdown
        category_breakdown = {}
        for category, degs in category_degradations.items():
            category_breakdown[category] = sum(degs) / len(degs)
        
        # Calculate recall performance (ability to maintain core information)
        recall_performance_rop = 1.0 - evolution_robustness_pdr
        
        # Multi-round degradation (simulate multiple evolutions)
        multi_round_degradation = min(1.0, evolution_robustness_pdr * 1.5)
        
        return {
            "evolution_robustness_pdr": evolution_robustness_pdr,
            "recall_performance_rop": recall_performance_rop,
            "multi_round_degradation": multi_round_degradation,
            "operation_breakdown": operation_breakdown,
            "category_breakdown": category_breakdown,
            "total_comparisons": len(degradations),
            "avg_degradation": avg_degradation
        }
    
    def _calculate_response_degradation(self, baseline_response: str, evolved_response: str, expected_answer: str) -> float:
        """Calculate performance degradation between baseline and evolved responses."""
        # Simple heuristic-based degradation calculation
        # In production, this would use more sophisticated similarity metrics
        
        baseline_lower = baseline_response.lower()
        evolved_lower = evolved_response.lower()
        expected_lower = expected_answer.lower()
        
        # Check if both responses contain expected answer
        baseline_correct = any(term in baseline_lower for term in expected_lower.split())
        evolved_correct = any(term in evolved_lower for term in expected_lower.split())
        
        # Length comparison
        length_ratio = len(evolved_response) / max(len(baseline_response), 1)
        length_degradation = abs(1.0 - length_ratio)
        
        # Content degradation
        if baseline_correct and not evolved_correct:
            content_degradation = 1.0  # Complete failure
        elif not baseline_correct and evolved_correct:
            content_degradation = 0.0  # Improvement
        elif baseline_correct and evolved_correct:
            content_degradation = 0.1  # Minor degradation
        else:
            content_degradation = 0.5  # Both incorrect
        
        # Overall degradation (weighted average)
        overall_degradation = 0.7 * content_degradation + 0.3 * min(1.0, length_degradation)
        
        return overall_degradation
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get evaluator metadata."""
        return {
            "framework": "AutoEvoEval",
            "version": "1.0",
            "description": "Robustness evaluation using 22 atomic evolution operations",
            "evolution_operations": list(self.evolution_operations.keys()),
            "operation_count": len(self.evolution_operations),
            "sample_size": self.sample_size,
            "random_seed": self.random_seed
        }