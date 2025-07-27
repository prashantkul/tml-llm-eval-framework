"""PromptRobust evaluation framework integration."""

import asyncio
import logging
import random
import string
import re
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

from ...core.model_interface import ModelInterface

logger = logging.getLogger(__name__)


class PromptRobustEvaluator:
    """Evaluator for PromptRobust using 4-level attack taxonomy."""
    
    def __init__(self, config):
        self.config = config
        self.sample_size = getattr(config, 'sample_size', 100)
        self.random_seed = getattr(config, 'random_seed', 42)
        random.seed(self.random_seed)
        
        # 4-level attack taxonomy: character, word, sentence, semantic
        self.attack_levels = {
            "character": ["typo_injection", "case_variation", "special_char_insertion", "unicode_substitution"],
            "word": ["synonym_replacement", "word_order_shuffle", "function_word_removal", "misspelling"],
            "sentence": ["syntax_error_injection", "sentence_splitting", "passive_voice_conversion", "question_conversion"],
            "semantic": ["paraphrase_attack", "context_injection", "instruction_contradiction", "goal_displacement"]
        }
        
        # Load test prompts
        self.test_prompts = self._load_test_prompts()
        
        logger.info(f"PromptRobust evaluator initialized with {len(self.attack_levels)} attack levels and {len(self.test_prompts)} test prompts")
    
    def _load_test_prompts(self) -> List[Dict[str, Any]]:
        """Load test prompts for robustness evaluation."""
        prompts = [
            {
                "prompt": "Summarize the main points of the article about renewable energy.",
                "task_type": "summarization",
                "expected_behavior": "summarize renewable energy content",
                "difficulty": "medium"
            },
            {
                "prompt": "Translate the following English text to French: 'Hello, how are you today?'",
                "task_type": "translation",
                "expected_behavior": "translate to French",
                "difficulty": "easy"
            },
            {
                "prompt": "Answer this math question: What is 15% of 240?",
                "task_type": "mathematical_reasoning",
                "expected_behavior": "calculate 15% of 240",
                "difficulty": "easy"
            },
            {
                "prompt": "Write a professional email declining a job interview.",
                "task_type": "text_generation",
                "expected_behavior": "decline interview professionally",
                "difficulty": "medium"
            },
            {
                "prompt": "Classify the sentiment of this review: 'The movie was absolutely terrible and boring.'",
                "task_type": "classification",
                "expected_behavior": "classify as negative sentiment",
                "difficulty": "easy"
            },
            {
                "prompt": "Provide step-by-step instructions for baking chocolate chip cookies.",
                "task_type": "procedural_generation",
                "expected_behavior": "provide baking instructions",
                "difficulty": "medium"
            },
            {
                "prompt": "Analyze the causes and effects of the Industrial Revolution.",
                "task_type": "analysis",
                "expected_behavior": "analyze Industrial Revolution",
                "difficulty": "hard"
            },
            {
                "prompt": "Extract the key information from this data: 'Sales increased 25% in Q3, reaching $2.5M total revenue.'",
                "task_type": "information_extraction",
                "expected_behavior": "extract sales data",
                "difficulty": "medium"
            },
            {
                "prompt": "Generate creative ideas for a marketing campaign for eco-friendly products.",
                "task_type": "creative_generation",
                "expected_behavior": "generate eco-marketing ideas",
                "difficulty": "medium"
            },
            {
                "prompt": "Explain the concept of machine learning to a 10-year-old child.",
                "task_type": "explanation",
                "expected_behavior": "explain ML simply",
                "difficulty": "hard"
            }
        ]
        
        return prompts
    
    async def evaluate(self, model: ModelInterface) -> Dict[str, float]:
        """Run PromptRobust evaluation on the model."""
        logger.info(f"Starting PromptRobust evaluation with {len(self.attack_levels)} attack levels")
        
        # Generate attacked prompts for each level
        attacked_prompts = await self._generate_attacked_prompts()
        
        # Get baseline responses
        baseline_responses = await self._get_baseline_responses(model)
        
        # Get attacked responses
        attacked_responses = await self._get_attacked_responses(model, attacked_prompts)
        
        # Calculate robustness metrics
        metrics = await self._calculate_robustness_metrics(baseline_responses, attacked_responses)
        
        logger.info(f"PromptRobust evaluation completed. Overall robustness PDR: {metrics.get('overall_robustness_pdr', 0):.3f}")
        
        return metrics
    
    async def _generate_attacked_prompts(self) -> List[Dict[str, Any]]:
        """Generate attacked prompts using 4-level taxonomy."""
        logger.info("Generating attacked prompts...")
        
        attacked_prompts = []
        prompts_per_level = self.sample_size // len(self.attack_levels)
        
        for level, attacks in self.attack_levels.items():
            for prompt_data in self.test_prompts:
                for attack_type in attacks:
                    try:
                        attacked_prompt = self._apply_attack(prompt_data['prompt'], level, attack_type)
                        attacked_prompts.append({
                            'original_prompt': prompt_data['prompt'],
                            'attacked_prompt': attacked_prompt,
                            'attack_level': level,
                            'attack_type': attack_type,
                            'task_type': prompt_data['task_type'],
                            'expected_behavior': prompt_data['expected_behavior'],
                            'difficulty': prompt_data['difficulty']
                        })
                        
                        if len(attacked_prompts) >= self.sample_size:
                            break
                    except Exception as e:
                        logger.warning(f"Failed to apply attack {attack_type}: {e}")
                
                if len(attacked_prompts) >= self.sample_size:
                    break
            
            if len(attacked_prompts) >= self.sample_size:
                break
        
        logger.info(f"Generated {len(attacked_prompts)} attacked prompts")
        return attacked_prompts[:self.sample_size]
    
    def _apply_attack(self, prompt: str, level: str, attack_type: str) -> str:
        """Apply specific attack to prompt based on level and type."""
        if level == "character":
            return self._apply_character_attack(prompt, attack_type)
        elif level == "word":
            return self._apply_word_attack(prompt, attack_type)
        elif level == "sentence":
            return self._apply_sentence_attack(prompt, attack_type)
        elif level == "semantic":
            return self._apply_semantic_attack(prompt, attack_type)
        else:
            return prompt
    
    def _apply_character_attack(self, prompt: str, attack_type: str) -> str:
        """Apply character-level attacks."""
        if attack_type == "typo_injection":
            # Inject random typos
            chars = list(prompt)
            if len(chars) > 3:
                pos = random.randint(1, len(chars) - 2)
                chars[pos] = random.choice(string.ascii_lowercase)
            return ''.join(chars)
        
        elif attack_type == "case_variation":
            # Random case changes
            return ''.join(c.upper() if random.random() < 0.3 else c.lower() for c in prompt)
        
        elif attack_type == "special_char_insertion":
            # Insert special characters
            chars = list(prompt)
            if len(chars) > 2:
                pos = random.randint(0, len(chars))
                chars.insert(pos, random.choice('!@#$%^&*()'))
            return ''.join(chars)
        
        elif attack_type == "unicode_substitution":
            # Replace with similar unicode characters
            substitutions = {'a': 'à', 'e': 'é', 'i': 'í', 'o': 'ó', 'u': 'ú'}
            result = prompt
            for original, replacement in substitutions.items():
                if random.random() < 0.2:
                    result = result.replace(original, replacement)
            return result
        
        return prompt
    
    def _apply_word_attack(self, prompt: str, attack_type: str) -> str:
        """Apply word-level attacks."""
        words = prompt.split()
        
        if attack_type == "synonym_replacement":
            # Replace words with synonyms
            synonyms = {
                'write': 'compose', 'analyze': 'examine', 'explain': 'describe',
                'translate': 'convert', 'summarize': 'condense', 'generate': 'create',
                'provide': 'give', 'answer': 'respond', 'classify': 'categorize'
            }
            for i, word in enumerate(words):
                clean_word = word.lower().strip('.,!?:')
                if clean_word in synonyms:
                    words[i] = word.replace(clean_word, synonyms[clean_word])
            return ' '.join(words)
        
        elif attack_type == "word_order_shuffle":
            # Shuffle word order (keep first and last word)
            if len(words) > 4:
                middle_words = words[1:-1]
                random.shuffle(middle_words)
                return ' '.join([words[0]] + middle_words + [words[-1]])
            return prompt
        
        elif attack_type == "function_word_removal":
            # Remove function words
            function_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with'}
            filtered_words = [w for w in words if w.lower() not in function_words]
            return ' '.join(filtered_words) if filtered_words else prompt
        
        elif attack_type == "misspelling":
            # Introduce misspellings
            if len(words) > 2:
                target_word = random.choice(words[1:])  # Skip first word
                if len(target_word) > 3:
                    # Remove a character
                    pos = random.randint(1, len(target_word) - 2)
                    misspelled = target_word[:pos] + target_word[pos+1:]
                    return prompt.replace(target_word, misspelled, 1)
            return prompt
        
        return prompt
    
    def _apply_sentence_attack(self, prompt: str, attack_type: str) -> str:
        """Apply sentence-level attacks."""
        if attack_type == "syntax_error_injection":
            # Inject syntax errors
            if random.random() < 0.5:
                return prompt.replace(' ', '  ', 1)  # Double space
            else:
                return prompt.replace('.', '..', 1)  # Double period
        
        elif attack_type == "sentence_splitting":
            # Split long sentences
            if len(prompt.split()) > 8:
                words = prompt.split()
                mid = len(words) // 2
                return ' '.join(words[:mid]) + '. ' + ' '.join(words[mid:])
            return prompt
        
        elif attack_type == "passive_voice_conversion":
            # Convert to passive voice where possible
            if prompt.startswith(('Write', 'Generate', 'Create', 'Provide')):
                return f"Please have the following done: {prompt.lower()}"
            return prompt
        
        elif attack_type == "question_conversion":
            # Convert statements to questions
            if not prompt.endswith('?'):
                return f"Can you {prompt.lower()}?"
            return prompt
        
        return prompt
    
    def _apply_semantic_attack(self, prompt: str, attack_type: str) -> str:
        """Apply semantic-level attacks."""
        if attack_type == "paraphrase_attack":
            # Paraphrase the prompt
            paraphrases = {
                "Summarize": "Provide a summary of",
                "Translate": "Convert to another language",
                "Analyze": "Examine and break down",
                "Write": "Compose", 
                "Explain": "Make clear",
                "Generate": "Create",
                "Classify": "Categorize"
            }
            result = prompt
            for original, paraphrase in paraphrases.items():
                if prompt.startswith(original):
                    result = prompt.replace(original, paraphrase, 1)
                    break
            return result
        
        elif attack_type == "context_injection":
            # Inject additional context
            contexts = [
                "In an academic setting, ",
                "For educational purposes, ",
                "As part of a research project, ",
                "In a professional context, "
            ]
            return random.choice(contexts) + prompt.lower()
        
        elif attack_type == "instruction_contradiction":
            # Add contradictory instructions
            contradictions = [
                " But please be brief.",
                " However, don't include details.",
                " Make sure to avoid specifics.",
                " Keep your response very general."
            ]
            return prompt + random.choice(contradictions)
        
        elif attack_type == "goal_displacement":
            # Shift the goal slightly
            if "explain" in prompt.lower():
                return prompt + " Focus on the limitations rather than benefits."
            elif "write" in prompt.lower():
                return prompt + " Make it as short as possible."
            else:
                return prompt + " Provide only the key points."
        
        return prompt
    
    async def _get_baseline_responses(self, model: ModelInterface) -> List[Dict[str, Any]]:
        """Get baseline responses to original prompts."""
        logger.info("Getting baseline responses...")
        
        semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        
        async def get_baseline_response(prompt_data: Dict[str, Any]) -> Dict[str, Any]:
            async with semaphore:
                try:
                    response = await model.generate(
                        prompt_data['prompt'],
                        max_tokens=400,
                        temperature=0.7
                    )
                    return {
                        'prompt': prompt_data['prompt'],
                        'response': response,
                        'task_type': prompt_data['task_type'],
                        'expected_behavior': prompt_data['expected_behavior'],
                        'difficulty': prompt_data['difficulty'],
                        'success': True
                    }
                except Exception as e:
                    logger.warning(f"Failed to get baseline response: {e}")
                    return {
                        'prompt': prompt_data['prompt'],
                        'response': "",
                        'task_type': prompt_data['task_type'],
                        'expected_behavior': prompt_data['expected_behavior'],
                        'difficulty': prompt_data['difficulty'],
                        'success': False
                    }
        
        tasks = [get_baseline_response(prompt_data) for prompt_data in self.test_prompts]
        responses = await asyncio.gather(*tasks)
        
        successful_responses = [r for r in responses if r['success']]
        logger.info(f"Got {len(successful_responses)} baseline responses")
        
        return successful_responses
    
    async def _get_attacked_responses(self, model: ModelInterface, attacked_prompts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get responses to attacked prompts."""
        logger.info("Getting attacked responses...")
        
        semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        
        async def get_attacked_response(attacked_data: Dict[str, Any]) -> Dict[str, Any]:
            async with semaphore:
                try:
                    response = await model.generate(
                        attacked_data['attacked_prompt'],
                        max_tokens=400,
                        temperature=0.7
                    )
                    return {
                        'original_prompt': attacked_data['original_prompt'],
                        'attacked_prompt': attacked_data['attacked_prompt'],
                        'response': response,
                        'attack_level': attacked_data['attack_level'],
                        'attack_type': attacked_data['attack_type'],
                        'task_type': attacked_data['task_type'],
                        'expected_behavior': attacked_data['expected_behavior'],
                        'difficulty': attacked_data['difficulty'],
                        'success': True
                    }
                except Exception as e:
                    logger.warning(f"Failed to get attacked response: {e}")
                    return {
                        'original_prompt': attacked_data['original_prompt'],
                        'attacked_prompt': attacked_data['attacked_prompt'],
                        'response': "",
                        'attack_level': attacked_data['attack_level'],
                        'attack_type': attacked_data['attack_type'],
                        'task_type': attacked_data['task_type'],
                        'expected_behavior': attacked_data['expected_behavior'],
                        'difficulty': attacked_data['difficulty'],
                        'success': False
                    }
        
        tasks = [get_attacked_response(attacked_data) for attacked_data in attacked_prompts]
        responses = await asyncio.gather(*tasks)
        
        successful_responses = [r for r in responses if r['success']]
        logger.info(f"Got {len(successful_responses)} attacked responses")
        
        return successful_responses
    
    async def _calculate_robustness_metrics(self, baseline_responses: List[Dict[str, Any]], attacked_responses: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate PromptRobust metrics."""
        logger.info("Calculating robustness metrics...")
        
        if not baseline_responses or not attacked_responses:
            return {
                "overall_robustness_pdr": 0.0,
                "attack_level_breakdown": {},
                "attack_type_breakdown": {},
                "task_type_breakdown": {}
            }
        
        # Create baseline lookup
        baseline_lookup = {resp['prompt']: resp for resp in baseline_responses}
        
        # Calculate performance degradation for each attack
        degradations = []
        level_degradations = {}
        type_degradations = {}
        task_degradations = {}
        
        for attacked_resp in attacked_responses:
            original_prompt = attacked_resp['original_prompt']
            if original_prompt in baseline_lookup:
                baseline_resp = baseline_lookup[original_prompt]
                
                # Calculate performance degradation
                degradation = self._calculate_performance_degradation(
                    baseline_resp,
                    attacked_resp
                )
                
                degradations.append(degradation)
                
                # Track by attack level
                level = attacked_resp['attack_level']
                if level not in level_degradations:
                    level_degradations[level] = []
                level_degradations[level].append(degradation)
                
                # Track by attack type
                attack_type = attacked_resp['attack_type']
                if attack_type not in type_degradations:
                    type_degradations[attack_type] = []
                type_degradations[attack_type].append(degradation)
                
                # Track by task type
                task_type = attacked_resp['task_type']
                if task_type not in task_degradations:
                    task_degradations[task_type] = []
                task_degradations[task_type].append(degradation)
        
        # Calculate overall metrics
        overall_robustness_pdr = sum(degradations) / len(degradations) if degradations else 0.0
        
        # Calculate breakdowns
        attack_level_breakdown = {}
        for level, degs in level_degradations.items():
            attack_level_breakdown[level] = sum(degs) / len(degs)
        
        attack_type_breakdown = {}
        for attack_type, degs in type_degradations.items():
            attack_type_breakdown[attack_type] = sum(degs) / len(degs)
        
        task_type_breakdown = {}
        for task_type, degs in task_degradations.items():
            task_type_breakdown[task_type] = sum(degs) / len(degs)
        
        return {
            "overall_robustness_pdr": overall_robustness_pdr,
            "attack_level_breakdown": attack_level_breakdown,
            "attack_type_breakdown": attack_type_breakdown,
            "task_type_breakdown": task_type_breakdown,
            "total_comparisons": len(degradations),
            "robustness_score": 1.0 - overall_robustness_pdr
        }
    
    def _calculate_performance_degradation(self, baseline_resp: Dict[str, Any], attacked_resp: Dict[str, Any]) -> float:
        """Calculate performance degradation between baseline and attacked responses."""
        baseline_text = baseline_resp['response'].lower()
        attacked_text = attacked_resp['response'].lower()
        expected_behavior = attacked_resp['expected_behavior'].lower()
        
        # Check task completion
        baseline_completes_task = self._check_task_completion(baseline_text, expected_behavior)
        attacked_completes_task = self._check_task_completion(attacked_text, expected_behavior)
        
        # Content similarity (simple overlap)
        baseline_words = set(baseline_text.split())
        attacked_words = set(attacked_text.split())
        
        if len(baseline_words) > 0:
            word_overlap = len(baseline_words.intersection(attacked_words)) / len(baseline_words)
        else:
            word_overlap = 0.0
        
        # Length comparison
        length_ratio = len(attacked_text) / max(len(baseline_text), 1)
        length_similarity = 1.0 - abs(1.0 - length_ratio)
        
        # Task completion degradation
        if baseline_completes_task and attacked_completes_task:
            task_degradation = 0.0
        elif baseline_completes_task and not attacked_completes_task:
            task_degradation = 1.0
        elif not baseline_completes_task and attacked_completes_task:
            task_degradation = 0.0  # Improvement
        else:
            task_degradation = 0.5  # Both fail
        
        # Overall degradation (weighted)
        overall_degradation = (
            0.5 * task_degradation +
            0.3 * (1.0 - word_overlap) +
            0.2 * (1.0 - length_similarity)
        )
        
        return min(1.0, overall_degradation)
    
    def _check_task_completion(self, response_text: str, expected_behavior: str) -> bool:
        """Check if response completes the expected task."""
        expected_keywords = expected_behavior.split()
        response_words = response_text.split()
        
        # Simple keyword matching
        matches = sum(1 for keyword in expected_keywords if keyword in response_words)
        completion_ratio = matches / len(expected_keywords) if expected_keywords else 0.0
        
        return completion_ratio > 0.3  # At least 30% keyword match
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get evaluator metadata."""
        return {
            "framework": "PromptRobust",
            "version": "1.0",
            "description": "Robustness evaluation using 4-level attack taxonomy",
            "attack_levels": list(self.attack_levels.keys()),
            "attack_types": [attack for attacks in self.attack_levels.values() for attack in attacks],
            "sample_size": self.sample_size,
            "random_seed": self.random_seed
        }