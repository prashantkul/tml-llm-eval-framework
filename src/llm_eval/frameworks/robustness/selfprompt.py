"""SelfPrompt evaluation framework for self-consistency and reliability."""

import asyncio
import logging
import random
import json
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter
from pathlib import Path

from ...core.model_interface import ModelInterface

logger = logging.getLogger(__name__)


class SelfPromptEvaluator:
    """Evaluator for SelfPrompt consistency and reliability assessment."""
    
    def __init__(self, config):
        self.config = config
        self.sample_size = getattr(config, 'sample_size', 100)
        self.random_seed = getattr(config, 'random_seed', 42)
        random.seed(self.random_seed)
        
        # Self-prompt evaluation categories
        self.evaluation_categories = [
            "factual_consistency",
            "logical_reasoning",
            "instruction_following",
            "creative_consistency",
            "mathematical_accuracy",
            "language_understanding"
        ]
        
        # Consistency test types
        self.consistency_tests = [
            "multi_generation",      # Same prompt, multiple generations
            "paraphrase_consistency", # Paraphrased prompts
            "context_variation",     # Different contexts, same question
            "temperature_variation", # Different temperature settings
            "multi_turn_consistency" # Multi-turn conversation consistency
        ]
        
        # Load test prompts for consistency evaluation
        self.test_prompts = self._load_consistency_prompts()
        
        logger.info(f"SelfPrompt evaluator initialized with {len(self.evaluation_categories)} categories and {len(self.test_prompts)} test prompts")
    
    def _load_consistency_prompts(self) -> List[Dict[str, Any]]:
        """Load prompts for consistency testing."""
        prompts = [
            {
                "prompt": "What is the capital of Japan?",
                "category": "factual_consistency",
                "expected_answer": "Tokyo",
                "variations": [
                    "Which city is the capital of Japan?",
                    "Name the capital city of Japan.",
                    "What city serves as Japan's capital?"
                ]
            },
            {
                "prompt": "If all cats are animals and all animals need food, do cats need food?",
                "category": "logical_reasoning",
                "expected_answer": "Yes",
                "variations": [
                    "Given that cats are animals and animals require food, must cats require food?",
                    "Since all cats are animals, and all animals need food, is it true that cats need food?",
                    "Cats are animals. Animals need food. Therefore, cats need what?"
                ]
            },
            {
                "prompt": "Write a haiku about summer.",
                "category": "creative_consistency",
                "expected_answer": "5-7-5 syllable poem about summer",
                "variations": [
                    "Compose a haiku poem with summer as the theme.",
                    "Create a traditional Japanese haiku about the summer season.",
                    "Write a three-line haiku poem describing summer."
                ]
            },
            {
                "prompt": "Calculate 25 × 4.",
                "category": "mathematical_accuracy",
                "expected_answer": "100",
                "variations": [
                    "What is 25 multiplied by 4?",
                    "Solve: 25 × 4 = ?",
                    "Find the product of 25 and 4."
                ]
            },
            {
                "prompt": "Follow these instructions: 1) Say hello, 2) Give your name, 3) Ask how I am.",
                "category": "instruction_following",
                "expected_answer": "Hello, I am [name], how are you?",
                "variations": [
                    "Please do the following: greet me, introduce yourself, and inquire about my wellbeing.",
                    "I need you to: say hello, tell me your name, then ask about my condition.",
                    "Execute these steps in order: greeting, self-introduction, health inquiry."
                ]
            },
            {
                "prompt": "Explain what 'running out of time' means.",
                "category": "language_understanding",
                "expected_answer": "Having little time remaining",
                "variations": [
                    "What does the phrase 'running out of time' mean?",
                    "Define the expression 'running out of time'.",
                    "Interpret the meaning of 'running out of time'."
                ]
            },
            {
                "prompt": "List three benefits of exercise.",
                "category": "factual_consistency",
                "expected_answer": "Health benefits like fitness, strength, mental health",
                "variations": [
                    "Name three advantages of physical exercise.",
                    "What are three positive effects of exercising?",
                    "Provide three benefits that come from regular exercise."
                ]
            },
            {
                "prompt": "If it's raining, you should take an umbrella. It's raining. What should you do?",
                "category": "logical_reasoning",
                "expected_answer": "Take an umbrella",
                "variations": [
                    "Given: If rain, then umbrella. It is raining. Therefore?",
                    "Rain means umbrella needed. It's raining now. Your action?",
                    "Rule: Rain → umbrella. Current state: raining. Conclusion?"
                ]
            },
            {
                "prompt": "Solve for x: 2x + 6 = 14",
                "category": "mathematical_accuracy",
                "expected_answer": "x = 4",
                "variations": [
                    "Find x in the equation 2x + 6 = 14",
                    "What is x when 2x + 6 = 14?",
                    "Determine the value of x: 2x + 6 = 14"
                ]
            },
            {
                "prompt": "Describe the color blue to someone who has never seen it.",
                "category": "creative_consistency",
                "expected_answer": "Cool, calming color like sky or ocean",
                "variations": [
                    "How would you explain the color blue to a person born blind?",
                    "Describe blue without using visual references.",
                    "Help someone understand blue who cannot see colors."
                ]
            }
        ]
        
        return prompts
    
    async def evaluate(self, model: ModelInterface) -> Dict[str, float]:
        """Run SelfPrompt consistency evaluation."""
        logger.info("Starting SelfPrompt consistency evaluation")
        
        # Run different consistency tests
        test_results = {}
        
        # Multi-generation consistency
        test_results['multi_generation'] = await self._test_multi_generation_consistency(model)
        
        # Paraphrase consistency
        test_results['paraphrase_consistency'] = await self._test_paraphrase_consistency(model)
        
        # Context variation consistency
        test_results['context_variation'] = await self._test_context_variation_consistency(model)
        
        # Temperature variation consistency  
        test_results['temperature_variation'] = await self._test_temperature_variation_consistency(model)
        
        # Multi-turn consistency
        test_results['multi_turn_consistency'] = await self._test_multi_turn_consistency(model)
        
        # Calculate overall metrics
        metrics = self._calculate_consistency_metrics(test_results)
        
        logger.info(f"SelfPrompt evaluation completed. Overall consistency: {metrics.get('overall_consistency', 0):.3f}")
        
        return metrics
    
    async def _test_multi_generation_consistency(self, model: ModelInterface) -> Dict[str, Any]:
        """Test consistency across multiple generations of the same prompt."""
        logger.info("Testing multi-generation consistency...")
        
        results = []
        generations_per_prompt = 5  # Generate 5 responses per prompt
        
        for prompt_data in self.test_prompts:
            prompt_results = await self._generate_multiple_responses(
                model, prompt_data['prompt'], generations_per_prompt
            )
            
            consistency_score = self._calculate_response_consistency(
                prompt_results, prompt_data['expected_answer']
            )
            
            results.append({
                'prompt': prompt_data['prompt'],
                'category': prompt_data['category'],
                'responses': prompt_results,
                'consistency_score': consistency_score
            })
        
        avg_consistency = sum(r['consistency_score'] for r in results) / len(results)
        
        return {
            'test_type': 'multi_generation',
            'avg_consistency': avg_consistency,
            'results': results,
            'generations_per_prompt': generations_per_prompt
        }
    
    async def _test_paraphrase_consistency(self, model: ModelInterface) -> Dict[str, Any]:
        """Test consistency across paraphrased versions of prompts."""
        logger.info("Testing paraphrase consistency...")
        
        results = []
        
        for prompt_data in self.test_prompts:
            original_response = await model.generate(prompt_data['prompt'], temperature=0.7)
            
            paraphrase_responses = []
            for variation in prompt_data['variations']:
                response = await model.generate(variation, temperature=0.7)
                paraphrase_responses.append(response)
            
            # Calculate consistency between original and paraphrases
            all_responses = [original_response] + paraphrase_responses
            consistency_score = self._calculate_response_consistency(
                all_responses, prompt_data['expected_answer']
            )
            
            results.append({
                'prompt': prompt_data['prompt'],
                'category': prompt_data['category'],
                'original_response': original_response,
                'paraphrase_responses': paraphrase_responses,
                'consistency_score': consistency_score
            })
        
        avg_consistency = sum(r['consistency_score'] for r in results) / len(results)
        
        return {
            'test_type': 'paraphrase_consistency',
            'avg_consistency': avg_consistency,
            'results': results
        }
    
    async def _test_context_variation_consistency(self, model: ModelInterface) -> Dict[str, Any]:
        """Test consistency with different contexts."""
        logger.info("Testing context variation consistency...")
        
        contexts = [
            "In an academic setting: ",
            "For a casual conversation: ",
            "In a professional context: ",
            "For educational purposes: ",
            ""  # No context
        ]
        
        results = []
        
        for prompt_data in self.test_prompts:
            context_responses = []
            
            for context in contexts:
                contextualized_prompt = context + prompt_data['prompt']
                response = await model.generate(contextualized_prompt, temperature=0.7)
                context_responses.append(response)
            
            consistency_score = self._calculate_response_consistency(
                context_responses, prompt_data['expected_answer']
            )
            
            results.append({
                'prompt': prompt_data['prompt'],
                'category': prompt_data['category'],
                'context_responses': context_responses,
                'consistency_score': consistency_score
            })
        
        avg_consistency = sum(r['consistency_score'] for r in results) / len(results)
        
        return {
            'test_type': 'context_variation',
            'avg_consistency': avg_consistency,
            'results': results,
            'contexts_tested': len(contexts)
        }
    
    async def _test_temperature_variation_consistency(self, model: ModelInterface) -> Dict[str, Any]:
        """Test consistency across different temperature settings."""
        logger.info("Testing temperature variation consistency...")
        
        temperatures = [0.1, 0.3, 0.5, 0.7, 0.9]
        results = []
        
        for prompt_data in self.test_prompts:
            temp_responses = []
            
            for temp in temperatures:
                response = await model.generate(prompt_data['prompt'], temperature=temp)
                temp_responses.append(response)
            
            consistency_score = self._calculate_response_consistency(
                temp_responses, prompt_data['expected_answer']
            )
            
            results.append({
                'prompt': prompt_data['prompt'],
                'category': prompt_data['category'],
                'temperature_responses': temp_responses,
                'consistency_score': consistency_score
            })
        
        avg_consistency = sum(r['consistency_score'] for r in results) / len(results)
        
        return {
            'test_type': 'temperature_variation',
            'avg_consistency': avg_consistency,
            'results': results,
            'temperatures_tested': temperatures
        }
    
    async def _test_multi_turn_consistency(self, model: ModelInterface) -> Dict[str, Any]:
        """Test consistency in multi-turn conversations."""
        logger.info("Testing multi-turn consistency...")
        
        # Create multi-turn scenarios
        multi_turn_scenarios = [
            {
                "turns": [
                    "What is the capital of France?",
                    "Can you confirm what you just said about France's capital?",
                    "Is that the same city you mentioned before?"
                ],
                "expected_consistency": "Paris mentioned consistently",
                "category": "factual_consistency"
            },
            {
                "turns": [
                    "Calculate 15 + 25",
                    "What was the result of that calculation?",
                    "Can you verify that sum again?"
                ],
                "expected_consistency": "40 mentioned consistently", 
                "category": "mathematical_accuracy"
            },
            {
                "turns": [
                    "What are the benefits of exercise?",
                    "Can you elaborate on those benefits you mentioned?",
                    "Are those the same advantages you listed before?"
                ],
                "expected_consistency": "Consistent exercise benefits",
                "category": "factual_consistency"
            }
        ]
        
        results = []
        
        for scenario in multi_turn_scenarios:
            turn_responses = []
            conversation_context = ""
            
            for turn in scenario['turns']:
                # Build context from previous turns
                current_prompt = conversation_context + turn
                response = await model.generate(current_prompt, temperature=0.7)
                turn_responses.append(response)
                
                # Update context for next turn
                conversation_context += f"Human: {turn}\nAssistant: {response}\n"
            
            # Analyze consistency across turns
            consistency_score = self._analyze_multi_turn_consistency(
                turn_responses, scenario['expected_consistency']
            )
            
            results.append({
                'scenario': scenario,
                'turn_responses': turn_responses,
                'consistency_score': consistency_score
            })
        
        avg_consistency = sum(r['consistency_score'] for r in results) / len(results)
        
        return {
            'test_type': 'multi_turn_consistency',
            'avg_consistency': avg_consistency,
            'results': results,
            'scenarios_tested': len(multi_turn_scenarios)
        }
    
    async def _generate_multiple_responses(self, model: ModelInterface, prompt: str, count: int) -> List[str]:
        """Generate multiple responses to the same prompt."""
        tasks = []
        semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        
        async def generate_single():
            async with semaphore:
                return await model.generate(prompt, temperature=0.7)
        
        tasks = [generate_single() for _ in range(count)]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_responses = [r for r in responses if isinstance(r, str)]
        return valid_responses
    
    def _calculate_response_consistency(self, responses: List[str], expected_answer: str) -> float:
        """Calculate consistency score for a set of responses."""
        if len(responses) < 2:
            return 1.0
        
        # Extract key information from responses
        key_info_sets = []
        for response in responses:
            key_info = self._extract_key_information(response, expected_answer)
            key_info_sets.append(key_info)
        
        # Calculate pairwise consistency
        consistency_scores = []
        for i in range(len(key_info_sets)):
            for j in range(i + 1, len(key_info_sets)):
                similarity = self._calculate_semantic_similarity(
                    key_info_sets[i], key_info_sets[j]
                )
                consistency_scores.append(similarity)
        
        return sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0.0
    
    def _extract_key_information(self, response: str, expected_answer: str) -> set:
        """Extract key information from response."""
        response_lower = response.lower()
        expected_lower = expected_answer.lower()
        
        # Extract key terms based on expected answer
        key_terms = set()
        
        # Add words from expected answer
        for word in expected_lower.split():
            if word in response_lower:
                key_terms.add(word)
        
        # Add numbers if present
        import re
        numbers = re.findall(r'\d+', response)
        key_terms.update(numbers)
        
        # Add first few words of response (for structure)
        response_words = response_lower.split()[:5]
        key_terms.update(response_words)
        
        return key_terms
    
    def _calculate_semantic_similarity(self, info1: set, info2: set) -> float:
        """Calculate semantic similarity between two information sets."""
        if not info1 and not info2:
            return 1.0
        if not info1 or not info2:
            return 0.0
        
        intersection = len(info1.intersection(info2))
        union = len(info1.union(info2))
        
        return intersection / union if union > 0 else 0.0
    
    def _analyze_multi_turn_consistency(self, turn_responses: List[str], expected_consistency: str) -> float:
        """Analyze consistency across multiple conversation turns."""
        if len(turn_responses) < 2:
            return 1.0
        
        # Extract key information from each turn
        key_info_by_turn = []
        for response in turn_responses:
            key_info = self._extract_key_information(response, expected_consistency)
            key_info_by_turn.append(key_info)
        
        # Check for consistency of key information across turns
        common_info = key_info_by_turn[0]
        for info_set in key_info_by_turn[1:]:
            common_info = common_info.intersection(info_set)
        
        # Calculate consistency based on persistent key information
        total_key_info = set()
        for info_set in key_info_by_turn:
            total_key_info.update(info_set)
        
        consistency_ratio = len(common_info) / len(total_key_info) if total_key_info else 1.0
        return consistency_ratio
    
    def _calculate_consistency_metrics(self, test_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate overall consistency metrics."""
        # Extract consistency scores from each test
        consistency_scores = []
        test_breakdowns = {}
        
        for test_name, test_data in test_results.items():
            avg_consistency = test_data.get('avg_consistency', 0.0)
            consistency_scores.append(avg_consistency)
            test_breakdowns[test_name] = avg_consistency
        
        # Overall consistency
        overall_consistency = sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0.0
        
        # Category-wise breakdown
        category_scores = {}
        category_counts = {}
        
        for test_data in test_results.values():
            if 'results' in test_data:
                for result in test_data['results']:
                    if 'category' in result:
                        category = result['category']
                        score = result['consistency_score']
                        
                        if category not in category_scores:
                            category_scores[category] = 0.0
                            category_counts[category] = 0
                        
                        category_scores[category] += score
                        category_counts[category] += 1
        
        category_breakdown = {}
        for category, total_score in category_scores.items():
            category_breakdown[category] = total_score / category_counts[category]
        
        # Reliability metrics
        self_consistency = overall_consistency
        hallucination_rate = 1.0 - overall_consistency  # Inverse relationship
        
        return {
            "overall_consistency": overall_consistency,
            "self_consistency": self_consistency,
            "hallucination_rate": hallucination_rate,
            "test_breakdown": test_breakdowns,
            "category_breakdown": category_breakdown,
            "reliability_score": overall_consistency,
            "total_tests": len(test_results)
        }
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get evaluator metadata."""
        return {
            "framework": "SelfPrompt",
            "version": "1.0",
            "description": "Self-consistency and reliability evaluation across multiple dimensions",
            "evaluation_categories": self.evaluation_categories,
            "consistency_tests": self.consistency_tests,
            "sample_size": self.sample_size,
            "random_seed": self.random_seed
        }