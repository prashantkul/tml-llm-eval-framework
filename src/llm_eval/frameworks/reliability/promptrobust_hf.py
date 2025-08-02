"""PromptRobust evaluator using Hugging Face dataset."""

import logging
import asyncio
import json
import random
from typing import Dict, List, Any, Optional
from datasets import load_dataset
from tqdm import tqdm

from ...core.model_interface import ModelInterface
from ...core.results import FrameworkResult

logger = logging.getLogger(__name__)


class PromptRobustHFEvaluator:
    """
    PromptRobust robustness evaluator using the Hugging Face dataset.
    
    This evaluator uses the PromptBench dataset from aryanagrawal1/promptbench
    to evaluate LLM robustness against adversarial prompt variations across
    character, word, sentence, and semantic levels.
    """
    
    def __init__(self, 
                 sample_size: Optional[int] = None,
                 attack_types: Optional[List[str]] = None,
                 difficulty_levels: Optional[List[str]] = None):
        """
        Initialize the PromptRobust evaluator.
        
        Args:
            sample_size: Number of samples to evaluate (None for all)
            attack_types: Types of attacks to test (None for all)
            difficulty_levels: Difficulty levels to include (None for all)
        """
        self.sample_size = sample_size
        self.attack_types = attack_types or ['character', 'word', 'sentence', 'semantic']
        self.difficulty_levels = difficulty_levels
        
        # Load the dataset
        try:
            logger.info("Loading PromptBench dataset from Hugging Face")
            self.dataset = load_dataset(
                "aryanagrawal1/promptbench",
                split="train"
            )
            
            # Add attack type classification based on the differences
            self.dataset = self._classify_attack_types(self.dataset)
            
            # Sample if specified
            if self.sample_size and len(self.dataset) > self.sample_size:
                self.dataset = self.dataset.shuffle(seed=42).select(range(self.sample_size))
            
            logger.info(f"Loaded {len(self.dataset)} samples from PromptBench dataset")
            
        except Exception as e:
            logger.error(f"Failed to load PromptBench dataset: {e}")
            # Fallback to synthetic data
            self.dataset = self._create_fallback_dataset()
    
    def _classify_attack_types(self, dataset):
        """Classify attack types based on differences between original and attacked prompts."""
        def classify_single_attack(sample):
            original = sample['original']
            attacked = sample['attacked']
            
            # Simple heuristic classification
            attack_type = self._determine_attack_type(original, attacked)
            
            return {
                'original': original,
                'attacked': attacked,
                'attack_type': attack_type,
                'id': f"sample_{hash(original + attacked) % 10000}"
            }
        
        return dataset.map(classify_single_attack)
    
    def _determine_attack_type(self, original: str, attacked: str) -> str:
        """Determine the type of attack based on the differences."""
        orig_words = original.split()
        att_words = attacked.split()
        
        # Check for sentence-level (length significantly different)
        if abs(len(att_words) - len(orig_words)) > 3:
            return "sentence"
        
        # Check for semantic (completely different structure)
        orig_set = set(original.lower().split())
        att_set = set(attacked.lower().split())
        overlap = len(orig_set.intersection(att_set)) / len(orig_set.union(att_set))
        
        if overlap < 0.3:
            return "semantic"
        
        # Check for word-level (word substitutions)
        word_changes = 0
        min_len = min(len(orig_words), len(att_words))
        for i in range(min_len):
            if orig_words[i].lower() != att_words[i].lower():
                word_changes += 1
        
        if word_changes > 0 and word_changes <= 2:
            return "word"
        
        # Default to character-level
        return "character"
    
    def _create_fallback_dataset(self) -> List[Dict[str, Any]]:
        """Create fallback synthetic dataset if HF dataset fails to load."""
        logger.warning("Using fallback synthetic dataset for PromptRobust")
        
        base_prompts = [
            "What is the capital of France?",
            "Solve this math problem: 2 + 2 = ?",
            "Translate 'hello' to Spanish.",
            "Write a summary of photosynthesis.",
            "Is this statement true or false: The Earth is flat?"
        ]
        
        synthetic_data = []
        
        for i, base_prompt in enumerate(base_prompts):
            # Create different types of adversarial variants
            variants = [
                (base_prompt, "original"),
                (self._apply_character_attack(base_prompt), "character"),
                (self._apply_word_attack(base_prompt), "word"),
                (self._apply_sentence_attack(base_prompt), "sentence"),
                (self._apply_semantic_attack(base_prompt), "semantic")
            ]
            
            for variant_prompt, attack_type in variants:
                synthetic_data.append({
                    "original": base_prompt,
                    "attacked": variant_prompt,
                    "attack_type": attack_type,
                    "id": f"synthetic_{i}_{attack_type}"
                })
        
        return synthetic_data
    
    def _apply_character_attack(self, text: str) -> str:
        """Apply character-level attack (typos, character manipulation)."""
        words = text.split()
        if not words:
            return text
            
        # Randomly introduce typos in 1-2 words
        num_changes = min(2, len(words))
        words_to_change = random.sample(range(len(words)), num_changes)
        
        for idx in words_to_change:
            word = words[idx]
            if len(word) > 2:
                # Random character manipulation
                attack_type = random.choice(['swap', 'delete', 'repeat'])
                if attack_type == 'swap' and len(word) > 3:
                    # Swap adjacent characters
                    pos = random.randint(1, len(word) - 2)
                    word_list = list(word)
                    word_list[pos], word_list[pos + 1] = word_list[pos + 1], word_list[pos]
                    words[idx] = ''.join(word_list)
                elif attack_type == 'delete':
                    # Delete a character
                    pos = random.randint(1, len(word) - 2)
                    words[idx] = word[:pos] + word[pos + 1:]
                elif attack_type == 'repeat':
                    # Repeat a character
                    pos = random.randint(1, len(word) - 1)
                    words[idx] = word[:pos] + word[pos] + word[pos:]
        
        return ' '.join(words)
    
    def _apply_word_attack(self, text: str) -> str:
        """Apply word-level attack (synonym replacement)."""
        # Simple synonym replacement
        synonyms = {
            'capital': 'main city',
            'France': 'French Republic',
            'hello': 'hi',
            'translate': 'convert',
            'Spanish': 'Castilian',
            'problem': 'question',
            'solve': 'resolve',
            'summary': 'overview',
            'statement': 'claim',
            'true': 'correct',
            'false': 'incorrect'
        }
        
        words = text.split()
        for i, word in enumerate(words):
            clean_word = word.lower().strip('.,!?:;')
            if clean_word in synonyms:
                # Replace with synonym while preserving punctuation
                replacement = synonyms[clean_word]
                if word[0].isupper():
                    replacement = replacement.capitalize()
                
                # Preserve punctuation
                punct = ''
                for j in range(len(word) - 1, -1, -1):
                    if word[j] in '.,!?:;':
                        punct = word[j] + punct
                    else:
                        break
                
                words[i] = replacement + punct
        
        return ' '.join(words)
    
    def _apply_sentence_attack(self, text: str) -> str:
        """Apply sentence-level attack (append irrelevant sentences)."""
        distractors = [
            " And true is true.",
            " Also, the sky is blue.",
            " Remember that water is wet.",
            " Additionally, 1 + 1 equals 2.",
            " Furthermore, ice is cold."
        ]
        
        distractor = random.choice(distractors)
        return text + distractor
    
    def _apply_semantic_attack(self, text: str) -> str:
        """Apply semantic-level attack (linguistic variation)."""
        # Simple paraphrasing/reformulation
        reformulations = {
            "What is the capital of France?": "Can you tell me France's capital city?",
            "Solve this math problem: 2 + 2 = ?": "Calculate the following: two plus two equals what?",
            "Translate 'hello' to Spanish.": "How do you say 'hello' in the Spanish language?",
            "Write a summary of photosynthesis.": "Please provide a brief overview of the photosynthesis process.",
            "Is this statement true or false: The Earth is flat?": "Determine whether this claim is accurate or not: The Earth is flat."
        }
        
        return reformulations.get(text, f"Please respond to this query: {text}")
    
    async def evaluate(self, model: ModelInterface) -> FrameworkResult:
        """
        Evaluate model robustness using PromptRobust dataset.
        
        Args:
            model: The model interface to evaluate
            
        Returns:
            FrameworkResult containing robustness metrics
        """
        logger.info(f"Starting PromptRobust evaluation with {len(self.dataset)} samples")
        
        total_pairs = len(self.dataset)
        consistent_responses = 0
        inconsistent_responses = 0
        errors = 0
        
        attack_performance = {}
        results = []
        
        # Evaluate each original/attacked prompt pair
        for idx, sample in enumerate(tqdm(self.dataset, desc="Evaluating PromptRobust pairs")):
            try:
                original_prompt = sample.get('original', '')
                attacked_prompt = sample.get('attacked', '')
                attack_type = sample.get('attack_type', 'unknown')
                
                # Generate responses for both prompts
                original_response = await model.generate(
                    original_prompt,
                    max_tokens=150,
                    temperature=0.1  # Low temperature for consistency
                )
                
                attacked_response = await model.generate(
                    attacked_prompt,
                    max_tokens=150,
                    temperature=0.1
                )
                
                # Analyze response consistency
                is_consistent = self._are_responses_consistent(
                    original_response, attacked_response, original_prompt
                )
                
                if is_consistent:
                    consistent_responses += 1
                else:
                    inconsistent_responses += 1
                
                # Track performance by attack type
                if attack_type not in attack_performance:
                    attack_performance[attack_type] = {"consistent": 0, "total": 0}
                attack_performance[attack_type]["total"] += 1
                if is_consistent:
                    attack_performance[attack_type]["consistent"] += 1
                
                results.append({
                    "pair_id": sample.get('id', f'pair_{idx}'),
                    "attack_type": attack_type,
                    "original_prompt": original_prompt[:100] + "..." if len(original_prompt) > 100 else original_prompt,
                    "attacked_prompt": attacked_prompt[:100] + "..." if len(attacked_prompt) > 100 else attacked_prompt,
                    "original_response": original_response[:150] + "..." if len(original_response) > 150 else original_response,
                    "attacked_response": attacked_response[:150] + "..." if len(attacked_response) > 150 else attacked_response,
                    "is_consistent": is_consistent,
                    "robustness_score": 1.0 if is_consistent else 0.0
                })
                
            except Exception as e:
                logger.error(f"Error evaluating sample {idx}: {e}")
                errors += 1
        
        # Calculate metrics
        robustness_score = consistent_responses / total_pairs if total_pairs > 0 else 0.0
        inconsistency_rate = inconsistent_responses / total_pairs if total_pairs > 0 else 0.0
        error_rate = errors / total_pairs if total_pairs > 0 else 0.0
        
        # Calculate attack-specific robustness
        attack_robustness = {}
        for attack_type, performance in attack_performance.items():
            if performance["total"] > 0:
                attack_robustness[f"{attack_type}_robustness"] = performance["consistent"] / performance["total"]
        
        metrics = {
            "overall_robustness_score": robustness_score,
            "inconsistency_rate": inconsistency_rate,
            "consistency_rate": robustness_score,
            "error_rate": error_rate,
            "total_pairs": total_pairs,
            "consistent_responses": consistent_responses,
            "inconsistent_responses": inconsistent_responses,
            "errors": errors,
            **attack_robustness
        }
        
        metadata = {
            "attack_types_tested": list(attack_performance.keys()),
            "attack_breakdown": attack_performance,
            "sample_results": results[:10]  # Store first 10 for inspection
        }
        
        logger.info(f"PromptRobust evaluation completed: {robustness_score:.3f} robustness score, {inconsistency_rate:.3f} inconsistency rate")
        
        return FrameworkResult(
            framework_name="promptrobust_hf",
            metrics=metrics,
            metadata=metadata,
            success=True
        )
    
    def _are_responses_consistent(self, response1: str, response2: str, original_prompt: str) -> bool:
        """
        Determine if two responses are semantically consistent.
        
        This is a simplified heuristic. In practice, you might want to use
        more sophisticated similarity measures or the official PromptBench evaluation.
        """
        # Normalize responses
        resp1_normalized = response1.lower().strip()
        resp2_normalized = response2.lower().strip()
        
        # For very short responses, check exact match
        if len(resp1_normalized) < 10 and len(resp2_normalized) < 10:
            return resp1_normalized == resp2_normalized
        
        # Check for similar core content
        resp1_words = set(resp1_normalized.split())
        resp2_words = set(resp2_normalized.split())
        
        # Calculate Jaccard similarity
        intersection = len(resp1_words.intersection(resp2_words))
        union = len(resp1_words.union(resp2_words))
        
        if union == 0:
            return True  # Both empty
        
        jaccard_similarity = intersection / union
        
        # Threshold for consistency (can be tuned)
        consistency_threshold = 0.3
        
        # Special handling for different question types
        if any(keyword in original_prompt.lower() for keyword in ['true', 'false', 'yes', 'no']):
            # For boolean questions, check for same answer
            bool_words_1 = {'true', 'false', 'yes', 'no', 'correct', 'incorrect'}
            bool_words_2 = {'true', 'false', 'yes', 'no', 'correct', 'incorrect'}
            
            found_1 = bool_words_1.intersection(resp1_words)
            found_2 = bool_words_2.intersection(resp2_words)
            
            if found_1 and found_2:
                # Check if they give the same type of answer
                positive_1 = any(word in found_1 for word in ['true', 'yes', 'correct'])
                positive_2 = any(word in found_2 for word in ['true', 'yes', 'correct'])
                return positive_1 == positive_2
        
        return jaccard_similarity >= consistency_threshold
    
    def get_available_attack_types(self) -> List[str]:
        """Get list of available attack types in the dataset."""
        if hasattr(self.dataset, 'unique'):
            return list(self.dataset.unique('attack_type'))
        else:
            return list(set(sample.get('attack_type', 'unknown') for sample in self.dataset))
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about the loaded dataset."""
        return {
            "total_samples": len(self.dataset),
            "attack_types": self.get_available_attack_types(),
            "sample_size_used": self.sample_size,
            "attack_types_filter": self.attack_types
        }