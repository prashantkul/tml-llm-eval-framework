"""Data loading utilities for evaluation frameworks."""

import json
import csv
import yaml
import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)


class DataLoader:
    """Utility class for loading evaluation datasets and test cases."""
    
    def __init__(self):
        self.supported_formats = ['.json', '.csv', '.yaml', '.yml', '.jsonl']
    
    def load_dataset(self, file_path: Union[str, Path], format: Optional[str] = None) -> List[Dict[str, Any]]:
        """Load dataset from file."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        # Auto-detect format if not specified
        if format is None:
            format = file_path.suffix.lower()
        
        if format not in self.supported_formats:
            raise ValueError(f"Unsupported format: {format}. Supported: {self.supported_formats}")
        
        logger.info(f"Loading dataset from {file_path} (format: {format})")
        
        try:
            if format == '.json':
                return self._load_json(file_path)
            elif format == '.jsonl':
                return self._load_jsonl(file_path)
            elif format == '.csv':
                return self._load_csv(file_path)
            elif format in ['.yaml', '.yml']:
                return self._load_yaml(file_path)
            else:
                raise ValueError(f"Format handler not implemented: {format}")
        
        except Exception as e:
            logger.error(f"Failed to load dataset from {file_path}: {e}")
            raise
    
    def _load_json(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load JSON dataset."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            if 'data' in data:
                return data['data']
            elif 'examples' in data:
                return data['examples']
            elif 'samples' in data:
                return data['samples']
            else:
                return [data]  # Single item
        else:
            raise ValueError("JSON must contain list or dict with 'data'/'examples'/'samples' key")
    
    def _load_jsonl(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load JSONL (JSON Lines) dataset."""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping invalid JSON on line {line_num}: {e}")
        return data
    
    def _load_csv(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load CSV dataset."""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(dict(row))
        return data
    
    def _load_yaml(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load YAML dataset."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        # Handle different YAML structures
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            if 'data' in data:
                return data['data']
            elif 'examples' in data:
                return data['examples']
            else:
                return [data]  # Single item
        else:
            raise ValueError("YAML must contain list or dict with 'data'/'examples' key")
    
    def load_multiple_datasets(self, file_paths: List[Union[str, Path]]) -> Dict[str, List[Dict[str, Any]]]:
        """Load multiple datasets."""
        datasets = {}
        
        for file_path in file_paths:
            file_path = Path(file_path)
            dataset_name = file_path.stem
            
            try:
                datasets[dataset_name] = self.load_dataset(file_path)
                logger.info(f"Loaded dataset '{dataset_name}' with {len(datasets[dataset_name])} samples")
            except Exception as e:
                logger.error(f"Failed to load dataset {file_path}: {e}")
                datasets[dataset_name] = []
        
        return datasets
    
    def filter_dataset(self, dataset: List[Dict[str, Any]], 
                      filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Filter dataset based on criteria."""
        filtered_data = []
        
        for item in dataset:
            include_item = True
            
            for key, value in filters.items():
                if key not in item:
                    include_item = False
                    break
                
                item_value = item[key]
                
                # Handle different filter types
                if isinstance(value, list):
                    # Value must be in list
                    if item_value not in value:
                        include_item = False
                        break
                elif isinstance(value, dict):
                    # Range or comparison filters
                    if 'min' in value and item_value < value['min']:
                        include_item = False
                        break
                    if 'max' in value and item_value > value['max']:
                        include_item = False
                        break
                    if 'equals' in value and item_value != value['equals']:
                        include_item = False
                        break
                else:
                    # Direct equality
                    if item_value != value:
                        include_item = False
                        break
            
            if include_item:
                filtered_data.append(item)
        
        logger.info(f"Filtered dataset: {len(filtered_data)}/{len(dataset)} items match criteria")
        return filtered_data
    
    def sample_dataset(self, dataset: List[Dict[str, Any]], 
                      sample_size: int, 
                      random_seed: Optional[int] = None) -> List[Dict[str, Any]]:
        """Sample a subset of the dataset."""
        if sample_size >= len(dataset):
            return dataset.copy()
        
        import random
        if random_seed is not None:
            random.seed(random_seed)
        
        sampled = random.sample(dataset, sample_size)
        logger.info(f"Sampled {len(sampled)} items from dataset of {len(dataset)}")
        return sampled
    
    def validate_dataset_schema(self, dataset: List[Dict[str, Any]], 
                               required_fields: List[str]) -> bool:
        """Validate that dataset contains required fields."""
        if not dataset:
            logger.warning("Dataset is empty")
            return False
        
        missing_fields = []
        for field in required_fields:
            if not any(field in item for item in dataset):
                missing_fields.append(field)
        
        if missing_fields:
            logger.error(f"Dataset missing required fields: {missing_fields}")
            return False
        
        # Check consistency across items
        inconsistent_items = []
        for i, item in enumerate(dataset):
            for field in required_fields:
                if field not in item:
                    inconsistent_items.append(i)
                    break
        
        if inconsistent_items:
            logger.warning(f"Items missing required fields: {len(inconsistent_items)} items")
            return False
        
        logger.info(f"Dataset validation passed for {len(required_fields)} required fields")
        return True
    
    def convert_to_standard_format(self, dataset: List[Dict[str, Any]], 
                                  field_mapping: Dict[str, str]) -> List[Dict[str, Any]]:
        """Convert dataset to standard field names."""
        converted_dataset = []
        
        for item in dataset:
            converted_item = {}
            
            # Apply field mapping
            for standard_field, source_field in field_mapping.items():
                if source_field in item:
                    converted_item[standard_field] = item[source_field]
            
            # Copy unmapped fields
            for key, value in item.items():
                if key not in field_mapping.values() and key not in converted_item:
                    converted_item[key] = value
            
            converted_dataset.append(converted_item)
        
        logger.info(f"Converted dataset with field mapping: {field_mapping}")
        return converted_dataset
    
    def merge_datasets(self, datasets: List[List[Dict[str, Any]]], 
                      add_source_label: bool = True) -> List[Dict[str, Any]]:
        """Merge multiple datasets."""
        merged_dataset = []
        
        for i, dataset in enumerate(datasets):
            for item in dataset:
                merged_item = item.copy()
                if add_source_label:
                    merged_item['source_dataset'] = i
                merged_dataset.append(merged_item)
        
        logger.info(f"Merged {len(datasets)} datasets into {len(merged_dataset)} total items")
        return merged_dataset
    
    def export_dataset(self, dataset: List[Dict[str, Any]], 
                      file_path: Union[str, Path], 
                      format: Optional[str] = None):
        """Export dataset to file."""
        file_path = Path(file_path)
        
        if format is None:
            format = file_path.suffix.lower()
        
        logger.info(f"Exporting {len(dataset)} items to {file_path} (format: {format})")
        
        try:
            if format == '.json':
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(dataset, f, indent=2, ensure_ascii=False)
            
            elif format == '.jsonl':
                with open(file_path, 'w', encoding='utf-8') as f:
                    for item in dataset:
                        f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
            elif format == '.csv':
                if dataset:
                    df = pd.DataFrame(dataset)
                    df.to_csv(file_path, index=False)
                else:
                    # Create empty CSV
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write('')
            
            elif format in ['.yaml', '.yml']:
                with open(file_path, 'w', encoding='utf-8') as f:
                    yaml.dump(dataset, f, default_flow_style=False, allow_unicode=True)
            
            else:
                raise ValueError(f"Unsupported export format: {format}")
        
        except Exception as e:
            logger.error(f"Failed to export dataset to {file_path}: {e}")
            raise
    
    def get_dataset_stats(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get basic statistics about the dataset."""
        if not dataset:
            return {"size": 0, "fields": []}
        
        # Collect all field names
        all_fields = set()
        for item in dataset:
            all_fields.update(item.keys())
        
        # Field statistics
        field_stats = {}
        for field in all_fields:
            values = [item.get(field) for item in dataset if field in item]
            field_stats[field] = {
                "count": len(values),
                "coverage": len(values) / len(dataset),
                "types": list(set(type(v).__name__ for v in values if v is not None))
            }
        
        stats = {
            "size": len(dataset),
            "fields": list(all_fields),
            "field_count": len(all_fields),
            "field_stats": field_stats
        }
        
        return stats