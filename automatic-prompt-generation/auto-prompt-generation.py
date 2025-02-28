import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any
import os

from transformers import pipeline

# import traceback
# import json
# import requests

# class LLMClientBase:
#     """
#     Base class for LLM API clients.
#     Subclass this to implement connections to different LLM providers.
#     """
#     def generate_text(self, prompt: str, max_tokens: int = 500) -> str:
#         """
#         Generate text from the LLM.
        
#         Args:
#             prompt: The input prompt
#             max_tokens: Maximum tokens to generate
            
#         Returns:
#             Generated text as a string
#         """
#         raise NotImplementedError("Subclasses must implement this method")


# class DummyLLMClient(LLMClientBase):
#     """
#     Dummy LLM client that returns a simple response.
#     Used when no LLM service is available.
#     """
#     def generate_text(self, prompt: str, max_tokens: int = 500) -> str:
#         return "Column in the dataset"


# class OpenAIClient(LLMClientBase):
#     """
#     Client for OpenAI API.
#     """
#     def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
#         self.api_key = api_key
#         self.model = model
#         self.api_url = "https://api.openai.com/v1/chat/completions"
        
#     def generate_text(self, prompt: str, max_tokens: int = 500) -> str:
#         headers = {
#             "Content-Type": "application/json",
#             "Authorization": f"Bearer {self.api_key}"
#         }
        
#         data = {
#             "model": self.model,
#             "messages": [{"role": "user", "content": prompt}],
#             "max_tokens": max_tokens
#         }
        
#         try:
#             response = requests.post(self.api_url, headers=headers, json=data)
#             response.raise_for_status()
#             return response.json()["choices"][0]["message"]["content"]
#         except Exception as e:
#             print(f"Error calling OpenAI API: {e}")
#             return "Column in the dataset"


# class AnthropicClient(LLMClientBase):
#     """
#     Client for Anthropic Claude API.
#     """
#     def __init__(self, api_key: str, model: str = "claude-instant-1"):
#         self.api_key = api_key
#         self.model = model
#         self.api_url = "https://api.anthropic.com/v1/messages"
        
#     def generate_text(self, prompt: str, max_tokens: int = 500) -> str:
#         headers = {
#             "Content-Type": "application/json",
#             "X-API-Key": self.api_key,
#             "anthropic-version": "2023-06-01"
#         }
        
#         data = {
#             "model": self.model,
#             "messages": [{"role": "user", "content": prompt}],
#             "max_tokens": max_tokens
#         }
        
#         try:
#             response = requests.post(self.api_url, headers=headers, json=data)
#             response.raise_for_status()
#             return response.json()["content"][0]["text"]
#         except Exception as e:
#             print(f"Error calling Anthropic API: {e}")
#             return "Column in the dataset"


# class HuggingFaceClient(LLMClientBase):
#     """
#     Client for Hugging Face Inference API.
#     """
#     def __init__(self, api_key: str, model: str = "google/flan-t5-xl"):
#         self.api_key = api_key
#         self.model = model
#         self.api_url = f"https://api-inference.huggingface.co/models/{model}"
        
#     def generate_text(self, prompt: str, max_tokens: int = 500) -> str:
#         headers = {
#             "Authorization": f"Bearer {self.api_key}"
#         }
        
#         data = {
#             "inputs": prompt,
#             "parameters": {
#                 "max_new_tokens": max_tokens,
#                 "return_full_text": False
#             }
#         }
        
#         try:
#             response = requests.post(self.api_url, headers=headers, json=data)
#             response.raise_for_status()
#             return response.json()[0]["generated_text"]
#         except Exception as e:
#             print(f"Error calling Hugging Face API: {e}")
#             return "Column in the dataset"


# class OllamaClient(LLMClientBase):
#     """
#     Client for Ollama local models.
#     Requires Ollama to be running locally.
#     """
#     def __init__(self, model: str = "llama2", api_url: str = "http://localhost:11434/api/generate"):
#         self.model = model
#         self.api_url = api_url
        
#     def generate_text(self, prompt: str, max_tokens: int = 500) -> str:
#         data = {
#             "model": self.model,
#             "prompt": prompt,
#             "max_tokens": max_tokens
#         }
        
#         try:
#             response = requests.post(self.api_url, json=data)
#             response.raise_for_status()
            
#             # Ollama streams responses, aggregate them
#             full_response = ""
#             for line in response.text.strip().split("\n"):
#                 if line:
#                     try:
#                         resp_json = json.loads(line)
#                         full_response += resp_json.get("response", "")
#                     except:
#                         continue
            
#             return full_response
#         except Exception as e:
#             print(f"Error calling Ollama API: {e}")
#             return "Column in the dataset"

class SyntheticDataPromptGenerator:
    """
    Automated generator for creating prompts for synthetic data generation
    from any tabular dataset.
    """
    
    def __init__(self, csv_path: str, target_column: str = None, id_column: str = None, fill_missing: str = None):
        """
        Initialize the prompt generator with a dataset.
        
        Args:
            csv_path: Path to the CSV file
            target_column: Name of the target/class column (if any)
            id_column: Name of the ID column (if any)
        """
        self.df = pd.read_csv(csv_path)
        self.file_name = os.path.basename(csv_path).split('.')[0]
        
        # Handle missing values first
        self._handle_missing_values()
        
        # Identify or set target column
        self.target_column = target_column
        if target_column is None and len(self.df.select_dtypes(include=['object', 'category']).columns) > 0:
            # Try to automatically identify a target column (categorical with few unique values)
            potential_targets = []
            for col in self.df.select_dtypes(include=['object', 'category']).columns:
                if self.df[col].nunique() <= 10:  # Assuming classification targets have few classes
                    potential_targets.append((col, self.df[col].nunique()))
            
            if potential_targets:
                # Select the categorical column with the fewest unique values
                self.target_column = sorted(potential_targets, key=lambda x: x[1])[0][0]
                print(f"Automatically selected target column: {self.target_column}")
        
        # Identify or set ID column
        self.id_column = id_column
        if id_column is None:
            # Try to automatically identify an ID column
            for col in self.df.columns:
                if col.lower() in ['id', 'idx', 'index', f'{self.file_name}_id']:
                    if self.df[col].nunique() == len(self.df):  # If each value is unique
                        self.id_column = col
                        print(f"Automatically selected ID column: {self.id_column}")
                        break
        
        # Identify numerical columns (excluding ID and target)
        self.numerical_columns = []
        for col in self.df.select_dtypes(include=np.number).columns:
            if (self.id_column is None or col != self.id_column) and (self.target_column is None or col != self.target_column):
                self.numerical_columns.append(col)
        
        # Generate column definitions (will be populated later)
        self.column_definitions = {}
    
    def _handle_missing_values(self):
        """Fill missing values in the DataFrame."""
        # Handle numerical columns
        numerical_cols = self.df.select_dtypes(include=np.number).columns
        for col in numerical_cols:
            if self.df[col].isna().any():
                # Check if all values are missing
                if self.df[col].count() == 0:
                    # Fill with 0 or another placeholder
                    self.df[col].fillna(0, inplace=True)
                else:
                    # Fill with median
                    median_val = self.df[col].median()
                    self.df[col].fillna(median_val, inplace=True)
        
        # Handle categorical columns
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if self.df[col].isna().any():
                # Check if all values are missing
                if self.df[col].count() == 0:
                    self.df[col].fillna('Unknown', inplace=True)
                else:
                    # Fill with mode
                    mode_val = self.df[col].mode()[0]
                    self.df[col].fillna(mode_val, inplace=True)

        
    # def generate_column_definitions(self) -> Dict[str, str]:
    #     """
    #     Generate basic definitions for each column based on column names.
    #     Can be manually refined later.
    #     """
    #     definitions = {}
        
    #     # Generate basic definition for ID column
    #     if self.id_column:
    #         definitions[self.id_column] = f"Unique identifier for each record"
        
    #     # Generate basic definition for target column
    #     if self.target_column:
    #         unique_values = self.df[self.target_column].unique()
    #         class_distribution = {}
    #         for val in unique_values:
    #             # Handle both categorical and numeric target columns
    #             if pd.isna(val):
    #                 continue  # Skip NaN values
                
    #             percentage = round(100 * self.df[self.target_column].value_counts(normalize=True)[val], 1)
    #             class_distribution[val] = percentage
            
    #         class_desc = ", ".join([f"'{val}' ({perc}% of cases)" for val, perc in class_distribution.items()])
    #         definitions[self.target_column] = f"Classification with {class_desc}"
        
    #     # Generate simple definitions for numerical columns based on column name
    #     for col in self.numerical_columns:
    #         clean_name = col.replace('_', ' ').lower()
    #         if 'mean' in clean_name:
    #             base_name = clean_name.replace('mean', '').strip()
    #             definitions[col] = f"Mean {base_name} value"
    #         elif 'std' in clean_name or 'se' in clean_name:
    #             base_name = clean_name.replace('std', '').replace('se', '').strip()
    #             definitions[col] = f"Standard error/deviation of {base_name}"
    #         elif 'max' in clean_name or 'min' in clean_name:
    #             base_name = clean_name.replace('max', '').replace('min', '').strip()
    #             definitions[col] = f"{'Maximum' if 'max' in clean_name else 'Minimum'} {base_name} value"
    #         else:
    #             definitions[col] = f"Measurement of {clean_name}"
        
    #     self.column_definitions = definitions
    #     return definitions
    

    def generate_column_definitions(self, use_llm=True, model_name="distilgpt2"):
        """
        Generate column definitions with optional LLM enhancement.
        Optimized for GPU acceleration with robust memory management.
        
        Args:
            use_llm: Whether to use LLM for enhanced definitions
            model_name: Name of the model to use
        """
        import gc
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        
        # Always generate basic definitions as fallback
        basic_definitions = self._generate_basic_column_definitions()
        
        # If LLM not requested, return basic definitions
        if not use_llm or not model_name:
            print("Using basic column definitions (no LLM requested)")
            self.column_definitions = basic_definitions
            return basic_definitions
    
        # Get device with memory management
        def clean_gpu_memory():
            """Clean GPU memory and cache"""
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    gc.collect()
            except Exception as e:
                print(f"Warning: Error cleaning GPU memory: {e}")
                gc.collect()
        
        def get_device():
            """Get appropriate device with error handling"""
            if not torch.cuda.is_available():
                print("CUDA not available, using CPU")
                return "cpu"
                
            try:
                # Set environment variable for memory allocation
                import os
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
                
                # Try to find best GPU
                best_device = 0
                
                # If multiple GPUs available, find one with most memory
                if torch.cuda.device_count() > 1:
                    max_free_memory = 0
                    for i in range(torch.cuda.device_count()):
                        try:
                            free_memory = torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)
                            if free_memory > max_free_memory:
                                max_free_memory = free_memory
                                best_device = i
                        except Exception as e:
                            print(f"Warning: Error checking GPU {i}: {e}")
                            
                    print(f"Selected GPU {best_device} with {max_free_memory / 1024**2:.2f} MB free")
                
                return f"cuda:{best_device}"
            except Exception as e:
                print(f"Error selecting GPU: {e}. Falling back to CPU")
                return "cpu"
    
        # Get appropriate device
        device = get_device()
        print(f"Using device: {device}")
        
        # Enhanced definitions dictionary (start with basic definitions)
        enhanced_definitions = dict(basic_definitions)
        
        try:
            print(f"Loading model {model_name}")
            clean_gpu_memory()
            
            # Try different loading strategies with progressive fallbacks
            try:
                # Try loading with 4-bit quantization first (for memory efficiency)
                print("Attempting to load with 4-bit quantization")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16
                )
                
                # Load tokenizer
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                # Load model with quantization
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    torch_dtype=torch.float16,
                    device_map={"": torch.cuda.current_device()} if "cuda" in device else "cpu"
                )
                
                print("Successfully loaded model with 4-bit quantization")
                
            except Exception as e:
                print(f"Error loading with 4-bit quantization: {e}")
                clean_gpu_memory()
                
                try:
                    # Try 8-bit quantization as fallback
                    print("Falling back to 8-bit quantization")
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_threshold=6.0
                    )
                    
                    # Load tokenizer
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    if tokenizer.pad_token is None:
                        tokenizer.pad_token = tokenizer.eos_token
                    
                    # Load model
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        quantization_config=quantization_config,
                        torch_dtype=torch.float16,
                        device_map={"": torch.cuda.current_device()} if "cuda" in device else "cpu"
                    )
                    
                    print("Successfully loaded model with 8-bit quantization")
                    
                except Exception as e2:
                    print(f"Error loading with 8-bit quantization: {e2}")
                    clean_gpu_memory()
                    
                    try:
                        # Final fallback: Load to CPU
                        print("Falling back to loading on CPU")
                        tokenizer = AutoTokenizer.from_pretrained(model_name)
                        if tokenizer.pad_token is None:
                            tokenizer.pad_token = tokenizer.eos_token
                        
                        model = AutoModelForCausalLM.from_pretrained(
                            model_name, 
                            torch_dtype=torch.float16 if "cuda" in device else torch.float32
                        )
                        
                        # Only move to GPU if available
                        if "cuda" in device:
                            model = model.to(device)
                        
                        print(f"Model loaded on {device} without quantization")
                    except Exception as e3:
                        print(f"Failed to load model: {e3}")
                        print("Falling back to basic definitions")
                        self.column_definitions = basic_definitions
                        return basic_definitions
            
            # Set model to evaluation mode
            model.eval()
            print(f"Model ready for inference")
            
            # Process each column
            columns_to_process = list(self.df.columns)
            print(f"Processing {len(columns_to_process)} columns for enhanced definitions")
            
            for col in columns_to_process:
                # Prepare descriptive context about the dataset
                dataset_name = self.file_name.replace('_', ' ').replace('-', ' ')
                
                # Create a focused, detailed prompt to help the model understand the context
                prompt = f"In a {dataset_name} dataset, what does the column '{col}' likely represent?\n\n"
                
                # Add column metadata
                col_type = str(self.df[col].dtype)
                unique_count = self.df[col].nunique()
                prompt += f"Column type: {col_type}\n"
                prompt += f"Number of unique values: {unique_count}\n"
                
                # Add sample values
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    sample_values = self.df[col].dropna().sample(min(5, len(self.df[col].dropna()))).tolist()
                    # Round numeric values for readability
                    sample_values = [round(val, 4) if isinstance(val, float) else val for val in sample_values]
                else:
                    sample_values = self.df[col].dropna().sample(min(5, self.df[col].dropna().nunique())).tolist()
                
                prompt += f"Sample values: {sample_values}\n"
                
                # Add context about other column names
                other_cols = [c for c in self.df.columns if c != col][:5]  # Limit to 5 other columns
                prompt += f"Other columns in dataset: {', '.join(other_cols)}\n\n"
                
                prompt += f"Definition: "
                
                try:
                    # Clear memory before each operation
                    clean_gpu_memory()
                    
                    # Tokenize prompt
                    inputs = tokenizer(prompt, return_tensors="pt")
                    if "cuda" in device:
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    # Generate using model with memory-efficient settings
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=25,  # Keep responses short to save memory
                            do_sample=True,
                            temperature=0.3,    # Lower temperature for more precise responses
                            top_p=0.95,
                            num_return_sequences=1,
                            pad_token_id=tokenizer.pad_token_id
                        )
                    
                    # Decode output
                    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                    # Extract just the generated part (after the prompt)
                    if prompt in generated_text:
                        definition = generated_text[len(prompt):].strip()
                    else:
                        definition = generated_text[len(generated_text.split("Definition: ")[0]) + len("Definition: "):].strip()
                    
                    # Clean up the definition
                    definition = definition.split('.')[0].strip()  # First sentence only
                    
                    # Only use if definition is meaningful
                    if len(definition) >= 5 and len(definition) <= 100:
                        enhanced_definitions[col] = definition
                        print(f"Enhanced definition for '{col}': {definition}")
                    else:
                        print(f"Generated definition for '{col}' was too short or too long, keeping basic definition")
                
                except Exception as e:
                    print(f"Error generating definition for column '{col}': {e}")
                    # Keep basic definition for this column
            
            # Final cleanup
            clean_gpu_memory()
            
            # Update definitions
            self.column_definitions = enhanced_definitions
            return enhanced_definitions
            
        except Exception as e:
            print(f"Error in column definition enhancement: {e}")
            import traceback
            traceback.print_exc()
            
            # Fall back to basic definitions
            print("Falling back to basic definitions due to error")
            self.column_definitions = basic_definitions
            return basic_definitions
    
    def _generate_basic_column_definitions(self):
        """Fallback basic definitions without LLM"""
        definitions = {}
        for col in self.df.columns:
            if col == self.id_column:
                definitions[col] = "Unique identifier for each record"
            elif col == self.target_column:
                # Logic for target column definitions
                unique_values = self.df[col].dropna().unique()
                class_distribution = {}
                
                for val in unique_values:
                    # Skip NaN values
                    if pd.isna(val):
                        continue
                        
                    # Calculate percentage distribution
                    percentage = round(100 * self.df[col].value_counts(normalize=True, dropna=True).get(val, 0), 1)
                    class_distribution[val] = percentage
                
                # Format class descriptions with appropriate formatting
                class_desc_items = []
                for val, perc in class_distribution.items():
                    # Format value based on its type
                    if isinstance(val, str):
                        formatted_val = f"'{val}'"
                    elif isinstance(val, (int, float)):
                        if isinstance(val, float) and val.is_integer():
                            formatted_val = f"{int(val)}"
                        else:
                            formatted_val = f"{val}"
                    else:
                        formatted_val = f"{val}"
                    
                    class_desc_items.append(f"{formatted_val} ({perc}% of cases)")
                
                class_desc = ", ".join(class_desc_items)
                definitions[col] = f"Classification with {class_desc}"
            else:
                # Handle numerical columns with more specific descriptions
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    clean_name = col.replace('_', ' ').lower()
                    if any(term in clean_name for term in ['mean', 'avg', 'average']):
                        base_name = ' '.join([word for word in clean_name.split() if word not in ['mean', 'avg', 'average']])
                        definitions[col] = f"Mean {base_name} value"
                    elif any(term in clean_name for term in ['std', 'stdev', 'deviation']):
                        base_name = ' '.join([word for word in clean_name.split() if word not in ['std', 'stdev', 'deviation']])
                        definitions[col] = f"Standard deviation of {base_name}"
                    elif any(term in clean_name for term in ['max', 'maximum']):
                        base_name = ' '.join([word for word in clean_name.split() if word not in ['max', 'maximum']])
                        definitions[col] = f"Maximum {base_name} value"
                    elif any(term in clean_name for term in ['min', 'minimum']):
                        base_name = ' '.join([word for word in clean_name.split() if word not in ['min', 'minimum']])
                        definitions[col] = f"Minimum {base_name} value"
                    elif any(term in clean_name for term in ['ratio', 'proportion']):
                        base_name = ' '.join([word for word in clean_name.split() if word not in ['ratio', 'proportion']])
                        definitions[col] = f"Ratio or proportion of {base_name}"
                    elif any(term in clean_name for term in ['count', 'number', 'num']):
                        base_name = ' '.join([word for word in clean_name.split() if word not in ['count', 'number', 'num']])
                        definitions[col] = f"Count or number of {base_name}"
                    else:
                        definitions[col] = f"Numerical measurement of {clean_name}"
                elif pd.api.types.is_object_dtype(self.df[col]) or pd.api.types.is_categorical_dtype(self.df[col]):
                    # Handle categorical columns
                    value_count = self.df[col].nunique()
                    if value_count <= 10:  # For categorical with few values
                        # List the unique values
                        unique_vals = self.df[col].dropna().unique()
                        if len(unique_vals) <= 5:  # If few enough to list
                            vals_formatted = []
                            for val in unique_vals:
                                if isinstance(val, str):
                                    vals_formatted.append(f"'{val}'")
                                else:
                                    vals_formatted.append(f"{val}")
                            vals_str = ", ".join(vals_formatted)
                            definitions[col] = f"Categorical variable with values: {vals_str}"
                        else:
                            definitions[col] = f"Categorical variable with {value_count} unique values"
                    else:
                        definitions[col] = f"Text or categorical variable"
                else:
                    # Fallback for other data types
                    definitions[col] = f"Feature representing {col.replace('_', ' ')}"
                    
        return definitions

    def _basic_definition_for_column(self, col):
        """
        Generate a basic definition for a single column when LLM fails.
        """
        if col == self.id_column:
            return "Unique identifier for each record"
        elif col == self.target_column:
            return "Target variable for classification or prediction"
        elif pd.api.types.is_numeric_dtype(self.df[col]):
            clean_name = col.replace('_', ' ').lower()
            return f"Numerical measurement of {clean_name}"
        elif pd.api.types.is_object_dtype(self.df[col]) or pd.api.types.is_categorical_dtype(self.df[col]):
            return f"Categorical variable representing {col.replace('_', ' ').lower()}"
        else:
            return f"Feature representing {col.replace('_', ' ').lower()}"
    
    def calculate_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate comprehensive statistics for each numerical column.
        """
        stats = {}
        
        for col in self.numerical_columns:
                        # Filter out NaN values for calculations
            col_data = self.df[col].dropna()
            
            if len(col_data) == 0:
                print(f"Warning: Column '{col}' has no valid numeric data")
                continue
            
            col_stats = {
                'mean': round(self.df[col].mean(), 4),
                'median': round(self.df[col].median(), 4),
                'min': round(self.df[col].min(), 4),
                'max': round(self.df[col].max(), 4),
                'std': round(self.df[col].std(), 4)
            }
            
            # Calculate class-specific statistics if target column exists
            if self.target_column:
                for class_val in self.df[self.target_column].dropna().unique():
                    # Filter data for this class, excluding NaNs in both target and current column
                    class_data = self.df[(self.df[self.target_column] == class_val) & (~self.df[col].isna())][col]
                    
                    if len(class_data) > 0:
                        col_stats[f'{class_val}_mean'] = round(class_data.mean(), 4)
                    else:
                        col_stats[f'{class_val}_mean'] = "N/A"
            
            stats[col] = col_stats
        
        return stats
    
    def calculate_correlations(self) -> pd.DataFrame:
        """
        Calculate correlation matrix for numerical columns.
        """
        return self.df[self.numerical_columns].corr()
    
    def get_example_records(self, n_examples: int = 3) -> List[Dict[str, Any]]:
        """
        Extract sample records from the dataset to use as examples.
        If target column exists, tries to get examples from different classes.
        """
        examples = []
        
        if self.target_column:
            # Get examples from different classes if possible
            classes = self.df[self.target_column].unique()
            n_per_class = max(1, n_examples // len(classes))
            remaining = n_examples - (n_per_class * len(classes))
            
            for i, class_val in enumerate(classes):
                n_from_class = n_per_class + (1 if i < remaining else 0)
                class_examples = self.df[self.df[self.target_column] == class_val].sample(
                    min(n_from_class, len(self.df[self.df[self.target_column] == class_val]))
                )
                
                for _, row in class_examples.iterrows():
                    example = {}
                    for col in self.df.columns:
                        # Format numbers with appropriate precision
                        if col in self.numerical_columns:
                            if abs(row[col]) < 0.01:
                                example[col] = f"{row[col]:.6f}"
                            elif abs(row[col]) < 1:
                                example[col] = f"{row[col]:.5f}"
                            elif abs(row[col]) < 10:
                                example[col] = f"{row[col]:.4f}"
                            elif abs(row[col]) < 100:
                                example[col] = f"{row[col]:.3f}"
                            elif abs(row[col]) < 1000:
                                example[col] = f"{row[col]:.2f}"
                            else:
                                example[col] = f"{row[col]:.1f}"
                            
                            # Remove trailing zeros
                            if '.' in example[col]:
                                example[col] = example[col].rstrip('0').rstrip('.')
                        else:
                            example[col] = row[col]
                    
                    examples.append(example)
        else:
            # If no target column, just sample random records
            sample_rows = self.df.sample(min(n_examples, len(self.df)))
            for _, row in sample_rows.iterrows():
                example = {}
                for col in self.df.columns:
                    # Format numbers with appropriate precision
                    if col in self.numerical_columns:
                        if abs(row[col]) < 0.01:
                            example[col] = f"{row[col]:.6f}"
                        elif abs(row[col]) < 1:
                            example[col] = f"{row[col]:.5f}"
                        elif abs(row[col]) < 10:
                            example[col] = f"{row[col]:.4f}"
                        elif abs(row[col]) < 100:
                            example[col] = f"{row[col]:.3f}"
                        elif abs(row[col]) < 1000:
                            example[col] = f"{row[col]:.2f}"
                        else:
                            example[col] = f"{row[col]:.1f}"
                        
                        # Remove trailing zeros
                        if '.' in example[col]:
                            example[col] = example[col].rstrip('0').rstrip('.')
                    else:
                        example[col] = row[col]
                
                examples.append(example)
        
        return examples
    
    def get_strong_correlations(self, threshold: float = 0.7) -> List[Tuple[str, str, float]]:
        """
        Identify pairs of columns with strong correlations.
        """
        corr_matrix = self.calculate_correlations()
        strong_correlations = []
        
        for i, col1 in enumerate(corr_matrix.columns):
            for j, col2 in enumerate(corr_matrix.columns):
                if i < j:  # Only get each pair once
                    corr_value = corr_matrix.loc[col1, col2]
                    if pd.isna(corr_value):
                        continue  # Skip if correlation is NaN
                    
                    if abs(corr_value) >= threshold:
                        direction = "positive" if corr_value > 0 else "negative"
                        strength = "strong" if abs(corr_value) > 0.8 else "moderate"
                        strong_correlations.append((col1, col2, corr_value, direction, strength))
        
        # Sort by correlation strength (absolute value)
        strong_correlations.sort(key=lambda x: abs(x[2]), reverse=True)
        return strong_correlations
    
    # every line only one feature with its value
    # def format_example(self, example: Dict[str, Any]) -> str:
    #     """
    #     Format a single example record as a string.
    #     """
    #     example_str = ""
        
    #     for col in example:
    #         example_str += f"- {col}: {example[col]}\n"
        
    #     return example_str.strip()
    
    # returns features and values as a comma-sepetated string
    # def format_example(self, example: Dict[str, Any]) -> str:
    #     """Format a single example record as a comma-separated string."""
    #     return ", ".join([f"{col}: {example[col]}" for col in self.df.columns])
    
    # returns only the values as a comma-separated string without the column names
    def format_example(self, example: Dict[str, Any]) -> str:
        """Format a single example record as a comma-separated string with only values."""
        return ", ".join(str(example[col]) for col in self.df.columns)
    
    def generate_basic_prompt(self, n_samples: int = 100) -> str:
        """
        Generate a basic prompt with minimal information and examples.
        """
        examples = self.get_example_records(3)
        
        # Generate dataset description
        if self.target_column:
            target_values = self.df[self.target_column].unique()
            target_dist = {val: round(100 * self.df[self.target_column].value_counts(normalize=True)[val], 1) 
                          for val in target_values}
            target_desc = ", ".join([f"'{val}' ({perc}%)" for val, perc in target_dist.items()])
        else:
            target_desc = "N/A"
        
        # Begin building the prompt
        prompt = f"""I need you to generate synthetic {self.file_name} data that closely resembles real-world data. The dataset should contain {n_samples} samples with the following columns:

"""
        
        # Add column descriptions
        for col in self.df.columns:
            if col in self.column_definitions:
                prompt += f"- {col}: {self.column_definitions[col]}\n"
            else:
                prompt += f"- {col}: A column in the dataset\n"
        
        # Add examples
        prompt += f"\nHere are {len(examples)} example records from a real dataset to guide your generation:\n\n"
        
        for i, example in enumerate(examples):
            prompt += f"Example {i+1}:\n"
            prompt += self.format_example(example) + "\n\n"
        
        # Add final instructions
        prompt += f"Please generate {n_samples} records in a CSV format that follows these patterns and maintains realistic relationships between the features. The data should be plausible and preserve the correlations between features that would be found in real {self.file_name} data."
        
        return prompt
    
    def generate_definition_prompt(self, n_samples: int = 100) -> str:
        """
        Generate a prompt with detailed column definitions and examples.
        """
        examples = self.get_example_records(3)
        stats = self.calculate_statistics()
        
        # Begin building the prompt
        prompt = f"""I need you to generate synthetic {self.file_name} data. Please create {n_samples} samples that realistically represent the patterns and relationships in this type of data.

Each record should include:

"""
        
        # Add column definitions with ranges
        for col in self.df.columns:
            if col in self.column_definitions:
                desc = self.column_definitions[col]
                
                if col in self.numerical_columns:
                    min_val = stats[col]['min']
                    max_val = stats[col]['max']
                    desc += f" (range: ~{min_val}-{max_val})"
                
                prompt += f"- {col}: {desc}\n"
            else:
                prompt += f"- {col}: A column in the dataset\n"
        
        # Add general relationships
        if self.target_column and len(self.numerical_columns) > 0:
            prompt += f"\nThe data should maintain realistic correlations: "
            
            # Find columns with significant differences between classes
            significant_cols = []
            for col in self.numerical_columns:
                class_vals = self.df[self.target_column].unique()
                if len(class_vals) >= 2:  # Need at least two classes
                    class1, class2 = class_vals[0], class_vals[1]
                    class1_mean = stats[col][f'{class1}_mean']
                    class2_mean = stats[col][f'{class2}_mean']
                    
                    # If the difference is substantial (arbitrary threshold of 15%)
                    if abs(class1_mean - class2_mean) / max(abs(class1_mean), abs(class2_mean)) > 0.15:
                        significant_cols.append(col)
            
            if significant_cols:
                prompt += f"{class_vals[0]} typically "
                if len(significant_cols) == 1:
                    prompt += f"has a {'higher' if stats[significant_cols[0]][f'{class_vals[0]}_mean'] > stats[significant_cols[0]][f'{class_vals[1]}_mean'] else 'lower'} value for {significant_cols[0]}"
                else:
                    prompt += "has "
                    comparisons = []
                    for col in significant_cols[:3]:  # Limit to 3 columns for brevity
                        comp = f"{'higher' if stats[col][f'{class_vals[0]}_mean'] > stats[col][f'{class_vals[1]}_mean'] else 'lower'} values for {col}"
                        comparisons.append(comp)
                    
                    prompt += f"{', '.join(comparisons[:-1])} and {comparisons[-1]}" if len(comparisons) > 1 else comparisons[0]
                    
                prompt += f" compared to {class_vals[1]}. "
            
            prompt += "There should be natural variance in the data while maintaining these relationships.\n"
        
        # Add examples
        prompt += f"\nHere are {len(examples)} example records from a real dataset:\n\n"
        
        for i, example in enumerate(examples):
            if self.target_column:
                target_value = example[self.target_column]
                prompt += f"Example {i+1} ({target_value}):\n"
            else:
                prompt += f"Example {i+1}:\n"
            
            prompt += self.format_example(example) + "\n\n"
        
        # Add final instructions
        prompt += f"Please provide {n_samples} synthetic records in CSV format, with values that are plausible and maintain the natural relationships between features."
        
        return prompt
    
    def generate_metadata_prompt(self, n_samples: int = 100) -> str:
        """
        Generate a prompt with detailed definitions, examples, and statistical metadata.
        """
        examples = self.get_example_records(3)
        stats = self.calculate_statistics()
        strong_correlations = self.get_strong_correlations(0.7)
        
        # Begin building the prompt
        prompt = f"""I need you to generate synthetic {self.file_name} data based on real statistical properties. Please generate {n_samples} records that accurately represent the data while maintaining the statistical properties and correlations found in real data.

Each record should include:

"""
        
        # Add basic column descriptions first
        for col in self.df.columns:
            if col not in self.numerical_columns:
                if col in self.column_definitions:
                    prompt += f"- {col}: {self.column_definitions[col]}\n"
                else:
                    prompt += f"- {col}: A column in the dataset\n"
        
        # Create statistical table for numerical columns
        prompt += f"\n{self.file_name.capitalize()} measurements with their definitions and statistical properties:\n\n"
        prompt += "| Feature | Definition | Overall Mean | "
        
        # Add class-specific columns if target exists
        if self.target_column:
            for class_val in sorted(self.df[self.target_column].unique()):
                prompt += f"{class_val} Mean | "
        
        prompt += "Min | Max | Std Dev |\n"
        prompt += "|---------|------------|--------------|"
        
        # Add class-specific header dividers if target exists
        if self.target_column:
            for _ in self.df[self.target_column].unique():
                prompt += "-------------|"
        
        prompt += "-----|-----|---------|\n"
        
        # Add rows for each numerical column
        for col in self.numerical_columns:
            definition = self.column_definitions.get(col, f"Measurement of {col}")
            mean = stats[col]['mean']
            min_val = stats[col]['min']
            max_val = stats[col]['max']
            std_dev = stats[col]['std']
            
            prompt += f"| {col} | {definition} | {mean} | "
            
            # Add class-specific means if target exists
            if self.target_column:
                for class_val in sorted(self.df[self.target_column].unique()):
                    class_mean = stats[col].get(f'{class_val}_mean', "N/A")
                    prompt += f"{class_mean} | "
            
            prompt += f"{min_val} | {max_val} | {std_dev} |\n"
        
        # Add correlation information
        prompt += "\nKey correlations to maintain:\n"
        
        if strong_correlations:
            # Group by strength
            strong = [c for c in strong_correlations if abs(c[2]) > 0.8]
            moderate = [c for c in strong_correlations if 0.7 <= abs(c[2]) <= 0.8]
            
            if strong:
                prompt += f"- Strong {strong[0][3]} correlation"
                if len(strong) == 1:
                    prompt += f" between {strong[0][0]} and {strong[0][1]}\n"
                else:
                    featured_cols = set()
                    for corr in strong[:3]:  # Limit to 3 correlations for brevity
                        featured_cols.add(corr[0])
                        featured_cols.add(corr[1])
                    
                    prompt += f" among {', '.join(list(featured_cols))}\n"
            
            if moderate:
                prompt += f"- Moderate {moderate[0][3]} correlation"
                if len(moderate) == 1:
                    prompt += f" between {moderate[0][0]} and {moderate[0][1]}\n"
                else:
                    featured_cols = set()
                    for corr in moderate[:3]:  # Limit to 3 correlations for brevity
                        featured_cols.add(corr[0])
                        featured_cols.add(corr[1])
                    
                    prompt += f" among {', '.join(list(featured_cols))}\n"
        else:
            prompt += "- No strong correlations were detected in the dataset\n"
        
        # Add class-specific differences if target exists
        if self.target_column:
            prompt += f"- {self.target_column} class differences: "
            
            # Find columns with significant differences between classes
            significant_cols = []
            for col in self.numerical_columns:
                class_vals = sorted(self.df[self.target_column].unique())
                if len(class_vals) >= 2:  # Need at least two classes
                    class1, class2 = class_vals[0], class_vals[1]
                    class1_mean = stats[col][f'{class1}_mean']
                    class2_mean = stats[col][f'{class2}_mean']
                    
                    # If the difference is substantial (arbitrary threshold of 15%)
                    if abs(class1_mean - class2_mean) / max(abs(class1_mean), abs(class2_mean)) > 0.15:
                        significant_cols.append((col, class1_mean, class2_mean))
            
            if significant_cols:
                significant_cols.sort(key=lambda x: abs(x[1] - x[2]) / max(abs(x[1]), abs(x[2])), reverse=True)
                significant_cols = significant_cols[:5]  # Limit to 5 most significant
                
                class_vals = sorted(self.df[self.target_column].unique())
                features_list = ", ".join([col[0] for col in significant_cols])
                prompt += f"{class_vals[0]} and {class_vals[1]} samples show significant differences in {features_list}\n"
            else:
                prompt += "No significant differences detected between classes\n"
        
        # Add examples
        prompt += f"\nHere are {len(examples)} example records from the real dataset:\n\n"
        
        for i, example in enumerate(examples):
            if self.target_column:
                target_value = example[self.target_column]
                prompt += f"Example {i+1} ({target_value}):\n"
            else:
                prompt += f"Example {i+1}:\n"
            
            prompt += self.format_example(example) + "\n\n"
        
        # Add final instructions
        prompt += f"Please provide {n_samples} synthetic records in CSV format, with values that are plausible and maintain both the statistical properties and natural relationships between features."
        
        if self.target_column:
            prompt += f" Ensure the data could be useful for machine learning algorithms to differentiate between different {self.target_column} values."
        
        return prompt
    
    def generate_no_examples_prompt(self, n_samples: int = 100) -> str:
        """
        Generate a prompt with metadata but no real examples (for privacy).
        """
        stats = self.calculate_statistics()
        strong_correlations = self.get_strong_correlations(0.7)
        
        # Begin building the prompt
        prompt = f"""I need you to generate synthetic {self.file_name} data based exclusively on statistical properties without using any real examples. Please create {n_samples} synthetic records that represent {self.file_name} measurements while preserving the statistical distributions and correlations in the real-world data.

Each record should include:

"""
        
        # Add basic column descriptions first
        for col in self.df.columns:
            if col not in self.numerical_columns:
                if col in self.column_definitions:
                    prompt += f"- {col}: {self.column_definitions[col]}\n"
                else:
                    prompt += f"- {col}: A column in the dataset\n"
        
        # Create statistical table for numerical columns
        prompt += f"\nThe data should contain {self.file_name} measurements with their statistical properties:\n\n"
        prompt += "| Feature | Definition | Overall Mean | "
        
        # Add class-specific columns if target exists
        if self.target_column:
            for class_val in sorted(self.df[self.target_column].unique()):
                prompt += f"{class_val} Mean | "
        
        prompt += "Min | Max | Std Dev |\n"
        prompt += "|---------|------------|--------------|"
        
        # Add class-specific header dividers if target exists
        if self.target_column:
            for _ in self.df[self.target_column].unique():
                prompt += "-------------|"
        
        prompt += "-----|-----|---------|\n"
        
        # Add rows for each numerical column
        for col in self.numerical_columns:
            definition = self.column_definitions.get(col, f"Measurement of {col}")
            mean = stats[col]['mean']
            min_val = stats[col]['min']
            max_val = stats[col]['max']
            std_dev = stats[col]['std']
            
            prompt += f"| {col} | {definition} | {mean} | "
            
            # Add class-specific means if target exists
            if self.target_column:
                for class_val in sorted(self.df[self.target_column].unique()):
                    class_mean = stats[col].get(f'{class_val}_mean', "N/A")
                    prompt += f"{class_mean} | "
            
            prompt += f"{min_val} | {max_val} | {std_dev} |\n"
        
        # Add important statistical relationships
        prompt += "\nImportant statistical relationships to maintain:\n\n"
        
        prompt += "1. Distribution pattern: Generate values following the statistical distributions in the table above, respecting min/max values and standard deviations\n"
        
        if self.target_column:
            prompt += f"2. Class-specific distributions: Maintain the statistical differences between different {self.target_column} values\n"
        else:
            prompt += "2. Overall distribution: Maintain the overall distribution of each variable\n"
        
        # Feature correlations
        prompt += "3. Feature correlations:\n"
        
        if strong_correlations:
            # Group by strength and type
            strong_positive = [(c[0], c[1]) for c in strong_correlations if c[2] > 0.8]
            strong_negative = [(c[0], c[1]) for c in strong_correlations if c[2] < -0.8]
            moderate_positive = [(c[0], c[1]) for c in strong_correlations if 0.7 <= c[2] <= 0.8]
            moderate_negative = [(c[0], c[1]) for c in strong_correlations if -0.8 <= c[2] <= -0.7]
            
            if strong_positive:
                prompt += f"   - Strong positive correlation (r > 0.8) between"
                for i, (col1, col2) in enumerate(strong_positive[:3]):  # Limit to 3 for brevity
                    if i == 0:
                        prompt += f" {col1} and {col2}"
                    else:
                        prompt += f", also between {col1} and {col2}"
                prompt += "\n"
            
            if strong_negative:
                prompt += f"   - Strong negative correlation (r < -0.8) between"
                for i, (col1, col2) in enumerate(strong_negative[:3]):
                    if i == 0:
                        prompt += f" {col1} and {col2}"
                    else:
                        prompt += f", also between {col1} and {col2}"
                prompt += "\n"
            
            if moderate_positive:
                prompt += f"   - Moderate positive correlation (r ~ 0.7) between"
                for i, (col1, col2) in enumerate(moderate_positive[:3]):
                    if i == 0:
                        prompt += f" {col1} and {col2}"
                    else:
                        prompt += f", also between {col1} and {col2}"
                prompt += "\n"
            
            if moderate_negative:
                prompt += f"   - Moderate negative correlation (r ~ -0.7) between"
                for i, (col1, col2) in enumerate(moderate_negative[:3]):
                    if i == 0:
                        prompt += f" {col1} and {col2}"
                    else:
                        prompt += f", also between {col1} and {col2}"
                prompt += "\n"
        else:
            prompt += "   - No strong correlations were detected in the dataset\n"
        
        # Add logical constraints where possible
        prompt += "\n4. Logical constraints:\n"
        # Find potential mathematical relationships (if any)
        for i, col1 in enumerate(self.numerical_columns):
            for j, col2 in enumerate(self.numerical_columns):
                if i < j:  # Check each pair once
                    # Look for area and radius/perimeter relationships
                    if ('area' in col1.lower() and 'radius' in col2.lower()) or \
                       ('area' in col2.lower() and 'radius' in col1.lower()):
                        prompt += f"   - If present, area should be proportional to radius squared (approximately πr²)\n"
                        break
                    # Look for perimeter and radius relationships
                    elif ('perimeter' in col1.lower() and 'radius' in col2.lower()) or \
                         ('perimeter' in col2.lower() and 'radius' in col1.lower()):
                        prompt += f"   - If present, perimeter should be proportional to radius (approximately 2πr)\n"
                        break
        
        # Add general constraints
        prompt += "   - All numerical values should be positive\n"
        prompt += f"   - Values should respect the min/max ranges in the table above\n"
        
        # Add final instructions
        prompt += f"\nPlease provide {n_samples} synthetic records in CSV format that satisfy these statistical properties and mathematical relationships. The synthetic data should be suitable for analysis or training machine learning models while preserving privacy by not containing any actual data points."
        
        return prompt
    
    def generate_all_prompts(self, n_samples: int = 100) -> Dict[str, str]:
        """
        Generate all four prompts.
        """
        # First, generate column definitions if not already done
        if not self.column_definitions:
            self.generate_column_definitions()
        
        # Generate all prompts
        prompts = {
            "task1_basic_prompt": self.generate_basic_prompt(n_samples),
            "task2_definition_prompt": self.generate_definition_prompt(n_samples),
            "task3_metadata_prompt": self.generate_metadata_prompt(n_samples),
            "task4_no_examples_prompt": self.generate_no_examples_prompt(n_samples)
        }
        
        return prompts
    
    def save_prompts_to_files(self, n_samples: int = 100, output_dir: str = "./prompts"):
        """
        Generate all prompts and save them to text files.
        
        Args:
            n_samples: Number of samples to request in the prompts
            output_dir: Directory to save the prompts in
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate all prompts
        prompts = self.generate_all_prompts(n_samples)
        
        # Save each prompt to a separate file
        for prompt_name, prompt_text in prompts.items():
            filename = f"{output_dir}/{self.file_name}_{prompt_name}.txt"
            with open(filename, 'w') as f:
                f.write(prompt_text)
            
            print(f"Saved {prompt_name} to {filename}")
        
        return [f"{output_dir}/{self.file_name}_{prompt_name}.txt" for prompt_name in prompts.keys()]


def main():
    """
    Example usage of the SyntheticDataPromptGenerator class with GPU support.
    """
    import argparse
    import gc
    import os
    import torch
    
    # Add this to avoid memory fragmentation
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    # Add explicit device selection for MIG environment
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    
    parser = argparse.ArgumentParser(description='Generate synthetic data prompts from a CSV file')
    parser.add_argument('csv_path', type=str, help='Path to the CSV file')
    parser.add_argument('--target', type=str, default=None, help='Name of the target/class column')
    parser.add_argument('--id', type=str, default=None, help='Name of the ID column')
    parser.add_argument('--n_samples', type=int, default=100, help='Number of samples to request')
    parser.add_argument('--output_dir', type=str, default='./prompts', help='Directory to save prompts')
    parser.add_argument('--model', type=str, default=None, help='Hugging Face model to use for definitions')
    parser.add_argument('--fill_missing', type=str, default=None, help='Strategy for missing values (mean, median, mode)')
    parser.add_argument('--cpu_only', action='store_true', help='Force CPU-only mode (no CUDA)')
    args = parser.parse_args()
    
    # Print hardware information
    if torch.cuda.is_available() and not args.cpu_only:
        print("CUDA is available with the following devices:")
        for i in range(torch.cuda.device_count()):
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
            free_memory = torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)
            print(f"  Memory available: {free_memory / 1024**3:.2f} GB")
    else:
        print("CUDA is not available or --cpu_only was specified. Using CPU only.")
    
    try:
        # Create the generator with clear parameter names to avoid confusion
        generator = SyntheticDataPromptGenerator(
            csv_path=args.csv_path,
            target_column=args.target,
            id_column=args.id,
            fill_missing=args.fill_missing
        )
        
        # Generate column definitions (with or without LLM)
        print("Generating column definitions...")
        if args.model:
            print(f"Using model {args.model} for enhanced definitions")
            
            # Force CPU if requested
            if args.cpu_only:
                # This requires modifying the generate_column_definitions method
                # to accept a force_cpu parameter
                definitions = generator.generate_column_definitions(
                    use_llm=True,
                    model_name=args.model,
                    force_cpu=True
                )
            else:
                definitions = generator.generate_column_definitions(
                    use_llm=True,
                    model_name=args.model
                )
        else:
            print("Using basic column definitions (no model specified)")
            definitions = generator.generate_column_definitions(use_llm=False)
        
        # Print the generated definitions
        print("\nGenerated column definitions:")
        for col, definition in definitions.items():
            print(f"  {col}: {definition}")
        
        print("\nGenerating prompts...")
        # Save all prompts to files
        generator.save_prompts_to_files(args.n_samples, args.output_dir)
        
        print("\nPrompt generation complete.")
        
        # Perform cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
    
    except Exception as e:
        print(f"Error: {e}")
        print("If this is a parameter error, check that your SyntheticDataPromptGenerator class constructor matches the parameters.")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()