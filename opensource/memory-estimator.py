import math
from typing import Dict, Tuple

import GPUtil
import psutil
import torch
from transformers import AutoConfig


class ModelRequirementsEstimator:
    @staticmethod
    def estimate_model_size(model_name: str, batch_size: int = 1) -> Dict[str, float]:
        """
        Estimate memory requirements for a model before loading it
        Returns sizes in GB
        """
        try:
            config = AutoConfig.from_pretrained(model_name)

            # Get model architecture details
            n_layers = config.num_hidden_layers
            n_heads = config.num_attention_heads
            hidden_size = config.hidden_size
            vocab_size = config.vocab_size
            seq_length = config.max_position_embeddings

            # Calculate parameters
            attention_params = 4 * n_layers * hidden_size * hidden_size
            mlp_params = 8 * n_layers * hidden_size * hidden_size
            embedding_params = vocab_size * hidden_size

            total_params = attention_params + mlp_params + embedding_params

            # Estimate memory requirements
            model_size_fp32 = total_params * 4 / (1024**3)  # Size in GB
            model_size_fp16 = model_size_fp32 / 2
            model_size_4bit = model_size_fp32 / 8

            # Estimate runtime memory (including activations)
            activation_memory = (
                batch_size * seq_length * hidden_size * 4 * n_layers / (1024**3)
            )

            # Estimate attention memory
            attention_memory = (
                batch_size * n_heads * seq_length * seq_length * 4 / (1024**3)
            )

            # Total runtime memory estimation
            runtime_memory = {
                "fp32": model_size_fp32 + activation_memory + attention_memory,
                "fp16": model_size_fp16 + (activation_memory + attention_memory) / 2,
                "4bit": model_size_4bit + (activation_memory + attention_memory) / 4,
            }

            return {
                "model_size_fp32": model_size_fp32,
                "model_size_fp16": model_size_fp16,
                "model_size_4bit": model_size_4bit,
                "activation_memory": activation_memory,
                "attention_memory": attention_memory,
                "total_runtime_memory": runtime_memory,
                "recommended_gpu_memory": runtime_memory["4bit"] * 1.5,  # 50% buffer
            }
        except Exception as e:
            raise RuntimeError(f"Error estimating model size: {str(e)}")

    @staticmethod
    def check_system_requirements(
        required_memory: float,
    ) -> Tuple[bool, Dict[str, any]]:
        """
        Check if system meets the requirements
        required_memory in GB
        """
        system_info = {
            "cpu_memory": psutil.virtual_memory().total / (1024**3),  # GB
            "gpu_info": [],
            "meets_requirements": False,
            "recommendations": [],
        }

        # Check GPU availability and memory
        if torch.cuda.is_available():
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                system_info["gpu_info"].append(
                    {
                        "id": gpu.id,
                        "name": gpu.name,
                        "memory_total": gpu.memoryTotal / 1024,  # GB
                        "memory_free": gpu.memoryFree / 1024,  # GB
                    }
                )

            # Check if any GPU has enough memory
            max_free_memory = max(gpu["memory_free"] for gpu in system_info["gpu_info"])
            system_info["meets_requirements"] = max_free_memory >= required_memory

            if not system_info["meets_requirements"]:
                system_info["recommendations"].append(
                    f"Need {required_memory:.2f}GB GPU memory, but only {max_free_memory:.2f}GB available. "
                    "Consider using a smaller model or increasing GPU memory."
                )
        else:
            system_info["recommendations"].append(
                "No GPU found. This model requires GPU acceleration."
            )

        return system_info["meets_requirements"], system_info


def estimate_requirements(
    model_name: str, batch_size: int = 1, print_report: bool = True
) -> Tuple[bool, Dict[str, any]]:
    """
    Main function to estimate requirements and check system compatibility
    """
    estimator = ModelRequirementsEstimator()

    # Get memory requirements
    memory_requirements = estimator.estimate_model_size(model_name, batch_size)

    # Check system capabilities
    meets_req, system_info = estimator.check_system_requirements(
        memory_requirements["recommended_gpu_memory"]
    )

    if print_report:
        print("\n=== Model Requirements Analysis ===")
        print(f"\nModel: {model_name}")
        print(f"Batch size: {batch_size}")
        print("\nEstimated Memory Requirements (GB):")
        print(f"- FP32: {memory_requirements['model_size_fp32']:.2f}")
        print(f"- FP16: {memory_requirements['model_size_fp16']:.2f}")
        print(f"- 4-bit: {memory_requirements['model_size_4bit']:.2f}")
        print(
            f"\nRuntime Memory (4-bit): {memory_requirements['total_runtime_memory']['4bit']:.2f}"
        )
        print(
            f"Recommended GPU Memory: {memory_requirements['recommended_gpu_memory']:.2f}"
        )

        print("\nSystem Information:")
        print(f"CPU Memory: {system_info['cpu_memory']:.2f}GB")
        if system_info["gpu_info"]:
            for gpu in system_info["gpu_info"]:
                print(f"\nGPU {gpu['id']} ({gpu['name']}):")
                print(f"- Total Memory: {gpu['memory_total']:.2f}GB")
                print(f"- Free Memory: {gpu['memory_free']:.2f}GB")

        print("\nRecommendations:")
        for rec in system_info["recommendations"]:
            print(f"- {rec}")

        print(f"\nSystem meets requirements: {'Yes' if meets_req else 'No'}")

    return meets_req, {
        "memory_requirements": memory_requirements,
        "system_info": system_info,
    }


# Usage example
if __name__ == "__main__":
    # Example usage for Llama-2-7b
    MODEL_NAME = "meta-llama/Llama-2-7b-hf"
    BATCH_SIZE = 5

    meets_requirements, requirements = estimate_requirements(
        model_name=MODEL_NAME, batch_size=BATCH_SIZE, print_report=True
    )
