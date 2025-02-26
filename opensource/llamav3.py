import gc
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List

import GPUtil
import numpy as np
import pandas as pd
import psutil
import torch
from memory_estimator import estimate_requirements
from prompts import PROMPTS
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Check requirements before loading model
meets_req, requirements = estimate_requirements(
    model_name="meta-llama/Llama-2-7b-hf", batch_size=5, print_report=True
)

if not meets_req:
    print("System does not meet requirements. Adjusting parameters...")
    # Here you could reduce batch size, use more quantization, etc.
    sys.exit(1)


# Enhanced Configuration
class Config:
    MODEL_NAME = "meta-llama/Llama-2-7b-hf"
    OUTPUT_DIR = "model_comparison_llama"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MAX_NEW_TOKENS = 200
    SAMPLES_PER_PROMPT = 100
    BATCH_SIZE = 5
    MAX_GPU_MEMORY = 0.9  # Maximum GPU memory usage threshold (90%)
    TEMPERATURE = 0.7
    MEMORY_THRESHOLD = 0.95  # 95% memory threshold for cleanup
    LOG_FILE = "experiment_log.txt"


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(Config.LOG_FILE), logging.StreamHandler(sys.stdout)],
)


class MemoryManager:
    """Enhanced memory management with monitoring and automatic cleanup"""

    @staticmethod
    def get_gpu_memory_map():
        return {
            i: torch.cuda.memory_allocated(i) / 1024**2
            for i in range(torch.cuda.device_count())
        }

    @staticmethod
    def memory_cleanup():
        """Aggressive memory cleanup"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    @staticmethod
    def check_memory_usage():
        """Check if memory usage exceeds threshold"""
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                memory_allocated = torch.cuda.memory_allocated(i)
                memory_reserved = torch.cuda.memory_reserved(i)
                if memory_allocated / memory_reserved > Config.MEMORY_THRESHOLD:
                    return True
        return False

    @staticmethod
    def get_memory_status():
        """Get detailed memory status"""
        memory_status = {"cpu": dict(psutil.virtual_memory()._asdict()), "gpu": []}
        if torch.cuda.is_available():
            for gpu in GPUtil.getGPUs():
                memory_status["gpu"].append(
                    {
                        "id": gpu.id,
                        "total": gpu.memoryTotal,
                        "used": gpu.memoryUsed,
                        "free": gpu.memoryFree,
                        "utilization": gpu.memoryUtil * 100,
                    }
                )
        return memory_status


class ExperimentMonitor:
    """Enhanced experiment monitoring"""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.start_time = time.time()
        self.metrics = {
            "timestamps": [],
            "gpu_memory": [],
            "cpu_memory": [],
            "generation_speeds": [],
            "memory_cleanups": 0,
            "oom_events": 0,
        }

        self.experiment_dir = os.path.join(
            output_dir, f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        os.makedirs(self.experiment_dir, exist_ok=True)

    def log_memory(self):
        memory_status = MemoryManager.get_memory_status()
        self.metrics["timestamps"].append(time.time() - self.start_time)
        self.metrics["gpu_memory"].append(memory_status["gpu"])
        self.metrics["cpu_memory"].append(memory_status["cpu"])

        # Log warning if memory usage is high
        if any(gpu["utilization"] > 90 for gpu in memory_status["gpu"]):
            logging.warning("High GPU memory utilization detected!")


def generate_with_auto_recovery(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    monitor: ExperimentMonitor,
) -> str:
    """Generate text with automatic error recovery"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            inputs = tokenizer(prompt, return_tensors="pt").to(Config.DEVICE)

            # Set gradients to None to save memory
            model.zero_grad(set_to_none=True)

            with torch.no_grad(), torch.cuda.amp.autocast():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=Config.MAX_NEW_TOKENS,
                    num_return_sequences=1,
                    temperature=Config.TEMPERATURE,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    top_k=50,
                    top_p=0.95,
                )

            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Clean up
            del outputs, inputs
            MemoryManager.memory_cleanup()

            return generated_text

        except RuntimeError as e:
            if "out of memory" in str(e):
                monitor.metrics["oom_events"] += 1
                logging.error(f"OOM error (attempt {attempt + 1}/{max_retries})")
                MemoryManager.memory_cleanup()
                if attempt == max_retries - 1:
                    raise RuntimeError("Max retries exceeded for generation")
            else:
                raise e


def main():
    meets_req, requirements = estimate_requirements(
        model_name=Config.MODEL_NAME, batch_size=Config.BATCH_SIZE
    )

    if not meets_req:
        logging.error("System does not meet memory requirements")
        recommended_memory = requirements["memory_requirements"][
            "recommended_gpu_memory"
        ]
        available_memory = max(
            gpu["memory_free"] for gpu in requirements["system_info"]["gpu_info"]
        )

        # Calculate maximum safe batch size
        max_batch_size = int(
            Config.BATCH_SIZE * (available_memory / recommended_memory)
        )
        logging.info(f"Recommended maximum batch size: {max_batch_size}")
        return

    logging.info("Starting experiment")
    monitor = ExperimentMonitor(Config.OUTPUT_DIR)

    # Configure quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    try:
        # Model loading with error handling
        try:
            model = AutoModelForCausalLM.from_pretrained(
                Config.MODEL_NAME,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

        except Exception as e:
            logging.error(f"Failed to load model: {str(e)}")
            raise

        # Set model to evaluation mode
        model.eval()

        # Generate for each prompt
        for prompt_name, prompt_text in PROMPTS.items():
            logging.info(f"Processing prompt: {prompt_name}")

            try:
                records = generate_batch(
                    model, tokenizer, prompt_text, Config.SAMPLES_PER_PROMPT, monitor
                )
                save_and_analyze_dataset(records, prompt_name, monitor.experiment_dir)

            except Exception as e:
                logging.error(f"Error processing prompt {prompt_name}: {str(e)}")
                continue

            finally:
                MemoryManager.memory_cleanup()

        # Save final metrics
        monitor.save_metrics(
            {
                "model_name": Config.MODEL_NAME,
                "device": Config.DEVICE,
                "samples_per_prompt": Config.SAMPLES_PER_PROMPT,
                "quantization": "4-bit",
                "batch_size": Config.BATCH_SIZE,
                "torch_version": torch.__version__,
                "cuda_version": (
                    torch.version.cuda if torch.cuda.is_available() else None
                ),
                "gpu_info": (
                    GPUtil.getGPUs()[0].name if torch.cuda.is_available() else None
                ),
                "oom_events": monitor.metrics["oom_events"],
                "memory_cleanups": monitor.metrics["memory_cleanups"],
            }
        )

    except Exception as e:
        logging.error(f"Experiment failed: {str(e)}")
        raise

    finally:
        MemoryManager.memory_cleanup()
        logging.info("Experiment completed")


if __name__ == "__main__":
    main()
