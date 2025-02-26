# convert the evaluation results to json format
import json
import os


def convert_evaluation_to_json(input_path, output_path):
    try:
        # Load the evaluation results
        with open(input_path, "r") as f:
            results = json.load(f)

        # Convert to JSON format
        json_results = json.dumps(results, indent=4)

        # Save to JSON file
        with open(output_path, "w") as f:
            f.write(json_results)

        print(f"Converted {input_path} to {output_path}")
    except FileNotFoundError:
        print(f"Error: Could not find {input_path}")
    except json.JSONDecodeError:
        print(f"Error: {input_path} is not a valid JSON file")


def main():
    # Define the paths for all four evaluation results
    evaluations = [
        (
            "/home/ailaty3088@id.sdsu.edu/SynLLM/evaluation_results_PROMPT_1_data.txt",
            "evaluation_results_PROMPT_1_data.json",
        ),
        (
            "/home/ailaty3088@id.sdsu.edu/SynLLM/evaluation_results_PROMPT_2_data.txt",
            "evaluation_results_PROMPT_2_data.json",
        ),
        (
            "/home/ailaty3088@id.sdsu.edu/SynLLM/evaluation_results_PROMPT_3_data.txt",
            "evaluation_results_PROMPT_3_data.json",
        ),
        (
            "/home/ailaty3088@id.sdsu.edu/SynLLM/evaluation_results_PROMPT_4_data.txt",
            "evaluation_results_PROMPT_4_data.json",
        ),
    ]

    # Process each evaluation file
    for input_file, output_file in evaluations:
        convert_evaluation_to_json(input_file, output_file)


if __name__ == "__main__":
    main()
