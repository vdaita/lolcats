import os
import argparse
import math


# Create a function to get filenames from the checkpoints directory
def get_filenames(directory):
    try:
        return sorted(os.listdir(directory))
    except FileNotFoundError:
        print(f"Error: Directory {directory} not found.")
        return []


def split_files(filenames, num_gpus):
    """Split filenames into chunks for each GPU."""
    chunk_size = math.ceil(len(filenames) / num_gpus)
    return [filenames[i * chunk_size : (i + 1) * chunk_size] for i in range(num_gpus)]


def write_gpu_scripts(output_dir, file_chunks):
    """Write individual GPU scripts."""
    gpu_scripts = []

    for gpu_id, chunk in enumerate(file_chunks):
        script_path = os.path.join(output_dir, f"run_gpu_{gpu_id}.sh")
        gpu_scripts.append(script_path)

        with open(script_path, "w") as script_file:
            script_file.write(f"#!/bin/bash\n")
            script_file.write(f"conda activate lolcats-env")
            script_file.write(f"export CUDA_VISIBLE_DEVICES={gpu_id}\n")

            for filename in chunk:
                script_file.write(f"echo Processing {filename} on GPU {gpu_id}\n")
                # Add the actual command to process the checkpoint file here
                filename_no_ext = filename.split("/")[-1].split(".")[0]

                script_file.write(
                    f"""WANDB_MODE="offline" python distill_llama.py --model_config {filename_no_ext} \
--distill_config distill_alpaca_clean_xent0_mse1000_lr1e-2 \
--finetune_config finetune_lora_qkvo_alpaca_clean \
--eval_config eval_alpaca_clean \
--lk_zero_init \
--verbose --seed 0 --replicate 0\n\n"""
                )

        os.chmod(script_path, 0o755)

    return gpu_scripts


def write_launcher_script(output_dir, gpu_scripts):
    """Write a launcher script to launch all GPU scripts in the background."""
    launcher_path = os.path.join(output_dir, "launch_all.sh")

    with open(launcher_path, "w") as launcher_file:
        launcher_file.write("#!/bin/bash\n")

        for script in gpu_scripts:
            launcher_file.write(f"bash {script} &\n")

    os.chmod(launcher_path, 0o755)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate GPU scripts for checkpoint processing."
    )
    parser.add_argument(
        "--configs_dir",
        type=str,
        default="./configs/model/",
        help="Path to the checkpoints directory.",
    )
    parser.add_argument(
        "--num_gpus", type=int, required=True, help="Number of GPUs to use."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./gpu_scripts/",
        help="Output directory for the generated scripts.",
    )

    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Get filenames from the checkpoints directory
    filenames = get_filenames(args.configs_dir)
    if not filenames:
        exit(1)

    # Split filenames among GPUs
    file_chunks = split_files(filenames, args.num_gpus)

    # Write GPU scripts
    gpu_scripts = write_gpu_scripts(args.output_dir, file_chunks)

    # Write the launcher script
    write_launcher_script(args.output_dir, gpu_scripts)

    print(f"Scripts generated in {args.output_dir}.")
