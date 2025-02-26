#!/usr/bin/env python3
import os
import signal
import subprocess
import sys
import time


def get_gpu_processes():
    """Get all processes using GPUs"""
    try:
        # Run nvidia-smi to get process information
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if result.returncode != 0:
            print("Error running nvidia-smi:", result.stderr)
            return []

        processes = []
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue

            parts = line.split(",")
            if len(parts) >= 1:
                pid = int(parts[0].strip())
                processes.append(pid)

        return processes
    except Exception as e:
        print(f"Error getting GPU processes: {e}")
        return []


def kill_gpu_processes(exclude_pid=None):
    """Kill all processes using GPUs except for the specified PID"""
    processes = get_gpu_processes()

    if not processes:
        print("No GPU processes found.")
        return

    current_pid = os.getpid()
    killed = 0

    for pid in processes:
        # Skip current process and excluded process
        if pid == current_pid or (exclude_pid and pid == exclude_pid):
            continue

        try:
            # Get process name for display
            process_name = subprocess.run(
                ["ps", "-p", str(pid), "-o", "comm="], stdout=subprocess.PIPE, text=True
            ).stdout.strip()

            print(f"Killing GPU process: {pid} ({process_name})")
            os.kill(pid, signal.SIGTERM)
            killed += 1
        except ProcessLookupError:
            print(f"Process {pid} not found")
        except PermissionError:
            print(f"Permission denied to kill process {pid}")

    print(f"Killed {killed} GPU processes")

    # Wait and check if processes were actually terminated
    if killed > 0:
        time.sleep(2)
        remaining = get_gpu_processes()
        remaining_count = len(
            [
                p
                for p in remaining
                if p != current_pid and (not exclude_pid or p != exclude_pid)
            ]
        )

        if remaining_count > 0:
            print(f"Warning: {remaining_count} GPU processes still running")
            print("You may need to use sudo or kill them manually")


def print_gpu_status():
    """Print current GPU status"""
    try:
        subprocess.run(["nvidia-smi"], check=True)
    except Exception as e:
        print(f"Error running nvidia-smi: {e}")


if __name__ == "__main__":
    print("Current GPU status:")
    print_gpu_status()

    # Ask for confirmation
    print("\nThis will kill all processes using GPUs (except this script).")
    response = input("Do you want to continue? (y/n): ")

    if response.lower() in ("y", "yes"):
        kill_gpu_processes()
        print("\nCleaned GPU memory. New status:")
        time.sleep(1)  # Give a moment for everything to clean up
        print_gpu_status()
    else:
        print("Operation cancelled.")
