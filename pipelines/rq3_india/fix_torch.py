"""
Run this first: python fix_torch.py
It will print your exact Python path and the fix command tailored to it.
"""
import sys, os, subprocess

exe = sys.executable
print(f"Python: {exe}")
print(f"Version: {sys.version.split()[0]}")
print()

# Check current torch state
try:
    import torch
    print(f"Current torch: {torch.__version__}")
    print(f"torch path:    {torch.__file__}")
    t = torch.tensor([1.0])
    print(f"Test tensor:   {t}  ← WORKS")
    print("\nTorch is working from THIS Python. No fix needed.")
    print("The pipeline must be using a different Python.")
    print(f"Make sure you run: {exe} rq3_india_run_pipeline.py --stage 3")
except (ImportError, OSError) as e:
    print(f"Torch FAILS from this Python: {e}")
    print()
    print("=" * 55)
    print("RUN THESE EXACT COMMANDS (copy-paste):")
    print("=" * 55)
    print(f'"{exe}" -m pip uninstall torch torchvision torchaudio -y')
    print(f'"{exe}" -m pip install torch '
          '--index-url https://download.pytorch.org/whl/cpu')
    print(f'"{exe}" -c "import torch; print(torch.__version__, torch.tensor([1.]))"')
    print()
    print("Then run the pipeline with THIS Python:")
    print(f'"{exe}" rq3_india_run_pipeline.py --stage 3')
