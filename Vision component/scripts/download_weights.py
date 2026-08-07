"""
scripts/download_weights.py
───────────────────────────
One-command utility to verify that all required model weights are present.

Note: ViT gender classifier weights are fine-tuned locally and included
in the repository under weights/. No external downloads are needed.
"""

import os

WEIGHTS_DIR = "weights"
REQUIRED_WEIGHTS = ["vit_b16_gender_phase2.pth", "vit_b16_gender_phase1.pth"]

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    target_dir = os.path.join(root_dir, WEIGHTS_DIR)

    if not os.path.exists(target_dir):
        print(f"Error: weights directory not found at {target_dir}")
        return

    all_present = True
    for fname in REQUIRED_WEIGHTS:
        path = os.path.join(target_dir, fname)
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"  [OK] {fname} ({size / 1024 / 1024:.1f} MB)")
        else:
            print(f"  [MISSING] {fname}")
            all_present = False

    if all_present:
        print("All weights present.")
    else:
        print("Some weights are missing. Ensure the fine-tuned weight files are in place.")

if __name__ == "__main__":
    main()
