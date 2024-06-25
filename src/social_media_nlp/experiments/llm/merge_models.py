import argparse
import os
from pathlib import Path

from social_media_nlp.models.transformers.utils import merge_adapters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="Base model local path")
    parser.add_argument("subfolder", help="Full fine-tuned model path")

    args = parser.parse_args()

    base_path = Path(str(args.model_path))
    folder = Path(str(args.subfolder))
    subfolder = folder.parts[-1]

    if not os.path.exists(f"{base_path}_merged/{subfolder}"):
        os.makedirs(f"{base_path}_merged/{subfolder}")

    try:
        merge_adapters(
            pretrained_model_name_or_path=folder,
            output_path=f"{base_path}_merged/{subfolder}",
        )
    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()
