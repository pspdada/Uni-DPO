import argparse
import json
import os


def change_image_paths(input_file: str, image_dir: str, output_file: str = None) -> None:
    """Replace image paths in JSON file with new image directory path.

    Args:
        input_file: Path to input JSON file
        image_dir: New base directory path (should end with MM-RLHF)
        output_file: Path to output JSON file (if None, overwrites input file)
    """
    # Validate image directory
    if not image_dir.endswith("MM-RLHF"):
        print(f"Warning: IMAGE_DIR does not end with 'MM-RLHF': {image_dir}")

    # Load JSON data
    with open(input_file, encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} entries from {input_file}")

    # Process each entry
    modified_count = 0
    for entry in data:
        if "images" in entry and isinstance(entry["images"], list):
            new_images = []
            for img_path in entry["images"]:
                # Extract relative path after MM-RLHF
                if "MM-RLHF" in img_path:
                    relative_path = img_path.split("MM-RLHF")[-1].lstrip("/")
                    new_path = os.path.join(image_dir, relative_path)
                    new_images.append(new_path)
                    modified_count += 1
                else:
                    print(f"Warning: Image path does not contain 'MM-RLHF': {img_path}")
                    new_images.append(img_path)

            entry["images"] = new_images

    # Save modified data
    output_path = output_file if output_file else input_file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"Modified {modified_count} image paths")
    print(f"Saved to {output_path}")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Replace image paths in JSON file with new base directory")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input JSON file")
    parser.add_argument(
        "--image_dir", type=str, required=True, help="New base directory path (should end with MM-RLHF)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to output JSON file (if not specified, overwrites input file)",
    )

    args = parser.parse_args()

    # Validate input file exists
    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input file not found: {args.input_file}")

    # Validate image directory exists
    if not os.path.exists(args.image_dir):
        raise FileNotFoundError(f"Image directory not found: {args.image_dir}")

    change_image_paths(args.input_file, args.image_dir, args.output_file)


if __name__ == "__main__":
    main()
