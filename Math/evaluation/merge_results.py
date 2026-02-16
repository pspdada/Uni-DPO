import json
import os

from openpyxl import Workbook

# Input folder path containing all validation results
# This folder contains multiple subfolders, each corresponding to a model's validation results, with multiple folders representing different datasets
INPUT_DIR = "results"

# Output filename for results, including text and Excel files
result_filename = "summary_results"

# Define datasets to display results, separated by commas
DISPLAY_DATASETS = "math500,minerva_math,olympiadbench,aime24,amc23,gaokao2023en,college_math,gsm8k"
# Define datasets to calculate average, separated by commas
CALCULATE_AVG_DATASETS = DISPLAY_DATASETS

# Convert strings to lists
DISPLAY_DATASETS_LIST = DISPLAY_DATASETS.split(",")
CALCULATE_AVG_DATASETS_LIST = CALCULATE_AVG_DATASETS.split(",")


def _extract_results_from_folder(folder_path: str):
    # Store results for each subfolder
    results = {}

    # Iterate through all subfolders in the main folder
    for subdir in os.listdir(folder_path):
        subdir_path = os.path.join(folder_path, subdir)

        # Ensure it's a folder and the folder name is in DISPLAY_DATASETS_LIST
        if os.path.isdir(subdir_path) and subdir.lower() in DISPLAY_DATASETS_LIST:
            # Look for JSON files containing "metrics"
            metrics_file = None
            for file in os.listdir(subdir_path):
                if "metrics" in file and file.endswith(".json"):
                    metrics_file = os.path.join(subdir_path, file)
                    break

            # If metrics file is found, read and extract acc value
            if metrics_file:
                with open(metrics_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if "acc" in data:
                        results[subdir] = data["acc"]

    return results


def _write_to_excel(excel_path, all_results):
    # Create a new Excel workbook
    wb = Workbook()
    ws = wb.active
    ws.title = "Summary Results"

    # Write headers
    headers = ["Folder Name"] + DISPLAY_DATASETS_LIST + ["Average"]
    ws.append(headers)

    # Write data
    for result in all_results:
        row = [result["folder_name"]]
        for dataset in DISPLAY_DATASETS_LIST:
            row.append(result["results"].get(dataset, None))  # Fill None if no result for this dataset
        row.append(result["avg"])
        ws.append(row)

    # Save Excel file
    wb.save(excel_path)


def _calculate_average(results: dict[str, float]):
    # Only calculate average for datasets in CALCULATE_AVG_DATASETS_LIST
    filtered_results = {k: v for k, v in results.items() if k in CALCULATE_AVG_DATASETS_LIST}
    if not filtered_results:
        return 0
    return sum(filtered_results.values()) / len(filtered_results)


def _get_results(folder_name, results):
    # Calculate average
    avg = _calculate_average(results)

    # Return results for Excel writing (no printing or saving yet)
    return {"folder_name": folder_name, "results": results, "avg": avg}


def process_folder(parent_folder: str):
    # Store all results for writing to Excel and text files
    all_results = []

    # Iterate through each subfolder in the parent folder
    for folder_name in os.listdir(parent_folder):
        folder_path = os.path.join(parent_folder, folder_name)

        # Ensure it's a folder
        if os.path.isdir(folder_path):
            # Extract results
            results = _extract_results_from_folder(folder_path)

            # If results are not empty, add to all_results
            if results:
                result_data = _get_results(folder_name, results)
                all_results.append(result_data)

    # Sort by avg value in descending order
    all_results.sort(key=lambda x: x["avg"], reverse=True)

    # Write to summary file and print results
    summary_file = os.path.join(parent_folder, result_filename + ".txt")
    with open(summary_file, "w", encoding="utf-8") as f:  # Clear and rewrite
        for result in all_results:
            folder_name = result["folder_name"]
            results = result["results"]
            avg = result["avg"]

            # Print header
            header = " ".join([key.ljust(15) for key in DISPLAY_DATASETS_LIST] + ["avg".ljust(15)])
            print(f"Result of {folder_name}:")
            print(header)

            # Print values
            values = " ".join(
                [
                    f"{results.get(key, None):.1f}".ljust(15)
                    if results.get(key, None) is not None
                    else "None".ljust(15)
                    for key in DISPLAY_DATASETS_LIST
                ]
            )
            print(values + f"{avg:.1f}".ljust(15))

            # Write to summary file
            f.write(f"Result of {folder_name}:\n")
            f.write(header + "\n")
            f.write(values + f"{avg:.1f}".ljust(15) + "\n\n")

    # Write to Excel file
    excel_path = os.path.join(parent_folder, result_filename + ".xlsx")
    _write_to_excel(excel_path, all_results)


def main() -> None:
    process_folder(INPUT_DIR)


if __name__ == "__main__":
    main()
