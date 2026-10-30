import json

file_name = "naruto_s2.json"

try:
    # 1. Load your season 2 data
    with open(file_name, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # 2. Loop through and update each ID to "2.x" based on its position
    for index, doc in enumerate(dataset, start=1):
            doc["id"] = 220 + index

    # 3. Overwrite the file with the updated IDs
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)

    print(f"Successfully updated IDs for {len(dataset)} items in {file_name}!")

except FileNotFoundError:
    print(f"Error: Could not find {file_name} in your current path.")
