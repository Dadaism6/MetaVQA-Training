import os
import json
from sklearn.model_selection import train_test_split

def count_items_in_subfolders(base_folder):
    subfolder_names = ["multi_frame_processed", "single_frame_processed", "safety_processed"]
    counts = {}

    for subfolder_name in subfolder_names:
        subfolder_path = os.path.join(base_folder, subfolder_name)
        total_items = 0

        if os.path.exists(subfolder_path) and os.path.isdir(subfolder_path):
            for json_file in os.listdir(subfolder_path):
                if json_file.endswith(".json"):
                    json_path = os.path.join(subfolder_path, json_file)
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                        total_items += len(data)

        counts[subfolder_name] = total_items

    return counts

def create_folders(new_base_folder):
    main_folders = ["training", "testing", "validation", "total"]
    subfolders = ["multi_frame_processed", "single_frame_processed", "safety_processed"]

    for main_folder in main_folders:
        main_folder_path = os.path.join(new_base_folder, main_folder)
        os.makedirs(main_folder_path, exist_ok=True)
        for subfolder in subfolders:
            os.makedirs(os.path.join(main_folder_path, subfolder), exist_ok=True)

def split_data(data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    train_data, temp_data = train_test_split(data, test_size=(val_ratio + test_ratio))
    val_data, test_data = train_test_split(temp_data, test_size=test_ratio / (val_ratio + test_ratio))
    return train_data, val_data, test_data

def process_and_split_data(base_folder, new_base_folder):
    subfolder_names = ["multi_frame_processed", "single_frame_processed", "safety_processed"]

    create_folders(new_base_folder)

    initial_counts = count_items_in_subfolders(base_folder)
    print("Initial Counts:", initial_counts)

    final_counts = {"training": {}, "validation": {}, "testing": {}, "total": {}}

    for subfolder_name in subfolder_names:
        subfolder_path = os.path.join(base_folder, subfolder_name)
        all_data = []

        # Collect all data
        if os.path.exists(subfolder_path) and os.path.isdir(subfolder_path):
            for json_file in os.listdir(subfolder_path):
                if json_file.endswith(".json"):
                    json_path = os.path.join(subfolder_path, json_file)
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                        print(f"Loaded {len(data)} items from {json_file} in {subfolder_name}")
                        all_data.extend(data.items())

        # Debugging information
        print(f"Total items collected for {subfolder_name}: {len(all_data)}")

        # Verify if total items match initial count
        if len(all_data) != initial_counts[subfolder_name]:
            print(f"Warning: Mismatch in total items collected for {subfolder_name}. Expected {initial_counts[subfolder_name]}, but got {len(all_data)}")

        # Split data
        train_data, val_data, test_data = split_data(all_data)

        # Generate new unique keys for each item
        def generate_unique_data(data):
            return {f"{i}": v for i, (k, v) in enumerate(data)}

        train_data = generate_unique_data(train_data)
        val_data = generate_unique_data(val_data)
        test_data = generate_unique_data(test_data)
        all_data_dict = generate_unique_data(all_data)

        # Print and store statistics
        print(f"{subfolder_name} - Total: {len(all_data)}, Training: {len(train_data.keys())}, Validation: {len(val_data.keys())}, Testing: {len(test_data.keys())}")
        final_counts["training"][subfolder_name] = len(train_data)
        final_counts["validation"][subfolder_name] = len(val_data)
        final_counts["testing"][subfolder_name] = len(test_data)
        final_counts["total"][subfolder_name] = len(all_data)

        # Save split data
        for dataset_name, dataset in [("training", train_data), ("validation", val_data), ("testing", test_data)]:
            new_folder_path = os.path.join(new_base_folder, dataset_name, subfolder_name)
            new_file_path = os.path.join(new_folder_path, f"{subfolder_name}.json")
            # with open(new_file_path, 'w') as f:
            #     json.dump(dataset, f)

        # Save total data
        total_folder_path = os.path.join(new_base_folder, "total", subfolder_name)
        total_file_path = os.path.join(total_folder_path, f"{subfolder_name}.json")
        # with open(total_file_path, 'w') as f:
        #     json.dump(all_data_dict, f)

    print("Final Counts:", final_counts)
# Usage example
base_folder = "/path/to/your/local/drive/metavqa_final/vqa/NuScenes_Mixed"
new_base_folder = "/path/to/your/local/drive/metavqa_final/vqa/NuScenes_Mixed"
process_and_split_data(base_folder, new_base_folder)





# import os
# import json
# from sklearn.model_selection import train_test_split
#
# def create_folders(base_folder, new_base_folder):
#     main_folders = ["training", "testing", "validation", "total"]
#     subfolders = ["multi_frame_processed", "single_frame_processed", "safety_processed"]
#
#     for main_folder in main_folders:
#         main_folder_path = os.path.join(new_base_folder, main_folder)
#         os.makedirs(main_folder_path, exist_ok=True)
#         for subfolder in subfolders:
#             os.makedirs(os.path.join(main_folder_path, subfolder), exist_ok=True)
#
# def split_data(data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
#     train_data, temp_data = train_test_split(data, test_size=(val_ratio + test_ratio))
#     val_data, test_data = train_test_split(temp_data, test_size=test_ratio / (val_ratio + test_ratio))
#     return train_data, val_data, test_data
#
# def process_and_split_data(base_folder, new_base_folder):
#     subfolder_names = ["multi_frame_processed", "single_frame_processed", "safety_processed"]
#
#     create_folders(base_folder, new_base_folder)
#
#     for subfolder_name in subfolder_names:
#         subfolder_path = os.path.join(base_folder, subfolder_name)
#         all_data = {}
#
#         # Collect all data
#         if os.path.exists(subfolder_path) and os.path.isdir(subfolder_path):
#             for json_file in os.listdir(subfolder_path):
#                 if json_file.endswith(".json"):
#                     json_path = os.path.join(subfolder_path, json_file)
#                     with open(json_path, 'r') as f:
#                         data = json.load(f)
#                         all_data.update(data)
#
#         # Split data
#         train_data, val_data, test_data = split_data(list(all_data.items()))
#
#         # Print statistics
#         print(f"{subfolder_name} - Total: {len(all_data)}, Training: {len(train_data)}, Validation: {len(val_data)}, Testing: {len(test_data)}")
#
#         # Save split data
#         for dataset_name, dataset in [("training", train_data), ("validation", val_data), ("testing", test_data)]:
#             new_folder_path = os.path.join(new_base_folder, dataset_name, subfolder_name)
#             new_file_path = os.path.join(new_folder_path, f"{subfolder_name}.json")
#             with open(new_file_path, 'w') as f:
#                 json.dump(dict(dataset), f)
#
#         # Save total data
#         total_folder_path = os.path.join(new_base_folder, "total", subfolder_name)
#         total_file_path = os.path.join(total_folder_path, f"{subfolder_name}.json")
#         with open(total_file_path, 'w') as f:
#             json.dump(all_data, f)
#
# # Usage example
# base_folder = "/path/to/your/local/drive/metavqa_final/vqa/NuScenes_Mixed"
# new_base_folder = "/path/to/your/local/drive/metavqa_final/vqa/NuScenes_Mixed"
# # process_and_split_data(base_folder, new_base_folder)
#
#
#
#
# def count_items_in_subfolders(base_folder):
#     subfolder_names = ["multi_frame_processed", "single_frame_processed", "safety_processed"]
#     counts = {}
#
#     for subfolder_name in subfolder_names:
#         subfolder_path = os.path.join(base_folder, subfolder_name)
#         total_items = 0
#
#         if os.path.exists(subfolder_path) and os.path.isdir(subfolder_path):
#             for json_file in os.listdir(subfolder_path):
#                 if json_file.endswith(".json"):
#                     json_path = os.path.join(subfolder_path, json_file)
#                     with open(json_path, 'r') as f:
#                         data = json.load(f)
#                         total_items += len(data)
#
#         counts[subfolder_name] = total_items
#
#     return counts
# #
# # # Usage example
# base_folder = "/path/to/your/local/drive/metavqa_final/vqa/NuScenes_Mixed"
# counts = count_items_in_subfolders(base_folder)
# print(counts)
