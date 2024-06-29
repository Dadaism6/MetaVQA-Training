import json
import random
from collections import Counter, defaultdict

def count_question_types(data):
    question_type_counts = Counter()
    for value in data.values():
        question_type_counts[value["question_type"]] += 1
    return question_type_counts

def balanced_sampling(data, num_samples, num_true_false_samples, answer_field='answer'):
    true_false_answers = defaultdict(list)
    other_answers = []

    for key, value in data.items():
        if value[answer_field] in ['True', 'False']:
            true_false_answers[value[answer_field]].append((key, value))
        else:
            other_answers.append((key, value))

    sampled_data = []

    # Sample fixed number of 'True' and 'False' answers
    sampled_data.extend(random.sample(true_false_answers['True'], min(num_true_false_samples, len(true_false_answers['True']))))
    sampled_data.extend(random.sample(true_false_answers['False'], min(num_true_false_samples, len(true_false_answers['False']))))

    # Sample from the remaining answer categories
    num_other_samples = num_samples - len(sampled_data)
    if num_other_samples > 0:
        sampled_data.extend(random.sample(other_answers, min(num_other_samples, len(other_answers))))

    return dict(sampled_data)

def count_question_types(data):
    question_type_counts = Counter()
    for value in data.values():
        question_type_counts[value["question_type"]] += 1
    return question_type_counts
# Load the JSON files
with open('/path/to/your/local/drive/metavqa_final/vqa/NuScenes_Mixed/training/multi_frame_processed/multi_frame_processed.json', 'r') as file:
    multi_frame_data = json.load(file)

with open('/path/to/your/local/drive/metavqa_final/vqa/NuScenes_Mixed/training/single_frame_processed/single_frame_processed.json', 'r') as file:
    single_frame_data = json.load(file)


# Count question types before sampling
multi_frame_counts_before = count_question_types(multi_frame_data)
single_frame_counts_before = count_question_types(single_frame_data)

# Sample 5000 items from each file
random.seed(42)
# sampled_multi_frame_data = random.sample(list(multi_frame_data.items()), 10000)
# sampled_single_frame_data = random.sample(list(single_frame_data.items()), 10000)
num_samples = 15000
num_true_false_samples = 3000  
sampled_multi_frame_dict = balanced_sampling(multi_frame_data, num_samples, num_true_false_samples)
sampled_single_frame_dict = balanced_sampling(single_frame_data, num_samples, num_true_false_samples)

# # Convert sampled data back to dictionaries
# sampled_multi_frame_dict = dict(sampled_multi_frame_data)
# sampled_single_frame_dict = dict(sampled_single_frame_data)

# Count question types after sampling
multi_frame_counts_after = count_question_types(sampled_multi_frame_dict)
single_frame_counts_after = count_question_types(sampled_single_frame_dict)

# Create a unified dictionary with newly named keys
unified_data = {}
for i, (key, value) in enumerate(sampled_multi_frame_dict.items()):
    unified_data[f'multi_frame_{i}'] = value

for i, (key, value) in enumerate(sampled_single_frame_dict.items()):
    unified_data[f'single_frame_{i}'] = value

def get_top_5_answers(data):
    answers = [value['answer'] for value in data.values()]
    answer_counts = Counter(answers)
    top_5_answers = answer_counts.most_common(5)
    return top_5_answers
print("Top 5 answers for all data:", get_top_5_answers(unified_data))
# Save the unified dictionary to a new JSON file
with open('/path/to/your/local/drive/dataset/nuscene_real_train/NuScenes_Mixed_train.json', 'w') as file:
    json.dump(unified_data, file, indent=4)

# Print counts before and after sampling
print("Question type counts before sampling (multi_frame):", multi_frame_counts_before)
print("Question type counts before sampling (single_frame):", single_frame_counts_before)
print("Question type counts after sampling (multi_frame):", multi_frame_counts_after)
print("Question type counts after sampling (single_frame):", single_frame_counts_after)

#=======================================================================================================

# import json
# import random
# from collections import defaultdict

# def load_json(file_path):
#     with open(file_path, 'r') as file:
#         data = json.load(file)
#     return data

# # def sample_by_supertype(data, supertype, sample_size):
# #     filtered_items = [key for key, value in data.items() if value['question_type'].startswith(supertype)]
# #     sampled_keys = random.sample(filtered_items, min(sample_size, len(filtered_items)))
# #     return {key: data[key] for key in sampled_keys}
# def sample_by_supertype(data, supertype, sample_size, max_false_answers):
#     filtered_items = [key for key, value in data.items() if value['question_type'].startswith(supertype)]
#     false_keys = [key for key in filtered_items if data[key]['answer'] == 'False']
#     non_false_keys = [key for key in filtered_items if data[key]['answer'] != 'False']

#     sampled_false_keys = random.sample(false_keys, min(max_false_answers, len(false_keys)))
#     remaining_sample_size = sample_size - len(sampled_false_keys)
#     sampled_non_false_keys = random.sample(non_false_keys, min(remaining_sample_size, len(non_false_keys)))

#     sampled_keys = sampled_false_keys + sampled_non_false_keys
#     return {key: data[key] for key in sampled_keys}

# def save_json(data, file_path):
#     with open(file_path, 'w') as file:
#         json.dump(data, file, indent=4)

# def print_stats(data):
#     question_type_counts = defaultdict(int)
#     # answer_counts = defaultdict(lambda: defaultdict(int))
#     answer_counts = Counter()

#     for key, value in data.items():
#         question_type = value['question_type']
#         answer = value['answer']
        
#         question_type_counts[question_type] += 1
#         answer_counts[answer] += 1
    
#     print("Question Type Counts:")
#     for question_type, count in question_type_counts.items():
#         print(f"{question_type}: {count}")
#     print("\nTop 5 Most Frequent Answers:")
#     top_answers = answer_counts.most_common(5)
#     for answer, count in top_answers:
#         print(f"{answer}: {count}")

# # Example usage
# input_file_path = '/path/to/your/local/drive/metavqa_final/vqa/training/train_all.json'  # Replace with the actual path to your input JSON file
# output_file_path = '/path/to/your/local/drive/dataset/waymo_train/waymo_train.json'  # Replace with the actual path to your output JSON file

# data = load_json(input_file_path)

# # Sample 5000 items from each of 'dynamic' and 'static'
# random.seed(42)
# max_false_answers = 1500  # Adjust this value based on your desired maximum number of "False" answers
# dynamic_sample = sample_by_supertype(data, 'dynamic', 15000, max_false_answers)
# static_sample = sample_by_supertype(data, 'static', 15000, max_false_answers)

# # Combine the sampled items
# sampled_data = {**dynamic_sample, **static_sample}
# save_json(sampled_data, output_file_path)

# # Print stats for the sampled data
# print("Stats for the Sampled Data:")
# print_stats(sampled_data)








# import json
# import random
# from collections import Counter
# #==========================================Nuscenes Real=====================================================
# def count_question_types(data):
#     question_type_counts = Counter()
#     for value in data.values():
#         question_type_counts[value["question_type"]] += 1
#     return question_type_counts
# # Load the JSON files
# with open('/path/to/your/local/drive/metavqa_final/vqa/NuScenes_Mixed/training/multi_frame_processed/multi_frame_processed.json', 'r') as file:
#     multi_frame_data = json.load(file)

# with open('/path/to/your/local/drive/metavqa_final/vqa/NuScenes_Mixed/training/single_frame_processed/single_frame_processed.json', 'r') as file:
#     single_frame_data = json.load(file)


# # Count question types before sampling
# multi_frame_counts_before = count_question_types(multi_frame_data)
# single_frame_counts_before = count_question_types(single_frame_data)

# # Sample 5000 items from each file
# random.seed(42)
# sampled_multi_frame_data = random.sample(list(multi_frame_data.items()), 10000)
# sampled_single_frame_data = random.sample(list(single_frame_data.items()), 10000)

# # Convert sampled data back to dictionaries
# sampled_multi_frame_dict = dict(sampled_multi_frame_data)
# sampled_single_frame_dict = dict(sampled_single_frame_data)

# # Count question types after sampling
# multi_frame_counts_after = count_question_types(sampled_multi_frame_dict)
# single_frame_counts_after = count_question_types(sampled_single_frame_dict)

# # Create a unified dictionary with newly named keys
# unified_data = {}
# for i, (key, value) in enumerate(sampled_multi_frame_data):
#     unified_data[f'multi_frame_{i}'] = value

# for i, (key, value) in enumerate(sampled_single_frame_data):
#     unified_data[f'single_frame_{i}'] = value

# def get_top_5_answers(data):
#     answers = [value['answer'] for value in data.values()]
#     answer_counts = Counter(answers)
#     top_5_answers = answer_counts.most_common(5)
#     return top_5_answers
# print("Top 5 answers for all data:", get_top_5_answers(unified_data))
# # Save the unified dictionary to a new JSON file
# # with open('/path/to/your/local/drive/dataset/nuscene_real_train/NuScenes_Mixed_train.json', 'w') as file:
# #     json.dump(unified_data, file, indent=4)

# # Print counts before and after sampling
# print("Question type counts before sampling (multi_frame):", multi_frame_counts_before)
# print("Question type counts before sampling (single_frame):", single_frame_counts_before)
# print("Question type counts after sampling (multi_frame):", multi_frame_counts_after)
# print("Question type counts after sampling (single_frame):", single_frame_counts_after)


#===========================================Waymo=====================================================

# import json
# import random
# from collections import defaultdict

# def load_json(file_path):
#     with open(file_path, 'r') as file:
#         data = json.load(file)
#     return data

# def sample_by_supertype(data, supertype, sample_size):
#     filtered_items = [key for key, value in data.items() if value['question_type'].startswith(supertype)]
#     sampled_keys = random.sample(filtered_items, min(sample_size, len(filtered_items)))
#     return {key: data[key] for key in sampled_keys}

# def save_json(data, file_path):
#     with open(file_path, 'w') as file:
#         json.dump(data, file, indent=4)

# def print_stats(data):
#     question_type_counts = defaultdict(int)
#     answer_counts = defaultdict(lambda: defaultdict(int))
#     valid_answers = {"Yes", "No"}

#     for key, value in data.items():
#         question_type = value['question_type']
#         answer = value['answer']
        
#         question_type_counts[question_type] += 1
        
    
#     print("Question Type Counts:")
#     for question_type, count in question_type_counts.items():
#         print(f"{question_type}: {count}")

# # Example usage
# input_file_path = '/path/to/your/local/drive/metavqa_final/vqa/training/train_all.json'  # Replace with the actual path to your input JSON file
# output_file_path = '/path/to/your/local/drive/dataset/waymo_train/waymo_train.json'  # Replace with the actual path to your output JSON file

# data = load_json(input_file_path)

# # Sample 5000 items from each of 'dynamic' and 'static'
# random.seed(42)
# dynamic_sample = sample_by_supertype(data, 'dynamic', 5000)
# static_sample = sample_by_supertype(data, 'static', 5000)

# # Combine the sampled items
# sampled_data = {**dynamic_sample, **static_sample}
# save_json(sampled_data, output_file_path)

# # Print stats for the sampled data
# print("Stats for the Sampled Data:")
# print_stats(sampled_data)
