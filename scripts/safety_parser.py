import json
import random
from collections import defaultdict
from tqdm import tqdm

def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def balance_subsample(data):
    question_type_counts = defaultdict(lambda: {'Yes': [], 'No': []})
    valid_answers = {"Yes", "No"}

    # for key, value in data.items():
    for key, value in tqdm(data.items()):
        question_type = value['question_type']
        answer = value['answer']
        
        if any(answer.startswith(valid) for valid in valid_answers):
            first_word = answer.split()[0]  # Get the first word of the answer
            first_word = first_word.rstrip('.')
            if first_word in valid_answers:
                question_type_counts[question_type][first_word].append(key)
        else:
            print(f"Warning: Answer '{answer}' for question type '{question_type}' does not start with Yes or No.")
    # print(question_type_counts)
    balanced_data = {}
    
    # for question_type, answers in question_type_counts.items():
    for question_type, answers in tqdm(question_type_counts.items()):
        yes_count = len(answers['Yes'])
        no_count = len(answers['No'])
        min_count = min(yes_count, no_count)
        
        if min_count > 0:
            sampled_yes_keys = random.sample(answers['Yes'], min_count)
            sampled_no_keys = random.sample(answers['No'], min_count)
            
            for key in sampled_yes_keys + sampled_no_keys:
                balanced_data[key] = data[key]
    
    return balanced_data

def save_json(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def print_stats(data):
    print("Total Number of Items:")
    print(len(data))
    question_type_counts = defaultdict(int)
    answer_counts = defaultdict(lambda: defaultdict(int))
    valid_answers = {"Yes", "No"}

    for key, value in data.items():
        question_type = value['question_type']
        answer = value['answer']
        
        question_type_counts[question_type] += 1
        
        if any(answer.startswith(valid) for valid in valid_answers):
            first_word = answer.split()[0]  # Get the first word of the answer
            answer_counts[question_type][first_word] += 1
    
    print("Question Type Counts:")
    for question_type, count in question_type_counts.items():
        print(f"{question_type}: {count}")
    
    print("\nAnswer Counts for Each Question Type:")
    for question_type, answers in answer_counts.items():
        print(f"{question_type}:")
        for answer, count in answers.items():
            print(f"  {answer}: {count}")

# Example usage
input_file_path = '/path/to/your/local/drive/metavqa_final/vqa/training/train_safety_all.json'  # Replace with the actual path to your input JSON file
output_file_path = '/path/to/your/local/drive/ELM/train_safety_balanced/train_safety_balanced.json'  # Replace with the actual path to your output JSON file

data = load_json(input_file_path)
print("Stats for the Original Data:")
print_stats(data)
balanced_data = balance_subsample(data)
# save_json(balanced_data, output_file_path)

# Print stats for the balanced subsample
print("Stats for the Balanced Subsample:")
print_stats(balanced_data)

# print first several element of data
for key, value in data.items():
    print(key, value)
    break
for key, value in balanced_data.items():
    print(key, value)
    break




# import json
# from collections import defaultdict
# from tqdm import tqdm
# def process_json(file_path):
#     with open(file_path, 'r') as file:
#         data = json.load(file)
    
#     question_type_counts = defaultdict(int)
#     answer_counts = defaultdict(lambda: defaultdict(int))
    
#     valid_answers = {"Yes", "No", "True", "False"}
    
#     # for key, value in data.items():
#     for key, value in tqdm(data.items()):
#         question_type = value['question_type']
#         answer = value['answer']
        
#         question_type_counts[question_type] += 1
        
#         if any(answer.startswith(valid) for valid in valid_answers):
#             first_word = answer.split()[0]  # Get the first word of the answer
#             answer_counts[question_type][first_word] += 1
#         else:
#             print(f"Warning: Answer '{answer}' for question type '{question_type}' does not start with Yes, No, True, or False.")
    
#     return question_type_counts, answer_counts

# def print_results(question_type_counts, answer_counts):
#     print("Question Type Counts:")
#     for question_type, count in question_type_counts.items():
#         print(f"{question_type}: {count}")
    
#     print("\nAnswer Counts for Each Question Type:")
#     for question_type, answers in answer_counts.items():
#         print(f"{question_type}:")
#         for answer, count in answers.items():
#             print(f"  {answer}: {count}")

# # Example usage
# file_path = '/path/to/your/local/drive/metavqa_final/vqa/training/train_safety_all.json'  # Replace with the actual path to your JSON file
# question_type_counts, answer_counts = process_json(file_path)
# print_results(question_type_counts, answer_counts)
