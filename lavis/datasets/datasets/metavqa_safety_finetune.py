from lavis.datasets.datasets.vqa_datasets import VQADataset, VQAEvalDataset
import json
import os
import torch
from PIL import Image
def convert_to_string(value):
    # If the value is not a list, simply convert it to string
    if not isinstance(value, list):
        return str(value)
    # If it's a list, we process each element
    else:
        converted_list = []
        for item in value:
            # For nested lists, we apply the same function recursively
            if isinstance(item, list):
                converted_list.append(convert_to_string(item))
            else:
                # For non-list items in the list, convert directly to string
                converted_list.append(str(item))
        # Join the list items into a single string representation
        # You might want to adjust this part based on how you wish to represent the list as a string
        return '[' + ', '.join(converted_list) + ']'
def process_image_path(annotation_val):
    image_path = []
    for view in ["front", "leftf", "leftb", "rightf", "rightb", "back"]:
        if view in annotation_val['rgb']:
            curr_view_path = []
            for path in annotation_val['rgb'][view]:
                curr_view_path.append(path)
            image_path.append(curr_view_path)
        else:
            raise ValueError(f"The view {view} is not in the annotation")
    # we should have equal number of images for each view
    for i in range(1, len(image_path)):
        if len(image_path[i]) != len(image_path[0]):
            raise ValueError(f"The number of images in each view should be the same")
    return image_path
class FinetuneSafetyDataset(VQADataset):
    def __init__(self, vis_processor=None, text_processor=None, vis_root=None, ann_paths=None):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        self.answers = []
        self.questions = []
        self.images_paths = []
        self.question_type = []
        self.answer_form = []
        self.dataset_source = []

        self.vis_root = vis_root
        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self.init_metavqa(ann_paths)
        print("Safety Finetune dataset: The number of Training data: ", len(self.questions))
    def init_metavqa(self, ann_paths):
        ann_paths = [
                "/path/to/your/local/drive/metavqa_final/vqa/training/multi_frame_processed/Waymo",
                     "/path/to/your/local/drive/metavqa_final/vqa/training/single_frame_processed/Waymo",
                #      "/path/to/your/local/drive/metavqa_final/vqa/training/safety_critical_processed/Waymo",
                     "/path/to/your/local/drive/ELM/train_safety_balanced"]
        sample_numbers = [2000, 2000, 75236]
        # sample_numbers = [1000, 1000, 500, 500]
        # sample_numbers = [5]
        # sample_number = 2000
        # sample_number = 5
        for ann_path, sample_number in zip(ann_paths,sample_numbers):
            # list files in ann path
            print("Load from: " + ann_path)
            print("Current Num: " + str(len(self.questions)))
            ann_files = os.listdir(ann_path)
            count = 0
            for ann_file in ann_files:
                if ann_file.endswith('.json'):
                    annotations = json.load(open(os.path.join(ann_path, ann_file), "r"))
                    # Calculate the step size
                    total_annotations = len(annotations)
                    step_size = max(1, total_annotations // sample_number)

                    # Initialize counters
                    count = 0
                    index = 0

                    # Get annotation keys
                    annotation_keys = list(annotations.keys())

                    # Loop through the annotation keys with a step size
                    while count < sample_number and index < total_annotations:
                        id = annotation_keys[index]
                        val = annotations[id]
                        image_path = process_image_path(val)
                        question = val['question']
                        answer = val['answer']
                        self.questions.append(question)
                        self.answers.append([answer])
                        self.images_paths.append(image_path)
                        self.dataset_source.append(val['source'])
                        self.question_type.append(val['question_type'])
                        self.answer_form.append(val['answer_form'])
                        count += 1
                        index += step_size

                    # In case the last step doesn't fill the sample_number
                    while count < sample_number and index < total_annotations:
                        id = annotation_keys[index]
                        val = annotations[id]
                        image_path = process_image_path(val)
                        question = val['question']
                        answer = val['answer']
                        self.questions.append(question)
                        self.answers.append([answer])
                        self.images_paths.append(image_path)
                        self.dataset_source.append(val['source'])
                        self.question_type.append(val['question_type'])
                        self.answer_form.append(val['answer_form'])
                        count += 1
                        index += 1


    def __getitem__(self, index):
        image_path = self.images_paths[index]
        image = self.load_and_process_images(image_path)


        question = self.questions[index]
        question = self.text_processor(question)
        answer = self.answers[index]
        dataset_source = self.dataset_source[index]
        question_type = self.question_type[index]
        answer_form = self.answer_form[index]


        return {
            "question": question,
            "answer": answer,
            "image": image,
            "dataset_source": dataset_source,
            "question_type": question_type,
            "answer_form": answer_form,
        }

    def __len__(self):
        return len(self.questions)

    def collater(self, samples):
        # merge samples into a list for each key
        questions = [s["question"] for s in samples]
        answers = [s["answer"] for s in samples]
        images = [s["image"] for s in samples]
        dataset_source = [s["dataset_source"] for s in samples]
        question_type = [s["question_type"] for s in samples]
        answer_form = [s["answer_form"] for s in samples]

        images = torch.stack(images, dim=0)
        # [][][] -> []
        answers = [item[0] for item in answers]

        return {
            "vfeats": images,
            "questions": questions,
            "answers": answers,
            "dataset_source": dataset_source,
            "question_type": question_type,
            "answer_form": answer_form,
        }

    def load_and_process_images(self, images_paths):
        # Define the views
        views = ["front", "leftf", "leftb", "rightf", "rightb", "back"]

        # Initialize a list to store processed images for each view
        all_views_images = [[] for _ in views]
        select_index = [0, 5, 10, 15, 19]
        # Iterate through the image paths
        for i, view_paths in enumerate(images_paths):
            if len(view_paths) == 1:
                curr_select_index = [0]
            else:
                curr_select_index = select_index
            for view_index in curr_select_index:
            # for view_index, view_path in enumerate(view_paths):
                view_path = view_paths[view_index]
                # Get the full path of the image
                image_full_path = os.path.join(self.vis_root, view_path.replace('./', '').replace('\\', '/'))
                # Load and process the image
                image = Image.open(image_full_path).convert("RGB")
                processed_image = self.vis_processor(image)
                all_views_images[i].append(processed_image)

        # Convert the lists of images into NumPy arrays
        all_views_images = [torch.stack(view_images, dim=0) for view_images in all_views_images]

        # Concatenate the processed images into the desired shape (t, views, height, width, channel)
        processed_images_tensor = torch.stack(all_views_images, dim=1)
        t, views, height, width, channel = processed_images_tensor.shape
        # If t is 1, repeat the tensor 20 times along the first dimension
        if t == 1:
            processed_images_tensor = processed_images_tensor.repeat(5, 1, 1, 1, 1)

        return processed_images_tensor

    def load_and_process_front_images(self, images_paths):
        # Find the index of the "front" view
        views = ["front", "leftf", "leftb", "rightf", "rightb", "back"]
        front_view_index = views.index("front")

        # Get the list of front images
        front_images_paths = images_paths[front_view_index]

        # Initialize a list to store processed front images
        processed_front_images = []

        # Iterate through the front image paths
        for image_path in front_images_paths:
            # Get the full path of the image
            image_full_path = os.path.join(self.vis_root, image_path.replace('./', '').replace('\\', '/'))
            # Load and process the image
            image = Image.open(image_full_path).convert("RGB")
            processed_image = self.vis_processor(image)
            processed_front_images.append(processed_image)

        # Convert the list of processed images into a single tensor
        processed_front_images_tensor = torch.stack(processed_front_images, dim=0)

        return processed_front_images_tensor

class FinetuneSafetyEvalDataset(VQADataset):
    def __init__(self, vis_processor=None, text_processor=None, vis_root=None, ann_paths=None):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        self.answers = []
        self.questions = []
        self.images_paths = []
        self.question_type = []
        self.answer_form = []
        self.dataset_source = []

        self.vis_root = vis_root
        self.vis_processor = vis_processor
        self.text_processor = text_processor


        self.init_metavqa(ann_paths)
        print("Eval Safety Finetune:The number of Validation data: ", len(self.questions))
    def init_metavqa(self, ann_paths):
        ann_paths = ["/path/to/your/local/drive/metavqa_final/vqa/validation/multi_frame_processed/Waymo",
                     "/path/to/your/local/drive/metavqa_final/vqa/validation/single_frame_processed/Waymo",
                     "/path/to/your/local/drive/metavqa_final/vqa/validation/safety_critical_processed/Waymo",
                     "/path/to/your/local/drive/metavqa_final/vqa/validation/safety_critical_processed/CAT"]
        sample_numbers = [200, 200, 100, 100]
        # sample_numbers = [100]
        for ann_path, sample_number in zip(ann_paths,sample_numbers):
            print("Load from: " + ann_path)
            print("Current Num: " + str(len(self.questions)))
            # list files in ann path
            ann_files = os.listdir(ann_path)
            count = 0
            for ann_file in ann_files:
                if ann_file.endswith('.json'):
                    annotations = json.load(open(os.path.join(ann_path, ann_file), "r"))
                    # Calculate the step size
                    total_annotations = len(annotations)
                    step_size = max(1, total_annotations // sample_number)

                    # Initialize counters
                    count = 0
                    index = 0

                    # Get annotation keys
                    annotation_keys = list(annotations.keys())

                    # Loop through the annotation keys with a step size
                    while count < sample_number and index < total_annotations:
                        id = annotation_keys[index]
                        val = annotations[id]
                        image_path = process_image_path(val)
                        question = val['question']
                        answer = val['answer']
                        self.questions.append(question)
                        self.answers.append([answer])
                        self.images_paths.append(image_path)
                        self.dataset_source.append(val['source'])
                        self.question_type.append(val['question_type'])
                        self.answer_form.append(val['answer_form'])
                        count += 1
                        index += step_size

                    # In case the last step doesn't fill the sample_number
                    while count < sample_number and index < total_annotations:
                        id = annotation_keys[index]
                        val = annotations[id]
                        image_path = process_image_path(val)
                        question = val['question']
                        answer = val['answer']
                        self.questions.append(question)
                        self.answers.append([answer])
                        self.images_paths.append(image_path)
                        self.dataset_source.append(val['source'])
                        self.question_type.append(val['question_type'])
                        self.answer_form.append(val['answer_form'])
                        count += 1
                        index += 1

    def __getitem__(self, index):
        image_path = self.images_paths[index]
        image = self.load_and_process_images(image_path)
        question = self.questions[index]
        question = self.text_processor(question)
        answer = self.answers[index]
        dataset_source = self.dataset_source[index]
        question_type = self.question_type[index]
        answer_form = self.answer_form[index]

        return {
            "question": question,
            "answer": answer,
            "image": image,
            "dataset_source": dataset_source,
            "question_type": question_type,
            "answer_form": answer_form,
        }

    def __len__(self):
        return len(self.questions)

    def collater(self, samples):
        # merge samples into a list for each key
        questions = [s["question"] for s in samples]
        answers = [s["answer"] for s in samples]
        images = [s["image"] for s in samples]
        dataset_source = [s["dataset_source"] for s in samples]
        question_type = [s["question_type"] for s in samples]
        answer_form = [s["answer_form"] for s in samples]

        images = torch.stack(images, dim=0)
        answers = [item[0] for item in answers]

        return {
            "vfeats": images,
            "questions": questions,
            "answers": answers,
            "dataset_source": dataset_source,
            "question_type": question_type,
            "answer_form": answer_form,
            # "vfeats": tmp_images,
        }
    def load_and_process_images(self, images_paths):
        # Define the views
        views = ["front", "leftf", "leftb", "rightf", "rightb", "back"]

        # Initialize a list to store processed images for each view
        all_views_images = [[] for _ in views]

        select_index = [0, 5, 10, 15, 19]
        # Iterate through the image paths
        for i, view_paths in enumerate(images_paths):
            if len(view_paths) == 1:
                curr_select_index = [0]
            else:
                curr_select_index = select_index
            for view_index in curr_select_index:
            # for view_index, view_path in enumerate(view_paths):
                # Get the full path of the image
                view_path = view_paths[view_index]
                image_full_path = os.path.join(self.vis_root, view_path.replace('./', '').replace('\\', '/'))
                # Load and process the image
                image = Image.open(image_full_path).convert("RGB")
                processed_image = self.vis_processor(image)
                all_views_images[i].append(processed_image)


        # Convert the lists of images into NumPy arrays
        all_views_images = [torch.stack(view_images, dim=0) for view_images in all_views_images]

        # Concatenate the processed images into the desired shape (t, views, height, width, channel)
        processed_images_tensor = torch.stack(all_views_images, dim=1)
        t, views, height, width, channel = processed_images_tensor.shape
        # If t is 1, repeat the tensor 20 times along the first dimension
        if t == 1:
            processed_images_tensor = processed_images_tensor.repeat(5, 1, 1, 1, 1)

        return processed_images_tensor

    def load_and_process_front_images(self, images_paths):
        # Find the index of the "front" view
        views = ["front", "leftf", "leftb", "rightf", "rightb", "back"]
        front_view_index = views.index("front")

        # Get the list of front images
        front_images_paths = images_paths[front_view_index]

        # Initialize a list to store processed front images
        processed_front_images = []

        # Iterate through the front image paths
        for image_path in front_images_paths:
            # Get the full path of the image
            image_full_path = os.path.join(self.vis_root, image_path.replace('./', '').replace('\\', '/'))
            # Load and process the image
            image = Image.open(image_full_path).convert("RGB")
            processed_image = self.vis_processor(image)
            processed_front_images.append(processed_image)

        # Convert the list of processed images into a single tensor
        processed_front_images_tensor = torch.stack(processed_front_images, dim=0)

        return processed_front_images_tensor
