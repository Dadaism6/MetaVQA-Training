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

class MetaVQADataset(VQADataset):
    def __init__(self, vis_processor=None, text_processor=None, vis_root=None, ann_paths=None):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        self.answers = []
        self.questions = []
        self.images_paths = []

        self.vis_root = vis_root
        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self.init_metavqa(ann_paths)
        print("The number of data: ", len(self.questions))

    def init_metavqa(self, ann_paths):
        self.annotations = json.load(open(ann_paths[0], "r"))
        count = 0
        for id, val in self.annotations.items():
            image_path = val['rgb']['front'][0]
            question = val['question']
            anwser = convert_to_string(val['answer'])
            self.questions.append(question)
            self.answers.append([anwser])
            self.images_paths.append(image_path)
            # count += 1
            # if count >= 20:
            #     break

    def __getitem__(self, index):
        image_path = self.images_paths[index]
        image_path = image_path.replace('./', '')
        image_full_path = os.path.join(self.vis_root, image_path.replace('\\', '/'))
        image = Image.open(image_full_path).convert("RGB")
        image = self.vis_processor(image)
        question = self.questions[index]
        question = self.text_processor(question)
        answer = self.answers[index]


        return {
            "question": question,
            "answer": answer,
            "image": image,
        }

    def __len__(self):
        return len(self.questions)

    def collater(self, samples):
        # merge samples into a list for each key
        questions = [s["question"] for s in samples]
        answers = [s["answer"] for s in samples]
        images = [s["image"] for s in samples]

        images = torch.stack(images, dim=0)
        # [][][] -> []
        answers = [item[0] for item in answers]

        return {
            "vfeats": images,
            "questions": questions,
            "answers": answers,
        }


class MetaVQAEvalDataset(VQADataset):
    def __init__(self, vis_processor=None, text_processor=None, vis_root=None, ann_paths=None):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        self.answers = []
        self.questions = []
        self.images_paths = []

        self.vis_root = vis_root
        self.vis_processor = vis_processor
        self.text_processor = text_processor


        self.init_metavqa(ann_paths)
        print("The number of data: ", len(self.questions))

    def init_metavqa(self, ann_paths):
        self.annotations = json.load(open(ann_paths[0], "r"))
        count = 0
        for id, val in self.annotations.items():
            image_path = val['rgb']['front'][0]
            question = val['question']
            anwser = convert_to_string(val['answer'])
            self.questions.append(question)
            self.answers.append([anwser])
            self.images_paths.append(image_path)
            # count += 1
            # if count >= 20:
            #     break

    def __getitem__(self, index):
        image_path = self.images_paths[index]
        image_path = image_path.replace('./', '')
        image_full_path = os.path.join(self.vis_root, image_path.replace('\\', '/'))
        image = Image.open(image_full_path).convert("RGB")
        image = self.vis_processor(image)
        question = self.questions[index]
        question = self.text_processor(question)
        answer = self.answers[index]

        return {
            "question": question,
            "answer": answer,
            "image": image,
        }

    def __len__(self):
        return len(self.questions)

    def collater(self, samples):
        # merge samples into a list for each key
        questions = [s["question"] for s in samples]
        answers = [s["answer"] for s in samples]
        images = [s["image"] for s in samples]

        images = torch.stack(images, dim=0)
        answers = [item[0] for item in answers]

        return {
            "vfeats": images,
            "questions": questions,
            "answers": answers,
            # "vfeats": tmp_images,
        }
