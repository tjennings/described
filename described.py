import argparse
import glob
import os
import json5
import torch
from PIL import Image
from lavis.models import load_model_and_preprocess
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import List, Any, Tuple, Union, Dict
from collections import OrderedDict


class Node:
    def __init__(self, data: Dict[str, Any]):
        self.data = data

    def get_template(self) -> str:
        return self.get("template", "{}")

    def get_question(self) -> str:
        return self.data["q"]

    def get_branches(self) -> Dict[str, Any]:
        return self.data.get("branches", {})

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

class Workflow:
    def __init__(self, nodes: List[Node]):
        self.nodes = nodes

    @staticmethod
    def from_json(json_data: List[Dict[str, Any]]):
        return Workflow([Node(node_data) for node_data in json_data])


class Inquisitor:
    def __init__(self, workflow: Workflow, model: Any, args: argparse.Namespace):
        self.workflow = workflow
        self.model = model
        self.args = args
        self.prefix = args.prefix or ""
        self.suffix = args.suffix or ""

    def ask(self, image: torch.Tensor) -> str:
        context: List[Tuple[str, str]] = []
        return f"{self.prefix} {self._traverse_workflow(image, context, self.workflow.nodes)} {self.suffix}"

    def _traverse_workflow(self, image: torch.Tensor, context: List[Tuple[str, str]], nodes: List[Node]) -> str:
        responses = []
        for node in nodes:
            response = self._process_node(image, context, node)
            if response:
                responses.append(response.strip())
        join_str = node.get("join", ' ')
        return join_str.join(responses)

    def _process_node(self, image: torch.Tensor, context: List[Tuple[str, str]], node: Node) -> str:
        answer, context = self._query(image, context, node)

        branches = node.get_branches()
        if answer in branches:
            branch = branches[answer]
            if isinstance(branch, list):
                return self._traverse_workflow(image, context, [Node(b) for b in branch])
            else:
                return self._process_node(image, context, Node(branch))
        elif not answer:
            return ""
        else:
            return node.get_template().format(answer)

    def _query(self, image: torch.Tensor, context: List[Tuple[str, str]], node: Node) -> Tuple[str, List[Tuple[str, str]]]:
        template = "Question: {}, Answer: {}."
        question = node.get_question()
        prompt = " ".join([template.format(q, a) for q, a in context]) + f" Question: {question} Answer:"
        answer = self.model.generate({"image": image, "prompt": prompt})[0].lower()
        answer = self._deduplicate(answer)

        #print(f"{prompt} {answer}")

        #if 'not enough information' in answer:
        #    raise Exception("Not enough information response")

        return answer, context + [(question, answer)]

    @staticmethod
    def _deduplicate(answer: str) -> str:
        return " ".join(OrderedDict.fromkeys(answer.split(" ")))

class ImageDataset(Dataset):
    def caption_path(self, path):
        return os.path.join(os.path.dirname(path), f"{os.path.basename(path).split('.')[0]}.txt")

    def caption_exists(self, path):
        caption_path = self.caption_path(path)
        return os.path.exists(caption_path)

    def __init__(self, dir, vis_processors, args):
        self.dir = dir
        self.args = args
        self.vis_processors = vis_processors

        image_paths = glob.glob(os.path.join(args.path, '**/*.*'), recursive=True)

        if args.overwrite:
            self.paths = [p for p in image_paths if
                          p.endswith(('jpg', 'jpeg', 'png', 'webp'))]
        else:
            self.paths = [p for p in image_paths if p.endswith(('jpg', 'jpeg', 'png', 'webp'))
                          and not self.caption_exists(p)]

    def __len__(self):
        return (len(self.paths))

    def __getitem__(self, idx):
        path = self.paths[idx]

        if args.resize is not None:
            self.resize_and_save(path)

        raw_image = self.resize_image(Image.open(path).convert("RGB"), 768)
        #raw_image = Image.open(path).convert("RGB")
        processed = self.vis_processors["eval"](raw_image).unsqueeze(0)
        return {"image": processed, "caption_path": self.caption_path(path)}

    @staticmethod
    def resize_and_save(path):
        img = Image.open(path)
        ImageDataset.resize_image(img, args.resize).save(path)

    @staticmethod
    def resize_image(image, target_size):
        # Get the current width and height of the image
        width, height = image.size

        # Check which dimension (width or height) is the maximum
        max_dimension = max(width, height)

        # If the image is smaller than target, abort
        if max(width, height) <= target_size:
            return image

        # Calculate the scale factor to resize the image
        scale_factor = 1
        if max_dimension > target_size:
            scale_factor = target_size / max_dimension

        # Calculate the new width and height using the scale factor
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)

        # Resize the image while maintaining the aspect ratio
        resized_image = image.resize((new_width, new_height))

        # Return the resized image
        return resized_image

    @staticmethod
    def collate_fn(batch):
        return batch


def load_workflow(file_path: str) -> Workflow:
    with open(file_path, 'r') as f:
        return Workflow.from_json(json5.load(f))


def main(args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model, vis_processors, _ = load_model_and_preprocess(name=args.model_name, model_type=args.model_type, is_eval=True, device=device)

    workflow = load_workflow(args.workflow)
    inquisitor = Inquisitor(workflow, model, args)

    dataset = ImageDataset(args.path, vis_processors, args)
    loader = DataLoader(dataset, batch_size=1, num_workers=5, collate_fn=ImageDataset.collate_fn)

    for batch in tqdm(loader):
        image, caption_path = batch[0]["image"], batch[0]["caption_path"]

        try:
            answer = inquisitor.ask(image.to(device))
            with open(caption_path, "w") as f:
                f.write(answer)
        except Exception as e:
            print(f"Failed to process {caption_path}, {str(e)}")


if __name__ == "__main__":
    args = argparse.ArgumentParser("described")
    args.add_argument("--workflow", type=str, default="./workflows/standard.json5", help="The workflow file to use")
    args.add_argument("--model_name", type=str, default="blip2_t5", help="One of: blip2_opt, blip2_t5, blip2")
    args.add_argument("--model_type", type=str, default="pretrain_flant5xl", help="A compatible model type. One of: blip2_opt(pretrain_opt2.7b, caption_coco_opt2.7b, pretrain_opt6.7b, caption_coco_opt6.7b), "
                                                                                  "blip2_t5(pretrain_flant5xl, caption_coco_flant5xl, pretrain_flant5xxl), "
                                                                                  "blip2(pretrain, coco)")
    args.add_argument("--path", type=str, required=True, help="Path to images to be captioned")
    args.add_argument("--overwrite", default=False, action="store_true", help="Overwrite existing captions")
    args.add_argument("--prefix", type=str, help="a string applied at the beginning of each caption")
    args.add_argument("--suffix", type=str, help="a string applied at the end of each caption")
    args.add_argument("--resize", type=int, help="additionally, resize and save the image where the longest side is the provided maximum ")
    args = args.parse_args()

    main(args)
