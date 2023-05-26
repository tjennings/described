import argparse
import glob
import os
import json5
import torch
from PIL import Image
from lavis.models import load_model_and_preprocess
from tqdm import tqdm
from typing import List, Any, Tuple, Union, Dict
from collections import OrderedDict


class Node:
    def __init__(self, data: Dict[str, Any]):
        self.data = data

    def get_template(self) -> str:
        return self.data.get("template", "{}")

    def get_question(self) -> str:
        return self.data["q"]

    def get_branches(self) -> Dict[str, Any]:
        return self.data.get("branches", {})


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
        return ' '.join(responses)

    def _process_node(self, image: torch.Tensor, context: List[Tuple[str, str]], node: Node) -> str:
        answer, context = self._query(image, context, node)

        branches = node.get_branches()
        if answer in branches:
            branch = branches[answer]
            if isinstance(branch, list):
                return self._traverse_workflow(image, context, [Node(b) for b in branch])
            else:
                return self._process_node(image, context, Node(branch))
        else:
            return node.get_template().format(answer)

    def _query(self, image: torch.Tensor, context: List[Tuple[str, str]], node: Node) -> Tuple[str, List[Tuple[str, str]]]:
        template = "Question: {}, Answer: {}"
        question = node.get_question()
        prompt = ", ".join([template.format(q, a) for q, a in context]) + f" Question: {question} Answer:"
        answer = self.model.generate({"image": image, "prompt": prompt})[0].lower()
        answer = self._deduplicate(answer)
        context.append((question, answer))
        return answer, context

    @staticmethod
    def _deduplicate(answer: str) -> str:
        return " ".join(OrderedDict.fromkeys(answer.split(" ")))


def load_workflow(file_path: str) -> Workflow:
    with open(file_path, 'r') as f:
        return Workflow.from_json(json5.load(f))


def main(args):
    device = torch.device("cuda") if torch.is_available() else torch.device("cpu")
    model, vis_processors, _ = load_model_and_preprocess(name=args.model_name, model_type=args.model_type, is_eval=True, device=device)

    workflow = load_workflow(args.workflow)
    inquisitor = Inquisitor(workflow, model, args)

    image_paths = glob.glob(os.path.join(args.path, '**/*.*'), recursive=True)
    image_paths = [p for p in image_paths if p.endswith(('jpg', 'jpeg', 'png', 'webp'))]

    for path in tqdm(image_paths):
        caption_path = os.path.join(os.path.dirname(path), f"{os.path.basename(path).split('.')[0]}.txt")

        if os.path.exists(caption_path):
            print(f"Skipping, caption exists for {path}")
            continue

        try:
            raw_image = Image.open(path).convert("RGB")
            image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

            answer = inquisitor.ask(image)
            with open(caption_path, "w") as f:
                f.write(answer)
        except Exception as e:
            print(f"Failed to process {path}, {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("described")
    parser.add_argument("--workflow", type=str, default="./workflows/standard.json5", help="The workflow file to use")
    parser.add_argument("--model_name", type=str, default="blip2_t5", help="One of: blip2_opt, blip2_t5, blip2")
    parser.add_argument("--model_type", type=str, default="pretrain_flant5xl", help="A compatible model type.")
    parser.add_argument("--path", type=str, required=True, help="Path to images to be captioned")
    parser.add_argument("--prefix", type=str, default="", help="a string applied at the beginning of each caption")
    parser.add_argument("--suffix", type=str, default="", help="a string applied at the end of each caption")

    args = parser.parse_args()
    main(args)
