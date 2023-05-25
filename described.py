import argparse
import glob
import os
import json5
from functools import partial

import torch
from PIL import Image
from lavis.models import load_model_and_preprocess
from tqdm import tqdm


class Inquisitor():
    def __init__(self, workflow, model, args):
        self.workflow = workflow
        self.model = model
        self.args = args

    def ask(self, image):
        return self.__ask(image, [], self.workflow)

    def __ask(self, image, context, node):
        if isinstance(node, list):
            return ' '.join(map(partial(self.__ask, image, context), node))

        template = "Question: {}, Answer: {}"
        question = node["q"]
        prompt = ", ".join([template.format(context[i][0], context[i][1]) for i in
                            range(len(context))]) + " Question: " + question + " Answer:"
        # prompt = " Question: " + question + " Answer:"
        answer = self.model.generate({"image": image, "prompt": prompt})[0].lower()
        context = context + [[question, answer]]
        # print(f"q: {prompt}, a: {answer}")

        if "branches" in node:
            if answer in node["branches"]:
                branch = node["branches"][answer]
                if isinstance(branch, list):
                    join_str = node.get("join", ' ')
                    res = join_str.join(
                        filter(bool, map(lambda x: x.strip(), map(partial(self.__ask, image, context), branch))))
                    if "template" in node:
                        return node["template"].format(res)
                    else:
                        return res
                else:
                    return self.__ask(image, context, branch)
            elif "default" in node["branches"]:
                return node["branches"]["default"].format(answer)
            else:
                raise Exception(f"Missing branch for {answer}")
        else:
            if "template" in node:
                return node["template"].format(answer)
            else:
                return answer

def main(args):
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    model, vis_processors, _ = load_model_and_preprocess(name=args.model_name, model_type=args.model_type, is_eval=True,
                                                         device=device)

    with open(args.workflow, 'r') as f:
        workflow = json5.load(f)

    inquisitor = Inquisitor(workflow, model, args)

    glob_pattern = os.path.join(args.path, '**/*.*')
    paths = glob.glob(glob_pattern, recursive=True)
    paths = [p for p in paths if p.endswith(('jpg', 'jpeg', 'png', 'webp'))]
    for path in tqdm(paths):
        file = os.path.basename(path)
        file_name = file.split(".")[0]
        caption_path = os.path.join(os.path.dirname(path), f"{file_name}.txt")

        if os.path.exists(caption_path):
            print(f"Skipping, caption exists for {path}")
            continue

        raw_image = Image.open(path).convert("RGB")
        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

        try:
            answer = inquisitor.ask(image)
            with open(caption_path, "w") as f:
                f.write(answer)
        except Exception as e:
            print(f"Failed to process {path}, {e}")


if __name__ == "__main__":
    args = argparse.ArgumentParser("described")
    args.add_argument("--workflow", type=str, default="./workflows/standard.json5", help="The workflow file to use")
    args.add_argument("--model_name", type=str, default="blip2_t5", help="One of: blip2_opt, blip2_t5, blip2")
    args.add_argument("--model_type", type=str, default="pretrain_flant5xl", help="A compatible model type. One of: blip2_opt(pretrain_opt2.7b, caption_coco_opt2.7b, pretrain_opt6.7b, caption_coco_opt6.7b), "
                                                                                  "blip2_t5(pretrain_flant5xl, caption_coco_flant5xl, pretrain_flant5xxl), "
                                                                                  "blip2(pretrain, coco)")
    args.add_argument("--path", type=str, required=True, help="Path to images to be captioned")
    args = args.parse_args()

    main(args)

