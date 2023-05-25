# Dscribed 

Described is a simple DAG workflow model for creating more detailed image captions built on [LAVIS/blip2](https://github.com/salesforce/LAVIS/tree/main/projects/blip2).
This is a useful tool I created to generate improved captions for my stable diffusion fine-tuning efforts.  

# Installation

NOTE:  The blip2 models are VERY large, ensure you have at least 60gb of free disk space on your root drive.
Huggingface will be default store the models in ~/.cache/huggingface.

```python -m venv venv
source ./venv/bin/activate
pip install -r requirements.txt
```

# Usage

```
usage: described [-h] [--workflow WORKFLOW] [--model_name MODEL_NAME] [--model_type MODEL_TYPE] --path PATH

options:
  -h, --help            show this help message and exit
  --workflow WORKFLOW   The workflow file to use
  --model_name MODEL_NAME
                        One of: blip2_opt, blip2_t5, blip2
  --model_type MODEL_TYPE
                        A compatible model type. One of: blip2_opt(pretrain_opt2.7b, caption_coco_opt2.7b, pretrain_opt6.7b, caption_coco_opt6.7b), blip2_t5(pretrain_flant5xl,
                        caption_coco_flant5xl, pretrain_flant5xxl), blip2(pretrain, coco)
  --path PATH           Path to images to be captioned
```


To use the standard workflow and model (blip2_t5/pretrain_flant5xl) simply provide the image path: 

`python described.py --path /path/to/my/images`

Captions are saved in the same path and with the same name as the source image with a .txt extension.  
If a caption already exists for an image, it will be skipped.  

# Workflows

The core idea behind described are workflows.  Workflows are defined in json and describe a line of 
questioning that will eventually result in a caption.  Unfortunately, we are still limited by the capabilities of available models, 
however you should expect captions that are generally superior to blip/blip2 single-question captions.

See [The Standard Workflow](https://github.com/tjennings/described/blob/main/workflows/standard.json5) for an example of usage.

# Contributing

These are early days and I would be very happy to have you help expanding the capabilities of described! 
Most importantly, we need more comprehensive workflows and these can be built by anyone, regardless of technical skills,
with a bit of patience.  To contribute fork this repository and send me a pull request with your changes.  