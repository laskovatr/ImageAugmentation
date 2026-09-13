"""
Module for images generating
"""
import argparse
import json
import os.path

import numpy as np
import yaml
import albumentations as A
from PIL import Image

import dataset_factory
import generator_factory


def create_json_cfg(json_name, source_name, result_name,
                    positive_prompts, negative_prompts,
                    augmentations=None):
    """
    Function to create result json file
    """
    json_data = {
        "source_name":source_name,
        "result_name": result_name,
        "positive_prompts": positive_prompts,
        "negative_prompts": negative_prompts,
        "augmentations": augmentations
    }
    with open(json_name, "w", encoding="utf-8") as file:
        json.dump(json_data, file, indent=4, ensure_ascii=False)


def generate_images(inference_config, ds_config, prompts, output_path,
                    augmentation_config=None):
    """
    Generate images function
    """
    ds = dataset_factory.create_ds_from_cfg_file(ds_config)
    generator = generator_factory.create_generator_from_cfg_file(inference_config)

    with open(prompts, 'r', encoding='utf-8') as file:
        prompts_data = yaml.safe_load(file)

    augmentation_flag = False
    if augmentation_config:
        augmentation_flag = True
        with open(augmentation_config, 'r', encoding='utf-8') as aug_file:
            aug_data = json.load(aug_file)
        loaded_transform = A.load(aug_data['albumentations_config'], data_format="yaml")
        if aug_data['random_seed']:
            loaded_transform.set_random_seed(aug_data['random_seed'])

    os.makedirs(output_path, exist_ok=True)

    for cur_img in ds:
        result_image = generator.generate(cur_img['img'],
                                          prompts_data['positive'],
                                          prompts_data['negative'])
        output_name = os.path.join(output_path, cur_img['img_name'])
        result_image.save(output_name)

        json_name = output_name.rsplit('.', 1)[0] + '.json'
        source_name = os.path.join(ds.img_path, cur_img['img_name'])
        create_json_cfg(json_name, source_name, output_name,
                        prompts_data['positive'], prompts_data['negative'])

        if augmentation_flag:
            result_image_np = np.array(result_image)

            for i in range(aug_data['sample_count']):
                transformed = loaded_transform(image=result_image_np)
                transformed_image_np = transformed["image"]
                final_image = Image.fromarray(transformed_image_np)
                cur_name = (cur_img['img_name'].rsplit('.', 1)[0] + f'_{i}'
                               + '.' + cur_img['img_name'].rsplit('.', 1)[1])
                output_name = os.path.join(output_path, cur_name)
                final_image.save(output_name)

                json_name = output_name.rsplit('.', 1)[0] + '.json'
                create_json_cfg(json_name, source_name, output_name,
                                prompts_data['positive'], prompts_data['negative'],
                                aug_data['albumentations_config'])


if __name__== '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inference-config', type=str,
                        help='Path to inference config file')
    parser.add_argument('--ds-config', type=str,
                        help='Path to DS config file')
    parser.add_argument('--prompts', type=str,
                        help='Path to prompts file')
    parser.add_argument('--output-path', type=str,
                        help='Path to output folder')
    parser.add_argument('--augmentation-config', type=str, default=None,
                        help='Path to augmentation config file')

    args = parser.parse_args()

    generate_images(args.inference_config, args.ds_config, args.prompts, args.output_path,
                    args.augmentation_config)
