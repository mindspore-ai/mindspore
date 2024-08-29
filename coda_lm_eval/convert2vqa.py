from PIL import Image, ImageDraw
import os
import numpy as np
import json
import argparse
from tqdm import tqdm
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--coda_root", type=str, default="./")
    parser.add_argument("--codalm_ann_name", type=str, default="CODA-LM", choices=["CODA-LM", "CODA-LM-chinese"])
    args = parser.parse_args()

    # user notice for directory structure
    print("=======================\nBefore started, please make sure your directory has been organized as in https://github.com/DLUT-LYZ/CODA-LM?tab=readme-ov-file#data-preparation\n=======================\n")
    
    ########################
    # Start pre-processing
    ########################
    ann_root = os.path.join(args.coda_root, args.codalm_ann_name)
    for split in os.listdir(ann_root):
        assert split in ['Train', 'Val', 'Test', 'Mini']
        split_root = os.path.join(ann_root, split)
        json_list = sorted([each for each in os.listdir(split_root) if each.endswith('.json')])

        stage1_index = -1
        stage2_index = -1
        stage3_index = -1
        stage1_all_data = []
        stage2_all_data = []
        stage3_all_data = []
        for json_name in tqdm(json_list, desc=split):
            with open(os.path.join(split_root, json_name), 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            img_dir = json_name.split("_")[0]
            img_name = json_name.split("_")[1][:-5] + '.jpg'

            ########################
            # Step 1: Prepare general perception data
            ########################
            stage1_index += 1
            stage1_data = dict(
                question_id=stage1_index,
                image=os.path.join(img_dir, 'images', img_name),
                question="There is an image of traffic captured from the perspective of the ego car. Focus on objects influencing the ego car's driving behavior: vehicles (cars, trucks, buses, etc.), vulnerable road users (pedestrians, cyclists, motorcyclists), traffic signs (no parking, warning, directional, etc.), traffic lights (red, green, yellow), traffic cones, barriers, miscellaneous(debris, dustbin, animals, etc.). You must not discuss any objects beyond the seven categories above. Please describe each object's appearance, position, direction, and explain why it affects the ego car's behavior."
            )
            if split != 'Test':
                stage1_data['answer'] = json_data['general_perception']['description and explanation']
            stage1_all_data.append(stage1_data)

            ########################
            # Step 2: Prepare driving suggestion data
            ########################
            stage2_index += 1
            stage2_data = dict(
                question_id=stage2_index,
                    image=os.path.join(img_dir, 'images', img_name),
                    question="There is an image of traffic captured from the perspective of the ego car. Focus on objects influencing the ego car's driving behavior: vehicles (cars, trucks, buses, etc.), vulnerable road users (pedestrians, cyclists, motorcyclists), traffic signs (no parking, warning, directional, etc.), traffic lights (red, green, yellow), traffic cones, barriers, miscellaneous(debris, dustbin, animals, etc.). You must not discuss any objects beyond the seven categories above. Please provide driving suggestions for the ego car based on the current scene."
            )
            if split != 'Test':
                stage2_data['answer'] = json_data['driving_suggestion']
            stage2_all_data.append(stage2_data)

            ########################
            # Step 3: Prepare region perception data
            ########################
            regional_data = json_data["region_perception"]
            for key, value in regional_data.items():
                # preprare image paths
                output_root = os.path.join(args.coda_root, img_dir, 'images_w_boxes')
                output_path = os.path.join(output_root, "{}_object_{}.jpg".format(json_name.split("_")[1][:-5], key))
                os.makedirs(output_root, exist_ok=True)
                
                # prepare images
                if not os.path.exists(output_path):
                    img = Image.open(os.path.join(args.coda_root, img_dir, 'images', img_name))
                    draw = ImageDraw.Draw(img)
                    rect = [value['box'][0], value['box'][1], value['box'][0] + value['box'][2], value['box'][1] + value['box'][3]]
                    draw.rectangle(rect, outline="red", width=2)
                    img.save(output_path)
                
                # prepare annotation
                stage3_index += 1
                stage3_data = dict(
                    question_id=stage3_index,
                    image=os.path.join(img_dir, 'images_w_boxes', "{}_object_{}.jpg".format(json_name.split("_")[1][:-5], key)),
                    question="Please describe the object inside the red rectangle in the image and explain why it affect ego car driving."
                )
                if split != 'Test':
                    stage3_data['answer'] = value['description and explanation']
                stage3_all_data.append(stage3_data)
        
        ########################
        # Step 4: Save annotation
        ########################
        save_root = os.path.join(split_root, 'vqa_anno')
        os.makedirs(save_root, exist_ok=True)

        # save stage1
        with open(os.path.join(save_root, 'general_perception.jsonl'), 'w') as file:
            for entry in stage1_all_data:
                json_str = json.dumps(entry)
                file.write(json_str + '\n')

        # save stage2
        with open(os.path.join(save_root, 'driving_suggestion.jsonl'), 'w') as file:
            for entry in stage2_all_data:
                json_str = json.dumps(entry)
                file.write(json_str + '\n')

        # save stage3
        with open(os.path.join(save_root, 'region_perception.jsonl'), 'w') as file:
            for entry in stage3_all_data:
                json_str = json.dumps(entry)
                file.write(json_str + '\n')
