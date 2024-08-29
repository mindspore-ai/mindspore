import os
import json
import argparse
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference_path", type=str, default="./CODA-LM/Test")
    parser.add_argument("--prediction_path", type=str, default="./region_perception_answer.jsonl")
    args = parser.parse_args()

    ######################
    # load ground truth
    ######################
    reference_data = {}
    for each in tqdm(os.listdir(args.reference_path), desc='Loading GT'):
        if not each.endswith('.json'):
            continue
        with open(os.path.join(args.reference_path, each), "r", encoding='utf-8') as f:
            each_data = json.load(f)
        reference_data[each[:-5]] = each_data

    ######################
    # load prediction
    ######################
    prediction_data = [json.loads(q) for q in open(os.path.expanduser(args.prediction_path), "r", encoding='utf-8')]
    for each in tqdm(prediction_data, desc='Processing prediction'):
        image_name = each['image'].split('/')[-1]
        json_name = 'test_' + image_name.split('_')[0]
        object_id = image_name.split('_')[-1][:-4]
        assert object_id.isdigit()
        each['label_name'] = reference_data[json_name]['region_perception'][object_id]['category_name']

    ######################
    # save converted prediction
    ######################
    with open(args.prediction_path[:-6] + '_w_label.jsonl', "w", encoding='utf-8') as file:
      for each in prediction_data:
          file.write(json.dumps(each) + "\n")