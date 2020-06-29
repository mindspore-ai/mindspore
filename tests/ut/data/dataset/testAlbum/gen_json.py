import json
import os

def dump_json_from_dict(structure, file_name):
    with open(file_name + '.json', 'w') as file_path:
        json.dump(structure, file_path)

if __name__ == '__main__':
    # iterate over directory
    DIRECTORY = "imagefolder"
    i = 0
    for filename in os.listdir(DIRECTORY):
        default_dict = {}
        default_dict.update(dataset='')
        default_dict.update(image=(os.path.join(DIRECTORY, filename)))
        default_dict.update(label=[1, 2])
        default_dict.update(_priority=0.8)
        default_dict.update(_embedding='sample.bin')
        default_dict.update(_segmented_image=(os.path.join(DIRECTORY, filename)))
        default_dict.update(_processed_image=(os.path.join(DIRECTORY, filename)))
        i = i + 1
        dump_json_from_dict(default_dict, 'images/'+str(i))
