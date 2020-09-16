import json
import os

def dump_json_from_dict(structure, file_name):
    with open(file_name + '.json', 'w') as fp:
        json.dump(structure, fp)

if __name__ == '__main__':
    # iterate over DIRECTORY
    DIRECTORY = os.path.dirname(os.path.realpath(__file__)) + "/original"
    PARENT_DIR = os.path.dirname(DIRECTORY)
    i = -1
    for filename in os.listdir(DIRECTORY):
        default_dict = {}
        default_dict.update(dataset='')
        default_dict.update(image=os.path.abspath(os.path.join(DIRECTORY, filename)))
        default_dict.update(label=['3', '2'])
        default_dict.update(_priority=[0.8, 0.3])
        default_dict.update(_embedding=os.path.abspath(os.path.join(PARENT_DIR, 'sample.bin')))
        default_dict.update(_processed_image=os.path.abspath(os.path.join(DIRECTORY, filename)))
        i = i + 1
        dump_json_from_dict(default_dict, PARENT_DIR + '/images/'+str(i))
