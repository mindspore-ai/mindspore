# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
""" build MindData lite minimum library """

import glob
import itertools
import json
from operator import itemgetter
import os
from pprint import pprint
import sys
import warnings

import parser

DEPENDENCY_FILENAME = 'dependencies.txt'
ASSOCIATION_FILENAME = 'associations.txt'
ALL_DEPS_FILENAME = 'needed_dependencies.txt'
OBJECTS_DIR = 'tmp/'

ESSENTIAL_OBJECTS = [
    # 'types.cc.o',
    # 'tensor_impl.cc.o',
    'random_sampler.cc.o',  # default value for datasets (may not exist in their code)
    'random_sampler_ir.cc.o',  # default value for datasets (may not exist in their code)
]


def load_dependencies():
    """
    Read dependencies.txt and load it into a dict.

    :return: a dict containing list of dependencies for almost any file in MindData lite
    """
    if not os.path.isfile(DEPENDENCY_FILENAME):
        raise FileNotFoundError("dependency file ({}) does not exist.\n"
                                "Please run cropper_configure.py first.".format(DEPENDENCY_FILENAME))
    with open(DEPENDENCY_FILENAME) as f:
        dep_dict = json.load(f)
    return dep_dict


def load_associations():
    """
    Read associations.txt and load it into a dict.

    :return: a dict containing entry point (a filename) for each op
    """
    if not os.path.isfile(ASSOCIATION_FILENAME):
        raise FileNotFoundError("association file ({}) does not exist.\n"
                                "Please run cropper_configure.py first.".format(ASSOCIATION_FILENAME))
    with open(ASSOCIATION_FILENAME) as f:
        _dict = json.load(f)
    return _dict


def get_unique_dependencies(dependencies_dict, associations_dict, user_ops):
    """
    Find which dependencies we need to include according to the ops found in the user code.

    :param dependencies_dict: a dict containing list of dependencies for almost any file in MindData lite
    :param associations_dict: a dcit containing entry point (a filename) for each op
    :param user_ops: a list of ops found in the user code
    :return: a list of dependencies needed based on the user code
    """
    selected_entries = []  # itemgetter(*user_ops)(associations_dict)
    for op in user_ops:
        print('{} --> {}'.format(op, associations_dict[op]))
        selected_entries.append(associations_dict[op])
    selected_files = itemgetter(*selected_entries)(dependencies_dict)
    selected_files = list(itertools.chain(*selected_files))
    return sorted(list(set().union(selected_files)))


def remove_unused_objects(final_deps, essentials, all_object_files):
    """
    Remove object files that are determined to be NOT needed to run user code
    as they are not in the dependencies of user code.

    :param final_deps: a list of dependencies needed based on the user code
    :param essentials: essential objects that should not be removed from final lib
    :param all_object_files: a lsit of all objects available in our static library
    :return: None
    """
    # find objects which are not part of any dependency (lstrip is needed for remove '_' added in crop.sh)
    to_be_removed = [x for x in all_object_files if not any(x.lstrip('_')[:-5] in y for y in final_deps)]
    # keep the ones that are not an essential object file. (lstrip is needed for remove '_' added in crop.sh)
    to_be_removed = [x for x in to_be_removed if not any(x.lstrip('_') in y for y in essentials)]

    print('Removing:', len(to_be_removed), 'unused objects.')
    pprint(sorted(to_be_removed))
    for filename in to_be_removed:
        os.remove(os.path.join(OBJECTS_DIR, filename))


def main():
    # load tables created using cropper.py
    dependencies_dict = load_dependencies()
    associations_dict = load_associations()

    # get all objects filename
    all_object_files = [os.path.basename(x) for x in glob.glob('{}*.o'.format(OBJECTS_DIR))]
    print("All Obj files: {}".format(len(all_object_files)))

    # find ops in user code
    my_parser = parser.SimpleParser()
    temp = [my_parser.parse(x) for x in user_code_filenames]
    user_ops = set(itertools.chain(*temp))
    print('user ops: {}'.format(user_ops))

    # user is not using any MindData op
    if not user_ops:
        warnings.warn('No MindData Ops detected in your code...')
        remove_unused_objects([], [], all_object_files)
        with os.fdopen(os.open(os.path.join(OBJECTS_DIR, ALL_DEPS_FILENAME), os.O_WRONLY | os.O_CREAT, 0o660),
                       "w+") as _:
            pass
        exit(0)

    # find dependencies required (based on user ops)
    unique_deps = get_unique_dependencies(dependencies_dict, associations_dict, user_ops)
    print('Unique Deps (.h): {}'.format(len(unique_deps)))
    print('Unique Deps (.cc): {}'.format(len(list(filter(lambda x: x[-2:] == 'cc', unique_deps)))))

    # add essential files to dependency files
    final_deps = set(unique_deps + dependencies_dict['ESSENTIAL'])
    print('Total Deps (.h): {}'.format(len(final_deps)))

    # delete the rest of the object files from directory.
    remove_unused_objects(final_deps, ESSENTIAL_OBJECTS, all_object_files)

    # write all dependencies to the file (for extracting external ones)
    with os.fdopen(os.open(os.path.join(OBJECTS_DIR, ALL_DEPS_FILENAME), os.O_WRONLY | os.O_CREAT, 0o660),
                   "w+") as fout:
        fout.write("\n".join(unique_deps) + '\n')


if __name__ == "__main__":
    # get user code filename(s) as argument(s) to code
    if len(sys.argv) <= 1:
        print("usage: python build_lib.py <xxx.y> [<xxx.z>]")
        exit(1)
    user_code_filenames = sys.argv[1:]
    main()
