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
""" configure cropper tool """

from functools import lru_cache
import glob
import json
import os
import queue
import re
import shlex
import subprocess

from mindspore import log as logger

DEFINE_STR = "-DENABLE_ANDROID -DENABLE_ARM -DENABLE_ARM64 -DENABLE_NEON -DNO_DLIB -DUSE_ANDROID_LOG -DANDROID"

ASSOCIATIONS_FILENAME = 'associations.txt'
DEPENDENCIES_FILENAME = 'dependencies.txt'
ERRORS_FILENAME = 'debug.txt'
OUTPUT_LOCATION = "mindspore/lite/tools/dataset/cropper"

# needed for gcc command for include directories
MANUAL_HEADERS = [
    ".",
    "mindspore",
    "mindspore/ccsrc",
    "mindspore/ccsrc/minddata/dataset",
    "mindspore/ccsrc/minddata/dataset/kernels/image",
    "mindspore/core",
    "mindspore/lite",
]

# To stop gcc command once reaching these external headers
# (not all of them may be used now in MindData lite)
EXTERNAL_DEPS = [
    "graphengine/910/inc/external",
    "akg/third_party/fwkacllib/inc",
    "third_party",
    "third_party/securec/include",
    "build/mindspore/_deps/sqlite-src",
    "build/mindspore/_deps/pybind11-src/include",
    "build/mindspore/_deps/tinyxml2-src",
    "build/mindspore/_deps/jpeg_turbo-src",
    "build/mindspore/_deps/jpeg_turbo-src/_build",
    "build/mindspore/_deps/icu4c-src/icu4c/source/i18n",
    "build/mindspore/_deps/icu4c-src/icu4c/source/common",
    "mindspore/lite/build/_deps/tinyxml2-src",
    "mindspore/lite/build/_deps/jpeg_turbo-src",
    "mindspore/lite/build/_deps/jpeg_turbo-src/_build",
    "mindspore/lite/build/_deps/nlohmann_json-src",
]

# API files which the corresponding objects and all objects for their dependencies must always be included.
ESSENTIAL_FILES_1 = [
    "api/data_helper.cc",
    "api/datasets.cc",
    "api/execute.cc",
    "api/iterator.cc",
]

# API files which the corresponding objects must always be included.
# (corresponding IR files will be included according to user ops)
ESSENTIAL_FILES_2 = [
    "api/text.cc",
    "api/transforms.cc",
    "api/samplers.cc",
    "api/vision.cc",
]

DATASET_PATH = "mindspore/ccsrc/minddata/dataset"

OPS_DIRS = [
    "engine/ir/datasetops",
    "engine/ir/datasetops/source",
    "engine/ir/datasetops/source/samplers",
    "kernels/ir/vision",
    "kernels/ir/data",
    "text/ir/kernels",
]


def extract_classname_samplers(header_content):
    """
    Use regex to find class names in header files of samplers

    :param header_content: string containing header of a sampler IR file
    :return: list of sampler classes found
    """
    return re.findall(r"(?<=class )[\w\d_]+(?=Obj : )", header_content)


def extract_classname_source_node(header_content):
    """
    Use regex to find class names in header files of source nodes

    :param header_content: string containing header of a source node IR file
    :return: list of source node classes found
    """
    return re.findall(r"(?<=class )[\w\d_]+(?=Node : )", header_content)


def extract_classname_nonsource_node(header_content):
    """
    Use regex to find class names in header files of non-source nodes

    :param header_content: string containing header of a non-source IR file
    :return: list of non-source node classes found
    """
    return re.findall(r"(?<=class )[\w\d_]+(?=Node : )", header_content)


def extract_classname_vision(header_content):
    """
    Use regex to find class names in header files of vision ops

    :param header_content: string containing header of a vision op IR file
    :return: list of vision ops found
    """
    return re.findall(r"(?<=class )[\w\d_]+(?=Operation : )", header_content)


def extract_classname_data(header_content):
    """
    Use regex to find class names in header files of data ops

    :param header_content: string containing header of a data op IR file
    :return: list of data ops found
    """
    return re.findall(r"(?<=class )[\w\d_]+(?=Operation : )", header_content)


def extract_classname_text(header_content):
    """
    Use regex to find class names in header files of text ops

    :param header_content: string containing header of a text op IR file
    :return: list of text ops found
    """
    return re.findall(r"(?<=class )[\w\d_]+(?=Operation : )", header_content)


# For each op type (directory) store the corresponding function which extracts op name
registered_functions = {
    os.path.join(DATASET_PATH, 'engine/ir/datasetops/source/samplers'): extract_classname_samplers,
    os.path.join(DATASET_PATH, 'engine/ir/datasetops/source'): extract_classname_source_node,
    os.path.join(DATASET_PATH, 'engine/ir/datasetops'): extract_classname_nonsource_node,
    os.path.join(DATASET_PATH, 'kernels/ir/vision'): extract_classname_vision,
    os.path.join(DATASET_PATH, 'kernels/ir/data'): extract_classname_data,
    os.path.join(DATASET_PATH, 'text/ir/kernels'): extract_classname_text,
}


def get_headers():
    """
    Get the headers flag: "-Ixx/yy -Ixx/zz ..."

    :return: a string to be passed to compiler
    """
    headers_paths = MANUAL_HEADERS + EXTERNAL_DEPS

    output = "-I{}/".format("/ -I".join(headers_paths))

    return output


@lru_cache(maxsize=1024)
def get_dependencies_of_file(headers_flag, filename):
    """
    Create dependency list for a file (file0.cc):
    file0.cc.o: file1.h, file2.h, ...

    :param headers_flag: string containing headers include paths with -I prepended to them.
    :param filename: a string containing path of a file.
    :return: a list of file names [file0.cc, file1.h, file2.h, file3.h] and error string
    """
    command = 'gcc -MM -MG {0} {1} {2}'.format(filename, DEFINE_STR, headers_flag)
    command_split = shlex.split(command)
    stdout, stderr = subprocess.Popen(command_split, shell=False, stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE).communicate()
    deps = re.split(r'[\s\\]+', stdout.decode('utf-8').strip(), flags=re.MULTILINE)[1:]

    return deps, stderr.decode('utf-8')


def needs_processing(dep_cc, processed_cc, queue_cc_set):
    """
    Determine if a file's dependencies need to be processed.

    :param dep_cc: the candidate file to be processed by gcc
    :param processed_cc: set of files that have been already processed.
    :param queue_cc_set: files currently in the queue (to be processed)
    :return: boolean, whether the file should be further processed by gcc.
    """
    # don't add the file to the queue if already processed
    if dep_cc in processed_cc:
        return False
    # don't add the file to the queue if it is already there
    if dep_cc in queue_cc_set:
        return False
    # if file doesn't exist, don't process as it will cause error (may happen for cache)
    if not os.path.isfile(dep_cc):
        return False
    return True


def build_source_file_path(dep_h):
    """
    Given the path to a header file, find the path for the associated source file.
    - if an external dependency, return "EXTERNAL"
    - if not found, keep the header file's path

    :param dep_h: a string containing path to the header file
    :return: dep_cc: a string containing path to the source file
    """
    for x in EXTERNAL_DEPS:
        if x in dep_h:
            dep_cc = "EXTERNAL"
            return dep_cc
    if 'include/api/types.h' in dep_h:
        dep_cc = "mindspore/ccsrc/cxx_api/types.cc"
        return dep_cc
    dep_cc = dep_h.replace('.hpp', '.cc').replace('.h', '.cc')
    if not os.path.isfile(dep_cc):
        dep_cc = dep_h
    return dep_cc


def get_all_dependencies_of_file(headers_flag, filename):
    """
    Create dependency list for a file (incl. all source files needed).

    :param headers_flag: string containing headers include paths with -I prepended to them.
    :param filename: a string containing path of a file.
    :return: all dependencies of that file and the error string
    """
    errors = []
    # a queue to process files
    queue_cc = queue.SimpleQueue()
    # a set of items that have ever been in queue_cc (faster access time)
    queue_cc_set = set()
    # store processed files
    processed_cc = set()

    # add the source file to the queue
    queue_cc.put(filename)
    queue_cc_set.add(filename)

    while not queue_cc.empty():
        # process the first item in the queue
        curr_cc = queue_cc.get()
        deps, error = get_dependencies_of_file(headers_flag, curr_cc)
        errors.append(error)
        processed_cc.add(curr_cc)
        # prepare its dependencies for processing
        for dep_h in deps:
            dep_cc = build_source_file_path(dep_h)
            # ignore if marked as an external dependency
            if dep_cc == "EXTERNAL":
                processed_cc.add(dep_h)
                continue
            # add to queue if needs processing
            if needs_processing(dep_cc, processed_cc, queue_cc_set):
                queue_cc.put(dep_cc)
                queue_cc_set.add(dep_cc)
    logger.debug('file: {} | deps: {}'.format(os.path.basename(filename), len(processed_cc)))

    return list(processed_cc), "".join(errors)


def get_deps_essential(headers_flag):
    """
    Return dependencies required for any run (essential).

    :param headers_flag: string containing headers include paths with -I prepended to them.
    :return: a list of essential files, and the error string
    """
    essentials = []
    errors = []

    # find dependencies for ESSENTIAL_FILES_1 as we need them too.
    for filename in [os.path.join(DATASET_PATH, x) for x in ESSENTIAL_FILES_1]:
        deps, err = get_all_dependencies_of_file(headers_flag, filename)
        errors.append(err)
        essentials.extend(deps)
        essentials.append(filename)
    # we only need ESSENTIAL_FILES_2 themselves (IR files are split)
    for filename in [os.path.join(DATASET_PATH, x) for x in ESSENTIAL_FILES_2]:
        essentials.append(filename)
    essentials = list(set(essentials))

    return essentials, "".join(errors)


def get_deps_non_essential(headers_flag):
    """
    Find the entry points (IR Level) for each op and write them in associations dict.
    Starting from these entry point, recursively find the dependencies for each file and write in a dict.

    :param headers_flag: string containing headers include paths with -I prepended to them.
    :return: dependencies dict, associations dict, the error string
    """
    dependencies = dict()  # what files each file imports
    associations = dict()  # what file each op is defined in (IR level)
    errors = []
    for dirname in [os.path.join(DATASET_PATH, x) for x in OPS_DIRS]:
        # Get the proper regex function for this directory
        if dirname not in registered_functions:
            raise ValueError("Directory has no registered regex function:", dirname)
        extract_classname = registered_functions[dirname]
        # iterate over source files in the directory
        for src_filename in glob.glob("{}/*.cc".format(dirname)):
            # get the dependencies of source file
            deps, err = get_all_dependencies_of_file(headers_flag, src_filename)
            dependencies[src_filename] = deps
            errors.append(err)
            # locate the corresponding header file and read it
            header_filename = src_filename.replace('.cc', '.h')
            if not os.path.isfile(header_filename):
                raise ValueError("Header file doesn't exist!")
            with open(header_filename, 'r') as f:
                content = f.read().strip()
            # extract ops from header file
            ops = extract_classname(content)
            # add the op to associations table
            for raw_op in ops:
                op = raw_op.lower().replace('_', '')
                associations[op] = src_filename
    return dependencies, associations, "".join(errors)


def main():
    """
    Configure the cropper tool by creating  associations.txt and dependencies.txt
    """
    errors = ""
    dependencies = {}

    # convert to a single string with '-I' prepended to each dir name
    headers_flag = get_headers()

    # get dependencies for essential files
    all_deps, err = get_deps_essential(headers_flag)
    dependencies['ESSENTIAL'] = all_deps
    errors += err
    logger.debug('len(ESSENTIAL): {}'.format(len(dependencies['ESSENTIAL'])))

    # get dependencies for other files (non-essentials)
    other_dependencies, all_associations, err = get_deps_non_essential(headers_flag)
    dependencies.update(other_dependencies)
    errors += err

    with os.fdopen(os.open(os.path.join(OUTPUT_LOCATION, DEPENDENCIES_FILENAME), os.O_WRONLY | os.O_CREAT, 0o660),
                   "w+") as f:
        json.dump(dependencies, f)

    with os.fdopen(os.open(os.path.join(OUTPUT_LOCATION, ASSOCIATIONS_FILENAME), os.O_WRONLY | os.O_CREAT, 0o660),
                   "w+") as f:
        json.dump(all_associations, f)

    with os.fdopen(os.open(os.path.join(OUTPUT_LOCATION, ERRORS_FILENAME), os.O_WRONLY | os.O_CREAT, 0o660), "w+") as f:
        f.write(errors)


if __name__ == "__main__":

    logger.info('STARTING: cropper_configure.py ')

    original_path = os.getcwd()
    script_path = os.path.dirname(os.path.abspath(__file__))

    try:
        # change directory to mindspore directory
        os.chdir(os.path.join(script_path, "../../../../.."))
        main()
    except (OSError, IndexError, KeyError):
        logger.critical('FAILED: cropper_configure.py!')
        raise
    else:
        logger.info('SUCCESS: cropper_configure.py ')
    finally:
        os.chdir(original_path)
