# Copyright 2025 Huawei Technologies Co., Ltd
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
"""
Interfaces for optimize module.
"""

from mindspore import log as logging
from mindspore.nn import Cell


def process_cell_init_args(cells, reusing_count, allow_list, denny_list):
    """
    process all the cells to use lazy inline with the conditions..

    Args:
        cells(Object map): All the cells information.
        reusing_count(int): The count of the same key of the cell instance.
        allow_list:(list):  The allow list of the cell class to apply lazy inline
        denny_list:(list):  The denny list of the cell class to apply lazy inline
    Returns:
        void
    """
    type_instance = {}
    class_name = {}

    for k in cells.keyrefs():
        v = cells[k]
        if k.construct.__code__.co_filename.find("mindspore/nn") != -1:
            continue
        if denny_list and any(deny in k.cls_name for deny in denny_list):
            continue
        if allow_list and not any(allow in k.cls_name for allow in allow_list):
            continue

        pickle_args = k.cls_name + str(id(k.__class__)) + "[" + str(v[1]) + ":kws:" + str(v[2]) + "]"

        instances = type_instance.get(pickle_args)
        if instances is not None:
            instances.append(k)
            if len(instances) > reusing_count:
                if not hasattr(k, "cell_init_args"):
                    setattr(k, "cell_init_args", "lazy_inline_" + pickle_args)
                    logging.debug("Reusing cell info: %s , id: %s , args: %s",
                                  k.construct.__code__.co_filename + "/" + k.cls_name, id(k), pickle_args)
                    setattr(k, "no_inline", False)
            elif len(instances) == reusing_count:
                class_name[v[0]] = k.construct.__code__.co_filename + "/" + k.cls_name
                logging.info("Reusing Cell: %s , args: %s", k.construct.__code__.co_filename + "/" + k.cls_name,
                             pickle_args)

                for i in instances:
                    if not hasattr(i, "cell_init_args"):
                        setattr(i, "cell_init_args", "lazy_inline_" + pickle_args)
                        logging.debug("Reusing cell info: %s , id: %s , args: %s",
                                      i.construct.__code__.co_filename + "/" + i.cls_name, id(i), pickle_args)
                        setattr(i, "no_inline", False)
        else:
            type_instance[pickle_args] = [k]

    return class_name


def set_lazy_inline(reusing_count=3, allow_list=None, denny_list=None):
    """
    Apply all the cells to use lazy inline with the conditions.

    Args:
        cells(Object map): All the cells information.
        reusing_count(int): The count of the same key of the cell instance.
        allow_list:(list):  The allow list of the cell class to apply lazy inline
        denny_list:(list):  The denny list of the cell class to apply lazy inline
    Returns:
        void
    """
    cells = Cell.global_cells
    if denny_list is None:
        denny_list = []
    if allow_list is None:
        allow_list = []

    denny_list.append("_Output")
    denny_list.append("_MicroBatch")
    reusing_cells = process_cell_init_args(cells, reusing_count, allow_list, denny_list)
    return reusing_cells
