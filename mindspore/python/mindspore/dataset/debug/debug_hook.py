# Copyright 2023 Huawei Technologies Co., Ltd
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
# ==============================================================================
"""
This module defines the class for minddata pipeline debugger.
class DebugHook is not exposed to users as an external API.
"""

from abc import ABC, abstractmethod


class DebugHook(ABC):
    """
    The base class for Dataset Pipeline Python Debugger hook. All user defined hook behaviors
    must inherit this base class.

    To debug the input and output data of `map` operation in dataset pipeline, users can add
    breakpoint in `compute` method, or print types and shapes of the data.

    Args:
        prev_op_name (str, optional): name of the operation before current debugging point. Default: ``None``.

    Examples:
        >>> import mindspore.dataset as ds
        >>> import mindspore.dataset.debug as debug
        >>>
        >>> class CustomizedHook(debug.DebugHook):
        ...     def __init__(self):
        ...         super().__init__()
        ...
        ...     def compute(self, *args):
        ...         import pdb
        ...         pdb.set_trace()
        ...         print("Data after decode", *args)
        ...         return args
        >>>
        >>> # Enable debug mode
        >>> ds.config.set_debug_mode(True, debug_hook_list=[CustomizedHook()])
        >>>
        >>> # Define dataset pipeline
        >>> dataset = ds.ImageFolderDataset(dataset_dir="/path/to/image_folder_dataset_directory")
        >>> # Insert debug hook after `Decode` operation.
        >>> dataset = dataset.map([vision.Decode(), CustomizedHook(), vision.CenterCrop(100)])
    """
    def __init__(self, prev_op_name=None):
        self.prev_op_name = prev_op_name
        self.is_first_op = None

    def __call__(self, *args):
        # If insert debug function into map, like [Decode(), debug_fun(), Resize],
        # the debug_fun does not have self.prev_op_name, so skip the common print.
        if not self.prev_op_name:
            pass
        else:
            # log op name
            if self.is_first_op:
                log_message = "[Dataset debugger] Print the [INPUT] of the operation [{}].".format(self.prev_op_name)
            else:
                log_message = "[Dataset debugger] Print the [OUTPUT] of the operation [{}].".format(self.prev_op_name)
            print(log_message, flush=True)

        ######################## NOTE ########################
        # Add a breakpoint to the following line to inspect
        # input and output of each transform.
        ######################################################
        self.compute(args)
        return args

    @abstractmethod
    def compute(self, *args):
        """
        Defines the debug behaviour to be performed. This method must be overridden by all subclasses.
        Refers to the example above to define a customized hook.

        Args:
            *args (Any): The input/output of the operation, just print it.
        """
        raise RuntimeError("compute() is not overridden in subclass of class DebugHook.")

    def set_previous_op_name(self, prev_op_name):
        # Set prev_op_name.
        self.prev_op_name = prev_op_name

    def set_is_first(self, is_first_op):
        # Set op is the first in map.
        self.is_first_op = is_first_op
