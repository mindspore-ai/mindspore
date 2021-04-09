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
"""Hierarchical occlusion edit tree searcher."""
from enum import Enum
import copy
import re
import math

import numpy as np
from scipy.ndimage import gaussian_filter

from mindspore import nn
from mindspore import Tensor
from mindspore.ops import Squeeze
from mindspore.train._utils import check_value_type


AUTO_LAYER_MAX = 3                       # maximum number of layer by auto settings
AUTO_WIN_SIZE_MIN = 28                   # minimum window size by auto settings
AUTO_WIN_SIZE_DIV = 2                    # denominator of windows size calculations by auto settings
AUTO_STRIDE_DIV = 5                      # denominator of stride calculations by auto settings
AUTO_MASK_GAUSSIAN_RADIUS_DIV = 25       # denominator of gaussian mask radius calculations by auto settings
DEFAULT_THRESHOLD = 0.5                  # default target prediction threshold
DEFAULT_BATCH_SIZE = 64                  # default batch size for batch inference search
MASK_GAUSSIAN_RE = r'^gaussian:(\d+)$'   # gaussian mask string pattern

# minimum length of input images' short side with auto settings
AUTO_IMAGE_SHORT_SIDE_MIN = AUTO_WIN_SIZE_MIN * AUTO_WIN_SIZE_DIV


def is_valid_str_mask(mask):
    """Check if it is a valid string mask."""
    check_value_type('mask', mask, str)
    match = re.match(MASK_GAUSSIAN_RE, mask)
    return match and int(match.group(1)) > 0


def compile_mask(mask, image):
    """Compile mask to a ready to use object."""
    if mask is None:
        return compile_str_mask(auto_str_mask(image), image)
    check_value_type('mask', mask, (str, tuple, float, np.ndarray))
    if isinstance(mask, str):
        return compile_str_mask(mask, image)

    if isinstance(mask, tuple):
        _check_iterable_type('mask', mask, tuple, float)
    elif isinstance(mask, np.ndarray):
        if len(image.shape) == 4 and len(mask.shape) == 3:
            mask = np.expand_dims(mask, axis=0)
        elif len(image.shape) == 3 and len(mask.shape) == 4 and mask.shape[0] == 1:
            mask = mask.squeeze(0)
        if image.shape != mask.shape:
            raise ValueError("Image and mask is not match in shape.")
    return mask


def auto_str_mask(image):
    """Generate auto string mask for the image."""
    check_value_type('image', image, np.ndarray)
    short_side = np.min(image.shape[-2:])
    radius = int(round(short_side/AUTO_MASK_GAUSSIAN_RADIUS_DIV))
    if radius == 0:
        raise ValueError(f"Input image's short side:{short_side} is too small for auto mask, "
                         f"at least {AUTO_MASK_GAUSSIAN_RADIUS_DIV}pixels is required.")
    return f'gaussian:{radius}'


def compile_str_mask(mask, image):
    """Concert string mask to numpy.ndarray."""
    check_value_type('mask', mask, str)
    check_value_type('image', image, np.ndarray)
    match = re.match(MASK_GAUSSIAN_RE, mask)
    if match:
        radius = int(match.group(1))
        if radius > 0:
            sigma = [0] * len(image.shape)
            sigma[-2] = radius
            sigma[-1] = radius
            return gaussian_filter(image, sigma=sigma, mode='nearest')
    raise ValueError(f"Invalid string mask: '{mask}'.")


class EditStep:
    """
    Edit step that describes a box region, also represents an edit tree.

    Args:
        layer (int): Layer number, -1 is root layer, 0 or above is normal edit layer.
        box (tuple[int, int, int, int]): Tuple of x, y, width, height.
    """
    def __init__(self, layer, box):
        self.layer = layer
        self.box = box
        self.network_output = 0
        self.step_change = 0
        self.children = None

    @property
    def x(self):
        """X-coordinate of the box."""
        return self.box[0]

    @property
    def y(self):
        """Y-coordinate of the box."""
        return self.box[1]

    @property
    def width(self):
        """Width of the box."""
        return self.box[2]

    @property
    def height(self):
        """Height of the box."""
        return self.box[3]

    @property
    def is_leaf(self):
        """Returns True if no child edit step."""
        return not self.children

    @property
    def leaf_steps(self):
        """Returns all leaf edit steps in the tree."""
        if self.is_leaf:
            return [self]
        steps = []
        for child in self.children:
            steps.extend(child.leaf_steps)
        return steps

    @property
    def max_layer(self):
        """Maximum layer number in the edit tree."""
        if self.is_leaf:
            return self.layer
        layer = self.layer
        for child in self.children:
            child_max_layer = child.max_layer
            if child_max_layer > layer:
                layer = child_max_layer
        return layer

    def add_child(self, child):
        """Add a child edit step."""
        if self.children is None:
            self.children = [child]
        else:
            self.children.append(child)

    def remove_all_children(self):
        """Remove all child steps."""
        self.children = None

    def get_layer_or_leaf_steps(self, layer):
        """Get all edit steps of the layer and all leaf edit steps above the layer."""
        if self.layer == layer or (self.layer < layer and self.is_leaf):
            return [self]
        steps = []
        if self.layer < layer and self.children:
            for child in self.children:
                steps.extend(child.get_layer_or_leaf_steps(layer))
        return steps

    def get_layer_steps(self, layer):
        """Get all edit steps of the layer."""
        if self.layer == layer:
            return [self]
        steps = []
        if self.layer < layer and self.children:
            for child in self.children:
                steps.extend(child.get_layer_steps(layer))
        return steps

    @classmethod
    def apply(cls,
              image,
              mask,
              edit_steps,
              by_masking=False,
              inplace=False):
        """
        Apply edit steps.

        Args:
            image (numpy.ndarray): Image tensor in CHW or NCHW(N=1) format.
            mask (Union[str, tuple[float, float, float], float, numpy.ndarray]): The mask, type can be
                str: String mask, e.g. 'gaussian:9' - Gaussian blur with radius of 9.
                tuple[float, float, float]: RGB solid color mask,
                float: Grey scale solid color mask.
                numpy.ndarray: Image mask in CHW or NCHW(N=1) format.
            edit_steps (list[EditStep], optional): Edit steps to be applied.
            by_masking (bool): Whether it is masking mode.
            inplace (bool): Whether the modification is going to take place in the input image tensor. False to
                construct a new image tensor as result.

        Returns:
            numpy.ndarray, the result image tensor.

        Raises:
            TypeError: Be raised for any argument or data type problem.
            ValueError: Be raised for any argument or data value problem.
        """
        if by_masking:
            return cls.apply_masking(image, mask, edit_steps, inplace)
        return cls.apply_unmasking(image, mask, edit_steps, inplace)

    @staticmethod
    def apply_masking(image,
                      mask,
                      edit_steps,
                      inplace=False):
        """
        Apply edit steps in masking mode.

        Args:
            image (numpy.ndarray): Image tensor in CHW or NCHW(N=1) format.
            mask (Union[str, tuple[float, float, float], float, numpy.ndarray]): The mask, type can be
                str: String mask, e.g. 'gaussian:9' - Gaussian blur with radius of 9.
                tuple[float, float, float]: RGB solid color mask,
                float: Grey scale solid color mask.
                numpy.ndarray: Image mask in CHW or NCHW(N=1) format.
            edit_steps (list[EditStep], optional): Edit steps to be applied.
            inplace (bool): Whether the modification is going to take place in the input image tensor. False to
                construct a new image tensor as result.

        Returns:
            numpy.ndarray, the result image tensor.

        Raises:
            TypeError: Be raised for any argument or data type problem.
            ValueError: Be raised for any argument or data value problem.
        """
        check_value_type('image', image, np.ndarray)
        check_value_type('mask', mask, (str, tuple, float, np.ndarray))
        if isinstance(mask, tuple):
            _check_iterable_type('mask', mask, tuple, float)

        if edit_steps is not None:
            _check_iterable_type('edit_steps', edit_steps, (tuple, list), EditStep)

        mask = compile_mask(mask, image)

        if inplace:
            background = image
        else:
            background = np.copy(image)

        if not edit_steps:
            return background

        for step in edit_steps:

            x_max = step.x + step.width
            y_max = step.y + step.height

            if x_max > background.shape[-1]:
                x_max = background.shape[-1]

            if y_max > background.shape[-2]:
                y_max = background.shape[-2]

            if x_max <= step.x or y_max <= step.y:
                continue

            if isinstance(mask, np.ndarray):
                background[..., step.y:y_max, step.x:x_max] = mask[..., step.y:y_max, step.x:x_max]
            else:
                if isinstance(mask, (int, float)):
                    mask = (mask, mask, mask)
                for c in range(3):
                    background[..., c, step.y:y_max, step.x:x_max] = mask[c]
        return background

    @staticmethod
    def apply_unmasking(image,
                        mask,
                        edit_steps,
                        inplace=False):
        """
        Apply edit steps in unmasking mode.

        Args:
            image (numpy.ndarray): Image tensor in CHW or NCHW(N=1) format.
            mask (Union[str, tuple[float, float, float], float, numpy.ndarray]): The mask, type can be
                str: String mask, e.g. 'gaussian:9' - Gaussian blur with radius of 9.
                tuple[float, float, float]: RGB solid color mask,
                float: Grey scale solid color mask.
                numpy.ndarray: Image mask in CHW or NCHW(N=1) format.
            edit_steps (list[EditStep]): Edit steps to be applied.
            inplace (bool): Whether the modification is going to take place in the input mask tensor. False to
                construct a new image tensor as result.

        Returns:
            numpy.ndarray, the result image tensor.

        Raises:
            TypeError: Be raised for any argument or data type problem.
            ValueError: Be raised for any argument or data value problem.
        """
        check_value_type('image', image, np.ndarray)
        check_value_type('mask', mask, (str, tuple, float, np.ndarray))
        if isinstance(mask, tuple):
            _check_iterable_type('mask', mask, tuple, float)

        if edit_steps is not None:
            _check_iterable_type('edit_steps', edit_steps, (tuple, list), EditStep)

        mask = compile_mask(mask, image)

        if isinstance(mask, np.ndarray):
            if inplace:
                background = mask
            else:
                background = np.copy(mask)
        else:
            if inplace:
                raise ValueError('Inplace cannot be True when mask is not a numpy.ndarray')

            background = np.zeros_like(image)
            if isinstance(mask, (int, float)):
                background.fill(mask)
            else:
                for c in range(3):
                    background[..., c, :, :] = mask[c]

        if not edit_steps:
            return background

        for step in edit_steps:

            x_max = step.x + step.width
            y_max = step.y + step.height

            if x_max > background.shape[-1]:
                x_max = background.shape[-1]

            if y_max > background.shape[-2]:
                y_max = background.shape[-2]

            if x_max <= step.x or y_max <= step.y:
                continue

            background[..., step.y:y_max, step.x:x_max] = image[..., step.y:y_max, step.x:x_max]

        return background


class NoValidResultError(RuntimeError):
    """Error for no edit step layer's network output meet the threshold."""


class OriginalOutputError(RuntimeError):
    """Error for network output of the original image is not strictly larger than the threshold."""


class Searcher:
    """
    Edit step searcher.

    Args:
        network (Cell): Image tensor in CHW or NCHW(N=1) format.
        win_sizes (Union(list[int], optional): Moving square window size (length of side) of layers,
            None means by auto calcuation.
        strides (Union(list[int], optional): Stride of layers, None means by auto calcuation.
        threshold (float): Threshold network output value of the target class.
        by_masking (bool): Whether it is masking mode.
    """

    def __init__(self,
                 network,
                 win_sizes=None,
                 strides=None,
                 threshold=DEFAULT_THRESHOLD,
                 by_masking=False):

        check_value_type('network', network, nn.Cell)

        if win_sizes is not None:
            _check_iterable_type('win_sizes', win_sizes, list, int)
            if not win_sizes:
                raise ValueError('Argument win_sizes is empty.')

            for i in range(1, len(win_sizes)):
                if win_sizes[i] >= win_sizes[i-1]:
                    raise ValueError('Argument win_sizes is not strictly descending.')

            if win_sizes[-1] <= 0:
                raise ValueError('Argument win_sizes has non-positive number.')
        elif strides is not None:
            raise ValueError('Argument win_sizes cannot be None if strides is not None.')

        if strides is not None:
            _check_iterable_type('strides', strides, list, int)
            for i in range(1, len(strides)):
                if strides[i] >= strides[i-1]:
                    raise ValueError('Argument win_sizes is not strictly descending.')

            if strides[-1] <= 0:
                raise ValueError('Argument strides has non-positive number.')

            if len(strides) != len(win_sizes):
                raise ValueError('Length of strides and win_sizes is not equal.')
        elif win_sizes is not None:
            raise ValueError('Argument strides cannot be None if win_sizes is not None.')

        self._network = copy.deepcopy(network)
        self._compiled_mask = None
        self._threshold = threshold
        self._win_sizes = copy.copy(win_sizes) if win_sizes else None
        self._strides = copy.copy(strides) if strides else None
        self._by_masking = by_masking

    @property
    def network(self):
        """Get the network."""
        return self._network

    @property
    def by_masking(self):
        """Check if it is masking mode."""
        return self._by_masking

    @property
    def threshold(self):
        """The network output threshold to stop the search."""
        return self._threshold

    @property
    def win_sizes(self):
        """Windows sizes in pixels."""
        return self._win_sizes

    @property
    def strides(self):
        """Strides in pixels."""
        return self._strides

    @property
    def compiled_mask(self):
        """The compiled mask after a successful search() call."""
        return self._compiled_mask

    def search(self, image, class_idx, mask=None):
        """
        Search smallest sufficient/destruction region on an image.

        Args:
            image (numpy.ndarray): Image tensor in CHW or NCHW(N=1) format.
            class_idx (int): Target class index.
            mask (Union[str, tuple[float, float, float], float], optional): The mask, type can be
                str: String mask, e.g. 'gaussian:9' - Gaussian blur with radius of 9.
                tuple[float, float, float]: RGB solid color mask,
                float: Grey scale solid color mask.
                None: By auto calculation.

        Returns:
            tuple[EditStep, list[float]], the root edit step and network output of each layer after applied the
                layer steps.

        Raise:
            TypeError: Be raised for any argument or data type problem.
            ValueError: Be raised for any argument or data value problem.
            NoValidResultError: Be raised if no valid result was found.
            OriginalOutputError: Be raised if network output of the original image is not strictly larger than
                the threshold.
        """
        check_value_type('image', image, (Tensor, np.ndarray))

        if isinstance(image, Tensor):
            image = image.asnumpy()

        if len(image.shape) == 4:
            if image.shape[0] != 1:
                raise ValueError("Argument image's batch size is not 1.")
        elif len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        else:
            raise ValueError("Argument image is not in CHW or NCHW(N=1) format.")

        check_value_type('class_idx', class_idx, int)

        if class_idx < 0:
            raise ValueError("Argument class_idx is less then zero.")

        self._compiled_mask = compile_mask(mask, image)

        short_side = np.min(image.shape[-2:])
        if self._win_sizes is None:
            win_sizes, strides = self._auto_win_sizes_strides(short_side)
        else:
            win_sizes, strides = self._win_sizes, self._strides

        if short_side <= win_sizes[0]:
            raise ValueError(f"Input image's short side is shorter then or "
                             f"equals to the first window size:{win_sizes[0]}.")

        self._network.set_train(False)

        # the search result will be store as a edit tree that attached to the root step.
        root_step = EditStep(-1, (0, 0, image.shape[-1], image.shape[-2]))
        root_job = _SearchJob(by_masking=self._by_masking,
                              class_idx=class_idx,
                              win_sizes=win_sizes,
                              strides=strides,
                              layer=0,
                              search_field=root_step.box,
                              pre_edit_steps=None,
                              parent_step=root_step)
        self._process_root_job(image, root_job)

        # the leaf layer's network output may not meet the threshold,
        # we have to cutoff the unqualified layers
        layer_count = root_step.max_layer + 1
        if layer_count == 0:
            raise NoValidResultError("No edit step layer was found.")

        # gather the network output of each layer
        layer_outputs = [None] * layer_count
        for layer in range(layer_count):
            steps = root_step.get_layer_or_leaf_steps(layer)
            if not steps:
                continue
            masked_image = EditStep.apply(image, self._compiled_mask, steps, by_masking=self._by_masking)
            output = self._network(Tensor(masked_image))
            output = output[0, class_idx].asnumpy().item()
            layer_outputs[layer] = output

        # determine which layer we have to cutoff
        cutoff_layer = None
        for layer in reversed(range(layer_count)):
            if layer_outputs[layer] is not None and self._is_threshold_met(layer_outputs[layer]):
                cutoff_layer = layer
                break

        if cutoff_layer is None or root_step.is_leaf:
            raise NoValidResultError(f"No edit step layer's network output meet the threshold: {self._threshold}.")

        # cutoff the layer by removing all children of the layer's steps.
        steps = root_step.get_layer_steps(cutoff_layer)
        for step in steps:
            step.remove_all_children()
        layer_outputs = layer_outputs[:cutoff_layer + 1]

        return root_step, layer_outputs

    def _process_root_job(self, sample_input, root_job):
        """
        Process job queue.

        Args:
            sample_input (numpy.ndarray): Image tensor in NCHW(N=1) format.
            root_job (_SearchJob): Root search job.
        """
        job_queue = [root_job]
        while job_queue:
            job = job_queue.pop(0)
            sub_job_queue = []
            job_edit_steps, stop_reason = self._process_job(job, sample_input, sub_job_queue)

            if stop_reason in (self._StopReason.THRESHOLD_MET, self._StopReason.STEP_CHANGE_MET):
                for step in job_edit_steps:
                    job.parent_step.add_child(step)
                job_queue.extend(sub_job_queue)

    def _process_job(self, job, sample_input, job_queue):
        """
        Process a job.

        Args:
            job (_SearchJob): Search job to be processed.
            sample_input (numpy.ndarray): Image tensor in NCHW(N=1) format.
            job_queue (list[_SearchJob]): Job queue.

        Returns:
            tuple[list[EditStep], _StopReason], result edit stop and the stop reason.
        """
        edit_steps = []

        # make the network output with the original image is strictly larger than the threshold
        if job.layer == 0:
            original_output = self._network(Tensor(sample_input))[0, job.class_idx].asnumpy().item()
            if original_output <= self._threshold:
                raise OriginalOutputError(f'The original output is not strictly larger the threshold: '
                                          f'{self._threshold}')

        # applying the pre-edit steps from the parent steps
        if job.pre_edit_steps:
            # use the latest leaf steps to increase the accuracy
            leaf_steps = []
            for step in job.pre_edit_steps:
                leaf_steps.extend(step.leaf_steps)
            pre_edit_steps = leaf_steps
        else:
            pre_edit_steps = None
        workpiece = EditStep.apply(sample_input,
                                   self._compiled_mask,
                                   pre_edit_steps,
                                   self._by_masking)

        job.on_start(sample_input, workpiece, self._compiled_mask, self._network)
        start_output = self._network(Tensor(workpiece))[0, job.class_idx].asnumpy().item()
        last_output = start_output

        # greedy search loop
        while True:

            if self._is_threshold_met(last_output):
                return edit_steps, self._StopReason.THRESHOLD_MET

            try:
                best_edit = job.find_best_edit()
            except _NoNewStepError:
                return edit_steps, self._StopReason.NO_NEW_STEP
            except _RepeatedStepError:
                return edit_steps, self._StopReason.REPEATED_STEP

            best_edit.step_change = best_edit.network_output - last_output

            if job.layer < job.layer_count - 1 and self._is_greedy(best_edit.step_change):
                # create net layer search job if new edit step is valid and not yet reaching
                # the final layer
                if job.pre_edit_steps:
                    pre_edit_steps = list(job.pre_edit_steps)
                    pre_edit_steps.extend(edit_steps)
                else:
                    pre_edit_steps = list(edit_steps)

                sub_job = job.create_sub_job(best_edit, pre_edit_steps)
                job_queue.append(sub_job)

            edit_steps.append(best_edit)

            if job.layer > 0:
                # stop if the step change meet the parent step change only after layer 0
                change = best_edit.network_output - start_output
                if self._is_step_change_met(job.parent_step.step_change, change):
                    return edit_steps, self._StopReason.STEP_CHANGE_MET

            last_output = best_edit.network_output

    def _is_threshold_met(self, network_output):
        """Check if the threshold was met."""
        if self._by_masking:
            return network_output <= self._threshold
        return network_output >= self._threshold

    def _is_step_change_met(self, target, step_change):
        """Check if the change target was met."""
        if self._by_masking:
            return step_change <= target
        return step_change >= target

    def _is_greedy(self, step_change):
        """Check if it is a greedy step."""
        if self._by_masking:
            return step_change < 0
        return step_change > 0

    @classmethod
    def _auto_win_sizes_strides(cls, short_side):
        """
        Calculate auto window sizes and strides.

        Args:
            short_side (int): Length of search space.

        Returns:
            tuple[list[int], list[int]], window sizes and strides.
        """
        win_sizes = []
        strides = []
        cur_len = int(short_side/AUTO_WIN_SIZE_DIV)
        while len(win_sizes) < AUTO_LAYER_MAX and cur_len >= AUTO_WIN_SIZE_MIN:
            stride = int(cur_len/AUTO_STRIDE_DIV)
            if stride <= 0:
                break
            win_sizes.append(cur_len)
            strides.append(stride)
            cur_len = int(cur_len/AUTO_WIN_SIZE_DIV)
        if not win_sizes:
            raise ValueError(f"Image's short side is less then {AUTO_IMAGE_SHORT_SIDE_MIN}, "
                             f"unable to calculates auto settings.")
        return win_sizes, strides

    class _StopReason(Enum):
        """Stop reason of search job."""
        THRESHOLD_MET = 0       # threshold was met.
        STEP_CHANGE_MET = 1     # parent step change was met.
        NO_NEW_STEP = 2         # no new step was found.
        REPEATED_STEP = 3       # repeated step was found.


def _check_iterable_type(arg_name, arg_value, container_type, elem_types):
    """Concert iterable argument data type."""
    check_value_type(arg_name, arg_value, container_type)
    for elem in arg_value:
        check_value_type(arg_name + ' element', elem, elem_types)


class _NoNewStepError(Exception):
    """Error for no new step was found."""


class _RepeatedStepError(Exception):
    """Error for repeated step was found."""


class _SearchJob:
    """
    Search job.

    Args:
        by_masking (bool): Whether it is masking mode.
        class_idx (int): Target class index.
        win_sizes (list[int]): Moving square window size (length of side) of layers.
        strides (list[int]): Strides of layers.
        layer (int): Layer number.
        search_field (tuple[int, int, int, int]): Search field in x, y, width, height format.
        pre_edit_steps (list[EditStep], optional): Edit steps to be applied before searching.
        parent_step (EditStep): Parent edit step.
        batch_size (int): Batch size of batched inferences.
    """

    def __init__(self,
                 by_masking,
                 class_idx,
                 win_sizes,
                 strides,
                 layer,
                 search_field,
                 pre_edit_steps,
                 parent_step,
                 batch_size=DEFAULT_BATCH_SIZE):

        if layer >= len(win_sizes):
            raise ValueError('Layer is larger then number of window sizes.')

        self.by_masking = by_masking
        self.class_idx = class_idx
        self.win_sizes = win_sizes
        self.strides = strides
        self.layer = layer
        self.search_field = search_field
        self.pre_edit_steps = pre_edit_steps
        self.parent_step = parent_step
        self.batch_size = batch_size
        self.network = None
        self.mask = None
        self.original_input = None

        self._workpiece = None
        self._found_best_edits = None
        self._found_uvs = None
        self._u_pixels = None
        self._v_pixels = None

    @property
    def layer_count(self):
        """Number of layers."""
        return len(self.win_sizes)

    def on_start(self, original_input, workpiece, mask, network):
        """
        Notification of the start of the search job.

        Args:
            original_input (numpy.ndarray): The original image tensor in CHW or NCHW(N=1) format.
            workpiece (numpy.ndarray): The intermediate image tensor in CHW or NCHW(N=1) format.
            mask (Union[tuple[float, float, float], float, numpy.ndarray]): The mask, type can be
                tuple[float, float, float]: RGB solid color mask,
                float: Grey scale solid color mask.
                numpy.ndarray: Image mask, has same format of original_input.
            network (nn.Cell): Classification network.
        """
        self.original_input = original_input
        self.mask = mask
        self.network = network

        self._workpiece = workpiece
        self._found_best_edits = []
        self._found_uvs = []
        self._u_pixels = self._calc_uv_pixels(self.search_field[0], self.search_field[2])
        self._v_pixels = self._calc_uv_pixels(self.search_field[1], self.search_field[3])

    def create_sub_job(self, parent_step, pre_edit_steps):
        """Create next layer search job."""
        return self.__class__(by_masking=self.by_masking,
                              class_idx=self.class_idx,
                              win_sizes=self.win_sizes,
                              strides=self.strides,
                              layer=self.layer + 1,
                              search_field=copy.copy(parent_step.box),
                              pre_edit_steps=pre_edit_steps,
                              parent_step=parent_step,
                              batch_size=self.batch_size)

    def find_best_edit(self):
        """
        Find the next best edit step.

        Returns:
            EditStep, the next best edit step.
        """
        workpiece = self._workpiece
        if len(workpiece.shape) == 3:
            workpiece = np.expand_dims(workpiece, axis=0)

        # generate input tensors with shifted masked/unmasked region and pack into a batch
        squeeze = Squeeze()
        best_new_workpiece = None
        best_output = None
        best_edit = None
        best_uv = None
        batch = np.repeat(workpiece, repeats=self.batch_size, axis=0)
        batch_uvs = []
        batch_steps = []
        batch_i = 0
        win_size = self.win_sizes[self.layer]
        for u, x in enumerate(self._u_pixels):
            for v, y in enumerate(self._v_pixels):
                if (u, v) in self._found_uvs:
                    continue

                edit_step = EditStep(self.layer, (x, y, win_size, win_size))

                if self.by_masking:
                    EditStep.apply(batch[batch_i],
                                   self.mask,
                                   [edit_step],
                                   self.by_masking,
                                   inplace=True)
                else:
                    EditStep.apply(self.original_input,
                                   batch[batch_i],
                                   [edit_step],
                                   self.by_masking,
                                   inplace=True)

                batch_i += 1
                batch_uvs.append((u, v))
                batch_steps.append(edit_step)
                if batch_i == self.batch_size:
                    # the batch is full, inference and empty it
                    batch_output = self.network(Tensor(batch))
                    batch_output = batch_output[:, self.class_idx]
                    if len(batch_output.shape) > 1:
                        batch_output = squeeze(batch_output)
                    if self.by_masking:
                        batch_best_i = np.argmin(batch_output.asnumpy())
                    else:
                        batch_best_i = np.argmax(batch_output.asnumpy())
                    batch_best_output = batch_output[int(batch_best_i)].asnumpy().item()

                    if best_output is None or self._is_output0_better(batch_best_output, best_output):
                        best_output = batch_best_output
                        best_uv = batch_uvs[batch_best_i]
                        best_edit = batch_steps[batch_best_i]
                        best_new_workpiece = batch[batch_best_i]

                    batch = np.repeat(workpiece, repeats=self.batch_size, axis=0)
                    batch_uvs = []
                    batch_i = 0

        if batch_i > 0:
            # don't forget the last half full batch
            batch_output = self.network(Tensor(batch))
            batch_output = batch_output[:, self.class_idx]
            if len(batch_output.shape) > 1:
                batch_output = squeeze(batch_output)
            if self.by_masking:
                batch_best_i = np.argmin(batch_output.asnumpy()[:batch_i, ...])
            else:
                batch_best_i = np.argmax(batch_output.asnumpy()[:batch_i, ...])

            batch_best_output = batch_output[int(batch_best_i)].asnumpy().item()
            if best_output is None or self._is_output0_better(batch_best_output, best_output):
                best_output = batch_best_output
                best_uv = batch_uvs[batch_best_i]
                best_edit = batch_steps[batch_best_i]
                best_new_workpiece = batch[batch_best_i]

        if best_edit is None:
            raise _NoNewStepError

        if best_uv in self._found_uvs:
            raise _RepeatedStepError

        self._found_uvs.append(best_uv)
        self._found_best_edits.append(best_edit)
        best_edit.network_output = best_output

        # continue on the best workpiece in the next function call
        self._workpiece = best_new_workpiece

        return best_edit

    def _is_output0_better(self, output0, output1):
        """Check if the network output0 is better."""
        if self.by_masking:
            return output0 < output1
        return output0 > output1

    def _calc_uv_pixels(self, begin, length):
        """
        Calculate the pixel coordinate of shifts.

        Args:
            begin (int): The beginning pixel coordinate of search field.
            length (int): The length of search field.

        Returns:
             list[int], pixel coordinate of shifts.
        """
        win_size = self.win_sizes[self.layer]
        stride = self.strides[self.layer]
        shift_count = self._calc_shift_count(length, win_size, stride)
        pixels = [0] * shift_count
        for i in range(shift_count):
            if i == shift_count - 1:
                pixels[i] = begin + length - win_size
            else:
                pixels[i] = begin + i*stride
        return pixels

    @staticmethod
    def _calc_shift_count(length, win_size, stride):
        """
        Calculate the number of shifts in search field.

        Args:
            length (int): The length of search field.
            win_size (int): The length of sides of moving window.
            stride (int): The stride.

        Returns:
             int, number of shifts.
        """
        if length <= win_size or win_size < stride or stride <= 0:
            raise ValueError("Invalid length, win_size or stride.")
        count = int(math.ceil((length - win_size)/stride))
        if (count - 1)*stride + win_size < length:
            return count + 1
        return count
