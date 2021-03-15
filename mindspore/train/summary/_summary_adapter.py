# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
"""Generate the summary event which conform to proto format."""
import platform
import time

import numpy as np
from PIL import Image

from mindspore import log as logger
from mindspore import context
from mindspore.communication.management import get_rank
from mindspore.communication.management import GlobalComm

from ..._checkparam import Validator
from ..anf_ir_pb2 import DataType, ModelProto
from ..summary_pb2 import Event

# define the MindSpore image format
MS_IMAGE_TENSOR_FORMAT = 'NCHW'
# Set the Event mark
EVENT_FILE_NAME_MARK = ".out.events.summary."
# Set the init event of version and mark
EVENT_FILE_INIT_VERSION_MARK = "MindSpore.Event:"
EVENT_FILE_INIT_VERSION = 1

F32_MIN, F32_MAX = np.finfo(np.float32).min, np.finfo(np.float32).max


def get_event_file_name(prefix, suffix, time_second):
    """
    Create file name: file_prefix + EVENT_FILE_NAME_MARK + time(seconds) + "." + Hostname + file_suffix.

    Args:
        prefix (str): The prefix of file name.
        suffix (str): The suffix of file name.
        time_second (str): The time stamp of file name.

    Returns:
        String, the name of event log file.
    """
    Validator.check_str_by_regular(prefix)
    Validator.check_str_by_regular(suffix)
    file_name = ""
    hostname = platform.node()

    device_num = context.get_auto_parallel_context('device_num')
    device_id = context.get_context('device_id')
    if device_num > 1 or GlobalComm.WORLD_COMM_GROUP == 'nccl_world_group':
        # Notice:
        # In GPU distribute training scene, get_context('device_id') will not work,
        # so we use get_rank instead of get_context.
        device_id = get_rank()

    file_name = f'{file_name}{EVENT_FILE_NAME_MARK}{time_second}.{device_id}.{hostname}'

    if prefix is not None:
        file_name = prefix + file_name

    if suffix is not None:
        file_name = file_name + suffix

    return file_name


def package_init_event():
    """Package the summary init event."""
    init_event = Event()
    init_event.wall_time = time.time()
    version = EVENT_FILE_INIT_VERSION_MARK + str(EVENT_FILE_INIT_VERSION)
    init_event.version = version
    return init_event


def package_graph_event(data):
    """
    Package the summary graph event.

    Args:
        data (Bytes): Graph bytes string.

    Returns:
        Event, event log object.
    """
    graph_event = Event()
    graph_event.wall_time = time.time()
    modelp = ModelProto()
    modelp.ParseFromString(data)
    graph_event.graph_def.CopyFrom(modelp.graph)
    return graph_event


def package_summary_event(data_list, step, wall_time):
    """
    Package the summary to event protobuffer.

    Args:
        data_list (list): Summary data list.
        step (Number): The recode step index.
        wall_time (float): The wall time.

    Returns:
        Summary, the summary event.
    """
    # create the event of summary
    summary_event = Event()
    summary = summary_event.summary
    summary_event.wall_time = wall_time
    summary_event.step = int(step)

    for value in data_list:
        summary_type = value["_type"]
        data = value["data"]
        tag = value["name"]

        logger.debug(f"Now process {summary_type} summary, tag = {tag}")

        summary_value = summary.value.add()
        summary_value.tag = tag
        # get the summary type and parse the tag
        if summary_type == 'Scalar':
            if not _fill_scalar_summary(tag, data, summary_value):
                del summary.value[-1]
        elif summary_type == 'Tensor':
            _fill_tensor_summary(tag, data, summary_value.tensor)
        elif summary_type == 'Image':
            if not _fill_image_summary(tag, data, summary_value.image, MS_IMAGE_TENSOR_FORMAT):
                del summary.value[-1]
        elif summary_type == 'Histogram':
            _fill_histogram_summary(tag, data, summary_value.histogram)
        else:
            # The data is invalid ,jump the data
            logger.error(f"Summary type({summary_type}) is error, tag = {tag}")
            del summary.value[-1]

    return summary_event


def _nptype_to_prototype(np_value):
    """
    Transform the np type to proto type.

    Args:
        np_value (Type): Numpy data type.

    Returns:
        Type, proto data type.
    """
    np2pt_tbl = {
        np.bool_: 'DT_BOOL',
        np.int8: 'DT_INT8',
        np.int16: 'DT_INT16',
        np.int32: 'DT_INT32',
        np.int64: 'DT_INT64',
        np.uint8: 'DT_UINT8',
        np.uint16: 'DT_UINT16',
        np.uint32: 'DT_UINT32',
        np.uint64: 'DT_UINT64',
        np.float16: 'DT_FLOAT16',
        np.float: 'DT_FLOAT64',
        np.float32: 'DT_FLOAT32',
        np.float64: 'DT_FLOAT64',
        None: 'DT_UNDEFINED'
    }
    np_type = None
    if np_value is None:
        logger.error("The numpy value is none")
    else:
        np_type = np_value.dtype.type

    proto = np2pt_tbl.get(np_type, None)
    if proto is None:
        raise TypeError("No match for proto data type.")

    return proto


def _fill_scalar_summary(tag: str, np_value, summary):
    """
    Package the scalar summary.

    Args:
        tag (str): Summary tag describe.
        np_value (Object): Scalary object.

    Returns:
        Summary, return scalar summary content.
    """
    logger.debug(f"Set({tag}) the scalar summary value")
    if np_value.size == 1:
        # is scalar
        summary.scalar_value = np_value.item()
        return True
    if np_value.size > 1:
        logger.warning(
            f"The tensor is not a single scalar, tag = {tag}, ndim = {np_value.ndim}, shape = {np_value.shape}")
        summary.scalar_value = next(np_value.flat).item()
        return True
    logger.error(f"There no values inside tensor, tag = {tag}, size = {np_value.size}")
    return False


def _fill_tensor_summary(tag: str, np_value, summary_tensor):
    """
    Package the tensor summary.

    Args:
        tag (str): Summary tag describe.
        np_value (Type): Summary data type.
        summary_tensor (Tensor): The tensor of summary.

    Returns:
        Summary, return tensor summary content.
    """
    logger.debug(f"Set({tag}) the tensor summary value")
    # get tensor dtype
    tensor_dtype = _nptype_to_prototype(np_value)
    summary_tensor.data_type = DataType.Value(tensor_dtype)

    # get the value list
    tensor_value_list = np_value.reshape(-1).tolist()
    summary_tensor.float_data.extend(tensor_value_list)

    # get the tensor dim
    for v in np_value.shape:
        summary_tensor.dims.append(v)

    return summary_tensor


def _calc_histogram_bins(count):
    """
    Calculates experience-based optimal bins number for histogram.

    There should be enough number in each bin. So we calc bin numbers according to count. For very small count(1 -
    10), we assign carefully chosen number. For large count, we tried to make sure there are 9-10 numbers in each
    bucket on average. Too many bins will slow down performance, so we set max number of bins to 90.

    Args:
        count (int): Valid number count for the tensor.

    Returns:
        int, number of histogram bins.
    """
    max_bins, max_per_bin = 90, 10

    if not count:
        return 1
    if count <= 5:
        return 2
    if count <= 10:
        return 3
    if count <= 880:
        # note that math.ceil(881/10) + 1 equals 90
        return count // max_per_bin + 1

    return max_bins


def _fill_histogram_summary(tag: str, np_value: np.ndarray, summary) -> None:
    """
    Package the histogram summary.

    Args:
        tag (str): Summary tag describe.
        np_value (np.ndarray): Summary data.
        summary (summary_pb2.Summary.Histogram): Summary histogram data.
    """
    logger.debug(f"Set({tag}) the histogram summary value")
    # Default bucket for tensor with no valid data.
    ma_value = np.ma.masked_invalid(np_value)
    total, valid = np_value.size, ma_value.count()
    invalids = []
    for isfn in np.isnan, np.isposinf, np.isneginf:
        if total - valid > sum(invalids):
            count = np.count_nonzero(isfn(np_value))
            invalids.append(count)
        else:
            invalids.append(0)

    summary.count = total
    summary.nan_count, summary.pos_inf_count, summary.neg_inf_count = invalids
    if not valid:
        logger.warning(f'There are no valid values in the ndarray(size={total}, shape={np_value.shape})')
        # summary.{min, max, sum} are 0s by default, no need to explicitly set
    else:
        # BUG: max of a masked array with dtype np.float16 returns inf
        # See numpy issue#15077
        if issubclass(np_value.dtype.type, np.floating):
            summary.min = ma_value.min(fill_value=np.PINF)
            summary.max = ma_value.max(fill_value=np.NINF)
            if summary.min < F32_MIN or summary.max > F32_MAX:
                logger.warning(f'Values({summary.min}, {summary.max}) are too large, '
                               f'you may encounter some undefined behaviours hereafter.')
        else:
            summary.min = ma_value.min()
            summary.max = ma_value.max()
        summary.sum = ma_value.sum(dtype=np.float64)
        bins = _calc_histogram_bins(valid)
        first_edge, last_edge = summary.min, summary.max

        if not first_edge < last_edge:
            first_edge -= 0.5
            last_edge += 0.5

        bins = np.linspace(first_edge, last_edge, bins + 1, dtype=np_value.dtype)
        hists, edges = np.histogram(np_value, bins=bins)

        for hist, edge1, edge2 in zip(hists, edges, edges[1:]):
            bucket = summary.buckets.add()
            bucket.width = edge2 - edge1
            bucket.count = hist
            bucket.left = edge1


def _fill_image_summary(tag: str, np_value, summary_image, input_format='NCHW'):
    """
    Package the image summary.

    Args:
        tag (str): Summary tag describe.
        np_value (Type): Summary data type.
        summary_image (Tensor): The tensor of summary.
        input_format (str): Data sort order index. Default: 'NCHW'.

    Returns:
        Summary, return image summary content.
    """
    logger.debug(f"Set({tag}) the image summary value")
    if np_value.ndim != 4 or np_value.shape[1] not in (1, 3):
        logger.error(f"The value is not Image, tag = {tag}, ndim = {np_value.ndim}, shape={np_value.shape}")
        return False

    if np_value.ndim != len(input_format):
        logger.error(
            f"The tensor with dim({np_value.ndim}) can't convert the format({input_format}) because dim not same")
        return False

    # convert the tensor format
    tensor = _convert_image_format(np_value, input_format)

    # convert the tensor dtype
    # Do not assume that user passes in values in [0, 255], use data type to detect
    scale_factor = 1
    if tensor.dtype == np.uint8:
        scale_factor = 1
    elif np.max(tensor) <= 1 and np.min(tensor) >= 0:
        scale_factor = 255
    tensor = tensor.astype(np.float32)
    tensor = (tensor * scale_factor).astype(np.uint8)

    # create the image summary
    height, width, channel, image_string = _make_image(tensor)
    summary_image.height = height
    summary_image.width = width
    summary_image.colorspace = channel
    summary_image.encoded_image = image_string
    return True


def _make_image(tensor, rescale=1):
    """
    Convert a numpy representation of an image to Image protobuf.

    Args:
        tensor (Tensor): The image data.
        rescale (Number): The rescale value. Default: 1.

    Returns:
        (Number, Number, Number, Bytes), return the height, width, channel, image string .
    """
    height, width, channel = tensor.shape
    scaled_height = int(height * rescale)
    scaled_width = int(width * rescale)
    image = Image.fromarray(tensor)
    image = image.resize((scaled_width, scaled_height), Image.ANTIALIAS)
    import io
    output = io.BytesIO()
    image.save(output, format='PNG')
    image_string = output.getvalue()
    output.close()
    return height, width, channel, image_string


def _convert_image_format(np_tensor, input_format, out_format='HWC'):
    """
    Convert the image format.

    Args:
        np_tensor (Tensor): The image data.
        input_format (str): Input data format.
        out_format (str): The output data format. Default: 'HWC'.

    Returns:
        Tensor, return format image.
    """
    input_format = input_format.upper()

    # convert the NCHW
    if input_format != 'NCHW':
        index = [input_format.find(c) for c in 'NCHW']
        tensor_nchw = np_tensor.transpose(index)
    else:
        tensor_nchw = np_tensor

    # make grid to expand N
    tensor_chw = _make_canvas_for_imgs(tensor_nchw)

    # convert to out format
    out_index = ['CHW'.find(c) for c in out_format]
    out_tensor = tensor_chw.transpose(out_index)
    return out_tensor


def _make_canvas_for_imgs(tensor, col_imgs=8):
    """
    Expand the N, show imgs on a canvs.

    Args:
        tensor (Tensor): The canvas value.
        col_imgs (Number): The image colume number. Default: 8.

    Returns:
        Tensor, return canvas of image.
    """
    # expand the N1HW to N3HW
    if tensor.shape[1] == 1:
        tensor = np.concatenate([tensor, tensor, tensor], 1)

    # expand the N
    n = tensor.shape[0]
    h = tensor.shape[2]
    w = tensor.shape[3]
    cols = min(n, col_imgs)
    rows = int(np.ceil(float(n) / cols))

    # create the canvas: expand the n
    out_canvas = np.zeros((3, h * rows, w * cols))
    i = 0
    for y in range(rows):
        for x in range(cols):
            if i >= n:
                break
            out_canvas[:, y * h:(y + 1) * h, x * w:(x + 1) * w] = tensor[i]
            i = i + 1
    return out_canvas
