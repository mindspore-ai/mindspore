# Copyright 2020 Huawei Technologies Co., Ltd
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
import time
import socket
import math
from enum import Enum, unique
import numpy as np
from PIL import Image

from mindspore import log as logger
from ..summary_pb2 import Event
from ..anf_ir_pb2 import ModelProto, DataType
from ..._checkparam import _check_str_by_regular

# define the MindSpore image format
MS_IMAGE_TENSOR_FORMAT = 'NCHW'
# Set the Event mark
EVENT_FILE_NAME_MARK = ".out.events.summary."
# Set the init event of version and mark
EVENT_FILE_INIT_VERSION_MARK = "Mindspore.Event:"
EVENT_FILE_INIT_VERSION = 1
# cache the summary data dict
# {id: SummaryData}
#           |---[{"name": tag_name, "data": numpy}, {"name": tag_name, "data": numpy},...]
g_summary_data_dict = {}

def save_summary_data(data_id, data):
    """Save the global summary cache."""
    global g_summary_data_dict
    g_summary_data_dict[data_id] = data


def del_summary_data(data_id):
    """Save the global summary cache."""
    global g_summary_data_dict
    if data_id in g_summary_data_dict:
        del g_summary_data_dict[data_id]
    else:
        logger.warning("Can't del the data because data_id(%r) "
                       "does not have data in g_summary_data_dict", data_id)

def get_summary_data(data_id):
    """Save the global summary cache."""
    ret = None
    global g_summary_data_dict
    if data_id in g_summary_data_dict:
        ret = g_summary_data_dict.get(data_id)
    else:
        logger.warning("The data_id(%r) does not have data in g_summary_data_dict", data_id)
    return ret

@unique
class SummaryType(Enum):
    """
    Summary type.

    Args:
        SCALAR (Number): Summary Scalar enum.
        TENSOR (Number): Summary TENSOR enum.
        IMAGE (Number): Summary image enum.
        GRAPH (Number): Summary graph enum.
        HISTOGRAM (Number): Summary histogram enum.
        INVALID (Number): Unknow type.
    """
    SCALAR = 1      # Scalar summary
    TENSOR = 2      # Tensor summary
    IMAGE = 3       # Image summary
    GRAPH = 4       # graph
    HISTOGRAM = 5   # Histogram Summary
    INVALID = 0xFF  # unknow type


def get_event_file_name(prefix, suffix):
    """
    Create file name: file_prefix + EVENT_FILE_NAME_MARK + time(seconds) + "." + Hostname + file_suffix.

    Args:
        prefix (str): The prefix of file name.
        suffix (str): The suffix of file name.

    Returns:
        String, the name of event log file.
    """
    _check_str_by_regular(prefix)
    _check_str_by_regular(suffix)
    file_name = ""
    time_second = str(int(time.time()))
    hostname = socket.gethostname()

    if prefix is not None:
        file_name = file_name + prefix

    file_name = file_name + EVENT_FILE_NAME_MARK + time_second + "." + hostname

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

    Retruns:
        Event, event log object.
    """
    graph_event = Event()
    graph_event.wall_time = time.time()
    modelp = ModelProto()
    modelp.ParseFromString(data)
    graph_event.graph_def.CopyFrom(modelp.graph)
    return graph_event


def package_summary_event(data_id, step):
    """
    Package the summary to event protobuffer.

    Args:
        data_id (Number): Summary data id.
        step (Number): The recode step index.

    Returns:
        Summary, the summary event.
    """
    data_list = get_summary_data(data_id)
    if data_list is None:
        logger.error("The step(%r) does not have record data.", step)
    del_summary_data(data_id)
    # create the event of summary
    summary_event = Event()
    summary = summary_event.summary

    for value in data_list:
        tag = value["name"]
        data = value["data"]
        summary_type = value["type"]

        # get the summary type and parse the tag
        if summary_type is SummaryType.SCALAR:
            logger.debug("Now process Scalar summary, tag = %r", tag)
            summary_value = summary.value.add()
            summary_value.tag = tag
            summary_value.scalar_value = _get_scalar_summary(tag, data)
        elif summary_type is SummaryType.TENSOR:
            logger.debug("Now process Tensor summary, tag = %r", tag)
            summary_value = summary.value.add()
            summary_value.tag = tag
            summary_tensor = summary_value.tensor
            _get_tensor_summary(tag, data, summary_tensor)
        elif summary_type is SummaryType.IMAGE:
            logger.debug("Now process Image summary, tag = %r", tag)
            summary_value = summary.value.add()
            summary_value.tag = tag
            summary_image = summary_value.image
            _get_image_summary(tag, data, summary_image, MS_IMAGE_TENSOR_FORMAT)
        elif summary_type is SummaryType.HISTOGRAM:
            logger.debug("Now process Histogram summary, tag = %r", tag)
            summary_value = summary.value.add()
            summary_value.tag = tag
            summary_histogram = summary_value.histogram
            _fill_histogram_summary(tag, data, summary_histogram)
        else:
            # The data is invalid ,jump the data
            logger.error("Summary type is error, tag = %r", tag)
            continue

    summary_event.wall_time = time.time()
    summary_event.step = int(step)
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


def _get_scalar_summary(tag: str, np_value):
    """
    Package the scalar summary.

    Args:
        tag (str): Summary tag describe.
        np_value (Object): Scalary object.

    Returns:
        Summary, return scalar summary content.
    """
    logger.debug("Set(%r) the scalar summary value", tag)
    if np_value.ndim == 0:
        # is scalar
        scalar_value = np_value.item()
    elif np_value.ndim == 1:
        # Because now GE can't providesumm the real shape info to convert the Tensor
        # So consider the dim = 1, shape = (1,) tensor is scalar
        scalar_value = np_value[0]
        if np_value.shape != (1,):
            logger.error("The tensor is not Scalar, tag = %r, Value = %r", tag, np_value)
    else:
        np_list = np_value.reshape(-1).tolist()
        scalar_value = np_list[0]
        logger.error("The value is not Scalar, tag = %r, Value = %r", tag, np_value)

    logger.debug("The tag(%r) value is: %r", tag, scalar_value)
    return scalar_value


def _get_tensor_summary(tag: str, np_value, summary_tensor):
    """
    Package the tensor summary.

    Args:
        tag (str): Summary tag describe.
        np_value (Type): Summary data type.
        summary_tensor (Tensor): The tensor of summary.

    Retruns:
        Summary, return tensor summary content.
    """
    logger.debug("Set(%r) the tensor summary value", tag)
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
    number_per_bucket = 10
    max_bins = 90

    if not count:
        return 1
    if count <= 5:
        return 2
    if count <= 10:
        return 3
    if count <= 880:
        # note that math.ceil(881/10) + 1 equals 90
        return int(math.ceil(count / number_per_bucket) + 1)

    return max_bins


def _fill_histogram_summary(tag: str, np_value: np.array, summary_histogram) -> None:
    """
    Package the histogram summary.

    Args:
        tag (str): Summary tag describe.
        np_value (np.array): Summary data.
        summary_histogram (summary_pb2.Summary.Histogram): Summary histogram data.
    """
    logger.debug("Set(%r) the histogram summary value", tag)
    # Default bucket for tensor with no valid data.
    default_bucket_left = -0.5
    default_bucket_width = 1.0

    if np_value.size == 0:
        bucket = summary_histogram.buckets.add()
        bucket.left = default_bucket_left
        bucket.width = default_bucket_width
        bucket.count = 0

        summary_histogram.nan_count = 0
        summary_histogram.pos_inf_count = 0
        summary_histogram.neg_inf_count = 0

        summary_histogram.max = 0
        summary_histogram.min = 0
        summary_histogram.sum = 0

        summary_histogram.count = 0

        return

    summary_histogram.nan_count = np.count_nonzero(np.isnan(np_value))
    summary_histogram.pos_inf_count = np.count_nonzero(np.isposinf(np_value))
    summary_histogram.neg_inf_count = np.count_nonzero(np.isneginf(np_value))
    summary_histogram.count = np_value.size

    masked_value = np.ma.masked_invalid(np_value)
    tensor_max = masked_value.max()
    tensor_min = masked_value.min()
    tensor_sum = masked_value.sum()

    # No valid value in tensor.
    if tensor_max is np.ma.masked:
        bucket = summary_histogram.buckets.add()
        bucket.left = default_bucket_left
        bucket.width = default_bucket_width
        bucket.count = 0

        summary_histogram.max = np.nan
        summary_histogram.min = np.nan
        summary_histogram.sum = 0

        return

    bin_number = _calc_histogram_bins(masked_value.count())
    counts, edges = np.histogram(np_value, bins=bin_number, range=(tensor_min, tensor_max))

    for ind, count in enumerate(counts):
        bucket = summary_histogram.buckets.add()
        bucket.left = edges[ind]
        bucket.width = edges[ind + 1] - edges[ind]
        bucket.count = count

    summary_histogram.max = tensor_max
    summary_histogram.min = tensor_min
    summary_histogram.sum = tensor_sum


def _get_image_summary(tag: str, np_value, summary_image, input_format='NCHW'):
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
    logger.debug("Set(%r) the image summary value", tag)
    if np_value.ndim != 4:
        logger.error("The value is not Image, tag = %r, Value = %r", tag, np_value)

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
    return summary_image


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
    out_tensor = None
    if np_tensor.ndim != len(input_format):
        logger.error("The tensor(%r) can't convert the format(%r) because dim not same",
                     np_tensor, input_format)
        return out_tensor

    input_format = input_format.upper()

    if len(input_format) == 4:
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
    else:
        logger.error("Don't support the format(%r) convert", input_format)
    return out_tensor


def _make_canvas_for_imgs(tensor, col_imgs=8):
    """
    Expand the N, show imgs on a canvs.

    Args:
        tensor (Tensor): The canvas value.
        col_imgs (Number): The image colume number. Default: 8.

    Returns:
        Tensor, retrun canvas of image.
    """
    # expand the N1HW to N3HW
    out_canvas = None
    if tensor.shape[1] == 1:
        tensor = np.concatenate([tensor, tensor, tensor], 1)

    # check the tensor format
    if tensor.ndim != 4 or tensor.shape[1] != 3:
        logger.error("The image tensor(%r) is not 'NCHW' format", tensor)
        return out_canvas

    # expand the N
    n = tensor.shape[0]
    h = tensor.shape[2]
    w = tensor.shape[3]
    cols = min(n, col_imgs)
    rows = int(np.ceil(float(n) / cols))

    # creat the canvas: expand the n
    out_canvas = np.zeros((3, h * rows, w * cols))
    i = 0
    for y in range(rows):
        for x in range(cols):
            if i >= n:
                break
            out_canvas[:, y * h:(y + 1) * h, x * w:(x + 1) * w] = tensor[i]
            i = i + 1
    return out_canvas
