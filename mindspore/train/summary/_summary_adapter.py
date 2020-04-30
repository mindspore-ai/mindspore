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
import socket
import time

import numpy as np
from PIL import Image

from mindspore import log as logger

from ..._checkparam import _check_str_by_regular
from ..anf_ir_pb2 import DataType, ModelProto
from ..summary_pb2 import Event

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
        logger.warning("Can't del the data because data_id(%r) " "does not have data in g_summary_data_dict", data_id)


def get_summary_data(data_id):
    """Save the global summary cache."""
    ret = None
    global g_summary_data_dict
    if data_id in g_summary_data_dict:
        ret = g_summary_data_dict.get(data_id)
    else:
        logger.warning("The data_id(%r) does not have data in g_summary_data_dict", data_id)
    return ret


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
    summary_event.wall_time = time.time()
    summary_event.step = int(step)

    for value in data_list:
        summary_type = value["_type"]
        data = value["data"]
        tag = value["name"]

        logger.debug("Now process %r summary, tag = %r", summary_type, tag)

        summary_value = summary.value.add()
        summary_value.tag = tag
        # get the summary type and parse the tag
        if summary_type == 'Scalar':
            summary_value.scalar_value = _get_scalar_summary(tag, data)
        elif summary_type == 'Tensor':
            summary_tensor = summary_value.tensor
            _get_tensor_summary(tag, data, summary_tensor)
        elif summary_type == 'Image':
            summary_image = summary_value.image
            _get_image_summary(tag, data, summary_image, MS_IMAGE_TENSOR_FORMAT)
        elif summary_type == 'Histogram':
            summary_histogram = summary_value.histogram
            _fill_histogram_summary(tag, data, summary_histogram)
        else:
            # The data is invalid ,jump the data
            logger.error("Summary type(%r) is error, tag = %r", summary_type, tag)

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
            logger.error("The tensor is not Scalar, tag = %r, Shape = %r", tag, np_value.shape)
    else:
        np_list = np_value.reshape(-1).tolist()
        scalar_value = np_list[0]
        logger.error("The value is not Scalar, tag = %r, ndim = %r", tag, np_value.ndim)

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
    logger.debug("Set(%r) the histogram summary value", tag)
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
        logger.warning('There are no valid values in the ndarray(size=%d, shape=%d)', total, np_value.shape)
        # summary.{min, max, sum} are 0s by default, no need to explicitly set
    else:
        summary.min = ma_value.min()
        summary.max = ma_value.max()
        summary.sum = ma_value.sum()
        bins = _calc_histogram_bins(valid)
        range_ = summary.min, summary.max
        hists, edges = np.histogram(np_value, bins=bins, range=range_)

        for hist, edge1, edge2 in zip(hists, edges, edges[1:]):
            bucket = summary.buckets.add()
            bucket.width = edge2 - edge1
            bucket.count = hist
            bucket.left = edge1


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
        logger.error("The value is not Image, tag = %r, ndim = %r", tag, np_value.ndim)

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
        logger.error("The tensor with dim(%r) can't convert the format(%r) because dim not same", np_tensor.ndim,
                     input_format)
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
        logger.error("The image tensor with ndim(%r) and shape(%r) is not 'NCHW' format", tensor.ndim, tensor.shape)
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
