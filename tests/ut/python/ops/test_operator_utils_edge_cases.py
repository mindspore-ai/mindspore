# Copyright 2026 Huawei Technologies Co., Ltd
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
"""Boundary cases for operator shape inference and distributed TCP addresses."""
import pytest

from mindspore.common import dtype as mstype
from mindspore.ops._utils import utils


@pytest.mark.parametrize('left,right,expected', [
    ([2, 3], [2, 3], [2, 3]),
    ([3], [2, 1], [2, 3]),
    ([2, 1, 3], [4, 3], [2, 4, 3]),
    ([2, -1], [1, 5], [2, 5]),
    ([2, 5], [1, -1], [2, 5]),
    ([-2], [2, 3], [-2]),
    ([2, 3], [-2], [-2]),
    ([0, 3], [1, 3], [0, 3]),
    ([], [2, 3], [2, 3]),
])
def test_broadcast_shape_dynamic_and_empty_dimensions(left, right, expected):
    """Unknown dimensions must not discard known broadcast sizes."""
    assert utils.get_broadcast_shape(left, right, 'Add') == expected


def test_broadcast_shape_rejects_incompatible_dimensions():
    """Shape inference must report incompatible operand shapes."""
    with pytest.raises(ValueError, match='broadcast'):
        utils.get_broadcast_shape([2, 3], [2, 4], 'Add', 'left', 'right')


@pytest.mark.parametrize('shapes,axis,expected', [
    ([(2, 3), (2, 4), (2, 1)], -1, ([0, 3, 7], 8, 1)),
    ([(2, 3), (4, 3)], 0, ([0, 2], 6, 0)),
    ([(2, -1), (2, 4), (2, 1)], 1, ([0, -1, -1], -1, 1)),
    ([(2, 3), (2, -1), (2, 1)], 1, ([0, 3, -1], -1, 1)),
    ([(-1, 3), (2, 4)], 1, ([0, 3], 7, 1)),
])
def test_concat_offsets_accumulate_and_preserve_unknowns(shapes, axis, expected):
    """Offsets must accumulate only along the normalized concatenation axis."""
    types = [mstype.TensorType(mstype.float32)] * len(shapes)
    assert utils.get_concat_offset(shapes, types, axis, 'Concat') == expected


@pytest.mark.parametrize('shapes,types,axis,error', [
    ([], [], 0, ValueError),
    ([(2, 3), (2, 3, 4)], [mstype.TensorType(mstype.float32)] * 2, 0, ValueError),
    ([(2, 3), (2, 3)], [mstype.TensorType(mstype.float32), mstype.TensorType(mstype.int32)], 0, ValueError),
    ([(2, 3), (4, 4)], [mstype.TensorType(mstype.float32)] * 2, 1, ValueError),
    ([(2, 3)], [mstype.TensorType(mstype.float32)], 2, ValueError),
    ([(2, 3)], [mstype.float32], 0, TypeError),
])
def test_concat_offsets_reject_invalid_inputs(shapes, types, axis, error):
    """Invalid rank, dtype, axis and non-concat dimensions must be rejected."""
    with pytest.raises(error):
        utils.get_concat_offset(shapes, types, axis, 'Concat')


@pytest.mark.parametrize('shape,expected', [((), (1,)), ((2, 3, 4), (24,)), ((2, 0, 4), (0,))])
def test_get_1d_shape_preserves_element_count(shape, expected):
    """Flattened shapes retain the number of elements, including empty tensors."""
    assert utils.get_1d_shape(shape) == expected


@pytest.mark.parametrize('out_shape,index_shape,axis,batch_dims,expected', [
    ((2, 3, 4), (5, 6), 1, 0, (1, 2, 0)),
    ((2, 3, 4), (5, 6), -1, 0, (1, 2, 0)),
    ((2, 3, 4, 5), (2, 6), 2, 1, (0, 2, 1, 3)),
])
def test_generate_shape_index_with_batch_dimensions(out_shape, index_shape, axis, batch_dims, expected):
    """Gather permutations keep batch dimensions ahead of index dimensions."""
    assert utils.generate_shape_index(out_shape, index_shape, axis, batch_dims) == expected


@pytest.mark.parametrize('url,expected', [
    ('tcp://127.0.0.1:1', ('127.0.0.1', 1)),
    ('tcp://255.255.255.255:65535', ('255.255.255.255', 65535)),
    ('tcp://0.0.0.0:8080', ('0.0.0.0', 8080)),
])
def test_parse_tcp_url_accepts_port_boundaries(url, expected):
    """Both valid port endpoints must be accepted without truncation."""
    assert utils._parse_tcp_url_for_ipv4(url) == expected


@pytest.mark.parametrize('url,reason', [
    ('http://127.0.0.1:80', 'must start'),
    ('tcp://', 'missing IP and port'),
    ('tcp://127.0.0.1', 'missing port separator'),
    ('tcp://::1:80', 'IPv6'),
    ('tcp://:80', 'missing IP address'),
    ('tcp://256.0.0.1:80', 'invalid IPv4'),
    ('tcp://localhost:80', 'invalid IPv4'),
    ('tcp://127.0.0.1:', 'missing port number'),
    ('tcp://127.0.0.1:abc', 'port must be numeric'),
    ('tcp://127.0.0.1:0', 'port out of range'),
    ('tcp://127.0.0.1:65536', 'port out of range'),
])
def test_parse_tcp_url_rejects_malformed_addresses(url, reason):
    """Malformed addresses fail with the parser's reason and original cause."""
    with pytest.raises(ValueError, match=reason) as exc:
        utils._parse_tcp_url_for_ipv4(url)
    assert isinstance(exc.value.__cause__, ValueError)


@pytest.mark.parametrize('url', [None, 123, b'tcp://127.0.0.1:80'])
def test_parse_tcp_url_requires_a_string(url):
    """Non-string values must fail before address parsing."""
    with pytest.raises(TypeError, match='string'):
        utils._parse_tcp_url_for_ipv4(url)
