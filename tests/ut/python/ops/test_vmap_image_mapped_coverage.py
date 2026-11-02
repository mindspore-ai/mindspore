# Copyright 2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
"""Real CPU image batching, with C++-ordered preprocessing only for direct eager calls."""
import importlib
import random

import numpy as np
import pytest
import mindspore as ms
import mindspore.dataset as dataset
from mindspore import ops, log as ms_log
from mindspore.common.generator import default_generator
from mindspore.ops.primitive import Primitive
from mindspore.ops.operations import _grad_ops as G, image_ops as IMG
from mindspore.ops._vmap import vmap_image_ops as I


def tensor(value, dtype=np.float32):
    return ms.Tensor(np.asarray(value, dtype=dtype))


def check(actual, expected):
    expected = np.asarray(expected)
    assert isinstance(actual, ms.Tensor) and actual.dtype == ms.float32
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual.asnumpy(), expected, rtol=2e-5, atol=2e-5)


@pytest.fixture(autouse=True)
def isolated_cpu(monkeypatch):
    context = {key: ms.get_context(key) for key in ('mode', 'device_target', 'pynative_synchronize')}
    python_state, numpy_state = random.getstate(), np.random.get_state()
    dataset_seed, generator_state, global_seed = dataset.config.get_seed(), ms.get_rng_state(), ms.get_seed()
    manual = default_generator._generator.attrs['manual_seed']
    seed_module = importlib.import_module('mindspore.common.seed')
    op_seeds = seed_module._KERNEL_SEED.copy()
    monkeypatch.setattr(ms_log._LogActionOnce, 'is_logged', ms_log._LogActionOnce.is_logged.copy())
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target='CPU', pynative_synchronize=True)
    yield
    ms.set_rng_state(generator_state)
    default_generator._generator.add_prim_attr('manual_seed', manual)
    dataset.config.set_seed(dataset_seed)
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    seed_module._GLOBAL_SEED = global_seed
    seed_module._KERNEL_SEED.clear()
    seed_module._KERNEL_SEED.update(op_seeds)
    ms.set_context(**context)


def coordinate(position, source_size, destination_size, mode):
    if mode == 'align_corners':
        result = position*(source_size-1)/(destination_size-1) if destination_size > 1 else 0.
    elif mode == 'half_pixel':
        result = (position+.5)*source_size/destination_size-.5
    else:
        result = position*source_size/destination_size
    return min(max(result, 0.), source_size-1.)


def linear_gradient_reference(gradient, input_size, mode):
    expected = np.zeros(gradient.shape[:-1]+(input_size,), np.float64)
    for output in range(gradient.shape[-1]):
        point = coordinate(output, input_size, gradient.shape[-1], mode)
        left = int(np.floor(point))
        right = min(left+1, input_size-1)
        weight = point-left
        expected[..., left] += gradient[..., output]*(1-weight)
        expected[..., right] += gradient[..., output]*weight
    return expected


def bilinear_gradient_reference(gradient, image_shape, align=False, half_pixel=False):
    expected = np.zeros(image_shape, np.float64)
    mode = 'align_corners' if align else 'half_pixel' if half_pixel else 'asymmetric'
    for out_y in range(gradient.shape[-2]):
        y = coordinate(out_y, image_shape[-2], gradient.shape[-2], mode)
        top, wy = int(np.floor(y)), y-np.floor(y)
        bottom = min(top+1, image_shape[-2]-1)
        for out_x in range(gradient.shape[-1]):
            x = coordinate(out_x, image_shape[-1], gradient.shape[-1], mode)
            left, wx = int(np.floor(x)), x-np.floor(x)
            right = min(left+1, image_shape[-1]-1)
            for row, y_weight in ((top, 1-wy), (bottom, wy)):
                for col, x_weight in ((left, 1-wx), (right, wx)):
                    expected[..., row, col] += gradient[..., out_y, out_x]*y_weight*x_weight
    return expected


def crop_reference(images, boxes, indices, size, method, extrapolation):
    output = np.empty((len(boxes), size[0], size[1], images.shape[-1]), np.float64)
    height, width = images.shape[1:3]
    for box, (y1, x1, y2, x2) in enumerate(boxes):
        image = images[indices[box]]
        for row in range(size[0]):
            y = (y1+y2)*(height-1)/2 if size[0] == 1 else (y1+(y2-y1)*row/(size[0]-1))*(height-1)
            for col in range(size[1]):
                x = (x1+x2)*(width-1)/2 if size[1] == 1 else (x1+(x2-x1)*col/(size[1]-1))*(width-1)
                if y < 0 or y > height-1 or x < 0 or x > width-1:
                    output[box, row, col] = extrapolation
                elif method == 'nearest':
                    output[box, row, col] = image[int(np.floor(y+.5)), int(np.floor(x+.5))]
                else:
                    top, left = int(np.floor(y)), int(np.floor(x))
                    bottom, right = min(top+1, height-1), min(left+1, width-1)
                    wy, wx = y-top, x-left
                    output[box, row, col] = ((1-wy)*((1-wx)*image[top, left]+wx*image[top, right])
                                              +wy*((1-wx)*image[bottom, left]+wx*image[bottom, right]))
    return output


def bilinear_inputs():
    gradient = np.arange(48, dtype=np.float32).reshape(2, 2, 1, 3, 4)/7-2
    image = np.arange(24, dtype=np.float32).reshape(2, 2, 1, 2, 3)/3
    return gradient, image


def linear_inputs():
    gradient = np.arange(20, dtype=np.float32).reshape(2, 2, 1, 5)/5-1
    image = np.arange(12, dtype=np.float32).reshape(2, 2, 1, 3)/4
    return gradient, image


def crop_inputs():
    first = (np.arange(48, dtype=np.float32).reshape(2, 3, 4, 2)**2)/20
    images = np.stack((first, first*1.5+70))
    boxes = np.array([[[0., 0., 1., 1.], [.25, .2, .75, .8]],
                      [[-.5, -.5, .5, .5], [.1, .3, .9, .7]]], np.float32)
    indices = np.array([[0, 1], [0, 0]], np.int32)
    return images, boxes, indices


def test_real_public_vmap_bilinear_gradient_both_mapped():
    gradient, image = bilinear_inputs()
    class Network(ms.nn.Cell):
        def __init__(self):
            super().__init__()
            self.gradient = G.ResizeBilinearGrad(align_corners=False, half_pixel_centers=True)
        def apply_gradient(self, dy, x):
            return self.gradient(dy, x)
        def construct(self, dy, x):
            return ops.vmap(self.apply_gradient, in_axes=(0, 0), out_axes=0)(dy, x)
    ms.set_context(mode=ms.GRAPH_MODE)
    result = Network()(tensor(gradient), tensor(image))
    check(result, bilinear_gradient_reference(gradient, image.shape, half_pixel=True))


def test_real_public_vmap_crop_all_tensor_operands_mapped():
    images, boxes, indices = crop_inputs()
    class Network(ms.nn.Cell):
        def __init__(self):
            super().__init__()
            self.crop = IMG.CropAndResize(method='bilinear', extrapolation_value=-9.)
        def apply_crop(self, x, regions, batch_indices, size):
            return self.crop(x, regions, batch_indices, size)
        def construct(self, x, regions, batch_indices):
            return ops.vmap(self.apply_crop, in_axes=(0, 0, 0, None), out_axes=0)(x, regions, batch_indices, (2, 3))
    ms.set_context(mode=ms.GRAPH_MODE)
    result = Network()(tensor(images), tensor(boxes), tensor(indices, np.int32))
    expected = np.stack([crop_reference(images[i], boxes[i], indices[i], (2, 3), 'bilinear', -9.) for i in range(2)])
    check(result, expected)


@pytest.mark.xfail(strict=True, raises=TypeError,
                   reason='Registered Linear1DGrad getter requires an extra coordinate_transformation_mode_bdim argument')
def test_real_public_vmap_linear_gradient_registration():
    gradient, image = linear_inputs()
    class Network(ms.nn.Cell):
        def __init__(self):
            super().__init__()
            self.gradient = G.ResizeLinear1DGrad(coordinate_transformation_mode='align_corners')
        def apply_gradient(self, dy, x):
            return self.gradient(dy, x)
        def construct(self, dy, x):
            return ops.vmap(self.apply_gradient, in_axes=(0, 0), out_axes=0)(dy, x)
    ms.set_context(mode=ms.GRAPH_MODE)
    try:
        result = Network()(tensor(gradient), tensor(image))
    except TypeError as error:
        assert "missing 1 required positional argument: 'coordinate_transformation_mode_bdim'" in str(error)
        raise
    check(result, linear_gradient_reference(gradient, image.shape[-1], 'align_corners'))


def cpp_flat_preprocess(first, *arguments):
    """Mirror vmap.cc:323-397 for the flat bdim tuples used by these image rules.

    The first formal argument is unused in the mapped branch. C++ inspects only
    subsequent arguments, stopping at the first non-None dimension, and calls
    the first argument only when every subsequent dimension is None. These
    closures omit prim, so first is their first data bdim, not a primitive.
    Grouped variable-arity arguments are outside this narrow adapter's scope.
    """
    assert arguments
    for argument in arguments:
        assert isinstance(argument, tuple) and len(argument) == 2
        if argument[1] is not None:
            return False, None
    if not callable(first):
        raise TypeError('C++ all-unmapped branch requires its first argument to be callable')
    return True, (first(*(argument[0] for argument in arguments)), None)


@pytest.fixture
def cpp_ordered_eager_frontend(monkeypatch):
    # Only substitute the graph-only frontend decision; every tensor operation
    # and CPU executor below is real. Graph proof tests above use actual C++.
    monkeypatch.setattr(I, 'vmap_general_preprocess', cpp_flat_preprocess)


def pack(value, axis):
    return tensor(value[0] if axis is None else np.moveaxis(value, 0, axis), value.dtype)


def canonical(value, axis):
    return np.broadcast_to(value[0], value.shape) if axis is None else value


def assert_tensor_argument(actual, expected):
    assert isinstance(actual, ms.Tensor)
    expected = np.asarray(expected)
    assert actual.dtype == (ms.int32 if expected.dtype == np.int32 else ms.float32)
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(actual.asnumpy(), expected)


def observe_native(monkeypatch, primitive, expected_arrays, expected_scalars, expected_attrs,
                   expected_init_attrs, generated_binding):
    """Observe full original ABI, then execute the unchanged real CPU call.

    Generated gradient __call__(dy, x) appends its constructor arguments.
    The original vmap rule already supplies those arguments, as native graph
    rewriting does. Bypass only that repeated Python binding, delegating all
    received arguments, without changing any data or flags, to Primitive.
    CropAndResize uses its ordinary, unmodified native call instead.
    """
    primitive_type = type(primitive)
    original_call = primitive_type.__call__
    calls = []

    def execute(actual_primitive, *arguments):
        assert actual_primitive is primitive and type(actual_primitive) is primitive_type
        assert actual_primitive.attrs == expected_attrs
        assert actual_primitive.init_attrs == expected_init_attrs
        assert len(arguments) == len(expected_arrays)+len(expected_scalars)
        for actual, expected in zip(arguments, expected_arrays):
            assert_tensor_argument(actual, expected)
        scalar_arguments = arguments[len(expected_arrays):]
        assert scalar_arguments == expected_scalars
        for actual, expected in zip(scalar_arguments, expected_scalars):
            assert type(actual) is type(expected)
        calls.append(arguments)
        if generated_binding:
            return Primitive.__call__(actual_primitive, *arguments)
        return original_call(actual_primitive, *arguments)

    monkeypatch.setattr(primitive_type, '__call__', execute)
    return calls


def test_cpp_flat_adapter_ignores_first_and_stops_at_first_mapped_argument():
    first_bdim = (tensor([7.]), 0)
    # C++ must return immediately, without examining the first argument or
    # validating arguments following the first mapped bdim.
    assert cpp_flat_preprocess(first_bdim, (tensor([8.]), 0), object()) == (False, None)
    with pytest.raises(TypeError, match='all-unmapped'):
        cpp_flat_preprocess(first_bdim, (tensor([8.]), None))
    identity = ops.Identity()
    source = tensor([7.])
    all_none, (result, axis) = cpp_flat_preprocess(identity, (source, None))
    assert all_none and axis is None
    check(result, [7.])


@pytest.mark.parametrize('align,half_pixel', [(True, False), (False, False), (False, True)])
@pytest.mark.parametrize('gradient_axis,image_axis', [(0, 0), (1, 2), (None, 1)])
def test_original_bilinear_gradient_cpu_native_abi(monkeypatch, cpp_ordered_eager_frontend,
                                                align, half_pixel, gradient_axis, image_axis):
    gradient, image = bilinear_inputs()
    primitive = G.ResizeBilinearGrad(align_corners=align, half_pixel_centers=half_pixel)
    assert type(primitive) is G.ResizeBilinearGrad
    actual_gradient = canonical(gradient, gradient_axis)
    calls = observe_native(monkeypatch, primitive,
                           (actual_gradient.reshape(4, 1, 3, 4), image.reshape(4, 1, 2, 3)),
                           (align, half_pixel), {'align_corners': align, 'half_pixel_centers': half_pixel},
                           {'align_corners': align, 'half_pixel_centers': half_pixel}, True)
    rule = I.get_resize_bilinear_grad_vmap_rule(primitive, 2)
    value, axis = rule((pack(gradient, gradient_axis), gradient_axis), (pack(image, image_axis), image_axis),
                       (align, None), (half_pixel, None))
    assert axis == 0 and len(calls) == 1
    check(value, bilinear_gradient_reference(actual_gradient, image.shape, align, half_pixel))


@pytest.mark.parametrize('mode', ['align_corners', 'half_pixel'])
@pytest.mark.parametrize('gradient_axis,image_axis', [(0, 0), (1, 2), (None, 1)])
def test_original_linear_gradient_cpu_native_abi(monkeypatch, cpp_ordered_eager_frontend,
                                              mode, gradient_axis, image_axis):
    gradient, image = linear_inputs()
    primitive = G.ResizeLinear1DGrad(coordinate_transformation_mode=mode)
    assert type(primitive) is G.ResizeLinear1DGrad
    native_mode = primitive.coordinate_transformation_mode
    assert type(native_mode) is int and native_mode == {'align_corners': 1, 'half_pixel': 2}[mode]
    actual_gradient = canonical(gradient, gradient_axis)
    calls = observe_native(monkeypatch, primitive,
                           (actual_gradient.reshape(4, 1, 5), image.reshape(4, 1, 3)), (native_mode,),
                           {'coordinate_transformation_mode': native_mode},
                           {'coordinate_transformation_mode': mode}, True)
    rule = I.get_resize_linear_1d_grad_vmap_rule(primitive, 2, (native_mode, None))
    value, axis = rule((pack(gradient, gradient_axis), gradient_axis), (pack(image, image_axis), image_axis))
    assert axis == 0 and len(calls) == 1
    check(value, linear_gradient_reference(actual_gradient, image.shape[-1], mode))


@pytest.mark.parametrize('method,size', [('bilinear', (2, 3)), ('nearest', (1, 2))])
@pytest.mark.parametrize('image_axis,boxes_axis,indices_axis', [(0, 0, 0), (1, 1, 1), (None, 0, 0), (0, None, 0)])
def test_original_crop_cpu_native_abi(monkeypatch, cpp_ordered_eager_frontend,
                                      method, size, image_axis, boxes_axis, indices_axis):
    images, boxes, indices = crop_inputs()
    actual_images = canonical(images, image_axis)
    actual_boxes = canonical(boxes, boxes_axis)
    primitive = IMG.CropAndResize(method=method, extrapolation_value=-9.)
    native_images = images[0] if image_axis is None else images.reshape(4, 3, 4, 2)
    native_indices = indices if image_axis is None else indices+np.arange(2, dtype=np.int32)[:, None]*2
    calls = observe_native(monkeypatch, primitive,
                           (native_images, actual_boxes.reshape(4, 4), native_indices.reshape(4)), (size,),
                           {'method': method, 'extrapolation_value': -9.,
                            'input_names': ['x', 'boxes', 'box_index', 'crop_size'], 'output_names': ['y']},
                           {'method': method, 'extrapolation_value': -9.}, False)
    rule = I.get_crop_and_resize_vmap_rule(primitive, 2)
    value, axis = rule((pack(images, image_axis), image_axis), (pack(boxes, boxes_axis), boxes_axis),
                       (pack(indices, indices_axis), indices_axis), (size, None))
    assert axis == 0 and len(calls) == 1
    expected = np.stack([crop_reference(actual_images[i], actual_boxes[i], indices[i], size, method, -9.)
                         for i in range(2)])
    check(value, expected)


def test_original_crop_rejects_mapped_crop_size(cpp_ordered_eager_frontend):
    images, boxes, indices = crop_inputs()
    rule = I.get_crop_and_resize_vmap_rule(IMG.CropAndResize(), 2)
    with pytest.raises(ValueError, match='axis of `crop_size`.*must be None'):
        rule((tensor(images), 0), (tensor(boxes), 0), (tensor(indices, np.int32), 0), ((2, 3), 0))


@pytest.mark.xfail(strict=True, raises=ValueError,
                   reason='Missing prim makes actual C++ all-unmapped branch call the grad bdim tuple')
def test_real_public_bilinear_first_only_mapped():
    gradient, image = bilinear_inputs()
    class Network(ms.nn.Cell):
        def __init__(self):
            super().__init__()
            self.gradient = G.ResizeBilinearGrad(False, True)
        def apply_gradient(self, dy, x):
            return self.gradient(dy, x)
        def construct(self, dy, x):
            return ops.vmap(self.apply_gradient, in_axes=(0, None), out_axes=0)(dy, x)
    ms.set_context(mode=ms.GRAPH_MODE)
    try:
        result = Network()(tensor(gradient), tensor(image[0]))
    except ValueError as error:
        assert 'The object is not callable' in str(error)
        assert 'vmap_image_ops.py:96' in str(error)
        raise
    check(result, bilinear_gradient_reference(gradient, image.shape, half_pixel=True))


@pytest.mark.xfail(strict=True, raises=ValueError,
                   reason='Missing prim makes actual C++ all-unmapped branch call the images bdim tuple')
def test_real_public_crop_first_only_mapped():
    images, boxes, indices = crop_inputs()
    class Network(ms.nn.Cell):
        def __init__(self):
            super().__init__()
            self.crop = IMG.CropAndResize(method='bilinear', extrapolation_value=-9.)
        def apply_crop(self, x, regions, batch_indices, size):
            return self.crop(x, regions, batch_indices, size)
        def construct(self, x, regions, batch_indices):
            return ops.vmap(self.apply_crop, in_axes=(0, None, None, None), out_axes=0)(x, regions, batch_indices, (2, 3))
    ms.set_context(mode=ms.GRAPH_MODE)
    try:
        result = Network()(tensor(images), tensor(boxes[0]), tensor(indices[0], np.int32))
    except ValueError as error:
        assert 'The object is not callable' in str(error)
        assert 'vmap_image_ops.py:160' in str(error)
        raise
    expected = np.stack([crop_reference(images[i], boxes[0], indices[0], (2, 3), 'bilinear', -9.) for i in range(2)])
    check(result, expected)


@pytest.mark.parametrize('operation', ['bilinear', 'linear', 'crop'])
@pytest.mark.xfail(strict=True, raises=ValueError,
                   reason='Image closure omits prim; real C++ all-unmapped branch attempts to call its first bdim tuple')
def test_real_jit_original_all_unmapped(operation):
    if operation == 'bilinear':
        gradient, image = bilinear_inputs()
        rule = I.get_resize_bilinear_grad_vmap_rule(G.ResizeBilinearGrad(False, True), 2)
        class Network(ms.nn.Cell):
            def construct(self, dy, x):
                return rule((dy, None), (x, None), (False, None), (True, None))
        inputs = (tensor(gradient[0]), tensor(image[0]))
        expected = bilinear_gradient_reference(gradient[0], image[0].shape, half_pixel=True)
        source_line = 96
    elif operation == 'linear':
        gradient, image = linear_inputs()
        primitive = G.ResizeLinear1DGrad(coordinate_transformation_mode='align_corners')
        rule = I.get_resize_linear_1d_grad_vmap_rule(primitive, 2, (primitive.coordinate_transformation_mode, None))
        class Network(ms.nn.Cell):
            def construct(self, dy, x):
                return rule((dy, None), (x, None))
        inputs = (tensor(gradient[0]), tensor(image[0]))
        expected = linear_gradient_reference(gradient[0], image.shape[-1], 'align_corners')
        source_line = 125
    else:
        images, boxes, indices = crop_inputs()
        rule = I.get_crop_and_resize_vmap_rule(IMG.CropAndResize('bilinear', -9.), 2)
        class Network(ms.nn.Cell):
            def construct(self, x, regions, batch_indices):
                return rule((x, None), (regions, None), (batch_indices, None), ((2, 3), None))
        inputs = (tensor(images[0]), tensor(boxes[0]), tensor(indices[0], np.int32))
        expected = crop_reference(images[0], boxes[0], indices[0], (2, 3), 'bilinear', -9.)
        source_line = 160
    ms.set_context(mode=ms.GRAPH_MODE)
    try:
        result, axis = Network()(*inputs)
    except ValueError as error:
        assert 'The object is not callable' in str(error)
        assert 'vmap_image_ops.py:'+str(source_line) in str(error)
        raise
    assert axis is None
    check(result, expected)
