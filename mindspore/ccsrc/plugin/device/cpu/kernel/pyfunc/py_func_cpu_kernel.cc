/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "plugin/device/cpu/kernel/pyfunc/py_func_cpu_kernel.h"

#include <memory>
#include <vector>
#include "Eigen/Core"
#include "abstract/utils.h"
#include "plugin/device/cpu/hal/device/cpu_common.h"
#include "include/common/utils/python_adapter.h"
#include "plugin/factory/ms_factory.h"
#include "utils/ms_utils_secure.h"

namespace mindspore {
namespace kernel {
namespace {
py::object RawMemoryToScalar(const void *data, const TypeId &type) {
  switch (type) {
    case kNumberTypeBool:
      return py::bool_(*reinterpret_cast<const bool *>(data));
    case kNumberTypeInt16:
      return py::int_(*reinterpret_cast<const int16_t *>(data));
    case kNumberTypeUInt16:
      return py::int_(*reinterpret_cast<const uint16_t *>(data));
    case kNumberTypeInt8:
      return py::int_(*reinterpret_cast<const int8_t *>(data));
    case kNumberTypeUInt8:
      return py::int_(*reinterpret_cast<const uint8_t *>(data));
    case kNumberTypeInt32:
      return py::int_(*reinterpret_cast<const int32_t *>(data));
    case kNumberTypeUInt32:
      return py::int_(*reinterpret_cast<const uint32_t *>(data));
    case kNumberTypeInt64:
      return py::int_(*reinterpret_cast<const int64_t *>(data));
    case kNumberTypeUInt64:
      return py::int_(*reinterpret_cast<const uint64_t *>(data));
    case kNumberTypeFloat16: {
      const Eigen::half_impl::__half_raw data_half(*reinterpret_cast<const uint16_t *>(data));
      return py::float_(Eigen::half_impl::half_to_float(data_half));
    }
    case kNumberTypeFloat32:
      return py::float_(*reinterpret_cast<const float *>(data));
    case kNumberTypeFloat64:
      return py::float_(*reinterpret_cast<const double *>(data));
    default:
      MS_LOG(EXCEPTION) << "Type: " << type << " not supported.";
  }
}

void ScalarToRawMemory(const py::object &obj, const TypeId &type, const AddressPtr &address) {
  MS_EXCEPTION_IF_NULL(address);
  switch (type) {
    case kNumberTypeBool: {
      bool data = py::cast<bool>(obj);
      CHECK_RET_WITH_EXCEPT(memcpy_s(address->addr, address->size, &data, sizeof(bool)), EOK, "memcpy failed.");
      return;
    }
    // ref: pybind11-src/include/pybind11/pytypes.h
    // py::int_ convert py::object to `long`, `unsigned long`, `long long`, `unsigned long long` with Python API
    // according to typename T, and then convert to target data type with C style cast.
    case kNumberTypeInt8: {
      int8_t data = py::cast<int8_t>(obj);
      CHECK_RET_WITH_EXCEPT(memcpy_s(address->addr, address->size, &data, sizeof(int8_t)), EOK, "memcpy failed.");
      return;
    }
    case kNumberTypeUInt8: {
      uint8_t data = py::cast<uint8_t>(obj);
      CHECK_RET_WITH_EXCEPT(memcpy_s(address->addr, address->size, &data, sizeof(uint8_t)), EOK, "memcpy failed.");
      return;
    }
    case kNumberTypeInt16: {
      int16_t data = py::cast<int16_t>(obj);
      CHECK_RET_WITH_EXCEPT(memcpy_s(address->addr, address->size, &data, sizeof(int16_t)), EOK, "memcpy failed.");
      return;
    }
    case kNumberTypeUInt16: {
      uint8_t data = py::cast<uint8_t>(obj);
      CHECK_RET_WITH_EXCEPT(memcpy_s(address->addr, address->size, &data, sizeof(uint8_t)), EOK, "memcpy failed.");
      return;
    }
    case kNumberTypeInt32: {
      int32_t data = py::cast<int32_t>(obj);
      CHECK_RET_WITH_EXCEPT(memcpy_s(address->addr, address->size, &data, sizeof(int32_t)), EOK, "memcpy failed.");
      return;
    }
    case kNumberTypeUInt32: {
      uint32_t data = py::cast<uint32_t>(obj);
      CHECK_RET_WITH_EXCEPT(memcpy_s(address->addr, address->size, &data, sizeof(uint32_t)), EOK, "memcpy failed.");
      return;
    }
    case kNumberTypeInt64: {
      int64_t data = py::cast<int64_t>(obj);
      CHECK_RET_WITH_EXCEPT(memcpy_s(address->addr, address->size, &data, sizeof(int64_t)), EOK, "memcpy failed.");
      return;
    }
    case kNumberTypeUInt64: {
      uint64_t data = py::cast<uint64_t>(obj);
      CHECK_RET_WITH_EXCEPT(memcpy_s(address->addr, address->size, &data, sizeof(uint64_t)), EOK, "memcpy failed.");
      return;
    }
    case kNumberTypeFloat16: {
      float data = py::cast<float>(obj);
      Eigen::half_impl::__half_raw data_half = Eigen::half_impl::float_to_half_rtne(data);
      CHECK_RET_WITH_EXCEPT(memcpy_s(address->addr, address->size, &data_half.x, sizeof(data_half.x)), EOK,
                            "memcpy failed.");
      return;
    }
    case kNumberTypeFloat32: {
      float data = py::cast<float>(obj);
      CHECK_RET_WITH_EXCEPT(memcpy_s(address->addr, address->size, &data, sizeof(float)), EOK, "memcpy failed.");
      return;
    }
    case kNumberTypeFloat64: {
      double data = py::cast<double>(obj);
      CHECK_RET_WITH_EXCEPT(memcpy_s(address->addr, address->size, &data, sizeof(double)), EOK, "memcpy failed.");
      return;
    }
    default:
      MS_LOG(EXCEPTION) << "Type: " << type << " not supported.";
  }
}

void ArrayToRawMemory(const py::array &array, const AddressPtr &address) {
  MS_EXCEPTION_IF_NULL(address);
  if (static_cast<unsigned int>(array.flags()) &
      static_cast<unsigned int>(pybind11::detail::npy_api::NPY_ARRAY_C_CONTIGUOUS_)) {
    const py::buffer_info &buf_info = array.request();
    CHECK_RET_WITH_EXCEPT(
      common::huge_memcpy(reinterpret_cast<uint8_t *>(address->addr), address->size,
                          reinterpret_cast<uint8_t *>(buf_info.ptr), LongToSize(buf_info.size * buf_info.itemsize)),
      EOK, "memcpy failed.");
  } else {
    // Transform numpy array to row major buffer.
    Py_buffer pybuf;
    if (PyObject_GetBuffer(array.ptr(), &pybuf, PyBUF_ANY_CONTIGUOUS) != 0) {
      MS_LOG(EXCEPTION) << "Failed to get buffer from the input!";
    }

    auto buffer = std::make_unique<char[]>(LongToSize(pybuf.len));
    if (PyBuffer_ToContiguous(buffer.get(), &pybuf, pybuf.len, 'C')) {
      PyBuffer_Release(&pybuf);
      MS_LOG(EXCEPTION) << "Can't copy numpy.ndarray to a contiguous buffer.";
    }
    PyBuffer_Release(&pybuf);
    CHECK_RET_WITH_EXCEPT(common::huge_memcpy(reinterpret_cast<uint8_t *>(address->addr), address->size,
                                              reinterpret_cast<uint8_t *>(buffer.get()), LongToSize(pybuf.len)),
                          EOK, "memcpy failed.");
  }
}

void ObjectToRawMemory(const py::object &object, const PythonOjectType &object_type, const TypeId &data_type,
                       const AddressPtr &address) {
  switch (object_type) {
    case PythonOjectType::kScalar:
      return ScalarToRawMemory(object, data_type, address);
    case PythonOjectType::kNumpyArray:
      return ArrayToRawMemory(object.cast<py::array>(), address);
    default:
      MS_LOG(EXCEPTION) << "python object not supported. type: " << object_type;
  }
}

py::tuple RawMemoryToPyObjects(const std::vector<AddressPtr> &inputs, const PyFuncArgumentInfo &input_infos,
                               const std::vector<tensor::TensorPtr> &input_tensors) {
  py::tuple result(inputs.size());
  for (size_t i = 0; i < inputs.size(); i++) {
    switch (input_infos.object_types[i]) {
      case PythonOjectType::kScalar:
        MS_EXCEPTION_IF_NULL(inputs[i]);
        result[i] = RawMemoryToScalar(inputs[i]->addr, input_infos.dtypes[i]);
        break;
      case PythonOjectType::kNumpyArray: {
        const tensor::TensorPtr &tensor = input_tensors[i];
        MS_EXCEPTION_IF_NULL(tensor);
        CHECK_RET_WITH_EXCEPT(common::huge_memcpy(reinterpret_cast<uint8_t *>(tensor->data_c()), tensor->Size(),
                                                  reinterpret_cast<uint8_t *>(inputs[i]->addr), inputs[i]->size),
                              EOK, "memcpy failed.");
        result[i] = python_adapter::PyAdapterCallback::TensorToNumpy(*tensor);
        break;
      }
      default:
        MS_LOG(EXCEPTION) << "Python args not support. Index: " << i << ", type" << input_infos.object_types[i];
    }
  }
  return result;
}

void PyObjectToRawMemorys(const py::object &object, const PyFuncArgumentInfo &output_infos,
                          const std::vector<AddressPtr> &outputs) {
  // Single output.
  if (!py::isinstance<py::tuple>(object)) {
    if (outputs.size() != 1) {
      MS_LOG(EXCEPTION) << "The output num is 1, with " << outputs.size() << " expect.";
    }
    return ObjectToRawMemory(object, output_infos.object_types[0], output_infos.dtypes[0], outputs[0]);
  }

  // Multiply outputs.
  auto result_tuple = object.cast<py::tuple>();
  if (result_tuple.size() != outputs.size()) {
    MS_LOG(EXCEPTION) << "The output num is: " << result_tuple.size() << ", with " << outputs.size() << " expect.";
  }
  if (output_infos.object_types.size() != outputs.size() || output_infos.dtypes.size() != outputs.size()) {
    MS_LOG(EXCEPTION) << "The output info size is: " << output_infos.object_types.size() << " and "
                      << output_infos.dtypes.size() << ", with " << outputs.size() << " expect.";
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    ObjectToRawMemory(result_tuple[i], output_infos.object_types[i], output_infos.dtypes[i], outputs[i]);
  }
}
}  // namespace

void PyFuncCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  func_id_ = common::AnfAlgo::GetNodeAttr<int64_t>(kernel_node, "fn_id");
  fake_output_ = common::AnfAlgo::GetNodeAttr<bool>(kernel_node, "fake_output");
  single_scalar_output_ = common::AnfAlgo::GetNodeAttr<bool>(kernel_node, "single_scalar_output");
  BuildFuncInfo(kernel_node);
}

bool PyFuncCpuKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                const std::vector<AddressPtr> &outputs) {
  if (!init_) {
    py_func_ = GetPythonFunc();
    init_ = true;
  }

  return ExecuteKernel(inputs, outputs);
}

void PyFuncCpuKernelMod::BuildFuncInfo(const CNodePtr &kernel_node) {
  std::vector<TypeId> in_types;
  std::vector<TypeId> out_types;
  std::vector<std::vector<int64_t>> in_shapes;
  std::vector<std::vector<int64_t>> out_shapes;

  if (common::AnfAlgo::HasNodeAttr("in_types", kernel_node)) {
    const auto &in_type_ptrs = common::AnfAlgo::GetNodeAttr<std::vector<TypePtr>>(kernel_node, "in_types");
    (void)std::for_each(in_type_ptrs.begin(), in_type_ptrs.end(), [&in_types](auto p) {
      MS_EXCEPTION_IF_NULL(p);
      (void)in_types.emplace_back(p->type_id());
    });
  } else {
    in_types = AnfAlgo::GetAllInputDeviceTypes(kernel_node);
  }

  if (common::AnfAlgo::HasNodeAttr("out_types", kernel_node)) {
    const auto &out_type_ptrs = common::AnfAlgo::GetNodeAttr<std::vector<TypePtr>>(kernel_node, "out_types");
    (void)std::for_each(out_type_ptrs.begin(), out_type_ptrs.end(), [&out_types](auto p) {
      MS_EXCEPTION_IF_NULL(p);
      (void)out_types.emplace_back(p->type_id());
    });
  } else {
    out_types = AnfAlgo::GetAllOutputDeviceTypes(kernel_node);
  }

  if (common::AnfAlgo::HasNodeAttr("in_shapes", kernel_node)) {
    in_shapes = common::AnfAlgo::GetNodeAttr<std::vector<std::vector<int64_t>>>(kernel_node, "in_shapes");
  } else {
    for (size_t i = 0; i < common::AnfAlgo::GetInputTensorNum(kernel_node); i++) {
      auto in_shape = AnfAlgo::GetInputDeviceShape(kernel_node, i);
      (void)in_shapes.emplace_back(in_shape);
    }
  }

  if (common::AnfAlgo::HasNodeAttr("out_shapes", kernel_node)) {
    out_shapes = common::AnfAlgo::GetNodeAttr<std::vector<std::vector<int64_t>>>(kernel_node, "out_shapes");
  } else {
    for (size_t i = 0; i < AnfAlgo::GetOutputTensorNum(kernel_node); i++) {
      auto out_shape = AnfAlgo::GetOutputDeviceShape(kernel_node, i);
      (void)out_shapes.emplace_back(out_shape);
    }
  }

  if (in_shapes.size() != in_types.size()) {
    MS_LOG(EXCEPTION) << "Input shapes'size is " << in_shapes.size() << ", while input types' size is "
                      << in_types.size();
  }
  if (out_shapes.size() != out_types.size()) {
    MS_LOG(EXCEPTION) << "Output shapes'size is " << out_shapes.size() << ", while output types' size is "
                      << out_types.size();
  }

  input_infos_.dtypes = in_types;
  input_infos_.shapes = in_shapes;
  for (size_t i = 0; i < in_shapes.size(); i++) {
    auto tensor = std::make_shared<tensor::Tensor>(in_types[i], in_shapes[i]);
    input_tensors_.push_back(tensor);

    const auto &object_type = in_shapes[i].empty() ? PythonOjectType::kScalar : PythonOjectType::kNumpyArray;
    (void)input_infos_.object_types.emplace_back(object_type);
  }

  output_infos_.dtypes = out_types;
  output_infos_.shapes = out_shapes;
  if (single_scalar_output_) {
    (void)output_infos_.object_types.emplace_back(PythonOjectType::kScalar);
  } else {
    for (size_t j = 0; j < out_shapes.size(); j++) {
      const auto &object_type = out_shapes[j].empty() ? PythonOjectType::kScalar : PythonOjectType::kNumpyArray;
      (void)output_infos_.object_types.emplace_back(object_type);
    }
  }
}

bool PyFuncCpuKernelMod::ExecuteKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs) {
  if (Py_IsInitialized() != true) {
    MS_LOG(ERROR) << "Py_IsInitialized failed.";
    return false;
  }

  py::gil_scoped_acquire gil_acquire;
  py::object result;
  if (inputs.size()) {
    py::tuple args = RawMemoryToPyObjects(inputs, input_infos_, input_tensors_);
    result = py_func_(*args);
  } else {
    result = py_func_();
  }

  if (fake_output_) {
    if (result.is_none()) {
      return true;
    } else {
      MS_LOG(ERROR) << "This CustomPyfunc must have no outputs, but got 1";
      return false;
    }
  }

  PyObjectToRawMemorys(result, output_infos_, outputs);

  return true;
}

py::function PyFuncCpuKernelMod::GetPythonFunc() const {
  py::gil_scoped_acquire gil_acquire;
  static const std::string &module_name = "mindspore.ops.operations._pyfunc_registry";
  static const std::string &entrance = "get_pyfunc";
  py::module module = py::module::import(module_name.c_str());
  py::object get_pyfunc_obj = module.attr(entrance.c_str());
  if (get_pyfunc_obj.is_none()) {
    MS_LOG(EXCEPTION) << "Cannot find a python function named " << entrance << "in module" << module_name;
  }

  py::function get_pyfunc = get_pyfunc_obj.cast<py::function>();
  py::object py_func_obj = get_pyfunc(py::int_(func_id_));
  if (py_func_obj.is_none()) {
    MS_LOG(EXCEPTION) << "Cannot find python func with id: " << func_id_;
  }

  return py_func_obj.cast<py::function>();
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, PyFunc, PyFuncCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
