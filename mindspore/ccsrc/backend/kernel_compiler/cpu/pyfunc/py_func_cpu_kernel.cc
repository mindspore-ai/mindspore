/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "backend/kernel_compiler/cpu/pyfunc/py_func_cpu_kernel.h"

#include <memory>
#include <vector>
#include "Eigen/Core"
#include "Eigen/src/Core/arch/CUDA/Half.h"
#include "abstract/utils.h"
#include "runtime/device/cpu/cpu_common.h"
#include "pybind_api/ir/tensor_py.h"
#include "backend/kernel_compiler/cpu/cpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
namespace {
py::object RawMemoryToScalar(const void *data, const TypePtr &type) {
  switch (type->type_id()) {
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
      MS_LOG(EXCEPTION) << "Type: " << type->type_id() << " not supported.";
  }
}

void ScalarToRawMemory(const py::object &obj, const TypePtr &type, const AddressPtr &address) {
  switch (type->type_id()) {
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
      MS_LOG(EXCEPTION) << "Type: " << type->type_id() << " not supported.";
  }
}

void ArrayToRawMemory(const py::array &array, const AddressPtr &address) {
  if (static_cast<unsigned int>(array.flags()) & pybind11::detail::npy_api::NPY_ARRAY_C_CONTIGUOUS_) {
    const py::buffer_info &buf_info = array.request();
    CHECK_RET_WITH_EXCEPT(memcpy_s(address->addr, address->size, buf_info.ptr, buf_info.size * buf_info.itemsize), EOK,
                          "memcpy failed.");
  } else {
    // Transform numpy array to row major buffer.
    Py_buffer pybuf;
    if (PyObject_GetBuffer(array.ptr(), &pybuf, PyBUF_ANY_CONTIGUOUS)) {
      MS_LOG(EXCEPTION) << "Failed to get buffer from the input!";
    }

    auto buffer = std::make_unique<char[]>(LongToSize(pybuf.len));
    if (PyBuffer_ToContiguous(buffer.get(), &pybuf, pybuf.len, 'C')) {
      PyBuffer_Release(&pybuf);
      MS_LOG(EXCEPTION) << "Can't copy numpy.ndarray to a contiguous buffer.";
    }
    PyBuffer_Release(&pybuf);
    CHECK_RET_WITH_EXCEPT(memcpy_s(address->addr, address->size, buffer.get(), LongToSize(pybuf.len)), EOK,
                          "memcpy failed.");
  }
}

void ObjectToRawMemory(const py::object &object, const PythonOjectType &object_type, const TypePtr &data_type,
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
        result[i] = RawMemoryToScalar(inputs[i]->addr, input_infos.dtypes[i]);
        break;
      case PythonOjectType::kNumpyArray: {
        const tensor::TensorPtr &tensor = input_tensors[i];
        CHECK_RET_WITH_EXCEPT(memcpy_s(tensor->data_c(), tensor->Size(), inputs[i]->addr, inputs[i]->size), EOK,
                              "memcpy failed.");
        result[i] = tensor::TensorPy::AsNumpy(*tensor);
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

  for (size_t i = 0; i < outputs.size(); i++) {
    ObjectToRawMemory(result_tuple[i], output_infos.object_types[i], output_infos.dtypes[i], outputs[i]);
  }
}
}  // namespace

void PyFuncCpuKernel::InitKernel(const CNodePtr &kernel_node) {
  func_id_ = AnfAlgo::GetNodeAttr<int64_t>(kernel_node, "fn_id");
  BuildFuncInfo(kernel_node);
}

bool PyFuncCpuKernel::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                             const std::vector<AddressPtr> &outputs) {
  if (!init_) {
    py_func_ = GetPythonFunc(func_id_);
    init_ = true;
  }

  return ExecuteKernel(inputs, outputs);
}

void PyFuncCpuKernel::BuildFuncInfo(const CNodePtr &kernel_node) {
  const auto &in_shapes = AnfAlgo::GetNodeAttr<std::vector<std::vector<int64_t>>>(kernel_node, "in_shapes");
  const auto &in_types = AnfAlgo::GetNodeAttr<std::vector<TypePtr>>(kernel_node, "in_types");
  const auto &out_shapes = AnfAlgo::GetNodeAttr<std::vector<std::vector<int64_t>>>(kernel_node, "out_shapes");
  const auto &out_types = AnfAlgo::GetNodeAttr<std::vector<TypePtr>>(kernel_node, "out_types");

  input_infos_.dtypes = in_types;
  input_infos_.shapes = in_shapes;
  for (size_t i = 0; i < in_shapes.size(); i++) {
    auto tensor = std::make_shared<tensor::Tensor>(in_types[i]->type_id(), in_shapes[i]);
    input_tensors_.push_back(tensor);

    const auto &object_type = in_shapes[i].empty() ? PythonOjectType::kScalar : PythonOjectType::kNumpyArray;
    (void)input_infos_.object_types.emplace_back(object_type);
  }

  output_infos_.dtypes = out_types;
  output_infos_.shapes = out_shapes;
  for (size_t j = 0; j < out_shapes.size(); j++) {
    const auto &object_type = out_shapes[j].empty() ? PythonOjectType::kScalar : PythonOjectType::kNumpyArray;
    (void)output_infos_.object_types.emplace_back(object_type);
  }
}

bool PyFuncCpuKernel::ExecuteKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs) {
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

  if (output_infos_.shapes.empty()) {
    return true;
  }

  PyObjectToRawMemorys(result, output_infos_, outputs);
  return true;
}

py::function PyFuncCpuKernel::GetPythonFunc(const int64_t &func_id) {
  py::gil_scoped_acquire gil_acquire;
  static const std::string &module_name = "mindspore.ops.operations.other_ops";
  static const std::string &func_name = "get_pyfunc";
  py::module module = py::module::import(module_name.c_str());
  py::object get_pyfunc_obj = module.attr(func_name.c_str());
  if (get_pyfunc_obj.is_none()) {
    MS_LOG(EXCEPTION) << "Cannot find a python function named " << func_name << "in module" << module_name;
  }

  py::function get_pyfunc = get_pyfunc_obj.cast<py::function>();
  py::object py_func_obj = get_pyfunc(py::int_(func_id));
  if (py_func_obj.is_none()) {
    MS_LOG(EXCEPTION) << "Cannot find python func with id: " << func_id;
  }

  return py_func_obj.cast<py::function>();
}

MS_REG_CPU_KERNEL(PyFunc, KernelAttr(), PyFuncCpuKernel)
}  // namespace kernel
}  // namespace mindspore
