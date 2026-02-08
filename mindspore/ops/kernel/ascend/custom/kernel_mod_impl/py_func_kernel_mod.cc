/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "kernel/ascend/custom/kernel_mod_impl/py_func_kernel_mod.h"

#include <algorithm>
#include <string>
#include <utility>

#include "acl/acl.h"
#include "include/utils/anfalgo.h"
#include "include/utils/convert_utils_py.h"
#include "include/utils/tensor_py.h"
#include "ir/tensor.h"
#include "ir/tensor_new.h"
#include "ops/op_def.h"
#include "runtime/pipeline/pipeline.h"

namespace mindspore::kernel {
using mindspore::ops::OP_DTYPE;
using mindspore::tensor::TensorPtr;
namespace {

ValuePtr ValueFromScalarKernelTensor(const kernel::KernelTensor *kernel_tensor) {
  MS_EXCEPTION_IF_NULL(kernel_tensor);
  if (kernel_tensor->GetValueTrack() != nullptr && !kernel_tensor->GetValueTrack()->isa<ValueAny>()) {
    return kernel_tensor->GetValueTrack();
  }

  if (IsShapeEmpty(kernel_tensor->GetShapeVector())) {
    auto type_id =
      (kernel_tensor->dtype_id() == TypeId::kTypeUnknown ? TypeId::kNumberTypeInt64 : kernel_tensor->dtype_id());
    return tensor::from_spec(type_id, kernel_tensor->GetShapeVector(), device::DeviceType::kAscend);
  }

  MS_LOG(DEBUG) << "Type:" << kernel_tensor->dtype_id() << " shape:" << kernel_tensor->GetShapeVector()
                << " size:" << kernel_tensor->size();

  auto real_value = kernel_tensor->GetValue();
  MS_EXCEPTION_IF_NULL(real_value);
  if (!real_value->isa<KernelTensorValue>()) {
    MS_LOG(EXCEPTION) << "Invalid kernel tensor value: " << real_value->ToString();
  }

  if (kernel_tensor->GetType() != nullptr && kernel_tensor->GetType()->isa<Number>()) {
    auto kernel_tensor_value = real_value->cast<KernelTensorValuePtr>();
    MS_EXCEPTION_IF_NULL(kernel_tensor_value);
    return common::AnfAlgo::ValueToScalar(kernel_tensor_value, kernel_tensor->GetType()->type_id());
  }

  return std::make_shared<tensor::Tensor>(kernel_tensor->dtype_id(), kernel_tensor->GetShapeVector(),
                                          kernel_tensor->device_address());
}

ValuePtr ValueFromKernelTensorList(const std::vector<kernel::KernelTensor *> &kernel_tensors) {
  std::vector<ValuePtr> values;
  values.reserve(kernel_tensors.size());
  std::transform(kernel_tensors.begin(), kernel_tensors.end(), std::back_inserter(values),
                 [](const kernel::KernelTensor *t) { return ValueFromScalarKernelTensor(t); });
  return std::make_shared<ValueTuple>(std::move(values));
}

template <typename T>
ValuePtr MakeScalarValue(kernel::KernelTensor *kt) {
  return MakeValue(kt->GetValueWithCheck<T>());
}

template <typename T>
ValuePtr MakeVectorValue(kernel::KernelTensor *kt) {
  return MakeValue(kt->GetValueWithCheck<std::vector<T>>());
}

using Handler = std::function<ValuePtr(const std::vector<kernel::KernelTensor *> &)>;

const std::unordered_map<OP_DTYPE, Handler> kKernelHandlers = {
  // Scalar
  {OP_DTYPE::DT_BOOL, [](const std::vector<kernel::KernelTensor *> &v) { return MakeScalarValue<bool>(v[0]); }},
  {OP_DTYPE::DT_INT, [](const std::vector<kernel::KernelTensor *> &v) { return MakeScalarValue<int64_t>(v[0]); }},
  {OP_DTYPE::DT_FLOAT, [](const std::vector<kernel::KernelTensor *> &v) { return MakeScalarValue<float>(v[0]); }},
  {OP_DTYPE::DT_STR, [](const std::vector<kernel::KernelTensor *> &v) { return MakeScalarValue<std::string>(v[0]); }},
  {OP_DTYPE::DT_NUMBER, [](const std::vector<kernel::KernelTensor *> &v) { return ValueFromScalarKernelTensor(v[0]); }},

  // Tensor
  {OP_DTYPE::DT_TENSOR, [](const std::vector<kernel::KernelTensor *> &v) { return ValueFromScalarKernelTensor(v[0]); }},

  // Tensor sequence
  {OP_DTYPE::DT_LIST_TENSOR, &ValueFromKernelTensorList},
  {OP_DTYPE::DT_TUPLE_TENSOR, &ValueFromKernelTensorList},

  // Sequence of scalars
  {OP_DTYPE::DT_LIST_BOOL, [](const std::vector<kernel::KernelTensor *> &v) { return MakeVectorValue<bool>(v[0]); }},
  {OP_DTYPE::DT_TUPLE_BOOL, [](const std::vector<kernel::KernelTensor *> &v) { return MakeVectorValue<bool>(v[0]); }},
  {OP_DTYPE::DT_LIST_INT, [](const std::vector<kernel::KernelTensor *> &v) { return MakeVectorValue<int64_t>(v[0]); }},
  {OP_DTYPE::DT_TUPLE_INT, [](const std::vector<kernel::KernelTensor *> &v) { return MakeVectorValue<int64_t>(v[0]); }},
  {OP_DTYPE::DT_LIST_FLOAT, [](const std::vector<kernel::KernelTensor *> &v) { return MakeVectorValue<float>(v[0]); }},
  {OP_DTYPE::DT_TUPLE_FLOAT, [](const std::vector<kernel::KernelTensor *> &v) { return MakeVectorValue<float>(v[0]); }},
  {OP_DTYPE::DT_LIST_STR,
   [](const std::vector<kernel::KernelTensor *> &v) { return MakeVectorValue<std::string>(v[0]); }},
  {OP_DTYPE::DT_TUPLE_STR,
   [](const std::vector<kernel::KernelTensor *> &v) { return MakeVectorValue<std::string>(v[0]); }},
};

ValuePtr ValueFromKernelTensor(const std::vector<kernel::KernelTensor *> &kernel_tensors, OP_DTYPE data_type) {
  auto it = kKernelHandlers.find(data_type);
  if (it == kKernelHandlers.end()) {
    MS_LOG(EXCEPTION) << "Unsupported data type for ValueFromKernelTensor: " << data_type;
  }
  return it->second(kernel_tensors);
}

std::vector<std::vector<KernelTensor *>> GroupInputs(const std::vector<KernelTensor *> &inputs,
                                                     const std::vector<int64_t> &group_sizes) {
  std::vector<std::vector<KernelTensor *>> groups;
  groups.reserve(group_sizes.size());

  size_t offset = 0;
  for (int64_t sz : group_sizes) {
    auto usize = LongToSize(sz);
    if (offset + usize > inputs.size()) {
      MS_LOG(EXCEPTION) << "Dynamic input sizes exceed the number of available inputs. "
                        << "Expected at least " << (offset + usize) << ", got " << inputs.size();
    }
    groups.emplace_back(inputs.begin() + offset, inputs.begin() + offset + usize);
    offset += usize;
  }

  if (offset != inputs.size()) {
    MS_LOG(EXCEPTION) << "Dynamic input sizes do not cover all inputs. "
                      << "Remaining ungrouped tensors: " << (inputs.size() - offset);
  }
  return groups;
}

}  // namespace

bool PyFuncKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  func_id_ = GetValue<int64_t>(primitive_->GetAttr("fn_id"));
  py_func_ = GetPythonFunc();
  return true;
}

bool PyFuncKernelMod::Launch(const std::vector<KernelTensor *> &inputs,
                             const std::vector<KernelTensor *> & /* workspace */,
                             const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  if (Py_IsInitialized() == 0) {
    MS_LOG(ERROR) << "Python interpreter is not initialized.";
    return false;
  }

  pybind11::gil_scoped_acquire gil_acquire;
  pybind11::tuple py_args = PreprocessInputs(inputs);
  pybind11::object result;

  try {
    result = inputs.empty() ? py_func_() : py_func_(*py_args);
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "Python function call failed: " << e.what();
    return false;
  }

  runtime::Pipeline::Get().WaitForward();
  return PostprocessOutputs(result, outputs, stream_ptr);
}

pybind11::function PyFuncKernelMod::GetPythonFunc() const {
  pybind11::gil_scoped_acquire gil_acquire;
  static const std::string kModuleName = "mindspore.ops.operations._pyfunc_registry";
  static const std::string kEntrance = "get_pyfunc";

  pybind11::module mod = pybind11::module::import(kModuleName.c_str());
  pybind11::object getter = mod.attr(kEntrance.c_str());
  if (getter.is_none()) {
    MS_LOG(EXCEPTION) << "Cannot find function '" << kEntrance << "' in module " << kModuleName;
  }

  pybind11::function get_pyfunc = getter.cast<pybind11::function>();
  pybind11::object func_obj = get_pyfunc(pybind11::int_(func_id_));
  if (func_obj.is_none()) {
    MS_LOG(EXCEPTION) << "Cannot find Python function with id " << func_id_;
  }

  return func_obj.cast<pybind11::function>();
}

pybind11::tuple PyFuncKernelMod::PreprocessInputs(const std::vector<KernelTensor *> &inputs) {
  auto group_sizes_value = primitive_->GetAttr("grouped_info");
  MS_EXCEPTION_IF_NULL(group_sizes_value);
  auto group_sizes = GetValue<std::vector<int64_t>>(group_sizes_value);
  auto grouped_inputs = GroupInputs(inputs, group_sizes);

  MS_EXCEPTION_IF_NULL(primitive_);
  const auto &op_name = primitive_->name();
  auto op_def = mindspore::ops::GetOpDef(op_name);
  MS_EXCEPTION_IF_NULL(op_def);
  const auto &input_args = op_def->args_;

  if (grouped_inputs.size() != input_args.size()) {
    MS_LOG(EXCEPTION) << "Grouped input count (" << grouped_inputs.size() << ") does not match OpDef input count ("
                      << input_args.size() << ")";
  }

  pybind11::tuple py_args(grouped_inputs.size());
  try {
    for (size_t i = 0; i < grouped_inputs.size(); ++i) {
      py_args[i] = ValueToPyData(ValueFromKernelTensor(grouped_inputs[i], input_args[i].arg_dtype_));
    }
  } catch (const std::exception &e) {
    MS_LOG(EXCEPTION) << "Failed to convert input " << grouped_inputs.size() << " to Python: " << e.what();
  }
  return py_args;
}

bool PyFuncKernelMod::PostprocessOutputs(pybind11::handle result, const std::vector<KernelTensor *> &outputs,
                                         void *stream_ptr) {
  if (outputs.empty()) {
    MS_LOG(ERROR) << "No output tensors provided.";
    return false;
  }

  std::vector<TensorPtr> cpp_tensors;
  try {
    if (tensor::IsTensorPy(result)) {
      cpp_tensors.push_back(pybind11::cast<TensorPtr>(result));
    } else if (pybind11::isinstance<pybind11::tuple>(result) || pybind11::isinstance<pybind11::list>(result)) {
      auto seq = pybind11::cast<pybind11::sequence>(result);
      cpp_tensors.reserve(seq.size());
      for (auto &&item : seq) {
        if (!tensor::IsTensorPy(item)) {
          MS_LOG(ERROR) << "Returned sequence contains non-tensor element.";
          return false;
        }
        cpp_tensors.push_back(pybind11::cast<TensorPtr>(item));
      }
    } else {
      MS_LOG(ERROR) << "Python kernel must return a tensor or a sequence of tensors.";
      return false;
    }
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "Failed to convert Python result to C++ tensors: " << e.what();
    return false;
  }

  if (cpp_tensors.size() != outputs.size()) {
    MS_LOG(ERROR) << "Tensor count mismatch: kernel returned " << cpp_tensors.size() << ", expected " << outputs.size();
    return false;
  }

  for (size_t i = 0; i < outputs.size(); ++i) {
    const auto &tensor = cpp_tensors[i];
    MS_EXCEPTION_IF_NULL(tensor);
    auto dev_addr = tensor->device_address();
    MS_EXCEPTION_IF_NULL(dev_addr);
    void *src_ptr = dev_addr->GetMutablePtr();
    MS_EXCEPTION_IF_NULL(src_ptr);

    MS_LOG(DEBUG) << "Output[" << i << "] shape=" << tensor->shape() << " dtype=" << tensor->data_type()
                  << " src_ptr=" << src_ptr << " dst_ptr=" << outputs[i]->device_ptr() << " size=" << tensor->Size();

    if (outputs[i]->size() != tensor->Size()) {
      MS_LOG(ERROR)
        << "Output[" << i << "] size mismatch: kernel returned " << tensor->Size() << " bytes, expected "
        << outputs[i]->size() << " bytes. "
        << "Returned tensor: " << tensor->ToString() << "; "
        << "Infer tensor: " << outputs[i]->ToString() << ". "
        << "Please (1) verify the infer function is consistent with the Python implementation, "
        << "(2) ensure the Python callback does not perform unintended cast/astype or other implicit conversion.";
      return false;
    }

    auto status = aclrtMemcpyAsync(outputs[i]->device_ptr(), outputs[i]->size(), src_ptr, tensor->Size(),
                                   ACL_MEMCPY_DEVICE_TO_DEVICE, stream_ptr);
    if (status != ACL_ERROR_NONE) {
      MS_LOG(ERROR) << "aclrtMemcpyAsync failed for output[" << i << "], ret=0x" << std::hex << status;
      return false;
    }
  }
  return true;
}

}  // namespace mindspore::kernel
