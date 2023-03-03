/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/pyexecute/py_execute_cpu_kernel.h"

#include <memory>
#include <vector>
#include <utility>

#include "Eigen/Core"
#include "abstract/utils.h"
#include "ir/anf.h"
#include "plugin/device/cpu/hal/device/cpu_common.h"
#include "include/common/fallback.h"
#include "include/common/utils/python_adapter.h"
#include "include/common/utils/convert_utils_py.h"
#include "include/common/utils/python_fallback_running.h"
#include "plugin/factory/ms_factory.h"
#include "mindspore/ccsrc/pipeline/jit/parse/resolve.h"
#include "utils/trace_base.h"

namespace mindspore {
namespace kernel {
namespace {
py::object CallPythonGetGlobalParams() {
  constexpr auto python_mod_parse = "mindspore._extends.parse";  // The same as PYTHON_MOD_PARSE_MODULE[]
  py::module mod = python_adapter::GetPyModule(python_mod_parse);
  constexpr auto python_get_dict = "get_global_params";
  return python_adapter::CallPyModFn(mod, python_get_dict);
}

// Call the python script string. The same codes as parse/data_converter.h, we must copy it here.
py::object CallPythonScript(const py::object &script, const py::tuple &args_kwargs) {
  constexpr auto python_mod_parse = "mindspore._extends.parse";  // The same as PYTHON_MOD_PARSE_MODULE[]
  py::module mod = python_adapter::GetPyModule(python_mod_parse);
  constexpr auto python_mode_eval = "eval_script";
  // The `args_kwargs` is a tuple(dict(global), dict(local)).
  return python_adapter::CallPyModFn(mod, python_mode_eval, script, args_kwargs);
}
}  // namespace

void PyExecuteCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_LOG(DEBUG) << "kernel_node: " << kernel_node << ", " << kernel_node->DebugString();
  inputs_info_.clear();
  kernel_node_ = kernel_node;
  for (size_t i = 1; i < kernel_node->size(); ++i) {
    const auto &input = kernel_node->inputs()[i];

    // Check if PyExecuteOutputUserData exists.
    py::object obj = py::none();
    if (input->has_user_data<PyExecuteOutputUserData>()) {
      py::gil_scoped_acquire gil_acquire;
      const auto &output_data = input->user_data<PyExecuteOutputUserData>();
      obj = output_data->obj;
      MS_LOG(DEBUG) << "Has \'PyExecuteOutputUserData\', obj: " << obj;
    }

    // Record the inputs' information by their abstract types.
    const auto &input_abstract = input->abstract();
    if (input_abstract->isa<abstract::AbstractMonad>()) {
      continue;
    }
    MS_EXCEPTION_IF_NULL(input_abstract);
    if (input_abstract->isa<abstract::AbstractRefTensor>()) {
      const auto &param = dyn_cast<Parameter>(input);
      MS_EXCEPTION_IF_NULL(param);
      MS_LOG(DEBUG) << "AbstractRefTensor, input[" << i << "]: " << param->default_param()->ToString();
      (void)inputs_info_.emplace_back(PyExecuteInputInfo({obj, input_abstract, kTypeUnknown, {}}));
    } else if (input_abstract->isa<abstract::AbstractTensor>()) {
      const auto &tensor_abstract = dyn_cast<abstract::AbstractTensor>(input_abstract);
      MS_EXCEPTION_IF_NULL(tensor_abstract);
      MS_LOG(DEBUG) << "AbstractTensor, input[" << i << "]: " << tensor_abstract->BuildType()->ToString() << ", "
                    << tensor_abstract->BuildShape()->ToString();
      const auto &in_type = AnfAlgo::GetInputDeviceDataType(kernel_node, i - 1);
      const auto &in_shape = AnfAlgo::GetInputDeviceShape(kernel_node, i - 1);
      (void)inputs_info_.emplace_back(PyExecuteInputInfo({obj, input_abstract, in_type, in_shape}));
    } else {
      MS_LOG(DEBUG) << "Other, input[" << i << "]: " << input->DebugString() << ", " << input_abstract->ToString();
      (void)inputs_info_.emplace_back(PyExecuteInputInfo({obj, input_abstract, kTypeUnknown, {}}));
    }
    MS_LOG(DEBUG) << "Kernel node's input[" << i << "]: " << input->DebugString() << ", " << input_abstract->ToString();
  }
}

void PyExecuteCpuKernelMod::AttachPyOutputData(const py::object &py_res) {
  const auto &py_output = std::make_shared<PyExecuteOutputUserData>();
  py_output->obj = py_res;
  // Set Python data for kernel node.
  kernel_node_->set_user_data<PyExecuteOutputUserData>(py_output);

  // Set Python data for front node.
  const auto &kernel_graph = std::dynamic_pointer_cast<session::KernelGraph>(kernel_node_->func_graph());
  MS_EXCEPTION_IF_NULL(kernel_graph);
  const auto &graph_output_map = kernel_graph->graph_output_map();
  session::AnfWithOutIndex anf_index = std::make_pair(kernel_node_, 0);
  const auto &iter = graph_output_map.find(anf_index);
  if (iter != graph_output_map.cend()) {
    const auto &front_node = iter->second.first;
    MS_LOG(INFO) << "Found front output for " << kernel_node_ << ", " << kernel_node_->DebugString();
    front_node->set_user_data<PyExecuteOutputUserData>(py_output);
  } else {
    MS_LOG(DEBUG) << "Not found, kernel node is not output, " << kernel_node_ << ", " << kernel_node_->DebugString();
    if (!IS_OUTPUT_ON(mindspore::kDebug)) {
      return;
    }
    for (const auto &output_pair : graph_output_map) {
      MS_EXCEPTION_IF_NULL(output_pair.first.first);
      MS_EXCEPTION_IF_NULL(output_pair.second.first);
      MS_LOG(DEBUG) << "backend node: " << output_pair.first.first << ", " << output_pair.first.first->DebugString()
                    << ", front node: " << output_pair.second.first << ", " << output_pair.second.first->DebugString();
    }
  }
}

std::pair<ValueTuplePtr, ValueTuplePtr> GetKeyAndValueArgsStructural(const ValuePtr &value) {
  if (value == nullptr) {
    MS_LOG(DEBUG) << "Structural info is miss return default structural.";
    auto structural = std::make_shared<ValueTuple>(std::vector<ValuePtr>{MakeValue<int64_t>(-1)});
    return std::make_pair(structural, structural);
  }
  auto tuple_structural_value = value->cast<ValueTuplePtr>();
  MS_EXCEPTION_IF_NULL(tuple_structural_value);
  constexpr auto tuple_structural_size = 3;
  if (tuple_structural_value->size() != tuple_structural_size) {
    MS_LOG(EXCEPTION) << "The " << kAttrDynInputSizes << " of PyExec should be a value tuple of size 3, but got "
                      << tuple_structural_value->ToString();
  }
  auto key_size_structural = (*tuple_structural_value)[1];
  auto value_size_structural = (*tuple_structural_value)[2];
  auto key_structural = key_size_structural->cast<ValueTuplePtr>();
  auto value_structural = value_size_structural->cast<ValueTuplePtr>();
  MS_EXCEPTION_IF_NULL(key_structural);
  MS_EXCEPTION_IF_NULL(value_structural);
  if (key_structural->size() != value_structural->size()) {
    MS_LOG(EXCEPTION) << "The key size must equal with the value size.";
  }
  return std::make_pair(key_structural, value_structural);
}

std::string ConstructLocalDictKey(const abstract::AbstractBasePtr &key_abs) {
  MS_EXCEPTION_IF_NULL(key_abs);
  const auto &input_type = key_abs->BuildType();
  MS_EXCEPTION_IF_NULL(input_type);
  if (key_abs->isa<abstract::AbstractScalar>() && input_type->isa<String>()) {
    const auto &value = key_abs->BuildValue();
    MS_EXCEPTION_IF_NULL(value);
    const auto &str_value = dyn_cast<StringImm>(value);
    MS_EXCEPTION_IF_NULL(str_value);
    const auto &str = str_value->value();
    return str;
  }
  MS_LOG(EXCEPTION) << "input should be a string but got :" << key_abs->type_name() << ", abs:" << key_abs->ToString();
}

py::object GenerateElementOfLocalDictValue(const AddressPtr &input, const PyExecuteInputInfo &input_info) {
  MS_EXCEPTION_IF_NULL(input);
  const auto &input_abstract = input_info.abstract;
  MS_EXCEPTION_IF_NULL(input_abstract);
  const auto &input_type = input_abstract->BuildType();
  MS_EXCEPTION_IF_NULL(input_type);
  if (input_abstract->isa<abstract::AbstractScalar>() && input_type->isa<String>()) {
    const auto &value = input_abstract->BuildValue();
    MS_EXCEPTION_IF_NULL(value);
    const auto &str_value = dyn_cast<StringImm>(value);
    MS_EXCEPTION_IF_NULL(str_value);
    const auto &str = str_value->value();
    MS_LOG(DEBUG) << "String, input" << input_abstract->ToString();
    return py::str(str);
  } else if (input_abstract->isa<abstract::AbstractTensor>()) {
    const auto &py_array_value = input_info.py_obj_output;
    bool is_py_middle_data = !py::isinstance<py::none>(py_array_value);
    MS_LOG(DEBUG) << "Tensor: " << input_abstract->ToString() << ", type: " << input_info.type
                  << ", shape: " << input_info.shape << ", addr: " << input->addr << ", size: " << input->size
                  << ", py_array_value: " << py_array_value << ", is_py_middle_data: " << is_py_middle_data;
    tensor::TensorPtr tensor = nullptr;
    if (!is_py_middle_data) {
      tensor = std::make_shared<tensor::Tensor>(input_info.type, input_info.shape, input->addr, input->size);
      return py::cast(tensor);
    }
    return py_array_value;
  }
  MS_LOG(EXCEPTION) << "unsupported value type." << input_abstract->ToString();
}

std::pair<py::object, size_t> ConstructLocalDictValue(
  const std::vector<AddressPtr>::const_iterator &addr_begin_iter,
  const std::vector<AddressPtr>::const_iterator &addr_end_iter,
  const std::vector<PyExecuteInputInfo>::const_iterator &info_begin_iter,
  const std::vector<PyExecuteInputInfo>::const_iterator &info_end_iter, const ValuePtr &structural) {
  if (addr_begin_iter == addr_end_iter) {
    MS_LOG(EXCEPTION) << "The address is out of range.";
  }
  if (info_begin_iter == info_end_iter) {
    MS_LOG(EXCEPTION) << "PyExecute Input info is out of range.";
  }
  MS_EXCEPTION_IF_NULL(structural);
  size_t offset = 0;
  if (structural->isa<Scalar>()) {
    offset = 1;
    const auto &input = *addr_begin_iter;
    const auto &input_info = *info_begin_iter;
    const auto &obj = GenerateElementOfLocalDictValue(input, input_info);
    return std::make_pair(obj, offset);
  } else if (structural->isa<ValueTuple>()) {
    auto tuple_structural = structural->cast_ptr<ValueTuple>();
    py::tuple py_args(tuple_structural->size());
    for (size_t i = 0; i < tuple_structural->size(); ++i) {
      auto element = (*tuple_structural)[i];
      const auto &res = ConstructLocalDictValue(addr_begin_iter + offset, addr_end_iter, info_begin_iter + offset,
                                                info_end_iter, element);
      offset += res.second;
      py_args[i] = res.first;
    }
    return std::make_pair(py_args, offset);
  }
  MS_LOG(EXCEPTION) << "The structural must be all Scalar or ValueTuple but got " << structural->type_name()
                    << ", value :" << structural->ToString();
}

py::dict PyExecuteCpuKernelMod::BuildLocalParameters(const std::vector<AddressPtr> &inputs) {
  auto prim = GetCNodePrimitive(cnode_ptr_.lock());
  // dyn_attr [-1(script), (key size), (value size)] and the value size may be a value tuple
  // record the input of real key struct
  // eg: the argument dict is {key1 : 1, key2 : (1,2), key3 :((1,2),3,4)} the front node is
  // {PyExec, Script, (key1, key2, key3), (1,(1,2),((1,2),3,4))}
  // the backend node is {PyExec, Script, key1, key2, key3, 1, 1, 2, 1, 2, 3, 4}
  // and the TupleInputStructural of PyExec is (-1, -1, (-1, (-1,-1), ((-1,-1),-1,-1)))
  auto tuple_structural = prim->GetAttr(kAttrTupleInputStructural);
  auto [key_structural, value_structural] = GetKeyAndValueArgsStructural(tuple_structural);
  MS_LOG(DEBUG) << "Value structural :" << value_structural->ToString();
  // To call the script with global and local parameters.
  py::dict local_dict;
  size_t offset_index = key_structural->size() + 1;
  size_t value_index = 0;
  for (size_t i = 0; i < key_structural->size(); ++i) {
    // skip the script address
    size_t key_index = i + 1;
    const auto &key_address = inputs.at(key_index);
    MS_EXCEPTION_IF_NULL(key_address);
    const auto &key_input_info = inputs_info_.at(key_index);
    const auto &key_abstract = key_input_info.abstract;
    const auto &key = ConstructLocalDictKey(key_abstract);
    MS_LOG(DEBUG) << "String, input[" << i << "]: " << key_abstract->ToString() << ", got the key :" << key;
    // Get Values
    auto structural = (*value_structural)[value_index++];
    MS_LOG(DEBUG) << "value structural :" << structural->ToString();

    const auto &value_with_offset = ConstructLocalDictValue(
      inputs.begin() + offset_index, inputs.end(), inputs_info_.begin() + offset_index, inputs_info_.end(), structural);
    offset_index += value_with_offset.second;
    local_dict[py::str(key)] = value_with_offset.first;
  }
  MS_LOG(DEBUG) << kernel_node_->DebugString() << " local_dict: " << local_dict;
  return local_dict;
}

void TensorToRawMemory(const tensor::TensorPtr &tensor, const AddressPtr &address) {
  MS_EXCEPTION_IF_NULL(tensor);
  MS_EXCEPTION_IF_NULL(address);
  const auto &res = memcpy_s(address->addr, address->size, tensor->data_c(), tensor->Size());
  if (res != EOK) {
    MS_LOG(EXCEPTION) << "memcpy failed. res: " << res << ", dest size: " << address->size
                      << ", src size: " << tensor->Size();
  }
}

bool PyExecuteCpuKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                   const std::vector<AddressPtr> &outputs) {
  MS_LOG(DEBUG) << "Launch PyExecute(), inputs.size: " << inputs.size() << ", outputs: " << outputs.size();
  if (Py_IsInitialized() == 0) {
    MS_LOG(ERROR) << "Py_IsInitialized failed.";
    return false;
  }
  if (outputs.size() != 1) {
    MS_LOG(EXCEPTION) << "The output num is 1, but got " << outputs.size();
  }
  py::gil_scoped_acquire gil_acquire;
  const auto &input0_info = inputs_info_[0];
  const auto &input0_abstract = input0_info.abstract;
  const auto &input0_abstract_scalar = dyn_cast<abstract::AbstractScalar>(input0_abstract);
  MS_EXCEPTION_IF_NULL(input0_abstract_scalar);
  if (!input0_abstract_scalar->BuildType()->isa<String>()) {
    MS_LOG(EXCEPTION) << "Should be a string, but got " << input0_abstract_scalar->ToString();
  }
  const auto &input0_value = input0_abstract_scalar->BuildValue();
  MS_EXCEPTION_IF_NULL(input0_value);
  const auto &input0_str = dyn_cast<StringImm>(input0_value);
  MS_LOG(DEBUG) << "Script: " << input0_str->ToString();
  // Check if output exists created by 'CppInferShapeAndType'.
  if (HasPyExecuteOutput(input0_str)) {
    const auto &output = PopPyExecuteOutput(input0_str);
    const auto &output_type = py::str(output.get_type());
    MS_LOG(DEBUG) << "Python *prebuilt* output type: " << output_type << ", output: " << output;
    if (py::isinstance<tensor::Tensor>(output)) {
      TensorToRawMemory(output.cast<tensor::TensorPtr>(), outputs[0]);
    } else if (IsStubTensor(output)) {
      TensorToRawMemory(ConvertStubTensor(output), outputs[0]);
    }
    AttachPyOutputData(output);
    return true;
  }
  const auto &script = input0_str->value();
  auto local_dict = BuildLocalParameters(inputs);
  auto global_dict = CallPythonGetGlobalParams();
  MS_LOG(INFO) << "Prebuilt output result not exists.";
  const auto &py_script = py::str(script);
  constexpr auto number_two = 2;
  auto params = py::tuple(number_two);
  params[0] = global_dict;
  params[1] = local_dict;
  MS_LOG(DEBUG) << "Python script: " << py_script << ", local_dict: " << local_dict;
  try {
    mindspore::ScopedFallbackRunning fallback_running;
    const auto &output = CallPythonScript(py_script, params);
    const auto &output_type = py::str(output.get_type());
    MS_LOG(DEBUG) << "Python output type: " << output_type << ", output: " << output;
    if (py::isinstance<tensor::Tensor>(output)) {
      TensorToRawMemory(output.cast<tensor::TensorPtr>(), outputs[0]);
    } else if (IsStubTensor(output)) {
      TensorToRawMemory(ConvertStubTensor(output), outputs[0]);
    }
    AttachPyOutputData(output);
  } catch (const py::error_already_set &e) {
    auto error_type_name = py::cast<std::string>(python_adapter::GetPyObjAttr(e.type(), "__name__"));
    auto error_iter = exception_types_map.find(error_type_name);
    if (error_iter != exception_types_map.end()) {
      auto &handler = LogWriter::GetExceptionHandler();
      if (handler != nullptr) {
        std::stringstream ss;
        ss << py::str(e.value()) << ".\n\n" << trace::GetDebugInfo(kernel_node_->debug_info());
        handler(error_iter->second, ss.str());
      }
    }
    throw std::runtime_error(py::str(e.value()));
  }
  return true;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, PyExecute, PyExecuteCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
