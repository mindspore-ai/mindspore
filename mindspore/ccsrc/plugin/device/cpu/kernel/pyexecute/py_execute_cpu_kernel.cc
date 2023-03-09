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
#include "include/common/utils/python_fallback_running.h"
#include "plugin/factory/ms_factory.h"
#include "mindspore/ccsrc/pipeline/jit/parse/resolve.h"
#include "utils/trace_base.h"

namespace mindspore {
namespace kernel {
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
  if (HasPyExecuteOutput()) {
    const auto &output = PopPyExecuteOutput();
    const auto &output_type = py::str(output.get_type());
    MS_LOG(DEBUG) << "Python *prebuilt* output type: " << output_type << ", output: " << output;
    if (py::isinstance<tensor::Tensor>(output)) {
      TensorToRawMemory(output.cast<tensor::TensorPtr>(), outputs[0]);
    }
    AttachPyOutputData(output);
    return true;
  }
  MS_LOG(EXCEPTION) << "Prebuilt output result not exists for " << input0_str->ToString();
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, PyExecute, PyExecuteCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
