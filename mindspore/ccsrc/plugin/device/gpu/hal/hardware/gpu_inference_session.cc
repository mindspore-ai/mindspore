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
#include "plugin/device/gpu/hal/hardware/gpu_inference_session.h"
#include <algorithm>
#include "ir/tensor.h"
#include "ir/anf.h"
#include "ir/param_info.h"
#include "runtime/device/kernel_runtime.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "utils/ms_utils.h"
#include "runtime/device/ms_device_shape_transfer.h"
#include "include/common/utils/config_manager.h"

namespace mindspore {
namespace session {
void GpuInferenceSession::LoadInputData(const std::shared_ptr<KernelGraph> &kernel_graph,
                                        const std::vector<tensor::TensorPtr> &inputs_const) const {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  std::vector<tensor::TensorPtr> inputs(inputs_const);
  auto input_nodes = kernel_graph->inputs();

  size_t no_weight_input = 0;
  for (size_t i = 0; i < input_nodes.size(); ++i) {
    tensor::TensorPtr tensor = nullptr;
    if (!input_nodes[i]->isa<Parameter>() || !AnfAlgo::OutputAddrExist(input_nodes[i], 0)) {
      MS_LOG(INFO) << "Kernel graph inputs is not Parameter or without user.";
      continue;
    }
    auto pk_node = input_nodes[i]->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(pk_node);
    if (!pk_node->IsUsedByRealKernelInGraph(kernel_graph->graph_id())) {
      MS_LOG(INFO) << "Kernel graph inputs have anfnode which has no user.";
      continue;
    }
    auto device_address = AnfAlgo::GetMutableOutputAddr(pk_node, 0);
    MS_EXCEPTION_IF_NULL(device_address);
    if (!common::AnfAlgo::IsParameterWeight(pk_node)) {
      tensor = inputs[no_weight_input++];
      MS_EXCEPTION_IF_NULL(tensor);
      if (!device_address->SyncHostToDevice(trans::GetRuntimePaddingShape(pk_node, 0),
                                            LongToSize(tensor->data().nbytes()), tensor->data_type(),
                                            tensor->data_c())) {
        MS_LOG(EXCEPTION) << "SyncHostToDevice failed.";
      }
    }
  }
}

GraphId GpuInferenceSession::CompileGraphImpl(NotNull<FuncGraphPtr> func_graph) {
  auto graph_id = GPUSession::CompileGraphImpl(func_graph);
  auto kernel_graph = GetGraph(graph_id);
  MS_EXCEPTION_IF_NULL(kernel_graph);
  // load weight data to device
  auto input_nodes = kernel_graph->inputs();
  for (size_t i = 0; i < input_nodes.size(); ++i) {
    if (!input_nodes[i]->isa<Parameter>() || !AnfAlgo::OutputAddrExist(input_nodes[i], 0)) {
      MS_LOG(INFO) << "Kernel graph inputs is not Parameter or without user.";
      continue;
    }
    auto pk_node = input_nodes[i]->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(pk_node);
    if (!pk_node->IsUsedByRealKernelInGraph(kernel_graph->graph_id())) {
      MS_LOG(INFO) << "Kernel graph inputs have anfnode which has no user.";
      continue;
    }
    auto device_address = AnfAlgo::GetMutableOutputAddr(pk_node, 0);
    MS_EXCEPTION_IF_NULL(device_address);
    if (common::AnfAlgo::IsParameterWeight(pk_node)) {
      const auto &param_value = pk_node->default_param();
      MS_EXCEPTION_IF_NULL(param_value);
      auto tensor = std::dynamic_pointer_cast<tensor::Tensor>(param_value);
      MS_EXCEPTION_IF_NULL(tensor);
      if (!device_address->SyncHostToDevice(trans::GetRuntimePaddingShape(pk_node, 0),
                                            LongToSize(tensor->data().nbytes()), tensor->data_type(),
                                            tensor->data_c())) {
        MS_LOG(EXCEPTION) << "SyncHostToDevice failed.";
      }
    }
  }
  return graph_id;
}

bool GpuInferenceSession::CheckModelInputs(uint32_t graph_id, const std::vector<tensor::TensorPtr> &inputs,
                                           std::string *error_msg) const {
  MS_LOG(INFO) << "Start check client inputs, graph id : " << graph_id;
  auto kernel_graph = GetGraph(graph_id);
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto kernel_graph_inputs = kernel_graph->inputs();
  size_t no_weight_input = 0;
  vector<ParameterPtr> paras;
  // find parameters of graph inputs
  for (size_t i = 0; i < kernel_graph_inputs.size(); ++i) {
    if (!kernel_graph_inputs[i]->isa<Parameter>()) {
      MS_LOG(ERROR) << "Kernel graph inputs have anfnode which is not Parameter.";
      continue;
    }
    auto parameter = kernel_graph_inputs[i]->cast<ParameterPtr>();
    if (!common::AnfAlgo::IsParameterWeight(parameter)) {
      paras.push_back(parameter);
    }
  }

  // check inputs
  for (size_t i = 0; i < paras.size(); ++i) {
    // compare input number
    if (paras.size() != inputs.size()) {
      MS_LOG(ERROR) << "Input number is inconsistent. The actual input number [" << inputs.size()
                    << "] but the graph input number is [" << paras.size() << "]";
      MS_LOG(ERROR) << "InputsInfo --" << InputsInfo(paras, inputs);
      if (error_msg != nullptr) {
        std::stringstream str_stream;
        str_stream << "Input number is inconsistent. The given input number [" << inputs.size()
                   << "] but the graph input number is [" << paras.size() << "]\n";
        str_stream << "InputsInfo --" << InputsInfo(paras, inputs);
        *error_msg = str_stream.str();
      }
      return false;
    }
    auto input = inputs[no_weight_input++];
    if (!CompareInput(input, paras[i])) {
      MS_LOG(ERROR) << "Please check the input information.";
      MS_LOG(ERROR) << "InputsInfo --" << InputsInfo(paras, inputs);
      if (error_msg != nullptr) {
        std::stringstream str_stream;
        str_stream << "Please check the input information.\n";
        str_stream << "InputsInfo --" << InputsInfo(paras, inputs);
        *error_msg = str_stream.str();
      }
      return false;
    }
  }
  return true;
}

bool GpuInferenceSession::CompareInput(const tensor::TensorPtr &input, const ParameterPtr &parameter) const {
  MS_EXCEPTION_IF_NULL(input);
  MS_EXCEPTION_IF_NULL(parameter);
  // compare dims
  auto parameter_shape = AnfAlgo::GetOutputDeviceShape(parameter, 0);

  // compare shape
  auto input_shape = input->shape();
  vector<int64_t> trans_input;
  (void)std::transform(input_shape.begin(), input_shape.end(), std::back_inserter(trans_input),
                       [](const int64_t dim) { return static_cast<size_t>(dim); });
  auto is_scalar_shape = [](const vector<int64_t> &shape) {
    return shape.empty() || (shape.size() == 1 && shape[0] == 1);
  };
  if ((!is_scalar_shape(trans_input) || !is_scalar_shape(parameter_shape)) && (trans_input != parameter_shape)) {
    MS_LOG(ERROR) << "Input shape is inconsistent. The actual shape is " << PrintInputShape(trans_input)
                  << ", but the parameter shape is " << PrintInputShape(parameter_shape)
                  << ". parameter : " << parameter->DebugString();
    return false;
  }

  // compare data type
  auto kernel_build_info = AnfAlgo::GetSelectKernelBuildInfo(parameter);
  if (input->data_type() != kernel_build_info->GetOutputDeviceType(0)) {
    MS_LOG(ERROR) << "Input data type is inconsistent. The actual data type is " << input->data_type()
                  << ", but the parameter data type is " << kernel_build_info->GetOutputDeviceType(0)
                  << ". parameter : " << parameter->DebugString();
    return false;
  }
  return true;
}

template <typename T>
std::string GpuInferenceSession::PrintInputShape(std::vector<T> shape) const {
  string res = "[";
  for (auto dim : shape) {
    res += " " + std::to_string(dim);
  }
  return res + " ]";
}

std::string GpuInferenceSession::InputsInfo(const std::vector<ParameterPtr> &paras,
                                            const std::vector<tensor::TensorPtr> &inputs) const {
  const std::map<TypeId, std::string> dtype_name_map{
    {TypeId::kNumberTypeBegin, "Unknown"},   {TypeId::kNumberTypeBool, "Bool"},
    {TypeId::kNumberTypeFloat64, "Float64"}, {TypeId::kNumberTypeInt8, "Int8"},
    {TypeId::kNumberTypeUInt8, "Uint8"},     {TypeId::kNumberTypeInt16, "Int16"},
    {TypeId::kNumberTypeUInt16, "Uint16"},   {TypeId::kNumberTypeInt32, "Int32"},
    {TypeId::kNumberTypeUInt32, "Uint32"},   {TypeId::kNumberTypeInt64, "Int64"},
    {TypeId::kNumberTypeUInt64, "Uint64"},   {TypeId::kNumberTypeFloat16, "Float16"},
    {TypeId::kNumberTypeFloat32, "Float32"},
  };
  auto data_type_to_string = [&dtype_name_map](TypeId type_id) {
    auto it = dtype_name_map.find(type_id);
    if (it == dtype_name_map.end()) {
      return std::string("Unknown");
    }
    return it->second;
  };

  std::string graph = "graph inputs:{ ";
  for (size_t i = 0; i < paras.size(); ++i) {
    auto &para = paras[i];
    graph += std::to_string(i) + ": dims " + std::to_string(AnfAlgo::GetOutputDeviceShape(para, 0).size()) +
             ", shape " + PrintInputShape(AnfAlgo::GetOutputDeviceShape(para, 0)) + ", data type " +
             data_type_to_string(AnfAlgo::GetSelectKernelBuildInfo(para)->GetOutputDeviceType(0)) + " }";
  }

  std::string actual = "given inputs:{ ";
  for (size_t i = 0; i < inputs.size(); ++i) {
    actual += std::to_string(i) + ": dims " + std::to_string(inputs[i]->shape().size()) + ", shape " +
              PrintInputShape(inputs[i]->shape()) + ", data type " + data_type_to_string(inputs[i]->data_type()) + " }";
  }
  return graph + "   " + actual;
}
}  // namespace session
}  // namespace mindspore
