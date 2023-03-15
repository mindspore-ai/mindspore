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

#include "src/extendrt/delegate/tensorrt/tensorrt_graph_executor.h"
#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include <set>
#include <map>
#include <string>
#include <utility>
#include <algorithm>
#include <fstream>
#include "src/extendrt/delegate/delegate_utils.h"
#include "ccsrc/kernel/common_utils.h"
#include "ccsrc/include/backend/optimizer/helper.h"
#include "ccsrc/include/common/utils/convert_utils.h"
#include "src/extendrt/utils/kernel_graph_utils.h"
#include "common/config_infos.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "src/extendrt/utils/func_graph_utils.h"

namespace mindspore::lite {
namespace {
const char tensorrt_provider[] = "tensorrt";

struct NodeWithOutputIndex {
  NodeWithOutputIndex() = default;
  NodeWithOutputIndex(session::KernelWithIndex kernel_index, TensorInfo tensor_info)
      : kernel_index(kernel_index), tensor_info(tensor_info) {}

  session::KernelWithIndex kernel_index;
  TensorInfo tensor_info;
};

ValuePtr GetNodeValuePtr(AnfNodePtr input_node) {
  if (input_node == nullptr) {
    return nullptr;
  }
  if (IsPrimitiveCNode(input_node, prim::kPrimDepend)) {
    input_node = AnfUtils::VisitKernel(input_node, 0).first;
  }
  ValuePtr value = nullptr;
  if (input_node->isa<ValueNode>() && !HasAbstractMonad(input_node)) {
    auto value_node = input_node->cast<ValueNodePtr>();
    if (value_node) {
      value = value_node->value();
    }
  } else if (input_node->isa<Parameter>()) {
    auto parameter = input_node->cast<ParameterPtr>();
    if (parameter->has_default()) {
      value = parameter->default_param();
    }
  }
  return value;
}

tensor::TensorPtr GetConstNodeValue(AnfNodePtr input_node) {
  ValuePtr value = GetNodeValuePtr(input_node);
  if (value == nullptr) {
    return nullptr;
  }
  if (value->isa<tensor::Tensor>()) {
    auto tensor = value->cast<tensor::TensorPtr>();
    if (tensor == nullptr || tensor->data().const_data() == nullptr) {
      return nullptr;
    }
    return tensor;
  }
  if (value->isa<Scalar>()) {
    return ScalarToTensor(value->cast<ScalarPtr>());
  }
  if (value->isa<ValueTuple>()) {
    return opt::CreateTupleTensor(value->cast<ValueTuplePtr>());
  }
  if (value->isa<Type>()) {
    auto type_ptr = value->cast<TypePtr>();
    if (type_ptr == nullptr) {
      return nullptr;
    }
    return std::make_shared<tensor::Tensor>(static_cast<int64_t>(type_ptr->type_id()), type_ptr->type());
  }
  MS_LOG(WARNING) << "Unexpected value type " << value->type_name() << " for " << input_node->fullname_with_scope();
  return nullptr;
}

TensorInfo KernelTensorAsTensorInfo(const session::KernelWithIndex &tensor_id) {
  auto prev_node = tensor_id.first;
  auto tensor_val = GetConstNodeValue(prev_node);

  constexpr auto tensorrt_format = mindspore::Format::NCHW;
  auto name = tensor_id.first->UniqueName();
  if (tensor_id.second > 0) {
    name += ":" + std::to_string(tensor_id.second);
  }
  auto shape = common::AnfAlgo::GetOutputInferShape(tensor_id.first, tensor_id.second);
  auto type_id = common::AnfAlgo::GetOutputInferDataType(tensor_id.first, tensor_id.second);

  auto datatype = static_cast<enum DataType>(type_id);
  auto format = tensorrt_format;
  const void *data = nullptr;
  size_t data_len = 0;
  if (tensor_val) {
    data = tensor_val->data_c();
    data_len = tensor_val->Size();
    shape = tensor_val->shape_c();
  }
  TensorInfo tensor_info(name, datatype, shape, format, data, data_len, tensor_val);
  return tensor_info;
}

Status GetAbstractArgsFromCNode(const CNodePtr &cnode, std::vector<NodeWithOutputIndex> *tensor_info_list_ptr,
                                BaseOperatorPtr *base_operator, std::vector<TensorInfo> *input_tensors,
                                std::vector<TensorInfo> *output_tensors) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(tensor_info_list_ptr);
  auto &tensor_info_list = *tensor_info_list_ptr;
  auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
  MS_EXCEPTION_IF_NULL(prim);
  auto kernel_name = prim->name();
  ops::PrimitiveCPtr primc_ptr = nullptr;
  static auto primc_fns = ops::OpPrimCRegister::GetInstance().GetPrimCMap();
  if (primc_fns.find(kernel_name) != primc_fns.end()) {
    primc_ptr = primc_fns[kernel_name]();
    (void)primc_ptr->SetAttrs(prim->attrs());
  }
  if (primc_ptr == nullptr) {
    MS_LOG(ERROR) << "OpPrimCRegister can not find " << kernel_name;
    return mindspore::kLiteError;
  }

  *base_operator = nullptr;
  static auto operator_fns = ops::OperatorRegister::GetInstance().GetOperatorMap();
  if (operator_fns.find(kernel_name) != operator_fns.end()) {
    *base_operator = operator_fns[kernel_name](primc_ptr);
  }
  MS_EXCEPTION_IF_NULL(*base_operator);
  // Makeup input tensors.
  input_tensors->clear();
  auto input_nodes = mindspore::GetNodeInputs(cnode);
  for (auto &tensor_id : input_nodes) {
    auto it = std::find_if(tensor_info_list.begin(), tensor_info_list.end(),
                           [&tensor_id](const NodeWithOutputIndex &index) { return index.kernel_index == tensor_id; });
    if (it != tensor_info_list.end()) {
      input_tensors->push_back(it->tensor_info);
    } else {
      auto tensor_info = KernelTensorAsTensorInfo(tensor_id);
      input_tensors->push_back(tensor_info);
      tensor_info_list.push_back(NodeWithOutputIndex(tensor_id, tensor_info));
    }
  }
  // Makeup output tensors.
  output_tensors->clear();
  auto output_num = AnfUtils::GetOutputTensorNum(cnode);
  for (size_t output_idx = 0; output_idx < output_num; ++output_idx) {
    session::KernelWithIndex tensor_id = {cnode, output_idx};
    auto it = std::find_if(tensor_info_list.begin(), tensor_info_list.end(),
                           [&tensor_id](const NodeWithOutputIndex &index) { return index.kernel_index == tensor_id; });
    if (it != tensor_info_list.end()) {
      output_tensors->push_back(it->tensor_info);
    } else {
      auto tensor_info = KernelTensorAsTensorInfo(tensor_id);
      output_tensors->push_back(tensor_info);
      tensor_info_list.push_back(NodeWithOutputIndex(tensor_id, tensor_info));
    }
  }
  return kSuccess;
}

Status GetModelInputsInfo(KernelGraphPtr kernel_graph, std::vector<NodeWithOutputIndex> *tensor_info_list_ptr,
                          std::vector<TensorInfo> *inputs) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  MS_EXCEPTION_IF_NULL(tensor_info_list_ptr);
  auto &tensor_info_list = *tensor_info_list_ptr;
  auto kernel_graph_inputs = kernel_graph->inputs();
  // find parameters of graph inputs
  for (size_t i = 0; i < kernel_graph_inputs.size(); ++i) {
    auto input = kernel_graph_inputs[i];
    if (!input->isa<Parameter>()) {
      MS_LOG(ERROR) << "Kernel graph inputs have anfnode which is not Parameter.";
      return mindspore::kLiteError;
    }
    auto parameter = input->cast<ParameterPtr>();
    if (common::AnfAlgo::IsParameterWeight(parameter)) {
      continue;
    }
    constexpr auto tensorrt_format = mindspore::Format::NCHW;

    std::vector<int64_t> input_shape = AnfAlgo::GetOutputDeviceShape(parameter, 0);
    auto kernel_build_info = AnfAlgo::GetSelectKernelBuildInfo(parameter);
    auto data_type = kernel_build_info->GetOutputDeviceType(0);
    auto format = tensorrt_format;
    NodeWithOutputIndex node_index;
    node_index.kernel_index.first = input;
    node_index.kernel_index.second = 0;

    auto abstract = parameter->abstract();
    MS_EXCEPTION_IF_NULL(abstract);
    auto name = abstract->name();
    if (name.empty()) {
      name = parameter->name();
    }
    TensorInfo tensor_info =
      TensorInfo(name, static_cast<enum DataType>(data_type), input_shape, format, nullptr, 0, nullptr);
    inputs->push_back(tensor_info);

    node_index.tensor_info = tensor_info;
    tensor_info_list.push_back(node_index);
  }
  return kSuccess;
}

Status GetModelOutputsInfo(KernelGraphPtr kernel_graph, std::vector<NodeWithOutputIndex> *tensor_info_list_ptr,
                           std::vector<TensorInfo> *output_tensors) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  MS_EXCEPTION_IF_NULL(tensor_info_list_ptr);
  auto &tensor_info_list = *tensor_info_list_ptr;
  auto outputs = KernelGraphUtils::GetKernelGraphOutputs(kernel_graph);
  // find parameters of graph inputs
  for (size_t i = 0; i < outputs.size(); ++i) {
    auto output = outputs[i];
    auto cur_abstract = output->abstract();
    size_t output_num = 1;
    if (cur_abstract->isa<abstract::AbstractTuple>()) {
      auto abs_tuple = cur_abstract->Clone()->cast<abstract::AbstractTuplePtr>();
      MS_EXCEPTION_IF_NULL(abs_tuple);
      output_num = abs_tuple->elements().size();
    }
    for (size_t output_idx = 0; output_idx < output_num; ++output_idx) {
      auto tensor_id = common::AnfAlgo::VisitKernelWithReturnType(output, output_idx);
      auto it =
        std::find_if(tensor_info_list.begin(), tensor_info_list.end(),
                     [&tensor_id](const NodeWithOutputIndex &index) { return index.kernel_index == tensor_id; });
      if (it != tensor_info_list.end()) {
        output_tensors->push_back(it->tensor_info);
      } else {
        auto tensor_info = KernelTensorAsTensorInfo(tensor_id);
        output_tensors->push_back(tensor_info);
        tensor_info_list.push_back(NodeWithOutputIndex(tensor_id, tensor_info));
      }
    }
  }
  return kSuccess;
}
}  // namespace
TensorRTExecutor::TensorRTExecutor(const std::shared_ptr<mindspore::Context> &context, const ConfigInfos &config_infos)
    : context_(context), config_infos_(config_infos) {}

TensorRTExecutor::~TensorRTExecutor() {
  // delete tensorrt_graph_ before delete runtime
  tensorrt_graph_.reset();
  if (runtime_ != nullptr) {
    delete runtime_;
  }
  if (stream_ != nullptr) {
    cudaStreamDestroy(stream_);
  }
  if (cublas_handle_ != nullptr) {
    cublasDestroy(cublas_handle_);
    cublas_handle_ = nullptr;
  }
  if (cublaslt_handle_ != nullptr) {
    cublasLtDestroy(cublaslt_handle_);
    cublaslt_handle_ = nullptr;
  }
}
bool IsHardwareSupport() {
  int driver_version = 0;
  int ret = cudaDriverGetVersion(&driver_version);
  if (ret != cudaSuccess || driver_version == 0) {
    MS_LOG(WARNING) << "No nvidia GPU driver.";
    return false;
  }
  return true;
}

bool TensorRTExecutor::Init() {
  if (!IsHardwareSupport()) {
    return false;
  }
  if (context_ == nullptr) {
    MS_LOG_ERROR << "Context cannot be nullptr";
    return false;
  }

  std::vector<std::shared_ptr<DeviceInfoContext>> device_list = context_->MutableDeviceInfo();
  auto iter = std::find_if(device_list.begin(), device_list.end(), [](std::shared_ptr<DeviceInfoContext> device) {
    return device->GetDeviceType() == DeviceType::kGPU;
  });
  if (iter == device_list.end()) {
    MS_LOG(ERROR) << "no gpu device info found for TensorRT.";
    return false;
  }
  auto gpu_info = (*iter)->Cast<GPUDeviceInfo>();
  if (gpu_info == nullptr) {
    MS_LOG(ERROR) << "no gpu device info found for TensorRT.";
    return false;
  }
  device_info_ = gpu_info;
  int ret = lite::SetCudaDevice(device_info_);
  if (ret != RET_OK) {
    return false;
  }
  if (runtime_ == nullptr) {
    runtime_ = new (std::nothrow) TensorRTRuntime();
    if (runtime_ == nullptr) {
      MS_LOG(ERROR) << "create TensorRTRuntime failed.";
      return false;
    }
  }
  if (runtime_->Init() != RET_OK) {
    MS_LOG(ERROR) << "TensorRTRuntime init failed.";
    return false;
  }
  runtime_->SetDeviceID(device_info_->GetDeviceID());

  auto cuda_ret = cudaStreamCreate(&stream_);
  if (cuda_ret != cudaSuccess) {
    MS_LOG(ERROR) << "Cuda create stream failed";
    return false;
  }

  auto cublas_ret = cublasCreate(&cublas_handle_);
  if (cublas_ret != CUBLAS_STATUS_SUCCESS) {
    MS_LOG(ERROR) << "Cuda create cublas handle failed";
    return false;
  }

  auto cublaslt_ret = cublasLtCreate(&cublaslt_handle_);
  if (cublaslt_ret != CUBLAS_STATUS_SUCCESS) {
    MS_LOG(ERROR) << "Cuda create cublaslt handle failed";
    return false;
  }

  ret = ParseOptimizationProfile();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Parse input ranges failed.";
    return false;
  }
  return true;
}

int TensorRTExecutor::ParseOptimizationProfile() {
  auto gpu_context_it = config_infos_.find(kGPUContextSection);
  if (gpu_context_it == config_infos_.end()) {
    MS_LOG(INFO) << "do not have input ranges config.";
    return RET_OK;
  }
  auto &gpu_context = gpu_context_it->second;
  ProfileConfigs profile_configs;
  if (!ProfileParser::Parse(gpu_context, true, &profile_configs)) {
    MS_LOG_WARNING << "Failed to parse profile info from '" << kGPUContextSection << "'";
    return RET_FAILED;
  }
  trt_profile_configs_ = profile_configs;
  auto precision_mode = ProfileParser::GetOption(gpu_context, lite::kPrecisionModeKey, "");
  if (precision_mode.empty()) {
    device_info_->SetPrecisionMode("enforce_fp32");
  } else {
    device_info_->SetPrecisionMode(precision_mode);
  }
  serialize_path_ = ProfileParser::GetOption(gpu_context, lite::kMSCacheSerializePathKey);
  return RET_OK;
}

int TensorRTExecutor::ParseDumpOptions(const std::map<std::string, std::string> &gpu_context) {
  auto dump_ops_str = ProfileParser::GetOption(gpu_context, lite::kDumpOpsKey, "");
  if (!dump_ops_str.empty()) {
    dump_ops_ = lite::StrSplit(dump_ops_str, ";");
    dump_dir_ = ProfileParser::GetOption(gpu_context, lite::kDumpDirKey, "");
    if (dump_dir_.empty()) {
      dump_dir_ = ".";
    }
  }
  return RET_OK;
}

Status TensorRTExecutor::BuildSubGraph(const KernelGraphPtr &kernel_graph) {
  KernelIter from, end;
  std::vector<TensorRTOp *> tensorrt_ops;
  int tensorrt_subgraph_index = 0;

  auto &nodes = kernel_graph->nodes();
  for (const auto &node : nodes) {
    std::string node_name = common::AnfAlgo::GetCNodeName(node);
    MS_LOG(INFO) << "TensorRTExecutor::Nodes " << node_name;
  }
  auto &kernel_nodes = kernel_graph->execution_order();
  if (kernel_nodes.empty()) {
    MS_LOG(ERROR) << "There are no nodes in the graph";
    return mindspore::kLiteNullptr;
  }
  std::vector<NodeWithOutputIndex> tensor_info_list;
  auto status = GetModelInputsInfo(kernel_graph, &tensor_info_list, &inputs_);
  if (status != kSuccess) {
    return status;
  }
  for (const auto &kernel_node : kernel_nodes) {
    auto node_name = kernel_node->fullname_with_scope();
    std::string kernel_name = common::AnfAlgo::GetCNodeName(kernel_node);
    BaseOperatorPtr base_operator = nullptr;
    std::vector<TensorInfo> input_tensors;
    std::vector<TensorInfo> output_tensors;
    status = GetAbstractArgsFromCNode(kernel_node, &tensor_info_list, &base_operator, &input_tensors, &output_tensors);
    if (status != kSuccess || base_operator == nullptr) {
      MS_LOG(ERROR) << "Failed to get operator of node " << kernel_name;
      return mindspore::kLiteError;
    }
    auto tensorrt_op = FindTensorRTOp(kernel_node, base_operator, input_tensors, output_tensors);
    if (tensorrt_op == nullptr) {
      MS_LOG(ERROR) << "FindTensorRTOp failed " << kernel_name;
      return mindspore::kLiteError;
    }
    tensorrt_op->SetRuntime(this->runtime_);
    tensorrt_ops.push_back(tensorrt_op);
    if (!dump_ops_.empty() && std::find(dump_ops_.begin(), dump_ops_.end(), node_name) != dump_ops_.end()) {
      std::copy(output_tensors.begin(), output_tensors.end(), std::back_inserter(dump_outputs_));
    }
  }
  status = GetModelOutputsInfo(kernel_graph, &tensor_info_list, &outputs_);
  if (status != kSuccess) {
    return status;
  }
  for (auto &out_tensor_info : outputs_) {
    if (out_tensor_info.DataType() == DataType::kNumberTypeFloat16) {
      MS_LOG(INFO) << "output " << out_tensor_info.Name() << " is Float16, set to Float32";
      out_tensor_info.SetDataType(DataType::kNumberTypeFloat32);
    }
  }
  std::vector<TensorInfo> trt_outputs = outputs_;
  std::copy(dump_outputs_.begin(), dump_outputs_.end(), std::back_inserter(trt_outputs));
  tensorrt_graph_ = CreateTensorRTGraph(tensorrt_ops, kernel_graph, tensorrt_subgraph_index, inputs_, trt_outputs);
  if (!tensorrt_graph_) {
    MS_LOG(ERROR) << "Create tensorrt graph failed";
    return mindspore::kLiteError;
  }
  return mindspore::kSuccess;
}

TensorRTOp *TensorRTExecutor::FindTensorRTOp(const CNodePtr &cnode, const BaseOperatorPtr &base_operator,
                                             const std::vector<TensorInfo> &input_tensors,
                                             const std::vector<TensorInfo> &output_tensors) {
  auto name = cnode->fullname_with_scope();
  auto node_type = base_operator->name();
  auto &plugin_factory = TensorRTRegistrationFactory::Get();
  if (plugin_factory.HasKey(node_type)) {
    TensorRTOp *tensorrt_op = plugin_factory.GetCreator(node_type)(base_operator, input_tensors, output_tensors, name);
    if (tensorrt_op == nullptr) {
      return nullptr;
    }
    if (!support_resize_) {
      return tensorrt_op;
    }
    support_resize_ = tensorrt_op->GetDynamicShapeParams().support_dynamic_ ? support_resize_ : false;
    if (!tensorrt_op->GetDynamicShapeParams().support_dynamic_) {
      MS_LOG(WARNING) << "TensorRT subgraph don't support dynamic shape resize, because of op " << name;
      support_hw_resize_ = false;
      return tensorrt_op;
    }
    if (!support_hw_resize_) {
      return tensorrt_op;
    }
    support_hw_resize_ = tensorrt_op->GetDynamicShapeParams().support_hw_dynamic_ ? support_hw_resize_ : false;
    if (!tensorrt_op->GetDynamicShapeParams().support_hw_dynamic_) {
      MS_LOG(WARNING) << "TensorRT subgraph don't support dynamic hw dims resize, because of op " << name;
    }
    return tensorrt_op;
  } else {
    MS_LOG(WARNING) << "Unsupported op type for TensorRT. kernel name:" << name << " type:" << node_type;
    return nullptr;
  }
}

std::shared_ptr<TensorRTSubGraph> TensorRTExecutor::CreateTensorRTGraph(const std::vector<TensorRTOp *> &ops,
                                                                        const KernelGraphPtr &graph, int index,
                                                                        const std::vector<TensorInfo> &inputs,
                                                                        const std::vector<TensorInfo> &outputs) {
  if (!trt_profile_configs_.input_infos.empty()) {
    std::vector<std::string> input_names;
    std::transform(inputs.begin(), inputs.end(), std::back_inserter(input_names),
                   [](auto &item) { return item.Name(); });
    if (!ProfileParser::ReorderByInputNames(input_names, &trt_profile_configs_)) {
      MS_LOG_ERROR << "Reorder profile by input names failed, input names: " << input_names;
      return nullptr;
    }
  }

  auto tensorrt_graph = std::make_shared<TensorRTSubGraph>(ops, inputs, outputs, context_.get(), device_info_, runtime_,
                                                           support_resize_, support_hw_resize_, trt_profile_configs_);
  if (tensorrt_graph == nullptr) {
    MS_LOG(ERROR) << "new tensorrt_graph failed.";
    return nullptr;
  }
  if (serialize_path_.size() > 0) {
    tensorrt_graph->SetSerializePath(serialize_path_ + "_trt" + std::to_string(GetRankID()) + ".bin_" +
                                     std::to_string(index));
  }
  // 1. For every op, find pre and next ops
  FindPreNextOps<TensorRTOp>(ops);

  // 2. Init TensorRT SubGraph.
  auto ret = tensorrt_graph->Init(stream_, cublas_handle_, cublaslt_handle_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "TensorRTGraph init failed.";
    return nullptr;
  }

  // 3. Build TensorRT Model.
  ret = tensorrt_graph->BuildTensorRTGraph();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "TensorRTGraph build failed.";
    return nullptr;
  }
  ret = tensorrt_graph->Prepare();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "TensorRTGraph prepare failed.";
    return nullptr;
  }
  return tensorrt_graph;
}

bool TensorRTExecutor::CompileGraph(const FuncGraphPtr &graph, const std::map<string, string> &compile_options) {
  int ret = lite::SetCudaDevice(device_info_);
  if (ret != RET_OK) {
    return false;
  }
  auto kernel_graph = dyn_cast<session::KernelGraph>(graph);
  MS_EXCEPTION_IF_NULL(kernel_graph);
  Status build_ret = BuildSubGraph(kernel_graph);
  if (build_ret != kSuccess) {
    MS_LOG(INFO) << "BuildSubGraph failed";
    return false;
  }
  return true;
}

bool TensorRTExecutor::RunGraph(const FuncGraphPtr &graph, const std::vector<tensor::Tensor> &inputs,
                                std::vector<tensor::Tensor> *outputs, const std::map<string, string> &compile_options) {
  if (inputs.size() != inputs_.size()) {
    MS_LOG(ERROR) << "Graph inputs size " << inputs_.size() << " != execute outputs size " << inputs.size();
    return false;
  }
  if (tensorrt_graph_ == nullptr) {
    MS_LOG(ERROR) << "TensorRT subgraph is nullptr.";
    return false;
  }
  if (dump_outputs_.empty()) {
    if (!outputs->empty() && outputs_.size() != outputs->size()) {
      MS_LOG(ERROR) << "Graph outputs size " << inputs_.size() << " != expected outputs size " << outputs->size();
      return false;
    }
    return tensorrt_graph_->Execute(inputs, outputs) == RET_OK;
  }

  if (!outputs->empty()) {
    MS_LOG(ERROR) << "Cannot has graph outputs when dump op";
    return false;
  }
  std::vector<tensor::Tensor> trt_outputs;
  if (tensorrt_graph_->Execute(inputs, &trt_outputs) != RET_OK) {
    return false;
  }
  if (trt_outputs.size() != outputs_.size() + dump_outputs_.size()) {
    MS_LOG(ERROR) << "TensorRT Graph outputs size " << trt_outputs.size() << " != graph outputs size "
                  << outputs_.size() << " + dump output size " << dump_outputs_.size();
    return false;
  }
  for (size_t i = 0; i < outputs_.size(); i++) {
    outputs->push_back(trt_outputs[i]);
  }
  if (!has_dumped_) {
    has_dumped_ = true;
    auto dump_tensor = [this](const std::string &file_name, const tensor::Tensor &tensor) {
      std::string new_file = file_name;
      for (size_t i = 0; i < new_file.size(); i++) {
        if (new_file[i] == '/' || new_file[i] == '\\') {
          new_file[i] = '_';
        }
      }
      std::ofstream fp(dump_dir_ + "/" + new_file, std::ofstream::binary);
      if (!fp.is_open()) {
        MS_LOG(WARNING) << "Failed to open file " << dump_dir_ + "/" + file_name;
        return;
      }
      fp.write(reinterpret_cast<const char *>(tensor.data_c()), tensor.Size());
    };
    for (size_t i = 0; i < inputs.size(); i++) {
      dump_tensor("input_" + std::to_string(i) + ".bin", inputs[i]);
    }
    for (size_t i = 0; i < outputs->size(); i++) {
      dump_tensor("output_" + std::to_string(i) + ".bin", (*outputs)[i]);
    }
    for (size_t i = outputs_.size(); i < trt_outputs.size(); i++) {
      auto tensor_info = dump_outputs_[i - outputs_.size()];
      dump_tensor(tensor_info.Name() + ".bin", trt_outputs[i]);
    }
  }
  return true;
}

bool TensorRTExecutor::Resize(const FuncGraphPtr &, const std::vector<tensor::Tensor> &inputs,
                              const std::vector<std::vector<int64_t>> &new_shapes) {
  if (tensorrt_graph_ == nullptr) {
    MS_LOG(ERROR) << "TensorRT subgraph is nullptr.";
    return false;
  }
  return tensorrt_graph_->Resize(inputs, new_shapes) == RET_OK;
}

std::vector<tensor::Tensor> TensorRTExecutor::GetInputInfos(const FuncGraphPtr &) {
  std::vector<tensor::Tensor> tensors;
  for (auto &tensor_info : inputs_) {
    auto type_id = static_cast<enum TypeId>(tensor_info.DataType());
    auto shape = tensor_info.Shape();
    tensors.push_back(tensor::Tensor(type_id, shape));
  }
  return tensors;
}

std::vector<tensor::Tensor> TensorRTExecutor::GetOutputInfos(const FuncGraphPtr &) {
  std::vector<tensor::Tensor> tensors;
  for (auto &tensor_info : outputs_) {
    auto type_id = static_cast<enum TypeId>(tensor_info.DataType());
    auto shape = tensor_info.Shape();
    tensors.push_back(tensor::Tensor(type_id, shape));
  }
  return tensors;
}

static std::shared_ptr<device::GraphExecutor> TensorRTGraphExecutorCreator(const std::shared_ptr<Context> &ctx,
                                                                           const ConfigInfos &config_infos) {
  auto executor = std::make_shared<TensorRTExecutor>(ctx, config_infos);
  if (!executor->Init()) {
    return nullptr;
  }
  return executor;
}

REG_DELEGATE(kGPU, tensorrt_provider, TensorRTGraphExecutorCreator);
}  // namespace mindspore::lite
