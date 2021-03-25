/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "src/scheduler.h"
#include <map>
#include <queue>
#include <string>
#include <vector>
#include <algorithm>
#include "src/tensorlist.h"
#include "include/errorcode.h"
#include "src/common/graph_util.h"
#include "src/common/utils.h"
#include "src/kernel_registry.h"
#include "src/sub_graph_kernel.h"
#include "src/ops/populate/populate_register.h"
#include "src/common/version_manager.h"
#include "src/common/prim_util.h"
#include "src/runtime/infer_manager.h"
#include "src/dequant.h"
#if GPU_OPENCL
#include "src/runtime/kernel/opencl/opencl_subgraph.h"
#include "src/runtime/gpu/opencl/opencl_runtime.h"
#endif
#if SUPPORT_NPU
#include "src/runtime/agent/npu/subgraph_npu_kernel.h"
#include "src/runtime/agent/npu/npu_manager.h"
#include "src/runtime/agent/npu/optimizer/npu_pass_manager.h"
#include "src/runtime/agent/npu/optimizer/npu_transform_pass.h"
#include "src/runtime/agent/npu/optimizer/npu_fusion_pass.h"
#include "src/runtime/agent/npu/optimizer/npu_insert_transform_pass.h"
#endif
namespace mindspore::lite {
using kernel::KERNEL_ARCH::kCPU;
using kernel::KERNEL_ARCH::kGPU;
using kernel::KERNEL_ARCH::kNPU;
namespace {
constexpr int kMainSubGraphIndex = 0;
}  // namespace

int Scheduler::Schedule(std::vector<kernel::LiteKernel *> *dst_kernels) {
  if (src_model_ == nullptr) {
    MS_LOG(ERROR) << "Input model is nullptr";
    return RET_PARAM_INVALID;
  }
  if (src_model_->sub_graphs_.empty()) {
    MS_LOG(ERROR) << "Model should have a subgraph at least";
    return RET_PARAM_INVALID;
  }

  this->graph_output_node_indexes_ = GetGraphOutputNodes(src_model_);
  bool infer_shape_interrupt = false;
  auto ret = InferSubGraphShape(kMainSubGraphIndex, &infer_shape_interrupt);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "op infer shape failed.";
    return ret;
  }
  ret = ScheduleSubGraphToKernels(kMainSubGraphIndex, dst_kernels, nullptr, nullptr);
  op_parameters_.clear();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Schedule main subgraph to kernels failed.";
    return ret;
  }
  FindAllInoutKernels(*dst_kernels);
  ret = RunPass(dst_kernels);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Schedule run pass failed.";
    return ret;
  }
  ret = ConstructSubGraphs(dst_kernels);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ConstructSubGraphs failed.";
    return ret;
  }
  MS_LOG(DEBUG) << "schedule kernels success.";
  return RET_OK;
}

void Scheduler::FindNodeInoutTensors(const lite::Model::Node &node, std::vector<Tensor *> *inputs,
                                     std::vector<Tensor *> *outputs) {
  MS_ASSERT(inputs != nullptr);
  MS_ASSERT(outputs != nullptr);
  auto in_size = node.input_indices_.size();
  inputs->reserve(in_size);
  for (size_t j = 0; j < in_size; ++j) {
    inputs->emplace_back(src_tensors_->at(node.input_indices_[j]));
  }
  auto out_size = node.output_indices_.size();
  outputs->reserve(out_size);
  for (size_t j = 0; j < out_size; ++j) {
    outputs->emplace_back(src_tensors_->at(node.output_indices_[j]));
  }
}

int Scheduler::InferNodeShape(const lite::Model::Node *node, bool *infer_shape_interrupt) {
  MS_ASSERT(node != nullptr);
  MS_ASSERT(infer_shape_interrupt != nullptr);
  auto primitive = node->primitive_;
  MS_ASSERT(primitive != nullptr);
  if (IsPartialNode(primitive)) {
    return InferPartialShape(node, infer_shape_interrupt);
  }
  std::vector<Tensor *> inputs;
  std::vector<Tensor *> outputs;
  FindNodeInoutTensors(*node, &inputs, &outputs);
  bool infer_valid = std::all_of(inputs.begin(), inputs.end(), [](const Tensor *tensor) {
    auto shape = tensor->shape();
    return std::all_of(shape.begin(), shape.end(), [](const int dim) { return dim != -1; });
  });
  if (!infer_valid) {
    *infer_shape_interrupt = true;
  }
  int schema_version = VersionManager::GetInstance()->GetSchemaVersion();
  auto parame_gen =
    PopulateRegistry::GetInstance()->GetParameterCreator(GetPrimitiveType(node->primitive_), schema_version);
  if (parame_gen == nullptr) {
    MS_LOG(ERROR) << "parameter generator is nullptr.";
    return RET_NULL_PTR;
  }
  auto parameter = parame_gen(primitive);
  if (parameter == nullptr) {
    MS_LOG(ERROR) << "PopulateParameter return nullptr, type: " << PrimitiveTypeName(GetPrimitiveType(primitive));
    return RET_ERROR;
  }
  parameter->quant_type_ = node->quant_type_;

  op_parameters_[node->output_indices_.at(0)] = parameter;
  parameter->infer_flag_ = !(*infer_shape_interrupt);
  auto ret = KernelInferShape(inputs, &outputs, parameter);
  if (ret == RET_INFER_INVALID) {
    parameter->infer_flag_ = false;
    *infer_shape_interrupt = true;
  }
  if (ret == RET_OK) {
    for (auto &output : outputs) {
      if (output->ElementsNum() >= MAX_MALLOC_SIZE / static_cast<int>(sizeof(int64_t))) {
        MS_LOG(ERROR) << "The size of output tensor is too big";
        return RET_ERROR;
      }
    }
  }
  return ret;
}

int Scheduler::InferPartialShape(const lite::Model::Node *node, bool *infer_shape_interrupt) {
  MS_ASSERT(src_model_ != nullptr);
  MS_ASSERT(node != nullptr);
  MS_ASSERT(infer_shape_interrupt != nullptr);
  if (!IsPartialNode(node->primitive_)) {
    MS_LOG(ERROR) << "Node is not a partial";
    return RET_PARAM_INVALID;
  }
  return InferSubGraphShape(GetPartialGraphIndex(node->primitive_), infer_shape_interrupt);
}

int Scheduler::InferSubGraphShape(size_t subgraph_index, bool *infer_shape_interrupt) {
  MS_ASSERT(infer_shape_interrupt != nullptr);
  MS_ASSERT(src_model_ != nullptr);
  MS_ASSERT(!src_model_->sub_graphs_.empty());
  MS_ASSERT(src_model_->sub_graphs_.size() > subgraph_index);
  auto subgraph = src_model_->sub_graphs_.at(subgraph_index);
  for (auto node_index : subgraph->node_indices_) {
    auto node = src_model_->all_nodes_[node_index];
    MS_ASSERT(node != nullptr);
    auto *primitive = node->primitive_;
    if (primitive == nullptr) {
      MS_LOG(ERROR) << "Op " << node->name_ << " should exist in model!";
      return RET_ERROR;
    }
    auto type = GetPrimitiveType(primitive);
    auto ret = InferNodeShape(node, infer_shape_interrupt);
    if (ret == RET_INFER_INVALID) {
      MS_LOG(INFO) << "InferShape interrupted, name: " << node->name_ << ", type: " << PrimitiveTypeName(type)
                   << ", set infer flag to false.";
      *infer_shape_interrupt = true;
    } else if (ret != RET_OK) {
      MS_LOG(ERROR) << "InferShape failed, name: " << node->name_ << ", type: " << PrimitiveTypeName(type);
      return RET_INFER_ERR;
    }
  }
  return RET_OK;
}

kernel::LiteKernel *Scheduler::FindBackendKernel(const std::vector<Tensor *> &in_tensors,
                                                 const std::vector<Tensor *> &out_tensors, const Model::Node *node,
                                                 TypeId prefer_data_type) {
  kernel::LiteKernel *kernel = nullptr;
  TypeId data_type = GetFirstFp32Fp16OrInt8Type(in_tensors);
  OpParameter *op_parameter = op_parameters_[node->output_indices_.at(0)];
  if (op_parameter == nullptr) {
    MS_LOG(ERROR) << "Can not find OpParameter!type: " << PrimitiveTypeName(GetPrimitiveType(node->primitive_));
    return nullptr;
  }
  bool infer_shape_interrupt = !op_parameter->infer_flag_;
  bool need_restore = true;
  if (node->quant_type_ == schema::QuantType_WeightQuant) {
    data_type = kNumberTypeFloat32;
  }
  if (!IsPackedOp(op_parameter->type_)) {
    need_restore = false;
  }
  kernel::KernelKey desc{kCPU, data_type, static_cast<schema::PrimitiveType>(op_parameter->type_)};
#if SUPPORT_GPU
  if (context_->IsGpuEnabled()) {
    // support more data type like int32
    kernel::KernelKey gpu_desc{kGPU, kNumberTypeFloat32, desc.type};
    if (context_->IsGpuFloat16Enabled()) gpu_desc.data_type = kNumberTypeFloat16;
    auto ret =
      KernelRegistry::GetInstance()->GetKernel(in_tensors, out_tensors, context_, gpu_desc, op_parameter, &kernel);
    if (ret == RET_OK) {
      MS_LOG(DEBUG) << "Get gpu op success: " << PrimitiveCurVersionTypeName(gpu_desc.type) << " " << node->name_;
      return kernel;
    } else {
      MS_LOG(DEBUG) << "Get gpu op failed, scheduler to cpu: " << PrimitiveCurVersionTypeName(gpu_desc.type) << " "
                    << node->name_;
      if (ret == RET_ERROR) {
        ret = InferNodeShape(node, &infer_shape_interrupt);
        if (ret == RET_INFER_INVALID || ret == RET_OK) {
          op_parameter = op_parameters_[node->output_indices_.at(0)];
        } else {
          MS_LOG(ERROR) << "Try repeat infer fail: " << node->name_;
          return nullptr;
        }
      }
    }
  }
#endif
#if SUPPORT_NPU
  if (context_->IsNpuEnabled()) {
    if (desc.data_type == kNumberTypeFloat16) {
      desc.data_type = kNumberTypeFloat32;
    }
    for (auto tensor : in_tensors) {
      if (tensor->data_type() == kNumberTypeFloat16) {
        tensor->set_data_type(kNumberTypeFloat32);
      }
    }
    kernel::KernelKey npu_desc{kNPU, desc.data_type, desc.type};
    auto ret =
      KernelRegistry::GetInstance()->GetKernel(in_tensors, out_tensors, context_, npu_desc, op_parameter, &kernel);
    if (ret == RET_OK) {
      MS_LOG(DEBUG) << "Get npu op success: " << PrimitiveCurVersionTypeName(npu_desc.type) << " " << node->name_;
      return kernel;
    } else {
      MS_LOG(DEBUG) << "Get npu op failed, scheduler to cpu: " << PrimitiveCurVersionTypeName(npu_desc.type) << " "
                    << node->name_;
      if (ret == RET_ERROR) {
        ret = InferNodeShape(node, &infer_shape_interrupt);
        if (ret == RET_INFER_INVALID || ret == RET_OK) {
          op_parameter = op_parameters_[node->output_indices_.at(0)];
        } else {
          MS_LOG(ERROR) << "Try repeat infer fail: " << node->name_;
          return nullptr;
        }
      }
    }
  }
#endif
  if ((prefer_data_type == kNumberTypeFloat16 || prefer_data_type == kTypeUnknown) &&
      mindspore::lite::IsSupportFloat16() &&
      ((context_->IsCpuFloat16Enabled() && data_type == kNumberTypeFloat32) || data_type == kNumberTypeFloat16)) {
    kernel::KernelKey fp16_cpu_desc{desc.arch, kNumberTypeFloat16, desc.type};
    auto tensor_origin_data_map =
      DequantUtil::DequantTensor(op_parameter, in_tensors, fp16_cpu_desc.data_type, need_restore);
    auto ret =
      KernelRegistry::GetInstance()->GetKernel(in_tensors, out_tensors, context_, fp16_cpu_desc, op_parameter, &kernel);
    DequantUtil::RestoreTensorData(tensor_origin_data_map);
    if (ret == RET_OK) {
      MS_LOG(DEBUG) << "Get fp16 op success: " << PrimitiveCurVersionTypeName(fp16_cpu_desc.type) << " " << node->name_;
      return kernel;
    } else {
      MS_LOG(DEBUG) << "Get fp16 op failed, scheduler to cpu: " << PrimitiveCurVersionTypeName(fp16_cpu_desc.type)
                    << " " << node->name_;
      if (ret == RET_ERROR) {
        ret = InferNodeShape(node, &infer_shape_interrupt);
        if (ret == RET_INFER_INVALID || ret == RET_OK) {
          op_parameter = op_parameters_[node->output_indices_.at(0)];
        } else {
          MS_LOG(ERROR) << "Try repeat infer fail: " << node->name_;
          return nullptr;
        }
      }
    }
  }
  if (data_type == kNumberTypeFloat16) {
    MS_LOG(DEBUG) << "Get fp16 op failed, back to fp32 op.";
    desc.data_type = kNumberTypeFloat32;
  }
  if (prefer_data_type == kNumberTypeFloat32 || prefer_data_type == kTypeUnknown) {
    auto tensor_origin_data_map = DequantUtil::DequantTensor(op_parameter, in_tensors, desc.data_type, need_restore);
    auto ret = KernelRegistry::GetInstance()->GetKernel(in_tensors, out_tensors, context_, desc, op_parameter, &kernel);
    DequantUtil::RestoreTensorData(tensor_origin_data_map);
    if (ret == RET_OK) {
      return kernel;
    } else if (ret == RET_ERROR) {
      ret = InferNodeShape(node, &infer_shape_interrupt);
      if (!(ret == RET_INFER_INVALID || ret == RET_OK)) {
        MS_LOG(ERROR) << "Try repeat infer fail: " << node->name_;
      }
    }
  }
  return nullptr;
}

kernel::LiteKernel *Scheduler::SchedulePartialToKernel(const lite::Model::Node *src_node) {
  MS_ASSERT(src_model_ != nullptr);
  MS_ASSERT(src_node != nullptr);
  auto *primitive = src_node->primitive_;
  MS_ASSERT(primitive != nullptr);
  if (!IsPartialNode(primitive)) {
    return nullptr;
  }
  auto sub_graph_index = GetPartialGraphIndex(src_node->primitive_);
  std::vector<kernel::LiteKernel *> sub_kernels;
  std::vector<lite::Tensor *> in_tensors;
  std::vector<lite::Tensor *> out_tensors;
  auto ret = ScheduleSubGraphToKernels(sub_graph_index, &sub_kernels, &in_tensors, &out_tensors, kNumberTypeFloat32);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Schedule partial failed, name: " << src_node->name_;
    return nullptr;
  }
  auto cur_sub_graph_type = mindspore::lite::Scheduler::GetKernelSubGraphType(sub_kernels.front());
  auto subgraph = CreateSubGraphKernel(sub_kernels, &in_tensors, &out_tensors, cur_sub_graph_type);
  subgraph->set_name("subgraph_" + src_node->name_);
  return subgraph;
}

kernel::LiteKernel *Scheduler::ScheduleNodeToKernel(const lite::Model::Node *src_node, TypeId prefer_data_type) {
  std::vector<Tensor *> inputs;
  std::vector<Tensor *> outputs;
  FindNodeInoutTensors(*src_node, &inputs, &outputs);
  auto *kernel = this->FindBackendKernel(inputs, outputs, src_node, prefer_data_type);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "FindBackendKernel return nullptr, name: " << src_node->name_
                  << ", type: " << PrimitiveTypeName(GetPrimitiveType(src_node->primitive_));
    return nullptr;
  }
  SetKernelTensorDataType(kernel);
  kernel->set_name(src_node->name_);
  return kernel;
}

int Scheduler::ScheduleSubGraphToKernels(size_t subgraph_index, std::vector<kernel::LiteKernel *> *dst_kernels,
                                         std::vector<lite::Tensor *> *in_tensors,
                                         std::vector<lite::Tensor *> *out_tensors, TypeId prefer_data_type) {
  MS_ASSERT(src_model_ != nullptr);
  MS_ASSERT(!src_model_->sub_graphs_.empty());
  MS_ASSERT(src_model_->sub_graphs_.size() > subgraph_index);
  MS_ASSERT(dst_kernels != nullptr);
  MS_ASSERT(dst_kernels->empty());
  auto subgraph = src_model_->sub_graphs_.at(subgraph_index);
  for (auto node_index : subgraph->node_indices_) {
    auto node = src_model_->all_nodes_[node_index];
    MS_ASSERT(node != nullptr);
    auto *primitive = node->primitive_;
    MS_ASSERT(primitive != nullptr);
    kernel::LiteKernel *kernel = nullptr;
    auto prim_type = GetPrimitiveType(primitive);
    if (IsPartialNode(primitive)) {  // sub_graph
      kernel = SchedulePartialToKernel(node);
    } else {  // kernel
      kernel = ScheduleNodeToKernel(node, prefer_data_type);
    }
    if (kernel == nullptr) {
      MS_LOG(ERROR) << "FindBackendKernel return nullptr, name: " << node->name_
                    << ", type: " << PrimitiveTypeName(prim_type);
      return RET_ERROR;
    }
    kernel->set_is_model_output(IsContain(graph_output_node_indexes_, size_t(node_index)));
    dst_kernels->emplace_back(kernel);
  }
  if (in_tensors != nullptr) {
    std::transform(subgraph->input_indices_.begin(), subgraph->input_indices_.end(), std::back_inserter(*in_tensors),
                   [&](const uint32_t index) { return this->src_tensors_->at(index); });
  }
  if (out_tensors != nullptr) {
    std::transform(subgraph->output_indices_.begin(), subgraph->output_indices_.end(), std::back_inserter(*out_tensors),
                   [&](const uint32_t index) { return this->src_tensors_->at(index); });
  }
  return RET_OK;
}

bool Scheduler::KernelFitCurrentSubGraph(const kernel::SubGraphType subgraph_type, const kernel::LiteKernel &kernel) {
  switch (subgraph_type) {
    case kernel::SubGraphType::kNotSubGraph:
    case kernel::SubGraphType::kApuSubGraph:
      return false;
    case kernel::SubGraphType::kGpuSubGraph:
      return kernel.desc().arch == kGPU;
    case kernel::SubGraphType::kNpuSubGraph:
      return kernel.desc().arch == kNPU;
    case kernel::SubGraphType::kCpuFP16SubGraph: {
      auto desc = kernel.desc();
      if (desc.arch != kCPU) {
        return false;
      }
      return (desc.data_type == kNumberTypeFloat16 || desc.data_type == kNumberTypeInt32 ||
              desc.data_type == kNumberTypeInt || desc.data_type == kNumberTypeBool);
    }
    case kernel::SubGraphType::kCpuFP32SubGraph: {
      auto desc = kernel.desc();
      if (desc.arch != kCPU) {
        return false;
      }
      return (desc.data_type == kNumberTypeFloat32 || desc.data_type == kNumberTypeFloat ||
              desc.data_type == kNumberTypeInt8 || desc.data_type == kNumberTypeInt ||
              desc.data_type == kNumberTypeInt32 || desc.data_type == kNumberTypeInt64 ||
              desc.data_type == kNumberTypeUInt8 || desc.data_type == kNumberTypeBool);
    }
    default:
      return false;
  }
}

std::vector<kernel::LiteKernel *> Scheduler::FindAllSubGraphKernels(
  kernel::LiteKernel *head_kernel, std::map<const kernel::LiteKernel *, bool> *sinked_kernel_map) {
  MS_ASSERT(head_kernel != nullptr);
  MS_ASSERT(sinked_kernel_map != nullptr);
  std::vector<kernel::LiteKernel *> sub_kernels;
  if (head_kernel->Type() == schema::PrimitiveType_Switch || head_kernel->Type() == schema::PrimitiveType_Merge) {
    (*sinked_kernel_map)[head_kernel] = true;
    sub_kernels.emplace_back(head_kernel);
    return sub_kernels;
  }
  std::queue<kernel::LiteKernel *> kernel_queue;
  kernel_queue.emplace(head_kernel);
  auto cur_sub_graph_type = mindspore::lite::Scheduler::GetKernelSubGraphType(head_kernel);
  while (!kernel_queue.empty()) {
    auto cur_kernel = kernel_queue.front();
    kernel_queue.pop();
    (*sinked_kernel_map)[cur_kernel] = true;
    sub_kernels.emplace_back(cur_kernel);
    auto post_kernels = cur_kernel->out_kernels();
    for (auto post_kernel : post_kernels) {
      if (post_kernel->subgraph_type() != kernel::kNotSubGraph || post_kernel->Type() == schema::PrimitiveType_Merge ||
          post_kernel->Type() == schema::PrimitiveType_Switch) {
        continue;
      }
      if (cur_sub_graph_type == mindspore::lite::Scheduler::GetKernelSubGraphType(post_kernel)) {
        auto post_kernel_inputs = post_kernel->in_kernels();
        if (std::all_of(post_kernel_inputs.begin(), post_kernel_inputs.end(),
                        [&](kernel::LiteKernel *kernel) { return (*sinked_kernel_map)[kernel]; })) {
          kernel_queue.emplace(post_kernel);
        }
      }
    }
  }
  return sub_kernels;
}

int Scheduler::ConstructSubGraphs(std::vector<kernel::LiteKernel *> *kernels) {
  auto old_kernels = *kernels;
  kernels->clear();
  std::map<const kernel::LiteKernel *, bool> is_kernel_finish;
  for (auto kernel : old_kernels) {
    is_kernel_finish[kernel] = false;
  }

  while (true) {
    auto head_kernel_iter = std::find_if(old_kernels.begin(), old_kernels.end(), [&](const kernel::LiteKernel *kernel) {
      auto kernel_inputs = kernel->in_kernels();
      if (is_kernel_finish[kernel]) {
        return false;
      }
      // when merge is removed, this if is removed automatically
      if (kernel->Type() == schema::PrimitiveType_Merge) {
        return MergeOpIsReady(kernel, is_kernel_finish);
      } else {
        return std::all_of(kernel_inputs.begin(), kernel_inputs.end(),
                           [&](kernel::LiteKernel *kernel) { return is_kernel_finish[kernel]; });
      }
    });
    if (head_kernel_iter == old_kernels.end()) {
      break;
    }
    auto head_kernel = *head_kernel_iter;
    if (head_kernel->subgraph_type() != kernel::kNotSubGraph) {
      is_kernel_finish[head_kernel] = true;
      kernels->emplace_back(head_kernel);
      continue;
    }
    if (head_kernel->desc().arch == mindspore::kernel::kAPU) {
      MS_LOG(ERROR) << "Not support APU now";
      return RET_NOT_SUPPORT;
    }
    auto cur_sub_graph_type = mindspore::lite::Scheduler::GetKernelSubGraphType(head_kernel);
    auto sub_kernels = FindAllSubGraphKernels(head_kernel, &is_kernel_finish);
    auto subgraph = CreateSubGraphKernel(sub_kernels, nullptr, nullptr, cur_sub_graph_type);
    if (subgraph == nullptr) {
      MS_LOG(ERROR) << "Create SubGraphKernel failed";
      return RET_ERROR;
    }
    kernels->emplace_back(subgraph);
  }
  for (auto *subgraph : *kernels) {
    auto ret = subgraph->Init();
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Init SubGraph failed: " << ret;
      return ret;
    }
  }
  return RET_OK;
}

bool Scheduler::MergeOpIsReady(const kernel::LiteKernel *kernel,
                               std::map<const kernel::LiteKernel *, bool> is_kernel_finish) {
  std::map<const lite::Tensor *, bool> merge_in_tensors_map;
  for (auto merge_in_tensor : kernel->in_tensors()) {
    merge_in_tensors_map[merge_in_tensor] = false;
    if (merge_in_tensor->category() == Tensor::CONST_TENSOR || merge_in_tensor->category() == Tensor::CONST_SCALAR ||
        merge_in_tensor->category() == Tensor::GRAPH_INPUT) {
      merge_in_tensors_map[merge_in_tensor] = true;
    }
    for (auto merge_in_kernel : kernel->in_kernels()) {
      for (auto tensor : merge_in_kernel->out_tensors()) {
        if (tensor == merge_in_tensor && is_kernel_finish[merge_in_kernel]) {
          merge_in_tensors_map[merge_in_tensor] = true;
        }
      }
    }
  }
  auto kernel_in_tensors_num = kernel->in_tensors().size();
  return std::all_of(kernel->in_tensors().begin(), kernel->in_tensors().begin() + kernel_in_tensors_num / 2,
                     [&](lite::Tensor *in_tensor) { return merge_in_tensors_map[in_tensor]; }) ||
         std::all_of(kernel->in_tensors().begin() + kernel_in_tensors_num / 2, kernel->in_tensors().end(),
                     [&](lite::Tensor *in_tensor) { return merge_in_tensors_map[in_tensor]; });
}

kernel::SubGraphKernel *Scheduler::CreateSubGraphKernel(const std::vector<kernel::LiteKernel *> &kernels,
                                                        const std::vector<lite::Tensor *> *in_tensors,
                                                        const std::vector<lite::Tensor *> *out_tensors,
                                                        kernel::SubGraphType type) {
  if (type == kernel::kApuSubGraph) {
    return nullptr;
  }
  std::vector<Tensor *> input_tensors;
  std::vector<Tensor *> output_tensors;
  if (in_tensors != nullptr) {
    input_tensors = *in_tensors;
  } else {
    input_tensors = kernel::LiteKernelUtil::SubgraphInputTensors(kernels);
  }
  if (out_tensors != nullptr) {
    output_tensors = *out_tensors;
  } else {
    output_tensors = kernel::LiteKernelUtil::SubgraphOutputTensors(kernels);
  }
  std::vector<kernel::LiteKernel *> input_kernels = kernel::LiteKernelUtil::SubgraphInputNodes(kernels);
  std::vector<kernel::LiteKernel *> output_kernels = kernel::LiteKernelUtil::SubgraphOutputNodes(kernels);
  if (type == kernel::kGpuSubGraph) {
#if GPU_OPENCL
    auto sub_kernel = new (std::nothrow)
      kernel::OpenCLSubGraph(input_tensors, output_tensors, input_kernels, output_kernels, kernels, context_);
    if (sub_kernel == nullptr) {
      MS_LOG(ERROR) << "Create OpenCLSubGraph failed";
      return nullptr;
    }
    return sub_kernel;
#elif GPU_VULKAN
    return nullptr;
#else
    return nullptr;
#endif
  }
  if (type == kernel::kNpuSubGraph) {
#if SUPPORT_NPU
    auto sub_kernel = new (std::nothrow) kernel::SubGraphNpuKernel(input_tensors, output_tensors, input_kernels,
                                                                   output_kernels, kernels, context_, npu_manager_);
    if (sub_kernel == nullptr) {
      MS_LOG(ERROR) << "NPU subgraph new failed.";
      return nullptr;
    }
    return sub_kernel;
#else
    return nullptr;
#endif
  }
  if (type == kernel::kCpuFP16SubGraph) {
#ifdef ENABLE_FP16
    auto sub_kernel = new (std::nothrow)
      kernel::CpuFp16SubGraph(input_tensors, output_tensors, input_kernels, output_kernels, kernels, context_);
    return sub_kernel;
#else
    MS_LOG(ERROR) << "FP16 subgraph is not supported!";
    return nullptr;
#endif
  }
  if (type == kernel::kCpuFP32SubGraph) {
    auto sub_kernel = new (std::nothrow)
      kernel::CpuFp32SubGraph(input_tensors, output_tensors, input_kernels, output_kernels, kernels, context_);
    return sub_kernel;
  }
  return nullptr;
}

TypeId Scheduler::GetFirstFp32Fp16OrInt8Type(const std::vector<Tensor *> &in_tensors) {
  for (const auto &tensor : in_tensors) {
    auto dtype = tensor->data_type();
    if (dtype == kObjectTypeString) {
      return kNumberTypeFloat32;
    }
    if (dtype == kObjectTypeTensorType) {
      auto tensor_list = reinterpret_cast<TensorList *>(tensor);
      auto tensor_list_dtype = tensor_list->data_type();
      if (tensor_list_dtype == kNumberTypeFloat32 || tensor_list_dtype == kNumberTypeFloat16 ||
          tensor_list_dtype == kNumberTypeInt8 || tensor_list_dtype == kNumberTypeInt32 ||
          tensor_list_dtype == kNumberTypeBool) {
        return tensor_list_dtype;
      }
    }
    if (dtype == kNumberTypeFloat32 || dtype == kNumberTypeFloat16 || dtype == kNumberTypeInt8 ||
        dtype == kNumberTypeInt32 || dtype == kNumberTypeBool) {
      return dtype;
    }
  }
  MS_ASSERT(!in_tensors.empty());
  return in_tensors[0]->data_type();
}

void Scheduler::SetKernelTensorDataType(kernel::LiteKernel *kernel) {
  if (kernel->desc().arch != kernel::KERNEL_ARCH::kCPU) {
    return;
  }
  if (kernel->desc().data_type == kNumberTypeFloat16) {
    for (auto tensor : kernel->out_tensors()) {
      if (tensor->data_type() == kNumberTypeFloat32) {
        tensor->set_data_type(kNumberTypeFloat16);
      }
    }
  } else if (kernel->desc().data_type == kNumberTypeFloat32) {
    for (auto tensor : kernel->in_tensors()) {
      if (!tensor->IsConst() && tensor->data_type() == kNumberTypeFloat16) {
        tensor->set_data_type(kNumberTypeFloat32);
      }
    }
    for (auto tensor : kernel->out_tensors()) {
      if (tensor->data_type() == kNumberTypeFloat16) {
        tensor->set_data_type(kNumberTypeFloat32);
      }
    }
  }
}

kernel::SubGraphType Scheduler::GetKernelSubGraphType(const kernel::LiteKernel *kernel) {
  if (kernel == nullptr) {
    return kernel::kNotSubGraph;
  }
  auto desc = kernel->desc();
  if (desc.arch == kernel::KERNEL_ARCH::kGPU) {
    return kernel::kGpuSubGraph;
  } else if (desc.arch == kernel::KERNEL_ARCH::kNPU) {
    return kernel::kNpuSubGraph;
  } else if (desc.arch == kernel::KERNEL_ARCH::kAPU) {
    return kernel::kApuSubGraph;
  } else if (desc.arch == kernel::KERNEL_ARCH::kCPU) {
    if (desc.data_type == kNumberTypeFloat16) {
      return kernel::kCpuFP16SubGraph;
    } else if (desc.data_type == kNumberTypeFloat32 || desc.data_type == kNumberTypeInt8 ||
               desc.data_type == kNumberTypeInt32 || desc.data_type == kNumberTypeInt64 ||
               desc.data_type == kNumberTypeUInt8 || desc.data_type == kNumberTypeBool) {
      return kernel::kCpuFP32SubGraph;
    }
  }
  return kernel::kNotSubGraph;
}

void Scheduler::FindAllInoutKernels(const std::vector<kernel::LiteKernel *> &kernels) {
  for (auto *kernel : kernels) {
    MS_ASSERT(kernel != nullptr);
    kernel->FindInoutKernels(kernels);
  }
}

int Scheduler::RunPass(std::vector<kernel::LiteKernel *> *dst_kernels) {
  int ret = RET_OK;
#if SUPPORT_NPU
  if (!context_->IsNpuEnabled()) {
    return RET_OK;
  }
  auto transform_pass = new NPUTransformPass(context_, dst_kernels, src_tensors_);
  MS_ASSERT(npu_pass_manager_ != nullptr);
  npu_pass_manager_->AddPass(transform_pass);
  auto concat_format_pass = new NPUInsertTransformPass(context_, dst_kernels, src_tensors_);
  npu_pass_manager_->AddPass(concat_format_pass);
  auto fusion_pass = new NPUFusionPass(dst_kernels);
  npu_pass_manager_->AddPass(fusion_pass);

  ret = npu_pass_manager_->Run();
#endif
  return ret;
}
}  // namespace mindspore::lite
