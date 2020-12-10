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
#include "include/errorcode.h"
#include "src/common/graph_util.h"
#include "src/common/utils.h"
#include "src/kernel_registry.h"
#include "src/sub_graph_kernel.h"
#if SUPPORT_GPU
#include "src/runtime/kernel/opencl/opencl_subgraph.h"
#include "src/runtime/opencl/opencl_runtime.h"
#endif
#if SUPPORT_NPU
#include "src/runtime/agent/npu/subgraph_npu_kernel.h"
#include "src/runtime/agent/npu/npu_manager.h"
#endif
namespace mindspore::lite {
using kernel::KERNEL_ARCH::kCPU;
using kernel::KERNEL_ARCH::kGPU;
using kernel::KERNEL_ARCH::kNPU;

int Scheduler::Schedule(const lite::Model *model, std::vector<Tensor *> *tensors,
                        std::vector<kernel::LiteKernel *> *kernels) {
  int ret = InferShape(model, tensors);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "op infer shape failed.";
    return ret;
  }
  ret = BuildKernels(model, tensors, kernels);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "init op to kernel failed.";
    return ret;
  }

  kernel::LiteKernelUtil::InitIOKernels(*kernels);

  ret = ConstructSubGraphs(kernels);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ConstructSubGraphs failed.";
    return ret;
  }

  kernel::LiteKernelUtil::InitIOKernels(*kernels);

  MS_LOG(DEBUG) << "schedule kernels success.";
  return RET_OK;
}

int Scheduler::ReSizeKernels(const std::vector<kernel::LiteKernel *> &kernels) {
  bool infer_shape_interrupt = false;
  for (auto kernel : kernels) {
    if (kernel == nullptr) {
      MS_LOG(ERROR) << "input kernel is nullptr!";
      return RET_ERROR;
    }
    if (kernel->subgraph_type() == kernel::kNotSubGraph) {
      MS_LOG(ERROR) << "All node in graph should be sub_graph";
      return RET_ERROR;
    }
    auto sub_graph = reinterpret_cast<kernel::SubGraphKernel *>(kernel);
    auto ret = sub_graph->ReSize(infer_shape_interrupt);
    if (ret == RET_INFER_INVALID) {
      MS_LOG(INFO) << "InferShape is interrupted";
      infer_shape_interrupt = true;
      continue;
    }
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "ReSize node " << kernel->name() << " failed";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

int Scheduler::InferShape(const lite::Model *model, std::vector<Tensor *> *tensors) {
  MS_ASSERT(model != nullptr);
  MS_ASSERT(tensors != nullptr);
  bool infer_shape_interrupt = false;
  uint32_t kernelCount = model->all_nodes_.size();
  for (uint32_t i = 0; i < kernelCount; ++i) {
    auto node = model->all_nodes_[i];
    MS_ASSERT(node != nullptr);
    std::vector<Tensor *> inputs;
    std::vector<Tensor *> outputs;
    auto in_size = node->input_indices_.size();
    inputs.reserve(in_size);
    for (size_t j = 0; j < in_size; ++j) {
      inputs.emplace_back(tensors->at(node->input_indices_[j]));
    }
    auto out_size = node->output_indices_.size();
    outputs.reserve(out_size);
    for (size_t j = 0; j < out_size; ++j) {
      outputs.emplace_back(tensors->at(node->output_indices_[j]));
    }
    auto *primitive = node->primitive_;
    if (primitive == nullptr) {
      MS_LOG(ERROR) << "Op " << node->name_ << " should exist in model!";
      return RET_ERROR;
    }
    bool infer_valid = std::all_of(inputs.begin(), inputs.end(), [](const Tensor *tensor) {
      auto shape = tensor->shape();
      return std::all_of(shape.begin(), shape.end(), [](const int dim) { return dim != -1; });
    });
    if (!infer_valid) {
      infer_shape_interrupt = true;
    }
    primitive->set_infer_flag(!infer_shape_interrupt);
    auto ret = primitive->InferShape(inputs, outputs);
    if (ret == RET_INFER_INVALID) {
      MS_LOG(INFO) << "InferShape shouldn't be done before runtime, name: " << node->name_
                   << ", type: " << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(primitive->Type()))
                   << "flag set to false.";
      primitive->set_infer_flag(false);
      infer_shape_interrupt = true;
    } else if (ret != RET_OK) {
      MS_LOG(ERROR) << "InferShape failed, name: " << node->name_ << ", type: "
                    << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(primitive->Type()));
      return RET_INFER_ERR;
    } else {
      for (auto &output : outputs) {
        if (output->ElementsNum() >= MAX_MALLOC_SIZE / static_cast<int>(sizeof(int64_t))) {
          MS_LOG(ERROR) << "The size of output tensor is too big";
          return RET_ERROR;
        }
      }
    }
  }

  return RET_OK;
}

int Scheduler::BuildKernels(const lite::Model *model, const std::vector<Tensor *> *tensors,
                            std::vector<kernel::LiteKernel *> *kernels) {
  MS_ASSERT(model != nullptr);
  MS_ASSERT(tensors != nullptr);
  uint32_t kernelCount = model->all_nodes_.size();
  auto graph_output_node_indexes = GetGraphOutputNodes(model);
  for (uint32_t i = 0; i < kernelCount; ++i) {
    auto node = model->all_nodes_[i];
    MS_ASSERT(node != nullptr);
    std::vector<Tensor *> inputs;
    std::vector<Tensor *> outputs;
    auto in_size = node->input_indices_.size();
    inputs.reserve(in_size);
    for (size_t j = 0; j < in_size; ++j) {
      inputs.emplace_back(tensors->at(node->input_indices_[j]));
    }
    auto out_size = node->output_indices_.size();
    outputs.reserve(out_size);
    for (size_t j = 0; j < out_size; ++j) {
      outputs.emplace_back(tensors->at(node->output_indices_[j]));
    }
    auto *primitive = node->primitive_;
    MS_ASSERT(primitive != nullptr);
    auto *kernel = this->ScheduleNode(inputs, outputs, primitive, node);
    if (kernel == nullptr) {
      MS_LOG(ERROR) << "ScheduleNode return nullptr, name: " << node->name_ << ", type: "
                    << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(primitive->Type()));
      return RET_ERROR;
    }
    SetKernelTensorDataType(kernel);
    kernel->set_name(node->name_);
    kernel->set_is_model_output(IsContain(graph_output_node_indexes, size_t(i)));
    kernels->emplace_back(kernel);
  }

  return RET_OK;
}

std::vector<kernel::LiteKernel *> Scheduler::FindAllSubGraphKernels(
  kernel::LiteKernel *head_kernel, std::map<const kernel::LiteKernel *, bool> *sinked_kernel_map) {
  MS_ASSERT(head_kernel != nullptr);
  MS_ASSERT(sinked_kernel_map != nullptr);
  std::vector<kernel::LiteKernel *> sub_kernels;
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
  std::map<const kernel::LiteKernel *, bool> is_kernel_sinked;
  for (auto kernel : old_kernels) {
    is_kernel_sinked[kernel] = false;
  }

  while (true) {
    auto head_kernel_iter = std::find_if(old_kernels.begin(), old_kernels.end(), [&](const kernel::LiteKernel *kernel) {
      auto kernel_inputs = kernel->in_kernels();
      return !is_kernel_sinked[kernel] &&
             std::all_of(kernel_inputs.begin(), kernel_inputs.end(),
                         [&](kernel::LiteKernel *kernel) { return is_kernel_sinked[kernel]; });
    });
    if (head_kernel_iter == old_kernels.end()) {
      break;
    }
    auto head_kernel = *head_kernel_iter;
    if (head_kernel->desc().arch == mindspore::kernel::kAPU) {
      MS_LOG(ERROR) << "Not support APU now";
      return RET_NOT_SUPPORT;
    }
    auto cur_sub_graph_type = mindspore::lite::Scheduler::GetKernelSubGraphType(head_kernel);
    auto sub_kernels = FindAllSubGraphKernels(head_kernel, &is_kernel_sinked);
    auto subgraph = CreateSubGraphKernel(sub_kernels, cur_sub_graph_type, kernels->size());
    if (subgraph == nullptr) {
      MS_LOG(ERROR) << "Create SubGraphKernel failed";
      return RET_ERROR;
    }
    kernels->emplace_back(subgraph);
  }
  return RET_OK;
}

kernel::SubGraphKernel *Scheduler::CreateSubGraphKernel(const std::vector<kernel::LiteKernel *> &kernels,
                                                        kernel::SubGraphType type, int index) {
  if (type == kernel::kApuSubGraph) {
    return nullptr;
  }
  std::vector<Tensor *> input_tensors = kernel::LiteKernelUtil::SubgraphInputTensors(kernels);
  std::vector<Tensor *> output_tensors = kernel::LiteKernelUtil::SubgraphOutputTensors(kernels);
  std::vector<kernel::LiteKernel *> input_kernels = kernel::LiteKernelUtil::SubgraphInputKernels(kernels);
  std::vector<kernel::LiteKernel *> output_kernels = kernel::LiteKernelUtil::SubgraphOutputKernels(kernels);
  if (type == kernel::kGpuSubGraph) {
#if SUPPORT_GPU
    auto sub_kernel = new (std::nothrow)
      kernel::OpenCLSubGraph(input_tensors, output_tensors, input_kernels, output_kernels, kernels, context_);
    return sub_kernel;
#else
    return nullptr;
#endif
  }
  if (type == kernel::kNpuSubGraph) {
#if SUPPORT_NPU
    auto sub_kernel =
      new kernel::SubGraphNpuKernel(input_tensors, output_tensors, input_kernels, output_kernels, kernels, context_);
    sub_kernel->SetIndex(index);
    if (sub_kernel->Init() != RET_OK) {
      return nullptr;
    }
    return sub_kernel;
#else
    return nullptr;
#endif
  }
  if (type == kernel::kCpuFP16SubGraph) {
    auto sub_kernel = new (std::nothrow)
      kernel::CpuFp16SubGraph(input_tensors, output_tensors, input_kernels, output_kernels, kernels, context_);
    return sub_kernel;
  }
  if (type == kernel::kCpuFP32SubGraph) {
    auto sub_kernel = new (std::nothrow)
      kernel::CpuFp32SubGraph(input_tensors, output_tensors, input_kernels, output_kernels, kernels, context_);
    return sub_kernel;
  }
  return nullptr;
}

kernel::LiteKernel *Scheduler::ScheduleNode(const std::vector<Tensor *> &in_tensors,
                                            const std::vector<Tensor *> &out_tensors,
                                            const mindspore::lite::PrimitiveC *primitive, const Model::Node *node) {
  MS_ASSERT(primitive != nullptr);
  TypeId data_type = GetFirstFp32Fp16OrInt8Type(in_tensors);
  kernel::KernelKey desc{kCPU, data_type, static_cast<schema::PrimitiveType>(primitive->Type())};
#if SUPPORT_NPU
  if (context_->IsNpuEnabled()) {
    kernel::KernelKey npu_desc{kNPU, desc.data_type, desc.type};
    auto *kernel = KernelRegistry::GetInstance()->GetKernel(in_tensors, out_tensors, primitive, context_, npu_desc);
    if (kernel != nullptr) {
      MS_LOG(DEBUG) << "Get npu op success: " << schema::EnumNamePrimitiveType(npu_desc.type) << " " << node->name_;
      return kernel;
    } else {
      MS_LOG(DEBUG) << "Get npu op failed, scheduler to cpu: " << schema::EnumNamePrimitiveType(npu_desc.type) << " "
                    << node->name_;
    }
  }
#endif
#if SUPPORT_GPU
  if (context_->IsGpuEnabled()) {
    kernel::KernelKey gpu_desc{kGPU, desc.data_type, desc.type};
    auto *kernel = KernelRegistry::GetInstance()->GetKernel(in_tensors, out_tensors, primitive, context_, gpu_desc);
    if (kernel != nullptr) {
      MS_LOG(DEBUG) << "Get gpu op success: " << schema::EnumNamePrimitiveType(gpu_desc.type) << " " << node->name_;
      return kernel;
    } else {
      MS_LOG(DEBUG) << "Get gpu op failed, scheduler to cpu: " << schema::EnumNamePrimitiveType(gpu_desc.type) << " "
                    << node->name_;
    }
  }
#endif
  if (mindspore::lite::IsSupportFloat16() &&
      ((context_->IsCpuFloat16Enabled() && data_type == kNumberTypeFloat32) || data_type == kNumberTypeFloat16)) {
    kernel::KernelKey fp16_cpu_desc{desc.arch, kNumberTypeFloat16, desc.type};
    auto *kernel =
      KernelRegistry::GetInstance()->GetKernel(in_tensors, out_tensors, primitive, context_, fp16_cpu_desc);
    if (kernel != nullptr) {
      MS_LOG(DEBUG) << "Get fp16 op success: " << schema::EnumNamePrimitiveType(fp16_cpu_desc.type) << " "
                    << node->name_;
      return kernel;
    }
  }
  if (data_type == kNumberTypeFloat16) {
    MS_LOG(DEBUG) << "Get fp16 op failed, back to fp32 op.";
    desc.data_type = kNumberTypeFloat32;
  }
  auto *kernel = KernelRegistry::GetInstance()->GetKernel(in_tensors, out_tensors, primitive, context_, desc);
  if (kernel != nullptr) {
    return kernel;
  }
  return nullptr;
}

TypeId Scheduler::GetFirstFp32Fp16OrInt8Type(const std::vector<Tensor *> &in_tensors) {
  for (const auto &tensor : in_tensors) {
    auto dtype = tensor->data_type();
    if (dtype == kObjectTypeString) {
      return kNumberTypeFloat32;
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
}  // namespace mindspore::lite
