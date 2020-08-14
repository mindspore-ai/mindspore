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
#include <vector>
#include <algorithm>
#include "include/errorcode.h"
#include "src/kernel_factory.h"
#include "src/common/graph_util.h"
#include "src/common/utils.h"
#if SUPPORT_GPU
#include "src/runtime/kernel/opencl/subgraph_opencl_kernel.h"
#endif

namespace mindspore::lite {
int Scheduler::Schedule(const lite::Model *model, std::vector<tensor::Tensor *> *tensors,
                        std::vector<kernel::LiteKernel *> *kernels) {
  // 1. op ---> kernel
  // 2. sub graph
  // 3. kernels (kernels --> subGraph)
  int ret = InitOp2Kernel(model, tensors, kernels);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "init op to kernel failed.";
    return RET_ERROR;
  }

  kernel::LiteKernelUtil::TopologicalSortKernels(*kernels);

  ConstructSubgraphs(kernels);

  MS_LOG(DEBUG) << "schedule kernels success.";
  return RET_OK;
}

int Scheduler::InitOp2Kernel(const lite::Model *model, std::vector<tensor::Tensor *> *tensors,
                             std::vector<kernel::LiteKernel *> *kernels) {
  MS_EXCEPTION_IF_NULL(model);
  MS_EXCEPTION_IF_NULL(tensors);
  MS_EXCEPTION_IF_NULL(kernels);
  auto meta_graph = model->GetMetaGraph();
  MS_EXCEPTION_IF_NULL(meta_graph);
  uint32_t kernelCount = meta_graph->nodes()->size();
  auto graph_output_node_indexes = GetGraphOutputNodes(meta_graph);
  for (uint32_t i = 0; i < kernelCount; i++) {
    auto cNode = meta_graph->nodes()->GetAs<schema::CNode>(i);
    std::vector<tensor::Tensor *> inputs;
    std::vector<tensor::Tensor *> outputs;
    auto inIndexes = cNode->inputIndex();
    for (size_t j = 0; j < inIndexes->size(); j++) {
      inputs.emplace_back(tensors->at(size_t(inIndexes->GetAs<uint32_t>(j))));
    }
    auto outIndexes = cNode->outputIndex();
    for (size_t j = 0; j < outIndexes->size(); j++) {
      outputs.emplace_back(tensors->at(size_t(outIndexes->GetAs<uint32_t>(j))));
    }
    auto *primitive = model->GetOp(cNode->name()->str());
    if (primitive == nullptr) {
      MS_LOG(ERROR) << "Op " << cNode->name()->str() << " should exist in model, type: "
                    << schema::EnumNamePrimitiveType(cNode->primitive()->value_type());
      return RET_ERROR;
    }
    if (!context_->infer_shape_interrupt_) {
      auto ret = primitive->InferShape(inputs, outputs);
      if (ret == RET_INFER_INVALID) {
        MS_LOG(INFO) << "InferShape shouldn't be done before runtime, name: " << cNode->name()->str()
                     << ", type: " << schema::EnumNamePrimitiveType(cNode->primitive()->value_type())
                     << "flag set to false.";
        primitive->SetInferFlag(false);
        context_->InferShapeInterrupt();
      } else if (ret != RET_OK) {
        MS_LOG(ERROR) << "InferShape failed, name: " << cNode->name()->str()
                      << ", type: " << schema::EnumNamePrimitiveType(cNode->primitive()->value_type());
        return RET_INFER_ERR;
      }
    } else {
      primitive->SetInferFlag(false);
    }

    auto *kernel = this->ScheduleNode(inputs, outputs, primitive);
    if (nullptr == kernel) {
      MS_LOG(ERROR) << "ScheduleNode return nullptr, name: " << cNode->name()->str()
                    << ", type: " << schema::EnumNamePrimitiveType(cNode->primitive()->value_type());
      return RET_ERROR;
    }
    kernel->set_name(cNode->name()->str());
    kernel->set_is_model_output(IsContain(graph_output_node_indexes, size_t(i)));
    kernels->emplace_back(kernel);
  }
  return RET_OK;
}

void Scheduler::ConstructSubgraphs(std::vector<kernel::LiteKernel *> *kernels) {
  uint32_t kernel_count = kernels->size();
  std::vector<kernel::LiteKernel *> sub_kernels;
  std::vector<std::vector<kernel::LiteKernel *>> sub_kernels_list;

  kernel::KERNEL_ARCH prev_arch = kernels->front()->desc().arch;
  for (uint32_t i = 0; i < kernel_count; ++i) {
    auto curr_kernel = kernels->at(i);
    auto curr_arch = curr_kernel->desc().arch;
    if (curr_arch == prev_arch) {
      sub_kernels.emplace_back(curr_kernel);
    }
    if ((curr_arch != prev_arch) || (i == kernel_count - 1)) {
      sub_kernels_list.emplace_back(sub_kernels);
      sub_kernels.clear();
      sub_kernels.emplace_back(curr_kernel);
    }
    prev_arch = curr_arch;
  }

  std::vector<kernel::LiteKernel *> subgraph_kernels;
  for (auto temp_kernels : sub_kernels_list) {
    kernel::KERNEL_ARCH arch = temp_kernels.front()->desc().arch;
    if (arch == kernel::KERNEL_ARCH::kCPU) {
      for (auto kernel : temp_kernels) {
        for (auto tensor : kernel->out_tensors()) {
          tensor->set_allocator(context_->allocator.get());
        }
      }
      std::copy(temp_kernels.begin(), temp_kernels.end(), std::back_inserter(subgraph_kernels));
    } else {
      auto subgraph_kernel = CreateSubKernel(temp_kernels, arch);
      subgraph_kernels.emplace_back(subgraph_kernel);
    }
  }
  kernels->clear();
  kernels->insert(kernels->begin(), subgraph_kernels.begin(), subgraph_kernels.end());
}

kernel::LiteKernel *Scheduler::CreateSubKernel(const std::vector<kernel::LiteKernel *> &kernels,
                                               kernel::KERNEL_ARCH arch) {
  kernel::LiteKernel *sub_kernel = nullptr;
#if SUPPORT_GPU
  if (arch == kernel::KERNEL_ARCH::kGPU) {
    auto head_kernel = kernels.front();
    auto tail_kernel = kernels.back();
    std::vector<kernel::LiteKernel *> input_kernels{head_kernel};
    std::vector<kernel::LiteKernel *> output_kernels{tail_kernel};
    std::vector<tensor::Tensor *> input_tensors;
    std::vector<tensor::Tensor *> output_tensors;
    for (auto tensor : head_kernel->in_tensors()) {
      if (tensor->Data() == nullptr) {
        input_tensors.emplace_back(tensor);
      }
    }
    for (auto tensor : tail_kernel->out_tensors()) {
      if (tensor->Data() == nullptr) {
        output_tensors.emplace_back(tensor);
      }
    }
    //    std::vector<tensor::Tensor *> input_tensors = kernel::LiteKernelUtil::SubgraphInputTensors(kernels);
    //    std::vector<tensor::Tensor *> output_tensors = kernel::LiteKernelUtil::SubgraphOutputTensors(kernels);
    //    std::vector<kernel::LiteKernel *> input_kernels = kernel::LiteKernelUtil::SubgraphInputKernels(kernels);
    //    std::vector<kernel::LiteKernel *> output_kernels = kernel::LiteKernelUtil::SubgraphOutputKernels(kernels);
    sub_kernel =
      new kernel::SubGraphOpenCLKernel(input_tensors, output_tensors, input_kernels, output_kernels, kernels);
    sub_kernel->Init();
  } else if (arch == kernel::KERNEL_ARCH::kNPU) {
    MS_LOG(ERROR) << "NPU kernel is not supported";
  } else {
    MS_LOG(ERROR) << "unsupported kernel arch: " << arch;
  }
#endif
  return sub_kernel;
}

kernel::LiteKernel *Scheduler::ScheduleNode(const std::vector<tensor::Tensor *> &in_tensors,
                                            const std::vector<tensor::Tensor *> &out_tensors,
                                            const lite::Primitive *primitive) {
  // todo: support NPU, APU
  MS_ASSERT(nullptr != primitive);
  auto data_type = in_tensors.front()->data_type();
  kernel::KernelKey desc{kernel::KERNEL_ARCH::kCPU, data_type, primitive->Type()};
  if (context_->device_ctx_.type == DT_GPU) {
    desc.arch = kernel::KERNEL_ARCH::kGPU;
    auto *kernel = KernelFactory::GetInstance()->GetKernel(in_tensors, out_tensors, primitive, context_, desc);
    if (nullptr != kernel) {
      kernel->set_desc(desc);
      return kernel;
    }
  }

  desc.arch = kernel::KERNEL_ARCH::kCPU;
  kernel::LiteKernel *kernel = nullptr;
  if (data_type == kNumberTypeFloat32) {
    // check if support fp16
    kernel::KernelKey key{desc.arch, kNumberTypeFloat16, desc.type};
    kernel = KernelFactory::GetInstance()->GetKernel(in_tensors, out_tensors, primitive, context_, key);
    if (kernel != nullptr) {
      MS_LOG(DEBUG) << "Get fp16 op success.";
      kernel->set_desc(desc);
      return kernel;
    }
    MS_LOG(DEBUG) << "Get fp16 op failed, back to fp32 op.";
    kernel = KernelFactory::GetInstance()->GetKernel(in_tensors, out_tensors, primitive, context_, desc);
  } else {
    kernel = KernelFactory::GetInstance()->GetKernel(in_tensors, out_tensors, primitive, context_, desc);
  }
  if (kernel != nullptr) {
    kernel->set_desc(desc);
    return kernel;
  }
  return nullptr;
}
}  // namespace mindspore::lite
