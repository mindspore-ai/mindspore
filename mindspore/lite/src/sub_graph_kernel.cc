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

#include "src/sub_graph_kernel.h"
#include <algorithm>
#include "src/tensor.h"
#ifndef CONTROLFLOW_TENSORLIST_CLIP
#include "src/tensorlist.h"
#endif
#ifdef ENABLE_FP16
#include "src/runtime/kernel/arm/fp16/fp16_op_handler.h"
#endif
#include "src/common/version_manager.h"
#include "src/runtime/infer_manager.h"
#include "src/common/tensor_util.h"
#include "src/common/utils.h"

namespace mindspore::kernel {
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_INFER_ERR;
using mindspore::lite::RET_INFER_INVALID;
using mindspore::lite::RET_OK;

int SubGraphKernel::Prepare() {
  for (auto node : this->nodes_) {
    if (node == nullptr) {
      MS_LOG(ERROR) << "node in Subgraph is nullptr";
      return mindspore::lite::RET_NULL_PTR;
    }
    auto ret = node->Prepare();
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "prepare node " << node->name() << " failed";
      return ret;
    }
  }
  return RET_OK;
}

std::string SubGraphKernel::ToString() const {
  std::ostringstream oss;
  oss << "===============================================" << std::endl << "Subgraph type : " << this->subgraph_type_;
  oss << std::endl << this->in_tensors().size() << "Subgraph inputTensors:";
  for (auto tensor : in_tensors()) {
    oss << " " << tensor;
  }
  oss << std::endl << this->out_tensors().size() << "Subgraph outputTensors:";
  for (auto tensor : out_tensors()) {
    oss << " " << tensor;
  }
  oss << std::endl << "Subgraph input nodes :" << std::endl;
  for (auto kernel : this->in_nodes_) {
    oss << " " << kernel->ToString() << std::endl;
  }
  oss << std::endl << "Subgraph output nodes :" << std::endl;
  for (auto kernel : this->out_nodes_) {
    oss << " " << kernel->ToString() << std::endl;
  }
  oss << std::endl << nodes_.size() << "　nodes in subgraph :";
  for (auto kernel : this->nodes_) {
    oss << " " << kernel->name();
  }
  return oss.str();
}

int SubGraphKernel::Execute(const KernelCallBack &before, const KernelCallBack &after) {
  if (this->executor_ == nullptr) {
    MS_LOG(ERROR) << "executor is nullptr";
    return RET_ERROR;
  }
  auto ret = executor_->Run(this->in_tensors(), this->out_tensors(), this->nodes_, before, after);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Run sub graph failed: " << ret;
    return ret;
  }

  return lite::RET_OK;
}

int SubGraphKernel::ReSize() {
  for (auto kernel : nodes_) {
    if (kernel == nullptr) {
      MS_LOG(ERROR) << "input kernel is nullptr!";
      return RET_ERROR;
    }
    if (kernel->subgraph_type() != kernel::kNotSubGraph) {
      MS_LOG(ERROR) << "all nodes in should be kernel";
      return RET_ERROR;
    }
    std::vector<lite::Tensor *> inputs = kernel->in_tensors();
    std::vector<lite::Tensor *> outputs = kernel->out_tensors();
    for (auto &output : outputs) {
      output->FreeData();
    }
    int ret;
#ifndef CUSTOM_KERNEL_REGISTRY_CLIP
    ret = lite::KernelInferShape(inputs, outputs, kernel->kernel()->primitive(), kernel->Context()->GetProviders(),
                                 schema_version_, kernel->kernel());
    if (ret == lite::RET_NOT_SUPPORT) {
#endif
      auto parameter = kernel->op_parameter();
      if (parameter == nullptr) {
        MS_LOG(ERROR) << "kernel(" << kernel->name() << ")'s op_parameter is nullptr!";
        return RET_ERROR;
      }
      ret = lite::KernelInferShape(inputs, outputs, parameter);
#ifndef CUSTOM_KERNEL_REGISTRY_CLIP
    }
#endif
    if (ret == RET_INFER_INVALID) {
      MS_LOG(INFO) << "InferShape shouldn't be done before runtime, type:"
                   << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(kernel->type()))
                   << "flag set to false.";
    } else if (ret != RET_OK) {
      MS_LOG(ERROR) << "InferShape failed, type: "
                    << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(kernel->type()));
      return RET_INFER_ERR;
    }
    if (ret == RET_OK) {
      ret = kernel->ReSize();
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "kernel " << kernel->name() << " resize fail!ret = " << ret;
        return ret;
      }
    }
  }
  return RET_OK;
}
void SubGraphKernel::InitInputTensorInitRefCount() {
  for (auto &input : this->in_tensors()) {
    int input_init_ref_count = input->init_ref_count();
    for (auto *node : nodes_) {
      input_init_ref_count += std::count_if(node->in_tensors().begin(), node->in_tensors().end(),
                                            [&input](lite::Tensor *item) { return item == input; });
    }
    input->set_init_ref_count(input_init_ref_count);
  }
}

void SubGraphKernel::InitOutTensorInitRefCount(const std::vector<LiteKernel *> *mask_kernels) {
  for (auto *node : nodes_) {
    node->InitOutTensorInitRefCount(mask_kernels);
  }
}

void SubGraphKernel::DropNode(LiteKernel *node) {
  lite::VectorErase(&nodes_, node);
  lite::VectorErase(&in_nodes_, node);
  lite::VectorErase(&out_nodes_, node);
}

int CustomSubGraph::Prepare() {
  auto ret = SubGraphKernel::Prepare();
  if (ret != RET_OK) {
    return ret;
  }
  if (nodes_.size() < 1) {
    return RET_OK;
  }
  auto provider = nodes_[0]->desc().provider;
  auto context = this->Context();
  AllocatorPtr allocator = context->allocator;
  auto iter = std::find_if(context->device_list_.begin(), context->device_list_.end(),
                           [&provider](const auto &dev) { return dev.provider_ == provider; });
  if (iter != context->device_list_.end()) {
    allocator = iter->allocator_;
  }

  for (size_t i = 0; i < nodes_.size() - 1; ++i) {
    auto node = nodes_[i];
    for (auto tensor : node->out_tensors()) {
      MS_ASSERT(tensor != nullptr);
      tensor->set_allocator(allocator);
    }
  }

  auto node = nodes_[nodes_.size() - 1];
  for (auto tensor : node->out_tensors()) {
    MS_ASSERT(tensor != nullptr);
    tensor->set_allocator(context->allocator);
  }
  return RET_OK;
}

int CustomSubGraph::Execute(const KernelCallBack &before, const KernelCallBack &after) {
  for (auto kernel : nodes_) {
    MS_ASSERT(kernel != nullptr);
    auto ret = kernel->Execute(before, after);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "run kernel failed, name: " << kernel->name();
      return ret;
    }
  }

  return RET_OK;
}

int CpuSubGraph::Prepare() {
  auto ret = SubGraphKernel::Prepare();
  if (ret != RET_OK) {
    return ret;
  }
  for (auto node : nodes_) {
    for (auto tensor : node->out_tensors()) {
      MS_ASSERT(tensor != nullptr);
      tensor->set_allocator(this->Context()->allocator);
    }
  }
  for (auto &out : this->out_tensors()) {
    out->set_allocator(this->Context()->allocator);
  }
  return RET_OK;
}

int CpuSubGraph::Execute(const KernelCallBack &before, const KernelCallBack &after) {
  MS_ASSERT(this->Context()->allocator.get() != nullptr);

  for (auto *kernel : nodes_) {
    MS_ASSERT(kernel != nullptr);
    auto ret = kernel->Execute(before, after);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "run kernel failed, name: " << kernel->name();
      return ret;
    }
  }
  return RET_OK;
}
}  // namespace mindspore::kernel
