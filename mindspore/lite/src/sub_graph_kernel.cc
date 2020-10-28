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
#include "src/tensor.h"
#ifdef ENABLE_ARM64
#include "nnacl/optimized_kernel.h"
#endif

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
  oss << std::endl << this->in_tensors_.size() << " InputTensors:";
  for (auto tensor : in_tensors_) {
    oss << " " << tensor << ":" << tensor->ToString();
  }
  oss << std::endl << this->out_tensors_.size() << " OutputTensors:";
  for (auto tensor : out_tensors_) {
    oss << " " << tensor << ":" << tensor->ToString();
  }
  oss << std::endl << "input kernels :";
  for (auto kernel : this->in_kernels_) {
    oss << " " << kernel->ToString();
  }
  oss << std::endl << "output kernels :";
  for (auto kernel : this->out_kernels_) {
    oss << " " << kernel->ToString();
  }
  oss << std::endl << nodes_.size() << "　nodes :";
  for (auto kernel : this->nodes_) {
    oss << " " << kernel->name();
  }
  return oss.str();
}

int SubGraphKernel::Run() {
  if (this->executor_ == nullptr) {
    MS_LOG(ERROR) << "executor is nullptr";
    return RET_ERROR;
  }
  auto ret = executor_->Prepare(this->nodes_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare failed: " << ret;
    return ret;
  }
  ret = executor_->Run(this->in_tensors_, this->out_tensors_, this->nodes_, this->context_->allocator.get());
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Run sub graph failed: " << ret;
    return ret;
  }
  return RET_OK;
}

int SubGraphKernel::Run(const KernelCallBack &before, const KernelCallBack &after) {
  if (this->executor_ == nullptr) {
    MS_LOG(ERROR) << "executor is nullptr";
    return RET_ERROR;
  }
  auto ret = executor_->Prepare(this->nodes_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare failed: " << ret;
    return ret;
  }
  ret =
    executor_->Run(this->in_tensors_, this->out_tensors_, this->nodes_, this->context_->allocator.get(), before, after);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Run sub graph failed: " << ret;
    return ret;
  }
  return RET_OK;
}

int SubGraphKernel::ReSize() { return ReSize(false); }

int SubGraphKernel::ReSize(bool is_interrupt) {
  for (auto kernel : nodes_) {
    if (kernel == nullptr) {
      MS_LOG(ERROR) << "input kernel is nullptr!";
      return RET_ERROR;
    }
    if (kernel->subgraph_type() != kernel::kNotSubGraph) {
      MS_LOG(ERROR) << "all nodes in should be kernel";
      return RET_ERROR;
    }
    auto primitive = const_cast<mindspore::lite::PrimitiveC *>(kernel->GetPrimitive());
    if (primitive == nullptr) {
      MS_LOG(ERROR) << "kernel(" << kernel->name() << ")'s primitive is nullptr!";
      return RET_ERROR;
    }
    std::vector<lite::Tensor *> inputs = kernel->in_tensors();
    std::vector<lite::Tensor *> outputs = kernel->out_tensors();
    for (auto &output : outputs) {
      output->FreeData();
    }
    primitive->SetInferFlag(!is_interrupt);
    auto ret = primitive->InferShape(inputs, outputs);
    if (ret == RET_INFER_INVALID) {
      MS_LOG(INFO) << "InferShape shouldn't be done before runtime, type:"
                   << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(primitive->Type()))
                   << "flag set to false.";
      primitive->SetInferFlag(false);
      is_interrupt = true;
    } else if (ret != RET_OK) {
      MS_LOG(ERROR) << "InferShape failed, type: "
                    << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(primitive->Type()));
      return RET_INFER_ERR;
    }
    if (!is_interrupt) {
      ret = kernel->ReSize();
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "kernel " << kernel->name() << " resize fail!ret = " << ret;
        return ret;
      }
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
      tensor->set_allocator(this->context_->allocator.get());
    }
  }
  return RET_OK;
}

int CpuFp32SubGraph::PreProcess() { return RET_OK; }

int CpuFp16SubGraph::PreProcess() {
  for (auto kernel : this->nodes_) {
    for (auto tensor : kernel->out_tensors()) {
      if (tensor->data_type() == kNumberTypeFloat32) {
        tensor->set_data_type(kNumberTypeFloat16);
      }
    }
  }
  return RET_OK;
}

int CpuFp16SubGraph::PostProcess() {
  auto fp16_to_fp32_cast_func = kernel::Float16CastUtil::GetInstance()->float16_to_float32_func_;
  if (fp16_to_fp32_cast_func == nullptr) {
    MS_LOG(ERROR) << "Can not find cast fp16 to fp32 func";
    return RET_ERROR;
  }
  for (auto tensor : this->out_tensors_) {
    if (tensor->data_type() == kNumberTypeFloat16) {
      void *float16_data = nullptr;
      if (this->context_ != nullptr && this->context_->allocator != nullptr) {
        float16_data = this->context_->allocator->Malloc(tensor->Size());
      } else {
        float16_data = malloc(tensor->Size());
      }
      if (float16_data == nullptr) {
        MS_LOG(ERROR) << "malloc data failed";
        return RET_ERROR;
      }
      memcpy(float16_data, tensor->data_c(), tensor->Size());
      auto ret = tensor->FreeData();
      if (RET_OK != ret) {
        MS_LOG(ERROR) << "free data failed";
        return RET_ERROR;
      }
      tensor->set_data_type(TypeId::kNumberTypeFloat32);
      ret = tensor->MallocData();
      if (RET_OK != ret) {
        MS_LOG(ERROR) << "malloc data failed";
        return RET_ERROR;
      }
      fp16_to_fp32_cast_func(float16_data, tensor->data_c(), tensor->ElementsNum());
      if (this->context_ != nullptr && this->context_->allocator != nullptr) {
        this->context_->allocator->Free(float16_data);
      } else {
        free(float16_data);
      }
    }
  }
  return RET_OK;
}
}  // namespace mindspore::kernel
