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
#if defined(ENABLE_ARM64) && defined(ENABLE_FP16)
#include "src/runtime/kernel/arm/fp16/fp16_op_handler.h"
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
  oss << std::endl << this->in_tensors_.size() << "Subgraph inputTensors:";
  for (auto tensor : in_tensors_) {
    oss << " " << tensor;
  }
  oss << std::endl << this->out_tensors_.size() << "Subgraph outputTensors:";
  for (auto tensor : out_tensors_) {
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

int SubGraphKernel::Run() {
  if (this->executor_ == nullptr) {
    MS_LOG(ERROR) << "executor is nullptr";
    return RET_ERROR;
  }
  auto ret = executor_->Run(this->in_tensors_, this->out_tensors_, this->nodes_, this->context_->allocator.get());
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
  auto ret =
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
    primitive->set_infer_flag(!is_interrupt);
    auto ret = primitive->InferShape(inputs, outputs);
    if (ret == RET_INFER_INVALID) {
      MS_LOG(INFO) << "InferShape shouldn't be done before runtime, type:"
                   << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(primitive->Type()))
                   << "flag set to false.";
      primitive->set_infer_flag(false);
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
  if (is_interrupt) {
    MS_LOG(INFO) << "Infer shape failed.";
    return RET_INFER_INVALID;
  }
  return RET_OK;
}

void SubGraphKernel::InitOutTensorInitRefCount() {
  for (auto *node : nodes_) {
    node->InitOutTensorInitRefCount();
  }
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
  this->executor_ = new (std::nothrow) mindspore::lite::CpuExecutor;
  if (this->executor_ == nullptr) {
    MS_LOG(ERROR) << "new CpuExecutor failed";
    return RET_ERROR;
  }
  ret = this->executor_->Prepare(this->nodes_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare CpuExecutor failed";
    return ret;
  }
  return RET_OK;
}

#ifdef ENABLE_FP16
void CpuFp16SubGraph::FreeOriginInputData() {
  for (auto *data_store : this->origin_input_data_) {
    if (data_store == nullptr) {
      continue;
    }
    // free data in data_store
    if (data_store->data_ != nullptr) {
      if (data_store->allocator_ == nullptr) {
        free(data_store->data_);
      } else {
        data_store->allocator_->Free(data_store->data_);
      }
    }
    // free data_store
    if (this->context_->allocator != nullptr) {
      this->context_->allocator->Free(data_store);
    } else {
      free(data_store);
    }
    data_store = nullptr;
  }
  this->origin_input_data_.clear();
}

int CpuFp16SubGraph::PreProcess() {
#ifdef ENABLE_ARM64
  if (!mindspore::lite::IsSupportFloat16()) {
    MS_LOG(ERROR) << "Unsupport fp16 in this devices";
    return RET_ERROR;
  }
  MS_ASSERT(origin_input_data_.empty());
  for (auto tensor : this->in_tensors_) {
    MS_ASSERT(tensor != nullptr);
    if (tensor->data_type() == kNumberTypeFloat32) {
      auto float32_data = tensor->data_c();
      MS_ASSERT(float32_data != nullptr);
      tensor->set_data(nullptr);
      tensor->set_data_type(TypeId::kNumberTypeFloat16);
      auto ret = tensor->MallocData();
      if (RET_OK != ret) {
        MS_LOG(ERROR) << "malloc data failed";
        this->FreeOriginInputData();
        return RET_ERROR;
      }
      MS_ASSERT(tensor->data_c() != nullptr);
      Float32ToFloat16_fp16_handler(float32_data, tensor->data_c(), tensor->ElementsNum());
      auto *data_store = DataStore::CreateDataStore(float32_data, tensor->allocator(), this->context_->allocator.get());
      if (data_store == nullptr) {
        MS_LOG(ERROR) << "Create DataStore failed";
        this->FreeOriginInputData();
        return RET_ERROR;
      }
      origin_input_data_.emplace_back(data_store);
    } else {
      origin_input_data_.emplace_back(nullptr);
    }
  }
  for (auto kernel : this->nodes_) {
    for (auto tensor : kernel->out_tensors()) {
      if (kernel->Type() == schema::PrimitiveType_Cast) {
        continue;
      }
      if (tensor->data_type() == kNumberTypeFloat32) {
        tensor->set_data_type(kNumberTypeFloat16);
      }
    }
  }
  return RET_OK;
#else
  return RET_OK;
#endif
}

int CpuFp16SubGraph::PostProcess() {
#ifdef ENABLE_ARM64
  if (!mindspore::lite::IsSupportFloat16()) {
    MS_LOG(ERROR) << "Unsupport fp16 in this devices";
    return RET_ERROR;
  }
  for (auto tensor : this->out_tensors_) {
    MS_ASSERT(tensor != nullptr);
    if (tensor->data_type() == kNumberTypeFloat16) {
      auto float16_data = tensor->data_c();
      MS_ASSERT(float16_data != nullptr);
      tensor->set_data(nullptr);
      tensor->set_data_type(TypeId::kNumberTypeFloat32);
      auto ret = tensor->MallocData();
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "malloc data failed";
        if (this->context_ != nullptr && this->context_->allocator != nullptr) {
          this->context_->allocator->Free(float16_data);
        } else {
          free(float16_data);
        }
        return RET_ERROR;
      }
      MS_ASSERT(tensor->data_c() != nullptr);
      Float16ToFloat32_fp16_handler(float16_data, tensor->data_c(), tensor->ElementsNum());
      if (tensor->allocator() != nullptr) {
        tensor->allocator()->Free(float16_data);
      } else {
        free(float16_data);
      }
    }
  }
  MS_ASSERT(this->origin_input_data_.size() == this->in_tensors_.size());
  for (size_t i = 0; i < this->in_tensors_.size(); i++) {
    auto tensor = in_tensors_.at(i);
    MS_ASSERT(tensor != nullptr);
    auto origin_tensor_data = origin_input_data_.at(i);
    if (tensor->data_type() == kNumberTypeFloat16 && origin_tensor_data != nullptr) {
      MS_ASSERT(tensor != nullptr);
      tensor->FreeData();
      MS_ASSERT(origin_tensor_data->data_ != nullptr);
      tensor->set_data(origin_tensor_data->data_);
      tensor->set_data_type(kNumberTypeFloat32);
      origin_tensor_data->data_ = nullptr;
    }
  }
  this->FreeOriginInputData();
  return RET_OK;
#else
  return RET_OK;
#endif
}
#endif
}  // namespace mindspore::kernel
