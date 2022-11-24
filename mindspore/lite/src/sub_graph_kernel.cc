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
#include "src/common/prim_inner.h"

namespace mindspore::kernel {
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_INFER_ERR;
using mindspore::lite::RET_INFER_INVALID;
using mindspore::lite::RET_OK;

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

int SubGraphKernel::ConstBatchResetAcc(int orig_batch, int new_batch) {
  if (batch_changed_inputs_.empty()) {
    MS_LOG(ERROR) << "The batch_changed_inputs_ can't be null while restore ";
    return RET_ERROR;
  }
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
    for (auto input : inputs) {
      auto iter = batch_changed_inputs_.find(input);
      if (iter == batch_changed_inputs_.end()) {
        continue;
      }

      if ((kernel->type() == schema::PrimitiveType_Reshape || kernel->type() == schema::PrimitiveType_SliceFusion) &&
          input->Size() > 0) {
        int *data = static_cast<int *>(input->data());
        if (data[0] == orig_batch) {
          std::stringstream err_info;
          err_info << "Const Tensor " << input->tensor_name() << " change value from " << input->ToString();
          data[0] = new_batch;
          err_info << " to " << input->ToString();
          MS_LOG(WARNING) << err_info.str();
        }
        continue;
      }

      if (input->IsConst() && !input->shape().empty() && input->shape()[0] == orig_batch) {
        void *data_orig = iter->second.orig_data;
        size_t size_orig = iter->second.orig_size;
        auto new_shape = input->shape();
        new_shape[0] = new_batch;
        MS_LOG(WARNING) << "Const tensor " << input->tensor_name() << " change shape from " << input->shape() << " to "
                        << new_shape;
        input->set_shape(new_shape);
        size_t size_new = input->Size();
        // need Create new buffer
        if (size_new > iter->second.new_size && size_new > size_orig) {
          if (size_new % size_orig != 0) {
            MS_LOG(ERROR) << "New size error, size_new:" << size_new << " origin size:" << size_orig;
            return RET_ERROR;
          }
          std::shared_ptr<unsigned char> data_new(new unsigned char[size_new], std::default_delete<unsigned char[]>());
          for (size_t i = 0; i < size_new; i += size_orig) {
            memcpy(data_new.get() + i, data_orig, size_orig);
          }
          input->set_data(data_new.get());
          input->set_own_data(false);
          iter->second.new_data = data_new;
          iter->second.new_size = size_new;
        }
      }
    }
  }
  return RET_OK;
}

int SubGraphKernel::ConstBatchReset(int orig_batch, int new_batch) {
  if (!batch_changed_inputs_.empty()) {
    return ConstBatchResetAcc(orig_batch, new_batch);
  }
  std::map<lite::Tensor *, DataInfo> changed_inputs;
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
    for (auto input : inputs) {
      if (input->IsConst() &&
          (kernel->type() == schema::PrimitiveType_Reshape || kernel->type() == schema::PrimitiveType_SliceFusion) &&
          input->Size() > 0) {
        int *data = static_cast<int *>(input->data());
        if (data[0] == orig_batch) {
          std::stringstream err_info;
          err_info << "Const Tensor " << input->tensor_name() << " change value from " << input->ToString();
          data[0] = new_batch;
          err_info << " to " << input->ToString();
          MS_LOG(WARNING) << err_info.str();
          DataInfo dataInfo{input->data(), input->Size(), nullptr, input->Size()};
          changed_inputs.emplace(input, dataInfo);
        }
        continue;
      }

      if (input->IsConst() && !input->shape().empty() && input->shape()[0] == orig_batch) {
        void *data_orig = input->data();
        size_t size_orig = input->Size();
        auto new_shape = input->shape();
        new_shape[0] = new_batch;
        MS_LOG(WARNING) << "Const tensor " << input->tensor_name() << " change shape from " << input->shape() << " to "
                        << new_shape;
        // const tensor never own data, origin data is from schema tensor
        // so no need free origin data
        MS_ASSERT((!input->own_data() || input->allocator() == nullptr));
        input->set_shape(new_shape);
        size_t size_new = input->Size();
        // need Create new buffer
        if (size_new > size_orig) {
          if (size_new % size_orig != 0) {
            MS_LOG(ERROR) << "New size error, size_new:" << size_new << " origin size:" << size_orig;
            return RET_ERROR;
          }
          std::shared_ptr<unsigned char> data_new(new unsigned char[size_new], std::default_delete<unsigned char[]>());
          for (size_t i = 0; i < size_new; i += size_orig) {
            memcpy(data_new.get() + i, data_orig, size_orig);
          }
          input->set_data(data_new.get());
          input->set_own_data(false);
          DataInfo dataInfo{data_orig, size_orig, data_new, size_new};
          changed_inputs.emplace(input, dataInfo);
        } else {
          // reuse origin buffer
          DataInfo dataInfo{data_orig, size_orig, nullptr, size_orig};
          changed_inputs.emplace(input, dataInfo);
        }
      }
    }
  }
  if (batch_changed_inputs_.empty()) {
    std::swap(batch_changed_inputs_, changed_inputs);
  }
  return RET_OK;
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
#ifndef CONTROLFLOW_TENSORLIST_CLIP
      // replace with custom op in the future.
      if (parameter->type_ == static_cast<int>(PrimType::PrimType_Inner_Identity)) {
        ret = kernel->ReSize();
        if (ret != RET_OK) {
          MS_LOG(ERROR) << "kernel " << kernel->name() << " resize fail!ret = " << ret;
          return ret;
        }
        continue;
      }
#endif
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
    int input_init_refcount = input->init_ref_count();
    for (auto *node : nodes_) {
      input_init_refcount += std::count_if(node->in_tensors().begin(), node->in_tensors().end(),
                                           [&input](const lite::Tensor *item) { return item == input; });
    }
    input->set_init_ref_count(input_init_refcount);
  }
}

void SubGraphKernel::InitOutTensorInitRefCount(const std::vector<LiteKernel *> *mask_kernels) {
  for (auto *node : nodes_) {
    node->InitOutTensorInitRefCount(mask_kernels);
  }
  for (auto &output : this->out_tensors()) {
    if (output->init_ref_count() == 0) {  // true only when output is also an input and only exist in control-flow model
      output->set_init_ref_count(1);
    }
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
      if (tensor->allocator() == nullptr) {
        tensor->set_allocator(allocator);
      }
    }
  }

  auto node = nodes_[nodes_.size() - 1];
  for (auto tensor : node->out_tensors()) {
    MS_ASSERT(tensor != nullptr);
    if (tensor->allocator() == nullptr) {
      tensor->set_allocator(context->allocator);
    }
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
      if (tensor->allocator() == nullptr) {
        tensor->set_allocator(this->Context()->allocator);
      }
    }
  }
  for (auto &out : this->out_tensors()) {
    if (out->allocator() == nullptr) {
      out->set_allocator(this->Context()->allocator);
    }
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
