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

#include "src/lite_kernel.h"
#include <algorithm>
#include <set>
#include "src/tensor.h"
#include "src/common/utils.h"
#include "src/runtime/infer_manager.h"
#include "src/common/version_manager.h"

namespace mindspore::kernel {
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
#ifdef SUPPORT_TRAIN
void *LiteKernel::workspace_ = nullptr;

void LiteKernel::AllocWorkspace(size_t size) {
  if (size == 0) {
    return;
  }
  workspace_ = malloc(size);
  if (workspace_ == nullptr) {
    MS_LOG(ERROR) << "fail to alloc " << size;
  }
}

void LiteKernel::FreeWorkspace() {
  free(workspace_);
  workspace_ = nullptr;
}

int LiteKernel::DecOutTensorRefCount() {
  for (auto *tensor : this->out_tensors_) {
    tensor->set_ref_count(tensor->ref_count() - 1);
    if (0 >= tensor->ref_count()) {
      tensor->FreeData();
    }
  }
  return 0;
}
#endif
bool LiteKernel::IsReady(const std::vector<lite::Tensor *> &scope_tensors) {
  return std::all_of(this->in_tensors().begin(), this->in_tensors().end(), [&](lite::Tensor *in_tensor) {
    if (IsContain(scope_tensors, in_tensor)) {
      return in_tensor->IsReady();
    } else {
      return true;
    }
  });
}

void LiteKernel::InitOutTensorInitRefCount() {
  for (auto *tensor : this->out_tensors_) {
    size_t init_ref_count = 0;
    for (auto *post_kernel : this->out_kernels_) {
      init_ref_count +=
        std::count_if(post_kernel->in_tensors_.begin(), post_kernel->in_tensors_.end(),
                      [&tensor](const lite::Tensor *post_kernel_in_tensor) { return post_kernel_in_tensor == tensor; });
    }
    tensor->set_init_ref_count(init_ref_count);
  }
}

int LiteKernel::FreeInWorkTensor() const {
  for (auto &in_tensor : this->in_tensors_) {
    MS_ASSERT(in_tensor != nullptr);
    if (in_tensor->root_tensor() == in_tensor) {
      continue;
    }
    in_tensor->DecRefCount();
  }
  return RET_OK;
}

int LiteKernel::PreProcess() {
  if (!InferShapeDone()) {
    op_parameter_->infer_flag_ = true;
    auto ret = lite::KernelInferShape(in_tensors_, &out_tensors_, op_parameter_);
    if (ret != 0) {
      op_parameter_->infer_flag_ = false;
      MS_LOG(ERROR) << "InferShape fail!";
      return ret;
    }
    ret = ReSize();
    if (ret != 0) {
      MS_LOG(ERROR) << "ReSize fail!ret: " << ret;
      return ret;
    }
  }

  for (auto *output : this->out_tensors()) {
    MS_ASSERT(output != nullptr);
    if (desc_.data_type == kNumberTypeFloat16 && output->data_type() == kNumberTypeFloat32) {
      output->set_data_type(kNumberTypeFloat16);
    }
    if (output->ElementsNum() >= MAX_MALLOC_SIZE / static_cast<int>(sizeof(int64_t))) {
      MS_LOG(ERROR) << "The size of output tensor is too big";
      return RET_ERROR;
    }
    auto ret = output->MallocData();
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "MallocData failed";
      return ret;
    }
  }
  return RET_OK;
}

int LiteKernel::PostProcess() {
#ifdef SUPPORT_TRAIN
  for (auto input_kernel : this->in_kernels()) {
    MS_ASSERT(input_kernel != nullptr);
    if (input_kernel->is_model_output()) {
      continue;
    }
    auto ret = input_kernel->DecOutTensorRefCount();
    if (0 != ret) {
      MS_LOG(WARNING) << "DecOutTensorRefCount for kernel" << this->name() << " failed";
    }
  }
  return RET_OK;
#else
  for (auto *output : this->out_tensors()) {
    MS_ASSERT(output != nullptr);
    output->ResetRefCount();
  }
  return FreeInWorkTensor();
#endif
}

int LiteKernel::Run(const KernelCallBack &before, const KernelCallBack &after) {
  if (before != nullptr) {
    if (!before(TensorVectorCast(this->in_tensors_), TensorVectorCast(this->out_tensors_),
                {this->name_, this->type_str()})) {
      MS_LOG(WARNING) << "run kernel before_callback failed, name: " << this->name_;
    }
  }
  // Support ZeroShape
  size_t zero_shape_num = 0;
  for (auto tensor : this->out_tensors_) {
    for (size_t i = 0; i < tensor->shape().size(); i++) {
      if (tensor->shape()[i] == 0) {
        zero_shape_num++;
        break;
      }
    }
  }
  if (zero_shape_num != this->out_tensors_.size()) {
    auto ret = Run();
    if (RET_OK != ret) {
      MS_LOG(ERROR) << "run kernel failed, name: " << this->name_;
      return ret;
    }
  }
  if (after != nullptr) {
    if (!after(TensorVectorCast(this->in_tensors_), TensorVectorCast(this->out_tensors_),
               {this->name_, this->type_str()})) {
      MS_LOG(WARNING) << "run kernel after_callback failed, name: " << this->name_;
    }
  }
  return RET_OK;
}

std::string LiteKernel::ToString() const {
  std::ostringstream oss;
  oss << "LiteKernel: " << this->name_;
  oss << ", Type: " << this->type_str();
  oss << ", " << this->in_tensors_.size() << " InputTensors:";
  for (auto tensor : in_tensors_) {
    oss << " " << tensor;
  }
  oss << ", " << this->out_tensors_.size() << " OutputTensors:";
  for (auto tensor : out_tensors_) {
    oss << " " << tensor;
  }
  oss << ", " << this->in_kernels_.size() << " InputKernels:";
  for (auto in_kernel : in_kernels_) {
    oss << " " << in_kernel->name_;
  }
  oss << ", " << this->out_kernels_.size() << " OutputKernels:";
  for (auto out_kernel : out_kernels_) {
    oss << " " << out_kernel->name_;
  }
  return oss.str();
}

void LiteKernel::FindInoutKernels(const std::vector<kernel::LiteKernel *> &scope_kernels) {
  // clean io kernels
  this->in_kernels_.clear();
  this->out_kernels_.clear();
  // find io kernels
  for (auto *scope_kernel : scope_kernels) {
    if (scope_kernel == this) {
      continue;
    }
    for (auto *tensor : this->in_tensors_) {
      if (lite::IsContain(scope_kernel->out_tensors(), tensor)) {
        if (!lite::IsContain(this->in_kernels(), scope_kernel)) {
          this->AddInKernel(scope_kernel);
        }
      }
    }
    for (auto *tensor : this->out_tensors_) {
      if (lite::IsContain(scope_kernel->in_tensors(), tensor)) {
        if (!lite::IsContain(this->out_kernels(), scope_kernel)) {
          this->AddOutKernel(scope_kernel);
        }
      }
    }
  }
}
}  // namespace mindspore::kernel
