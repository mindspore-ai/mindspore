/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "nnacl/nnacl_kernel.h"
#include "src/tensor.h"
#include "include/errorcode.h"
#include "src/litert/thread_cost_model.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
namespace mindspore::nnacl {
NNACLKernel::~NNACLKernel() {
  if (in_ != nullptr) {
    free(in_);
    in_ = nullptr;
  }
  if (out_ != nullptr) {
    free(out_);
    out_ = nullptr;
  }

  if (kernel_ != nullptr) {
    kernel_->release(kernel_);

    free(kernel_);
    kernel_ = nullptr;
  }
}

int NNACLKernel::Prepare() {
  if (kernel_ == nullptr) {
    return RET_ERROR;
  }

  int ret = kernel_->prepare(kernel_);
  if (ret != RET_OK) {
    return ret;
  }

  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int NNACLKernel::ReSize() {
  if (kernel_ == nullptr) {
    return RET_ERROR;
  }
  UpdateTensorC();
  return kernel_->resize(kernel_);
}

int NNACLKernel::Run() {
  if (kernel_ == nullptr) {
    return RET_ERROR;
  }
  UpdateTensorC();
  kernel_->workspace_ = workspace();
  return kernel_->compute(kernel_);
}

int NNACLKernel::InferShape() {
  if (kernel_ == nullptr) {
    return RET_ERROR;
  }
  return kernel_->infershape(kernel_);
}

void NNACLKernel::UpdateTensorC() {
  for (size_t i = 0; i < in_size_; i++) {
    in_[i] = in_tensors().at(i)->ConvertToTensorC();
  }
  for (size_t i = 0; i < out_size_; i++) {
    out_[i] = out_tensors().at(i)->ConvertToTensorC();
  }
}

int NNACLKernel::OptimizeDataCopy() {
  auto input_tensor = in_tensors().front();
  CHECK_NULL_RETURN(input_tensor);
  CHECK_NULL_RETURN(input_tensor->data());
  auto output_tensor = out_tensors().front();
  CHECK_NULL_RETURN(output_tensor);
  CHECK_NULL_RETURN(output_tensor->data());

  if (input_tensor->allocator() == nullptr || input_tensor->allocator() != output_tensor->allocator() ||
      input_tensor->allocator() != ms_context_->allocator || /* runtime allocator */
      op_parameter_->is_train_session_) {
    return NNACLKernel::Run();
  }

  output_tensor->FreeData();
  output_tensor->ResetRefCount();
  output_tensor->set_data(input_tensor->data());
  if (input_tensor->IsConst()) {
    output_tensor->set_own_data(false);
  } else {
    output_tensor->set_own_data(input_tensor->own_data());
  }
  return RET_OK;
}

int NNACLKernel::InitKernel(const TypeId &data_type, const lite::InnerContext *ctx) {
  CHECK_NULL_RETURN(ctx);

  in_size_ = in_tensors_.size();
  if (in_size_ == 0 || in_size_ > MAX_MALLOC_SIZE) {
    return RET_ERROR;
  }
  in_ = reinterpret_cast<TensorC **>(malloc(in_size_ * sizeof(TensorC *)));
  if (in_ == nullptr) {
    return RET_ERROR;
  }

  out_size_ = out_tensors_.size();
  if (out_size_ == 0 || out_size_ > MAX_MALLOC_SIZE) {
    return RET_ERROR;
  }
  out_ = reinterpret_cast<TensorC **>(malloc(out_size_ * sizeof(TensorC *)));
  if (out_ == nullptr) {
    return RET_ERROR;
  }

  UpdateTensorC();
  kernel_ =
    CreateKernel(op_parameter_, in_, in_size_, out_, out_size_, data_type, const_cast<ExecEnv *>(ctx->GetExecEnv()));
  if (kernel_ == nullptr) {
    return RET_ERROR;
  }
  kernel_->update_thread_ = lite::UpdateThreadNum;
  return RET_OK;
}
}  // namespace mindspore::nnacl
