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
#include "nnacl/cxx_utils.h"
#include "src/tensor.h"
#include "include/errorcode.h"
#include "nnacl/errorcode.h"

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
    kernel_->Release(kernel_);

    free(kernel_);
    kernel_ = nullptr;
  }
}

int NNACLKernel::Prepare() {
  if (kernel_ == nullptr) {
    return RET_ERROR;
  }

  int ret = kernel_->Prepare(kernel_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "NNACL prepare failed. Kernel: " << name() << ", ret: " << ret;
    MS_LOG(ERROR) << NNACLErrorMsg(ret);
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

  int ret = kernel_->Resize(kernel_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "NNACL resize failed. Kernel: " << name() << ", ret: " << ret;
    MS_LOG(ERROR) << NNACLErrorMsg(ret);
    return ret;
  }
  return RET_OK;
}

int NNACLKernel::Run() {
  if (kernel_ == nullptr) {
    return RET_ERROR;
  }
  UpdateTensorC();
  kernel_->workspace_ = workspace();

  int ret = kernel_->Compute(kernel_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "NNACL compute failed. Kernel: " << name() << ", ret: " << ret;
    MS_LOG(ERROR) << NNACLErrorMsg(ret);
    return ret;
  }
  return RET_OK;
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

int NNACLKernel::NNACLCheckArgs() {
  if (op_parameter_ == nullptr) {
    MS_LOG(ERROR) << "NNACL check failed. Invalid parameter.";
    return RET_ERROR;
  }

  if (in_size_ == 0 || in_size_ * sizeof(TensorC *) > MAX_MALLOC_SIZE) {
    MS_LOG(ERROR) << "NNACL check failed. Invalid input size: " << in_size_;
    return RET_ERROR;
  }

  if (out_size_ == 0 || out_size_ * sizeof(TensorC *) > MAX_MALLOC_SIZE) {
    MS_LOG(ERROR) << "NNACL check failed. Invalid output size: " << out_size_;
    return RET_ERROR;
  }

  if (op_parameter_->thread_num_ <= 0 || op_parameter_->thread_num_ > MAX_THREAD_NUM) {
    MS_LOG(ERROR) << "NNACL check failed. Invalid thread number: " << op_parameter_->thread_num_;
    return RET_ERROR;
  }

  return RET_OK;
}

int NNACLKernel::InitKernel(const TypeId &data_type, const lite::InnerContext *ctx) {
  CHECK_NULL_RETURN(ctx);

  in_size_ = in_tensors_.size();
  out_size_ = out_tensors_.size();

  int ret = NNACLCheckArgs();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "NNACL check args failed. Kernel: " << name();
    return ret;
  }

  in_ = reinterpret_cast<TensorC **>(malloc(in_size_ * sizeof(TensorC *)));
  if (in_ == nullptr) {
    return RET_ERROR;
  }

  out_ = reinterpret_cast<TensorC **>(malloc(out_size_ * sizeof(TensorC *)));
  if (out_ == nullptr) {
    return RET_ERROR;
  }

  UpdateTensorC();
  kernel_ = CreateKernel(op_parameter_, in_, in_size_, out_, out_size_, data_type, exec_env_);
  if (kernel_ == nullptr) {
    MS_LOG(WARNING) << "NNACL create kernel failed. Kernel: " << name();
    return RET_ERROR;
  }
  kernel_->UpdateThread = DefaultUpdateThreadNumPass;
  return RET_OK;
}
}  // namespace mindspore::nnacl
