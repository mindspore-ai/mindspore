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

#include "src/runtime/kernel/arm/fp32/transpose_fp32.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "src/runtime/runtime_api.h"
#include "nnacl/pack.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;
using mindspore::lite::RET_OP_EXECUTE_FAILURE;
using mindspore::schema::PrimitiveType_Transpose;

namespace mindspore::kernel {
int TransposeCPUKernel::Init() {
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int TransposeCPUKernel::ReSize() {
  TransposeParameter *param = reinterpret_cast<TransposeParameter *>(op_parameter_);
  if (in_tensors_.size() == 2) {
    param->num_axes_ = in_tensors_.at(1)->ElementsNum();
  }
  if (in_tensors_.at(kInputIndex)->shape().size() != static_cast<size_t>(param->num_axes_)) {
    return RET_OK;
  }
  // get perm data
  MS_ASSERT(in_tensors_.size() == 2);
  auto perm_tensor = in_tensors_.at(1);
  int *perm_data = reinterpret_cast<int *>(perm_tensor->data_c());
  MS_ASSERT(perm_data != nullptr);
  for (int i = 0; i < param->num_axes_; ++i) {
    param->perm_[i] = perm_data[i];
  }
  auto &inTensor = in_tensors_.front();
  auto &outTensor = out_tensors_.front();
  auto in_shape = inTensor->shape();
  auto out_shape = outTensor->shape();
  param->strides_[param->num_axes_ - 1] = 1;
  param->out_strides_[param->num_axes_ - 1] = 1;
  param->data_size_ = inTensor->Size();
  for (int i = param->num_axes_ - 2; i >= 0; i--) {
    param->strides_[i] = in_shape.at(i + 1) * param->strides_[i + 1];
    param->out_strides_[i] = out_shape.at(i + 1) * param->out_strides_[i + 1];
  }

  if (this->out_shape_ != nullptr) {
    free(this->out_shape_);
    this->out_shape_ = nullptr;
  }

  out_shape_ = reinterpret_cast<int *>(malloc(out_shape.size() * sizeof(int)));
  if (out_shape_ == nullptr) {
    MS_LOG(ERROR) << "malloc out_shape_ failed.";
    return RET_ERROR;
  }
  memcpy(out_shape_, out_shape.data(), in_shape.size() * sizeof(int));
  return RET_OK;
}

TransposeCPUKernel::~TransposeCPUKernel() {
  if (this->out_shape_ != nullptr) {
    free(this->out_shape_);
  }
}

void TransposeCPUKernel::GetNHNCTransposeFunc(lite::Tensor *in_tensor, lite::Tensor *out_tensor,
                                              TransposeParameter *param) {
  auto out_shape = out_tensor->shape();
  if (in_tensor->shape().size() == 4 && param->perm_[0] == 0 && param->perm_[1] == 2 && param->perm_[2] == 3 &&
      param->perm_[3] == 1) {
    nhnc_param_[0] = out_shape[0];
    nhnc_param_[1] = out_shape[1] * out_shape[2];
    nhnc_param_[2] = out_shape[3];
    if (in_tensor->data_type() == kNumberTypeFloat32) {
      NHNCTransposeFunc_ = PackNCHWToNHWCFp32;
    }
  }
  if (in_tensor->shape().size() == 4 && param->perm_[0] == 0 && param->perm_[1] == 3 && param->perm_[2] == 1 &&
      param->perm_[3] == 2) {
    nhnc_param_[0] = out_shape[0];
    nhnc_param_[1] = out_shape[2] * out_shape[3];
    nhnc_param_[2] = out_shape[1];
    if (in_tensor->data_type() == kNumberTypeFloat32) {
      NHNCTransposeFunc_ = PackNHWCToNCHWFp32;
    }
  }
}

int TransposeCPUKernel::RunImpl(int task_id) {
  if (NHNCTransposeFunc_ != nullptr) {
    NHNCTransposeFunc_(in_data_, out_data_, nhnc_param_[0], nhnc_param_[1], nhnc_param_[2], task_id, thread_count_);
  } else {
    TransposeDimsFp32(in_data_, out_data_, out_shape_, dim_size_, position_ + dims_ * task_id, param_, task_id,
                      thread_count_);
  }
  return RET_OK;
}

int TransposeImpl(void *kernel, int task_id) {
  auto transpose = reinterpret_cast<TransposeCPUKernel *>(kernel);
  auto ret = transpose->RunImpl(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "TransposeImpl Run error task_id[" << task_id << "] error_code[" << ret << "]";
  }
  return ret;
}

int TransposeCPUKernel::Run() {
  MS_ASSERT(in_tensors_.size() == 1 || in_tensors_.size() == 2);
  MS_ASSERT(out_tensors_.size() == 1);
  auto &in_tensor = in_tensors_.front();
  auto &out_tensor = out_tensors_.front();
  if (in_tensor == nullptr || out_tensor == nullptr) {
    MS_LOG(ERROR) << "null pointer dreferencing.";
    return RET_ERROR;
  }
  in_data_ = reinterpret_cast<float *>(in_tensor->MutableData());
  out_data_ = reinterpret_cast<float *>(out_tensor->MutableData());
  MS_ASSERT(in_data_);
  MS_ASSERT(out_data_);

  param_ = reinterpret_cast<TransposeParameter *>(this->op_parameter_);
  if (in_tensor->shape().size() != static_cast<size_t>(param_->num_axes_)) {
    memcpy(out_data_, in_data_, in_tensor->ElementsNum() * sizeof(float));
    return RET_OK;
  }
  if (in_tensors_.size() == 2) {
    auto input_perm = in_tensors_.at(1);
    MS_ASSERT(input_perm != nullptr);
    MS_ASSERT(input_perm->data_c() != nullptr);
    int *perm_data = reinterpret_cast<int *>(input_perm->data_c());
    for (int i = 0; i < input_perm->ElementsNum(); ++i) {
      param_->perm_[i] = perm_data[i];
    }
    for (int i = input_perm->ElementsNum(); i < MAX_SHAPE_SIZE; ++i) {
      param_->perm_[i] = 0;
    }
  }
  thread_count_ = op_parameter_->thread_num_;
  GetNHNCTransposeFunc(in_tensor, out_tensor, param_);
  if (NHNCTransposeFunc_ != nullptr) {
    auto ret = ParallelLaunch(this->context_->thread_pool_, TransposeImpl, this, thread_count_);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "NHNCTransposeFunc_ is error!";
    }
    return ret;
  }

  MS_ASSERT(out_shape_);
  dims_ = out_tensor->shape().size();
  if (dims_ > MAX_TRANSPOSE_DIM_SIZE) {
    dim_size_ = reinterpret_cast<int *>(context_->allocator->Malloc(dims_ * sizeof(int)));
    if (dim_size_ == nullptr) {
      MS_LOG(ERROR) << "Malloc data failed";
      return RET_NULL_PTR;
    }
    *(dim_size_ + dims_ - 1) = 1;
    for (int i = dims_ - 1; i > 0; --i) {
      *(dim_size_ + i - 1) = *(dim_size_ + i) * out_shape_[i];
    }
    position_ = reinterpret_cast<int *>(context_->allocator->Malloc(dims_ * sizeof(int) * thread_count_));
    if (position_ == nullptr) {
      context_->allocator->Free(dim_size_);
      MS_LOG(ERROR) << "Malloc data failed";
      return RET_NULL_PTR;
    }
  }
  int ret;
  if (dims_ > MAX_TRANSPOSE_DIM_SIZE) {
    ret = ParallelLaunch(this->context_->thread_pool_, TransposeImpl, this, thread_count_);
  } else {
    ret = DoTransposeFp32(in_data_, out_data_, out_shape_, param_);
  }
  if (dims_ > MAX_TRANSPOSE_DIM_SIZE) {
    context_->allocator->Free(dim_size_);
    context_->allocator->Free(position_);
    dim_size_ = nullptr;
    position_ = nullptr;
  }
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Transpose run failed";
  }
  return ret;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Transpose, LiteKernelCreator<TransposeCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_Transpose, LiteKernelCreator<TransposeCPUKernel>)
}  // namespace mindspore::kernel
