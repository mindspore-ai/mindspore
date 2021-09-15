/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include "nnacl/pack.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;
using mindspore::lite::RET_OP_EXECUTE_FAILURE;
using mindspore::schema::PrimitiveType_Transpose;

namespace mindspore::kernel {
int TransposeCPUKernel::Init() {
  CHECK_LESS_RETURN(in_tensors_.size(), C2NUM);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int TransposeCPUKernel::ReSize() {
  if (in_tensors_.size() == 2) {
    param_->num_axes_ = in_tensors_.at(1)->ElementsNum();
  }
  int trans3d[3] = {0, 2, 1};
  int *perm_data = nullptr;
  auto input_tensor = in_tensors_.at(kInputIndex);
  if (input_tensor->shape().size() != static_cast<size_t>(param_->num_axes_)) {
    if (input_tensor->shape().size() == 3 && param_->num_axes_ == 4) {
      param_->num_axes_ = 3;
      perm_data = trans3d;
    } else {
      return RET_OK;
    }
  } else {
    MS_ASSERT(in_tensors_.size() == 2);
    auto perm_tensor = in_tensors_.at(1);
    perm_data = reinterpret_cast<int *>(perm_tensor->data());
    MSLITE_CHECK_PTR(perm_data);
  }
  if (param_->num_axes_ > MAX_TRANSPOSE_DIM_SIZE || param_->num_axes_ < 0) {
    MS_LOG(ERROR) << "num_axes_ " << param_->num_axes_ << "is invalid.";
    return RET_ERROR;
  }
  for (int i = 0; i < param_->num_axes_; ++i) {
    param_->perm_[i] = perm_data[i];
  }
  auto &inTensor = in_tensors_.front();
  auto &outTensor = out_tensors_.front();
  auto in_shape = inTensor->shape();
  auto out_shape = outTensor->shape();
  param_->strides_[param_->num_axes_ - 1] = 1;
  param_->out_strides_[param_->num_axes_ - 1] = 1;
  param_->data_num_ = inTensor->ElementsNum();
  MS_CHECK_LE(static_cast<size_t>(param_->num_axes_), in_shape.size(), RET_ERROR);
  MS_CHECK_LE(static_cast<size_t>(param_->num_axes_), out_shape.size(), RET_ERROR);
  for (int i = param_->num_axes_ - 2; i >= 0; i--) {
    param_->strides_[i] = in_shape.at(i + 1) * param_->strides_[i + 1];
    param_->out_strides_[i] = out_shape.at(i + 1) * param_->out_strides_[i + 1];
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

void TransposeCPUKernel::GetNchwToNhwcFunc(TypeId dtype) {
  if (dtype == kNumberTypeFloat32) {
    NHNCTransposeFunc_ = PackNCHWToNHWCFp32;
  }
}

void TransposeCPUKernel::GetNhwcToNchwFunc(TypeId dtype) {
  if (dtype == kNumberTypeFloat32) {
    NHNCTransposeFunc_ = PackNHWCToNCHWFp32;
  }
}

int TransposeCPUKernel::TransposeDim2to6() {
  return DoTransposeFp32(static_cast<const float *>(in_data_), static_cast<float *>(out_data_), out_shape_, param_);
}

int TransposeCPUKernel::TransposeDimGreaterThan6(int task_id) {
  TransposeDimsFp32(static_cast<const float *>(in_data_), static_cast<float *>(out_data_), out_shape_, param_, task_id,
                    op_parameter_->thread_num_);
  return RET_OK;
}

int TransposeCPUKernel::GetNHNCTransposeFunc(const lite::Tensor *in_tensor, const lite::Tensor *out_tensor) {
  if (in_tensor->shape().size() != 4) {
    return RET_OK;
  }
  auto out_shape = out_tensor->shape();
  if (param_->perm_[0] == 0 && param_->perm_[1] == 2 && param_->perm_[2] == 3 && param_->perm_[3] == 1) {
    nhnc_param_[0] = out_shape[0];
    MS_CHECK_FALSE(INT_MUL_OVERFLOW(out_shape[1], out_shape[2]), RET_ERROR);
    nhnc_param_[1] = out_shape[1] * out_shape[2];
    nhnc_param_[2] = out_shape[3];
    GetNchwToNhwcFunc(in_tensor->data_type());
  }
  if (param_->perm_[0] == 0 && param_->perm_[1] == 3 && param_->perm_[2] == 1 && param_->perm_[3] == 2) {
    nhnc_param_[0] = out_shape[0];
    MS_CHECK_FALSE(INT_MUL_OVERFLOW(out_shape[2], out_shape[3]), RET_ERROR);
    nhnc_param_[1] = out_shape[2] * out_shape[3];
    nhnc_param_[2] = out_shape[1];
    GetNhwcToNchwFunc(in_tensor->data_type());
  }
  return RET_OK;
}

int TransposeCPUKernel::RunImpl(int task_id) {
  if (NHNCTransposeFunc_ != nullptr) {
    NHNCTransposeFunc_(in_data_, out_data_, nhnc_param_[0], nhnc_param_[1], nhnc_param_[2], task_id,
                       op_parameter_->thread_num_);
  } else {
    return TransposeDimGreaterThan6(task_id);
  }
  return RET_OK;
}

int TransposeImpl(void *kernel, int task_id, float lhs_scale, float rhs_scale) {
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
  in_data_ = in_tensor->data();
  out_data_ = out_tensor->data();
  CHECK_NULL_RETURN(in_data_);
  CHECK_NULL_RETURN(out_data_);

  if (in_tensor->shape().size() != static_cast<size_t>(param_->num_axes_)) {
    memcpy(out_data_, in_data_, in_tensor->Size());
    return RET_OK;
  }
  if (GetNHNCTransposeFunc(in_tensor, out_tensor) != RET_OK) {
    MS_LOG(ERROR) << "Get NHWC tranpose func fail!";
    return RET_ERROR;
  }
  if (NHNCTransposeFunc_ != nullptr) {
    return ParallelLaunch(this->ms_context_, TransposeImpl, this, op_parameter_->thread_num_);
  }
  if (out_tensor->shape().size() <= DIMENSION_6D) {
    return TransposeDim2to6();
  } else {
    return ParallelLaunch(this->ms_context_, TransposeImpl, this, op_parameter_->thread_num_);
  }
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Transpose, LiteKernelCreator<TransposeCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_Transpose, LiteKernelCreator<TransposeCPUKernel>)
}  // namespace mindspore::kernel
