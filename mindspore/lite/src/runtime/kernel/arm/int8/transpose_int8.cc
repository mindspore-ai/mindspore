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

#include "src/runtime/kernel/arm/int8/transpose_int8.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::lite::RET_OP_EXECUTE_FAILURE;
using mindspore::schema::PrimitiveType_Transpose;

namespace mindspore::kernel {
namespace {
constexpr size_t kMaxShapeSize = 20;
}  // namespace
int TransposeInt8CPUKernel::Init() {
  CHECK_LESS_RETURN(in_tensors_.size(), C2NUM);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int TransposeInt8Run(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  auto transpose_int8 = reinterpret_cast<TransposeInt8CPUKernel *>(cdata);
  auto ret = transpose_int8->DoTranspose(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "DoTranspose error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_OP_EXECUTE_FAILURE;
  }
  return RET_OK;
}

int TransposeInt8CPUKernel::ReSize() {
  auto in_tensor = in_tensors_.front();
  auto out_tensor = out_tensors_.front();
  auto in_shape = in_tensor->shape();
  auto out_shape = out_tensor->shape();

  transpose_param_->data_num_ = in_tensor->ElementsNum();

  // get perm data
  auto perm_tensor = in_tensors_.at(1);
  int *perm_data = reinterpret_cast<int *>(perm_tensor->data());
  CHECK_NULL_RETURN(perm_data);
  transpose_param_->num_axes_ = perm_tensor->ElementsNum();
  for (int i = 0; i < transpose_param_->num_axes_; ++i) {
    transpose_param_->perm_[i] = perm_data[i];
  }

  transpose_param_->strides_[transpose_param_->num_axes_ - 1] = 1;
  transpose_param_->out_strides_[transpose_param_->num_axes_ - 1] = 1;
  for (int i = transpose_param_->num_axes_ - 2; i >= 0; i--) {
    transpose_param_->strides_[i] = in_shape.at(i + 1) * transpose_param_->strides_[i + 1];
    transpose_param_->out_strides_[i] = out_shape.at(i + 1) * transpose_param_->out_strides_[i + 1];
  }
  return RET_OK;
}

int TransposeInt8CPUKernel::DoTranspose(int task_id) {
  CHECK_NULL_RETURN(in_ptr_);
  CHECK_NULL_RETURN(out_ptr_);
  CHECK_NULL_RETURN(in_shape_);
  CHECK_NULL_RETURN(out_shape_);
  CHECK_NULL_RETURN(transpose_param_);
  TransposeDimsInt8(in_ptr_, out_ptr_, out_shape_, transpose_param_, task_id, op_parameter_->thread_num_);
  return RET_OK;
}

void TransposeInt8CPUKernel::GetNHNCTransposeFunc(const lite::Tensor *in_tensor, const lite::Tensor *out_tensor,
                                                  const TransposeParameter *param) {
  auto out_shape = out_tensor->shape();
  if (in_tensor->shape().size() == DIMENSION_4D && param->perm_[0] == 0 && param->perm_[1] == 2 &&
      param->perm_[2] == 3 && param->perm_[3] == 1) {
    nhnc_param_[0] = out_shape[0];
    nhnc_param_[1] = out_shape[1] * out_shape[2];
    nhnc_param_[2] = out_shape[3];
    NHNCTransposeFunc_ = PackNCHWToNHWCInt8;
  }
  if (in_tensor->shape().size() == DIMENSION_4D && param->perm_[0] == 0 && param->perm_[1] == 3 &&
      param->perm_[2] == 1 && param->perm_[3] == 2) {
    nhnc_param_[0] = out_shape[0];
    nhnc_param_[1] = out_shape[2] * out_shape[3];
    nhnc_param_[2] = out_shape[1];
    NHNCTransposeFunc_ = PackNHWCToNCHWInt8;
  }
}

int TransposeInt8CPUKernel::Run() {
  auto in_tensor = in_tensors_.front();
  auto out_tensor = out_tensors_.front();

  auto in_dims = in_tensor->shape();
  auto out_dims = out_tensor->shape();

  in_ptr_ = reinterpret_cast<int8_t *>(in_tensor->data());
  CHECK_NULL_RETURN(in_ptr_);
  out_ptr_ = reinterpret_cast<int8_t *>(out_tensor->data());
  CHECK_NULL_RETURN(out_ptr_);
  GetNHNCTransposeFunc(in_tensor, out_tensor, transpose_param_);
  if (NHNCTransposeFunc_ != nullptr) {
    NHNCTransposeFunc_(in_ptr_, out_ptr_, nhnc_param_[0], nhnc_param_[1], nhnc_param_[2]);
    return RET_OK;
  }
  if (in_dims.size() > kMaxShapeSize) {
    MS_LOG(ERROR) << "in_dims size > " << kMaxShapeSize << " cannot copy data.";
    return RET_ERROR;
  }
  memcpy(in_shape_, in_dims.data(), in_dims.size() * sizeof(int));
  if (out_dims.size() > kMaxShapeSize) {
    MS_LOG(ERROR) << "out_dims size > " << kMaxShapeSize << " cannot copy data.";
    return RET_ERROR;
  }
  memcpy(out_shape_, out_dims.data(), out_dims.size() * sizeof(int));

  if (out_tensor->shape().size() > DIMENSION_6D) {
    return ParallelLaunch(this->ms_context_, TransposeInt8Run, this, op_parameter_->thread_num_);
  } else {
    return DoTransposeInt8(in_ptr_, out_ptr_, out_shape_, transpose_param_);
  }
}

REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_Transpose, LiteKernelCreator<TransposeInt8CPUKernel>)
REG_KERNEL(kCPU, kNumberTypeBool, PrimitiveType_Transpose, LiteKernelCreator<TransposeInt8CPUKernel>)
}  // namespace mindspore::kernel
