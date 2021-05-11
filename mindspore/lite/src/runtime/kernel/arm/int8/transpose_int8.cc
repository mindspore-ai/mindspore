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
#include "src/runtime/runtime_api.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::lite::RET_OP_EXECUTE_FAILURE;
using mindspore::schema::PrimitiveType_Transpose;

namespace mindspore::kernel {
int TransposeInt8CPUKernel::Init() {
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int TransposeInt8Run(void *cdata, int task_id) {
  auto transpose_int8 = reinterpret_cast<TransposeInt8CPUKernel *>(cdata);
  auto ret = transpose_int8->DoTranspose(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "DoTranspose error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_OP_EXECUTE_FAILURE;
  }
  return RET_OK;
}

void TransposeInt8CPUKernel::FreeTmpBuf() {
  if (!extra_dims_) {
    return;
  }
  if (dim_size_ != nullptr) {
    context_->allocator->Free(dim_size_);
    dim_size_ = nullptr;
  }
  if (position_ != nullptr) {
    context_->allocator->Free(position_);
    position_ = nullptr;
  }
  return;
}

int TransposeInt8CPUKernel::MallocTmpBuf() {
  if (!extra_dims_) {
    return RET_OK;
  }

  int dims = out_tensors_.at(0)->shape().size();

  dim_size_ = reinterpret_cast<int *>(context_->allocator->Malloc(dims * sizeof(int)));
  if (dim_size_ == nullptr) {
    MS_LOG(ERROR) << "Malloc data failed";
    return RET_ERROR;
  }
  *(dim_size_ + dims - 1) = 1;
  for (int i = dims - 1; i > 0; --i) {
    *(dim_size_ + i - 1) = *(dim_size_ + i) * out_shape_[i];
  }
  position_ = reinterpret_cast<int *>(context_->allocator->Malloc(dims * sizeof(int) * op_parameter_->thread_num_));
  if (position_ == nullptr) {
    MS_LOG(ERROR) << "Malloc data failed";
    context_->allocator->Free(dim_size_);
    dim_size_ = nullptr;
    return RET_ERROR;
  }
  return RET_OK;
}

int TransposeInt8CPUKernel::ReSize() {
  auto in_tensor = in_tensors_.front();
  auto out_tensor = out_tensors_.front();
  auto in_shape = in_tensor->shape();
  auto out_shape = out_tensor->shape();

  transpose_param_->data_size_ = in_tensor->Size();

  // get perm data
  auto perm_tensor = in_tensors_.at(1);
  int *perm_data = reinterpret_cast<int *>(perm_tensor->data_c());
  MS_ASSERT(perm_data != nullptr);
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

  extra_dims_ = out_shape.size() > DIMENSION_6D;
  return RET_OK;
}

int TransposeInt8CPUKernel::DoTranspose(int task_id) {
  int dims = out_tensors_.at(0)->shape().size();
  MS_ASSERT(in_ptr_);
  MS_ASSERT(out_ptr_);
  MS_ASSERT(in_shape_);
  MS_ASSERT(out_shape_);
  MS_ASSERT(transpose_param_);
  TransposeDimsInt8(in_ptr_, out_ptr_, out_shape_, dim_size_, position_ + dims * task_id, transpose_param_, task_id,
                    op_parameter_->thread_num_);
  return RET_OK;
}

void TransposeInt8CPUKernel::GetNHNCTransposeFunc(lite::Tensor *in_tensor, lite::Tensor *out_tensor,
                                                  TransposeParameter *param) {
  auto out_shape = out_tensor->shape();
  if (in_tensor->shape().size() == 4 && param->perm_[0] == 0 && param->perm_[1] == 2 && param->perm_[2] == 3 &&
      param->perm_[3] == 1) {
    nhnc_param_[0] = out_shape[0];
    nhnc_param_[1] = out_shape[1] * out_shape[2];
    nhnc_param_[2] = out_shape[3];
    NHNCTransposeFunc_ = PackNCHWToNHWCInt8;
  }
  if (in_tensor->shape().size() == 4 && param->perm_[0] == 0 && param->perm_[1] == 3 && param->perm_[2] == 1 &&
      param->perm_[3] == 2) {
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

  in_ptr_ = reinterpret_cast<int8_t *>(in_tensor->data_c());
  out_ptr_ = reinterpret_cast<int8_t *>(out_tensor->data_c());
  GetNHNCTransposeFunc(in_tensor, out_tensor, transpose_param_);
  if (NHNCTransposeFunc_ != nullptr) {
    NHNCTransposeFunc_(in_ptr_, out_ptr_, nhnc_param_[0], nhnc_param_[1], nhnc_param_[2]);
    return RET_OK;
  }
  memcpy(in_shape_, in_dims.data(), in_dims.size() * sizeof(int));
  memcpy(out_shape_, out_dims.data(), out_dims.size() * sizeof(int));

  int ret = MallocTmpBuf();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "MallocTmpBuf error_code[" << ret << "]";
  }
  if (extra_dims_) {
    ret = ParallelLaunch(static_cast<const lite::InnerContext *>(this->context_)->thread_pool_, TransposeInt8Run, this,
                         op_parameter_->thread_num_);
  } else {
    ret = DoTransposeInt8(in_ptr_, out_ptr_, out_shape_, transpose_param_);
  }
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Tranpose error error_code[" << ret << "]";
  }

  FreeTmpBuf();
  return ret;
}

REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_Transpose, LiteKernelCreator<TransposeInt8CPUKernel>)
}  // namespace mindspore::kernel
