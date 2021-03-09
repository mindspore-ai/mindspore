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

  int dims = out_tensors_[0]->shape().size();

  dim_size_ = reinterpret_cast<int *>(context_->allocator->Malloc(dims * thread_h_num_ * sizeof(int)));
  if (dim_size_ == nullptr) {
    MS_LOG(ERROR) << "Malloc data failed";
    return RET_ERROR;
  }
  position_ = reinterpret_cast<int *>(context_->allocator->Malloc(dims * thread_h_num_ * sizeof(int)));
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

  extra_dims_ = out_shape.size() > MAX_SHAPE_SIZE;

  num_unit_ = static_cast<int>(in_shape.at(transpose_param_->perm_[kNHWC_H]));
  thread_h_num_ = MSMIN(thread_num_, num_unit_);
  thread_h_stride_ = UP_DIV(num_unit_, thread_h_num_);
  return RET_OK;
}

int TransposeInt8CPUKernel::DoTranspose(int task_id) {
  int num_unit_thread = MSMIN(thread_h_stride_, num_unit_ - task_id * thread_h_stride_);
  if (num_unit_thread <= 0) {
    return RET_OK;
  }
  int thread_offset = task_id * thread_h_stride_;

  int *dim_size = nullptr;
  int *position = nullptr;
  if (extra_dims_) {
    MS_ASSERT(dim_size_);
    dim_size = dim_size_ + task_id * transpose_param_->num_axes_;
    MS_ASSERT(position_);
    position = position_ + task_id * transpose_param_->num_axes_;
  }

  MS_ASSERT(in_ptr_);
  MS_ASSERT(out_ptr_);
  MS_ASSERT(in_shape_);
  MS_ASSERT(out_shape_);
  MS_ASSERT(transpose_param_);
  auto ret = DoTransposeInt8(in_ptr_, out_ptr_, out_shape_, transpose_param_, thread_offset,
                             thread_offset + num_unit_thread, dim_size, position);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Transpose error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_ERROR;
  }

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

  ret = ParallelLaunch(this->context_->thread_pool_, TransposeInt8Run, this, thread_h_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Tranpose error error_code[" << ret << "]";
  }

  FreeTmpBuf();
  return ret;
}

REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_Transpose, LiteKernelCreator<TransposeInt8CPUKernel>)
}  // namespace mindspore::kernel
