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

#include "src/runtime/kernel/arm/fp32/fullconnection.h"
#include "src/runtime/runtime_api.h"
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_MEMORY_FAILED;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
FullconnectionCPUKernel::~FullconnectionCPUKernel() {
  FreeBuf();
  return;
}

void FullconnectionCPUKernel::FreeBuf() {
  if (a_c12_ptr_ != nullptr) {
    free(a_c12_ptr_);
    a_c12_ptr_ = nullptr;
  }
  if (b_r8_ptr_ != nullptr) {
    free(b_r8_ptr_);
    b_r8_ptr_ = nullptr;
  }
  if (bias_ptr_ != nullptr) {
    free(bias_ptr_);
    bias_ptr_ = nullptr;
  }
}

int FullconnectionCPUKernel::ReSize() {
  FreeBuf();
  fc_param_->row_ = (in_tensors_[0]->shape())[0];
  fc_param_->col_ = (in_tensors_[1]->shape())[0];
  fc_param_->deep_ = (in_tensors_[1]->shape())[1];

  fc_param_->row_12_ = UP_ROUND(fc_param_->row_, C12NUM);
  fc_param_->col_8_ = UP_ROUND(fc_param_->col_, C8NUM);

  thread_count_ = MSMIN(thread_count_, UP_DIV(fc_param_->col_8_, 8));
  thread_stride_ = UP_DIV(UP_DIV(fc_param_->col_8_, 8), thread_count_);

  bias_ptr_ = reinterpret_cast<float *>(malloc(fc_param_->col_8_ * sizeof(float)));
  memset(bias_ptr_, 0, fc_param_->col_8_ * sizeof(float));
  if (in_tensors_.size() == 3) {
    memcpy(bias_ptr_, in_tensors_[2]->Data(), fc_param_->col_ * sizeof(float));
  }

  a_c12_ptr_ = reinterpret_cast<float *>(malloc(fc_param_->row_12_ * fc_param_->deep_ * sizeof(float)));
  if (a_c12_ptr_ == nullptr) {
    return RET_MEMORY_FAILED;
  }
  memset(a_c12_ptr_, 0, fc_param_->row_12_ * fc_param_->deep_ * sizeof(float));

  b_r8_ptr_ = reinterpret_cast<float *>(malloc(fc_param_->col_8_ * fc_param_->deep_ * sizeof(float)));
  if (b_r8_ptr_ == nullptr) {
    FreeBuf();
    return RET_MEMORY_FAILED;
  }
  memset(b_r8_ptr_, 0, fc_param_->col_8_ * fc_param_->deep_ * sizeof(float));

  fc_param_->a_const_ = (in_tensors_[0]->Data() != nullptr);
  fc_param_->b_const_ = (in_tensors_[1]->Data() != nullptr);
  if (fc_param_->a_const_) InitMatrixA(reinterpret_cast<float *>(in_tensors_[0]->Data()), a_c12_ptr_);
  if (fc_param_->b_const_) InitMatrixB(reinterpret_cast<float *>(in_tensors_[1]->Data()), b_r8_ptr_);
  return RET_OK;
}

int FullconnectionCPUKernel::Init() {
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

void FullconnectionCPUKernel::InitMatrixA(float *src_ptr, float *dst_ptr) {
  RowMajor2Col12Major(src_ptr, a_c12_ptr_, fc_param_->row_, fc_param_->deep_);
}

void FullconnectionCPUKernel::InitMatrixB(float *src_ptr, float *dst_ptr) {
  RowMajor2Col8Major(src_ptr, dst_ptr, fc_param_->col_, fc_param_->deep_);
}

int FcFp32MatmulRun(void *cdata, int task_id) {
  auto fc = reinterpret_cast<FullconnectionCPUKernel *>(cdata);
  auto error_code = fc->DoMatmul(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "FcFp32MatmulRun error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int FullconnectionCPUKernel::DoMatmul(int task_id) {
  int cur_oc = MSMIN(thread_stride_ * C8NUM, fc_param_->col_ - task_id * thread_stride_ * C8NUM);
  if (cur_oc <= 0) {
    return RET_OK;
  }

  MatMulOpt(a_c12_ptr_, b_r8_ptr_ + task_id * thread_stride_ * C8NUM * fc_param_->deep_,
            c_r_ptr + task_id * thread_stride_ * C8NUM, bias_ptr_ + task_id * thread_stride_ * C8NUM,
            fc_param_->act_type_, fc_param_->deep_, fc_param_->row_, cur_oc, fc_param_->col_, OutType_Nhwc);
  return RET_OK;
}

int FullconnectionCPUKernel::Run() {
  auto prepare_ret = Prepare();
  if (prepare_ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare fail!ret: " << prepare_ret;
    return prepare_ret;
  }
  auto a_ptr = reinterpret_cast<float *>(in_tensors_.at(0)->Data());
  auto b_ptr = reinterpret_cast<float *>(in_tensors_.at(1)->Data());
  c_r_ptr = reinterpret_cast<float *>(out_tensors_.at(0)->Data());

  if (!fc_param_->a_const_) InitMatrixA(a_ptr, a_c12_ptr_);
  if (!fc_param_->b_const_) InitMatrixB(b_ptr, b_r8_ptr_);

  ParallelLaunch(THREAD_POOL_DEFAULT, FcFp32MatmulRun, this, thread_count_);

  return RET_OK;
}
}  // namespace mindspore::kernel
