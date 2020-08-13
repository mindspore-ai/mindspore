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
  if (a_c8_ptr_ != nullptr) {
    free(a_c8_ptr_);
    a_c8_ptr_ = nullptr;
  }
  if (b_r8_ptr_ != nullptr) {
    free(b_r8_ptr_);
    b_r8_ptr_ = nullptr;
  }
  if (c_r8x8_ptr_ != nullptr) {
    free(c_r8x8_ptr_);
    c_r8x8_ptr_ = nullptr;
  }
  if (bias_ptr_ != nullptr) {
    free(bias_ptr_);
    bias_ptr_ = nullptr;
  }
}

int FullconnectionCPUKernel::ReSize() { return RET_OK; }

int FullconnectionCPUKernel::Init() {
  if (context_->infer_shape_interrupt_ && !context_->running_) {
    set_need_reinit();
    return RET_OK;
  }
  fc_param_->row_ = (in_tensors_[0]->shape())[0];
  fc_param_->col_ = (in_tensors_[1]->shape())[0];
  fc_param_->deep_ = (in_tensors_[1]->shape())[1];

  fc_param_->row_8_ = UP_ROUND(fc_param_->row_, 8);
  fc_param_->col_8_ = UP_ROUND(fc_param_->col_, 8);

  thread_count_ = MSMIN(thread_count_, UP_DIV(fc_param_->col_8_, 8));
  thread_stride_ = UP_DIV(UP_DIV(fc_param_->col_8_, 8), thread_count_);

  bias_ptr_ = reinterpret_cast<float *>(malloc(fc_param_->col_8_ * sizeof(float)));
  memset(bias_ptr_, 0, fc_param_->col_8_ * sizeof(float));
  if (in_tensors_.size() == 3) {
    memcpy(bias_ptr_, in_tensors_[2]->Data(), fc_param_->col_ * sizeof(float));
  }

  a_c8_ptr_ = reinterpret_cast<float *>(malloc(fc_param_->row_8_ * fc_param_->deep_ * sizeof(float)));
  if (a_c8_ptr_ == nullptr) {
    return RET_MEMORY_FAILED;
  }
  memset(a_c8_ptr_, 0, fc_param_->row_8_ * fc_param_->deep_ * sizeof(float));

  b_r8_ptr_ = reinterpret_cast<float *>(malloc(fc_param_->col_8_ * fc_param_->deep_ * sizeof(float)));
  if (b_r8_ptr_ == nullptr) {
    return RET_MEMORY_FAILED;
  }
  memset(b_r8_ptr_, 0, fc_param_->col_8_ * fc_param_->deep_ * sizeof(float));
  RowMajor2Col8Major(reinterpret_cast<float *>(in_tensors_[1]->Data()), b_r8_ptr_, fc_param_->col_, fc_param_->deep_);

  c_r8x8_ptr_ = reinterpret_cast<float *>(malloc(fc_param_->row_8_ * fc_param_->col_8_ * sizeof(float)));
  if (c_r8x8_ptr_ == nullptr) {
    return RET_MEMORY_FAILED;
  }
  memset(c_r8x8_ptr_, 0, fc_param_->row_8_ * fc_param_->col_8_ * sizeof(float));
  return RET_OK;
}

int FcFp32MatmulRun(int task_id, LiteParallelGroupEnv *penv, void *cdata) {
  auto fc = reinterpret_cast<FullconnectionCPUKernel *>(cdata);
  auto error_code = fc->DoMatmul(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "FcFp32MatmulRun error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int FullconnectionCPUKernel::DoMatmul(int task_id) {
  int cur_oc = MSMIN(thread_stride_, UP_DIV(fc_param_->col_8_, 8) - task_id * thread_stride_);
  if (cur_oc <= 0) {
    return RET_OK;
  }

  MatMul(a_c8_ptr_, b_r8_ptr_ + task_id * thread_stride_ * C8NUM * fc_param_->deep_,
         c_r8x8_ptr_ + task_id * thread_stride_ * C8NUM * fc_param_->row_8_,
         bias_ptr_ + task_id * thread_stride_ * C8NUM, fc_param_->act_type_, fc_param_->deep_, fc_param_->row_8_,
         cur_oc * 8, 0, false);
  return RET_OK;
}

int FullconnectionCPUKernel::Run() {
  auto prepare_ret = Prepare();
  if (prepare_ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare fail!ret: " << prepare_ret;
    return prepare_ret;
  }
  auto a_ptr = reinterpret_cast<float *>(in_tensors_.at(0)->Data());
  auto output_ptr = reinterpret_cast<float *>(out_tensors_.at(0)->Data());

  RowMajor2Col8Major(a_ptr, a_c8_ptr_, fc_param_->row_, fc_param_->deep_);

  LiteBackendParallelLaunch(FcFp32MatmulRun, this, thread_count_);

  Row8x8Major2RowMajor(c_r8x8_ptr_, output_ptr, fc_param_->row_, fc_param_->col_, fc_param_->col_);
  return RET_OK;
}
}  // namespace mindspore::kernel
