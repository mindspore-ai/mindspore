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

#include "src/runtime/kernel/arm/int8/softmax_int8.h"
#include "src/runtime/kernel/arm/nnacl/int8/softmax_int8.h"
#include "schema/model_generated.h"
#include "src/runtime/runtime_api.h"
#include "include/errorcode.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {

int SoftmaxInt8CPUKernel::Init() {
  auto ret = SoftmaxBaseCPUKernel::Init();
  if (ret != RET_OK) {
    return ret;
  }

  auto *input_tensor = in_tensors_.at(kInputIndex);
  MS_ASSERT(input_tensor);

  auto in_quant_args = input_tensor->GetQuantParams();
  quant_params_.in_quant_args_.scale_ = in_quant_args.front().scale;
  quant_params_.in_quant_args_.zp_ = in_quant_args.front().zeroPoint;

  auto *out_tensor = out_tensors_.at(kOutputIndex);
  MS_ASSERT(out_tensor);

  auto out_quant_args = out_tensor->GetQuantParams();
  quant_params_.out_quant_arg_.scale_ = out_quant_args.front().scale;
  quant_params_.out_quant_arg_.zp_ = out_quant_args.front().zeroPoint;

  if (!InferShapeDone()) {
    return RET_OK;
  }

  return ReSize();
}

void SoftmaxInt8CPUKernel::FreeTmpBuffer() {
  if (exp_data_ != nullptr) {
    free(exp_data_);
    exp_data_ = nullptr;
  }
  if (sum_data_ != nullptr) {
    free(sum_data_);
    sum_data_ = nullptr;
  }
}

int SoftmaxInt8CPUKernel::ReSize() {
  auto ret = SoftmaxBaseCPUKernel::ReSize();
  if (ret != RET_OK) {
    return ret;
  }
  FreeTmpBuffer();
  exp_data_ = reinterpret_cast<float *>(malloc(softmax_param_->element_size_ * sizeof(float)));
  int inner_size = 1;
  for (int i = softmax_param_->axis_ + 1; i < softmax_param_->n_dim_; i++) {
    inner_size *= softmax_param_->input_shape_[i];
  }
  sum_data_ = reinterpret_cast<float *>(malloc(inner_size * sizeof(float)));
  return RET_OK;
}

int SoftmaxInt8CPUKernel::DoSoftmax(int task_id) {
  MS_ASSERT(in_tensors_.size() == 1);
  MS_ASSERT(out_tensors_.size() == 1);

  auto input_ptr = reinterpret_cast<int8_t *>(in_tensors_.at(0)->Data());
  auto output_ptr = reinterpret_cast<int8_t *>(out_tensors_.at(0)->Data());

  int outter_size = 1, inner_size = 1;
  for (int i = 0; i < softmax_param_->axis_; i++) {
    outter_size *= softmax_param_->input_shape_[i];
  }
  for (int i = softmax_param_->axis_; i < softmax_param_->n_dim_; i++) {
    inner_size *= softmax_param_->input_shape_[i];
  }

  int stride = UP_DIV(outter_size, thread_count_);
  int count = MSMIN(stride, outter_size - stride * task_id);

  input_ptr += stride * task_id * inner_size;
  output_ptr += stride * task_id * inner_size;
  exp_data_ += stride * task_id * inner_size;

  auto error_code = Int8Softmax(input_ptr, output_ptr, count, exp_data_, sum_data_, quant_params_, softmax_param_);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "DoSoftmax error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int SoftmaxRun(int task_id, LiteParallelGroupEnv *penv, void *cdata) {
  auto softmax_kernel = reinterpret_cast<SoftmaxInt8CPUKernel *>(cdata);
  auto error_code = softmax_kernel->DoSoftmax(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "SoftmaxRun error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int SoftmaxInt8CPUKernel::Run() {
  auto ret = Prepare();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare fail!ret: " << ret;
    return RET_ERROR;
  }
  auto input_ptr = reinterpret_cast<int8_t *>(in_tensors_.at(0)->Data());
  int ele_size = softmax_param_->element_size_;
  for (int i = 0; i < ele_size; i++) {
    float input_scaled = ((input_ptr[i] - quant_params_.in_quant_args_.zp_) * quant_params_.in_quant_args_.scale_);
    exp_data_[i] = exp(input_scaled);
  }
  int error_code = LiteBackendParallelLaunch(SoftmaxRun, this, thread_count_);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Softmax function error error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}
}  // namespace mindspore::kernel
