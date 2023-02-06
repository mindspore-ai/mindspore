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

#include "src/litert/kernel/cpu/int8/softmax_int8.h"
#include <limits>
#include "nnacl/int8/softmax_int8.h"
#include "schema/model_generated.h"
#include "include/errorcode.h"
#include "src/litert/kernel_registry.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_NULL_PTR;
using mindspore::schema::PrimitiveType_Softmax;

namespace mindspore::kernel {
SoftmaxInt8CPUKernel::~SoftmaxInt8CPUKernel() {
  if (quant_param_ != nullptr) {
    free(quant_param_);
    quant_param_ = nullptr;
  }
}

int SoftmaxInt8CPUKernel::Prepare() {
  auto ret = SoftmaxBaseCPUKernel::Prepare();
  if (ret != RET_OK) {
    return ret;
  }
  if (in_tensors_[0]->data_type() != mindspore::kNumberTypeInt8 ||
      out_tensors_[0]->data_type() != mindspore::kNumberTypeInt8) {
    MS_LOG(ERROR) << "Datatype error, input0 data_type is " << in_tensors_[0]->data_type() << ", output data_type is "
                  << out_tensors_[0]->data_type();
    return RET_ERROR;
  }
  quant_param_ = reinterpret_cast<SoftmaxQuantArg *>(malloc(sizeof(SoftmaxQuantArg)));
  if (quant_param_ == nullptr) {
    MS_LOG(ERROR) << "Malloc SoftmaxQuantArg for Softmax int8 op failed!";
    return RET_ERROR;
  }

  auto *input_tensor = in_tensors_.at(kInputIndex);
  MS_ASSERT(input_tensor != nullptr);

  auto in_quant_args = input_tensor->quant_params();
  CHECK_LESS_RETURN(in_quant_args.size(), 1);
  quant_param_->in_quant_args_.scale_ = static_cast<float>(in_quant_args.front().scale);
  quant_param_->in_quant_args_.zp_ = -in_quant_args.front().zeroPoint;

  auto *out_tensor = out_tensors_.at(kOutputIndex);
  MS_ASSERT(out_tensor != nullptr);

  auto out_quant_args = out_tensor->quant_params();
  CHECK_LESS_RETURN(out_quant_args.size(), 1);
  quant_param_->out_quant_arg_.scale_ = static_cast<float>(out_quant_args.front().scale);
  quant_param_->out_quant_arg_.zp_ = -out_quant_args.front().zeroPoint;
  quant_param_->output_activation_min_ = std::numeric_limits<int8_t>::min();
  quant_param_->output_activation_max_ = std::numeric_limits<int8_t>::max();

  const double input_real_multiplier =
    MSMIN(quant_param_->in_quant_args_.scale_ * (1 << (unsigned int)(31 - 5)), (1LL << 31) - 1.0);
  int right_shift = 0;
  QuantizeMultiplierSmallerThanOne(input_real_multiplier, &quant_param_->output_multiplier_, &right_shift);
  quant_param_->shift_left_ = right_shift < 0 ? -right_shift : 0;
  quant_param_->shift_right_ = right_shift > 0 ? right_shift : 0;

  if (!InferShapeDone()) {
    return RET_OK;
  }

  return ReSize();
}

int SoftmaxInt8CPUKernel::ReSize() { return SoftmaxBaseCPUKernel::ReSize(); }

int SoftmaxInt8CPUKernel::DoSoftmax(int task_id) {
  MS_ASSERT(in_tensors_.size() == 1);
  MS_ASSERT(out_tensors_.size() == 1);

  auto input_ptr = reinterpret_cast<int8_t *>(in_tensors_.at(0)->MutableData());
  MS_ASSERT(input_ptr);
  auto output_ptr = reinterpret_cast<int8_t *>(out_tensors_.at(0)->MutableData());
  MS_ASSERT(output_ptr);

  int outter_size = 1;
  int inner_size = 1;
  for (int i = 0; i < softmax_param_->axis_; i++) {
    outter_size *= softmax_param_->input_shape_[i];
  }
  for (int i = softmax_param_->axis_; i < softmax_param_->n_dim_; i++) {
    inner_size *= softmax_param_->input_shape_[i];
  }

  int stride = UP_DIV(outter_size, thread_count_);
  if (INT_MUL_OVERFLOW(task_id, stride)) {
    MS_LOG(ERROR) << "int mul overflow.";
    return RET_ERROR;
  }
  int count = MSMIN(stride, outter_size - stride * task_id);
  int stride_size = stride * task_id * inner_size;

  auto error_code = SoftmaxInt8(input_ptr + stride_size, output_ptr + stride_size, count, exp_data_ + stride_size,
                                sum_data_, quant_param_, softmax_param_);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "DoSoftmax error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int SoftmaxRun(void *cdata, int task_id, float, float) {
  CHECK_NULL_RETURN(cdata);
  auto softmax_kernel = reinterpret_cast<SoftmaxInt8CPUKernel *>(cdata);
  auto error_code = softmax_kernel->DoSoftmax(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "SoftmaxRun error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int SoftmaxInt8CPUKernel::Run() {
  CHECK_LESS_RETURN(MAX_MALLOC_SIZE, softmax_param_->element_size_ * sizeof(int));
  exp_data_ = reinterpret_cast<int *>(ms_context_->allocator->Malloc(softmax_param_->element_size_ * sizeof(int)));
  int inner_size = 1;
  for (int i = softmax_param_->axis_ + 1; i < softmax_param_->n_dim_; i++) {
    if (INT_MUL_OVERFLOW(inner_size, softmax_param_->input_shape_[i])) {
      MS_LOG(ERROR) << "int mul overflow.";
      return RET_ERROR;
    }
    inner_size *= softmax_param_->input_shape_[i];
  }
  sum_data_ = reinterpret_cast<int *>(ms_context_->allocator->Malloc(inner_size * sizeof(int)));
  if (exp_data_ == nullptr || sum_data_ == nullptr) {
    MS_LOG(ERROR) << "Memory allocation failed";
    ms_context_->allocator->Free(exp_data_);
    ms_context_->allocator->Free(sum_data_);
    return RET_ERROR;
  }
  auto ret = ParallelLaunch(this->ms_context_, SoftmaxRun, this, thread_count_);
  ms_context_->allocator->Free(exp_data_);
  ms_context_->allocator->Free(sum_data_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Softmax function error error_code[" << ret << "]";
  }
  return ret;
}

REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_Softmax, LiteKernelCreator<SoftmaxInt8CPUKernel>)
}  // namespace mindspore::kernel
