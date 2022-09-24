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

#include "src/litert/kernel/cpu/int8/div_int8.h"
#include <limits>
#include <algorithm>
#include "nnacl/int8/arithmetic_int8.h"
#include "src/litert/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_DivFusion;

namespace mindspore::kernel {
DivInt8CPUKernel::~DivInt8CPUKernel() {
  if (quant_args_ != nullptr) {
    free(quant_args_);
    quant_args_ = nullptr;
  }
}
int DivInt8CPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), C2NUM);
  CHECK_NULL_RETURN(in_tensors_.at(0));
  CHECK_NULL_RETURN(in_tensors_.at(1));
  if (in_tensors_[0]->data_type() != mindspore::kNumberTypeInt8 ||
      in_tensors_[1]->data_type() != mindspore::kNumberTypeInt8 ||
      out_tensors_[0]->data_type() != mindspore::kNumberTypeInt8) {
    MS_LOG(ERROR) << "Datatype error, input0 data_type is " << in_tensors_[0]->data_type() << ", input1 data_type is "
                  << in_tensors_[1]->data_type() << ", output data_type is " << out_tensors_[0]->data_type();
    return RET_ERROR;
  }
  lite::Tensor *input0 = in_tensors_.at(0);
  lite::Tensor *input1 = in_tensors_.at(1);
  lite::Tensor *output = out_tensors_.at(0);

  quant_args_ = reinterpret_cast<DivQuantArg *>(malloc(sizeof(DivQuantArg)));
  if (quant_args_ == nullptr) {
    MS_LOG(ERROR) << "Malloc DivQuantArg for Div int8 op failed!";
    return RET_ERROR;
  }
  const auto &input0_params = input0->quant_params();
  const auto &input1_params = input1->quant_params();
  const auto &output_params = output->quant_params();
  MS_CHECK_TRUE_MSG(!input0_params.empty(), RET_ERROR, "Input 0 quant param cannot be empty.");
  MS_CHECK_TRUE_MSG(!input1_params.empty(), RET_ERROR, "Input 1 quant param cannot be empty.");
  MS_CHECK_TRUE_MSG(!output_params.empty(), RET_ERROR, "Output quant param cannot be empty.");

  quant_args_->in0_args_.scale_ = static_cast<float>(input0_params.front().scale);
  quant_args_->in0_args_.zp_ = -input0_params.front().zeroPoint;
  quant_args_->in1_args_.scale_ = static_cast<float>(input1_params.front().scale);
  quant_args_->in1_args_.zp_ = -input1_params.front().zeroPoint;
  quant_args_->out_args_.scale_ = static_cast<float>(output_params.front().scale);
  quant_args_->out_args_.zp_ = output_params.front().zeroPoint;

  const double real_multiplier =
    quant_args_->in0_args_.scale_ / (quant_args_->in1_args_.scale_ * quant_args_->out_args_.scale_);

  QuantizeMultiplier(real_multiplier, &quant_args_->output_multiplier_, &quant_args_->output_shift_);

  quant_args_->output_activation_min_ = std::numeric_limits<int8_t>::min();
  quant_args_->output_activation_max_ = std::numeric_limits<int8_t>::max();

  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int DivInt8CPUKernel::ReSize() {
  lite::Tensor *input0 = in_tensors_.at(0);
  lite::Tensor *input1 = in_tensors_.at(1);
  MS_CHECK_GT(input0->ElementsNum(), 0, RET_ERROR);
  MS_CHECK_GT(input1->ElementsNum(), 0, RET_ERROR);

  broadcast_ = input0->ElementsNum() != input1->ElementsNum();
  div_scalar_ = input1->ElementsNum() == 1;
  return RET_OK;
}

int DivInt8CPUKernel::DoExecute(int task_id) {
  auto input0_data_ = static_cast<int8_t *>(in_tensors_.at(0)->MutableData());
  auto input1_data_ = static_cast<int8_t *>(in_tensors_.at(1)->MutableData());
  auto output_data_ = static_cast<int8_t *>(out_tensors_.at(0)->MutableData());
  auto element_num = out_tensors_[0]->ElementsNum();
  MS_CHECK_GT(element_num, 0, RET_ERROR);

  MS_ASSERT(op_parameter_->thread_num_ != 0);
  int stride = UP_DIV(element_num, op_parameter_->thread_num_);
  int count = MSMIN(stride, element_num - stride * task_id);

  auto ret = RET_OK;
  if (broadcast_) {
    ret = DivInt8(tile0_data_ + task_id * count, tile1_data_ + task_id * count, output_data_ + task_id * count, count,
                  quant_args_);
  } else {
    ret = DivInt8(input0_data_ + task_id * count, input1_data_ + task_id * count, output_data_ + task_id * count, count,
                  quant_args_);
  }

  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Divint8 function error error_code[" << ret << "]";
  }
  return ret;
}

int DivInt8Run(void *cdata, int task_id, float, float) {
  auto div_kernel = reinterpret_cast<DivInt8CPUKernel *>(cdata);
  auto ret = div_kernel->DoExecute(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "DivInt8 DoExecute error task_id[" << task_id << "] error_code[" << ret << "]";
  }
  return ret;
}

int DivInt8CPUKernel::DivScalarDoExecute(int task_id) {
  auto input0_data_ = static_cast<int8_t *>(in_tensors_.at(0)->MutableData());
  auto input1_data_ = static_cast<int8_t *>(in_tensors_.at(1)->MutableData());
  auto output_data_ = static_cast<int8_t *>(out_tensors_.at(0)->MutableData());
  auto element_num = out_tensors_[0]->ElementsNum();

  MS_ASSERT(op_parameter_->thread_num_ != 0);
  int stride = UP_DIV(element_num, op_parameter_->thread_num_);
  int count = MSMIN(stride, element_num - stride * task_id);

  auto ret =
    DivScalarInt8(input0_data_ + task_id * stride, input1_data_, output_data_ + task_id * stride, count, quant_args_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Divint8 function error error_code[" << ret << "]";
  }
  return ret;
}

int DivScalarInt8Run(void *cdata, int task_id, float, float) {
  auto div_kernel = reinterpret_cast<DivInt8CPUKernel *>(cdata);
  auto ret = div_kernel->DivScalarDoExecute(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "DivInt8 DoExecute error task_id[" << task_id << "] error_code[" << ret << "]";
  }
  return ret;
}

int DivInt8CPUKernel::Run() {
  if (div_scalar_) {
    auto ret = ParallelLaunch(this->ms_context_, DivScalarInt8Run, this, op_parameter_->thread_num_);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "DivInt8Run function error error_code[" << ret << "]";
    }
    return ret;
  }

  if (broadcast_) {
    ArithmeticParameter tile_para;
    auto out_shape = out_tensors_[FIRST_INPUT]->shape();
    tile_para.ndim_ = out_shape.size();
    auto in_shape0 = in_tensors_[FIRST_INPUT]->shape();
    MS_CHECK_TRUE_MSG(out_shape.size() >= in_shape0.size(), RET_ERROR,
                      "Sub first-input shape size is larger than out.");
    for (size_t i = 0; i < out_shape.size() - in_shape0.size(); ++i) {
      tile_para.in_shape0_[i] = 1;
    }
    for (size_t i = 0; i < in_shape0.size(); ++i) {
      tile_para.in_shape0_[i + out_shape.size() - in_shape0.size()] = in_shape0[i];
    }
    auto in_shape1 = in_tensors_[SECOND_INPUT]->shape();
    MS_CHECK_TRUE_MSG(out_shape.size() >= in_shape1.size(), RET_ERROR,
                      "Sub second-input shape size is larger than out.");
    for (size_t i = 0; i < out_shape.size() - in_shape1.size(); ++i) {
      tile_para.in_shape1_[i] = 1;
    }
    for (size_t i = 0; i < in_shape1.size(); ++i) {
      tile_para.in_shape1_[i + out_shape.size() - in_shape1.size()] = in_shape1[i];
    }
    for (size_t i = 0; i < out_shape.size(); ++i) {
      tile_para.out_shape_[i] = out_shape[i];
    }
    tile0_data_ = static_cast<int8_t *>(ms_context_->allocator->Malloc(out_tensors_.at(0)->Size()));
    tile1_data_ = static_cast<int8_t *>(ms_context_->allocator->Malloc(out_tensors_.at(0)->Size()));
    if (tile0_data_ == nullptr || tile1_data_ == nullptr) {
      MS_LOG(ERROR) << "Memory allocation failed";
      ms_context_->allocator->Free(tile0_data_);
      ms_context_->allocator->Free(tile1_data_);
      tile0_data_ = nullptr;
      tile1_data_ = nullptr;
      return RET_ERROR;
    }
    TileDimensionsInt8(static_cast<int8_t *>(in_tensors_.at(0)->MutableData()),
                       static_cast<int8_t *>(in_tensors_.at(1)->MutableData()), reinterpret_cast<int8_t *>(tile0_data_),
                       reinterpret_cast<int8_t *>(tile1_data_), &tile_para);
  }
  auto ret = ParallelLaunch(this->ms_context_, DivInt8Run, this, op_parameter_->thread_num_);
  if (broadcast_) {
    ms_context_->allocator->Free(tile0_data_);
    ms_context_->allocator->Free(tile1_data_);
    tile0_data_ = nullptr;
    tile1_data_ = nullptr;
  }
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "DivInt8Run function error error_code[" << ret << "]";
  }
  return ret;
}

REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_DivFusion, LiteKernelCreator<DivInt8CPUKernel>)
}  // namespace mindspore::kernel
