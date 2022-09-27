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

#include "src/litert/kernel/cpu/int8/sub_int8.h"
#include "src/litert/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_SubFusion;

namespace mindspore::kernel {
SubInt8CPUKernel::~SubInt8CPUKernel() {
  if (quant_param_ != nullptr) {
    free(quant_param_);
    quant_param_ = nullptr;
  }
}

int SubInt8CPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), C2NUM);
  CHECK_LESS_RETURN(out_tensors_.size(), C1NUM);
  CHECK_NULL_RETURN(in_tensors_[0]);
  CHECK_NULL_RETURN(in_tensors_[1]);
  CHECK_NULL_RETURN(out_tensors_[0]);
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

  broadcast_ = input0->ElementsNum() != input1->ElementsNum();

  quant_param_ = reinterpret_cast<SubQuantArg *>(malloc(sizeof(SubQuantArg)));
  if (quant_param_ == nullptr) {
    MS_LOG(ERROR) << "Malloc SubQuantArg for Sub int8 op failed!";
    return RET_ERROR;
  }
  const auto &input0_params = input0->quant_params();
  const auto &input1_params = input1->quant_params();
  const auto &output_params = output->quant_params();
  MS_CHECK_TRUE_MSG(!input0_params.empty(), RET_ERROR, "Input 0 quant param cannot be empty.");
  MS_CHECK_TRUE_MSG(!input1_params.empty(), RET_ERROR, "Input 1 quant param cannot be empty.");
  MS_CHECK_TRUE_MSG(!output_params.empty(), RET_ERROR, "Output quant param cannot be empty.");

  quant_param_->in0_args_.scale_ = static_cast<float>(input0_params.front().scale);
  quant_param_->in0_args_.zp_ = -input0_params.front().zeroPoint;
  quant_param_->in1_args_.scale_ = static_cast<float>(input1_params.front().scale);
  quant_param_->in1_args_.zp_ = -input1_params.front().zeroPoint;
  quant_param_->out_args_.scale_ = static_cast<float>(output_params.front().scale);
  quant_param_->out_args_.zp_ = output_params.front().zeroPoint;

  const int left_shift = 20;
  const double twice_max_input_scale = 2 * std::max(quant_param_->in0_args_.scale_, quant_param_->in1_args_.scale_);
  const double real_input0_multiplier = quant_param_->in0_args_.scale_ / twice_max_input_scale;
  const double real_input1_multiplier = quant_param_->in1_args_.scale_ / twice_max_input_scale;
  const double real_output_multiplier = twice_max_input_scale / ((1 << left_shift) * quant_param_->out_args_.scale_);

  QuantizeMultiplierSmallerThanOne(real_input0_multiplier, &quant_param_->input0_multiplier_,
                                   &quant_param_->input0_shift_);
  QuantizeMultiplierSmallerThanOne(real_input1_multiplier, &quant_param_->input1_multiplier_,
                                   &quant_param_->input1_shift_);
  QuantizeMultiplierSmallerThanOne(real_output_multiplier, &quant_param_->output_multiplier_,
                                   &quant_param_->output_shift_);

  quant_param_->output_activation_min_ = std::numeric_limits<int8_t>::min();
  quant_param_->output_activation_max_ = std::numeric_limits<int8_t>::max();

  int left_shift0 = -quant_param_->input0_shift_ > 0 ? -quant_param_->input0_shift_ : 0;
  quant_param_->right_shift0_ = -quant_param_->input0_shift_ > 0 ? 0 : quant_param_->input0_shift_;

  int left_shift1 = -quant_param_->input1_shift_ > 0 ? -quant_param_->input1_shift_ : 0;
  quant_param_->right_shift1_ = -quant_param_->input1_shift_ > 0 ? 0 : quant_param_->input1_shift_;

  quant_param_->left_shift_out_ = -quant_param_->output_shift_ > 0 ? -quant_param_->output_shift_ : 0;
  quant_param_->right_shift_out_ = -quant_param_->output_shift_ > 0 ? 0 : quant_param_->output_shift_;

  quant_param_->left_shift_result0_ = (1 << left_shift) * ((1 << left_shift0));
  quant_param_->left_shift_result1_ = (1 << left_shift) * ((1 << left_shift1));

  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int SubInt8CPUKernel::ReSize() { return RET_OK; }

int SubInt8CPUKernel::DoExecute(int task_id) {
  auto input0_data_ = static_cast<int8_t *>(in_tensors_.at(0)->MutableData());
  auto input1_data_ = static_cast<int8_t *>(in_tensors_.at(1)->MutableData());
  auto output_data_ = static_cast<int8_t *>(out_tensors_.at(0)->MutableData());
  auto element_num = out_tensors_[0]->ElementsNum();

  MS_ASSERT(op_parameter_->thread_num_ != 0);
  int stride = UP_DIV(element_num, op_parameter_->thread_num_);
  int count = MSMIN(stride, element_num - stride * task_id);
  if (count <= 0) {
    return RET_OK;
  }

  auto ret = RET_OK;
  if (broadcast_) {
    ret = SubInt8(tile0_data_ + task_id * stride, tile1_data_ + task_id * stride, output_data_ + task_id * stride,
                  count, quant_param_);
  } else {
    ret = SubInt8(input0_data_ + task_id * stride, input1_data_ + task_id * stride, output_data_ + task_id * stride,
                  count, quant_param_);
  }

  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Subint8 function error error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int SubInt8Run(void *cdata, int task_id, float, float) {
  auto sub_kernel = reinterpret_cast<SubInt8CPUKernel *>(cdata);
  auto ret = sub_kernel->DoExecute(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "SubInt8 DoExecute error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int SubInt8CPUKernel::Run() {
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
    if (tile0_data_ == nullptr) {
      MS_LOG(ERROR) << "malloc memory fail!";
      return RET_ERROR;
    }
    tile1_data_ = static_cast<int8_t *>(ms_context_->allocator->Malloc(out_tensors_.at(0)->Size()));
    if (tile1_data_ == nullptr) {
      MS_LOG(ERROR) << "malloc memory fail!";
      ms_context_->allocator->Free(tile0_data_);
      return RET_ERROR;
    }
    TileDimensionsInt8(static_cast<int8_t *>(in_tensors_.at(0)->data()),
                       static_cast<int8_t *>(in_tensors_.at(1)->data()), reinterpret_cast<int8_t *>(tile0_data_),
                       reinterpret_cast<int8_t *>(tile1_data_), &tile_para);
  }
  auto ret = ParallelLaunch(this->ms_context_, SubInt8Run, this, op_parameter_->thread_num_);
  if (broadcast_) {
    ms_context_->allocator->Free(tile0_data_);
    ms_context_->allocator->Free(tile1_data_);
  }
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "SubInt8Run function error error_code[" << ret << "]";
  }
  return ret;
}

REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_SubFusion, LiteKernelCreator<SubInt8CPUKernel>)
}  // namespace mindspore::kernel
