/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "coder/opcoders/cmsis-nn/int8/add_int8_coder.h"
#include <algorithm>
#include <limits>
#include "coder/opcoders/serializers/serializer.h"
#include "nnacl/arithmetic.h"
#include "nnacl/int8/quantize.h"
#include "coder/opcoders/file_collector.h"
#include "coder/log.h"

using mindspore::schema::PrimitiveType_AddFusion;

namespace mindspore::lite::micro::cmsis {

int AddInt8Coder::Prepare(CoderContext *const context) {
  input1_ = input_tensors_.at(0);
  input2 = input_tensors_.at(1);

  MS_CHECK_PTR(input1_);
  MS_CHECK_PTR(input2);

  MS_CHECK_TRUE(!input1_->quant_params().empty(), "input1_ quant_params is empty");
  MS_CHECK_TRUE(!input2->quant_params().empty(), "input2_ quant_params is empty");
  MS_CHECK_TRUE(!output_tensor_->quant_params().empty(), "output quant_params is empty");

  input_1_offset_ = -input1_->quant_params().at(0).zeroPoint;
  input_2_offset_ = -input2->quant_params().at(0).zeroPoint;
  out_offset_ = output_tensor_->quant_params().at(0).zeroPoint;
  const double input1_scale = input1_->quant_params().at(0).scale;
  const double input2_scale = input2->quant_params().at(0).scale;
  const double output_scale = output_tensor_->quant_params().at(0).scale;
  left_shift_ = 20;
  const double twice_max_input_scale = 2 * std::max(input1_scale, input2_scale);
  const double real_input1_multiplier = static_cast<double>(input1_scale) / twice_max_input_scale;
  const double real_input2_multiplier = static_cast<double>(input2_scale) / twice_max_input_scale;
  const double real_output_multiplier =
    twice_max_input_scale / ((1 << static_cast<size_t>(left_shift_)) * static_cast<double>(output_scale));

  MS_CHECK_TRUE(0 <= real_input1_multiplier && real_input1_multiplier <= 1,
                "real_input1_multiplier should be in (0, 1)");
  QuantizeMultiplier(real_input1_multiplier, &input_1_mult_, &input_1_shift_);
  MS_CHECK_TRUE(0 <= real_input2_multiplier && real_input2_multiplier <= 1,
                "real_input2_multiplier should be in (0, 1)");
  QuantizeMultiplier(real_input2_multiplier, &input_2_mult_, &input_2_shift_);
  MS_CHECK_TRUE(0 <= real_output_multiplier && real_output_multiplier <= 1,
                "real_output_multiplier should be in (0, 1)");
  QuantizeMultiplier(real_output_multiplier, &out_mult_, &out_shift_);

  out_activation_min_ = std::numeric_limits<int8_t>::min();
  out_activation_max_ = std::numeric_limits<int8_t>::max();

  MS_CHECK_TRUE(input1_->ElementsNum() == input2->ElementsNum(), "tensor length not match");

  block_size_ = input1_->ElementsNum();

  return RET_OK;
}

int AddInt8Coder::DoCode(CoderContext *const context) {
  Serializer code;
  code.precision(kPrecision);

  Collect(context, {"CMSIS/NN/Include/arm_nnfunctions.h"}, {"arm_elementwise_add_s8.c"});

  code.CodeFunction("arm_elementwise_add_s8", input1_, input2, input_1_offset_, input_1_mult_, input_1_shift_,
                    input_2_offset_, input_2_mult_, input_2_shift_, left_shift_, output_tensor_, out_offset_, out_mult_,
                    out_shift_, out_activation_min_, out_activation_max_, block_size_);

  MS_LOG(INFO) << "AddInt8Coder has been called";
  context->AppendCode(code.str());
  return RET_OK;
}
REG_OPERATOR_CODER(kARM32M, kNumberTypeInt8, PrimitiveType_AddFusion, CPUOpCoderCreator<AddInt8Coder>)
}  // namespace mindspore::lite::micro::cmsis
