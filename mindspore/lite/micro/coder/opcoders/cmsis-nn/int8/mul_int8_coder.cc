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

#include "coder/opcoders/cmsis-nn/int8/mul_int8_coder.h"
#include <string>
#include "coder/opcoders/serializers/serializer.h"
#include "nnacl/int8/quantize.h"
#include "coder/opcoders/file_collector.h"

using mindspore::schema::PrimitiveType_MulFusion;

namespace mindspore::lite::micro::cmsis {

int MulInt8Coder::Prepare(CoderContext *const context) {
  input1_ = OperatorCoder::input_tensors().at(0);
  input2_ = OperatorCoder::input_tensors().at(1);

  MS_CHECK_PTR(input1_);
  MS_CHECK_PTR(input2_);

  MS_CHECK_TRUE(!input1_->quant_params().empty(), "input1_ quant_params is empty");
  MS_CHECK_TRUE(!input2_->quant_params().empty(), "input2_ quant_params is empty");
  MS_CHECK_TRUE(!output_tensor_->quant_params().empty(), "output quant_params is empty");

  input_1_offset_ = -input1_->quant_params().at(0).zeroPoint;
  input_2_offset_ = -input2_->quant_params().at(0).zeroPoint;
  out_offset_ = output_tensor_->quant_params().at(0).zeroPoint;
  const double input1_scale = input1_->quant_params().at(0).scale;
  const double input2_scale = input2_->quant_params().at(0).scale;
  const double output_scale = output_tensor_->quant_params().at(0).scale;

  const double real_multiplier = input1_scale * input2_scale / output_scale;

  QuantizeMultiplier(real_multiplier, &out_mult_, &out_shift_);

  CalculateActivationRangeQuantized(false, false, out_offset_, output_scale, &out_activation_min_,
                                    &out_activation_max_);

  MS_CHECK_TRUE(input1_->ElementsNum() == input2_->ElementsNum(), "tensor length not match");

  block_size_ = input1_->ElementsNum();

  return RET_OK;
}

int MulInt8Coder::DoCode(CoderContext *const context) {
  Serializer code;
  code.precision(kPrecision);

  Collect(context, {"CMSIS/NN/Include/arm_nnfunctions.h"}, {"arm_elementwise_mul_s8.c"});

  code.CodeFunction("arm_elementwise_mul_s8", input1_, input2_, input_1_offset_, input_2_offset_, output_tensor_,
                    out_offset_, out_mult_, out_shift_, out_activation_min_, out_activation_max_, block_size_);

  MS_LOG(INFO) << "MulInt8Coder has been called";
  context->AppendCode(code.str());
  return RET_OK;
}
REG_OPERATOR_CODER(kARM32M, kNumberTypeInt8, PrimitiveType_MulFusion, CPUOpCoderCreator<MulInt8Coder>)
}  // namespace mindspore::lite::micro::cmsis
