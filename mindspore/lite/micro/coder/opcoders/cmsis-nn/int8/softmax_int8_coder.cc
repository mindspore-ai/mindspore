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

#include "coder/opcoders/cmsis-nn/int8/softmax_int8_coder.h"
#include <limits>
#include "coder/opcoders/serializers/serializer.h"
#include "coder/opcoders/file_collector.h"

using mindspore::schema::PrimitiveType_Softmax;
namespace mindspore::lite::micro::cmsis {

int SoftMaxInt8Coder::Prepare(CoderContext *const context) {
  SoftmaxBaseCoder::Init();

  MS_CHECK_TRUE(!input_tensor_->quant_params().empty(), "input quant_params is empty");
  QuantArg in_quant_arg = input_tensor_->quant_params().at(0);
  quant_params_.in_quant_args_.zp_ = -in_quant_arg.zeroPoint;

  std::vector<QuantArg> out_quant_args = output_tensor_->quant_params();
  MS_CHECK_TRUE(!out_quant_args.empty(), "output quant_params is empty");
  quant_params_.out_quant_arg_.scale_ = static_cast<float>(out_quant_args.at(0).scale);
  quant_params_.out_quant_arg_.zp_ = out_quant_args.at(0).zeroPoint;
  quant_params_.output_activation_min_ = std::numeric_limits<int8_t>::min();
  quant_params_.output_activation_max_ = std::numeric_limits<int8_t>::max();

  const int total_signed_bits = 31;
  const int input_integer_bits = 5;
  const double input_real_multiplier =
    MSMIN(in_quant_arg.scale * (1 << (unsigned int)(total_signed_bits - input_integer_bits)),
          (1ll << total_signed_bits) - 1.0);
  // mult, shift
  QuantizeMultiplier(input_real_multiplier, &mult_, &shift_);
  // Calculate Input Radius
  const double max_input_rescaled = 1.0 * ((1 << input_integer_bits) - 1) *
                                    (1ll << static_cast<size_t>((total_signed_bits - input_integer_bits))) /
                                    (1ll << static_cast<size_t>(shift_));
  diff_min_ = -1.0 * static_cast<int>(std::floor(max_input_rescaled));

  const int trailing_dim = static_cast<int>(input_tensor_->shape().size()) - 1;
  const int dims_count = input_tensor_->shape().size();
  MS_CHECK_TRUE(0 <= trailing_dim && trailing_dim < dims_count, "trailing_dim should be in [0, dims_count)");
  num_rows_ = 1;
  for (int i = 0; i < dims_count; ++i) {
    num_rows_ *= (i == trailing_dim) ? 1 : input_tensor_->DimensionSize(i);
  }

  MS_CHECK_TRUE(input_tensor_->DimensionSize(trailing_dim) == output_tensor_->DimensionSize(trailing_dim),
                "input and output DimensionSize mismatch");
  row_size_ = input_tensor_->DimensionSize(trailing_dim);

  ReSize();
  return RET_OK;
}

int SoftMaxInt8Coder::DoCode(CoderContext *const context) {
  Serializer code;
  code.precision(kPrecision);

  Collect(context, {"CMSIS/NN/Include/arm_nnfunctions.h"}, {"arm_softmax_s8.c"});
  code.CodeFunction("arm_softmax_s8", input_tensor_, num_rows_, row_size_, mult_, shift_, diff_min_, output_tensor_);

  MS_LOG(INFO) << "SoftMaxInt8Coder has been called";
  context->AppendCode(code.str());
  return RET_OK;
}
REG_OPERATOR_CODER(kARM32M, kNumberTypeInt8, PrimitiveType_Softmax, CPUOpCoderCreator<SoftMaxInt8Coder>)

}  // namespace mindspore::lite::micro::cmsis
