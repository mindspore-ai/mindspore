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

#include "coder/opcoders/nnacl/int8/relux_int8_coder.h"
#include "nnacl/fp32/activation_fp32.h"
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_int8_serializer.h"
#include "coder/opcoders/file_collector.h"
#include "coder/log.h"
#include "include/errorcode.h"

namespace mindspore::lite::micro::nnacl {

int ReluxInt8Coder::Prepare(CoderContext *const context) {
  MS_CHECK_PTR(parameter_);
  type_ = (reinterpret_cast<ActivationParameter *>(parameter_))->type_;

  quant_arg_.input_arg.scale_ = input_tensor_->quant_params().front().scale;
  quant_arg_.input_arg.zp_ = input_tensor_->quant_params().front().zeroPoint;
  quant_arg_.output_arg.scale_ = output_tensor_->quant_params().front().scale;
  quant_arg_.output_arg.zp_ = output_tensor_->quant_params().front().zeroPoint;

  const double multiplier = quant_arg_.input_arg.scale_ / quant_arg_.output_arg.scale_;
  QuantizeRoundParameterWithDoublePrecision(multiplier, &quant_arg_.input_multiplier_, &quant_arg_.left_shift_,
                                            &quant_arg_.right_shift_);

  return RET_OK;
}

int ReluxInt8Coder::DoCode(CoderContext *const context) {
  Collect(context, {"nnacl/int8/relux_int8.h"}, {"relux_int8.c"});

  NNaclInt8Serializer code;

  int length = input_tensor_->ElementsNum();

  code.CodeStruct("quant_arg", quant_arg_);
  code.CodeFunction("ReluXInt8", input_tensor_, length, output_tensor_, "&quant_arg");

  context->AppendCode(code.str());

  return RET_OK;
}

}  // namespace mindspore::lite::micro::nnacl
