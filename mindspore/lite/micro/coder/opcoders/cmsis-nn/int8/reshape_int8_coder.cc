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

#include "coder/opcoders/cmsis-nn/int8/reshape_int8_coder.h"
#include <vector>
#include <string>
#include "coder/opcoders/serializers/serializer.h"

using mindspore::schema::PrimitiveType_Reshape;

namespace mindspore::lite::micro::cmsis {

int ReshapeInt8Coder::DoCode(CoderContext *const context) {
  int elements_num = input_tensor_->ElementsNum();

  std::vector<QuantArg> input_quant_args = input_tensor_->quant_params();
  std::vector<QuantArg> output_quant_args = output_tensor_->quant_params();
  MS_CHECK_TRUE(!input_quant_args.empty(), "input quant_params is empty");
  MS_CHECK_TRUE(!output_quant_args.empty(), "output quant_params is empty");
  // in Int8Reshape, the following values are checked. then it will do a memory copy
  // para.in_args_.scale_ == para.out_args_.scale_ && para.in_args_.zp_ == para.out_args_.zp_
  MS_CHECK_TRUE((input_quant_args.at(0).scale == output_quant_args.at(0).scale &&
                 input_quant_args.at(0).zeroPoint == output_quant_args.at(0).zeroPoint),
                "the quant arg of input and output should be the same!");

  Serializer code;
  code.precision(kPrecision);

  code.CodeFunction("memcpy", output_tensor_, input_tensor_, elements_num);

  MS_LOG(INFO) << "ReshapeInt8Coder has been called";
  context->AppendCode(code.str());
  return RET_OK;
}

REG_OPERATOR_CODER(kARM32M, kNumberTypeInt8, PrimitiveType_Reshape, CPUOpCoderCreator<ReshapeInt8Coder>)

}  // namespace mindspore::lite::micro::cmsis
