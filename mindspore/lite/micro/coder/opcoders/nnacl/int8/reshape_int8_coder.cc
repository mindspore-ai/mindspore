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

#include "coder/opcoders/nnacl/int8/reshape_int8_coder.h"
#include <vector>
#include <string>
#include "coder/opcoders/file_collector.h"
#include "coder/log.h"
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_int8_serializer.h"

using mindspore::schema::PrimitiveType_Reshape;

namespace mindspore::lite::micro::nnacl {

int ReshapeInt8Coder::DoCode(CoderContext *const context) {
  Tensor *input = OperatorCoder::input_tensors().at(kInputIndex);
  Tensor *output = OperatorCoder::output_tensors().at(kOutputIndex);
  MS_CHECK_PTR(input);
  MS_CHECK_PTR(output);
  int elements_num = input->ElementsNum();
  std::vector<QuantArg> input_quant_args = input->quant_params();
  std::vector<QuantArg> output_quant_args = output->quant_params();

  Collect(context, {"nnacl/int8/reshape_int8.h"}, {"reshape_int8.c"});
  NNaclInt8Serializer code;
  code.precision(kPrecision);
  ReshapeQuantArg reshape_quant_arg = {
    {static_cast<float>(input_quant_args.at(0).scale), input_quant_args.at(0).zeroPoint},
    {static_cast<float>(output_quant_args.at(0).scale), output_quant_args.at(0).zeroPoint},
    INT8_MIN,
    INT8_MAX};
  code.CodeStruct("reshape_quant_arg", reshape_quant_arg);
  code.CodeFunction("Int8Reshape", input, output, elements_num, "reshape_quant_arg");

  context->AppendCode(code.str());
  return RET_OK;
}

REG_OPERATOR_CODER(kAllTargets, kNumberTypeInt8, PrimitiveType_Reshape, CPUOpCoderCreator<ReshapeInt8Coder>)
}  // namespace mindspore::lite::micro::nnacl
