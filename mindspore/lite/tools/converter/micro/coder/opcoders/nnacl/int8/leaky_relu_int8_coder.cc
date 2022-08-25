/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "coder/opcoders/nnacl/int8/leaky_relu_int8_coder.h"
#include "include/errorcode.h"
#include "coder/log.h"
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_int8_serializer.h"
#include "coder/opcoders/file_collector.h"

using mindspore::schema::PrimitiveType_LeakyRelu;

namespace mindspore::lite::micro::nnacl {
int LeakyReluInt8Coder::Prepare(CoderContext *context) {
  quant_prelu_parm_.thread_num_ = parameter_->thread_num_;
  quant_prelu_parm_.slope_ = reinterpret_cast<ActivationParameter *>(parameter_)->alpha_;

  auto *input_tensor = input_tensors_.at(kInputIndex);
  auto in_quant_args = input_tensor->quant_params();
  MS_CHECK_TRUE_MSG(!in_quant_args.empty(), RET_ERROR, "Input quant param cannot be empty.");
  quant_prelu_parm_.in_args_.scale_ = static_cast<float>(in_quant_args.front().scale);
  quant_prelu_parm_.in_args_.zp_ = in_quant_args.front().zeroPoint;

  auto *out_tensor = output_tensors_.at(kOutputIndex);
  auto out_quant_args = out_tensor->quant_params();
  MS_CHECK_TRUE_MSG(!out_quant_args.empty(), RET_ERROR, "Output quant param cannot be empty.");
  quant_prelu_parm_.out_args_.scale_ = static_cast<float>(out_quant_args.front().scale);
  quant_prelu_parm_.out_args_.zp_ = out_quant_args.front().zeroPoint;

  auto input_dim = input_tensor->shape().size();
  quant_prelu_parm_.input_dim_ = input_dim;
  MS_CHECK_GT(input_tensor->ElementsNum(), 0, RET_ERROR);
  quant_prelu_parm_.element_num = input_tensor->ElementsNum();

  return RET_OK;
}

int LeakyReluInt8Coder::DoCode(CoderContext *const context) {
  Collect(context,
          {
            "nnacl/int8/leaky_relu_int8.h",
          },
          {
            "leaky_relu_int8.c",
          });
  NNaclInt8Serializer code;
  quant_prelu_parm_.thread_num_ = 1;
  code.CodeStruct("param", quant_prelu_parm_);
  code.CodeFunction("DoLeakReluInt8", input_tensor_, output_tensor_, "&param", 0);

  context->AppendCode(code.str());
  return RET_OK;
}

REG_OPERATOR_CODER(kAllTargets, kNumberTypeInt8, PrimitiveType_LeakyRelu, CPUOpCoderCreator<LeakyReluInt8Coder>)
}  // namespace mindspore::lite::micro::nnacl
