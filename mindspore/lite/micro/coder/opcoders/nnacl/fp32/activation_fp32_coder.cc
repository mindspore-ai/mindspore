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
#include "micro/coder/opcoders/nnacl/fp32/activation_fp32_coder.h"
#include <string>
#include "nnacl/fp32/activation_fp32.h"
#include "nnacl/op_base.h"
#include "micro/coder/opcoders/serializers/nnacl_serializer/nnacl_fp32_serializer.h"
#include "micro/coder/opcoders/file_collector.h"

using mindspore::schema::PrimitiveType_Activation;

namespace mindspore::lite::micro::nnacl {

int ActivationFP32Coder::DoCode(CoderContext *const context) {
  // attribute
  auto *activation_parameter = reinterpret_cast<ActivationParameter *>(parameter_);
  int task_id = 0;
  int length = input_tensor_->ElementsNum();
  MS_CHECK_TRUE(thread_num_ > 0, "thread_num_ <= 0");
  int stride = UP_DIV(length, thread_num_);
  int count = MSMIN(stride, length - stride * task_id);

  if (activation_parameter->type_ == schema::ActivationType_SIGMOID) {
    Collect(context, {"runtime/kernel/fp32/sigmoid.h"}, {"sigmoid.c"});
  } else {
    Collect(context, {"nnacl/fp32/activation.h"}, {"activation.c"});
  }
  NNaclFp32Serializer code;
  switch (activation_parameter->type_) {
    case schema::ActivationType_RELU:
      code.CodeFunction("Fp32Relu", input_tensor_, count, output_tensor_);
      break;
    case schema::ActivationType_RELU6:
      code.CodeFunction("Fp32Relu6", input_tensor_, count, output_tensor_);
      break;
    case schema::ActivationType_LEAKY_RELU:
      code.CodeFunction("LRelu", input_tensor_, count, output_tensor_, activation_parameter->alpha_);
      break;
    case schema::ActivationType_SIGMOID:
      code.CodeFunction("Sigmoid", input_tensor_, count, output_tensor_);
      break;
    case schema::ActivationType_TANH:
      code.CodeFunction("Tanh", input_tensor_, count, output_tensor_);
      break;
    case schema::ActivationType_HSWISH:
      code.CodeFunction("HSwish", input_tensor_, count, output_tensor_);
      break;
    default:
      MS_LOG(ERROR) << "Activation type error";
      return RET_ERROR;
  }
  MS_LOG(DEBUG) << "ActivationFP32Code has been called";
  context->AppendCode(code.str());
  return lite::RET_OK;
}

REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_Activation, CPUOpCoderCreator<ActivationFP32Coder>)
}  // namespace mindspore::lite::micro::nnacl
