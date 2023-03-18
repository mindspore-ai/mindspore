/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "coder/opcoders/nnacl/fp16/avg_pooling_fp16_coder.h"
#include <cfloat>
#include <string>
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_fp32_serializer.h"
#include "coder/log.h"
#include "coder/opcoders/parallel.h"
#include "coder/opcoders/file_collector.h"

using mindspore::schema::PrimitiveType_AvgPoolFusion;

namespace mindspore::lite::micro::nnacl {
int PoolingFP16Coder::Prepare(CoderContext *const context) {
  if (input_tensor_->data_type() != kNumberTypeFloat16) {
    MS_LOG(INFO) << "Input tensor data type is invalid";
    return RET_INPUT_PARAM_INVALID;
  }
  return RET_OK;
}

int PoolingFP16Coder::DoCode(CoderContext *const context) {
  // attribute
  auto pooling_parameter = reinterpret_cast<PoolingParameter *>(parameter_);
  MS_CHECK_PTR(pooling_parameter);
  // init struct PoolingParameters
  pooling_parameter->input_batch_ = input_tensor_->Batch();
  pooling_parameter->input_channel_ = input_tensor_->Channel();
  pooling_parameter->input_h_ = input_tensor_->Height();
  pooling_parameter->input_w_ = input_tensor_->Width();
  pooling_parameter->output_batch_ = output_tensor_->Batch();
  pooling_parameter->output_channel_ = output_tensor_->Channel();
  pooling_parameter->output_h_ = output_tensor_->Height();
  pooling_parameter->output_w_ = output_tensor_->Width();

  pooling_parameter->thread_num_ = pooling_parameter->op_parameter_.thread_num_;

  NNaclFp32Serializer code;
  std::string param_name = "pooling_parameter";
  code.CodeStruct(param_name, *pooling_parameter);
  float minf = -FLT16_MAX;
  float maxf = FLT16_MAX;
  Collect(context,
          {
            "nnacl/fp16/pooling_fp16.h",
          },
          {
            "pooling_fp16.c",
          });
  switch (pooling_parameter->act_type_) {
    case ActType_Relu: {
      minf = 0.f;
      break;
    }
    case ActType_Relu6: {
      minf = 0.f;
      maxf = 6.f;
      break;
    }
    default: {
      MS_LOG(INFO) << "no actype";
      break;
    }
  }
  code.CodeFunction("AvgPoolingFp16", input_tensor_, output_tensor_, "&pooling_parameter", kDefaultTaskId, minf, maxf);

  MS_LOG(INFO) << "PoolingFp16Code has been called";
  context->AppendCode(code.str());
  return lite::RET_OK;
}

REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat16, PrimitiveType_AvgPoolFusion, CPUOpCoderCreator<PoolingFP16Coder>)
}  // namespace mindspore::lite::micro::nnacl
