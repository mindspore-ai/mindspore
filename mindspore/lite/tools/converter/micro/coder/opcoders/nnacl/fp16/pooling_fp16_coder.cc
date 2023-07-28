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
#include "coder/opcoders/nnacl/fp16/pooling_fp16_coder.h"
#include <cfloat>
#include <string>
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_fp32_serializer.h"
#include "coder/log.h"
#include "coder/opcoders/parallel.h"
#include "coder/opcoders/file_collector.h"

using mindspore::schema::PrimitiveType_AvgPoolFusion;
using mindspore::schema::PrimitiveType_MaxPoolFusion;

namespace mindspore::lite::micro::nnacl {
int PoolingFP16Coder::Prepare(CoderContext *const context) {
  if (input_tensor_->data_type() != kNumberTypeFloat16 || output_tensor_->data_type() != kNumberTypeFloat16) {
    MS_LOG(ERROR) << "Tensor data type is invalid";
    return lite::RET_INPUT_PARAM_INVALID;
  }
  return RET_OK;
}

int PoolingFP16Coder::DoCode(CoderContext *const context) {
  // attribute
  auto pooling_parameter = reinterpret_cast<PoolingParameter *>(parameter_);
  MS_CHECK_PTR(pooling_parameter);
  // init struct PoolingParameters
  compute_param_.input_batch_ = input_tensor_->Batch();
  compute_param_.input_channel_ = input_tensor_->Channel();
  compute_param_.input_h_ = input_tensor_->Height();
  compute_param_.input_w_ = input_tensor_->Width();
  compute_param_.output_batch_ = output_tensor_->Batch();
  compute_param_.output_channel_ = output_tensor_->Channel();
  compute_param_.output_h_ = output_tensor_->Height();
  compute_param_.output_w_ = output_tensor_->Width();
  compute_param_.window_h_ = pooling_parameter->window_h_;
  compute_param_.window_w_ = pooling_parameter->window_w_;

  NNaclFp32Serializer code;
  std::string param_name = "pooling_parameter";
  code.CodeStruct(param_name, *pooling_parameter);
  float minf = -FLT16_MAX;
  float maxf = FLT16_MAX;
  Collect(context,
          {
            "nnacl/fp16/pooling_fp16.h",
            "nnacl/kernel/pooling.h",
          },
          {
            "pooling_fp16.c",
            "pooling.c",
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
  compute_param_.minf = minf;
  compute_param_.maxf = maxf;
  code.CodeStruct("pooling_args", compute_param_);

  if (pooling_parameter->pool_mode_ == PoolMode_MaxPool) {
    code.CodeFunction("MaxPoolingFp16", input_tensor_, output_tensor_, "&pooling_parameter", "&pooling_args",
                      kDefaultTaskId, parameter_->thread_num_);
  } else if (pooling_parameter->pool_mode_ == PoolMode_AvgPool) {
    code.CodeFunction("AvgPoolingFp16", input_tensor_, output_tensor_, "&pooling_parameter", "&pooling_args",
                      kDefaultTaskId, parameter_->thread_num_);
  } else {
    MS_LOG(ERROR) << "Unsupported pooling mode.";
    return lite::RET_ERROR;
  }

  MS_LOG(INFO) << "PoolingFp16Code has been called";
  context->AppendCode(code.str());
  return lite::RET_OK;
}

REG_OPERATOR_CODER(kARM32, kNumberTypeFloat16, PrimitiveType_AvgPoolFusion, CPUOpCoderCreator<PoolingFP16Coder>)
REG_OPERATOR_CODER(kARM64, kNumberTypeFloat16, PrimitiveType_AvgPoolFusion, CPUOpCoderCreator<PoolingFP16Coder>)
REG_OPERATOR_CODER(kARM32, kNumberTypeFloat16, PrimitiveType_MaxPoolFusion, CPUOpCoderCreator<PoolingFP16Coder>)
REG_OPERATOR_CODER(kARM64, kNumberTypeFloat16, PrimitiveType_MaxPoolFusion, CPUOpCoderCreator<PoolingFP16Coder>)
}  // namespace mindspore::lite::micro::nnacl
