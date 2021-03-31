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
#include "coder/opcoders/nnacl/fp32/pooling_fp32_coder.h"
#include <cfloat>
#include <string>
#include "nnacl/fp32/pooling_fp32.h"
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_fp32_serializer.h"
#include "coder/log.h"
#include "coder/opcoders/parallel.h"
#include "coder/opcoders/file_collector.h"

using mindspore::schema::PrimitiveType_AvgPoolFusion;
using mindspore::schema::PrimitiveType_MaxPoolFusion;

namespace mindspore::lite::micro::nnacl {

int PoolingFP32Coder::DoCode(CoderContext *const context) {
  // attribute
  auto pooling_parameter = reinterpret_cast<PoolingParameter *>(parameter_);
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
  code.CodeStruct("pooling_parameter", *pooling_parameter);
  float minf = -FLT_MAX;
  float maxf = FLT_MAX;
  if (pooling_parameter->pool_mode_ == PoolMode_MaxPool) {
    Collect(context, {"nnacl/fp32/pooling_fp32.h"}, {"pooling_fp32.c"});
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

    code.CodeFunction("MaxPooling", input_tensor_, output_tensor_, "&pooling_parameter", kDefaultTaskId, minf, maxf);
  } else {
    Collect(context, {"nnacl/fp32/pooling_fp32.h"}, {"pooling_fp32.c"});
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
    code.CodeFunction("AvgPooling", input_tensor_, output_tensor_, "&pooling_parameter", kDefaultTaskId, minf, maxf);
  }

  MS_LOG(INFO) << "PoolingFp32Code has been called";
  context->AppendCode(code.str());
  return lite::RET_OK;
}

REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_AvgPoolFusion, CPUOpCoderCreator<PoolingFP32Coder>)
REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_MaxPoolFusion, CPUOpCoderCreator<PoolingFP32Coder>)
}  // namespace mindspore::lite::micro::nnacl
