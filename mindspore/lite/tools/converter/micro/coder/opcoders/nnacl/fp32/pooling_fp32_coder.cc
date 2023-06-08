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
  int thread_num = pooling_parameter->op_parameter_.thread_num_;

  // init struct PoolingComputeParam
  compute_param_.input_batch_ = input_tensor_->Batch();
  compute_param_.input_channel_ = input_tensor_->Channel();
  compute_param_.input_h_ = input_tensor_->Height();
  compute_param_.input_w_ = input_tensor_->Width();
  compute_param_.output_batch_ = output_tensor_->Batch();
  compute_param_.output_channel_ = output_tensor_->Channel();
  compute_param_.output_h_ = output_tensor_->Height();
  compute_param_.output_w_ = output_tensor_->Width();

  NNaclFp32Serializer code;
  std::string param_name = "pooling_parameter";
  code.CodeStruct(param_name, *pooling_parameter);
  float minf = -FLT_MAX;
  float maxf = FLT_MAX;
  Collect(context,
          {
            "wrapper/fp32/pooling_fp32_wrapper.h",
            "nnacl/kernel/pooling.h",
            "nnacl/fp32/pooling_fp32.h",
          },
          {
            "pooling_fp32_wrapper.c",
            "pooling_fp32.c",
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
  compute_param_.window_w_ = pooling_parameter->window_w_;
  compute_param_.window_h_ = pooling_parameter->window_h_;
  compute_param_.minf = minf;
  compute_param_.maxf = maxf;
  code.CodeStruct("pooling_args", compute_param_);

  if (pooling_parameter->pool_mode_ == PoolMode_MaxPool) {
    if (!support_parallel_) {
      code.CodeFunction("MaxPooling", input_tensor_, output_tensor_, "&pooling_parameter", "&pooling_args",
                        kDefaultTaskId, kDefaultThreadNum);
    } else {
      code.CodeBaseStruct("PoolingFp32Args", kRunArgs, input_tensor_, output_tensor_, "&pooling_parameter",
                          "&pooling_args");
      code.CodeFunction(kParallelLaunch, "DoMaxPooling", kRunArgsAddr, thread_num);
    }
  } else {
    code.CodeFunction("AvgPooling", input_tensor_, output_tensor_, "&pooling_parameter", "&pooling_args",
                      kDefaultTaskId, kDefaultThreadNum);
  }

  MS_LOG(INFO) << "PoolingFp32Code has been called";
  context->AppendCode(code.str());
  return lite::RET_OK;
}

REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_AvgPoolFusion, CPUOpCoderCreator<PoolingFP32Coder>)
REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_MaxPoolFusion, CPUOpCoderCreator<PoolingFP32Coder>)
}  // namespace mindspore::lite::micro::nnacl
