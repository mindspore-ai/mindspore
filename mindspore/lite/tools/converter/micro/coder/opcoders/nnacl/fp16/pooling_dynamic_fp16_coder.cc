/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include "coder/opcoders/nnacl/fp16/pooling_dynamic_fp16_coder.h"
#include <cfloat>
#include <string>
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_fp32_serializer.h"
#include "coder/log.h"
#include "coder/opcoders/parallel.h"
#include "coder/opcoders/file_collector.h"
#include "coder/utils/coder_utils.h"

using mindspore::schema::PrimitiveType_AvgPoolFusion;
using mindspore::schema::PrimitiveType_MaxPoolFusion;

namespace mindspore::lite::micro::nnacl {
int PoolingDynamicFP16Coder::Prepare(CoderContext *const context) {
  if (input_tensor_->data_type() != kNumberTypeFloat16 || output_tensor_->data_type() != kNumberTypeFloat16) {
    MS_LOG(ERROR) << "Tensor data type is invalid";
    return lite::RET_INPUT_PARAM_INVALID;
  }
  param_ = reinterpret_cast<PoolingParameter *>(parameter_);
  MS_CHECK_PTR(param_);
  MS_CHECK_TRUE_MSG(memset_s(&compute_, sizeof(compute_), 0, sizeof(compute_)) == EOK, RET_ERROR, "memset_s failed.");
  param_->op_parameter_.thread_num_ = 1;
  dynamic_param_.input_batch_ = shape_info_container_->GetTemplateShape(input_tensor_)[0];
  compute_.input_channel_ = input_tensor_->Channel();
  compute_.input_h_ = input_tensor_->Height();
  compute_.input_w_ = input_tensor_->Width();
  dynamic_param_.output_batch_ = shape_info_container_->GetTemplateShape(output_tensor_)[0];
  compute_.output_channel_ = output_tensor_->Channel();
  compute_.output_h_ = output_tensor_->Height();
  compute_.output_w_ = output_tensor_->Width();
  compute_.window_h_ = param_->window_h_;
  compute_.window_w_ = param_->window_w_;
  if (param_->global_) {
    compute_.window_h_ = compute_.input_h_;
    compute_.window_w_ = compute_.input_w_;
  }
  float minf = -FLT16_MAX;
  float maxf = FLT16_MAX;
  if (param_->act_type_ == ActType_Relu) {
    minf = 0.f;
  } else if (param_->act_type_ == ActType_Relu6) {
    minf = 0.f;
    maxf = 6.f;
  }
  compute_.minf = minf;
  compute_.maxf = maxf;
  return RET_OK;
}

int PoolingDynamicFP16Coder::DoCode(CoderContext *const context) {
  Collect(context,
          {
            "nnacl/fp16/pooling_fp16.h",
          },
          {
            "pooling_fp16.c",
          });
  NNaclFp32Serializer code;
  code.CodeStruct("pooling_parameter", *param_);
  code.CodeStruct("pooling_compute", compute_, dynamic_param_);

  auto input_data =
    "(float16_t *)(" + GetTensorAddr(input_tensor_, input_tensor_->IsConst(), dynamic_mem_manager_, allocator_) + ")";
  auto output_data =
    "(float16_t *)(" + GetTensorAddr(output_tensor_, output_tensor_->IsConst(), dynamic_mem_manager_, allocator_) + ")";
  if (param_->pool_mode_ == PoolMode_MaxPool) {
    code.CodeFunction("MaxPoolingFp16", input_data, output_data, "&pooling_parameter", "&pooling_compute",
                      kDefaultTaskId, param_->op_parameter_.thread_num_);
  } else if (param_->pool_mode_ == PoolMode_AvgPool) {
    code.CodeFunction("AvgPoolingFp16", input_data, output_data, "&pooling_parameter", "&pooling_compute",
                      kDefaultTaskId, param_->op_parameter_.thread_num_);
  } else {
    MS_LOG(ERROR) << "Unsupported pooling mode.";
    return lite::RET_ERROR;
  }
  context->AppendCode(code.str());
  return lite::RET_OK;
}

REG_DYNAMIC_OPERATOR_CODER(kARM32, kNumberTypeFloat16, PrimitiveType_AvgPoolFusion,
                           CPUOpCoderCreator<PoolingDynamicFP16Coder>)
REG_DYNAMIC_OPERATOR_CODER(kARM64, kNumberTypeFloat16, PrimitiveType_AvgPoolFusion,
                           CPUOpCoderCreator<PoolingDynamicFP16Coder>)
REG_DYNAMIC_OPERATOR_CODER(kARM32, kNumberTypeFloat16, PrimitiveType_MaxPoolFusion,
                           CPUOpCoderCreator<PoolingDynamicFP16Coder>)
REG_DYNAMIC_OPERATOR_CODER(kARM64, kNumberTypeFloat16, PrimitiveType_MaxPoolFusion,
                           CPUOpCoderCreator<PoolingDynamicFP16Coder>)
}  // namespace mindspore::lite::micro::nnacl
