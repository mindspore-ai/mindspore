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

#include "coder/opcoders/nnacl/fp32/exp_fp32_coder.h"
#include <cmath>
#include "coder/opcoders/file_collector.h"
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_fp32_serializer.h"
#include "coder/opcoders/parallel.h"
using mindspore::schema::PrimitiveType_ExpFusion;

namespace mindspore::lite::micro::nnacl {
int ExpFP32Coder::Prepare(CoderContext *context) {
  exp_parameter_ = reinterpret_cast<ExpParameter *>(parameter_);
  float log_ = (exp_parameter_->base_ == -1) ? 1 : logf(exp_parameter_->base_);
  exp_parameter_->in_scale_ = exp_parameter_->scale_ * log_;
  if (exp_parameter_->shift_ == 0) {
    exp_parameter_->out_scale_ = 1;
  } else {
    if (log_ == 1) {
      exp_parameter_->out_scale_ = expf(exp_parameter_->shift_);
    } else {
      exp_parameter_->out_scale_ = powf(exp_parameter_->base_, exp_parameter_->shift_);
    }
  }
  exp_parameter_->element_num_ = input_tensor_->ElementsNum();
  return RET_OK;
}

int ExpFP32Coder::DoCode(CoderContext *ctx) {
  Collect(ctx,
          {
            "nnacl/fp32/exp_fp32.h",
          },
          {
            "exp_fp32.c",
          });
  nnacl::NNaclFp32Serializer code;
  code.CodeStruct("exp_parameter", *exp_parameter_);
  code.CodeFunction("ExpFusionFp32", input_tensor_, output_tensor_, "(ExpParameter *)&exp_parameter", kDefaultTaskId);
  ctx->AppendCode(code.str());
  return RET_OK;
}
REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_ExpFusion, CPUOpCoderCreator<ExpFP32Coder>)
}  // namespace mindspore::lite::micro::nnacl
