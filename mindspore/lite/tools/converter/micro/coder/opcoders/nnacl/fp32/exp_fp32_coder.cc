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
  ExpParameter *exp_param = reinterpret_cast<ExpParameter *>(parameter_);
  exp_struct_.base_.param_ = parameter_;

  float log_ = (exp_param->base_ == -1) ? 1 : logf(exp_param->base_);
  exp_struct_.in_scale_ = exp_param->scale_ * log_;
  if (exp_param->shift_ == 0) {
    exp_struct_.out_scale_ = 1;
  } else {
    if (log_ == 1) {
      exp_struct_.out_scale_ = expf(exp_param->shift_);
    } else {
      exp_struct_.out_scale_ = powf(exp_param->base_, exp_param->shift_);
    }
  }
  exp_struct_.element_num_ = input_tensor_->ElementsNum();
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
  code.CodeStruct("exp_struct", exp_struct_);
  code.CodeFunction("ExpFusionFp32", input_tensor_, output_tensor_, "&exp_struct", kDefaultTaskId);
  ctx->AppendCode(code.str());
  return RET_OK;
}
REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_ExpFusion, CPUOpCoderCreator<ExpFP32Coder>)
}  // namespace mindspore::lite::micro::nnacl
