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
#include "coder/opcoders/nnacl/int8/pooling_int8_coder.h"
#include <memory>
#include <vector>
#include "nnacl/int8/pooling_int8.h"
#include "coder/log.h"
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_int8_serializer.h"
#include "coder/opcoders/file_collector.h"
#include "coder/opcoders/parallel.h"
#include "coder/utils/common.h"

using mindspore::schema::PrimitiveType_AvgPoolFusion;
using mindspore::schema::PrimitiveType_MaxPoolFusion;
namespace mindspore::lite::micro::nnacl {
int PoolingInt8Coder::DoCode(CoderContext *const context) {
  // attribute
  auto *pooling_parameter = reinterpret_cast<PoolingParameter *>(parameter_);
  MS_CHECK_PTR(pooling_parameter);
  // init struct PoolingParameters
  Tensor *in_tensor = input_tensors_.at(kInputIndex);
  Tensor *out_tensor = output_tensors_.at(kOutputIndex);
  MS_CHECK_PTR(in_tensor);
  MS_CHECK_PTR(out_tensor);

  compute_param_.input_batch_ = in_tensor->Batch();
  compute_param_.input_channel_ = in_tensor->Channel();
  compute_param_.input_h_ = in_tensor->Height();
  compute_param_.input_w_ = in_tensor->Width();
  compute_param_.output_batch_ = out_tensor->Batch();
  compute_param_.output_channel_ = out_tensor->Channel();
  compute_param_.output_h_ = out_tensor->Height();
  compute_param_.output_w_ = out_tensor->Width();
  compute_param_.window_w_ = pooling_parameter->window_w_;
  compute_param_.window_h_ = pooling_parameter->window_h_;
  compute_param_.minf = INT8_MIN;
  compute_param_.maxf = INT8_MAX;

  // get quant params
  std::vector<LiteQuantParam> in_quant_args = in_tensor->quant_params();
  std::vector<LiteQuantParam> out_quant_args = out_tensor->quant_params();
  Collect(context,
          {
            "nnacl/int8/pooling_int8.h",
            "nnacl/kernel/pooling.h",
            "nnacl/errorcode.h",
          },
          {
            "pooling_int8.c",
          });
  NNaclInt8Serializer code;
  code.precision(kPrecision);
  // code op parameter
  ::QuantArg quant_arg_in = {static_cast<float>(in_quant_args.at(0).scale), in_quant_args.at(0).zeroPoint};
  ::QuantArg quant_arg_out = {static_cast<float>(out_quant_args.at(0).scale), out_quant_args.at(0).zeroPoint};

  code.CodeStruct("pooling_parameter", *pooling_parameter);
  code.CodeStruct("compute", compute_param_);
  code.CodeStruct("quant", quant_arg_in, quant_arg_out);

  if (pooling_parameter->pool_mode_ == PoolMode_MaxPool) {
    code.CodeFunction("MaxPoolingInt8", in_tensor, out_tensor, "&pooling_parameter", "&compute", "quant");
  } else {
    code.CodeFunction("AvgPoolingInt8", in_tensor, out_tensor, "&pooling_parameter", "&compute", "&quant");
  }
  context->AppendCode(code.str());
  return lite::RET_OK;
}

REG_OPERATOR_CODER(kAllTargets, kNumberTypeInt8, PrimitiveType_AvgPoolFusion, CPUOpCoderCreator<PoolingInt8Coder>)
REG_OPERATOR_CODER(kAllTargets, kNumberTypeInt8, PrimitiveType_MaxPoolFusion, CPUOpCoderCreator<PoolingInt8Coder>)
}  // namespace mindspore::lite::micro::nnacl
