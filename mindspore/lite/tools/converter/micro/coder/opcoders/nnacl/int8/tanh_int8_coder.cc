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

#include "coder/opcoders/nnacl/int8/tanh_int8_coder.h"
#include <limits>
#include <algorithm>
#include "coder/log.h"
#include "include/errorcode.h"
#include "coder/opcoders/file_collector.h"
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_int8_serializer.h"
#include "nnacl/int8/tanh_int8.h"

namespace mindspore::lite::micro::nnacl {
int TanhInt8Coder::Prepare(CoderContext *const context) { return RET_OK; }

int TanhInt8Coder::DoCode(CoderContext *const context) {
  Collect(context,
          {
            "nnacl/int8/tanh_int8.h",
          },
          {"tanh_int8.c", "activation_fp32.c"});

  NNaclInt8Serializer code;
  const float input_scale = input_tensor_->quant_params().at(0).scale;
  const int32_t input_zp = input_tensor_->quant_params().at(0).zeroPoint;
  const float output_scale = output_tensor_->quant_params().at(0).scale;
  const int32_t output_zp = output_tensor_->quant_params().at(0).zeroPoint;

  code.CodeBaseStruct("TanhQuantParameter", "tanh_quant_param", input_zp, output_zp, input_scale, output_scale);
  code.CodeFunction("TanhInt8", input_tensor_, output_tensor_, input_tensor_->ElementsNum(), "&tanh_quant_param");

  context->AppendCode(code.str());

  return RET_OK;
}
}  // namespace mindspore::lite::micro::nnacl
