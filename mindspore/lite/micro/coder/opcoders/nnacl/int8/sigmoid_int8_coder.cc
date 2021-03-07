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

#include "coder/opcoders/nnacl/int8/sigmoid_int8_coder.h"
#include <limits>
#include <algorithm>
#include "coder/log.h"
#include "include/errorcode.h"
#include "coder/opcoders/file_collector.h"
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_int8_serializer.h"

namespace mindspore::lite::micro::nnacl {

void CalculateTableList(int8_t *table, const float input_scale, const int32_t input_zp) {
  int32_t min_value = std::numeric_limits<int8_t>::min();
  int32_t max_value = std::numeric_limits<int8_t>::max();
  const float output_scale = 1.0f / 256;
  const int32_t output_zp = -128;

  for (int i = min_value; i < max_value; ++i) {
    const float real_input_value = input_scale * (i - input_zp);
    const float sigmoid_value = 1.0f / (1.0f + std::exp(-real_input_value));
    const int32_t quantized = std::round(sigmoid_value / output_scale) + output_zp;
    auto out_value = static_cast<int8_t>(std::max(std::min(quantized, max_value), min_value));
    auto index = static_cast<uint8_t>(i);
    table[index] = out_value;
  }
}

int SigmodInt8Coder::Prepare(CoderContext *const context) {
  size_t int8_range = 256;
  table_list_ = static_cast<int8_t *>(allocator_->Malloc(kNumberTypeInt8, int8_range, kOfflinePackWeight));
  MS_CHECK_PTR(table_list_);

  const float input_scale = input_tensor_->quant_params().at(0).scale;
  const int32_t input_zp = input_tensor_->quant_params().at(0).zeroPoint;
  const float output_scale = output_tensor_->quant_params().at(0).scale;
  const int32_t output_zp = output_tensor_->quant_params().at(0).zeroPoint;
  if (output_scale != (1.0f / 256) || output_zp != -128) {
    MS_LOG(ERROR) << "Output scale is : " << output_scale << ", should be 1/256. Output zp is : " << output_zp
                  << ", should be -128.";
    return RET_ERROR;
  }
  CalculateTableList(table_list_, input_scale, input_zp);
  return RET_OK;
}

int SigmodInt8Coder::DoCode(CoderContext *const context) {
  Collect(context, {"nnacl/int8/sigmoid_int8.h"}, {"sigmoid_int8.c"});

  NNaclInt8Serializer code;

  int length = input_tensor_->ElementsNum();
  code.CodeFunction("SigmoidInt8", input_tensor_, length, output_tensor_, table_list_);

  context->AppendCode(code.str());

  return RET_OK;
}

}  // namespace mindspore::lite::micro::nnacl
