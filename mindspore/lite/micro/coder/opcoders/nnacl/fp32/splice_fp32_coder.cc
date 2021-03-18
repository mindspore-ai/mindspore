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

#include "coder/opcoders/nnacl/fp32/splice_fp32_coder.h"
#include <string>
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_fp32_serializer.h"
#include "coder/opcoders/file_collector.h"
#include "src/common/log_adapter.h"
#include "nnacl/splice_parameter.h"
using mindspore::schema::PrimitiveType_Splice;
namespace mindspore::lite::micro::nnacl {
int SpliceFP32Coder::DoCode(CoderContext *const context) {
  auto splice_parameter = reinterpret_cast<SpliceParameter *>(parameter_);
  // to make forward_indexes nullptr
  splice_parameter->forward_indexes_ = nullptr;
  std::vector<int> src_shape = input_tensor_->shape();
  std::vector<int> dst_shape = output_tensor_->shape();
  if (src_shape.size() != dst_shape.size() || src_shape.size() != kInputSize2 || dst_shape.size() != kInputSize2) {
    MS_LOG(ERROR) << "SpliceFP32Coder src_shape size not equal to dst_shape";
    return RET_ERROR;
  }
  int src_row = src_shape.at(kWeightIndex);
  int dst_row = dst_shape.at(kWeightIndex);
  int src_col = src_shape.at(kBiasIndex);
  int dst_col = dst_shape.at(kBiasIndex);
  if (src_col * splice_parameter->context_dim_ != dst_col) {
    MS_LOG(ERROR) << "SpliceFP32Coder src_col not match to dst_col";
    return RET_ERROR;
  }
  Collect(context, {"nnacl/splice_parameter.h", "nnacl/fp32/splice_fp32.h"}, {"splice_fp32.c"});
  NNaclFp32Serializer code;
  code.CodeStruct("splice_parameter", *splice_parameter);
  code.CodeFunction("SpliceFp32", input_tensor_, src_row, src_col, "&splice_parameter", output_tensor_, dst_row,
                    dst_col);
  context->AppendCode(code.str());
  MS_LOG(DEBUG) << "SpliceFP32Coder do_code ok";
  return RET_OK;
}
REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_Splice, CPUOpCoderCreator<SpliceFP32Coder>)
}  // namespace mindspore::lite::micro::nnacl
