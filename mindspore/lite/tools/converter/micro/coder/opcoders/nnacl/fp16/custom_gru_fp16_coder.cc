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
#include "coder/opcoders/nnacl/fp16/custom_gru_fp16_coder.h"
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_fp32_serializer.h"
#include "coder/opcoders/file_collector.h"

using mindspore::schema::PrimitiveType_Custom;

namespace mindspore::lite::micro::nnacl {
void CustomGruFP16Coder::InitNnaclFile(CoderContext *const context) {
  Collect(context, {"nnacl/fp16/custom_gru_fp16.h"},
          {"custom_gru_fp16.c", "pack_fp16.c", "matmul_fp16.c", "arithmetic_fp16.c", "activation_fp16.c"});
}

void CustomGruFP16Coder::InitPackMatrixB(NNaclFp32Serializer *init_code, const std::string &src, const std::string &dst,
                                         int row, int col) {
  init_code->CodeFunction("RowMajor2Col8MajorFp16", src, dst, row, col, false);
}

REG_OPERATOR_CODER(kARM64, kNumberTypeFloat16, PrimitiveType_Custom, CPUOpCoderCreator<CustomGruFP16Coder>)
}  // namespace mindspore::lite::micro::nnacl
