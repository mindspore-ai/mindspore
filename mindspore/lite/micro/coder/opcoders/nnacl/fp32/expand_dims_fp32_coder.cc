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

#include "coder/opcoders/nnacl/fp32/expand_dims_fp32_coder.h"
#include <string>
#include "coder/opcoders/file_collector.h"
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_fp32_serializer.h"

using mindspore::schema::PrimitiveType_ExpandDims;

namespace mindspore::lite::micro::nnacl {
int ExpandDimsFP32Coder::Prepare(CoderContext *const context) { return ReSize(); }

int ExpandDimsFP32Coder::ReSize() {
  data_size_ = input_tensor_->ElementsNum();
  thread_sz_count_ = MSMIN(thread_num_, static_cast<int>(data_size_));
  MS_CHECK_TRUE(thread_sz_count_ > 0, "thread_sz_count_ is less or equal to 0");
  thread_sz_stride_ = UP_DIV(data_size_, thread_sz_count_);
  return RET_OK;
}

int ExpandDimsFP32Coder::DoCode(CoderContext *const context) {
  // generate code .h .c
  Collect(context, {"nnacl/fp32/expandDims.h"}, {"nnacl/fp32/expandDims.c"});
  NNaclFp32Serializer code;
  int task_id = 0;
  size_t size = MSMIN(thread_sz_stride_, static_cast<int>(data_size_ - task_id * thread_sz_stride_));
  if (!size) {
    return RET_OK;
  }
  code.CodeFunction("ExpandDims", input_tensor_, output_tensor_, size * sizeof(float));
  context->AppendCode(code.str());
  return RET_OK;
}

REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_ExpandDims, CPUOpCoderCreator<ExpandDimsFP32Coder>)

REG_OPERATOR_CODER(kAllTargets, kNumberTypeInt32, PrimitiveType_ExpandDims, CPUOpCoderCreator<ExpandDimsFP32Coder>)
}  // namespace mindspore::lite::micro::nnacl
