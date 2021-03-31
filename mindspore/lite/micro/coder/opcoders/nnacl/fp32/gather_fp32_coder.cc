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

#include "coder/opcoders/nnacl/fp32/gather_fp32_coder.h"
#include <string>
#include "nnacl/gather_parameter.h"
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_fp32_serializer.h"
#include "coder/log.h"
#include "coder/opcoders/parallel.h"
#include "coder/opcoders/file_collector.h"

using mindspore::schema::PrimitiveType_Gather;

namespace mindspore::lite::micro::nnacl {

int GatherFP32Coder::Prepare(CoderContext *const context) { return RET_OK; }

int GatherFP32Coder::DoCode(CoderContext *context) {
  Tensor *input0 = input_tensors_.at(0);
  Tensor *input1 = input_tensors_.at(1);

  // generate code .h .c
  Collect(context, {"nnacl/fp32/gather.h"}, {"nnacl/fp32/gather.c"});

  NNaclFp32Serializer code;
  std::vector<int> in_shape = input0->shape();
  int in_rank = in_shape.size();
  int indices_element_size = input1->ElementsNum();
  int axis = (reinterpret_cast<GatherParameter *>(parameter_))->axis_;
  MS_CHECK_TRUE(static_cast<int>(in_shape.size()) >= axis, "invalid axis in gather parameter");
  const int limit = in_shape.at(axis);

  int outer_size = 1, inner_size = 1;
  for (int i = 0; i < axis; ++i) {
    outer_size *= in_shape.at(i);
  }
  for (int i = axis + 1; i < in_rank; ++i) {
    inner_size *= in_shape.at(i);
  }
  MS_CHECK_TRUE(thread_num_ > 0, "thread_num_ <= 0");
  int stride = UP_DIV(outer_size, thread_num_);
  int count = MSMIN(stride, outer_size - stride * kDefaultTaskId);

  // call the op function
  if (input0->data_type() == kNumberTypeInt32) {
    code.CodeFunction("GatherInt32", input0, count, inner_size, limit, input1, indices_element_size, output_tensor_);
  } else {
    code.CodeFunction("Gather", input0, count, inner_size, limit, input1, indices_element_size, output_tensor_);
  }
  context->AppendCode(code.str());
  return RET_OK;
}

REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_Gather, CPUOpCoderCreator<GatherFP32Coder>)

}  // namespace mindspore::lite::micro::nnacl
