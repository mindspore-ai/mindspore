/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
  MS_CHECK_PTR(input0);
  MS_CHECK_PTR(input1);
  MS_CHECK_TRUE_MSG(input1->data_type() == kNumberTypeInt32 || input1->data_type() == kNumberTypeInt, RET_ERROR,
                    "index's data-type is not int32");
  // generate code .h .c
  Collect(context,
          {
            "nnacl/base/gather_base.h",
          },
          {
            "gather_base.c",
          });

  NNaclFp32Serializer code;
  std::vector<int> in_shape = input0->shape();
  int in_rank = static_cast<int>(in_shape.size());
  MS_CHECK_PTR(parameter_);
  int axis = *(reinterpret_cast<int *>(input_tensors_.at(THIRD_INPUT)->data()));
  MS_CHECK_TRUE(static_cast<int>(in_shape.size()) >= axis, "invalid axis in gather parameter");
  const int limit = in_shape.at(axis);

  int outer_size = 1, inner_size = 1;
  for (int i = 0; i < axis; ++i) {
    outer_size *= in_shape.at(i);
  }
  for (int i = axis + 1; i < in_rank; ++i) {
    inner_size *= in_shape.at(i);
  }
  auto data_size = static_cast<int>(lite::DataTypeSize(input0->data_type()));
  int64_t byte_inner_size = inner_size * data_size;
  int indices_element_size = input1->ElementsNum();
  int64_t byte_out_stride = indices_element_size * byte_inner_size;
  MS_CHECK_TRUE(thread_num_ > 0, "thread_num_ <= 0");
  int stride = UP_DIV(outer_size, thread_num_);
  int start = stride * kDefaultTaskId;
  int count = MSMIN(stride, outer_size - stride * kDefaultTaskId);
  std::string input0_data = MemoryAllocator::GetInstance()->GetRuntimeAddr(input0, true);
  if (input0_data.empty()) {
    MS_LOG(ERROR) << "pointer is not allocated by the allocator";
    return RET_ERROR;
  }
  std::string input1_data = MemoryAllocator::GetInstance()->GetRuntimeAddr(input1, true);
  if (input1_data.empty()) {
    MS_LOG(ERROR) << "pointer is not allocated by the allocator";
    return RET_ERROR;
  }
  std::string output_data = MemoryAllocator::GetInstance()->GetRuntimeAddr(output_tensor_, true);
  if (output_data.empty()) {
    MS_LOG(ERROR) << "pointer is not allocated by the allocator";
    return RET_ERROR;
  }
  code << "\t\tconst int8_t *int8_in = (const int8_t *)" << input0_data << ";\n";
  code << "\t\tint8_in += " << std::to_string(start * limit * byte_inner_size) << ";\n";
  code << "\t\tconst int *index_data = (const int *)" << input1_data << ";\n";
  code << "\t\tint8_t *int8_out = (int8_t *)" << output_data << ";\n";
  code << "\t\tint8_out += " << std::to_string(start * byte_out_stride) << ";\n";
  // call the op function
  code.CodeFunction("Gather", "int8_in", count, byte_inner_size, limit, "index_data", indices_element_size, "int8_out",
                    byte_out_stride);
  context->AppendCode(code.str());
  return RET_OK;
}

REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_Gather, CPUOpCoderCreator<GatherFP32Coder>)
REG_OPERATOR_CODER(kAllTargets, kNumberTypeInt32, PrimitiveType_Gather, CPUOpCoderCreator<GatherFP32Coder>)
}  // namespace mindspore::lite::micro::nnacl
