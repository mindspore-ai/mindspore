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

#include "coder/opcoders/nnacl/fp32/slice_fp32_coder.h"
#include <string>
#include "nnacl/slice_parameter.h"
#include "src/ops/slice.h"
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_fp32_serializer.h"
#include "coder/opcoders/file_collector.h"

using mindspore::schema::PrimitiveType_Slice;
namespace mindspore::lite::micro::nnacl {
int SliceFP32Coder::Prepare(CoderContext *const context) { return RET_OK; }

int SliceFP32Coder::DoCode(CoderContext *const context) {
  // generate code .h .c
  Collect(context, {"nnacl/slice_parameter.h", "nnacl/fp32/slice.h"}, {"slice.c"});

  auto param = reinterpret_cast<SliceParameter *>(parameter_);
  auto primitive_slice = reinterpret_cast<const mindspore::lite::Slice *>(OperatorCoder::primitive());
  std::vector<int> begin = primitive_slice->GetPostProcessBegin();
  std::vector<int> size = primitive_slice->GetPostProcessSize();
  std::vector<int> input_shape = input_tensor_->shape();
  NNaclFp32Serializer code;
  for (int i = 0; i < param->param_length_; i++) {
    param->shape_[i] = input_shape.at(i);
  }

  for (int i = 0; i < param->param_length_; i++) {
    param->begin_[i] = begin.at(i);
  }

  for (int i = 0; i < param->param_length_; i++) {
    int tmp_size = size.at(i);
    if (size.at(i) < 0) {
      tmp_size = input_shape.at(i) - begin.at(i);
    }
    param->end_[i] = (begin.at(i) + tmp_size);
  }

  for (int i = 0; i < param->param_length_; i++) {
    if (size.at(i) < 0) {
      param->size_[i] = (input_shape.at(i) - begin.at(i));
      continue;
    }
    param->size_[i] = size.at(i);
  }

  code.CodeStruct("slice_parameter", *param);

  // call the op function
  if (param->param_length_ < DIMENSION_4D) {
    code.CodeFunction("PadSliceParameterTo4D", "&slice_parameter");
  }
  code.CodeFunction("DoSliceNoParallel", input_tensor_, output_tensor_, "&slice_parameter");
  context->AppendCode(code.str());
  return RET_OK;
}

REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_Slice, CPUOpCoderCreator<SliceFP32Coder>)
}  // namespace mindspore::lite::micro::nnacl
