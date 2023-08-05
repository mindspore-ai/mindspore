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

#include "mindspore/lite/tools/converter/micro/coder/opcoders/nnacl/fp32/slice_fp32_coder.h"
#include "mindspore/lite/tools/converter/micro/coder/opcoders/file_collector.h"
#include "mindspore/lite/tools/converter/micro/coder/opcoders/serializers/nnacl_serializer/nnacl_fp32_serializer.h"
#include "nnacl/slice_parameter.h"
#include "nnacl/base/slice_base.h"
#include "coder/opcoders/parallel.h"

using mindspore::schema::PrimitiveType_SliceFusion;

namespace mindspore::lite::micro::nnacl {
namespace {
constexpr int kNumInput0 = 0;
constexpr int kNumInput1 = 1;
constexpr int kNumInput2 = 2;
constexpr int kNumInputSize = 3;
}  // namespace

int SliceFP32Coder::Prepare(CoderContext *const context) {
  this->slice_struct_.base_.param_ = parameter_;

  CHECK_LESS_RETURN(input_tensors_.size(), kNumInputSize);
  CHECK_LESS_RETURN(output_tensors_.size(), 1);
  CHECK_NULL_RETURN(input_tensors_[kNumInput0]);
  CHECK_NULL_RETURN(input_tensors_[kNumInput1]);
  CHECK_NULL_RETURN(input_tensors_[kNumInput2]);
  CHECK_NULL_RETURN(output_tensor_);

  auto begin_tensor = input_tensors_[kNumInput1];
  auto size_tensor = input_tensors_[kNumInput2];
  MS_CHECK_TRUE_MSG(input_tensor_->shape().size() == static_cast<size_t>(begin_tensor->ElementsNum()), RET_ERROR,
                    "The begin tensor is invalid.");
  MS_CHECK_TRUE_MSG(input_tensor_->shape().size() == static_cast<size_t>(size_tensor->ElementsNum()), RET_ERROR,
                    "The size tensor is invalid.");
  auto begin = reinterpret_cast<int32_t *>(begin_tensor->data());
  CHECK_NULL_RETURN(begin);
  auto size = reinterpret_cast<int32_t *>(size_tensor->data());
  CHECK_NULL_RETURN(size);

  slice_struct_.data_type_size_ = static_cast<int>(lite::DataTypeSize(input_tensor_->data_type()));
  slice_struct_.param_length_ = static_cast<int>(input_tensor_->shape().size());
  if (slice_struct_.param_length_ > DIMENSION_8D) {
    MS_LOG(ERROR) << "input dimension num should <= " << DIMENSION_8D;
    return RET_ERROR;
  }
  for (int i = 0; i < slice_struct_.param_length_; ++i) {
    slice_struct_.shape_[i] = input_tensor_->DimensionSize(i);
    slice_struct_.begin_[i] = begin[i];
    slice_struct_.size_[i] = size[i] < 0 ? slice_struct_.shape_[i] - slice_struct_.begin_[i] : size[i];
    slice_struct_.end_[i] = slice_struct_.begin_[i] + slice_struct_.size_[i];
  }
  if (slice_struct_.param_length_ < DIMENSION_8D) {
    PadSliceParameterTo8D(&slice_struct_);
  }
  return RET_OK;
}

int SliceFP32Coder::DoCode(CoderContext *const context) {
  Collect(context,
          {
            "nnacl/base/slice_base.h",
            "wrapper/fp32/slice_fp32_wrapper.h",
          },
          {
            "slice_base.c",
            "slice_fp32_wrapper.c",
          });
  NNaclFp32Serializer code;
  code.CodeStruct("slice_struct", slice_struct_);

  if (support_parallel_) {
    code.CodeBaseStruct("SliceFp32Args", kRunArgs, input_tensor_, output_tensor_, "&slice_struct", thread_num_);
    code.CodeFunction(kParallelLaunch, "DoSliceRun", kRunArgsAddr, gThreadNum);
  } else {
    code.CodeFunction("DoSliceNoParallel", input_tensor_, output_tensor_, "&slice_struct",
                      slice_struct_.data_type_size_);
  }
  context->AppendCode(code.str());
  return NNACL_OK;
}
REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_SliceFusion, CPUOpCoderCreator<SliceFP32Coder>)
};  // namespace mindspore::lite::micro::nnacl
