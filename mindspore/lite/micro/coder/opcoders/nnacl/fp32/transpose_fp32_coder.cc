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

#include "coder/opcoders/nnacl/fp32/transpose_fp32_coder.h"
#include <vector>
#include <string>
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_fp32_serializer.h"
#include "coder/opcoders/file_collector.h"
#include "coder/opcoders/parallel.h"

using mindspore::schema::PrimitiveType_Transpose;
namespace mindspore::lite::micro::nnacl {

int TransposeFp32Coder::Resize() {
  num_unit_ = static_cast<int>(input_tensor_->shape().at(transpose_parameter_->perm_[kNHWC_H]));
  thread_h_num_ = MSMIN(thread_num_, num_unit_);
  MS_CHECK_TRUE(thread_h_num_ > 0, "thread_h_num_ <= 0");
  thread_h_stride_ = UP_DIV(num_unit_, thread_h_num_);

  std::vector<int> in_shape = input_tensor_->shape();
  std::vector<int> out_shape = output_tensor_->shape();
  transpose_parameter_->strides_[transpose_parameter_->num_axes_ - 1] = 1;
  transpose_parameter_->out_strides_[transpose_parameter_->num_axes_ - 1] = 1;
  transpose_parameter_->data_size_ = static_cast<int>(input_tensor_->Size());
  for (int i = transpose_parameter_->num_axes_ - 2; i >= 0; i--) {
    transpose_parameter_->strides_[i] = in_shape.at(i + 1) * transpose_parameter_->strides_[i + 1];
    transpose_parameter_->out_strides_[i] = out_shape.at(i + 1) * transpose_parameter_->out_strides_[i + 1];
  }
  MS_CHECK_TRUE(in_shape.size() > 0, "invalid shape size");
  MS_CHECK_TRUE(out_shape.size() > 0, "invalid shape size");
  auto in_shape_data_size = static_cast<size_t>(in_shape.size() * sizeof(int));
  auto out_shape_data_size = static_cast<size_t>(out_shape.size() * sizeof(int));
  in_shape_ = reinterpret_cast<int *>(allocator_->Malloc(kNumberTypeInt, in_shape_data_size, kOfflinePackWeight));
  MS_CHECK_PTR(in_shape_);
  out_shape_ =
    reinterpret_cast<int *>(allocator_->Malloc(kNumberTypeInt, out_shape.size() * sizeof(int), kOfflinePackWeight));
  MS_CHECK_PTR(out_shape_);
  MS_CHECK_RET_CODE(memcpy_s(in_shape_, in_shape_data_size, in_shape.data(), in_shape_data_size), "memcpy failed");
  MS_CHECK_RET_CODE(memcpy_s(out_shape_, out_shape_data_size, out_shape.data(), out_shape_data_size), "memcpy failed");
  return RET_OK;
}

int TransposeFp32Coder::Init() {
  transpose_parameter_ = reinterpret_cast<TransposeParameter *>(parameter_);
  MS_CHECK_PTR(transpose_parameter_);
  return Resize();
}

int TransposeFp32Coder::Prepare(CoderContext *const context) {
  MS_CHECK_RET_CODE(Init(), "init failed");
  int out_dims = static_cast<int>(output_tensor_->shape().size());
  auto out_data_dims_size = static_cast<size_t>(out_dims * thread_h_num_ * sizeof(int));
  if (out_dims > MAX_TRANSPOSE_DIM_SIZE) {
    dim_size_ = reinterpret_cast<int *>(allocator_->Malloc(kNumberTypeInt, out_data_dims_size, kWorkspace));
    MS_CHECK_PTR(dim_size_);
    position_ = reinterpret_cast<int *>(allocator_->Malloc(kNumberTypeInt, out_data_dims_size, kWorkspace));
    MS_CHECK_PTR(position_);
  }
  return RET_OK;
}

int TransposeFp32Coder::DoCode(CoderContext *const context) {
  int num_unit_thread = MSMIN(thread_h_stride_, num_unit_ - kDefaultTaskId * thread_h_stride_);
  if (num_unit_thread <= 0) {
    return RET_OK;
  }

  Collect(context, {"nnacl/transpose.h", "nnacl/fp32/transpose.h", "nnacl/errorcode.h"}, {"transpose.c"});

  NNaclFp32Serializer code;
  code.CodeStruct("transpose_parameter", *transpose_parameter_);

  code.CodeFunction("DoTransposeFp32", input_tensor_, output_tensor_, in_shape_, out_shape_,
                    "(TransposeParameter *)&transpose_parameter", kDefaultTaskId, num_unit_thread, dim_size_,
                    position_);

  context->AppendCode(code.str());
  return RET_OK;
}

REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_Transpose, CPUOpCoderCreator<TransposeFp32Coder>)
}  // namespace mindspore::lite::micro::nnacl
