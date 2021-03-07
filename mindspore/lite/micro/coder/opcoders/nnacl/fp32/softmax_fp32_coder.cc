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

#include "coder/opcoders/nnacl/fp32/softmax_fp32_coder.h"
#include <string>
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_fp32_serializer.h"
#include "schema/inner/ops_generated.h"
#include "coder/opcoders/file_collector.h"

using mindspore::schema::PrimitiveType_Softmax;

namespace mindspore::lite::micro::nnacl {

int SoftMaxFP32Coder::Prepare(CoderContext *const context) {
  SoftmaxBaseCoder::Init();
  // malloc tmp buffer
  int n_dim = softmax_param_->n_dim_;
  int32_t axis = softmax_param_->axis_;
  if (axis == -1) {
    softmax_param_->axis_ += n_dim;
    axis = softmax_param_->axis_;
  }
  auto in_shape = input_tensor_->shape();
  int out_plane_size = 1;
  for (int i = 0; i < axis; ++i) {
    out_plane_size *= in_shape.at(i);
  }
  int in_plane_size = 1;
  for (int i = axis + 1; i < n_dim; i++) {
    in_plane_size *= in_shape.at(i);
  }
  sum_data_size_ = out_plane_size * in_plane_size * sizeof(float);
  sum_data_ = static_cast<float *>(allocator_->Malloc(kNumberTypeFloat, sum_data_size_, kWorkspace));
  return RET_OK;
}

int SoftMaxFP32Coder::DoCode(CoderContext *const context) {
  Collect(context, {"nnacl/fp32/softmax_fp32.h"}, {"softmax_fp32.c", "exp_fp32.c"});
  NNaclFp32Serializer code;
  code.CodeStruct("softmax_parameter", *softmax_param_);
  code.CodeFunction("memset", sum_data_, "0", sum_data_size_);
  code.CodeFunction("Softmax", input_tensor_, output_tensor_, sum_data_, "&softmax_parameter");
  context->AppendCode(code.str());

  return RET_OK;
}

REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_Softmax, CPUOpCoderCreator<SoftMaxFP32Coder>)

}  // namespace mindspore::lite::micro::nnacl
