/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "coder/opcoders/nnacl/fp32_grad/softmax_cross_entropy_with_logits_coder.h"
#include <string>
#include "nnacl/fp32_grad/softmax_crossentropy_parameter.h"
#include "coder/opcoders/file_collector.h"
#include "schema/inner/ops_generated.h"
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_fp32_serializer.h"

namespace mindspore::lite::micro::nnacl {
using mindspore::schema::PrimitiveType_SoftmaxCrossEntropyWithLogits;

int SoftmaxCrossEntropyWithLogitsCoder::Prepare(CoderContext *const context) {
  MS_CHECK_TRUE(input_tensor_ != nullptr, "input_tensor is nullptr.");
  size_t data_size = input_tensor_->ElementsNum();
  auto dims = input_tensor_->shape();
  auto *softmax_cross_entropy_param = reinterpret_cast<SoftmaxCrossEntropyParameter *>(parameter_);
  softmax_cross_entropy_param->n_dim_ = DIMENSION_2D;
  CHECK_LESS_RETURN(dims.size(), DIMENSION_2D);
  softmax_cross_entropy_param->number_of_classes_ = dims.at(1);
  softmax_cross_entropy_param->batch_size_ = dims.at(0);

  losses_ = reinterpret_cast<float *>(allocator_->Malloc(kNumberTypeFloat32, data_size * sizeof(float), kWorkspace));
  sum_data_ = reinterpret_cast<float *>(allocator_->Malloc(kNumberTypeFloat32, dims[0] * sizeof(float), kWorkspace));
  n_dim_ = DIMENSION_2D;
  element_size_ = data_size;
  softmax_params_.axis_ = 1;
  for (size_t i = 0; i < dims.size(); i++) {
    input_shape_[i] = dims.at(i);
  }
  return RET_OK;
}

int SoftmaxCrossEntropyWithLogitsCoder::DoCode(CoderContext *const context) {
  MS_CHECK_TRUE(input_tensors_.size() == DIMENSION_2D, "inputs size is not equal to two");
  Collect(context,
          {
            "nnacl/fp32/softmax_fp32.h",
            "nnacl/fp32_grad/softmax_cross_entropy_with_logits.h",
          },
          {
            "softmax_fp32.c",
            "softmax_cross_entropy_with_logits.c",
          });
  NNaclFp32Serializer code, init_code;
  code.CodeStruct("softmax_params", softmax_params_);
  code.CodeStruct("input_shape", input_shape_, DIMENSION_5D);

  // Get Tensor Pointer
  std::string in_str = allocator_->GetRuntimeAddr(input_tensor_);
  std::string labels_str = allocator_->GetRuntimeAddr(input_tensors_.at(1));
  std::string out_str = allocator_->GetRuntimeAddr(output_tensor_);
  std::string grad_str = "NULL";
  if (output_tensors_.size() > 1) {
    grad_str = allocator_->GetRuntimeAddr(output_tensors_.at(1));
  }
  auto *softmax_cross_entropy_param = reinterpret_cast<SoftmaxCrossEntropyParameter *>(parameter_);
  code.CodeFunction("Softmax", in_str, losses_, sum_data_, "softmax_params.axis_", n_dim_, "input_shape");
  code.CodeFunction("ForwardPostExecute", labels_str, losses_, grad_str, out_str,
                    softmax_cross_entropy_param->number_of_classes_, softmax_cross_entropy_param->batch_size_);

  context->AppendCode(code.str());
  return RET_OK;
}

REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_SoftmaxCrossEntropyWithLogits,
                   CPUOpCoderCreator<SoftmaxCrossEntropyWithLogitsCoder>)
}  // namespace mindspore::lite::micro::nnacl
