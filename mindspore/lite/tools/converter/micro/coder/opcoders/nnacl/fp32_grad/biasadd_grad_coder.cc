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

#include "coder/opcoders/nnacl/fp32_grad/biasadd_grad_coder.h"
#include <string>
#include "schema/inner/ops_generated.h"
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_fp32_serializer.h"

namespace mindspore::lite::micro::nnacl {
using mindspore::schema::PrimitiveType_BiasAddGrad;

int BiasAddGradCoder::Prepare(CoderContext *const context) {
  auto dims = input_tensor_->shape();
  auto *bias_param = reinterpret_cast<ArithmeticParameter *>(parameter_);
  bias_param->ndim_ = dims.size();
  for (unsigned int i = 0; i < bias_param->ndim_; i++) {
    bias_param->in_shape0_[i] = dims[i];
    bias_param->out_shape_[i] = 1;  // 1 dimension for N,H,W,
  }
  bias_param->out_shape_[bias_param->ndim_ - 1] = dims[bias_param->ndim_ - 1];
  for (auto i = bias_param->ndim_; i < DIMENSION_4D; i++) {
    bias_param->in_shape0_[i] = 0;
    bias_param->out_shape_[i] = 0;
  }
  return RET_OK;
}

int BiasAddGradCoder::DoCode(CoderContext *const context) {
  auto *bias_param = reinterpret_cast<ArithmeticParameter *>(parameter_);
  size_t nhw_size = 1;
  size_t channels = bias_param->in_shape0_[bias_param->ndim_ - 1];  // C in NHWC
  for (size_t i = 0; i < bias_param->ndim_ - 1; i++) {
    nhw_size *= static_cast<size_t>(bias_param->in_shape0_[i]);
  }

  size_t total_size = channels * nhw_size;

  NNaclFp32Serializer code;
  // Get Tensor Pointer
  std::string input_str = allocator_->GetRuntimeAddr(input_tensor_);
  std::string output_str = allocator_->GetRuntimeAddr(output_tensor_);

  code << "\t\tfor (size_t c = 0; c < " << channels << "; ++c) {\n";
  code << "\t\t\t(" << output_str << ")[c] = 0;\n";
  code << "\t\t\tfor (size_t offset = 0; offset < " << total_size << "; offset += " << channels << ") {\n";
  code << "\t\t\t\t(" << output_str << ")[c] += (" << input_str << ")[offset + c];\n";
  code << "\t\t\t}\n";
  code << "\t\t}\n";

  context->AppendCode(code.str());
  return RET_OK;
}

REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_BiasAddGrad, CPUOpCoderCreator<BiasAddGradCoder>)
}  // namespace mindspore::lite::micro::nnacl
