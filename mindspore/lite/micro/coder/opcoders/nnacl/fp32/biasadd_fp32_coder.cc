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

#include "coder/opcoders/nnacl/fp32/biasadd_fp32_coder.h"
#include <string>
#include "coder/opcoders/file_collector.h"
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_fp32_serializer.h"

using mindspore::schema::PrimitiveType_BiasAdd;

namespace mindspore::lite::micro::nnacl {

int BiasAddFP32Coder::Prepare(CoderContext *context) {
  arithmetic_parameter_ = reinterpret_cast<ArithmeticParameter *>(parameter_);
  size_t data_size = input_tensors_.at(0)->ElementsNum();
  tile_in_ = reinterpret_cast<float *>(allocator_->Malloc(kNumberTypeFloat32, data_size * sizeof(float), kWorkspace));
  tile_bias_ = reinterpret_cast<float *>(allocator_->Malloc(kNumberTypeFloat32, data_size * sizeof(float), kWorkspace));
  return RET_OK;
}

int BiasAddFP32Coder::DoCode(CoderContext *ctx) {
  if (input_tensors_.size() < kBiasIndex) {
    return RET_ERROR;
  }
  size_t data_size = input_tensor_->ElementsNum();
  std::string bias_str = allocator_->GetRuntimeAddr(input_tensors_.at(kWeightIndex), true);
  Collect(ctx,
          {"nnacl/arithmetic.h", "nnacl/nnacl_utils.h", "nnacl/nnacl_common.h", "nnacl/base/arithmetic_base.h",
           "nnacl/fp32/add_fp32.h", "nnacl/fp32/arithmetic_fp32.h"},
          {"arithmetic_base.c", "arithmetic_fp32.c", "add_fp32.c"});
  nnacl::NNaclFp32Serializer code;
  std::vector<int> dims = input_tensor_->shape();
  arithmetic_parameter_->broadcasting_ = false;
  arithmetic_parameter_->ndim_ = dims.size();
  arithmetic_parameter_->activation_type_ = 0;
  for (size_t i = 0; i < dims.size(); i++) {
    arithmetic_parameter_->in_shape0_[i] = dims[i];
  }
  arithmetic_parameter_->in_elements_num0_ = 0;

  for (size_t i = 0; i < dims.size(); i++) {
    if (i == dims.size() - 1) {
      arithmetic_parameter_->in_shape1_[i] = dims[dims.size() - 1];
      continue;
    }
    arithmetic_parameter_->in_shape1_[i] = 1;
  }
  arithmetic_parameter_->in_elements_num1_ = 0;

  for (size_t i = 0; i < dims.size(); i++) {
    arithmetic_parameter_->out_shape_[i] = dims[i];
  }
  arithmetic_parameter_->out_elements_num_ = 0;
  // other rest elements is not sure

  code.CodeStruct("arith_param", *arithmetic_parameter_);
  code.CodeFunction("BroadcastAdd", input_tensor_, bias_str, tile_in_, tile_bias_, output_tensor_, data_size,
                    "(ArithmeticParameter *)&arith_param");
  ctx->AppendCode(code.str());
  return RET_OK;
}

REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_BiasAdd, CPUOpCoderCreator<BiasAddFP32Coder>)
}  // namespace mindspore::lite::micro::nnacl
