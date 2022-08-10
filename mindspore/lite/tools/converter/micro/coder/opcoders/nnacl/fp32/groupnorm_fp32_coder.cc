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
#include "coder/opcoders/nnacl/fp32/groupnorm_fp32_coder.h"
#include <string>
#include <vector>
#include "nnacl/fp32/group_norm_fp32.h"
#include "nnacl/op_base.h"
#include "coder/opcoders/file_collector.h"
#include "coder/opcoders/parallel.h"
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_fp32_serializer.h"

using mindspore::schema::PrimitiveType_GroupNormFusion;

namespace mindspore::lite::micro::nnacl {
int GroupNormFP32Coder::Init() {
  auto gn_parameter = reinterpret_cast<GroupNormParameter *>(OperatorCoder::parameter_);
  std::vector<int> input_shapes = input_tensor_->shape();
  if (input_shapes.empty()) {
    return RET_ERROR;
  }
  // only NCHW is supported
  auto fmt = input_tensor_->format();
  CHECK_NOT_EQUAL_RETURN(fmt, NCHW);

  auto in_n_dim = input_shapes.size();
  CHECK_LESS_RETURN(in_n_dim, 1);

  gn_parameter->unit_ = input_tensor_->Height() * input_tensor_->Width();
  gn_parameter->batch_ = input_tensor_->Batch();
  gn_parameter->channel_ = input_tensor_->Channel();
  return RET_OK;
}

int GroupNormFP32Coder::Prepare(CoderContext *const context) {
  auto gn_parameter = reinterpret_cast<GroupNormParameter *>(parameter_);
  int mean_var_size = gn_parameter->num_groups_ * sizeof(float);
  mean_ = reinterpret_cast<float *>(allocator_->Malloc(kNumberTypeFloat32, mean_var_size, kWorkspace));
  MS_CHECK_PTR(mean_);
  variance_ = reinterpret_cast<float *>(allocator_->Malloc(kNumberTypeFloat32, mean_var_size, kWorkspace));
  MS_CHECK_PTR(variance_);
  return RET_OK;
}

int GroupNormFP32Coder::DoCode(CoderContext *const context) {
  // attribute
  auto gn_parameter = reinterpret_cast<GroupNormParameter *>(parameter_);
  if (Init() != RET_OK) {
    MS_LOG(ERROR) << "GroupFP32Coder Init error";
    return RET_ERROR;
  }
  MS_CHECK_TRUE(input_tensors_.size() == DIMENSION_3D, "inputs size is not equal to three");
  Tensor *scale_tensor = input_tensors_.at(kWeightIndex);
  Tensor *offset_tensor = input_tensors_.at(kBiasIndex);
  MS_CHECK_PTR(scale_tensor);
  MS_CHECK_PTR(offset_tensor);
  Collect(context,
          {
            "nnacl/fp32/group_norm_fp32.h",
          },
          {
            "group_norm_fp32.c",
          });
  NNaclFp32Serializer code;
  std::string param_name = "gn_parameter";
  code.CodeStruct(param_name, *gn_parameter);
  if (support_parallel_) {
    code << "    " << param_name << ".op_parameter_.thread_num_ = 1;\n";
  }
  code.CodeFunction("GroupNormFp32", input_tensor_, scale_tensor, offset_tensor, mean_, variance_, "&gn_parameter",
                    kDefaultTaskId, output_tensor_);
  MS_LOG(INFO) << "GroupNormFp32Code has been called";
  context->AppendCode(code.str());
  return lite::RET_OK;
}

REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_GroupNormFusion,
                   CPUOpCoderCreator<GroupNormFP32Coder>)
}  // namespace mindspore::lite::micro::nnacl
