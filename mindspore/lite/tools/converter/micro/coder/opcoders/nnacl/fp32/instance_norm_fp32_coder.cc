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
#include "coder/opcoders/nnacl/fp32/instance_norm_fp32_coder.h"
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_fp32_serializer.h"
#include "coder/opcoders/file_collector.h"
#include "coder/opcoders/parallel.h"

using mindspore::schema::PrimitiveType_InstanceNorm;

namespace mindspore::lite::micro::nnacl {
int InstanceNormFP32Coder::Prepare(CoderContext *const context) {
  param_ = reinterpret_cast<InstanceNormParameter *>(parameter_);
  if (support_parallel_) {
    param_->op_parameter_.thread_num_ = 1;
  }
  param_->op_parameter_.thread_num_ = MSMIN(UP_DIV(param_->channel_, C8NUM), param_->op_parameter_.thread_num_);
  if (input_tensors_[0]->format() == NHWC) {
    param_->batch_ = input_tensor_->Batch();
    param_->inner_size_ = input_tensor_->Height() * input_tensor_->Width();
    param_->channel_ = input_tensor_->Channel();
    tmp_src_data_ =
      reinterpret_cast<float *>(allocator_->Malloc(kNumberTypeFloat32, input_tensors_[0]->Size(), kWorkspace));
  }
  return RET_OK;
}

int InstanceNormFP32Coder::DoCode(CoderContext *const context) {
  NNaclFp32Serializer code;
  code.CodeStruct("instance_norm_param", *param_);
  Collect(context, {"nnacl/fp32/pack_fp32.h", "nnacl/fp32/instance_norm_fp32.h"},
          {"pack_fp32.c", "instance_norm_fp32.c"});
  if (input_tensors_[0]->format() == NHWC) {
    code.CodeFunction("PackNHWCToNC4HW4NotAlignedFp32", input_tensor_, tmp_src_data_, param_->batch_,
                      param_->inner_size_, param_->channel_);
    code.CodeFunction("InstanceNormNC4HW4", tmp_src_data_, input_tensor_, input_tensors_.at(SECOND_INPUT),
                      input_tensors_.at(THIRD_INPUT), "&instance_norm_param", 0);
  } else {
    code.CodeFunction("InstanceNorm", input_tensor_, input_tensor_, input_tensors_.at(SECOND_INPUT),
                      input_tensors_.at(THIRD_INPUT), "&instance_norm_param", 0);
  }

  context->AppendCode(code.str());
  return RET_OK;
}

REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_InstanceNorm,
                   CPUOpCoderCreator<InstanceNormFP32Coder>)
}  // namespace mindspore::lite::micro::nnacl
