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

#include "coder/opcoders/base/unstack_base_coder.h"
#include <cmath>
#include <string>
#include "mindspore/lite/src/common/log_util.h"
#include "coder/opcoders/file_collector.h"
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_fp32_serializer.h"
#include "coder/opcoders/parallel.h"
#include "base/float16.h"
using mindspore::schema::PrimitiveType_Unstack;

namespace mindspore::lite::micro {
int UnstackBaseCoder::Prepare(CoderContext *context) {
  size_t shape_size = input_tensor_->shape().size();
  unstack_parameter_ = reinterpret_cast<UnstackParameter *>(parameter_);
  unstack_parameter_->pre_dims_ = 1;
  unstack_parameter_->axis_dim_ = 1;
  unstack_parameter_->after_dims_ = 1;
  if (unstack_parameter_->axis_ < 0) {
    unstack_parameter_->axis_ += static_cast<int>(shape_size);
  }
  for (size_t i = 0; i < shape_size; ++i) {
    if (unstack_parameter_->axis_ > static_cast<int>(i)) {
      unstack_parameter_->pre_dims_ *= input_tensor_->DimensionSize(i);
    } else if (unstack_parameter_->axis_ < static_cast<int>(i)) {
      unstack_parameter_->after_dims_ *= input_tensor_->DimensionSize(i);
    } else {
      unstack_parameter_->axis_dim_ = input_tensor_->DimensionSize(i);
    }
  }

  if (output_addr_array_ != nullptr) {
    free(output_addr_array_);
    output_addr_array_ = nullptr;
  }
  MS_CHECK_FALSE_MSG(SIZE_MUL_OVERFLOW(sizeof(Tensor *), output_tensors_.size()), RET_ERROR, "mul overflow");
  output_addr_array_ = reinterpret_cast<Tensor **>(malloc(sizeof(Tensor *) * output_tensors_.size()));
  if (output_addr_array_ == nullptr) {
    MS_LOG(ERROR) << "Failed to malloc memory";
    return RET_ERROR;
  }
  return RET_OK;
}

int UnstackBaseCoder::DoCode(CoderContext *ctx) {
  Collect(ctx,
          {
            "nnacl/base/unstack_base.h",
          },
          {
            "unstack_base.c",
          });

  size_t out_num = output_tensors_.size();
  unstack_parameter_->num_ = static_cast<int>(out_num);
  int data_type_len = input_tensor_->data_type() == kNumberTypeFloat16 ? sizeof(float16) : sizeof(float);

  nnacl::NNaclFp32Serializer code;
  code.CodeStruct("unstack_parameter", *unstack_parameter_);
  code << "    void* output_addr_array[" << out_num << "] = {";
  for (size_t i = 0; i < out_num; i++) {
    std::string output = output_tensors_.at(i) == nullptr ? "NULL" : allocator_->GetRuntimeAddr(output_tensors_.at(i));
    code << output << ", ";
  }
  code << "};\n";
  code.CodeFunction("Unstack", input_tensor_, "output_addr_array", "&unstack_parameter", data_type_len);
  ctx->AppendCode(code.str());
  return RET_OK;
}

REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_Unstack, CPUOpCoderCreator<UnstackBaseCoder>)
REG_OPERATOR_CODER(kAllTargets, kNumberTypeInt32, PrimitiveType_Unstack, CPUOpCoderCreator<UnstackBaseCoder>)
REG_OPERATOR_CODER(kARM32, kNumberTypeFloat16, PrimitiveType_Unstack, CPUOpCoderCreator<UnstackBaseCoder>)
REG_OPERATOR_CODER(kARM64, kNumberTypeFloat16, PrimitiveType_Unstack, CPUOpCoderCreator<UnstackBaseCoder>)
}  // namespace mindspore::lite::micro
