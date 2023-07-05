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

#include "coder/opcoders/nnacl/fp32/fill_fp32_coder.h"
#include <cmath>
#include "coder/opcoders/file_collector.h"
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_fp32_serializer.h"
#include "coder/opcoders/parallel.h"

using mindspore::schema::PrimitiveType_Fill;

namespace mindspore::lite::micro::nnacl {
int FillFP32Coder::Prepare(CoderContext *context) {
  fill_struct_.base_.param_ = parameter_;
  fill_struct_.data_size_ = static_cast<int>(output_tensors_.size());
  fill_struct_.thread_sz_count_ = MSMIN(thread_num_, fill_struct_.data_size_);
  if (fill_struct_.thread_sz_count_ != 0) {
    fill_struct_.thread_sz_stride_ = UP_DIV(fill_struct_.data_size_, fill_struct_.thread_sz_count_);
  }
  return RET_OK;
}

int FillFP32Coder::DoCode(CoderContext *ctx) {
  Collect(ctx,
          {
            "nnacl/kernel/fill.h",
          },
          {
            "fill.c",
          });
  nnacl::NNaclFp32Serializer code;
  fill_struct_.src_data_ = static_cast<void *>(input_tensor_);
  fill_struct_.out_ptr_ = static_cast<void *>(output_tensor_);
  int size =
    MSMIN(fill_struct_.thread_sz_stride_, fill_struct_.data_size_ - kDefaultTaskId * fill_struct_.thread_sz_stride_);
  MS_CHECK_FALSE(size <= 0, RET_OK);
  int offset = kDefaultTaskId * fill_struct_.thread_sz_stride_;
  auto input_str = MemoryAllocator::GetInstance()->GetRuntimeAddr(input_tensor_);
  auto output_str = MemoryAllocator::GetInstance()->GetRuntimeAddr(output_tensor_) + "+" + std::to_string(offset);
  switch (input_tensor_->data_type()) {
    case kNumberTypeFloat32:
    case kNumberTypeFloat:
      if (!support_parallel_) {
        code.CodeFunction("FillFp32", output_str, size, static_cast<float *>(input_tensor_->data())[0]);
      } else {
        Collect(ctx, {"wrapper/fp32/fill_fp32_wrapper.h"}, {"fill_fp32_wrapper.c"});
        code.CodeBaseStruct("FillFp32Args", kRunArgs, output_str, size, static_cast<float *>(input_tensor_->data())[0]);
        code.CodeFunction(kParallelLaunch, "DoFillFp32", kRunArgsAddr, gThreadNum);
      }
      break;
    case kNumberTypeInt32:
      if (!support_parallel_) {
        code.CodeFunction("FillInt32", output_str, size, static_cast<int *>(input_tensor_->data())[0]);
      } else {
        Collect(ctx, {"wrapper/fp32/fill_fp32_wrapper.h"}, {"fill_fp32_wrapper.c"});
        code.CodeBaseStruct("FillInt32Args", kRunArgs, output_str, size, static_cast<int *>(input_tensor_->data())[0]);
        code.CodeFunction(kParallelLaunch, "DoFillInt32", kRunArgsAddr, gThreadNum);
      }
      break;
    case kNumberTypeBool:
      if (!support_parallel_) {
        code.CodeFunction("FillBool", output_str, size, static_cast<bool *>(input_tensor_->data())[0]);
      } else {
        Collect(ctx, {"wrapper/fp32/fill_fp32_wrapper.h"}, {"fill_fp32_wrapper.c"});
        code.CodeBaseStruct("FillBoolArgs", kRunArgs, output_str, size, static_cast<bool *>(input_tensor_->data())[0]);
        code.CodeFunction(kParallelLaunch, "DoFillBool", kRunArgsAddr, gThreadNum);
      }
      break;
    default:
      MS_LOG(ERROR) << "input_tensor_ data type is invalid";
      return RET_PARAM_INVALID;
  }

  ctx->AppendCode(code.str());
  return RET_OK;
}
REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_Fill, CPUOpCoderCreator<FillFP32Coder>)
REG_OPERATOR_CODER(kAllTargets, kNumberTypeInt32, PrimitiveType_Fill, CPUOpCoderCreator<FillFP32Coder>)
REG_OPERATOR_CODER(kAllTargets, kNumberTypeBool, PrimitiveType_Fill, CPUOpCoderCreator<FillFP32Coder>)
}  // namespace mindspore::lite::micro::nnacl
