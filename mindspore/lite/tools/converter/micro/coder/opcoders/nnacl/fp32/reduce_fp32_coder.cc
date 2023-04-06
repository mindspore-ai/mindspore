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

#include "coder/opcoders/nnacl/fp32/reduce_fp32_coder.h"
#include <string>
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_fp32_serializer.h"
#include "coder/opcoders/file_collector.h"

using mindspore::schema::PrimitiveType_ReduceFusion;
namespace mindspore::lite::micro::nnacl {
int ReduceFP32Coder::Prepare(CoderContext *const context) {
  MS_CHECK_RET_CODE(ReduceBaseCoder::Init(), "init failed");
  MS_CHECK_RET_CODE(ReSize(), "resize failed");
  MS_CHECK_RET_CODE(MallocTmpBuffer(kNumberTypeFloat), "malloc buffer failed");
  return RET_OK;
}

int ReduceFP32Coder::MallocTmpBuffer(mindspore::TypeId type_id) {
  data_buffers_.clear();
  for (auto size : buffer_sizes_) {
    auto *buffer = static_cast<float *>(allocator_->Malloc(type_id, size * lite::DataTypeSize(type_id), kWorkspace));
    MS_CHECK_PTR(buffer);
    data_buffers_.emplace_back(buffer);
  }
  return RET_OK;
}

int ReduceFP32Coder::ReSize() {
  if (input_tensors_.at(0)->data_type() == kNumberTypeFloat32) {
    data_type_ = ::kNumberTypeFloat32;
  } else {
    data_type_ = ::kNumberTypeInt32;
  }
  return ReduceBaseCoder::ReSize();
}

int ReduceFP32Coder::DoCode(CoderContext *const context) {
  Collect(context,
          {
            "nnacl/fp32/reduce_fp32.h",
          },
          {
            "reduce_fp32.c",
          });

  // call the op function
  switch (mode_) {
    case static_cast<int>(schema::ReduceMode_ReduceSum): {
      reduce_ = "ReduceSum";
      break;
    }
    case static_cast<int>(schema::ReduceMode_ReduceMean): {
      reduce_ = "ReduceMean";
      break;
    }
    case static_cast<int>(schema::ReduceMode_ReduceMax): {
      reduce_ = "ReduceMax";
      break;
    }
    case static_cast<int>(schema::ReduceMode_ReduceMin): {
      reduce_ = "ReduceMin";
      break;
    }
    case static_cast<int>(schema::ReduceMode_ReduceProd): {
      reduce_ = "ReduceProd";
      int_reduce_ = "IntReduceProd";
      break;
    }
    case static_cast<int>(schema::ReduceMode_ReduceSumSquare): {
      reduce_ = "ReduceSumSquare";
      break;
    }
    default:
      MS_LOG(ERROR) << "Reduce unsupported reduce_ mode: " << mode_;
      return RET_ERROR;
  }
  GenerateCode(context);
  return RET_OK;
}
void ReduceFP32Coder::GenerateCode(CoderContext *const context) {
  NNaclFp32Serializer code;
  std::string src_addr = allocator_->GetRuntimeAddr(input_tensor_);
  std::string dst_addr;
  for (int i = 0; i < num_axes_; ++i) {
    if (i != num_axes_ - 1) {
      dst_addr = allocator_->GetRuntimeAddr(data_buffers_.at(i));
    } else {
      dst_addr = allocator_->GetRuntimeAddr(output_tensor_);
    }
    outer_size_ = outer_sizes_.at(i);
    inner_size_ = inner_sizes_.at(i);
    axis_size_ = axis_sizes_.at(i);
    if (data_type_ == ::kNumberTypeInt32) {
      code.CodeFunction(int_reduce_, outer_size_, inner_size_, axis_size_, src_addr, dst_addr, 0, thread_num_);
    } else {
      code.CodeFunction(reduce_, outer_size_, inner_size_, axis_size_, src_addr, dst_addr, 0, thread_num_);
    }
    src_addr = dst_addr;
  }
  context->AppendCode(code.str());
}

REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_ReduceFusion, CPUOpCoderCreator<ReduceFP32Coder>)
REG_OPERATOR_CODER(kAllTargets, kNumberTypeInt32, PrimitiveType_ReduceFusion, CPUOpCoderCreator<ReduceFP32Coder>)
}  // namespace mindspore::lite::micro::nnacl
