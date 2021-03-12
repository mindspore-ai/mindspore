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
#include "coder/log.h"
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_fp32_serializer.h"
#include "coder/opcoders/file_collector.h"

using mindspore::schema::PrimitiveType_ReduceFusion;

namespace mindspore::lite::micro::nnacl {
int ReduceFP32Coder::Prepare(CoderContext *const context) {
  MS_CHECK_RET_CODE(ReduceBaseCoder::Init(), "init failed");
  MS_CHECK_RET_CODE(ReSize(), "resize failed");
  MS_CHECK_RET_CODE(MallocTmpBuffer(), "malloc buffer failed");
  return RET_OK;
}

int ReduceFP32Coder::MallocTmpBuffer() {
  data_buffers_.clear();
  for (auto size : buffer_sizes_) {
    auto *buffer = static_cast<float *>(allocator_->Malloc(kNumberTypeFloat, size * sizeof(float), kWorkspace));
    MS_CHECK_PTR(buffer);
    data_buffers_.emplace_back(buffer);
  }
  return RET_OK;
}

int ReduceFP32Coder::ReSize() {
  if (input_tensors_.at(0)->data_type() == kNumberTypeFloat32) {
    data_type_ = kDataTypeFloat;
  } else {
    data_type_ = kDataTypeInt;
  }
  return ReduceBaseCoder::ReSize();
}

int ReduceFP32Coder::DoCode(CoderContext *const context) {
  // generate code .h .c
  if (mode_ == static_cast<int>(schema::ReduceMode_ReduceSum)) {
    Collect(context, {"runtime/kernel/fp32/reduce_sum.h"}, {"reduce_sum.c"});
  } else if (mode_ == static_cast<int>(schema::ReduceMode_ReduceMean)) {
    Collect(context, {"runtime/kernel/fp32/reduce_mean.h"}, {"reduce_mean.c"});
  } else {
    Collect(context, {"runtime/kernel/fp32/reduce.h"}, {"reduce.c"});
  }

  NNaclFp32Serializer code;
  // call the op function
  std::string reduce;
  std::string int_reduce;
  switch (mode_) {
    case static_cast<int>(schema::ReduceMode_ReduceSum): {
      reduce = "ReduceSum";
      break;
    }
    case static_cast<int>(schema::ReduceMode_ReduceMean): {
      reduce = "ReduceMean";
      break;
    }
    case static_cast<int>(schema::ReduceMode_ReduceMax): {
      reduce = "ReduceMax";
      break;
    }
    case static_cast<int>(schema::ReduceMode_ReduceMin): {
      reduce = "ReduceMin";
      break;
    }
    case static_cast<int>(schema::ReduceMode_ReduceProd): {
      reduce = "ReduceProd";
      int_reduce = "IntReduceProd";
      break;
    }
    case static_cast<int>(schema::ReduceMode_ReduceSumSquare): {
      reduce = "ReduceSumSquare";
      break;
    }
    default:
      MS_LOG(ERROR) << "Reduce unsupported reduce mode: " << mode_;
      return RET_ERROR;
  }

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
    if (data_type_ == kDataTypeFloat) {
      code.CodeFunction(reduce, outer_size_, inner_size_, axis_size_, src_addr, dst_addr, 0, thread_num_);
    } else {
      code.CodeFunction(int_reduce, outer_size_, inner_size_, axis_size_, src_addr, dst_addr, 0, thread_num_);
    }
    src_addr = dst_addr;
  }
  context->AppendCode(code.str());
  return RET_OK;
}

REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_ReduceFusion, CPUOpCoderCreator<ReduceFP32Coder>)

}  // namespace mindspore::lite::micro::nnacl
