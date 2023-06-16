/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "src/litert/kernel/cpu/fp32/reduce_fp32.h"
#include "src/litert/kernel_registry.h"
#include "include/errorcode.h"
#include "nnacl/fp32/reduce_fp32.h"
#include "src/litert/kernel/cpu/base/reduce_base.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
int ReduceCPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), 1);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  auto ret = ReduceBaseCPUKernel::Prepare();
  if (ret != RET_OK) {
    return ret;
  }

  InitialKernelList();

  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int ReduceImpl(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  auto reduce = reinterpret_cast<ReduceCPUKernel *>(cdata);
  CHECK_NULL_RETURN(reduce);
  auto error_code = reduce->CallReduceUnit(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Reduce Run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int ReduceCPUKernel::Run() {
  if (only_copy_) {
    return CopyInputToOutput();
  }
  data_type_ = in_tensors().at(0)->data_type();
  auto ret = MallocTmpBuffer();
  if (ret != RET_OK) {
    FreeTmpBuffer();
    return ret;
  }

  src_data_ = in_tensors_.at(0)->data();
  HandleASumAndSumSquare();
  for (size_t i = 0; i < static_cast<size_t>(num_axes_); ++i) {
    if (i != static_cast<size_t>(num_axes_ - 1)) {
      dst_data_ = data_buffers_.at(i);
    } else {
      dst_data_ = out_tensors_.at(0)->data();
    }
    outer_size_ = outer_sizes_.at(i);
    inner_size_ = inner_sizes_.at(i);
    axis_size_ = axis_sizes_.at(i);
    if (axis_size_ == 0) {
      MS_LOG(ERROR) << "axis_size_ is must not be zero!";
      FreeTmpBuffer();
      return RET_ERROR;
    }
    auto error_code = ParallelLaunch(this->ms_context_, ReduceImpl, this, thread_num_);
    if (error_code != RET_OK) {
      MS_LOG(ERROR) << "Reduce run error, error_code[" << error_code << "]";
      FreeTmpBuffer();
      return RET_ERROR;
    }
    src_data_ = dst_data_;
  }
  if (reduce_param_->reduce_to_end_ && abs(reduce_param_->coeff) > 1e-5) {
    ret = CalculateCoeffOutput();
    if (ret != RET_OK) {
      FreeTmpBuffer();
      return ret;
    }
  }

  FreeTmpBuffer();
  return RET_OK;
}

int ReduceCPUKernel::MallocTmpBuffer() {
  data_buffers_.clear();
  for (auto size : buffer_sizes_) {
    void *buffer = nullptr;
    if (data_type_ == kNumberTypeFloat16) {
      buffer = ms_context_->allocator->Malloc(size * FP16_DATA_TYPE_LEN);
    }
    if (buffer == nullptr) {
      MS_LOG(ERROR) << "Malloc data failed.";
      return RET_ERROR;
    }
    data_buffers_.emplace_back(buffer);
  }
  return RET_OK;
}

void ReduceCPUKernel::FreeTmpBuffer() {
  for (auto &buffer : data_buffers_) {
    if (buffer != nullptr) {
      ms_context_->allocator->Free(buffer);
      buffer = nullptr;
    }
  }
  data_buffers_.clear();
}
}  // namespace mindspore::kernel
