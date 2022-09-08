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
#include "src/litert/kernel/cpu/fp16/concat_fp16.h"
#include "nnacl/fp16/cast_fp16.h"
#include "src/litert/kernel_registry.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Concat;

namespace mindspore::kernel {
int ConcatFp16Run(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  auto concat_kernel = reinterpret_cast<ConcatFp16CPUKernel *>(cdata);
  auto error_code = concat_kernel->DoConcat(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "ConcatRun error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConcatFp16CPUKernel::EnsureFp16InputsAndOutput() {
  inputs_ptr_.clear();
  for (size_t i = 0; i < in_tensors_.size(); ++i) {
    if (!is_with_data_[i]) {
      continue;
    }
    auto input = in_tensors_[i]->data();
    MS_CHECK_TRUE_MSG(input != nullptr, RET_ERROR, "input-data is a nullptr.");
    if (in_tensors_[i]->data_type() == kNumberTypeFloat16) {
      inputs_ptr_.push_back(static_cast<const uint8_t *>(input));
      continue;
    }
    if (in_tensors_[i]->data_type() == kNumberTypeFloat32 || in_tensors_[i]->data_type() == kNumberTypeFloat) {
      auto *tmp =
        reinterpret_cast<uint8_t *>(ms_context_->allocator->Malloc(sizeof(float16_t) * in_tensors_[i]->ElementsNum()));
      if (tmp == nullptr) {
        MS_LOG(ERROR) << "malloc failed";
        return RET_ERROR;
      }
      inputs_ptr_.push_back(tmp);
      tmp_buffers_.push_back(tmp);
      Float32ToFloat16(static_cast<float *>(input), reinterpret_cast<float16_t *>(tmp), in_tensors_[i]->ElementsNum());
    } else {
      MS_LOG(ERROR) << "input's data-type is invalid.";
      return RET_ERROR;
    }
  }
  auto &out_tensor = out_tensors_.at(0);
  if (out_tensor->data_type() == kNumberTypeFloat16) {
    output_ = reinterpret_cast<uint8_t *>(out_tensor->data());
    return RET_OK;
  }
  if (out_tensor->data_type() == kNumberTypeFloat32 || out_tensor->data_type() == kNumberTypeFloat) {
    output_ =
      reinterpret_cast<uint8_t *>(ms_context_->allocator->Malloc(sizeof(float16_t) * out_tensor->ElementsNum()));
    if (output_ == nullptr) {
      MS_LOG(ERROR) << "malloc failed";
      return RET_ERROR;
    }
    tmp_buffers_.push_back(output_);
  } else {
    MS_LOG(ERROR) << "output's data-type is invalid.";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConcatFp16CPUKernel::Run() {
  if (outer_size_ == 0 || inner_sizes_.back() == 0) {
    return RET_OK;
  }
  auto ret = EnsureFp16InputsAndOutput();
  if (ret != RET_OK) {
    for (auto tmp_buffer : tmp_buffers_) {
      ms_context_->allocator->Free(tmp_buffer);
    }
    tmp_buffers_.clear();
    MS_LOG(ERROR) << "EnsureFp16InputsAndOutput failed.";
    return RET_ERROR;
  }
  MS_CHECK_TRUE_MSG(output_ != nullptr, RET_ERROR, "output data is a nullptr.");
  ret = ParallelLaunch(this->ms_context_, ConcatFp16Run, this, block_splits_.size());
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "the kernel run failed. name is " << name_;
  } else if (out_tensors_.at(0)->data_type() == kNumberTypeFloat32 ||
             out_tensors_.at(0)->data_type() == kNumberTypeFloat) {
    auto output = reinterpret_cast<float *>(out_tensors_.at(0)->data());
    if (output == nullptr) {
      MS_LOG(ERROR) << "output data is a nullptr.";
      ret = RET_ERROR;
    } else {
      Float16ToFloat32(reinterpret_cast<float16_t *>(output_), output, out_tensors_.at(0)->ElementsNum());
    }
  }
  for (auto tmp_buffer : tmp_buffers_) {
    ms_context_->allocator->Free(tmp_buffer);
  }
  tmp_buffers_.clear();
  return ret;
}

REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_Concat, LiteKernelCreator<ConcatFp16CPUKernel>)
}  // namespace mindspore::kernel
