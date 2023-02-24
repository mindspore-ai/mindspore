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
#include <cmath>
#include "schema/model_generated.h"
#include "src/litert/kernel_registry.h"
#include "include/errorcode.h"
#include "nnacl/fp32/reduce_fp32.h"
#include "src/litert/kernel/cpu/base/reduce_base.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_ReduceFusion;
using mindspore::schema::ReduceMode;
using mindspore::schema::ReduceMode_ReduceAll;
using mindspore::schema::ReduceMode_ReduceASum;
using mindspore::schema::ReduceMode_ReduceL2;
using mindspore::schema::ReduceMode_ReduceMax;
using mindspore::schema::ReduceMode_ReduceMean;
using mindspore::schema::ReduceMode_ReduceMin;
using mindspore::schema::ReduceMode_ReduceProd;
using mindspore::schema::ReduceMode_ReduceSum;
using mindspore::schema::ReduceMode_ReduceSumSquare;

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

int ReduceCPUKernel::CallReduceUnit(int task_id) {
  CHECK_NULL_RETURN(src_data_);
  CHECK_NULL_RETURN(dst_data_);
  if (data_type_ == kNumberTypeFloat32) {
    if (reducer_ == nullptr) {
      MS_LOG(ERROR) << "function reducer_ is null.";
      return RET_NULL_PTR;
    }
    if (inner_size_ == 1 && float_last_axis_func_ != nullptr) {
      float_last_axis_func_(outer_size_, inner_size_, axis_size_, static_cast<const float *>(src_data_),
                            static_cast<float *>(dst_data_), task_id, thread_num_);
    } else {
      reducer_(outer_size_, inner_size_, axis_size_, static_cast<const float *>(src_data_),
               static_cast<float *>(dst_data_), task_id, thread_num_);
    }
  } else if (data_type_ == kNumberTypeBool) {
    if (bool_reducer_ == nullptr) {
      MS_LOG(ERROR) << "function bool_reducer_ is null.";
      return RET_NULL_PTR;
    }
    bool_reducer_(outer_size_, inner_size_, axis_size_, static_cast<const bool *>(src_data_),
                  static_cast<bool *>(dst_data_), task_id, thread_num_);
  } else {
    if (int_reducer_ == nullptr) {
      MS_LOG(ERROR) << "function int_reducer_ is null.";
      return RET_NULL_PTR;
    }
    int_reducer_(outer_size_, inner_size_, axis_size_, static_cast<const int *>(src_data_),
                 static_cast<int *>(dst_data_), task_id, thread_num_);
  }
  return RET_OK;
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

void ReduceCPUKernel::HandleASumAndSumSquare() {
  if (data_type_ == kNumberTypeInt32 || data_type_ == kNumberTypeBool) {
    return;
  }
  int num = in_tensors_[kInputIndex]->ElementsNum();
  float *data = static_cast<float *>(in_tensors_[kInputIndex]->data());
  NNACL_CHECK_NULL_RETURN_VOID(data);

  if (reduce_param_->mode_ == static_cast<int>(ReduceMode_ReduceASum)) {
    for (int i = 0; i < num; ++i) {
      if (data[i] < 0.0f) {
        data[i] = 0.0f - data[i];
      }
    }
  }
  if (reduce_param_->mode_ == static_cast<int>(ReduceMode_ReduceSumSquare)) {
    for (int i = 0; i < num; ++i) {
      data[i] = data[i] * data[i];
    }
  }
}

int ReduceCPUKernel::CalculateCoeffOutput() {
  auto out_tensor = out_tensors_.at(0);
  int num = out_tensor->ElementsNum();
  if (data_type_ != kNumberTypeFloat32) {
    return RET_ERROR;
  }
  auto *out_data = reinterpret_cast<float *>(out_tensor->data());
  if (out_data == nullptr) {
    return RET_NULL_PTR;
  }
  for (int i = 0; i < num; ++i) {
    out_data[i] *= reduce_param_->coeff;
  }
  return RET_OK;
}

int ReduceCPUKernel::MallocTmpBuffer() {
  data_buffers_.clear();
  for (auto size : buffer_sizes_) {
    void *buffer = nullptr;
    if (data_type_ == kNumberTypeFloat32) {
      buffer = ms_context_->allocator->Malloc(size * sizeof(float));
    } else if (data_type_ == kNumberTypeFloat16) {
      buffer = ms_context_->allocator->Malloc(size * FP16_DATA_TYPE_LEN);
    } else if (data_type_ == kNumberTypeBool) {
      buffer = ms_context_->allocator->Malloc(size * sizeof(bool));
    } else {
      buffer = ms_context_->allocator->Malloc(size * sizeof(int));
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

void ReduceCPUKernel::InitialKernelList() {
  ReduceKernelList func_list[] = {{ReduceMode_ReduceSum, ReduceSum, IntReduceSum, nullptr, ReduceSumByLastAxis},
                                  {ReduceMode_ReduceMean, ReduceMean, IntReduceMean, nullptr, nullptr},
                                  {ReduceMode_ReduceMax, ReduceMax, IntReduceMax, nullptr, ReduceMaxByLastAxis},
                                  {ReduceMode_ReduceMin, ReduceMin, IntReduceMin, nullptr, nullptr},
                                  {ReduceMode_ReduceProd, ReduceProd, IntReduceProd, nullptr, nullptr},
                                  {ReduceMode_ReduceSumSquare, ReduceSum, IntReduceSum, nullptr, nullptr},
                                  {ReduceMode_ReduceASum, ReduceSum, IntReduceSum, nullptr, nullptr},
                                  {ReduceMode_ReduceAll, nullptr, nullptr, ReduceAll, nullptr},
                                  {ReduceMode_ReduceL2, ReduceL2Norm, nullptr, nullptr, nullptr}};
  size_t list_len = sizeof(func_list) / sizeof(ReduceKernelList);
  for (size_t i = 0; i < list_len; ++i) {
    if (mode_ == func_list[i].type_) {
      reducer_ = func_list[i].float_func_;
      int_reducer_ = func_list[i].int_func_;
      bool_reducer_ = func_list[i].bool_func_;
      float_last_axis_func_ = func_list[i].float_last_axis_func_;
      break;
    }
  }
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_ReduceFusion, LiteKernelCreator<ReduceCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_ReduceFusion, LiteKernelCreator<ReduceCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeBool, PrimitiveType_ReduceFusion, LiteKernelCreator<ReduceCPUKernel>)
}  // namespace mindspore::kernel
