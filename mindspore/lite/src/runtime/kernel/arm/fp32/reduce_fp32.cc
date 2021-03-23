/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "src/runtime/kernel/arm/fp32/reduce_fp32.h"
#include <cmath>
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "src/runtime/runtime_api.h"
#include "nnacl/fp32/reduce_fp32.h"
#include "src/runtime/kernel/arm/base/reduce_base.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_ReduceFusion;
using mindspore::schema::ReduceMode;
using mindspore::schema::ReduceMode_ReduceAll;
using mindspore::schema::ReduceMode_ReduceASum;
using mindspore::schema::ReduceMode_ReduceMax;
using mindspore::schema::ReduceMode_ReduceMean;
using mindspore::schema::ReduceMode_ReduceMin;
using mindspore::schema::ReduceMode_ReduceProd;
using mindspore::schema::ReduceMode_ReduceSum;
using mindspore::schema::ReduceMode_ReduceSumSquare;

namespace mindspore::kernel {
int ReduceCPUKernel::Init() {
  auto ret = ReduceBaseCPUKernel::Init();
  if (ret != RET_OK) {
    return ret;
  }

  InitialKernelList();

  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int ReduceCPUKernel::ReSize() { return ReduceBaseCPUKernel::ReSize(); }

int ReduceCPUKernel::CallReduceUnit(int task_id) {
  if (data_type_ == kDataTypeFloat) {
    if (!reducer_) {
      MS_LOG(ERROR) << "function reducer_ is null.";
      return RET_NULL_PTR;
    }
    reducer_(outer_size_, inner_size_, axis_size_, static_cast<const float *>(src_data_),
             static_cast<float *>(dst_data_), task_id, context_->thread_num_);
  } else if (data_type_ == KDataTypeBool) {
    if (!bool_reducer_) {
      MS_LOG(ERROR) << "function bool_reducer_ is null.";
      return RET_NULL_PTR;
    }
    bool_reducer_(outer_size_, inner_size_, axis_size_, static_cast<const bool *>(src_data_),
                  static_cast<bool *>(dst_data_), task_id, context_->thread_num_);
  } else {
    if (!int_reducer_) {
      MS_LOG(ERROR) << "function int_reducer_ is null.";
      return RET_NULL_PTR;
    }
    int_reducer_(outer_size_, inner_size_, axis_size_, static_cast<const int *>(src_data_),
                 static_cast<int *>(dst_data_), task_id, context_->thread_num_);
  }
  return RET_OK;
}

int ReduceImpl(void *cdata, int task_id) {
  auto reduce = reinterpret_cast<ReduceCPUKernel *>(cdata);
  auto error_code = reduce->CallReduceUnit(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Reduce Run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int ReduceCPUKernel::Run() {
  if (in_tensors().at(0)->data_type() == kNumberTypeFloat32) {
    data_type_ = kDataTypeFloat;
  } else if (in_tensors().at(0)->data_type() == kNumberTypeBool) {
    data_type_ = KDataTypeBool;
  } else {
    data_type_ = kDataTypeInt;
  }
  auto ret = MallocTmpBuffer();
  if (ret != RET_OK) {
    FreeTmpBuffer();
    return ret;
  }

  src_data_ = in_tensors_.at(0)->data_c();
  HandleASumAndSumSquare();
  for (size_t i = 0; i < static_cast<size_t>(num_axes_); ++i) {
    if (i != static_cast<size_t>(num_axes_ - 1)) {
      dst_data_ = data_buffers_.at(i);
    } else {
      dst_data_ = out_tensors_.at(0)->data_c();
    }
    outer_size_ = outer_sizes_.at(i);
    inner_size_ = inner_sizes_.at(i);
    axis_size_ = axis_sizes_.at(i);
    auto error_code = ParallelLaunch(this->context_->thread_pool_, ReduceImpl, this, context_->thread_num_);
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
  if (data_type_ == kDataTypeInt) {
    return;
  }
  int num = in_tensors_.at(0)->ElementsNum();
  auto *data = reinterpret_cast<float *>(in_tensors_.at(0)->data_c());
  if (data == nullptr) {
    return;
  }
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
  if (data_type_ != kDataTypeFloat) {
    return RET_ERROR;
  }
  auto *out_data = reinterpret_cast<float *>(out_tensor->data_c());
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
    if (data_type_ == kDataTypeFloat) {
      buffer = context_->allocator->Malloc(size * sizeof(float));
    } else if (data_type_ == KDataTypeBool) {
      buffer = context_->allocator->Malloc(size * sizeof(bool));
    } else {
      buffer = context_->allocator->Malloc(size * sizeof(int));
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
      context_->allocator->Free(buffer);
      buffer = nullptr;
    }
  }
  data_buffers_.clear();
}

void ReduceCPUKernel::InitialKernelList() {
  ReduceKernelList func_list[] = {{ReduceMode_ReduceSum, ReduceSum, IntReduceSum, nullptr},
                                  {ReduceMode_ReduceMean, ReduceMean, IntReduceMean, nullptr},
                                  {ReduceMode_ReduceMax, ReduceMax, IntReduceMax, nullptr},
                                  {ReduceMode_ReduceMin, ReduceMin, IntReduceMin, nullptr},
                                  {ReduceMode_ReduceProd, ReduceProd, IntReduceProd, nullptr},
                                  {ReduceMode_ReduceSumSquare, ReduceSum, IntReduceSum, nullptr},
                                  {ReduceMode_ReduceASum, ReduceSum, IntReduceSum, nullptr},
                                  {ReduceMode_ReduceAll, nullptr, nullptr, ReduceAll}};
  int list_len = sizeof(func_list) / sizeof(ReduceKernelList);
  for (int i = 0; i < list_len; ++i) {
    if (mode_ == func_list[i].type_) {
      reducer_ = func_list[i].float_func_;
      int_reducer_ = func_list[i].int_func_;
      bool_reducer_ = func_list[i].bool_func_;
      break;
    }
  }
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_ReduceFusion, LiteKernelCreator<ReduceCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt, PrimitiveType_ReduceFusion, LiteKernelCreator<ReduceCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_ReduceFusion, LiteKernelCreator<ReduceCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeBool, PrimitiveType_ReduceFusion, LiteKernelCreator<ReduceCPUKernel>)
}  // namespace mindspore::kernel
