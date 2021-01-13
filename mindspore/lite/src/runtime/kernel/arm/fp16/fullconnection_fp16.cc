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

#include "src/runtime/kernel/arm/fp16/fullconnection_fp16.h"
#include "src/runtime/runtime_api.h"
#include "src/kernel_registry.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_INPUT_TENSOR_ERROR;
using mindspore::lite::RET_MEMORY_FAILED;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_FullConnection;

namespace mindspore::kernel {
FullconnectionFP16CPUKernel::~FullconnectionFP16CPUKernel() { FreeTmpBuffer(); }

void FullconnectionFP16CPUKernel::FreeTmpBuffer() {
  if (a_pack_ptr_ != nullptr) {
    context_->allocator->Free(a_pack_ptr_);
    a_pack_ptr_ = nullptr;
  }
  if (b_pack_ptr_ != nullptr) {
    context_->allocator->Free(b_pack_ptr_);
    b_pack_ptr_ = nullptr;
  }
  if (bias_ptr_ != nullptr) {
    context_->allocator->Free(bias_ptr_);
    bias_ptr_ = nullptr;
  }
  if (output_fp16_ != nullptr) {
    context_->allocator->Free(output_fp16_);
    output_fp16_ = nullptr;
  }
}

int FullconnectionFP16CPUKernel::ReSize() {
  FreeTmpBuffer();
  int row = 1;
  for (size_t i = 0; i < out_tensors_.at(0)->shape().size() - 1; ++i) row *= (out_tensors_.at(0)->shape())[i];
  fc_param_->row_ = row;
  fc_param_->col_ = out_tensors_.at(0)->shape().back();
  fc_param_->deep_ = (in_tensors_.at(1)->shape()).at(1);
  fc_param_->row_16_ = UP_ROUND(fc_param_->row_, C16NUM);
  fc_param_->col_8_ = UP_ROUND(fc_param_->col_, C8NUM);
  thread_count_ = MSMIN(op_parameter_->thread_num_, UP_DIV(fc_param_->col_, C8NUM));
  thread_stride_ = UP_DIV(UP_DIV(fc_param_->col_, C8NUM), thread_count_) * C8NUM;

  if (row == 1) is_vector_input_ = true;
  int a_pack_row = 0;
  int b_pack_col = 0;
  if (is_vector_input_) {
    a_pack_row = 1;
    b_pack_col = fc_param_->col_;
  } else {
    a_pack_row = fc_param_->row_16_;
    b_pack_col = fc_param_->col_8_;
  }
  a_pack_ptr_ =
    reinterpret_cast<float16_t *>(context_->allocator->Malloc(a_pack_row * fc_param_->deep_ * sizeof(float16_t)));
  if (a_pack_ptr_ == nullptr) {
    FreeTmpBuffer();
    return RET_MEMORY_FAILED;
  }
  memset(a_pack_ptr_, 0, a_pack_row * fc_param_->deep_ * sizeof(float16_t));

  b_pack_ptr_ =
    reinterpret_cast<float16_t *>(context_->allocator->Malloc(b_pack_col * fc_param_->deep_ * sizeof(float16_t)));
  if (b_pack_ptr_ == nullptr) {
    FreeTmpBuffer();
    return RET_MEMORY_FAILED;
  }
  memset(b_pack_ptr_, 0, b_pack_col * fc_param_->deep_ * sizeof(float16_t));

  fc_param_->b_const_ = (in_tensors_.at(1)->data_c() != nullptr);
  if (fc_param_->b_const_) {
    if (in_tensors_.at(1)->data_type() == kNumberTypeFloat32) {
      if (is_vector_input_) {
        Float32ToFloat16(reinterpret_cast<float *>(in_tensors_.at(1)->data_c()), b_pack_ptr_,
                         fc_param_->col_ * fc_param_->deep_);
      } else {
        InitMatrixB(reinterpret_cast<float *>(in_tensors_.at(1)->data_c()), b_pack_ptr_);
      }
    } else {
      if (is_vector_input_) {
        memcpy(b_pack_ptr_, reinterpret_cast<float16_t *>(in_tensors_.at(1)->data_c()),
               fc_param_->col_ * fc_param_->deep_ * sizeof(float16_t));
      } else {
        InitMatrixB(reinterpret_cast<float16_t *>(in_tensors_.at(1)->data_c()), b_pack_ptr_);
      }
    }
    b_ptr_ = b_pack_ptr_;
  }

  if (in_tensors_.size() == 3) {
    bias_ptr_ = reinterpret_cast<float16_t *>(context_->allocator->Malloc(b_pack_col * sizeof(float16_t)));
    if (bias_ptr_ == nullptr) {
      FreeTmpBuffer();
      return RET_MEMORY_FAILED;
    }
    memset(bias_ptr_, 0, b_pack_col * sizeof(float16_t));
    Float32ToFloat16(reinterpret_cast<float *>(in_tensors_.at(2)->data_c()), bias_ptr_, fc_param_->col_);
  }

  if (out_tensors_.at(0)->data_type() == kNumberTypeFloat32) {
    output_fp16_ =
      reinterpret_cast<float16_t *>(context_->allocator->Malloc(fc_param_->row_ * fc_param_->col_ * sizeof(float16_t)));
    if (output_fp16_ == nullptr) {
      FreeTmpBuffer();
      return RET_MEMORY_FAILED;
    }
  }
  return RET_OK;
}  // namespace mindspore::kernel

void FullconnectionFP16CPUKernel::InitMatrixA(float *a_ptr, float16_t *a_pack_ptr) {
  RowMajor2Col16MajorFp16(reinterpret_cast<void *>(a_ptr), a_pack_ptr, fc_param_->row_, fc_param_->deep_, true);
}

void FullconnectionFP16CPUKernel::InitMatrixA(float16_t *a_ptr, float16_t *a_pack_ptr) {
  RowMajor2Col16MajorFp16(reinterpret_cast<void *>(a_ptr), a_pack_ptr, fc_param_->row_, fc_param_->deep_, false);
}

void FullconnectionFP16CPUKernel::InitMatrixB(float *b_ptr, float16_t *b_pack_ptr) {
  RowMajor2Col8MajorFp16(reinterpret_cast<void *>(b_ptr), b_pack_ptr, fc_param_->col_, fc_param_->deep_, true);
}

void FullconnectionFP16CPUKernel::InitMatrixB(float16_t *b_ptr, float16_t *b_pack_ptr) {
  RowMajor2Col8MajorFp16(reinterpret_cast<void *>(b_ptr), b_pack_ptr, fc_param_->col_, fc_param_->deep_, false);
}

int FullconnectionFP16CPUKernel::Init() {
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int FullconnectionFP16CPUKernel::RunImpl(int task_id) {
  int cur_stride = fc_param_->col_ - task_id * thread_stride_;
  int cur_oc = MSMIN(thread_stride_, cur_stride);
  if (cur_oc <= 0) {
    return RET_OK;
  }
  auto b = b_ptr_ + task_id * thread_stride_ * fc_param_->deep_;
  auto bias = (bias_ptr_ == nullptr) ? nullptr : bias_ptr_ + thread_stride_ * task_id;
  auto c = output_ptr_ + task_id * thread_stride_;
  if (is_vector_input_) {
    MatVecMulFp16(a_ptr_, b, c, bias, fc_param_->act_type_, fc_param_->deep_, cur_oc);
  } else {
    MatMulFp16(a_ptr_, b, c, bias, fc_param_->act_type_, fc_param_->deep_, fc_param_->row_, cur_oc, fc_param_->col_,
               OutType_Nhwc);
  }

  return RET_OK;
}

int FcFP16Run(void *cdata, int task_id) {
  auto op = reinterpret_cast<FullconnectionFP16CPUKernel *>(cdata);
  auto error_code = op->RunImpl(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "MatmulFp32Run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int FullconnectionFP16CPUKernel::Run() {
  auto out_tensor = out_tensors_.at(0);
  if (out_tensor->data_type() == kNumberTypeFloat32) {
    output_ptr_ = output_fp16_;
  } else {
    output_ptr_ = reinterpret_cast<float16_t *>(out_tensor->data_c());
  }

  if (in_tensors_.at(0)->data_type() == kNumberTypeFloat32) {
    if (is_vector_input_) {
      Float32ToFloat16(reinterpret_cast<float *>(in_tensors_.at(0)->data_c()), a_pack_ptr_, fc_param_->deep_);
    } else {
      InitMatrixA(reinterpret_cast<float *>(in_tensors_.at(0)->data_c()), a_pack_ptr_);
    }
    a_ptr_ = a_pack_ptr_;
  } else {
    if (is_vector_input_) {
      a_ptr_ = reinterpret_cast<float16_t *>(in_tensors_.at(0)->data_c());
    } else {
      InitMatrixA(reinterpret_cast<float16_t *>(in_tensors_.at(0)->data_c()), a_pack_ptr_);
      a_ptr_ = a_pack_ptr_;
    }
  }

  if (!fc_param_->b_const_) {
    if (in_tensors_.at(1)->data_type() == kNumberTypeFloat32) {
      if (is_vector_input_) {
        Float32ToFloat16(reinterpret_cast<float *>(in_tensors_.at(1)->data_c()), b_pack_ptr_,
                         fc_param_->col_ * fc_param_->deep_);
      } else {
        InitMatrixB(reinterpret_cast<float *>(in_tensors_.at(1)->data_c()), b_pack_ptr_);
      }
      b_ptr_ = b_pack_ptr_;
    } else {
      if (is_vector_input_) {
        b_ptr_ = reinterpret_cast<float16_t *>(in_tensors_.at(1)->data_c());
      } else {
        InitMatrixB(reinterpret_cast<float16_t *>(in_tensors_.at(1)->data_c()), b_pack_ptr_);
        b_ptr_ = b_pack_ptr_;
      }
    }
  }
  ParallelLaunch(this->context_->thread_pool_, FcFP16Run, this, thread_count_);
  if (out_tensor->data_type() == kNumberTypeFloat32) {
    auto size = out_tensor->ElementsNum();
    auto out_tensor_data = reinterpret_cast<float *>(out_tensor->data_c());
    Float16ToFloat32(output_fp16_, out_tensor_data, size);
  }
  return RET_OK;
}

kernel::LiteKernel *CpuFullConnectionFp16KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                                       const std::vector<lite::Tensor *> &outputs,
                                                       OpParameter *opParameter, const lite::InnerContext *ctx,
                                                       const kernel::KernelKey &desc,
                                                       const mindspore::lite::PrimitiveC *primitive) {
  auto *kernel = new (std::nothrow) FullconnectionFP16CPUKernel(opParameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "kernel is nullptr.";
    free(opParameter);
    return nullptr;
  }
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init kernel failed, name: " << opParameter->name_ << ", type: "
                  << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(opParameter->type_));
    delete kernel;
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_FullConnection, CpuFullConnectionFp16KernelCreator)
}  // namespace mindspore::kernel
