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

#include "src/runtime/kernel/arm/fp16/matmul_base_fp16.h"
#include "nnacl/fp16/matmul_fp16.h"
#include "nnacl/fp16/cast_fp16.h"
#include "include/errorcode.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_INPUT_TENSOR_ERROR;
using mindspore::lite::RET_MEMORY_FAILED;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
int MatmulBaseFP16Run(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  CHECK_NULL_RETURN(cdata);
  auto op = reinterpret_cast<MatmulBaseFP16CPUKernel *>(cdata);
  auto error_code = op->RunImpl(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "MatmulFp16Run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

MatmulBaseFP16CPUKernel::~MatmulBaseFP16CPUKernel() {
  if (src_b_ != nullptr) {
    free(src_b_);
    src_b_ = nullptr;
  }
  if (bias_ptr_ != nullptr) {
    free(bias_ptr_);
    bias_ptr_ = nullptr;
  }
  FreeResizeBufA();
  FreeResizeBufB();
}

void MatmulBaseFP16CPUKernel::FreeResizeBufA() {
  if (a_pack_ptr_ != nullptr) {
    ms_context_->allocator->Free(a_pack_ptr_);
    a_pack_ptr_ = nullptr;
  }
  return;
}

void MatmulBaseFP16CPUKernel::FreeResizeBufB() {
  if (b_pack_ptr_ != nullptr) {
    ms_context_->allocator->Free(b_pack_ptr_);
    b_pack_ptr_ = nullptr;
  }
  return;
}

void MatmulBaseFP16CPUKernel::InitParameter() {
  NNACL_CHECK_NULL_RETURN_VOID(in_tensors_[0]);
  NNACL_CHECK_NULL_RETURN_VOID(in_tensors_[1]);
  params_->a_const_ = (in_tensors_[0]->data() != nullptr);
  params_->b_const_ = (in_tensors_[1]->data() != nullptr);
}

int MatmulBaseFP16CPUKernel::InitBias() {
  if (params_->col_ != 0 && bias_ptr_ == nullptr) {
    int max_bias_data = UP_ROUND(params_->col_, C16NUM);
    bias_ptr_ = reinterpret_cast<float16_t *>(malloc(max_bias_data * sizeof(float16_t)));
    if (bias_ptr_ == nullptr) {
      MS_LOG(ERROR) << "malloc bias_ptr_ failed";
      return RET_ERROR;
    }
    if (in_tensors_.size() == 3) {
      auto bias_tensor = in_tensors_[2];
      CHECK_NULL_RETURN(bias_tensor);
      memcpy(bias_ptr_, bias_tensor->data(), bias_tensor->ElementsNum() * sizeof(float16_t));
    } else {
      memset(bias_ptr_, 0, max_bias_data * sizeof(float16_t));
    }
  }
  return RET_OK;
}

int MatmulBaseFP16CPUKernel::ReSize() {
  ResizeParameter();

  if (params_->b_const_ == true && src_b_ != nullptr) {
    InitBufferB();
    InitMatrixB(src_b_, kNumberTypeFloat16);
    free(src_b_);
    src_b_ = nullptr;
  }
  if (vec_matmul_) {
#ifdef ENABLE_ARM64
    thread_count_ = MSMIN(op_parameter_->thread_num_, UP_DIV(params_->col_, C16NUM));
    thread_stride_ = UP_DIV(UP_DIV(params_->col_, C16NUM), thread_count_) * C16NUM;
#else
    thread_count_ = MSMIN(op_parameter_->thread_num_, UP_DIV(params_->col_, C8NUM));
    thread_stride_ = UP_DIV(UP_DIV(params_->col_, C8NUM), thread_count_) * C8NUM;
#endif
  } else {
    thread_count_ = MSMIN(op_parameter_->thread_num_, UP_DIV(params_->col_, C8NUM));
    thread_stride_ = UP_DIV(UP_DIV(params_->col_, C8NUM), thread_count_) * C8NUM;
  }
  return RET_OK;
}

void MatmulBaseFP16CPUKernel::ResizeParameter() {
  if (params_->row_ == 1) {
    vec_matmul_ = true;
  }

  if (vec_matmul_) {
    params_->row_align_ = 1;
#ifdef ENABLE_ARM64
    params_->col_align_ = UP_ROUND(params_->col_, C16NUM);
#else
    params_->col_align_ = params_->col_;
#endif
  } else {
    params_->row_align_ = UP_ROUND(params_->row_, row_tile_);
    params_->col_align_ = UP_ROUND(params_->col_, C8NUM);
  }
}

int MatmulBaseFP16CPUKernel::InitBufferA() {
  a_pack_ptr_ = reinterpret_cast<float16_t *>(
    ms_context_->allocator->Malloc(params_->batch * params_->row_align_ * params_->deep_ * sizeof(float16_t)));
  if (a_pack_ptr_ == nullptr) {
    return RET_MEMORY_FAILED;
  }

  memset(a_pack_ptr_, 0, params_->batch * params_->row_align_ * params_->deep_ * sizeof(float16_t));
  return RET_OK;
}

int MatmulBaseFP16CPUKernel::InitBufferB() {
  if (b_pack_ptr_ != nullptr) {
    return RET_OK;
  }

  b_pack_ptr_ = reinterpret_cast<float16_t *>(
    ms_context_->allocator->Malloc(params_->batch * params_->col_align_ * params_->deep_ * sizeof(float16_t)));
  if (b_pack_ptr_ == nullptr) {
    return RET_MEMORY_FAILED;
  }

  memset(b_pack_ptr_, 0, params_->batch * params_->col_align_ * params_->deep_ * sizeof(float16_t));
  return RET_OK;
}

void MatmulBaseFP16CPUKernel::InitMatrixA(const void *src_ptr) {
  NNACL_CHECK_NULL_RETURN_VOID(src_ptr);
  auto src_data_type = in_tensors_[0]->data_type();

  if (vec_matmul_) {
    if (src_data_type == kNumberTypeFloat32) {
      Float32ToFloat16(reinterpret_cast<const float *>(src_ptr), a_pack_ptr_, params_->batch * params_->deep_);
    } else {
      memcpy(a_pack_ptr_, src_ptr, params_->batch * params_->deep_ * sizeof(float16_t));
    }
    return;
  }

  const int8_t *int8_src = reinterpret_cast<const int8_t *>(src_ptr);
  for (int i = 0; i < params_->batch; i++) {
    const int8_t *src = int8_src + i * params_->deep_ * params_->row_ * lite::DataTypeSize(src_data_type);
    float16_t *dst = a_pack_ptr_ + i * params_->deep_ * params_->row_align_;
    if (params_->a_transpose_) {
#ifdef ENABLE_ARM64
      RowMajor2RowNMajorFp16((const float16_t *)src, dst, params_->deep_, params_->row_);
#else
      RowMajor2Row12MajorFp16(src, dst, params_->deep_, params_->row_, src_data_type == kNumberTypeFloat32);
#endif
    } else {
#ifdef ENABLE_ARM64
      RowMajor2ColNMajorFp16((const float16_t *)src, dst, params_->row_, params_->deep_);
#else
      RowMajor2Col12MajorFp16(src, dst, params_->row_, params_->deep_, src_data_type == kNumberTypeFloat32);
#endif
    }
  }
  return;
}

void MatmulBaseFP16CPUKernel::InitMatrixB(const void *src_ptr, TypeId src_data_type) {
  NNACL_CHECK_NULL_RETURN_VOID(src_ptr);
  const int8_t *int8_src = reinterpret_cast<const int8_t *>(src_ptr);

  if (vec_matmul_) {
    if (params_->b_transpose_) {
      if (src_data_type == kNumberTypeFloat32) {
        Float32ToFloat16(reinterpret_cast<const float *>(src_ptr), b_pack_ptr_,
                         params_->batch * params_->col_ * params_->deep_);
      } else {
#ifdef ENABLE_ARM64
        for (auto i = 0; i < params_->batch; ++i) {
          const auto *b_src = reinterpret_cast<const float16_t *>(src_ptr) + i * params_->col_align_ * params_->deep_;
          auto *dst = b_pack_ptr_ + i * params_->col_align_ * params_->deep_;
          RowMajor2Col16MajorFp16Opt(b_src, dst, params_->col_, params_->deep_);
        }
#else
        memcpy(b_pack_ptr_, src_ptr, params_->batch * params_->col_ * params_->deep_ * sizeof(float16_t));
#endif
      }
    } else {
      for (int i = 0; i < params_->batch; i++) {
#ifdef ENABLE_ARM64
        const auto *b_src = reinterpret_cast<const float16_t *>(src_ptr) + i * params_->col_align_ * params_->deep_;
        auto *dst = b_pack_ptr_ + i * params_->col_align_ * params_->deep_;
        RowMajor2Row16MajorFp16Opt(b_src, dst, params_->deep_, params_->col_);
#else
        const int8_t *batch_src = int8_src + i * params_->deep_ * params_->col_ * lite::DataTypeSize(src_data_type);
        float16_t *dst = b_pack_ptr_ + i * params_->deep_ * params_->col_;
        RowMajor2ColMajorFp16(batch_src, dst, params_->deep_, params_->col_, src_data_type == kNumberTypeFloat32);
#endif
      }
    }
    return;
  }

  for (int i = 0; i < params_->batch; i++) {
    const int8_t *src = int8_src + i * params_->deep_ * params_->col_ * lite::DataTypeSize(src_data_type);
    float16_t *dst = b_pack_ptr_ + i * params_->deep_ * params_->col_align_;
    if (params_->b_transpose_) {
      RowMajor2Col8MajorFp16(src, dst, params_->col_, params_->deep_, src_data_type == kNumberTypeFloat32);
    } else {
      RowMajor2Row8MajorFp16(src, dst, params_->deep_, params_->col_, src_data_type == kNumberTypeFloat32);
    }
  }
  return;
}

int MatmulBaseFP16CPUKernel::Init() {
  CHECK_LESS_RETURN(in_tensors_.size(), 2);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  ResizeParameter();
  if (params_->a_const_ == true) {
    if (RET_OK != InitBufferA()) {
      return RET_ERROR;
    }
    MS_ASSERT(in_tensors_[0] != nullptr);
    MS_ASSERT(in_tensors_[0]->data() != nullptr);
    InitMatrixA(reinterpret_cast<float *>(in_tensors_[0]->data()));
  }

  if (params_->b_const_ == true) {
    /* copy origin b data, pack in resize
     * pack after a infershape done */
    auto b_tensor = in_tensors_[1];
    MS_ASSERT(b_tensor != nullptr);
    MS_ASSERT(b_tensor->data() != nullptr);
    src_b_ = reinterpret_cast<float16_t *>(malloc(params_->batch * params_->col_ * params_->deep_ * sizeof(float16_t)));
    if (src_b_ == nullptr) {
      MS_LOG(ERROR) << "Matmul fp16 malloc src_b_ failed";
      return RET_ERROR;
    }

    if (b_tensor->data_type() == kNumberTypeFloat32) {
      Float32ToFloat16(reinterpret_cast<float *>(b_tensor->data()), src_b_,
                       params_->batch * params_->col_ * params_->deep_);
    } else {
      memcpy(src_b_, b_tensor->data(), params_->batch * params_->col_ * params_->deep_ * sizeof(float16_t));
    }
  }

  auto ret = InitBias();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Matmul fp16 malloc matrix A buffer failed";
    return RET_ERROR;
  }
  return RET_OK;
}

int MatmulBaseFP16CPUKernel::RunImpl(int task_id) {
  int cur_stride = params_->col_ - task_id * thread_stride_;
  int cur_oc = MSMIN(thread_stride_, cur_stride);
  if (cur_oc <= 0) {
    return RET_OK;
  }

  auto bias = bias_ptr_ + thread_stride_ * task_id;
  auto b = batch_b_ptr_ + task_id * thread_stride_ * params_->deep_;
  auto c = batch_c_ptr_ + task_id * thread_stride_;

  if (vec_matmul_) {
#ifdef ENABLE_ARM64
    VecMatmulFp16(batch_a_ptr_, b, c, bias, params_->act_type_, params_->deep_, cur_oc);
#else
    MatVecMulFp16(batch_a_ptr_, b, c, bias, params_->act_type_, params_->deep_, cur_oc);
#endif
  } else {
#ifdef ENABLE_ARM64
    MatmulBaseFp16Neon(batch_a_ptr_, b, c, bias, params_->act_type_, params_->deep_, params_->row_, cur_oc,
                       params_->col_, OutType_Nhwc);
#else
    MatMulFp16(batch_a_ptr_, b, c, bias, params_->act_type_, params_->deep_, params_->row_, cur_oc, params_->col_,
               OutType_Nhwc);
#endif
  }
  return RET_OK;
}

int MatmulBaseFP16CPUKernel::Run() {
  auto c_ptr = reinterpret_cast<float16_t *>(out_tensors_[0]->data());
  CHECK_NULL_RETURN(c_ptr);

  if ((params_->a_const_ == false) || IsRepack()) {
    if (RET_OK != InitBufferA()) {
      return RET_ERROR;
    }
    InitMatrixA(in_tensors_[0]->data());
  }
  if ((params_->b_const_ == false) || IsRepack()) {
    if (RET_OK != InitBufferB()) {
      FreeResizeBufA();
      return RET_ERROR;
    }
    InitMatrixB(in_tensors_[1]->data(), in_tensors_[1]->data_type());
    InitBias();
  }

  for (int i = 0; i < params_->batch; ++i) {
    if (vec_matmul_) {
      batch_a_ptr_ = a_pack_ptr_ + i * params_->deep_;
#ifdef ENABLE_ARM64
      batch_b_ptr_ = b_pack_ptr_ + i * params_->deep_ * params_->col_align_;
#else
      batch_b_ptr_ = b_pack_ptr_ + i * params_->deep_ * params_->col_;
#endif
      batch_c_ptr_ = c_ptr + i * params_->row_ * params_->col_;
    } else {
      batch_a_ptr_ = a_pack_ptr_ + i * params_->row_align_ * params_->deep_;
      batch_b_ptr_ = b_pack_ptr_ + i * params_->deep_ * params_->col_align_;
      batch_c_ptr_ = c_ptr + i * params_->row_ * params_->col_;
    }
    auto ret = ParallelLaunch(this->ms_context_, MatmulBaseFP16Run, this, thread_count_);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "MatmulBaseFloatRun failed";
      return ret;
    }
  }

  if (params_->a_const_ == false) {
    FreeResizeBufA();
  }

  if (params_->b_const_ == false) {
    FreeResizeBufB();
  }
  return RET_OK;
}
}  // namespace mindspore::kernel
