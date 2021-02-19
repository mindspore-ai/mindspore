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

#include "src/runtime/kernel/arm/fp32/matmul_fp32_base.h"
#include "nnacl/fp32/matmul_fp32.h"

namespace mindspore::kernel {
int MatmulBaseFloatRun(void *cdata, int task_id) {
  auto op = reinterpret_cast<MatmulFp32BaseCPUKernel *>(cdata);
  auto error_code = op->FloatRun(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "MatmulFp32Run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

MatmulFp32BaseCPUKernel::~MatmulFp32BaseCPUKernel() {
  FreeResizeBufA();
  FreeResizeBufB();
  FreeBiasBuf();
  return;
}

void MatmulFp32BaseCPUKernel::InitParameter() {
  params_->a_const_ = (in_tensors_.at(0)->data_c() != nullptr);
  params_->b_const_ = (in_tensors_.at(1)->data_c() != nullptr);

#ifdef ENABLE_AVX
  row_tile_ = C6NUM;
  col_tile_ = C16NUM;
#elif defined(ENABLE_ARM32)
  row_tile_ = C12NUM;
  col_tile_ = C4NUM;
#elif defined(ENABLE_SSE)
  row_tile_ = C4NUM;
  col_tile_ = C8NUM;
#else
  row_tile_ = C12NUM;
  col_tile_ = C8NUM;
#endif
  return;
}

void MatmulFp32BaseCPUKernel::ResizeParameter() {
  if (params_->row_ == 1) {
    vec_matmul_ = true;
  }
  params_->row_align_ = vec_matmul_ ? 1 : UP_ROUND(params_->row_, row_tile_);
  params_->col_align_ = vec_matmul_ ? params_->col_ : UP_ROUND(params_->col_, col_tile_);
  return;
}

int MatmulFp32BaseCPUKernel::InitBufferA() {
  if (a_pack_ptr_ != nullptr) {
    return RET_OK;
  }
  a_pack_ptr_ = reinterpret_cast<float *>(
    context_->allocator->Malloc(params_->batch * params_->row_align_ * params_->deep_ * sizeof(float)));
  if (a_pack_ptr_ == nullptr) {
    MS_LOG(ERROR) << "malloc a_pack_ptr_ failed";
    return RET_ERROR;
  }
  return RET_OK;
}

int MatmulFp32BaseCPUKernel::InitBufferB() {
  if (b_pack_ptr_ != nullptr) {
    return RET_OK;
  }
  b_pack_ptr_ = reinterpret_cast<float *>(
    context_->allocator->Malloc(params_->batch * params_->col_align_ * params_->deep_ * sizeof(float)));
  if (b_pack_ptr_ == nullptr) {
    MS_LOG(ERROR) << "malloc b_pack_ptr_ failed";
    return RET_ERROR;
  }
  return RET_OK;
}

int MatmulFp32BaseCPUKernel::InitBiasData() {
  if (in_tensors_.size() == 3) {
    auto bias_tensor = in_tensors_[2];
    int max_bias_data = UP_ROUND(bias_tensor->ElementsNum(), C16NUM);
    bias_ptr_ = reinterpret_cast<float *>(malloc(max_bias_data * sizeof(float)));
    if (bias_ptr_ == nullptr) {
      MS_LOG(ERROR) << "malloc bias_ptr_ failed";
      return RET_ERROR;
    }
    memset(bias_ptr_, 0, max_bias_data * sizeof(float));
    memcpy(bias_ptr_, bias_tensor->data_c(), bias_tensor->ElementsNum() * sizeof(float));
  }
  return RET_OK;
}

int MatmulFp32BaseCPUKernel::InitMatrixA(const float *src_ptr) {
  if (vec_matmul_) {
    memcpy(a_pack_ptr_, src_ptr, params_->batch * params_->deep_ * sizeof(float));
    return RET_OK;
  }

  for (int i = 0; i < params_->batch; i++) {
    const float *src = src_ptr + i * params_->deep_ * params_->row_;
    float *dst = a_pack_ptr_ + i * params_->deep_ * params_->row_align_;
#ifdef ENABLE_AVX
    if (params_->a_transpose_) {
      RowMajor2Row6Major(src, dst, params_->deep_, params_->row_);
    } else {
      RowMajor2Col6Major(src, dst, params_->row_, params_->deep_);
    }
#elif defined(ENABLE_SSE)
    if (params_->a_transpose_) {
      RowMajor2Row4Major(src, dst, params_->deep_, params_->row_);
    } else {
      RowMajor2Col4Major(src, dst, params_->row_, params_->deep_);
    }
#else
    if (params_->a_transpose_) {
      RowMajor2Row12Major(src, dst, params_->deep_, params_->row_);
    } else {
      RowMajor2Col12Major(src, dst, params_->row_, params_->deep_);
    }
#endif
  }
  return RET_OK;
}

int MatmulFp32BaseCPUKernel::InitMatrixB(const float *src_ptr) {
  if (vec_matmul_) {
    if (params_->b_transpose_) {
      memcpy(b_pack_ptr_, src_ptr, params_->batch * params_->col_ * params_->deep_ * sizeof(float));
    } else {
      for (int i = 0; i < params_->batch; i++) {
        const float *src = src_ptr + i * params_->deep_ * params_->col_;
        float *dst = b_pack_ptr_ + i * params_->deep_ * params_->col_;
        RowMajor2ColMajor(src, dst, params_->deep_, params_->col_);
      }
    }
    return RET_OK;
  }

  for (int i = 0; i < params_->batch; i++) {
    const float *src = src_ptr + i * params_->deep_ * params_->col_;
    float *dst = b_pack_ptr_ + i * params_->deep_ * params_->col_align_;
#ifdef ENABLE_AVX
    if (params_->b_transpose_) {
      RowMajor2Col16Major(src, dst, params_->col_, params_->deep_);
    } else {
      RowMajor2Row16Major(src, dst, params_->deep_, params_->col_);
    }
#elif defined(ENABLE_ARM32)
    if (params_->b_transpose_) {
      RowMajor2Col4Major(src, dst, params_->col_, params_->deep_);
    } else {
      RowMajor2Row4Major(src, dst, params_->deep_, params_->col_);
    }
#else
    if (params_->b_transpose_) {
      RowMajor2Col8Major(src, dst, params_->col_, params_->deep_);
    } else {
      RowMajor2Row8Major(src, dst, params_->deep_, params_->col_);
    }
#endif
  }
  return RET_OK;
}

void MatmulFp32BaseCPUKernel::FreeBiasBuf() {
  if (bias_ptr_ != nullptr) {
    free(bias_ptr_);
    bias_ptr_ = nullptr;
  }
  return;
}

void MatmulFp32BaseCPUKernel::FreeResizeBufA() {
  if (a_pack_ptr_ != nullptr) {
    context_->allocator->Free(a_pack_ptr_);
    a_pack_ptr_ = nullptr;
  }
  return;
}

void MatmulFp32BaseCPUKernel::FreeResizeBufB() {
  if (b_pack_ptr_ != nullptr) {
    context_->allocator->Free(b_pack_ptr_);
    b_pack_ptr_ = nullptr;
  }
  return;
}

int MatmulFp32BaseCPUKernel::FloatRun(int task_id) {
  int current_stride_oc = thread_stride_ * col_tile_;
  int current_rest_oc = params_->col_ - task_id * thread_stride_ * col_tile_;
  int cur_oc = MSMIN(current_stride_oc, current_rest_oc);
  if (cur_oc <= 0) {
    return RET_OK;
  }

  auto b = batch_b_ptr_ + task_id * thread_stride_ * col_tile_ * params_->deep_;
  auto c = batch_c_ptr_ + task_id * thread_stride_ * col_tile_;
  auto bias = (bias_ptr_ == nullptr) ? nullptr : bias_ptr_ + task_id * thread_stride_ * col_tile_;
  if (vec_matmul_) {
    MatVecMulFp32(batch_a_ptr_, b, c, bias, params_->act_type_, params_->deep_, cur_oc);
  } else {
    MatMulOpt(batch_a_ptr_, b, c, bias, params_->act_type_, params_->deep_, params_->row_, cur_oc, params_->col_,
              OutType_Nhwc);
  }
  return RET_OK;
}

int MatmulFp32BaseCPUKernel::Init() {
  ResizeParameter();

  auto ret = InitBiasData();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "InitBiasData failed";
    return ret;
  }

  if (params_->a_const_ == true) {
    if (RET_OK != InitBufferA()) {
      return RET_ERROR;
    }
    InitMatrixA(reinterpret_cast<float *>(in_tensors_[0]->data_c()));
  }

  if (params_->b_const_ == true) {
    /* copy origin b data, pack in resize
     * pack after a infershape done */
    auto b_tensor = in_tensors_[1];
    src_b_ = reinterpret_cast<float *>(malloc(params_->batch * params_->col_ * params_->deep_ * sizeof(float)));
    if (src_b_ == nullptr) {
      MS_LOG(ERROR) << "Matmul fp16 malloc src_b_ failed";
      return RET_ERROR;
    }
    memcpy(src_b_, b_tensor->data_c(), params_->batch * params_->col_ * params_->deep_ * sizeof(float));
  }
  return RET_OK;
}

int MatmulFp32BaseCPUKernel::ReSize() {
  ResizeParameter();

  if (params_->b_const_ == true && src_b_ != nullptr) {
    if (RET_OK != InitBufferB()) {
      return RET_ERROR;
    }
    InitMatrixB(src_b_);
    free(src_b_);
    src_b_ = nullptr;
  }

  thread_count_ = MSMIN(op_parameter_->thread_num_, UP_DIV(params_->col_align_, col_tile_));
  thread_stride_ = UP_DIV(UP_DIV(params_->col_align_, col_tile_), thread_count_);
  return RET_OK;
}

int MatmulFp32BaseCPUKernel::Run() {
  auto a_ptr = reinterpret_cast<float *>(in_tensors_.at(0)->data_c());
  auto b_ptr = reinterpret_cast<float *>(in_tensors_.at(1)->data_c());
  auto c_ptr = reinterpret_cast<float *>(out_tensors_.at(0)->data_c());

  if (params_->a_const_ == false) {
    if (RET_OK != InitBufferA()) {
      return RET_ERROR;
    }
    InitMatrixA(a_ptr);
  }
  if (params_->b_const_ == false) {
    if (RET_OK != InitBufferB()) {
      FreeResizeBufA();
      return RET_ERROR;
    }
    InitMatrixB(b_ptr);
  }

  for (int i = 0; i < params_->batch; ++i) {
    if (vec_matmul_) {
      batch_a_ptr_ = a_pack_ptr_ + i * params_->deep_;
      batch_b_ptr_ = b_pack_ptr_ + i * params_->deep_ * params_->col_;
      batch_c_ptr_ = c_ptr + i * params_->row_ * params_->col_;
    } else {
      batch_a_ptr_ = a_pack_ptr_ + i * params_->row_align_ * params_->deep_;
      batch_b_ptr_ = b_pack_ptr_ + i * params_->deep_ * params_->col_align_;
      batch_c_ptr_ = c_ptr + i * params_->row_ * params_->col_;
    }
    auto ret = ParallelLaunch(this->context_->thread_pool_, MatmulBaseFloatRun, this, thread_count_);
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
