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
#include "nnacl/fp32/pack_fp32.h"

using mindspore::lite::RET_NULL_PTR;

namespace mindspore::kernel {
int MatmulBaseFloatRun(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  CHECK_NULL_RETURN(cdata);
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
  FreeBuffSrcB();
}

void MatmulFp32BaseCPUKernel::InitParameter() {
  NNACL_CHECK_NULL_RETURN_VOID(in_tensors_[kInputIndex]);
  NNACL_CHECK_NULL_RETURN_VOID(in_tensors_[kWeightIndex]);
  params_->a_const_ = (in_tensors_[kInputIndex]->data() != nullptr);
  params_->b_const_ = (in_tensors_[kWeightIndex]->data() != nullptr);

  if (op_parameter_->is_train_session_) {
    params_->a_const_ = false;
    params_->b_const_ = false;
  }
}

void MatmulFp32BaseCPUKernel::ResizeParameter() {
  init_global_variable();
  if (params_->row_ == 1) {
    vec_matmul_ = true;
#ifdef ENABLE_AVX
    // vector matmul col is aligned to C8NUM in avx
    col_tile_ = C8NUM;
#elif defined(ENABLE_ARM64)
    col_tile_ = C8NUM;
#endif
    row_tile_ = 1;
  }
  params_->row_align_ = UP_ROUND(params_->row_, row_tile_);
#ifdef ENABLE_AVX
  // avx is aligned to col_tile_
  params_->col_align_ = UP_ROUND(params_->col_, col_tile_);
#elif defined(ENABLE_ARM64)
  // no matter vec_matmul_ or not, use col_tile_ to get col_align_
  params_->col_align_ = UP_ROUND(params_->col_, col_tile_);
#else
  params_->col_align_ = vec_matmul_ ? params_->col_ : UP_ROUND(params_->col_, col_tile_);
#endif
  oc_res_ = params_->col_ % col_tile_;
}

int MatmulFp32BaseCPUKernel::InitBufferA() {
  if (a_pack_ptr_ != nullptr) {
    return RET_OK;
  }
  if (!op_parameter_->is_train_session_) {
#ifdef ENABLE_ARM64
    if (vec_matmul_) {
      a_pack_ptr_ = reinterpret_cast<float *>(in_tensors().at(0)->data());
    } else {
      a_pack_ptr_ = reinterpret_cast<float *>(ms_context_->allocator->Malloc(matrix_a_pack_size_ * sizeof(float)));
    }
#else
    a_pack_ptr_ = reinterpret_cast<float *>(ms_context_->allocator->Malloc(matrix_a_pack_size_ * sizeof(float)));
#endif
  } else {
    a_pack_ptr_ = reinterpret_cast<float *>(workspace());
  }
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
  if (op_parameter_->is_train_session_) {
    b_pack_ptr_ = reinterpret_cast<float *>(workspace()) + matrix_a_pack_size_;
  } else {
    b_pack_ptr_ = reinterpret_cast<float *>(
      ms_context_->allocator->Malloc(static_cast<size_t>(matrix_b_pack_size_) * sizeof(float)));
  }
  if (b_pack_ptr_ == nullptr) {
    MS_LOG(ERROR) << "malloc b_pack_ptr_ failed";
    return RET_ERROR;
  }
  return RET_OK;
}

int MatmulFp32BaseCPUKernel::CalBroadCastBiasDataElements() {
  lite::Tensor *bias_tensor = in_tensors_.at(2);
  int max_bias_data = UP_ROUND(bias_tensor->ElementsNum(), col_tile_);
  if (!params_->b_const_) {
    MS_LOG(WARNING) << "matmul do not support broadcast bias data";
  } else {
    lite::Tensor *const_tensor = in_tensors_.at(1);
    size_t shape_size = const_tensor->shape().size();
    if (params_->b_transpose_) {
      MS_CHECK_TRUE_RET(shape_size >= kBiasIndex, max_bias_data);
      max_bias_data = UP_ROUND(const_tensor->shape()[shape_size - kBiasIndex], col_tile_);
    } else {
      MS_CHECK_TRUE_RET(shape_size >= kWeightIndex, max_bias_data);
      max_bias_data = UP_ROUND(const_tensor->shape()[shape_size - kWeightIndex], col_tile_);
    }
  }
  return max_bias_data;
}

int MatmulFp32BaseCPUKernel::InitBiasData() {
  if (in_tensors_.size() == 3) {
    auto bias_tensor = in_tensors_[2];
    size_t max_bias_data = UP_ROUND(bias_tensor->ElementsNum(), col_tile_);
    // malloc addr need to aligned to 32 bytes
    bias_ptr_ = reinterpret_cast<float *>(malloc(max_bias_data * static_cast<int>(sizeof(float))));
    if (bias_ptr_ == nullptr) {
      MS_LOG(ERROR) << "malloc bias_ptr_ failed";
      return RET_ERROR;
    }
    // whether to broadcast bias data
    if (bias_tensor->ElementsNum() == 1) {
      max_bias_data = CalBroadCastBiasDataElements();
      float broadcast_data = (reinterpret_cast<float *>(bias_tensor->data()))[0];
      // broadcast bias data
      for (size_t i = 0; i < max_bias_data; ++i) {
        bias_ptr_[i] = broadcast_data;
      }
    } else {
      memset(bias_ptr_, 0, max_bias_data * static_cast<int>(sizeof(float)));
      memcpy(bias_ptr_, bias_tensor->data(), bias_tensor->ElementsNum() * static_cast<int>(sizeof(float)));
    }
  }
  return RET_OK;
}

int MatmulFp32BaseCPUKernel::InitMatrixA(const float *src_ptr) {
  CHECK_NULL_RETURN(src_ptr);
#ifdef ENABLE_ARM64
  if (vec_matmul_) {
    return RET_OK;
  }
#else
  if (vec_matmul_) {
    memcpy(a_pack_ptr_, src_ptr, params_->batch * params_->deep_ * static_cast<int>(sizeof(float)));
    return RET_OK;
  }
#endif
  for (int i = 0; i < params_->batch; i++) {
    const float *src = src_ptr + i * params_->deep_ * params_->row_;
    float *dst = a_pack_ptr_ + i * params_->deep_ * params_->row_align_;
    if (params_->a_transpose_) {
      matrix_a_pack_fun_(src, dst, params_->deep_, params_->row_);
    } else {
      matrix_a_pack_fun_(src, dst, params_->row_, params_->deep_);
    }
  }
  return RET_OK;
}

int MatmulFp32BaseCPUKernel::InitMatrixB(const float *src_ptr) {
  CHECK_NULL_RETURN(src_ptr);
  if (vec_matmul_) {
    for (int i = 0; i < params_->batch; i++) {
      const float *src_data = src_ptr + i * params_->deep_ * params_->col_;
      float *dst = b_pack_ptr_ + i * params_->deep_ * params_->col_align_;
      if (params_->b_transpose_) {
#ifdef ENABLE_AVX
        RowMajor2Col32Major(src_data, dst, params_->deep_, params_->col_);
#elif defined(ENABLE_ARM64)
        RowMajor2Col8Major(src_data, dst, params_->col_, params_->deep_);
#else
        memcpy(dst, src_data, params_->col_ * params_->deep_ * static_cast<int>(sizeof(float)));
#endif
      } else {
#ifdef ENABLE_AVX
        RowMajor2Row32Major(src_data, dst, params_->col_, params_->deep_);
#elif defined(ENABLE_ARM64)
        RowMajor2Row8Major(src_data, dst, params_->deep_, params_->col_);
#else
        RowMajor2ColMajor(src_data, dst, params_->deep_, params_->col_);
#endif
      }
    }
    return RET_OK;
  }

  for (int i = 0; i < params_->batch; i++) {
    const float *src = src_ptr + i * params_->deep_ * params_->col_;
    float *dst = b_pack_ptr_ + i * params_->deep_ * params_->col_align_;
    if (params_->b_transpose_) {
      matrix_b_pack_fun_(src, dst, params_->col_, params_->deep_);
    } else {
      matrix_b_pack_fun_(src, dst, params_->deep_, params_->col_);
    }
  }
  return RET_OK;
}

void MatmulFp32BaseCPUKernel::FreeBiasBuf() {
  if (bias_ptr_ != nullptr) {
    free(bias_ptr_);
    bias_ptr_ = nullptr;
  }
}

void MatmulFp32BaseCPUKernel::FreeResizeBufA() {
  if (!op_parameter_->is_train_session_) {
#ifdef ENABLE_ARM64
    if (vec_matmul_) {
      a_pack_ptr_ = nullptr;
    } else {
      if (a_pack_ptr_ != nullptr) {
        ms_context_->allocator->Free(a_pack_ptr_);
        a_pack_ptr_ = nullptr;
      }
    }
#else
    if (a_pack_ptr_ != nullptr) {
      ms_context_->allocator->Free(a_pack_ptr_);
      a_pack_ptr_ = nullptr;
    }
#endif
  } else {
    a_pack_ptr_ = nullptr;
  }
}

void MatmulFp32BaseCPUKernel::FreeResizeBufB() {
  if (!op_parameter_->is_train_session_) {
    if (b_pack_ptr_ != nullptr) {
      ms_context_->allocator->Free(b_pack_ptr_);
      b_pack_ptr_ = nullptr;
    }
  } else {
    b_pack_ptr_ = nullptr;
  }
}

int MatmulFp32BaseCPUKernel::FloatRun(int task_id) const {
  int current_start_oc = task_id * thread_stride_ * col_tile_;
  int current_rest_oc = 0;
#if defined(ENABLE_AVX)
  if (vec_matmul_) {
    current_rest_oc = params_->col_align_ - current_start_oc;
  } else {
    current_rest_oc = params_->col_ - current_start_oc;
  }
#else
  current_rest_oc = params_->col_ - current_start_oc;
#endif
  int cur_oc = MSMIN(thread_stride_ * col_tile_, current_rest_oc);
  if (cur_oc <= 0) {
    return RET_OK;
  }

  auto b = batch_b_ptr_ + current_start_oc * params_->deep_;
  auto c = batch_c_ptr_ + current_start_oc;
  auto bias = (bias_ptr_ == nullptr) ? nullptr : bias_ptr_ + current_start_oc;
  if (vec_matmul_) {
#ifdef ENABLE_AVX
    MatVecMulAvxFp32(batch_a_ptr_, b, c, bias, params_->act_type_, params_->deep_, cur_oc, params_->col_align_);
#elif defined(ENABLE_ARM64)
    int rest_align_col = MSMIN(params_->col_align_ - current_start_oc, thread_stride_ * col_tile_);
    MatVecMulFp32Neon64(batch_a_ptr_, b, c, bias, params_->act_type_, params_->deep_, cur_oc, rest_align_col);
#else
    MatVecMulFp32(batch_a_ptr_, b, c, bias, params_->act_type_, params_->deep_, cur_oc);
#endif
  } else {
    MatMulOpt(batch_a_ptr_, b, c, bias, params_->act_type_, params_->deep_, params_->row_, cur_oc, params_->col_,
              OutType_Nhwc);
  }
  return RET_OK;
}

void MatmulFp32BaseCPUKernel::init_global_variable() {
#ifdef ENABLE_AVX
  matrix_a_pack_fun_ = params_->a_transpose_ ? RowMajor2Row6Major : RowMajor2Col6Major;
  matrix_b_pack_fun_ = params_->b_transpose_ ? RowMajor2Col16Major : RowMajor2Row16Major;
  row_tile_ = C6NUM;
  col_tile_ = C16NUM;
#elif defined(ENABLE_ARM32)
  matrix_a_pack_fun_ = params_->a_transpose_ ? RowMajor2Row12Major : RowMajor2Col12Major;
  matrix_b_pack_fun_ = params_->b_transpose_ ? RowMajor2Col4Major : RowMajor2Row4Major;
  row_tile_ = C12NUM;
  col_tile_ = C4NUM;
#elif defined(ENABLE_SSE)
  matrix_a_pack_fun_ = params_->a_transpose_ ? RowMajor2Row4Major : RowMajor2Col4Major;
  matrix_b_pack_fun_ = params_->b_transpose_ ? RowMajor2Col8Major : RowMajor2Row8Major;
  row_tile_ = C4NUM;
  col_tile_ = C8NUM;
#else
  matrix_a_pack_fun_ = params_->a_transpose_ ? RowMajor2Row12Major : RowMajor2Col12Major;
  matrix_b_pack_fun_ = params_->b_transpose_ ? RowMajor2Col8Major : RowMajor2Row8Major;
  row_tile_ = C12NUM;
  col_tile_ = C8NUM;
#endif
  params_->row_align_ = UP_ROUND(params_->row_, row_tile_);
  vec_matmul_ = false;
}

int MatmulFp32BaseCPUKernel::Init() {
  CHECK_LESS_RETURN(in_tensors_.size(), C2NUM);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  init_global_variable();
  matrix_a_pack_size_ = params_->batch * params_->row_align_ * params_->deep_;
  if (matrix_a_pack_size_ < 0) {
    MS_LOG(ERROR) << "Matrix pack size is negative "
                  << "matrix_a_pack_size=" << matrix_a_pack_size_;
    return RET_ERROR;
  }
  auto ret = InitBiasData();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "InitBiasData failed";
    return ret;
  }
  if (params_->a_const_) {
    if (RET_OK != InitBufferA()) {
      return RET_ERROR;
    }
    ret = InitMatrixA(reinterpret_cast<float *>(in_tensors_[0]->data()));
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "InitMatrixA failed!";
      return ret;
    }
  }
  if (params_->b_const_) {
    // only copy weight data
    // resize or run to pack
    auto b_tensor = in_tensors_.at(1);
    src_b_ = reinterpret_cast<float *>(
      malloc(params_->batch * params_->deep_ * params_->col_ * static_cast<int>(sizeof(float))));
    if (src_b_ == nullptr) {
      MS_LOG(ERROR) << "matmul fp16 src_b_ is failed!";
      return RET_ERROR;
    }
    memcpy(src_b_, b_tensor->data(), params_->batch * params_->deep_ * params_->col_ * static_cast<int>(sizeof(float)));
  }
  return RET_OK;
}

void MatmulFp32BaseCPUKernel::FreeBuffSrcB() {
  if (src_b_ != nullptr) {
    free(src_b_);
    src_b_ = nullptr;
  }
}

int MatmulFp32BaseCPUKernel::ReSize() {
  ResizeParameter();
  matrix_a_pack_size_ = params_->batch * params_->row_align_ * params_->deep_;
  matrix_b_pack_size_ = params_->batch * params_->col_align_ * params_->deep_;
  if (matrix_a_pack_size_ < 0 || matrix_b_pack_size_ < 0) {
    MS_LOG(ERROR) << "Matrix pack size is negative "
                  << "matrix_a_pack_size=" << matrix_a_pack_size_ << "matrix_b_pack_size=" << matrix_b_pack_size_;
    return RET_ERROR;
  }
  if (op_parameter_->is_train_session_) {
    set_workspace_size((matrix_a_pack_size_ + matrix_b_pack_size_) * static_cast<int>(sizeof(float)));
  }

  if (params_->b_const_ && src_b_ != nullptr) {
    if (InitBufferB() != RET_OK) {
      FreeBuffSrcB();
      return RET_ERROR;
    }
    if (InitMatrixB(src_b_) != RET_OK) {
      FreeBuffSrcB();
      MS_LOG(ERROR) << "InitMatrixB failed!";
      return RET_ERROR;
    }
    FreeBuffSrcB();
  }
  thread_count_ = MSMIN(op_parameter_->thread_num_, UP_DIV(params_->col_align_, col_tile_));
#if defined(ENABLE_AVX)
  if (vec_matmul_) {
    thread_stride_ = UP_DIV(UP_DIV(params_->col_align_, col_tile_ * C4NUM), thread_count_) * C4NUM;
  } else {
    thread_stride_ = UP_DIV(UP_DIV(params_->col_align_, col_tile_), thread_count_);
  }
#else
  thread_stride_ = UP_DIV(UP_DIV(params_->col_align_, col_tile_), thread_count_);
#endif
  return RET_OK;
}

int MatmulFp32BaseCPUKernel::InitTmpOutBuffer() {
  auto out_data = reinterpret_cast<float *>(out_tensors_.front()->data());
  MS_ASSERT(out_data != nullptr);
#ifdef ENABLE_AVX
  if (oc_res_ != 0 && vec_matmul_) {  // vec matmul need to malloc dst
    int out_channel = params_->col_;
    int oc_block_num = UP_DIV(out_channel, col_tile_);
    MS_ASSERT(ms_context_->allocator != nullptr);
    output_data_ = reinterpret_cast<float *>(ms_context_->allocator->Malloc(
      params_->batch * params_->row_ * oc_block_num * col_tile_ * static_cast<int>(sizeof(float))));
    if (output_data_ == nullptr) {
      MS_LOG(ERROR) << "malloc tmp output data failed.";
      return RET_NULL_PTR;
    }
  } else {  // need to malloc dst to algin block
    output_data_ = out_data;
  }
#else
  output_data_ = out_data;
#endif
  return RET_OK;
}

int MatmulFp32BaseCPUKernel::Run() {
  if (!params_->a_const_) {
    auto a_ptr = reinterpret_cast<float *>(in_tensors_[0]->data());
    CHECK_NULL_RETURN(a_ptr);
    if (RET_OK != InitBufferA()) {
      return RET_ERROR;
    }
    auto ret = InitMatrixA(a_ptr);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "InitMatrixA failed!";
      return ret;
    }
  }
  if (!params_->b_const_) {
    auto b_ptr = reinterpret_cast<float *>(in_tensors_[1]->data());
    CHECK_NULL_RETURN(b_ptr);
    if (RET_OK != InitBufferB()) {
      FreeResizeBufA();
      return RET_ERROR;
    }
    auto ret = InitMatrixB(b_ptr);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "InitMatrixB failed!";
      return ret;
    }
  }

  auto ret = InitTmpOutBuffer();
  if (ret != RET_OK) {
    FreeResizeBufA();
    FreeResizeBufB();
    MS_LOG(ERROR) << "InitTmpOutBuffer error!";
    return ret;
  }

  for (int i = 0; i < params_->batch; ++i) {
    batch_a_ptr_ = a_pack_ptr_ + i * params_->row_align_ * params_->deep_;
    batch_b_ptr_ = b_pack_ptr_ + i * params_->deep_ * params_->col_align_;
    if (vec_matmul_) {
      batch_c_ptr_ = output_data_ + i * params_->row_ * params_->col_align_;
    } else {
      // need not aligned
      batch_c_ptr_ = output_data_ + i * params_->row_ * params_->col_;
    }
    ret = ParallelLaunch(this->ms_context_, MatmulBaseFloatRun, this, thread_count_);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "MatmulBaseFloatRun failed";
    }
  }

#ifdef ENABLE_AVX
  if (oc_res_ != 0 && vec_matmul_) {
    auto out_data = reinterpret_cast<float *>(out_tensors_.front()->MutableData());
    PackNHWCXToNHWCFp32(output_data_, out_data, params_->batch, params_->row_, params_->col_, col_tile_);
    ms_context_->allocator->Free(output_data_);
    output_data_ = nullptr;
  }
#endif
  if (!params_->a_const_) {
    FreeResizeBufA();
  }

  if (!params_->b_const_) {
    FreeResizeBufB();
  }
  return ret;
}
}  // namespace mindspore::kernel
