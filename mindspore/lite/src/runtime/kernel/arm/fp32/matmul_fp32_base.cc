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
#include <algorithm>
#include "nnacl/fp32/matmul_fp32.h"
#include "nnacl/fp32/pack_fp32.h"
#ifdef ENABLE_AVX512
#include "nnacl/fp32/matmul_avx512_fp32.h"
#endif

using mindspore::lite::RET_NULL_PTR;

namespace mindspore::kernel {
int MatmulRun(const void *cdata, int task_id, float, float) {
  CHECK_NULL_RETURN(cdata);
  auto op = reinterpret_cast<const MatmulFp32BaseCPUKernel *>(cdata);
  auto error_code = (op->*(op->parallel_fun_))(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "MatmulRun error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

MatmulFp32BaseCPUKernel::~MatmulFp32BaseCPUKernel() {
  FreeResizeBufA();
  FreeResizeBufB();
  FreeBiasBuf();
}

void MatmulFp32BaseCPUKernel::InitParameter() {
  NNACL_CHECK_NULL_RETURN_VOID(in_tensors_[kInputIndex]);
  NNACL_CHECK_NULL_RETURN_VOID(in_tensors_[kWeightIndex]);
  params_->a_const_ = in_tensors_[kInputIndex]->IsConst();
  params_->b_const_ = in_tensors_[kWeightIndex]->IsConst();

  if (op_parameter_->is_train_session_) {
    params_->a_const_ = false;
    params_->b_const_ = false;
  }
}

int MatmulFp32BaseCPUKernel::InitBufferA() {
  if (a_pack_ptr_ != nullptr) {
    return RET_OK;
  }
  if (vec_matmul_) {
    a_pack_ptr_ = reinterpret_cast<float *>(in_tensors().at(0)->data());
  } else {
    if (op_parameter_->is_train_session_) {
      a_pack_ptr_ = reinterpret_cast<float *>(workspace());
    } else {
      a_pack_ptr_ = reinterpret_cast<float *>(ms_context_->allocator->Malloc(matrix_a_pack_size_ * sizeof(float)));
    }
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
  if (in_tensors_.size() != FOURTH_INPUT) {
    return RET_OK;
  }
  auto bias_tensor = in_tensors_[THIRD_INPUT];
  if (bias_tensor == nullptr) {
    MS_LOG(ERROR) << "bias_tensor invalid";
    return RET_ERROR;
  }
  auto bias_num = static_cast<size_t>(bias_tensor->ElementsNum());
  MS_CHECK_TRUE_RET(bias_num > 0, RET_ERROR);
  if (bias_num == 1) {
    // broadcast bias data
    size_t max_bias_data = CalBroadCastBiasDataElements();
    bias_ptr_ = reinterpret_cast<float *>(malloc(max_bias_data * sizeof(float)));
    if (bias_ptr_ == nullptr) {
      MS_LOG(ERROR) << "malloc bias_ptr_ failed";
      return RET_ERROR;
    }
    float broadcast_data = (reinterpret_cast<float *>(bias_tensor->data()))[0];
    // broadcast bias data
    for (size_t i = 0; i < max_bias_data; ++i) {
      bias_ptr_[i] = broadcast_data;
    }
    return RET_OK;
  }

  auto max_bias_data = static_cast<size_t>(UP_ROUND(bias_num, col_tile_));
  // malloc addr need to aligned to 32 bytes
  bias_ptr_ = reinterpret_cast<float *>(malloc(max_bias_data * static_cast<int>(sizeof(float))));
  if (bias_ptr_ == nullptr) {
    MS_LOG(ERROR) << "malloc bias_ptr_ failed";
    return RET_ERROR;
  }
  memcpy(bias_ptr_, bias_tensor->data(), bias_num * static_cast<int>(sizeof(float)));
  memset(bias_ptr_ + bias_num, 0, (max_bias_data - bias_num) * sizeof(float));
  return RET_OK;
}

int MatmulFp32BaseCPUKernel::InitMatrixA(const float *src_ptr) const {
  CHECK_NULL_RETURN(src_ptr);
  if (vec_matmul_) {
    return RET_OK;
  }
  for (int i = 0; i < a_batch_; i++) {
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

int MatmulFp32BaseCPUKernel::InitMatrixB(const float *src_ptr) const {
  CHECK_NULL_RETURN(src_ptr);
  for (int i = 0; i < b_batch_; i++) {
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
  if (!vec_matmul_ && !op_parameter_->is_train_session_ && a_pack_ptr_ != nullptr) {
    ms_context_->allocator->Free(a_pack_ptr_);
  }
  a_pack_ptr_ = nullptr;
}

void MatmulFp32BaseCPUKernel::FreeResizeBufB() {
  if (!op_parameter_->is_train_session_ && b_pack_ptr_ != nullptr) {
    ms_context_->allocator->Free(b_pack_ptr_);
  }
  b_pack_ptr_ = nullptr;
}

int MatmulFp32BaseCPUKernel::ParallelRunByBatch(int task_id) const {
  int start_batch = task_id * batch_stride_;
  int end_batch = MSMIN(params_->batch, start_batch + batch_stride_);
#if defined(ENABLE_AVX) || defined(ENABLE_AVX512)
  int col_step = params_->col_align_;
#else
  // col need not aligned
  int col_step = params_->col_;
#endif

  for (int index = start_batch; index < end_batch; ++index) {
    const float *a = a_pack_ptr_ + a_offset_[index] * params_->row_align_ * params_->deep_;
    const float *b = b_pack_ptr_ + b_offset_[index] * params_->deep_ * params_->col_align_;
    float *c = output_data_ + index * params_->row_ * col_step;

    auto bias = (bias_ptr_ == nullptr) ? nullptr : bias_ptr_;
    if (vec_matmul_) {
#if defined(ENABLE_AVX) || defined(ENABLE_AVX512)
      gemvCalFun(a, b, c, bias, params_->act_type_, params_->deep_, col_step, params_->col_align_);
#elif defined(ENABLE_ARM64)
      MatVecMulFp32Neon64(a, b, c, bias, params_->act_type_, params_->deep_, col_step, params_->col_align_);
#elif defined(ENABLE_ARM32)
      MatVecMulFp32Block4(a, b, c, bias, params_->act_type_, params_->deep_, col_step);
#else
      MatVecMulFp32Block8(a, b, c, bias, params_->act_type_, params_->deep_, col_step);
#endif
    } else {
#if defined(ENABLE_AVX512) || defined(ENABLE_AVX)
      gemmCalFun(a, b, c, bias, params_->act_type_, params_->deep_, col_step, params_->col_align_, params_->row_);
#else
      MatMulOpt(a, b, c, bias, params_->act_type_, params_->deep_, params_->row_, col_step, params_->col_,
                OutType_Nhwc);
#endif
    }
  }
  return RET_OK;
}

int MatmulFp32BaseCPUKernel::ParallelRunByOC(int task_id) const {
  int current_start_oc = task_id * oc_stride_ * col_tile_;
#if defined(ENABLE_AVX) || defined(ENABLE_AVX512)
  int current_rest_oc = params_->col_align_ - current_start_oc;
#else
  int current_rest_oc = params_->col_ - current_start_oc;
#endif
  int cur_oc = MSMIN(oc_stride_ * col_tile_, current_rest_oc);
  if (cur_oc <= 0) {
    return RET_OK;
  }

  auto b = batch_b_ptr_ + current_start_oc * params_->deep_;
  auto c = batch_c_ptr_ + current_start_oc;
  auto bias = (bias_ptr_ == nullptr) ? nullptr : bias_ptr_ + current_start_oc;
  if (vec_matmul_) {
#ifdef ENABLE_AVX512
    MatVecMulAvx512Fp32(batch_a_ptr_, b, c, bias, params_->act_type_, params_->deep_, cur_oc, params_->col_align_);
#elif defined(ENABLE_AVX)
    MatVecMulAvxFp32(batch_a_ptr_, b, c, bias, params_->act_type_, params_->deep_, cur_oc, params_->col_align_);
#elif defined(ENABLE_ARM64)
    int rest_align_col = MSMIN(params_->col_align_ - current_start_oc, oc_stride_ * col_tile_);
    MatVecMulFp32Neon64(batch_a_ptr_, b, c, bias, params_->act_type_, params_->deep_, cur_oc, rest_align_col);
#elif defined(ENABLE_ARM32)
    MatVecMulFp32Block4(batch_a_ptr_, b, c, bias, params_->act_type_, params_->deep_, cur_oc);
#else
    MatVecMulFp32Block8(batch_a_ptr_, b, c, bias, params_->act_type_, params_->deep_, cur_oc);
#endif
  } else {
#ifdef ENABLE_AVX512
    MatMulAvx512Fp32(batch_a_ptr_, b, c, bias, params_->act_type_, params_->deep_, cur_oc, params_->col_align_,
                     params_->row_);
#elif defined(ENABLE_AVX)
    MatMulAvxFp32(batch_a_ptr_, b, c, bias, params_->act_type_, params_->deep_, cur_oc, params_->col_align_,
                  params_->row_);
#else
    MatMulOpt(batch_a_ptr_, b, c, bias, params_->act_type_, params_->deep_, params_->row_, cur_oc, params_->col_,
              OutType_Nhwc);
#endif
  }
  return RET_OK;
}

void MatmulFp32BaseCPUKernel::init_global_variable() {
#ifdef ENABLE_AVX512
  matrix_a_pack_fun_ = params_->a_transpose_ ? RowMajor2ColMajor : RowMajor2RowMajor;
  matrix_b_pack_fun_ = params_->b_transpose_ ? RowMajor2Col64Major : RowMajor2Row64Major;
  row_tile_ = C1NUM;
  col_tile_ = C16NUM;
  gemmCalFun = MatMulAvx512Fp32;
  gemvCalFun = MatVecMulAvx512Fp32;
#elif defined(ENABLE_AVX)
  matrix_a_pack_fun_ = params_->a_transpose_ ? RowMajor2ColMajor : RowMajor2RowMajor;
  matrix_b_pack_fun_ = params_->b_transpose_ ? RowMajor2Col32Major : RowMajor2Row32Major;
  row_tile_ = C1NUM;
  col_tile_ = C8NUM;
  gemmCalFun = MatMulAvxFp32;
  gemvCalFun = MatVecMulAvxFp32;
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
#else  // ARM64
  matrix_a_pack_fun_ = params_->a_transpose_ ? RowMajor2Row12Major : RowMajor2Col12Major;
  matrix_b_pack_fun_ = params_->b_transpose_ ? RowMajor2Col8Major : RowMajor2Row8Major;
  row_tile_ = C12NUM;
  col_tile_ = C8NUM;
#endif
  params_->row_align_ = UP_ROUND(params_->row_, row_tile_);
  params_->col_align_ = UP_ROUND(params_->col_, col_tile_);
}

int MatmulFp32BaseCPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), C2NUM);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  init_global_variable();
  MS_CHECK_INT_MUL_NOT_OVERFLOW(a_batch_, params_->row_align_, RET_ERROR);
  MS_CHECK_INT_MUL_NOT_OVERFLOW(a_batch_ * params_->row_align_, params_->deep_, RET_ERROR);
  matrix_a_pack_size_ = a_batch_ * params_->row_align_ * params_->deep_;
  MS_CHECK_INT_MUL_NOT_OVERFLOW(a_batch_, params_->col_align_, RET_ERROR);
  MS_CHECK_INT_MUL_NOT_OVERFLOW(a_batch_ * params_->col_align_, params_->deep_, RET_ERROR);
  matrix_b_pack_size_ = b_batch_ * params_->col_align_ * params_->deep_;
  if (matrix_a_pack_size_ < 0 || matrix_b_pack_size_ < 0) {
    MS_LOG(ERROR) << "Matrix pack size is negative "
                  << "matrix_a_pack_size=" << matrix_a_pack_size_ << "matrix_b_pack_size=" << matrix_b_pack_size_;
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
    auto b_tensor = in_tensors_[1];
    CHECK_NULL_RETURN(b_tensor);
    if (InitBufferB() != RET_OK) {
      return RET_ERROR;
    }
    if (InitMatrixB(static_cast<float *>(b_tensor->data())) != RET_OK) {
      MS_LOG(ERROR) << "InitMatrixB failed!";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

int MatmulFp32BaseCPUKernel::ReSize() {
  ResizeParameter();
  matrix_a_pack_size_ = a_batch_ * params_->row_align_ * params_->deep_;
  matrix_b_pack_size_ = b_batch_ * params_->col_align_ * params_->deep_;
  if (matrix_a_pack_size_ < 0 || matrix_b_pack_size_ < 0) {
    MS_LOG(ERROR) << "Matrix pack size is negative "
                  << "matrix_a_pack_size=" << matrix_a_pack_size_ << "matrix_b_pack_size=" << matrix_b_pack_size_;
    return RET_ERROR;
  }
  if (op_parameter_->is_train_session_) {
    set_workspace_size((matrix_a_pack_size_ + matrix_b_pack_size_) * static_cast<int>(sizeof(float)));
  }
  GetThreadCuttingPolicy();
  return RET_OK;
}

void MatmulFp32BaseCPUKernel::ResizeParameter() {
  init_global_variable();
  if (params_->row_ == 1) {
    vec_matmul_ = true;
    row_tile_ = 1;
  } else {
    vec_matmul_ = false;
  }
  params_->row_align_ = UP_ROUND(params_->row_, row_tile_);
  oc_res_ = params_->col_ % col_tile_;
}

int MatmulFp32BaseCPUKernel::InitTmpOutBuffer() {
  auto out_data = reinterpret_cast<float *>(out_tensors_.front()->data());
  MS_ASSERT(out_data != nullptr);
#if defined(ENABLE_AVX) || defined(ENABLE_AVX512)
  if (oc_res_ != 0) {  // avx matmul need to malloc dst aligned to C8NUM and avx512 need to aligned to C16NUM
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

void MatmulFp32BaseCPUKernel::GetThreadCuttingPolicy() {
  if (params_->batch >= op_parameter_->thread_num_) {
    thread_count_ = op_parameter_->thread_num_;
    batch_stride_ = UP_DIV(params_->batch, thread_count_);
    batch_split_ = true;
    parallel_fun_ = &MatmulFp32BaseCPUKernel::ParallelRunByBatch;
  } else {
    thread_count_ = MSMIN(op_parameter_->thread_num_, UP_DIV(params_->col_align_, col_tile_));
#if defined(ENABLE_AVX) || defined(ENABLE_AVX512)  // thread tile by col_tile * C4NUM
    oc_stride_ = UP_DIV(UP_DIV(params_->col_align_, col_tile_ * C4NUM), thread_count_) * C4NUM;
#else
    oc_stride_ = UP_DIV(UP_DIV(params_->col_align_, col_tile_), thread_count_);
#endif
    batch_split_ = false;
    parallel_fun_ = &MatmulFp32BaseCPUKernel::ParallelRunByOC;
  }
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

  if (batch_split_) {
    ret = ParallelLaunch(this->ms_context_, MatmulRun, this, thread_count_);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "MatmulRun failed in split by batch";
      return ret;
    }
  } else {
#if defined(ENABLE_AVX) || defined(ENABLE_AVX512)
    int col_step = params_->col_align_;
#else
    // need not aligned
    int col_step = params_->col_;
#endif
    for (int i = 0; i < params_->batch; ++i) {
      batch_a_ptr_ = a_pack_ptr_ + a_offset_[i] * params_->row_align_ * params_->deep_;
      batch_b_ptr_ = b_pack_ptr_ + b_offset_[i] * params_->deep_ * params_->col_align_;
      batch_c_ptr_ = output_data_ + i * params_->row_ * col_step;
      ret = ParallelLaunch(this->ms_context_, MatmulRun, this, thread_count_);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "MatmulRun failed in split by oc";
        return ret;
      }
    }
  }

#if defined(ENABLE_AVX) || defined(ENABLE_AVX512)
  if (oc_res_ != 0) {
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
