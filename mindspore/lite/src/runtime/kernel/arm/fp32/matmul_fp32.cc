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

#include "src/runtime/kernel/arm/fp32/matmul_fp32.h"
#include "include/errorcode.h"
#include "nnacl/fp32/matmul_fp32.h"
#include "src/runtime/runtime_api.h"
#include "src/kernel_registry.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_INPUT_TENSOR_ERROR;
using mindspore::lite::RET_MEMORY_FAILED;
using mindspore::lite::RET_OK;

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::schema::PrimitiveType_MatMul;

namespace mindspore::kernel {
MatmulCPUKernel::~MatmulCPUKernel() {
  if (a_pack_ptr_ != nullptr) {
    free(a_pack_ptr_);
    a_pack_ptr_ = nullptr;
  }
  if (b_pack_ptr_ != nullptr) {
    free(b_pack_ptr_);
    b_pack_ptr_ = nullptr;
  }
  if (bias_ptr_ != nullptr) {
    free(bias_ptr_);
    bias_ptr_ = nullptr;
  }
}

void MatmulCPUKernel::FreeTmpBuffer() {
  if (a_pack_ptr_ != nullptr) {
    params_->a_const_ ? free(a_pack_ptr_) : context_->allocator->Free(a_pack_ptr_);
    a_pack_ptr_ = nullptr;
  }
  if (b_pack_ptr_ != nullptr) {
    params_->b_const_ ? free(b_pack_ptr_) : context_->allocator->Free(b_pack_ptr_);
    b_pack_ptr_ = nullptr;
  }
}

int MatmulCPUKernel::MallocMatrixABuffer() {
  auto a_shape = in_tensors_.at(0)->shape();
  int batch = 1;
  MS_ASSERT(a_shape.size() >= 2);
  for (size_t i = 0; i < a_shape.size() - 2; ++i) {
    batch *= a_shape[i];
  }
  params_->batch = batch;
  params_->row_ = params_->a_transpose_ ? a_shape[a_shape.size() - 1] : a_shape[a_shape.size() - 2];
#ifdef ENABLE_ARM
  if (params_->a_init_shape_ && params_->row_ == 1) {
    is_vector_a_ = true;
  } else {
    is_vector_a_ = false;
  }
#endif
  params_->deep_ = params_->a_transpose_ ? a_shape[a_shape.size() - 2] : a_shape[a_shape.size() - 1];
#ifdef ENABLE_AVX
  params_->row_align_ = UP_ROUND(params_->row_, C6NUM);
#elif defined(ENABLE_SSE)
  params_->row_align_ = UP_ROUND(params_->row_, C4NUM);
#else
  params_->row_align_ = UP_ROUND(params_->row_, C12NUM);
#endif

  int row_tmp = is_vector_a_ ? 1 : params_->row_align_;
  if (params_->a_const_) {
    a_pack_ptr_ = reinterpret_cast<float *>(malloc(params_->batch * row_tmp * params_->deep_ * sizeof(float)));
  } else {
    a_pack_ptr_ =
      reinterpret_cast<float *>(context_->allocator->Malloc(params_->batch * row_tmp * params_->deep_ * sizeof(float)));
  }
  if (a_pack_ptr_ == nullptr) {
    FreeTmpBuffer();
    return RET_MEMORY_FAILED;
  }

  return RET_OK;
}

int MatmulCPUKernel::MallocMatrixBBuffer() {
  auto b_shape = in_tensors_.at(1)->shape();
  if (b_shape.empty()) {
    return RET_OK;
  }
  int batch = 1;
  MS_ASSERT(b_shape.size() >= 2);
  for (size_t i = 0; i < b_shape.size() - 2; ++i) {
    batch *= b_shape[i];
  }
  params_->batch = batch;
  params_->col_ = params_->b_transpose_ ? b_shape[b_shape.size() - 2] : b_shape[b_shape.size() - 1];
  params_->col_align_ = UP_ROUND(params_->col_, col_tile_);
  params_->deep_ = params_->b_transpose_ ? b_shape[b_shape.size() - 1] : b_shape[b_shape.size() - 2];

  int col_tmp = is_vector_a_ ? params_->col_ : params_->col_align_;
  if (params_->b_const_) {
    b_pack_ptr_ = reinterpret_cast<float *>(malloc(params_->batch * col_tmp * params_->deep_ * sizeof(float)));
  } else {
    b_pack_ptr_ =
      reinterpret_cast<float *>(context_->allocator->Malloc(params_->batch * col_tmp * params_->deep_ * sizeof(float)));
  }
  if (b_pack_ptr_ == nullptr) {
    FreeTmpBuffer();
    return RET_MEMORY_FAILED;
  }

  thread_count_ = MSMIN(op_parameter_->thread_num_, UP_DIV(params_->col_align_, col_tile_));
  thread_stride_ = UP_DIV(UP_DIV(params_->col_align_, col_tile_), thread_count_);
  return RET_OK;
}

int MatmulCPUKernel::InitBias() {
  auto b_shape = in_tensors_.at(1)->shape();
  auto c_shape = out_tensors_.at(0)->shape();
  params_->col_ = params_->b_const_
                    ? (params_->b_transpose_ ? b_shape.at(b_shape.size() - 2) : b_shape.at(b_shape.size() - 1))
                    : (c_shape.at(c_shape.size() - 1));
  params_->col_align_ = UP_ROUND(params_->col_, col_tile_);
  auto col_tmp = is_vector_a_ ? params_->col_ : params_->col_align_;
  if (bias_ptr_ == nullptr) {
    bias_ptr_ = reinterpret_cast<float *>(malloc(col_tmp * sizeof(float)));
    if (bias_ptr_ == nullptr) {
      FreeTmpBuffer();
      return RET_MEMORY_FAILED;
    }
  }
  memset(bias_ptr_, 0, col_tmp * sizeof(float));
  if (in_tensors_.size() == 3) {
    memcpy(bias_ptr_, in_tensors_[2]->data_c(), in_tensors_[2]->ElementsNum() * sizeof(float));
  }
  return RET_OK;
}

int MatmulCPUKernel::ReSize() {
  if (!params_->b_const_) {
    free(bias_ptr_);
    bias_ptr_ = nullptr;
    auto ret = InitBias();
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Matmul fp32 init bias failed";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

void MatmulCPUKernel::InitMatrixA(const float *src_ptr, float *dst_ptr) {
  if (is_vector_a_) {
    memcpy(dst_ptr, src_ptr, params_->batch * params_->deep_ * sizeof(float));
    return;
  }

  for (int i = 0; i < params_->batch; i++) {
    const float *src = src_ptr + i * params_->deep_ * params_->row_;
    float *dst = dst_ptr + i * params_->deep_ * params_->row_align_;
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
  return;
}

void MatmulCPUKernel::InitMatrixB(const float *src_ptr, float *dst_ptr) {
  if (is_vector_a_) {
    if (params_->b_transpose_) {
      memcpy(dst_ptr, src_ptr, params_->batch * params_->col_ * params_->deep_ * sizeof(float));
    } else {
      for (int i = 0; i < params_->batch; i++) {
        const float *src = src_ptr + i * params_->deep_ * params_->col_;
        float *dst = dst_ptr + i * params_->deep_ * params_->col_;
        RowMajor2ColMajor(src, dst, params_->deep_, params_->col_);
      }
    }
    return;
  }

  for (int i = 0; i < params_->batch; i++) {
    const float *src = src_ptr + i * params_->deep_ * params_->col_;
    float *dst = dst_ptr + i * params_->deep_ * params_->col_align_;
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
  return;
}

int MatmulCPUKernel::Init() {
#ifdef ENABLE_AVX
  col_tile_ = C16NUM;
#elif defined(ENABLE_ARM32)
  col_tile_ = C4NUM;
#else
  col_tile_ = C8NUM;
#endif
  params_->a_const_ = (in_tensors_.at(0)->data_c() != nullptr);
  params_->b_const_ = (in_tensors_.at(1)->data_c() != nullptr);
  if (params_->a_const_) {
    auto ret = MallocMatrixABuffer();
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Matmul fp32 malloc matrix Ａ buffer failed";
      return RET_ERROR;
    }
    InitMatrixA(reinterpret_cast<float *>(in_tensors_.at(0)->data_c()), a_pack_ptr_);
    a_ptr_ = a_pack_ptr_;
  }
  if (params_->b_const_) {
    auto ret = MallocMatrixBBuffer();
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Matmul fp32 malloc matrix B buffer failed";
      return RET_ERROR;
    }
    InitMatrixB(reinterpret_cast<float *>(in_tensors_.at(1)->data_c()), b_pack_ptr_);
    b_ptr_ = b_pack_ptr_;
    // init bias
    ret = InitBias();
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Matmul fp32 init bias failed";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

int MatmulCPUKernel::RunImpl(int task_id) {
  int cur_oc = MSMIN(thread_stride_ * col_tile_, params_->col_ - task_id * thread_stride_ * col_tile_);
  if (cur_oc <= 0) {
    return RET_OK;
  }
  auto b = cur_b_ptr_ + task_id * thread_stride_ * col_tile_ * params_->deep_;
  auto c = cur_c_ptr_ + task_id * thread_stride_ * col_tile_;
  auto bias = bias_ptr_ ? bias_ptr_ + task_id * thread_stride_ * col_tile_ : NULL;
  MS_ASSERT(cur_a_ptr_);
  MS_ASSERT(b);
  MS_ASSERT(c);
  if (is_vector_a_) {
    MatVecMul(cur_a_ptr_, b, c, bias, ActType_No, params_->deep_, cur_oc);
  } else {
    MatMulOpt(cur_a_ptr_, b, c, bias, ActType_No, params_->deep_, params_->row_, cur_oc, params_->col_, OutType_Nhwc);
  }
  return RET_OK;
}

int MatmulFloatRun(void *cdata, int task_id) {
  auto op = reinterpret_cast<MatmulCPUKernel *>(cdata);
  auto error_code = op->RunImpl(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "MatmulFp32Run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int MatmulCPUKernel::Run() {
  auto a_src = reinterpret_cast<float *>(in_tensors_.at(0)->data_c());
  auto b_src = reinterpret_cast<float *>(in_tensors_.at(1)->data_c());
  auto c_src = reinterpret_cast<float *>(out_tensors_.at(0)->data_c());

  if (!params_->a_const_ || IsTrain()) {
    if (a_pack_ptr_ != nullptr) {
      params_->a_const_ ? free(a_pack_ptr_) : context_->allocator->Free(a_pack_ptr_);
      a_pack_ptr_ = nullptr;
    }
    auto ret = MallocMatrixABuffer();
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Matmul fp32 malloc matrix a buffer failed";
      return RET_ERROR;
    }
    if (is_vector_a_) {
      a_ptr_ = a_src;
    } else {
      InitMatrixA(a_src, a_pack_ptr_);
      a_ptr_ = a_pack_ptr_;
    }
  }
  if (!params_->b_const_ || IsTrain()) {
    if (b_pack_ptr_ != nullptr) {
      params_->b_const_ ? free(b_pack_ptr_) : context_->allocator->Free(b_pack_ptr_);
      b_pack_ptr_ = nullptr;
    }
    auto ret = MallocMatrixBBuffer();
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Matmul fp32 malloc matrix b buffer failed";
      return RET_ERROR;
    }
    if (is_vector_a_ && params_->b_transpose_) {
      b_ptr_ = b_src;
    } else {
      InitMatrixB(b_src, b_pack_ptr_);
      b_ptr_ = b_pack_ptr_;
    }
  }
  if (IsTrain()) {
    InitBias();
  }
  for (int i = 0; i < params_->batch; ++i) {
    if (is_vector_a_) {
      cur_a_ptr_ = a_ptr_ + i * params_->deep_;
      cur_b_ptr_ = b_ptr_ + i * params_->deep_ * params_->col_;
      cur_c_ptr_ = c_src + i * params_->row_ * params_->col_;
    } else {
      cur_a_ptr_ = a_ptr_ + i * params_->row_align_ * params_->deep_;
      cur_b_ptr_ = b_ptr_ + i * params_->deep_ * params_->col_align_;
      cur_c_ptr_ = c_src + i * params_->row_ * params_->col_;
    }
    auto ret = ParallelLaunch(this->context_->thread_pool_, MatmulFloatRun, this, thread_count_);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Matmul fp32 run function MatmulFloatRun failed";
      FreeTmpBuffer();
      return RET_ERROR;
    }
  }
  if (!params_->a_const_ || IsTrain()) {
    params_->a_const_ ? free(a_pack_ptr_) : context_->allocator->Free(a_pack_ptr_);
    a_pack_ptr_ = nullptr;
  }
  if (!params_->b_const_ || IsTrain()) {
    params_->b_const_ ? free(b_pack_ptr_) : context_->allocator->Free(b_pack_ptr_);
    b_pack_ptr_ = nullptr;
  }
  return RET_OK;
}

int MatmulCPUKernel::Eval() {
  // Copy weights after training
  auto a_src = reinterpret_cast<float *>(in_tensors_.at(0)->data_c());
  auto b_src = reinterpret_cast<float *>(in_tensors_.at(1)->data_c());
  LiteKernel::Eval();
  if (params_->a_const_) {
    if (a_pack_ptr_ == nullptr) {
      auto ret = MallocMatrixABuffer();
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "Matmul fp32 malloc matrix a buffer failed";
        return RET_ERROR;
      }
    }
    if (is_vector_a_) {
      a_ptr_ = a_src;
    } else {
      InitMatrixA(a_src, a_pack_ptr_);
      a_ptr_ = a_pack_ptr_;
    }
  }
  if (params_->b_const_) {
    if (b_pack_ptr_ == nullptr) {
      auto ret = MallocMatrixBBuffer();
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "Matmul fp32 malloc matrix b buffer failed";
        return RET_ERROR;
      }
    }
    if (is_vector_a_ && params_->b_transpose_) {
      b_ptr_ = b_src;
    } else {
      InitMatrixB(b_src, b_pack_ptr_);
      b_ptr_ = b_pack_ptr_;
    }
  }
  InitBias();
  return RET_OK;
}

kernel::LiteKernel *CpuMatmulFp32KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                               const std::vector<lite::Tensor *> &outputs, OpParameter *opParameter,
                                               const lite::InnerContext *ctx, const kernel::KernelKey &desc,
                                               const mindspore::lite::PrimitiveC *primitive) {
  MS_ASSERT(opParameter != nullptr);
  MS_ASSERT(desc.type == schema::PrimitiveType_MatMul);
  auto kernel = new (std::nothrow) MatmulCPUKernel(opParameter, inputs, outputs, ctx, primitive);
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

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_MatMul, CpuMatmulFp32KernelCreator)
}  // namespace mindspore::kernel
