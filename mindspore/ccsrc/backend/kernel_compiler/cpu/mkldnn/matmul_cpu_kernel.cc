/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#include "backend/kernel_compiler/cpu/mkldnn/matmul_cpu_kernel.h"
#include <utility>
#include "common/thread_pool.h"
#include "backend/kernel_compiler/cpu/mkldnn/mkl_kernel_engine.h"
#include "backend/kernel_compiler/cpu/nnacl/op_base.h"
#include "runtime/device/cpu/cpu_device_address.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kMatMulInputsNum = 2;
constexpr size_t kMatMulOutputsNum = 1;
const size_t kIndexOffset = 2;
}  // namespace

void MatMulCPUKernel::InitTile() {
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
}

void MatMulCPUKernel::InitMatrixA(const float *src_ptr) {
  const size_t size = param_.batch * param_.row_align_ * param_.deep_;
  a_pack_ptr_ = new (std::nothrow) float[size];
  if (a_pack_ptr_ == nullptr) {
    MS_LOG(EXCEPTION) << "MatMul new a_pack_ptr_ failed.";
  }

  if (vec_matmul_) {
    const size_t count = size * sizeof(float);
    if (memcpy_s(a_pack_ptr_, count, src_ptr, count) != EOK) {
      FreeBuffer();
      MS_LOG(EXCEPTION) << "MatMul memcpy a_pack_ptr_ failed.";
    }
    return;
  }

  for (int i = 0; i < param_.batch; i++) {
    const float *src = src_ptr + i * param_.row_ * param_.deep_;
    float *dst = a_pack_ptr_ + i * param_.row_align_ * param_.deep_;
#ifdef ENABLE_AVX
    if (param_.a_transpose_) {
      RowMajor2Row6Major(src, dst, param_.deep_, param_.row_);
    } else {
      RowMajor2Col6Major(src, dst, param_.row_, param_.deep_);
    }
#elif defined(ENABLE_SSE)
    if (param_.a_transpose_) {
      RowMajor2Row4Major(src, dst, param_.deep_, param_.row_);
    } else {
      RowMajor2Col4Major(src, dst, param_.row_, param_.deep_);
    }
#else
    if (param_.a_transpose_) {
      RowMajor2Row12Major(src, dst, param_.deep_, param_.row_);
    } else {
      RowMajor2Col12Major(src, dst, param_.row_, param_.deep_);
    }
#endif
  }
}

void MatMulCPUKernel::InitMatrixB(const float *src_ptr) {
  const size_t size = param_.batch * param_.col_align_ * param_.deep_;
  b_pack_ptr_ = new (std::nothrow) float[size];
  if (b_pack_ptr_ == nullptr) {
    FreeBuffer();
    MS_LOG(EXCEPTION) << "MatMul new b_pack_ptr_ failed";
  }
  if (vec_matmul_) {
    if (param_.b_transpose_) {
      const size_t count = size * sizeof(float);
      if (memcpy_s(b_pack_ptr_, count, src_ptr, count) != EOK) {
        FreeBuffer();
        MS_LOG(EXCEPTION) << "MatMul memcpy b_pack_ptr_ failed.";
      }
    } else {
      for (int i = 0; i < param_.batch; i++) {
        const float *src = src_ptr + i * param_.deep_ * param_.col_;
        float *dst = b_pack_ptr_ + i * param_.deep_ * param_.col_;
        RowMajor2ColMajor(src, dst, param_.deep_, param_.col_);
      }
    }
    return;
  }

  for (int i = 0; i < param_.batch; i++) {
    const float *src = src_ptr + i * param_.deep_ * param_.col_;
    float *dst = b_pack_ptr_ + i * param_.deep_ * param_.col_align_;
#ifdef ENABLE_AVX
    if (param_.b_transpose_) {
      RowMajor2Col16Major(src, dst, param_.col_, param_.deep_);
    } else {
      RowMajor2Row16Major(src, dst, param_.deep_, param_.col_);
    }
#elif defined(ENABLE_ARM32)
    if (param_.b_transpose_) {
      RowMajor2Col4Major(src, dst, param_.col_, param_.deep_);
    } else {
      RowMajor2Row4Major(src, dst, param_.deep_, param_.col_);
    }
#else
    if (param_.b_transpose_) {
      RowMajor2Col8Major(src, dst, param_.col_, param_.deep_);
    } else {
      RowMajor2Row8Major(src, dst, param_.deep_, param_.col_);
    }
#endif
  }
}

void MatMulCPUKernel::InitArmKernel(bool trans_a, bool trans_b, const std::vector<size_t> &a_shape,
                                    const std::vector<size_t> &o_shape) {
  InitTile();
  param_.batch = SizeToInt(batch_);
  param_.a_transpose_ = trans_a;
  param_.b_transpose_ = trans_b;
  param_.row_ = SizeToInt(o_shape[rank_ - kIndexOffset]);
  param_.deep_ = SizeToInt(trans_a ? a_shape[rank_ - kIndexOffset] : a_shape[rank_ - 1]);
  param_.col_ = SizeToInt(o_shape[rank_ - 1]);
  vec_matmul_ = (param_.row_ == 1);
  param_.row_align_ = vec_matmul_ ? 1 : UP_ROUND(param_.row_, row_tile_);
  param_.col_align_ = vec_matmul_ ? param_.col_ : UP_ROUND(param_.col_, col_tile_);
  size_t max_thread_num = common::ThreadPool::GetInstance().GetSyncRunThreadNum();
  thread_count_ = MSMIN(max_thread_num, UP_DIV(param_.col_align_, col_tile_));
  thread_stride_ = UP_DIV(UP_DIV(param_.col_align_, col_tile_), thread_count_);
}

void MatMulCPUKernel::InitX64Kernel(bool trans_a, bool trans_b, const std::vector<size_t> &a_shape,
                                    const std::vector<size_t> &b_shape, const std::vector<size_t> &o_shape) {
  size_mat_a_ = a_shape[rank_ - kIndexOffset] * a_shape[rank_ - 1];
  size_mat_b_ = b_shape[rank_ - kIndexOffset] * b_shape[rank_ - 1];
  size_mat_o_ = o_shape[rank_ - kIndexOffset] * o_shape[rank_ - 1];
  if (trans_a) {
    trans_a_ = TRANSPOSE_YES;
    dim_k_ = static_cast<dnnl_dim_t>(a_shape[rank_ - kIndexOffset]);
  } else {
    dim_k_ = static_cast<dnnl_dim_t>(a_shape[rank_ - 1]);
  }
  if (trans_b) {
    trans_b_ = TRANSPOSE_YES;
  }
  dim_m_ = static_cast<dnnl_dim_t>(o_shape[rank_ - kIndexOffset]);
  dim_n_ = static_cast<dnnl_dim_t>(o_shape[rank_ - 1]);
}

void MatMulCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = AnfAlgo::GetCNodeName(kernel_node);
  std::vector<size_t> a_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  std::vector<size_t> b_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 1);
  std::vector<size_t> o_shape = AnfAlgo::GetOutputDeviceShape(kernel_node, 0);
  const size_t rank_min = 2;
  if (a_shape.size() < rank_min || b_shape.size() < rank_min || o_shape.size() < rank_min) {
    MS_LOG(EXCEPTION) << "The tensor rank of MatMul should be greater than or equal to 2.";
  }
  bool trans_a = AnfAlgo::GetNodeAttr<bool>(kernel_node, TRANSPOSE_A);
  bool trans_b = AnfAlgo::GetNodeAttr<bool>(kernel_node, TRANSPOSE_B);
  rank_ = a_shape.size();
  batch_ = 1;
  for (size_t i = 0; i < rank_ - kIndexOffset; ++i) {
    batch_ *= a_shape[i];
  }
#ifdef ENABLE_ARM
  InitArmKernel(trans_a, trans_b, a_shape, o_shape);
#else
  InitX64Kernel(trans_a, trans_b, a_shape, b_shape, o_shape);
#endif
}

int MatMulCPUKernel::FloatRun(size_t task_id) const {
  size_t current_stride_oc = thread_stride_ * col_tile_;
  if (IntToSize(param_.col_) <= task_id * current_stride_oc) {
    return common::SUCCESS;
  }

  size_t current_rest_oc = IntToSize(param_.col_) - task_id * current_stride_oc;
  size_t cur_oc = MSMIN(current_stride_oc, current_rest_oc);
  auto b = batch_b_ptr_ + task_id * thread_stride_ * col_tile_ * IntToSize(param_.deep_);
  auto output = batch_o_ptr_ + task_id * thread_stride_ * col_tile_;
  float *bias = nullptr;
  if (vec_matmul_) {
    MatVecMulFp32(batch_a_ptr_, b, output, bias, param_.act_type_, param_.deep_, SizeToInt(cur_oc));
  } else {
    MatMulOpt(batch_a_ptr_, b, output, bias, param_.act_type_, param_.deep_, param_.row_, SizeToInt(cur_oc),
              param_.col_, OutType_Nhwc);
  }
  return common::SUCCESS;
}

void MatMulCPUKernel::ParallelRun(float *output) {
  for (int i = 0; i < param_.batch; ++i) {
    if (vec_matmul_) {
      batch_a_ptr_ = a_pack_ptr_ + i * param_.deep_;
      batch_b_ptr_ = b_pack_ptr_ + i * param_.deep_ * param_.col_;
      batch_o_ptr_ = output + i * param_.row_ * param_.col_;
    } else {
      batch_a_ptr_ = a_pack_ptr_ + i * param_.row_align_ * param_.deep_;
      batch_b_ptr_ = b_pack_ptr_ + i * param_.deep_ * param_.col_align_;
      batch_o_ptr_ = output + i * param_.row_ * param_.col_;
    }
    std::vector<common::Task> tasks;
    size_t thread_index = 0;
    while (thread_index < thread_count_) {
      (void)tasks.emplace_back(std::bind(&MatMulCPUKernel::FloatRun, this, thread_index));
      thread_index++;
    }
    (void)common::ThreadPool::GetInstance().SyncRun(tasks);
  }
}

void MatMulCPUKernel::LaunchARM(const float *input_a, const float *input_b, float *output) {
  InitMatrixA(input_a);
  InitMatrixB(input_b);
  ParallelRun(output);
  FreeBuffer();
}

void MatMulCPUKernel::LaunchX64(const float *input_a, const float *input_b, float *output) const {
  dnnl_dim_t lda = (trans_a_ == TRANSPOSE_YES ? dim_m_ : dim_k_);
  dnnl_dim_t ldb = (trans_b_ == TRANSPOSE_YES ? dim_k_ : dim_n_);
  dnnl_dim_t ldc = dim_n_;
  float alpha = 1.0;
  float beta = 0.0;
  for (size_t i = 0; i < batch_; i++) {
    (void)dnnl_sgemm(trans_a_, trans_b_, dim_m_, dim_n_, dim_k_, alpha, input_a + i * size_mat_a_, lda,
                     input_b + i * size_mat_b_, ldb, beta, output + i * size_mat_o_, ldc);
  }
}

bool MatMulCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &,
                             const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kMatMulInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kMatMulOutputsNum, kernel_name_);
  const auto input_a = reinterpret_cast<float *>(inputs[0]->addr);
  const auto input_b = reinterpret_cast<float *>(inputs[1]->addr);
  auto output = reinterpret_cast<float *>(outputs[0]->addr);

#ifdef ENABLE_ARM
  LaunchARM(input_a, input_b, output);
#else
  LaunchX64(input_a, input_b, output);
#endif
  return true;
}

void MatMulCPUKernel::FreeBuffer() {
  if (a_pack_ptr_ != nullptr) {
    delete[] a_pack_ptr_;
    a_pack_ptr_ = nullptr;
  }
  if (b_pack_ptr_ != nullptr) {
    delete[] b_pack_ptr_;
    b_pack_ptr_ = nullptr;
  }
}
}  // namespace kernel
}  // namespace mindspore
