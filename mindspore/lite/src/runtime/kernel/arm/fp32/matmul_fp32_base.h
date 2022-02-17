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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_MATMUL_FP32_BASE_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_MATMUL_FP32_BASE_H_

#include <vector>
#include "src/inner_kernel.h"
#ifdef SERVER_INFERENCE
#include "src/pack_weight_manager.h"
#endif
#include "nnacl/matmul_parameter.h"
#include "include/errorcode.h"
#include "src/common/common.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_MEMORY_FAILED;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
using MatrixPackFun = void (*)(const float *src_ptr, float *dst_ptr, int row, int col);
using GemmFun = void (*)(const float *a, const float *b, float *c, const float *bias, const int act_type,
                         const int depth, const int cur_col, const int col_align, const int row);
using GemvFun = void (*)(const float *a, const float *b, float *c, const float *bias, const int act_type,
                         const int depth, const int cur_col, const int col_align);
using GemmIsNotPackFun = void (*)(const float *a, const float *b, float *c, const float *bias, int m, int k);

class MatmulFp32BaseCPUKernel : public InnerKernel {
 public:
  MatmulFp32BaseCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                          const std::vector<lite::Tensor *> &outputs, const mindspore::lite::InnerContext *ctx)
      : InnerKernel(parameter, inputs, outputs, ctx) {
    params_ = reinterpret_cast<MatMulParameter *>(op_parameter_);
  }
  ~MatmulFp32BaseCPUKernel() override;
  int Prepare() override;
  int ReSize() override;
  int Run() override;

  using ParallelRun = int (MatmulFp32BaseCPUKernel::*)(int task_id) const;
  ParallelRun parallel_fun_ = nullptr;

 private:
  struct MatrixInfo {
    bool need_pack;
    bool has_packed;  // only valid for constant, only do once throughout the process.
    bool has_origin;  // only valid for constant, only true when failing to infer shape, then false after packing.
    int pack_size;
    float *origin_ptr;  // only valid for constant, which is synchronized with the 'has_origin'.
    float *pack_ptr;
    MatrixInfo()
        : need_pack(false),
          has_packed(false),
          has_origin(false),
          pack_size(-1),
          origin_ptr(nullptr),
          pack_ptr(nullptr) {}
  };

#if defined(ENABLE_AVX) || defined(ENABLE_AVX512) || defined(ENABLE_ARM64)
  int ParallelRunByRow(int task_id) const;
#endif
  int ParallelRunByOC(int task_id) const;
  int ParallelRunByBatch(int task_id) const;
  int ParallelRunIsNotPackByBatch(int task_id) const;
  int BackupConstMatrix(MatrixInfo *matrix_info, int index);
  void InitGlobalVariable();
  int PackMatrixA();
  int PackMatrixB();
  int PackBiasMatrix();
  void FreePackedMatrixA();
  void FreePackedMatrixB();
  int InitParameter();
  int InitTmpOutBuffer();
  int GetThreadCuttingPolicy();
  bool CheckThreadCuttingByRow();
  void GetThreadCuttingInfoByRow();

 protected:
  MatMulParameter *params_ = nullptr;
  int a_batch_ = 1;
  int b_batch_ = 1;
  std::vector<int> a_offset_;
  std::vector<int> b_offset_;

 private:
  int col_tile_ = 0;
  int row_tile_ = 0;
  int batch_stride_ = 0;
  int oc_stride_ = 0;
  int thread_count_ = 0;
  float *output_data_ = nullptr;
  bool out_need_aligned_ = false;
  int col_step_ = 0;
#if defined(ENABLE_AVX) || defined(ENABLE_AVX512)
  GemmFun gemmCalFun = nullptr;
  GemvFun gemvCalFun = nullptr;
#endif
  GemmIsNotPackFun gemmIsNotPackFun = nullptr;
  int row_num_;
  std::vector<int> row_split_points_;
  MatrixInfo matrix_a_;
  MatrixInfo matrix_b_;
  MatrixInfo matrix_c_;
  MatrixPackFun matrix_a_pack_fun_ = nullptr;
  MatrixPackFun matrix_b_pack_fun_ = nullptr;
};
}  // namespace mindspore::kernel
#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_MATMUL_FP32_BASE_H_
