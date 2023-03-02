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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP32_MATMUL_FP32_BASE_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP32_MATMUL_FP32_BASE_H_

#include <vector>
#include "src/litert/lite_kernel.h"
#include "src/litert/pack_weight_manager.h"
#include "nnacl/matmul_parameter.h"
#include "include/errorcode.h"
#include "src/common/common.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_MEMORY_FAILED;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
using MatrixPackFun = void (*)(const float *src_ptr, float *dst_ptr, int row, int col, int start_row, int end_row);
using GemmIsNotPackFun = void (*)(const float *a, const float *b, float *c, const float *bias, int m, int k,
                                  int act_type);

class MatmulFp32BaseCPUKernel : public LiteKernel {
 public:
  MatmulFp32BaseCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                          const std::vector<lite::Tensor *> &outputs, const mindspore::lite::InnerContext *ctx)
      : LiteKernel(parameter, inputs, outputs, ctx) {
    params_ = reinterpret_cast<MatMulParameter *>(op_parameter_);
    params_->matmul_type_ = kMatmulFp32BaseCpu;
  }
  ~MatmulFp32BaseCPUKernel() override;
  int Prepare() override;
  int FullConnectionPrepare();
  int MatmulPrepare();
  void SetConv1x1OriginWeight(float *conv1x1_origin_weight) { conv1x1_origin_weight_ = conv1x1_origin_weight; }
  void SetConv1x1OriginBias(float *conv1x1_origin_bias) { conv1x1_origin_bias_ = conv1x1_origin_bias; }
  int Conv1x1Prepare();
  int ReSize() override;
  int FullConnectionReSize();
  int MatmulReSize();
  int Conv1x1ReSize();
  int Run() override;
  static int InitBroadcastParams(const std::vector<int> &a_shape_const, const std::vector<int> &b_shape_const,
                                 MatMulParameter *params, std::vector<int> *a_offsets, std::vector<int> *b_offsets);
  int PackMatrixBParallelRunByBatch(int task_id) const;
  inline void SetSharingPack(bool is_sharing) { is_sharing_pack_ = is_sharing; }

  using ParallelRun = int (MatmulFp32BaseCPUKernel::*)(int task_id) const;
  ParallelRun parallel_fun_ = nullptr;
  void SetRunByGEMM() { parallel_fun_ = &MatmulFp32BaseCPUKernel::ParallelRunByGEMM; }
  void SetRunByGEPDOT() { parallel_fun_ = &MatmulFp32BaseCPUKernel::ParallelRunByGEPDOT; }
  void SetRunByGEPM() { parallel_fun_ = &MatmulFp32BaseCPUKernel::ParallelRunByGEPM; }
  void SetRunByBatchColRowGEMM() { parallel_fun_ = &MatmulFp32BaseCPUKernel::ParallelRunByBatchColRowGEMM; }
  void SetRunByRow1Deep1GEPDOT() { parallel_fun_ = &MatmulFp32BaseCPUKernel::ParallelRunByRow1Deep1GEPDOT; }

  virtual int ParallelRunByGEMM(int task_id) const { return RET_ERROR; }
  virtual int ParallelRunByGEPDOT(int task_id) const { return RET_ERROR; }
  virtual int ParallelRunByGEPM(int task_id) const { return RET_ERROR; }
  virtual int ParallelRunByBatchColRowGEMM(int task_id) const { return RET_ERROR; }
  virtual int ParallelRunByRow1Deep1GEPDOT(int task_id) const { return RET_ERROR; }
  virtual int GetThreadCuttingPolicy();

  const float *GetPackBPtr() const { return matrix_b_.pack_ptr; }
  const int GetBBatch() const { return b_batch_; }
  void SetWeightIsPacked(bool weight_is_packed) { this->weight_is_packed_ = weight_is_packed; }

 public:
  struct MatrixInfo {
    bool need_pack{false};
    bool has_packed{false};  // only valid for constant, only do once throughout the process.
    bool has_origin{false};  // only valid for constant, only true when failing to infer shape, then false after packed.
    int pack_size{-1};
    float *origin_ptr{nullptr};  // only valid for constant, which is synchronized with the 'has_origin'.
    float *pack_ptr{nullptr};
  };

  virtual int ParallelRunByRow(int task_id) const;
  virtual int ParallelRunByOC(int task_id) const;
  virtual int ParallelRunByBatch(int task_id) const;
  virtual int ParallelRunByAllScene(int task_id) const { return RET_ERROR; }
  int ParallelRunIsNotPackByBatch(int task_id) const;
  int BackupConstMatrix(MatrixInfo *matrix_info, int index);
  virtual void InitGlobalVariable();
  int PackMatrixA();
  int PackMatrixB();
  int PackMatrixAImpl();
  int PackMatrixBImpl();
  virtual int PackMatrixAImplOpt();
  bool CheckRow1OptimalConditions();
  virtual bool SupportMulBatchCuttingByRow() { return false; }
  int PackBiasMatrix();
  void FreePackedMatrixA();
  void FreePackedMatrixB();
  virtual int InitParameter();
  int InitTmpOutBuffer();
  virtual bool CheckThreadCuttingByRow();
  void GetThreadCuttingInfoByRow();
  void InitShapeA();
  void InitShapeB();
  int InitBroadcastParams();

 protected:
  MatMulParameter *params_ = nullptr;
  GemmIsNotPackFun gemmIsNotPackFun = nullptr;
  int a_batch_ = 1;
  int b_batch_ = 1;
  std::vector<int> a_offset_;
  std::vector<int> b_offset_;

  int pack_b_stride_ = 0;
  const float *pack_b_src_;
  float *pack_b_dst_;
  int col_tile_ = 0;
  int row_tile_ = 0;
  int batch_stride_ = 0;
  int row_num_;
  int row_min_unit_{1};
  int col_min_unit_{1};
  float *output_data_ = nullptr;
  bool out_need_aligned_ = false;
  int col_step_ = 0;
  std::vector<int> split_points_;
  std::vector<int> col_split_points_;
  std::vector<int> row_split_points_;
  int block_col_unit_ = 0;
  MatrixInfo matrix_a_;
  MatrixInfo matrix_b_;
  MatrixInfo matrix_c_;
  bool pack_opt_{false};  // indicate whether packing can be multi-threads, currently, only support in ARM64 && packA.
  MatrixPackFun matrix_a_pack_fun_ = nullptr;
  MatrixPackFun matrix_b_pack_fun_ = nullptr;
  float *conv1x1_origin_weight_ = nullptr;
  float *conv1x1_origin_bias_ = nullptr;
  bool is_sharing_pack_ = true;
  bool weight_is_packed_{false};
};
}  // namespace mindspore::kernel
#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP32_MATMUL_FP32_BASE_H_
