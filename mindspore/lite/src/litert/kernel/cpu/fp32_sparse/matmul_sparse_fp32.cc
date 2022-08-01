/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "src/litert/kernel/cpu/fp32_sparse/matmul_sparse_fp32.h"
#include <cstdlib>
#include <cstdio>
#include <cstdarg>
#include "include/errorcode.h"
#include "src/litert/kernel_registry.h"
#ifdef ENABLE_ARM64
#include <arm_neon.h>
#endif
#include "nnacl/fp32/matmul_fp32.h"
#include "nnacl/fp32_sparse/matmul_sparse_x1_fp32.h"
#include "nnacl/fp32/pack_fp32.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_MatMulFusion;

namespace mindspore::kernel {
constexpr size_t kFirstDimIdx = 0;
constexpr size_t kSecondDimIdx = 1;
constexpr size_t kThirdDimIdx = 2;

void MatmulSparseCPUKernel::InitParameter() {
  params_->a_const_ = true;
  params_->b_const_ = false;
  auto a_shape = in_tensors_.at(0)->shape();
  int a_batch = 1;
  constexpr size_t batch_matmul_split = -2;
  for (size_t i = 0; i < a_shape.size() + batch_matmul_axis; ++i) {
    a_batch *= a_shape[i];
  }
  params_->batch = a_batch;
  constexpr size_t left_row_axis_transpose = -1;
  constexpr size_t left_row_axis_not_transpose = -2;
  constexpr size_t left_col_axis_transpose = -2;
  constexpr size_t left_col_axis_not_transpose = -1;
  params_->row_ = params_->a_transpose_ ? (a_shape[a_shape.size() + left_row_axis_transpose])
                                        : (a_shape[a_shape.size() + left_row_axis_not_transpose]);
  params_->deep_ = params_->a_transpose_ ? (a_shape[a_shape.size() + left_col_axis_transpose])
                                         : (a_shape[a_shape.size() + left_col_axis_not_transpose]);

  auto b_shape = in_tensors_.at(1)->shape();
  int b_batch = 1;
  for (size_t i = 0; i < b_shape.size() + batch_matmul_split; ++i) {
    b_batch *= b_shape[i];
  }
  MS_ASSERT(a_batch == b_batch);
  constexpr size_t right_row_axis_transpose = -2;
  constexpr size_t right_row_axis_not_transpose = -1;
  constexpr size_t right_col_axis_transpose = -1;
  constexpr size_t right_col_axis_not_transpose = -2;
  params_->col_ = params_->b_transpose_ ? (b_shape[b_shape.size() + right_row_axis_transpose])
                                        : (b_shape[b_shape.size() + right_row_axis_not_transpose]);
  params_->deep_ = params_->b_transpose_ ? (b_shape[b_shape.size() + right_col_axis_transpose])
                                         : (b_shape[b_shape.size() + right_col_axis_not_transpose]);

  params_->row_align_ = UP_ROUND(params_->row_, C8NUM);
  params_->col_align_ = UP_ROUND(params_->col_, C8NUM);

  // if a_transpose, no need to pack activation
  if (params_->a_transpose_) {
    return;
  }
  constexpr int perm_1 = 2;
  auto area = params_->row_ * params_->deep_;
  trans_param_.num_axes_ = kNumIntThree;
  trans_param_.perm_[kFirstDimIdx] = 0;
  trans_param_.perm_[kSecondDimIdx] = perm_1;
  trans_param_.perm_[kThirdDimIdx] = 1;
  trans_param_.strides_[kThirdDimIdx] = 1;
  trans_param_.strides_[kSecondDimIdx] = params_->deep_;
  trans_param_.strides_[kFirstDimIdx] = area;
  trans_param_.out_strides_[kThirdDimIdx] = 1;
  trans_param_.out_strides_[kSecondDimIdx] = params_->row_;
  trans_param_.out_strides_[kFirstDimIdx] = area;
}

namespace {
constexpr float kFpPrecision = 1e-6;
constexpr size_t kBlockSize = 8;
constexpr size_t bias_tensor_index = 2;
}  // namespace

int kernel::MatmulSparseCPUKernel::PrepareWeight() {
  auto weight_data = reinterpret_cast<float *>(in_tensors_.at(1)->data());
  MS_ASSERT(weight_data != nullptr);
  sparsity_weight_ = new SparsityWeight;
  size_t non_zeros = 0;
  for (int i = 0; i < in_tensors_.at(1)->ElementsNum(); i++) {
    if (weight_data[i] > kFpPrecision) {
      non_zeros++;
    }
  }
  sparsity_weight_->nnz = non_zeros;
  sparsity_weight_->data = reinterpret_cast<float *>(malloc(non_zeros * sizeof(float)));
  CHECK_NULL_RETURN(sparsity_weight_->data);
  sparsity_weight_->non_zero_num = reinterpret_cast<uint32_t *>(malloc(sizeof(uint32_t) * params_->col_));
  CHECK_NULL_RETURN(sparsity_weight_->non_zero_num);
  memset(sparsity_weight_->non_zero_num, 0, sizeof(uint32_t) * params_->col_);
  sparsity_weight_->act_stride = reinterpret_cast<size_t *>(malloc(non_zeros * sizeof(size_t)));
  CHECK_NULL_RETURN(sparsity_weight_->act_stride);
  size_t act_stride_index = 0;
  size_t weight_data_index = 0;
  for (int j = 0; j < params_->col_; j++) {
    for (int i = 0; i < params_->deep_; i++) {
      auto cur_data = weight_data[i * params_->col_ + j];
      if (cur_data > kFpPrecision) {
        sparsity_weight_->data[weight_data_index++] = cur_data;
        sparsity_weight_->act_stride[act_stride_index++] = i * kBlockSize * sizeof(float);
        (*(sparsity_weight_->non_zero_num + j))++;
      }
    }
  }
  return RET_OK;
}

int MatmulSparseCPUKernel::PrepareBias() {
  constexpr size_t has_bias_tensor_num = 3;
  if (in_tensors_.size() == has_bias_tensor_num) {
    auto bias_tensor = in_tensors_[bias_tensor_index];
    if (bias_tensor->ElementsNum() != params_->col_) {
      MS_LOG(ERROR) << "Not support broadcast bias data now";
      return lite::RET_NOT_SUPPORT;
    }
    // malloc addr need to aligned to 32 bytes
    bias_pack_ = reinterpret_cast<float *>(malloc(params_->col_align_ * static_cast<int>(sizeof(float))));
    if (bias_pack_ == nullptr) {
      MS_LOG(ERROR) << "malloc bias_ptr_ failed";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

int MatmulSparseCPUKernel::Prepare() {
  if (!InferShapeDone()) {
    return RET_ERROR;
  }
  if (params_ == nullptr) {
    MS_LOG(ERROR) << "Params is nullptr";
    return RET_ERROR;
  }
  if ((in_tensors_.at(0)->data() != nullptr) || (in_tensors_.at(1)->data() == nullptr)) {
    MS_LOG(ERROR) << "Only support Activation X filter now";
    return RET_ERROR;
  }
  InitParameter();
  matrix_a_pack_size_ = params_->batch * params_->row_align_ * params_->deep_;
  if (params_->batch != 1) {
    MS_LOG(ERROR) << "Only support batch == 1 now";
    return lite::RET_NOT_SUPPORT;
  }
  if (params_->row_ % kBlockSize != 0) {
    MS_LOG(ERROR) << "Only support 8n rows now";
    return lite::RET_NOT_SUPPORT;
  }
  if (params_->col_ % kBlockSize != 0) {
    MS_LOG(ERROR) << "Only support 8n cols now";
    return lite::RET_NOT_SUPPORT;
  }
  if (params_->a_transpose_) {  // conv nchw or a_transpose matmul
    MS_LOG(ERROR) << "Not support a transpose now";
    return lite::RET_NOT_SUPPORT;
  }
  auto ret = PrepareWeight();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "PrepareWeight failed";
    return ret;
  }
  ret = PrepareBias();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "PrepareBias failed";
    return ret;
  }
  return RET_OK;
}

int MatmulSparseCPUKernel::PackInput() {
  a_pack_ =
    reinterpret_cast<float *>(ms_context_->allocator->Malloc(static_cast<size_t>(matrix_a_pack_size_) * sizeof(float)));
  if (a_pack_ == nullptr) {
    MS_LOG(ERROR) << "Malloc input pack buffer failed";
    return lite::RET_NOT_SUPPORT;
  }

  // not support multi-thread now
  auto task_num = params_->row_align_ / kBlockSize;
  auto stride = matrix_a_pack_size_ / task_num;

  auto *src = reinterpret_cast<const float *>(in_tensors_[0]->data());
  for (size_t i = 0; i < task_num; i++) {
    PackNHWCToNCHWFp32(src + i * stride, a_pack_ + i * stride, params_->batch, kBlockSize, params_->deep_, 0, 0);
  }
  return RET_OK;
}

int MatmulSparseCPUKernel::ReSize() { return RET_ERROR; }

int MatmulSparseCPUKernel::RunInstrinsics() {
#ifndef ENABLE_ARM64
  MS_LOG(ERROR) << "Only support aarch64 now";
  return lite::RET_NOT_SUPPORT;
#else
  auto ret = PackInput();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Pack input failed";
    return ret;
  }
  CHECK_NULL_RETURN(sparsity_weight_);
#ifdef Debug
  auto start_time = lite::GetTimeUs();
  auto end_time = lite::GetTimeUs();
  std::cout << "Test time : " << (end_time - start_time) << "us" << std::endl;
  {
    printf("=========================================input:\r\n");
    float *src = reinterpret_cast<float *>(in_tensors_[0]->data());
    for (size_t i = 0; i < params_->row_; i++) {
      for (size_t j = 0; j < params_->deep_; j++) {
        printf(" %2.2f", src[i * params_->deep_ + j]);
      }
      printf("\r\n");
    }
    printf("\r\n");
  }
  {
    auto wdata = reinterpret_cast<float *>(in_tensors_.at(1)->data());
    printf("=========================================weight:\r\n");
    for (size_t i = 0; i < params_->deep_; i++) {
      for (size_t j = 0; j < params_->col_; j++) {
        printf(" %2.2f", wdata[i * params_->col_ + j]);
      }
      printf("\r\n");
    }
    printf("\r\n");
  }
  {
    auto bias = reinterpret_cast<float *>(in_tensors_.at(bias_tensor_index)->data());
    printf("=========================================bias:\r\n");
    for (size_t i = 0; i < params_->col_; i++) {
      printf(" %2.2f", bias[i]);
    }
    printf("\r\n");
  }
  {
    printf("=========================================weight:\r\n");
    for (uint32_t i = 0; i < sparsity_weight_->nnz; i++) {
      printf(" %2.2f", sparsity_weight_->data[i]);
    }
    printf("\r\n");
  }
  {
    printf("=========================================widx_dmap:\r\n");
    for (size_t i = 0; i < sparsity_weight_->nnz; i++) {
      printf(" %4zu", sparsity_weight_->act_stride[i]);
    }
    printf("\r\n");
  }
  {
    printf("=========================================nidx_nnzmap:\r\n");
    for (size_t i = 0; i < params_->col_; i++) {
      printf(" %4d", sparsity_weight_->non_zero_num[i]);
    }
    printf("\r\n");
  }
#endif
  auto bias = reinterpret_cast<float *>(in_tensors_.at(bias_tensor_index)->data());
  auto output = reinterpret_cast<float *>(out_tensors_.front()->data());
  for (int i = 0; i < params_->row_align_ / kBlockSize; i++) {
    MatMulSparse8x8(a_pack_ + i * kBlockSize * params_->deep_, sparsity_weight_->data, sparsity_weight_->non_zero_num,
                    sparsity_weight_->act_stride, output + i * kBlockSize * kBlockSize, bias, ActType_No, kBlockSize);
  }

#ifdef Debug
  {
    printf("=========================================output:\r\n");
    for (size_t i = 0; i < params_->row_; i++) {
      for (size_t j = 0; j < params_->col_; j++) {
        printf(" %2.2f", output[i * params_->col_ + j]);
      }
      printf("\r\n");
    }
    printf("\r\n");
  }
#endif
  ms_context_->allocator->Free(a_pack_);
  return RET_OK;
#endif
}

int MatmulSparseCPUKernel::Run() {
#ifndef ENABLE_ARM64
  MS_LOG(ERROR) << "Only support aarch64 now";
  return lite::RET_NOT_SUPPORT;
#else
  auto ret = PackInput();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Pack input failed";
    return ret;
  }
  CHECK_NULL_RETURN(sparsity_weight_);
  auto bias = reinterpret_cast<float *>(in_tensors_.at(bias_tensor_index)->data());
  auto output = reinterpret_cast<float *>(out_tensors_.front()->data());
  for (int i = 0; i < params_->row_align_ / kBlockSize; i++) {
    SPMM8x8Fp32(a_pack_ + i * params_->deep_ * kBlockSize, sparsity_weight_->data, sparsity_weight_->non_zero_num,
                sparsity_weight_->act_stride, output + i * params_->col_align_ * kBlockSize, bias, ActType_No,
                kBlockSize * sizeof(float));
  }
  ms_context_->allocator->Free(a_pack_);
  return RET_OK;
#endif
}

MatmulSparseCPUKernel::~MatmulSparseCPUKernel() {
  if (this->sparsity_weight_ == nullptr) {
    return;
  }
  free(sparsity_weight_->non_zero_num);
  free(sparsity_weight_->act_stride);
  free(sparsity_weight_->data);
  delete (this->sparsity_weight_);
}
}  // namespace mindspore::kernel
