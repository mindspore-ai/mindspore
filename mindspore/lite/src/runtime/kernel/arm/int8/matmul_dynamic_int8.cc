/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "src/runtime/kernel/arm/int8/matmul_dynamic_int8.h"
#include "src/runtime/kernel/arm/int8/opt_op_handler.h"
#include "nnacl/int8/matmul_int8.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_MEMORY_FAILED;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
namespace {
constexpr int kHasBiasSize = 3;
constexpr int kMinInputSize = 2;
constexpr int kOutputSize = 1;
constexpr int kSize1 = 1;
constexpr int kSize2 = 2;
}  // namespace

int MatmulDynamicInt8Run(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  CHECK_NULL_RETURN(cdata);
  auto op = reinterpret_cast<MatmulDynamicInt8CPUKernel *>(cdata);
  auto ret = op->RunImpl(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "MatmulInt8Run error task_id[" << task_id << "] error_code[" << ret << "]";
    return ret;
  }
  return RET_OK;
}

int MatmulDynamicInt8CPUKernel::RunImpl(int task_id) {
  int stride = thread_stride_ * col_tile_;
  int cur_stride = task_id * stride;
  int res_stride = param_->col_ - cur_stride;
  int cur_oc = MSMIN(stride, res_stride);
  if (cur_oc <= 0) {
    return RET_OK;
  }
  DynamicMatmulInt8AIWI(pack_a_ptr_, batch_b_ptr_ + cur_stride * param_->deep_align_, fp32_bias_ptr_,
                        batch_c_ptr_ + cur_stride, param_->row_, cur_oc, param_->deep_align_,
                        quant_param_->input_scale_, quant_param_->filter_scale_, param_->col_, filter_per_channel_);
  return RET_OK;
}

MatmulDynamicInt8CPUKernel::~MatmulDynamicInt8CPUKernel() {
  FreeQuantParam();
  FreeTmpBuffer();
}

void MatmulDynamicInt8CPUKernel::FreeQuantParam() {
  if (quant_param_ != nullptr) {
    if (quant_param_->filter_scale_ != nullptr) {
      free(quant_param_->filter_scale_);
      quant_param_->filter_scale_ = nullptr;
    }
    if (quant_param_->filter_zp_ != nullptr) {
      free(quant_param_->filter_zp_);
      quant_param_->filter_zp_ = nullptr;
    }
    free(quant_param_);
    quant_param_ = nullptr;
  }
}

int MatmulDynamicInt8CPUKernel::MallocQuantParam() {
  quant_param_ = reinterpret_cast<MatmulDynamicQuantParameter *>(malloc(sizeof(MatmulQuantParameter)));
  if (quant_param_ == nullptr) {
    MS_LOG(ERROR) << "Malloc MatmulDynamicQuantParameter for Matmul int8 op failed!";
    return RET_ERROR;
  }
  memset(quant_param_, 0, sizeof(MatmulQuantParameter));
  return RET_OK;
}

int MatmulDynamicInt8CPUKernel::InitFilterQuantParam() {
  if (quant_param_->filter_scale_ != nullptr) {
    free(quant_param_->filter_scale_);
    quant_param_->filter_scale_ = nullptr;
  }
  if (quant_param_->filter_zp_ != nullptr) {
    free(quant_param_->filter_zp_);
    quant_param_->filter_zp_ = nullptr;
  }

  auto weight_tensor = in_tensors_.at(kWeightIndex);
  auto weight_quant_params = weight_tensor->quant_params();
  auto w_shape = weight_tensor->shape();
  if (w_shape.size() < DIMENSION_2D) {
    MS_LOG(ERROR) << weight_tensor->tensor_name() << " dims < 2.";
    return RET_ERROR;
  }
  int col = param_->b_transpose_ ? w_shape[w_shape.size() - kSize2] : w_shape[w_shape.size() - kSize1];
  filter_per_channel_ = (weight_quant_params.size() > 1);
  channel_num_ = filter_per_channel_ ? col : 1;
  if (static_cast<int>(weight_quant_params.size()) != channel_num_) {
    MS_LOG(ERROR) << weight_tensor->tensor_name() << " quant params size:" << weight_quant_params.size()
                  << " != channel_num_:" << channel_num_;
    return RET_ERROR;
  }
  quant_param_->filter_scale_ = reinterpret_cast<float *>(malloc(channel_num_ * sizeof(float)));
  CHECK_NULL_RETURN(quant_param_->filter_scale_);
  memset(quant_param_->filter_scale_, 0, sizeof(channel_num_));
  quant_param_->filter_zp_ = reinterpret_cast<int32_t *>(malloc(channel_num_ * sizeof(int32_t)));
  CHECK_NULL_RETURN(quant_param_->filter_zp_);
  memset(quant_param_->filter_zp_, 0, sizeof(channel_num_));

  for (int i = 0; i < channel_num_; i++) {
    quant_param_->filter_scale_[i] = static_cast<float>(weight_quant_params[i].scale);
    quant_param_->filter_zp_[i] = weight_quant_params[i].zeroPoint;
  }
  return RET_OK;
}

int MatmulDynamicInt8CPUKernel::InitInputQuantParam() {
  auto in_quant_params = in_tensors_.at(kInputIndex)->quant_params();
  if (in_quant_params.empty()) {
    MS_LOG(ERROR) << "invalid in quant param";
    return RET_ERROR;
  }
  quant_param_->input_zp_ = in_quant_params.front().zeroPoint;
  quant_param_->input_scale_ = static_cast<float>(in_quant_params.front().scale);
  return RET_OK;
}

void MatmulDynamicInt8CPUKernel::InitParameter() {
  param_->a_const_ = (in_tensors_[kInputIndex]->data() != nullptr);
  param_->b_const_ = (in_tensors_[kWeightIndex]->data() != nullptr);
#ifdef ENABLE_ARM32
  row_tile_ = C4NUM;
  col_tile_ = C2NUM;
  deep_tile_ = C16NUM;
#elif ENABLE_ARM64
  support_sdot_ = mindspore::lite::IsSupportSDot();
  row_tile_ = C4NUM;
  if (support_sdot_) {
    col_tile_ = C16NUM;
    deep_tile_ = C4NUM;
  } else {
    col_tile_ = C4NUM;
    deep_tile_ = C16NUM;
  }
#else
  row_tile_ = C4NUM;
  col_tile_ = C4NUM;
  deep_tile_ = C16NUM;
#endif
  if (param_->a_transpose_) {
    a_pack_func_ = RowMajor2Col16x4MajorInt8;
  } else {
    a_pack_func_ = RowMajor2Row16x4MajorInt8;
  }
  if (param_->b_transpose_) {
#ifdef ENABLE_ARM32
    b_pack_func_ = RowMajor2Row2x16MajorInt8;
#elif ENABLE_ARM64
    if (support_sdot_) {
      b_pack_func_ = RowMajor2Row4x16MajorInt8;
    } else {
      b_pack_func_ = RowMajor2Row16x4MajorInt8;
    }
#else
    b_pack_func_ = RowMajor2Row16x4MajorInt8;
#endif
  } else {
#ifdef ENABLE_ARM32
    b_pack_func_ = RowMajor2Col16x2MajorInt8;
#elif ENABLE_ARM64
    if (support_sdot_) {
      b_pack_func_ = RowMajor2Col4x16MajorInt8;
    } else {
      b_pack_func_ = RowMajor2Col16x4MajorInt8;
    }
#else
    b_pack_func_ = RowMajor2Col16x4MajorInt8;
#endif
  }
  return;
}

void MatmulDynamicInt8CPUKernel::ResizeParameter() {
  param_->row_align_ = UP_ROUND(param_->row_, row_tile_);
  param_->col_align_ = UP_ROUND(param_->col_, col_tile_);
  param_->deep_align_ = UP_ROUND(param_->deep_, deep_tile_);

  thread_count_ = MSMIN(op_parameter_->thread_num_, UP_DIV(param_->col_align_, col_tile_));
  thread_stride_ = UP_DIV(UP_DIV(param_->col_align_, col_tile_), thread_count_);
  return;
}

void MatmulDynamicInt8CPUKernel::FreeTmpBuffer() {
  if (pack_a_ptr_ != nullptr) {
    free(pack_a_ptr_);
    pack_a_ptr_ = nullptr;
  }
  if (pack_b_ptr_ != nullptr) {
    free(pack_b_ptr_);
    pack_b_ptr_ = nullptr;
  }
  return;
}

int MatmulDynamicInt8CPUKernel::TransferB() {
  auto weight_data = reinterpret_cast<int8_t *>(in_tensors_.at(kWeightIndex)->data());
  CHECK_NULL_RETURN(weight_data);
  for (int i = 0; i < param_->batch; i++) {
    auto current_weight = weight_data + i * param_->deep_ * param_->col_;
    auto current_b_pack = pack_b_ptr_ + i * param_->col_align_ * param_->deep_align_;
    CHECK_NULL_RETURN(b_pack_func_);
    if (param_->b_transpose_) {
      b_pack_func_(current_weight, current_b_pack, param_->col_, param_->deep_);
    } else {
      b_pack_func_(current_weight, current_b_pack, param_->deep_, param_->col_);
    }
  }
  return RET_OK;
}

int MatmulDynamicInt8CPUKernel::InitTmpBuffer() {
  pack_a_ptr_ = reinterpret_cast<int8_t *>(malloc(param_->row_align_ * param_->deep_align_ * sizeof(int8_t)));
  if (pack_a_ptr_ == nullptr) {
    FreeTmpBuffer();
    return RET_ERROR;
  }
  pack_b_ptr_ =
    reinterpret_cast<int8_t *>(malloc(param_->batch * param_->col_align_ * param_->deep_align_ * sizeof(int8_t)));
  if (pack_b_ptr_ == nullptr) {
    FreeTmpBuffer();
    return RET_ERROR;
  }
  memset(pack_a_ptr_, 0, param_->row_align_ * param_->deep_align_ * sizeof(int8_t));
  memset(pack_b_ptr_, 0, param_->batch * param_->col_align_ * param_->deep_align_ * sizeof(int8_t));
  return RET_OK;
}

int MatmulDynamicInt8CPUKernel::CopyBias() {
  if (in_tensors_.size() == kHasBiasSize) {
    auto bias_tensor = in_tensors_[kBiasIndex];
    fp32_bias_ptr_ = reinterpret_cast<float *>(bias_tensor->data());
    if (fp32_bias_ptr_ == nullptr) {
      MS_LOG(ERROR) << "Memory allocation failed";
      FreeTmpBuffer();
      return RET_MEMORY_FAILED;
    }
    memcpy(fp32_bias_ptr_, bias_tensor->data(), bias_tensor->ElementsNum() * sizeof(float));
  } else {
    fp32_bias_ptr_ = nullptr;
  }
  return RET_OK;
}

int MatmulDynamicInt8CPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), kMinInputSize);
  CHECK_LESS_RETURN(out_tensors_.size(), kOutputSize);
  InitParameter();
  auto ret = MallocQuantParam();
  if (ret != RET_OK) {
    FreeQuantParam();
    return ret;
  }
  if (param_->b_const_) {
    ret = InitFilterQuantParam();
    if (ret != RET_OK) {
      FreeQuantParam();
      return ret;
    }
  }
  ret = CopyBias();
  if (ret != RET_OK) {
    FreeQuantParam();
    return ret;
  }
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int MatmulDynamicInt8CPUKernel::ReSize() {
  int batch = 1;
  auto x_shape = in_tensors_.at(0)->shape();
  auto o_shape = out_tensors_.at(0)->shape();
  MS_ASSERT(x_shape.size() >= kSize2);
  for (size_t i = 0; i < x_shape.size() - kSize2; ++i) {
    batch *= x_shape[i];
  }
  param_->batch = batch;
  MS_ASSERT(o_shape.size() >= kSize2);
  param_->row_ = o_shape[o_shape.size() - kSize2];
  param_->col_ = o_shape[o_shape.size() - kSize1];
  param_->deep_ = param_->a_transpose_ ? x_shape[x_shape.size() - kSize2] : x_shape[x_shape.size() - kSize1];

  FreeTmpBuffer();

  ResizeParameter();

  auto ret = InitTmpBuffer();
  if (ret != RET_OK) {
    FreeQuantParam();
    return ret;
  }
  if (param_->b_const_ == true) {
    TransferB();
  }
  return RET_OK;
}

int MatmulDynamicInt8CPUKernel::Run() {
  auto ret = InitInputQuantParam();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init input quant param failed.";
    return ret;
  }
  if (!param_->b_const_) {
    ret = InitFilterQuantParam();
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Init filter quant param failed.";
      FreeQuantParam();
      return ret;
    }
    ret = TransferB();
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "TransferB failed.";
      return ret;
    }
  }
  auto *a_ptr = reinterpret_cast<int8_t *>(in_tensors_.at(0)->data());
  auto *c_ptr = reinterpret_cast<float *>(out_tensors_.at(0)->data());
  CHECK_NULL_RETURN(a_ptr);
  CHECK_NULL_RETURN(c_ptr);
  for (int i = 0; i < param_->batch; i++) {
    auto current_src_a = a_ptr + i * param_->row_ * param_->deep_;
    if (param_->a_transpose_) {
      MS_CHECK_TRUE_RET(a_pack_func_ != nullptr, RET_ERROR);
      a_pack_func_(current_src_a, pack_a_ptr_, param_->deep_, param_->row_);
    } else {
      MS_CHECK_TRUE_RET(a_pack_func_ != nullptr, RET_ERROR);
      a_pack_func_(current_src_a, pack_a_ptr_, param_->row_, param_->deep_);
    }

    batch_b_ptr_ = pack_b_ptr_ + i * param_->col_align_ * param_->deep_align_;
    batch_c_ptr_ = c_ptr + i * param_->row_ * param_->col_;

    ret = ParallelLaunch(this->ms_context_, MatmulDynamicInt8Run, this, thread_count_);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "MatmulInt8Run error: [" << ret << "]";
      return ret;
    }
  }
  return RET_OK;
}
}  // namespace mindspore::kernel
