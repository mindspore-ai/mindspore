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

#include "src/litert/kernel/cpu/int8/matmul_dynamic_sdot_int8.h"
#include <vector>
#include "nnacl/int8/dynamic_matmul_int8.h"
#include "nnacl/int8/matmul_int8.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_MEMORY_FAILED;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
namespace {
int Arm64SdotPreRun(void *cdata, int task_id, float, float) {
  CHECK_NULL_RETURN(cdata);
  auto op = reinterpret_cast<MatMulDynamicSdotInt8Kernel *>(cdata);
  auto ret = op->MatMulDynamicArm64SdotPre(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "MatmulInt8Run error task_id[" << task_id << "] error_code[" << ret << "]";
    return ret;
  }
  return RET_OK;
}

int Arm64SdotRun(void *cdata, int task_id, float, float) {
  CHECK_NULL_RETURN(cdata);
  auto op = reinterpret_cast<MatMulDynamicSdotInt8Kernel *>(cdata);
  auto ret = op->MatMulDynamicArm64SdotImpl(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "MatmulInt8Run error task_id[" << task_id << "] error_code[" << ret << "]";
    return ret;
  }
  return RET_OK;
}
}  // namespace

int MatMulDynamicSdotInt8Kernel::MatMulDynamicArm64SdotPre(int task_id) {
  int row_thread_count = MSMIN(op_parameter_->thread_num_, UP_DIV(param_->row_align_, row_tile_));
  int row_stride = UP_DIV(UP_DIV(param_->row_align_, row_tile_), row_thread_count) * row_tile_;

  int row_current_stride = task_id * row_stride;
  int row_res_stride = param_->row_ - row_current_stride;
  int cur_r = MSMIN(row_res_stride, row_stride);
  if (cur_r <= 0) {
    return RET_OK;
  }

  auto current_a_pack = pack_a_ptr_ + row_current_stride * param_->deep_align_;
  int weight_zp = quant_param_->filter_zp_[0];
  if (param_->a_transpose_) {
    auto current_src_a = batch_input_ptr_ + row_current_stride;
    if (weight_zp == 0) {
      PackInput2Col4x4(current_src_a, current_a_pack, param_->deep_, cur_r, param_->row_);
    } else {
      PackInput2Col4x4AndInputSumPert(current_src_a, current_a_pack, input_sums_ + row_current_stride, param_->deep_,
                                      cur_r, param_->row_, weight_zp);
    }
  } else {
    auto current_src_a = batch_input_ptr_ + row_current_stride * param_->deep_;
    if (weight_zp == 0) {
      PackInput4x4(current_src_a, current_a_pack, param_->deep_, cur_r);
    } else {
      PackInput4x4AndInputSumPert(current_src_a, current_a_pack, input_sums_ + row_current_stride, param_->deep_, cur_r,
                                  weight_zp);
    }
  }
  return RET_OK;
}

void MatMulDynamicSdotInt8Kernel::ComputeMultiScaleAhead(std::vector<float> *multi_scale, int col_start,
                                                         size_t col_num) {
  auto &scales = *multi_scale;
  if (!input_per_channel_) {
    if (!filter_per_channel_) {
      scales.resize(1);
      scales[0] = quant_param_->input_scale_[0] * quant_param_->filter_scale_[0];
    } else {
      scales.resize(UP_ROUND(col_num, col_tile_));
      float *filter_scales = quant_param_->filter_scale_ + col_start;
      for (size_t i = 0; i < col_num; ++i) {
        scales[i] = quant_param_->input_scale_[0] * filter_scales[i];
      }
    }
  } else if (!filter_per_channel_) {
    scales.resize(param_->row_align_);
    for (int i = 0; i < param_->row_; ++i) {
      scales[i] = quant_param_->input_scale_[i] * quant_param_->filter_scale_[0];
    }
  }
}

void MatMulDynamicSdotInt8Kernel::ComputeMultiScaleChannelByChannel(std::vector<float> *multi_scale, int row_start,
                                                                    size_t row_num, int col_start, size_t col_num) {
  auto &scales = *multi_scale;
  scales.resize(row_tile_ * col_tile_, 0);
  float *in_scales = quant_param_->input_scale_ + row_start;
  float *filter_scales = quant_param_->filter_scale_ + col_start;
  for (size_t i = 0; i < row_num; ++i) {
    for (size_t j = 0; j < col_num; ++j) {
      scales[i * col_tile_ + j] = in_scales[i] * filter_scales[j];
    }
  }
}

int MatMulDynamicSdotInt8Kernel::MatMulDynamicArm64SdotImpl(int task_id) {
  // Multi-thread split by col.
  int stride = thread_stride_ * col_tile_;
  int cur_stride = task_id * stride;
  int res_stride = param_->col_ - cur_stride;
  int cur_oc = MSMIN(stride, res_stride);
  if (cur_oc <= 0) {
    return RET_OK;
  }
  auto current_sums = batch_sums_ + cur_stride;
  if (!param_->b_const_) {
    auto current_b_pack = batch_b_ptr_ + cur_stride * param_->deep_align_;
    if (param_->b_transpose_) {
      auto current_weight = batch_weight_ptr_ + cur_stride * param_->deep_;
      RowMajor2Row4x16MajorInt8(current_weight, current_b_pack, cur_oc, param_->deep_);
      CalcPartWeightSums(current_weight, param_->deep_, param_->col_, cur_oc, current_sums, ColMajor);
    } else {
      auto current_weight = batch_weight_ptr_ + cur_stride;
      RowMajor2Col4x16MajorPartInt8(current_weight, current_b_pack, param_->deep_, param_->col_, cur_oc);
      CalcPartWeightSums(current_weight, param_->deep_, param_->col_, cur_oc, current_sums, RowMajor);
    }
  }

  std::vector<float> multi_scale;
  ComputeMultiScaleAhead(&multi_scale, cur_stride, cur_oc);
  int64_t mode = input_per_channel_ * C2NUM + filter_per_channel_;

  size_t data_type_size = enable_fp16_ ? sizeof(uint16_t) : sizeof(float);
  auto out_stride = param_->col_ * data_type_size;
  int64_t act_type = static_cast<int64_t>(param_->act_type_);
  for (int r = 0; r < param_->row_; r += C4NUM) {
    size_t row = MSMIN(C4NUM, param_->row_ - r);
    auto a_ptr = pack_a_ptr_ + r * param_->deep_align_;
    int *input_sums_ptr = input_sums_ + r;
    for (int c = 0; c < cur_oc; c += C16NUM) {
      size_t col = MSMIN(C16NUM, cur_oc - c);
      auto col_offset = cur_stride + c;
      auto b_ptr = batch_b_ptr_ + col_offset * param_->deep_align_;
      int *weight_sums_ptr = current_sums + c;

      void *out_ptr = static_cast<int8_t *>(batch_c_ptr_) + (r * param_->col_ + col_offset) * data_type_size;
      auto bias = bias_ptr_;
      if (bias_ptr_ != nullptr) {
        bias = static_cast<int8_t *>(bias) + col_offset * data_type_size;
      }
      if (mode == C3NUM) {
        ComputeMultiScaleChannelByChannel(&multi_scale, r, row, col_offset, col);
      }
      int multi_scale_offset =
        (input_per_channel_ == filter_per_channel_ ? 0 : input_per_channel_ * r + filter_per_channel_ * c);
      if (!enable_fp16_) {
        dynamic_matmul_compute_fp32(a_ptr, b_ptr, reinterpret_cast<float *>(out_ptr), param_->deep_align_,
                                    multi_scale.data() + multi_scale_offset, reinterpret_cast<float *>(bias), row, col,
                                    out_stride, input_sums_ptr, weight_sums_ptr, quant_param_->input_zp_[0],
                                    quant_param_->filter_zp_[0] * param_->deep_, act_type, mode);
      } else {
#ifdef ENABLE_FP16
        dynamic_matmul_compute_fp16(a_ptr, b_ptr, reinterpret_cast<float16_t *>(out_ptr), param_->deep_align_,
                                    multi_scale.data() + multi_scale_offset, reinterpret_cast<float16_t *>(bias), row,
                                    col, out_stride, input_sums_ptr, weight_sums_ptr, quant_param_->input_zp_[0],
                                    quant_param_->filter_zp_[0] * param_->deep_, act_type, mode);
#endif
      }
    }
  }
  return RET_OK;
}

void MatMulDynamicSdotInt8Kernel::InitParameter() {
  param_->a_const_ = (in_tensors_[0]->data() != nullptr);
  param_->b_const_ = (in_tensors_[1]->data() != nullptr);

  row_tile_ = C4NUM;
  col_tile_ = C16NUM;
  deep_tile_ = C4NUM;

  if (param_->b_transpose_) {
    b_pack_func_ = RowMajor2Row4x16MajorInt8;
  } else {
    b_pack_func_ = RowMajor2Col4x16MajorInt8;
  }
#if defined(ENABLE_ARM64) && !defined(SUPPORT_NNIE) && !defined(SUPPORT_34XX) && (!defined(MACHINE_LINUX_ARM64)) && \
  !defined(USE_AOS_GCC_TOOLCHAIN)
  dynamic_matmul_compute_fp32 = DynamicMatmulSdot4x4x16AIWI;
#else
  dynamic_matmul_compute_fp32 = DynamicMatmul4x4x16AIWI;
#endif
#ifdef ENABLE_FP16
#if defined(ENABLE_ARM64) && !defined(SUPPORT_NNIE) && !defined(SUPPORT_34XX) && (!defined(MACHINE_LINUX_ARM64)) && \
  !defined(USE_AOS_GCC_TOOLCHAIN)
  dynamic_matmul_compute_fp16 = DynamicMatmulSdot4x4x16AIWIForFp16;
#else
  dynamic_matmul_compute_fp16 = DynamicMatmul4x4x16AIWIForFp16;
#endif
#endif
}

int MatMulDynamicSdotInt8Kernel::MatMulDynamicRunArm64Sdot() {
  int8_t *a_ptr = reinterpret_cast<int8_t *>(in_tensors_.at(0)->data());
  int8_t *b_ptr = reinterpret_cast<int8_t *>(in_tensors_.at(1)->data());
  void *c_ptr = out_tensors_.at(0)->data();
  CHECK_NULL_RETURN(a_ptr);
  CHECK_NULL_RETURN(b_ptr);
  CHECK_NULL_RETURN(c_ptr);

  size_t data_type_size = enable_fp16_ ? sizeof(uint16_t) : sizeof(float);
  for (int i = 0; i < param_->batch; i++) {
    batch_input_ptr_ = a_ptr + a_offset_[i] * param_->row_ * param_->deep_;
    auto ret = ParallelLaunch(this->ms_context_, Arm64SdotPreRun, this, op_parameter_->thread_num_);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Arm64SdotPreRun error: [" << ret << "]";
      return ret;
    }

    batch_weight_ptr_ = b_ptr + b_offset_[i] * param_->col_ * param_->deep_;
    batch_sums_ = weight_sums_ + b_offset_[i] * param_->col_align_;
    batch_b_ptr_ = pack_b_ptr_ + b_offset_[i] * param_->col_align_ * param_->deep_align_;
    batch_c_ptr_ = static_cast<uint8_t *>(c_ptr) + i * param_->row_ * param_->col_ * data_type_size;
    ret = ParallelLaunch(this->ms_context_, Arm64SdotRun, this, thread_num_);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Arm64SdotRun error: [" << ret << "]";
      return ret;
    }
  }
  return RET_OK;
}

int MatMulDynamicSdotInt8Kernel::Run() {
  std::vector<float> input_scales;
  std::vector<int32_t> input_zp;
  auto ret = InitInputQuantParam(&input_scales, &input_zp);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init input quant param failed.";
    return ret;
  }
  ret = InitMatrixABuffer();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Alloc run-buffer for matrix-a failed.";
    return ret;
  }
  if (!param_->b_const_) {
    ret = InitFilterQuantParam();
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Init filter quant param failed.";
      FreeQuantParam();
      return ret;
    }
  }
  ret = MatMulDynamicRunArm64Sdot();
  FreeMatrixABuffer();
  return ret;
}
}  // namespace mindspore::kernel
