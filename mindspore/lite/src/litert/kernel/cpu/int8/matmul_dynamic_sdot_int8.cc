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

  std::vector<float> multi_scale(cur_oc);
  for (int i = 0; i < cur_oc; ++i) {
    if (!param_->b_const_) {
      multi_scale[i] = quant_param_->input_scale_ * quant_param_->filter_scale_[0];
    } else {
      multi_scale[i] = quant_param_->input_scale_ * quant_param_->filter_scale_[cur_stride + i];
    }
  }
  auto out_stride = param_->col_ * sizeof(float);
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
      auto out_ptr = batch_c_ptr_ + r * param_->col_ + col_offset;
      auto bias = fp32_bias_ptr_;
      if (bias != nullptr) {
        bias += col_offset;
      }

#if defined(ENABLE_ARM64) && !defined(SUPPORT_NNIE) && !defined(SUPPORT_34XX) && (!defined(MACHINE_LINUX_ARM64))
      DynamicMatmulSdot4x4x16AIWI(a_ptr, b_ptr, out_ptr, param_->deep_align_, multi_scale.data() + c, bias, row, col,
                                  out_stride, input_sums_ptr, weight_sums_ptr, quant_param_->input_zp_,
                                  quant_param_->filter_zp_[0] * param_->deep_, act_type);
#else
      DynamicMatmul4x4x16AIWI(a_ptr, b_ptr, out_ptr, param_->deep_align_, multi_scale.data() + c, bias, row, col,
                              out_stride, input_sums_ptr, weight_sums_ptr, quant_param_->input_zp_,
                              quant_param_->filter_zp_[0] * param_->deep_, act_type);
#endif
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
  return;
}

int MatMulDynamicSdotInt8Kernel::MatMulDynamicRunArm64Sdot() {
  int8_t *a_ptr = reinterpret_cast<int8_t *>(in_tensors_.at(0)->data());
  int8_t *b_ptr = reinterpret_cast<int8_t *>(in_tensors_.at(1)->data());
  float *c_ptr = reinterpret_cast<float *>(out_tensors_.at(0)->data());
  CHECK_NULL_RETURN(a_ptr);
  CHECK_NULL_RETURN(b_ptr);
  CHECK_NULL_RETURN(c_ptr);

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
    batch_c_ptr_ = c_ptr + i * param_->row_ * param_->col_;

    ret = ParallelLaunch(this->ms_context_, Arm64SdotRun, this, thread_count_);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Arm64SdotRun error: [" << ret << "]";
      return ret;
    }
  }
  return RET_OK;
}

int MatMulDynamicSdotInt8Kernel::Run() {
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
  }
  return MatMulDynamicRunArm64Sdot();
}
}  // namespace mindspore::kernel
