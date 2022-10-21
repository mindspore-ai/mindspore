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

#include "src/litert/kernel/cpu/int8/matmul_dynamic_int8.h"
#include "src/litert/kernel/cpu/int8/opt_op_handler.h"
#include "nnacl/int8/matmul_int8.h"
#include "nnacl/int8/dynamic_matmul_int8.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_MEMORY_FAILED;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
namespace {
int MatmulDynamicInt8Run(void *cdata, int task_id, float, float) {
  CHECK_NULL_RETURN(cdata);
  auto op = reinterpret_cast<MatmulDynamicInt8CPUKernel *>(cdata);
  auto ret = op->RunImpl(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "MatmulInt8Run error task_id[" << task_id << "] error_code[" << ret << "]";
    return ret;
  }
  return RET_OK;
}
}  // namespace

int MatmulDynamicInt8CPUKernel::RunImpl(int task_id) {
  int stride = thread_stride_ * col_tile_;
  int cur_stride = task_id * stride;
  int res_stride = param_->col_ - cur_stride;
  int cur_oc = MSMIN(stride, res_stride);
  if (cur_oc <= 0) {
    return RET_OK;
  }
  float *bias_ptr = fp32_bias_ptr_;
  if (fp32_bias_ptr_ != nullptr) {
    bias_ptr += cur_stride;
  }
  float *filter_scale = quant_param_->filter_scale_;
  int32_t filter_zp = quant_param_->filter_zp_[0];
  if (filter_per_channel_) {
    filter_scale += cur_stride;
  }
  int64_t act_type = static_cast<int64_t>(param_->act_type_);

  DynamicMatmul4x16x4AIWI(batch_a_ptr_, batch_b_ptr_ + cur_stride * param_->deep_align_, bias_ptr,
                          batch_c_ptr_ + cur_stride, param_->row_, cur_oc, param_->deep_, param_->deep_align_,
                          param_->col_, quant_param_->input_zp_, quant_param_->input_scale_, filter_scale, filter_zp,
                          filter_per_channel_, act_type);
  return RET_OK;
}

void MatmulDynamicInt8CPUKernel::InitParameter() {
  param_->a_const_ = (in_tensors_[kInputIndex]->data() != nullptr);
  param_->b_const_ = (in_tensors_[kWeightIndex]->data() != nullptr);
  row_tile_ = C4NUM;
  col_tile_ = C4NUM;
  deep_tile_ = C16NUM;
  if (param_->a_transpose_) {
    a_pack_func_ = RowMajor2Col16x4MajorInt8;
  } else {
    a_pack_func_ = RowMajor2Row16x4MajorInt8;
  }
  if (param_->b_transpose_) {
    b_pack_func_ = RowMajor2Row16x4MajorInt8;
  } else {
    b_pack_func_ = RowMajor2Col16x4MajorInt8;
  }
  return;
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
    memset(pack_a_ptr_, quant_param_->input_zp_, param_->row_align_ * param_->deep_align_ * sizeof(int8_t));
    auto current_src_a = a_ptr + a_offset_[i] * param_->row_ * param_->deep_;
    if (param_->a_transpose_) {
      MS_CHECK_TRUE_RET(a_pack_func_ != nullptr, RET_ERROR);
      a_pack_func_(current_src_a, pack_a_ptr_, param_->deep_, param_->row_);
    } else {
      MS_CHECK_TRUE_RET(a_pack_func_ != nullptr, RET_ERROR);
      a_pack_func_(current_src_a, pack_a_ptr_, param_->row_, param_->deep_);
    }

    batch_a_ptr_ = pack_a_ptr_;
    batch_b_ptr_ = pack_b_ptr_ + b_offset_[i] * param_->col_align_ * param_->deep_align_;
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
