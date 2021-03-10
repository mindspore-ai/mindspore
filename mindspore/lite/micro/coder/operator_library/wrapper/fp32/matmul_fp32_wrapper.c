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

#include "wrapper/fp32/matmul_fp32_wrapper.h"
void InitMatrixA(const float *src_ptr, float *dst_ptr, const MatMulParameter *params_, bool is_vector_a) {
  if (is_vector_a) {
    memcpy(dst_ptr, src_ptr, params_->batch * params_->deep_ * sizeof(float));
    return;
  }
  for (int i = 0; i < params_->batch; i++) {
    const float *src = src_ptr + i * params_->deep_ * params_->row_;
    float *dst = dst_ptr + i * params_->deep_ * params_->row_align_;
    if (params_->a_transpose_) {
      RowMajor2Row12Major(src, dst, params_->deep_, params_->row_);
    } else {
      RowMajor2Col12Major(src, dst, params_->row_, params_->deep_);
    }
  }
}

void InitMatrixB(const float *src_ptr, float *dst_ptr, const MatMulParameter *params_, bool is_vector_a) {
  if (is_vector_a) {
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
#ifdef ENABLE_ARM32
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
}
