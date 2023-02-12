/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "nnacl/fp32/sparse_segment_sum_fp32.h"
#include "nnacl/errorcode.h"
#include "nnacl/common_func.h"
#include "nnacl/sparse_segment_sum_fp32_simd.h"

int SparseSegmentSumCalcInt32(int32_t *in_data_ptr_int32, int32_t *in_indcie_ptr, int32_t *in_segment_ids_ptr,
                              int32_t *out_data_ptr_int32, size_t data_ids_num, size_t data_num) {
  int oldindex = -1;
  for (size_t i = 0; i < data_ids_num; i++) {
    if (oldindex != in_segment_ids_ptr[i]) {
      oldindex = in_segment_ids_ptr[i];
      for (size_t j = 0; j < data_num; j++) {
        out_data_ptr_int32[j + oldindex * data_num] = 0;
      }
    }

    int32_t *in_data_ptr_int32_tmp = in_data_ptr_int32 + in_indcie_ptr[i] * data_num;
    int32_t *out_data_ptr_int32_tmp = out_data_ptr_int32 + oldindex * data_num;

    size_t index = 0;
    SIMD_RUN_NO_SCALAR(SparseSegmentSumCalcInt32, index, in_data_ptr_int32_tmp, out_data_ptr_int32_tmp, data_num);
    for (; index < data_num; index++) {
      out_data_ptr_int32_tmp[index] += in_data_ptr_int32_tmp[index];
    }
  }
  return NNACL_OK;
}

int SparseSegmentSumCalcFp32(float *in_data_ptr_fp32, int32_t *in_indcie_ptr, int32_t *in_segment_ids_ptr,
                             float *out_data_ptr_fp32, size_t data_ids_num, size_t data_num) {
  int oldindex = -1;
  for (size_t i = 0; i < data_ids_num; i++) {
    if (oldindex != in_segment_ids_ptr[i]) {
      oldindex = in_segment_ids_ptr[i];
      for (size_t j = 0; j < data_num; j++) {
        out_data_ptr_fp32[j + oldindex * data_num] = 0;
      }
    }

    float *in_data_ptr_fp32_tmp = in_data_ptr_fp32 + in_indcie_ptr[i] * data_num;
    float *out_data_ptr_fp32_tmp = out_data_ptr_fp32 + oldindex * data_num;
    size_t index = 0;
    SIMD_RUN_NO_SCALAR(SparseSegmentSumCalcFp32, index, in_data_ptr_fp32_tmp, out_data_ptr_fp32_tmp, data_num);
    for (; index < data_num; index++) {
      out_data_ptr_fp32_tmp[index] += in_data_ptr_fp32_tmp[index];
    }
  }

  return NNACL_OK;
}
