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

#ifndef NNACL_BASE_UNSORTED_SEGMENT_SUM_BASE_H_
#define NNACL_BASE_UNSORTED_SEGMENT_SUM_BASE_H_

#include "nnacl/op_base.h"

#ifdef __cplusplus
extern "C" {
#endif
#define UnsortedSegmentSum(type, type1, input, unit_num, input_dim1, indices, output, output_dim0, output_dim1) \
  UnsortedSegmentSum_##type##_##type1(input, unit_num, input_dim1, indices, output, output_dim0, output_dim1)
int UnsortedSegmentSum_int_int(const int *input, int unit_num, int input_dim1, const int *indices, int *output,
                               int output_dim0, int output_dim1);
int UnsortedSegmentSum_float_int(const float *input, int unit_num, int input_dim1, const int *indices, float *output,
                                 int output_dim0, int output_dim1);
int UnsortedSegmentSum_int_int64_t(const int *input, int unit_num, int input_dim1, const int64_t *indices, int *output,
                                   int output_dim0, int output_dim1);
int UnsortedSegmentSum_float_int64_t(const float *input, int unit_num, int input_dim1, const int64_t *indices,
                                     float *output, int output_dim0, int output_dim1);
#ifdef __cplusplus
}
#endif
#endif  //  NNACL_BASE_UNSORTED_SEGMENT_SUM_BASE_H_
