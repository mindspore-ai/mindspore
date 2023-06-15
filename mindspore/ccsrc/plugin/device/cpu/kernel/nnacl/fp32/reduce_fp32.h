/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_NNACL_FP32_REDUCE_H_
#define MINDSPORE_NNACL_FP32_REDUCE_H_
#include <stddef.h>
#include "nnacl/op_base.h"
#include "nnacl/reduce_parameter.h"

#ifdef __cplusplus
extern "C" {
#endif
int ReduceMean(int outer_size, int inner_size, int axis_size, const float *src_data, float *dst_data, int tid,
               int thread_num);
int IntReduceMean(int outer_size, int inner_size, int axis_size, const int32_t *src_data, int32_t *dst_data, int tid,
                  int thread_num);
int ReduceSum(int outer_size, int inner_size, int axis_size, const float *src_data, float *dst_data, int tid,
              int thread_num);
int ReduceSumByLastAxis(int outer_size, int inner_size, int axis_size, const float *src_data, float *dst_data, int tid,
                        int thread_num);
int IntReduceSum(int outer_size, int inner_size, int axis_size, const int32_t *src_data, int32_t *dst_data, int tid,
                 int thread_num);
int ReduceMax(int outer_size, int inner_size, int axis_size, const float *src_data, float *dst_data, int tid,
              int thread_num);
int ReduceMaxByLastAxis(int outer_size, int inner_size, int axis_size, const float *src_data, float *dst_data, int tid,
                        int thread_num);
int IntReduceMax(int outer_size, int inner_size, int axis_size, const int32_t *src_data, int32_t *dst_data, int tid,
                 int thread_num);
int ReduceMin(int outer_size, int inner_size, int axis_size, const float *src_data, float *dst_data, int tid,
              int thread_num);
int IntReduceMin(int outer_size, int inner_size, int axis_size, const int32_t *src_data, int32_t *dst_data, int tid,
                 int thread_num);
int ReduceProd(int outer_size, int inner_size, int axis_size, const float *src_data, float *dst_data, int tid,
               int thread_num);
int IntReduceProd(int outer_size, int inner_size, int axis_size, const int32_t *src_data, int32_t *dst_data, int tid,
                  int thread_num);
int ReduceSumSquare(int outer_size, int inner_size, int axis_size, const float *src_data, float *dst_data, int tid,
                    int thread_num);
int ReduceL2Norm(int outer_size, int inner_size, int axis_size, const float *src_data, float *dst_data, int tid,
                 int thread_num);
int ReduceAll(int outer_size, int inner_size, int axis_size, const bool *src_data, bool *dst_data, int tid,
              int thread_num);
int ReduceSumDim2Axis0(size_t col_size, size_t col_len, size_t row_len, const float *src_data, float *dst_data);
int ReduceSumDim2Axis1(size_t col_len, const float *src_data, float *dst_data);
int ReduceMeanWithAxis(const float *src_data, float *mean, int64_t size);
int ReduceDeviation(const float *src_data, int64_t size, float mean, float *deviation);

#ifdef ENABLE_NNACL_INFER_SHAPE
int ReduceInferShape(int32_t **in_shape, size_t *dim_size, int32_t *out_shape, int32_t *in_format, int32_t *out_format,
                     int32_t *in_datatype, int32_t *out_datatype, OpParameter *param);
#endif
#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_NNACL_FP32_REDUCE_H_
