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

#ifndef MINDSPORE_LITE_NNACL_FP32_REDUCE_H_
#define MINDSPORE_LITE_NNACL_FP32_REDUCE_H_
#include "nnacl/op_base.h"
#include "nnacl/reduce_parameter.h"

#ifdef __cplusplus
extern "C" {
#endif
int ReduceMean(int outer_size, int inner_size, int axis_size, const float *src_data, float *dst_data, int tid,
               int thread_num);
int IntReduceMean(int outer_size, int inner_size, int axis_size, const int *src_data, int *dst_data, int tid,
                  int thread_num);
int ReduceSum(int outer_size, int inner_size, int axis_size, const float *src_data, float *dst_data, int tid,
              int thread_num);
int IntReduceSum(int outer_size, int inner_size, int axis_size, const int *src_data, int *dst_data, int tid,
                 int thread_num);
int ReduceMax(int outer_size, int inner_size, int axis_size, const float *src_data, float *dst_data, int tid,
              int thread_num);
int IntReduceMax(int outer_size, int inner_size, int axis_size, const int *src_data, int *dst_data, int tid,
                 int thread_num);
int ReduceMin(int outer_size, int inner_size, int axis_size, const float *src_data, float *dst_data, int tid,
              int thread_num);
int IntReduceMin(int outer_size, int inner_size, int axis_size, const int *src_data, int *dst_data, int tid,
                 int thread_num);
int ReduceProd(int outer_size, int inner_size, int axis_size, const float *src_data, float *dst_data, int tid,
               int thread_num);
int IntReduceProd(int outer_size, int inner_size, int axis_size, const int *src_data, int *dst_data, int tid,
                  int thread_num);
int ReduceSumSquare(int outer_size, int inner_size, int axis_size, const float *src_data, float *dst_data, int tid,
                    int thread_num);
int ReduceAll(int outer_size, int inner_size, int axis_size, const bool *src_data, bool *dst_data, int tid,
              int thread_num);

#ifdef ENABLE_NNACL_INFER_SHAPE
int ReduceInferShape(int **in_shape, size_t *dim_size, int *out_shape, int *in_format, int *out_format,
                     int *in_datatype, int *out_datatype, OpParameter *param);
#endif
#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_LITE_NNACL_FP32_REDUCE_H_
