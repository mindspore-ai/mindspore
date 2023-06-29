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

#ifndef NNACL_KERNEL_REDUCE_H_
#define NNACL_KERNEL_REDUCE_H_

#include "nnacl/op_base.h"
#include "nnacl/tensor_c.h"
#include "nnacl/kernel.h"

typedef struct ReduceKernelList {
  int type_;
  int (*float_function_)(const int outer_size, const int inner_size, const int axis_size, const float *src_data,
                         float *dst_data, const int tid, const int thread_num);
  int (*int_function_)(const int outer_size, const int inner_size, const int axis_size, const int *src_data,
                       int *dst_data, const int tid, const int thread_num);
  int (*bool_function_)(const int outer_size, const int inner_size, const int axis_size, const bool *src_data,
                        bool *dst_data, const int tid, const int thread_num);
  int (*float_last_axis_func_)(const int outer_size, const int inner_size, const int axis_size, const float *src_data,
                               float *dst_data, const int tid, const int thread_num);
} ReduceKernelList;

typedef struct ReduceStruct {
  KernelBase base_;
  bool only_copy_;
  int num_axes_;
  TypeIdC data_type_;
  int axes_[MAX_SHAPE_SIZE];

  void *data_buffers_[MAX_SHAPE_SIZE];
  size_t data_buffer_sizes_[MAX_SHAPE_SIZE];
  int data_buffers_size_;
  ReduceModeC mode_;

  int outer_sizes_[MAX_SHAPE_SIZE];
  int inner_sizes_[MAX_SHAPE_SIZE];
  int axis_sizes_[MAX_SHAPE_SIZE];
  int offset_size_;

  int outer_size_;
  int inner_size_;
  int axis_size_;

  void *src_data_;
  void *dst_data_;
  ReduceKernelList compute_;

  void (*handle_sum_square_)(KernelBase *base);
  void (*init_kernel_list_)(KernelBase *base);
  int (*calculate_coeff_)(KernelBase *base);
  int (*call_uint_)(KernelBase *base, int task_id);
} ReduceStruct;

KernelBase *CreateReduce(OpParameter *param, int data_type);
int ReducePrepare(KernelBase *self);
int ReduceResize(KernelBase *self);
int ReduceCompute(KernelBase *self);

#endif  // NNACL_KERNEL_RESHAPE_H_
