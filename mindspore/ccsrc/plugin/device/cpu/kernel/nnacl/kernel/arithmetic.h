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
#ifndef NNACL_KERNEL_ARITHMETIC_H_
#define NNACL_KERNEL_ARITHMETIC_H_

#include "nnacl/op_base.h"
#include "nnacl/tensor_c.h"
#include "nnacl/kernel.h"
#include "nnacl/arithmetic_parameter.h"

typedef struct ArithmeticFuncions {
  int primitive_type_;
  int activation_type_;
  int (*compute_f32_)(const float *in1, const float *in2, float *out, int ele);
  int (*compute_int_)(const int *in1, const int *in2, int *out, int ele);
  int (*compute_bool_)(const bool *in1, const bool *in2, bool *out, int ele);
  int (*optimzie_f32_)(const float *in1, const float *in2, float *out, int ele, bool scalar);
  int (*optimzie_int_)(const int *in1, const int *in2, int *out, int ele, bool scalar);
  int (*optimzie_bool_)(const bool *in1, const bool *in2, bool *out, int ele, bool scalar);
} ArithmeticFuncions;

typedef struct ArithmeticMatrixInfo {
  bool is_const_;
  bool is_valid_;
  void *data_;
  int64_t inner_size_;
  int shape_[ARITHMETIC_SUPPORT_DIMS_NUM];
  int shape_size_;
  int *batch_post_sum_; /* shape size + 1 */
} ArithmeticMatrixInfo;

typedef struct ArithmeticBlockBoundaryInfo {
  int batch_begin_;
  int batch_end_;
  int size_begin_;  // start-offset under the begin batch
  int size_end_;    // end-num under the ending batch
  int *a_offset_;
  int *b_offset_;
  bool init_offset_;
} ArithmeticBlockBoundaryInfo;

typedef struct ArithmeticStruct {
  KernelBase base_;
  bool scalar_opt_;
  int primitive_type_;
  int ndim_;
  int in_data_size_;
  int out_data_size_;
  int batch_tail_dim_;

  ArithmeticMatrixInfo a_matrix_;
  ArithmeticMatrixInfo b_matrix_;
  ArithmeticMatrixInfo c_matrix_;
  ArithmeticFuncions functions_;

  void *broadcast_buffer_[TWO_TENSOR];
  int block_boundary_infos_size_;
  ArithmeticBlockBoundaryInfo block_boundary_infos_[MAX_THREAD_NUM];

  int in_shape0_[ARITHMETIC_SUPPORT_DIMS_NUM];
  int in_elements_num0_;
  int in_shape1_[ARITHMETIC_SUPPORT_DIMS_NUM];
  int in_elements_num1_;
  int out_shape_[ARITHMETIC_SUPPORT_DIMS_NUM];
  int out_elements_num_;
  int in_strides0_[ARITHMETIC_SUPPORT_DIMS_NUM];
  int in_strides1_[ARITHMETIC_SUPPORT_DIMS_NUM];
  int out_strides_[ARITHMETIC_SUPPORT_DIMS_NUM];
  int multiples0_[ARITHMETIC_SUPPORT_DIMS_NUM];
  int multiples1_[ARITHMETIC_SUPPORT_DIMS_NUM];

  void (*tile_function_)(const void *inPtr, void *outPtr, int dim, size_t ndim, const int *inShape,
                         const int *inStrides, const int *outStrides, const int *multiple);
  int (*execute_)(KernelBase *base, const void *input0, const void *input1, void *output, int64_t size);
  void (*init_function_)(KernelBase *base);
} ArithmeticStruct;

KernelBase *CreateArithmetic(OpParameter *param, int data_type);
int ArithmeticPrepare(struct KernelBase *self);
int ArithmeticRelease(struct KernelBase *self);
int ArithmeticCompute(struct KernelBase *self);
int ArithmeticResize(struct KernelBase *self);

#endif  // NNACL_KERNEL_ARITHMETIC_H_
