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

#include "nnacl/kernel/arithmetic_compare.h"
#include "nnacl/kernel/arithmetic.h"
#include "nnacl/fp32/arithmetic_fp32.h"
#include "nnacl/fp32/arithmetic_compare_fp32.h"

typedef struct ArithmeticCompareFuncions {
  int primitive_type_;
  int (*compute_f32_)(const float *input0, const float *input1, uint8_t *output, int element_size);
  int (*compute_i32_)(const int *input0, const int *input1, uint8_t *output, int element_size);
  int (*optimize_f32)(const float *input0, const float *input1, uint8_t *output, int element_size, bool first_scalar);
  int (*optimize_i32)(const int *input0, const int *input1, uint8_t *output, int element_size, bool first_scalar);
  int (*compute_i64)(const int64_t *input0, const int64_t *input1, uint8_t *output, int element_size);
  int (*optimize_i64)(const int64_t *input0, const int64_t *input1, uint8_t *output, int element_size,
                      bool first_scalar);
  int (*compute_bool)(const bool *input0, const bool *input1, uint8_t *output, int element_size);
} ArithmeticCompareFuncions;

typedef struct ArithmeticCompareStruct {
  ArithmeticStruct arithmetic_;
  ArithmeticCompareFuncions functions_;
} ArithmeticCompareStruct;

void InitArithmeticCompareRunFunction(KernelBase *self) {
  ArithmeticCompareStruct *arithmetic_compare = (ArithmeticCompareStruct *)self;
  NNACL_CHECK_NULL_RETURN_VOID(arithmetic_compare);

  ArithmeticCompareFuncions fun_table[] = {
    {PrimType_Equal, ElementEqualFp32, ElementEqualInt32, ElementOptEqualFp32, ElementOptEqualInt32, NULL, NULL,
     ElementEqualBool},
    {PrimType_NotEqual, ElementNotEqualFp32, ElementNotEqualInt32, ElementOptNotEqualFp32, ElementOptNotEqualInt32,
     ElementNotEqualInt64, ElementOptNotEqualInt64, NULL},
    {PrimType_Less, ElementLessFp32, ElementLessInt32, ElementOptLessFp32, ElementOptLessInt32, NULL, NULL, NULL},
    {PrimType_LessEqual, ElementLessEqualFp32, ElementLessEqualInt32, ElementOptLessEqualFp32, ElementOptLessEqualInt32,
     NULL, NULL, NULL},
    {PrimType_Greater, ElementGreaterFp32, ElementGreaterInt32, ElementOptGreaterFp32, ElementOptGreaterInt32, NULL,
     NULL, NULL},
    {PrimType_GreaterEqual, ElementGreaterEqualFp32, ElementGreaterEqualInt32, ElementOptGreaterEqualFp32,
     ElementOptGreaterEqualInt32, NULL, NULL, NULL}};

  size_t length = sizeof(fun_table) / sizeof(ArithmeticCompareFuncions);
  for (size_t i = 0; i < length; i++) {
    if (fun_table[i].primitive_type_ == arithmetic_compare->arithmetic_.primitive_type_) {
      arithmetic_compare->functions_ = fun_table[i];
      return;
    }
  }
}

int ArithmeticCompareExecute(KernelBase *base, const void *input0, const void *input1, void *output, int64_t size) {
  ArithmeticCompareStruct *arithmetic_compare = (ArithmeticCompareStruct *)base;
  NNACL_CHECK_NULL_RETURN_ERR(input0);
  NNACL_CHECK_NULL_RETURN_ERR(input1);

  int data_type = base->in_[FIRST_INPUT]->data_type_;
  bool first_scalar = arithmetic_compare->arithmetic_.in_elements_num0_ == 1;

  if (data_type == kNumberTypeFloat32) {
    if (arithmetic_compare->arithmetic_.scalar_opt_) {
      NNACL_CHECK_NULL_RETURN_ERR(arithmetic_compare->functions_.optimize_f32);
      return arithmetic_compare->functions_.optimize_f32((const float *)input0, (const float *)input1,
                                                         (uint8_t *)output, size, first_scalar);
    } else {
      NNACL_CHECK_NULL_RETURN_ERR(arithmetic_compare->functions_.compute_f32_);
      return arithmetic_compare->functions_.compute_f32_((const float *)input0, (const float *)input1,
                                                         (uint8_t *)output, size);
    }
  }

  if (data_type == kNumberTypeInt || data_type == kNumberTypeInt32) {
    if (arithmetic_compare->arithmetic_.scalar_opt_) {
      NNACL_CHECK_NULL_RETURN_ERR(arithmetic_compare->functions_.optimize_i32);
      return arithmetic_compare->functions_.optimize_i32((const int *)input0, (const int *)input1, (uint8_t *)output,
                                                         size, first_scalar);
    } else {
      NNACL_CHECK_NULL_RETURN_ERR(arithmetic_compare->functions_.compute_i32_);
      return arithmetic_compare->functions_.compute_i32_((const int *)input0, (const int *)input1, (uint8_t *)output,
                                                         size);
    }
  }

  if (data_type == kNumberTypeInt64) {
    if (arithmetic_compare->arithmetic_.scalar_opt_) {
      NNACL_CHECK_NULL_RETURN_ERR(arithmetic_compare->functions_.optimize_i64);
      return arithmetic_compare->functions_.optimize_i64((const int64_t *)input0, (const int64_t *)input1,
                                                         (uint8_t *)output, size, first_scalar);
    } else {
      NNACL_CHECK_NULL_RETURN_ERR(arithmetic_compare->functions_.compute_i64);
      return arithmetic_compare->functions_.compute_i64((const int64_t *)input0, (const int64_t *)input1,
                                                        (uint8_t *)output, size);
    }
  }
  if (data_type == kNumberTypeBool) {
    if (!arithmetic_compare->arithmetic_.scalar_opt_) {
      NNACL_CHECK_NULL_RETURN_ERR(arithmetic_compare->functions_.compute_bool);
      return arithmetic_compare->functions_.compute_bool((const bool *)input0, (const bool *)input1, (uint8_t *)output,
                                                         size);
    } else {
      return NNACL_UNSUPPORTED_DATA_TYPE;
    }
  }

  return NNACL_UNSUPPORTED_DATA_TYPE;
}

int ArithmeticCompareResize(KernelBase *self) {
  ArithmeticStruct *arithmetic = (ArithmeticStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(arithmetic);
  arithmetic->in_data_size_ = DataTypeCSize(self->in_[FIRST_INPUT]->data_type_);
  arithmetic->out_data_size_ = DataTypeCSize(self->out_[OUTPUT_INDEX]->data_type_);
  return ArithmeticResize(self);
}

KernelBase *CreateArithmeticCompare(OpParameter *param, int data_type) {
  ArithmeticCompareStruct *arithmetic_compare = (ArithmeticCompareStruct *)malloc(sizeof(ArithmeticCompareStruct));
  NNACL_MALLOC_CHECK_NULL_RETURN_NULL(arithmetic_compare);
  memset(arithmetic_compare, 0, sizeof(ArithmeticCompareStruct));

  ArithmeticStruct *arithmetic = (ArithmeticStruct *)arithmetic_compare;
  arithmetic->in_data_size_ = DataTypeCSize(data_type);
  arithmetic->out_data_size_ = DataTypeCSize(data_type);
  arithmetic->block_boundary_infos_size_ = 0;
  arithmetic->a_matrix_.batch_post_sum_ = NULL;
  arithmetic->b_matrix_.batch_post_sum_ = NULL;
  arithmetic->c_matrix_.batch_post_sum_ = NULL;
  arithmetic->broadcast_buffer_[FIRST_INPUT] = NULL;
  arithmetic->broadcast_buffer_[SECOND_INPUT] = NULL;
  arithmetic->tile_function_ = TileOneDimensionFp32;
  arithmetic->init_function_ = InitArithmeticCompareRunFunction;
  arithmetic->execute_ = ArithmeticCompareExecute;
  arithmetic->base_.Prepare = ArithmeticPrepare;
  arithmetic->base_.Resize = ArithmeticCompareResize;
  arithmetic->base_.Release = ArithmeticRelease;
  arithmetic->base_.Compute = ArithmeticCompute;
  return (KernelBase *)arithmetic_compare;
}

REG_KERNEL_CREATOR(PrimType_Equal, kNumberTypeFloat32, CreateArithmeticCompare)
REG_KERNEL_CREATOR(PrimType_Equal, kNumberTypeBool, CreateArithmeticCompare)
REG_KERNEL_CREATOR(PrimType_Equal, kNumberTypeInt32, CreateArithmeticCompare)
REG_KERNEL_CREATOR(PrimType_NotEqual, kNumberTypeFloat32, CreateArithmeticCompare)
REG_KERNEL_CREATOR(PrimType_NotEqual, kNumberTypeInt32, CreateArithmeticCompare)
REG_KERNEL_CREATOR(PrimType_NotEqual, kNumberTypeInt64, CreateArithmeticCompare)
REG_KERNEL_CREATOR(PrimType_Less, kNumberTypeFloat32, CreateArithmeticCompare)
REG_KERNEL_CREATOR(PrimType_Less, kNumberTypeInt32, CreateArithmeticCompare)
REG_KERNEL_CREATOR(PrimType_LessEqual, kNumberTypeFloat32, CreateArithmeticCompare)
REG_KERNEL_CREATOR(PrimType_LessEqual, kNumberTypeInt32, CreateArithmeticCompare)
REG_KERNEL_CREATOR(PrimType_Greater, kNumberTypeFloat32, CreateArithmeticCompare)
REG_KERNEL_CREATOR(PrimType_Greater, kNumberTypeInt32, CreateArithmeticCompare)
REG_KERNEL_CREATOR(PrimType_GreaterEqual, kNumberTypeFloat32, CreateArithmeticCompare)
REG_KERNEL_CREATOR(PrimType_GreaterEqual, kNumberTypeInt32, CreateArithmeticCompare)
