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

#include "nnacl/kernel/f16/arithmetic_compare_f16.h"
#include "nnacl/kernel/f16/arithmetic_f16.h"
#include "nnacl/fp16/arithmetic_fp16.h"

typedef struct ArithmeticCompareF16Funcions {
  int primitive_type_;
  int activation_type_;
  int (*compute_)(const float16_t *input0, const float16_t *input1, uint8_t *output, int element_size);
  int (*optimzie_)(const float16_t *input0, const float16_t *input1, uint8_t *output, int element_size,
                   bool first_scalar);
} ArithmeticCompareF16Funcions;

typedef struct ArithmeticCompareF16Struct {
  ArithmeticF16Struct arithmetic_f16_;
  ArithmeticCompareF16Funcions functions_;
} ArithmeticCompareF16Struct;

void InitArithmeticCompareF16RunFunction(KernelBase *base) {
  ArithmeticCompareF16Struct *arithmetic_compare_f16 = (ArithmeticCompareF16Struct *)base;
  ArithmeticParameter *arithmetic_param = (ArithmeticParameter *)base->param_;

  ArithmeticCompareF16Funcions arithmetic_cp_fun_table_fp16[] = {
    {PrimType_NotEqual, ActType_No, ElementNotEqualFp16, ElementOptNotEqualFp16},
    {PrimType_Equal, ActType_No, ElementEqualFp16, ElementOptEqualFp16},
    {PrimType_Less, ActType_No, ElementLessFp16, ElementOptLessFp16},
    {PrimType_LessEqual, ActType_No, ElementLessEqualFp16, ElementOptLessEqualFp16},
    {PrimType_Greater, ActType_No, ElementGreaterFp16, ElementOptGreaterFp16},
    {PrimType_GreaterEqual, ActType_No, ElementGreaterEqualFp16, ElementOptGreaterEqualFp16}};

  size_t length = sizeof(arithmetic_cp_fun_table_fp16) / sizeof(ArithmeticCompareF16Funcions);
  for (size_t i = 0; i < length; i++) {
    if (arithmetic_cp_fun_table_fp16[i].primitive_type_ ==
          arithmetic_compare_f16->arithmetic_f16_.arithmetic_.primitive_type_ &&
        arithmetic_cp_fun_table_fp16[i].activation_type_ == arithmetic_param->activation_type_) {
      arithmetic_compare_f16->functions_ = arithmetic_cp_fun_table_fp16[i];
      return;
    }
  }
}

int ArithmeticCompareF16DoExecute(KernelBase *base, const void *input0, const void *input1, void *output,
                                  int64_t size) {
  ArithmeticCompareF16Struct *arithmetic_compare_f16 = (ArithmeticCompareF16Struct *)base;

  if (arithmetic_compare_f16->arithmetic_f16_.arithmetic_.scalar_opt_) {
    bool first_scalar = arithmetic_compare_f16->arithmetic_f16_.arithmetic_.in_elements_num0_ == 1;
    NNACL_CHECK_NULL_RETURN_ERR(arithmetic_compare_f16->functions_.optimzie_);
    return arithmetic_compare_f16->functions_.optimzie_((const float16_t *)input0, (const float16_t *)input1,
                                                        (uint8_t *)output, size, first_scalar);
  }

  NNACL_CHECK_NULL_RETURN_ERR(arithmetic_compare_f16->functions_.compute_);
  return arithmetic_compare_f16->functions_.compute_((const float16_t *)input0, (const float16_t *)input1,
                                                     (uint8_t *)output, size);
}
int ArithmeticCompareF16Compute(KernelBase *self) {
  ArithmeticStruct *arithmetic = (ArithmeticStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(arithmetic);
  arithmetic->in_data_size_ = DataTypeCSize(self->in_[FIRST_INPUT]->data_type_);
  arithmetic->out_data_size_ = DataTypeCSize(self->out_[OUTPUT_INDEX]->data_type_);
  return ArithmeticF16Compute(self);
}

KernelBase *CreateArithmeticCompareF16(OpParameter *param, int data_type) {
  ArithmeticCompareF16Struct *arithmetic_compare_f16 =
    (ArithmeticCompareF16Struct *)malloc(sizeof(ArithmeticCompareF16Struct));
  NNACL_CHECK_NULL_RETURN_NULL(arithmetic_compare_f16);
  memset(arithmetic_compare_f16, 0, sizeof(ArithmeticF16Struct));

  ArithmeticStruct *arithmetic = &arithmetic_compare_f16->arithmetic_f16_.arithmetic_;
  arithmetic->block_boundary_infos_size_ = 0;
  arithmetic->a_matrix_.batch_post_sum_ = NULL;
  arithmetic->b_matrix_.batch_post_sum_ = NULL;
  arithmetic->c_matrix_.batch_post_sum_ = NULL;
  arithmetic->broadcast_buffer_[FIRST_INPUT] = NULL;
  arithmetic->broadcast_buffer_[SECOND_INPUT] = NULL;
  arithmetic->base_.Prepare = ArithmeticPrepare;
  arithmetic->base_.Resize = ArithmeticF16Resize;
  arithmetic->base_.Release = ArithmeticRelease;
  arithmetic->base_.Compute = ArithmeticCompareF16Compute;

  arithmetic->execute_ = ArithmeticCompareF16DoExecute;
  arithmetic->tile_function_ = TileOneDimensionFp16;
  arithmetic->init_function_ = InitArithmeticCompareF16RunFunction;

  return (KernelBase *)arithmetic_compare_f16;
}

REG_KERNEL_CREATOR(PrimType_NotEqual, kNumberTypeFloat16, CreateArithmeticCompareF16)
REG_KERNEL_CREATOR(PrimType_Equal, kNumberTypeFloat16, CreateArithmeticCompareF16)
REG_KERNEL_CREATOR(PrimType_Less, kNumberTypeFloat16, CreateArithmeticCompareF16)
REG_KERNEL_CREATOR(PrimType_LessEqual, kNumberTypeFloat16, CreateArithmeticCompareF16)
REG_KERNEL_CREATOR(PrimType_Greater, kNumberTypeFloat16, CreateArithmeticCompareF16)
REG_KERNEL_CREATOR(PrimType_GreaterEqual, kNumberTypeFloat16, CreateArithmeticCompareF16)
