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

#include "nnacl/kernel/arithmetic_self.h"
#include "nnacl/fp32/arithmetic_self_fp32.h"
#include "nnacl/kernel/default_kernel_base.h"
#include "nnacl/tensor_c_utils.h"
#ifdef ENABLE_FP16
#include "nnacl/fp16/arithmetic_self_fp16.h"
#endif

void ArithmeticSelfGetArithmeticSelfFunction(ArithmeticSelfStruct *arithmetic_self, int primitive_type) {
  ArithmeticSelfFunction type_func_table[] = {
    {PrimType_Abs, ElementAbs, NULL, ElementAbsInt, NULL},
    {PrimType_Cos, ElementCos, NULL, NULL, NULL},
    {PrimType_Log, ElementLog, NULL, NULL, NULL},
    {PrimType_Log1p, ElementLog1p, NULL, NULL, NULL},
    {PrimType_Square, ElementSquare, NULL, NULL, NULL},
    {PrimType_Sqrt, ElementSqrt, NULL, NULL, NULL},
    {PrimType_Rsqrt, ElementRsqrt, NULL, NULL, NULL},
    {PrimType_Sin, ElementSin, NULL, NULL, NULL},
    {PrimType_LogicalNot, ElementLogicalNot, ElementLogicalNotBool, NULL, NULL},
    {PrimType_Floor, ElementFloor, NULL, NULL, NULL},
    {PrimType_Ceil, ElementCeil, NULL, NULL, NULL},
    {PrimType_Round, ElementRound, NULL, NULL, NULL},
    {PrimType_Neg, ElementNegative, NULL, ElementNegativeInt, NULL},
    {PrimType_Reciprocal, ElementReciprocal, NULL, NULL, NULL},
    {PrimType_Erf, ElementErf, NULL, NULL, NULL},
    {PrimType_IsFinite, NULL, NULL, NULL, ElementIsFinite}};
  for (size_t i = 0; i < sizeof(type_func_table) / sizeof(ArithmeticSelfFunction); i++) {
    if (type_func_table[i].primitive_type_ == primitive_type) {
      arithmetic_self->function_ = type_func_table[i];
      return;
    }
  }
}

void ArithmeticSelfGetArithmeticSelfF16Function(ArithmeticSelfStruct *arithmetic_self, int primitive_type) {
#ifdef ENABLE_FP16
  ArithmeticSelfF16Function type_func_table[] = {{PrimType_Abs, ElementAbsFp16},
                                                 {PrimType_Cos, ElementCosFp16},
                                                 {PrimType_Log, ElementLogFp16},
                                                 {PrimType_Square, ElementSquareFp16},
                                                 {PrimType_Sqrt, ElementSqrtFp16},
                                                 {PrimType_Rsqrt, ElementRsqrtFp16},
                                                 {PrimType_Sin, ElementSinFp16},
                                                 {PrimType_LogicalNot, ElementLogicalNotFp16},
                                                 {PrimType_Floor, ElementFloorFp16},
                                                 {PrimType_Ceil, ElementCeilFp16},
                                                 {PrimType_Round, ElementRoundFp16},
                                                 {PrimType_Neg, ElementNegativeFp16},
                                                 {PrimType_Reciprocal, ElementReciprocalFp16},
                                                 {PrimType_Erf, ElementErfFp16}};
  for (size_t i = 0; i < sizeof(type_func_table) / sizeof(ArithmeticSelfF16Function); i++) {
    if (type_func_table[i].primitive_type_ == primitive_type) {
      arithmetic_self->f16_function_ = type_func_table[i];
      return;
    }
  }
#endif
  arithmetic_self->f16_function_.primitive_type_ = primitive_type;
  return;
}

int ArithmeticSelfExecute(ArithmeticSelfStruct *arithmetic_self, int task_id) {
  int elements_num = GetElementNum(arithmetic_self->base_.in_[FIRST_INPUT]);
  NNACL_CHECK_TRUE_RET(arithmetic_self->base_.thread_nr_, NNACL_ERR);
  int stride = UP_DIV(elements_num, arithmetic_self->base_.thread_nr_);
  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(task_id, stride, NNACL_ERR);
  int offset = task_id * stride;
  int count = NNACL_MIN(stride, elements_num - offset);
  if (count <= 0) {
    return NNACL_OK;
  }

  void *in_data = arithmetic_self->base_.in_[FIRST_INPUT]->data_;
  NNACL_CHECK_NULL_RETURN_ERR(in_data);
  void *out_data = arithmetic_self->base_.out_[OUTPUT_INDEX]->data_;
  NNACL_CHECK_NULL_RETURN_ERR(out_data);
  int in_data_type = arithmetic_self->base_.in_[FIRST_INPUT]->data_type_;
  int out_data_type = arithmetic_self->base_.out_[OUTPUT_INDEX]->data_type_;

  if (in_data_type == kNumberTypeFloat32 && out_data_type == kNumberTypeBool) {
    NNACL_CHECK_NULL_RETURN_ERR(arithmetic_self->function_.func_float_bool_);
    return arithmetic_self->function_.func_float_bool_((float *)in_data + offset, (bool *)out_data + offset, count);
  }

  if (in_data_type == kNumberTypeFloat32) {
    NNACL_CHECK_NULL_RETURN_ERR(arithmetic_self->function_.func_);
    return arithmetic_self->function_.func_((float *)in_data + offset, (float *)out_data + offset, count);
  }

  if (in_data_type == kNumberTypeBool) {
    NNACL_CHECK_NULL_RETURN_ERR(arithmetic_self->function_.func_bool_);
    return arithmetic_self->function_.func_bool_((bool *)in_data + offset, (bool *)out_data + offset, count);
  }

  if (in_data_type == kNumberTypeInt32) {
    NNACL_CHECK_NULL_RETURN_ERR(arithmetic_self->function_.func_int_);
    return arithmetic_self->function_.func_int_((int32_t *)in_data + offset, (int32_t *)out_data + offset, count);
  }

#ifdef ENABLE_FP16
  if (in_data_type == kNumberTypeFloat16) {
    NNACL_CHECK_NULL_RETURN_ERR(arithmetic_self->f16_function_.func_);
    return arithmetic_self->f16_function_.func_((float16_t *)in_data + offset, (float16_t *)out_data + offset, count);
  }
#endif
  return NNACL_ARITHMETIC_SELF_DATA_TYPE_UNSUPPORT;
}

int ArithmeticSelfRun(void *cdata, int task_id, float l, float r) {
  ArithmeticSelfStruct *arithmetic_self = (ArithmeticSelfStruct *)cdata;
  NNACL_CHECK_NULL_RETURN_ERR(arithmetic_self);
  return ArithmeticSelfExecute(arithmetic_self, task_id);
}

int ArithmeticSelfResize(KernelBase *self) {
  ArithmeticSelfStruct *arithmetic_self = (ArithmeticSelfStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(arithmetic_self);
  self->thread_nr_ = arithmetic_self->base_.UpdateThread(TC_PTYPE(arithmetic_self->op_type_), 1, 1,
                                                         GetElementNum(self->out_[OUTPUT_INDEX]), self->thread_nr_);
  return NNACL_OK;
}

int ArithmeticSelfCompute(KernelBase *self) {
  return self->env_->ParallelLaunch(self->env_->thread_pool_, ArithmeticSelfRun, self, self->thread_nr_);
}

int ArithmeticSelfPrepare(KernelBase *self) {
  NNACL_CHECK_FALSE(self->in_size_ != ONE_TENSOR, NNACL_INPUT_TENSOR_ERROR);
  NNACL_CHECK_FALSE(self->out_size_ != ONE_TENSOR, NNACL_OUTPUT_TENSOR_ERROR);
  NNACL_CHECK_FALSE(self->out_[OUTPUT_INDEX]->category_ == ConstTensor, NNACL_OUTPUT_TENSOR_ERROR);
  NNACL_CHECK_FALSE(self->out_[OUTPUT_INDEX]->category_ == ConstScalar, NNACL_OUTPUT_TENSOR_ERROR);
  return NNACL_OK;
}

KernelBase *CreateArithmeticSelf(OpParameter *param, int data_type) {
  ArithmeticSelfStruct *arithmetic_self = (ArithmeticSelfStruct *)malloc(sizeof(ArithmeticSelfStruct));
  NNACL_MALLOC_CHECK_NULL_RETURN_NULL(arithmetic_self);
  ArithmeticSelfGetArithmeticSelfFunction(arithmetic_self, param->type_);
  ArithmeticSelfGetArithmeticSelfF16Function(arithmetic_self, param->type_);
  arithmetic_self->op_type_ = param->type_;
  arithmetic_self->base_.Prepare = ArithmeticSelfPrepare;
  arithmetic_self->base_.Resize = ArithmeticSelfResize;
  arithmetic_self->base_.Release = DefaultRelease;
  arithmetic_self->base_.Compute = ArithmeticSelfCompute;
  return (KernelBase *)arithmetic_self;
}

REG_KERNEL_CREATOR(PrimType_LogicalNot, kNumberTypeBool, CreateArithmeticSelf)

REG_KERNEL_CREATOR(PrimType_Abs, kNumberTypeInt32, CreateArithmeticSelf)
REG_KERNEL_CREATOR(PrimType_Neg, kNumberTypeInt32, CreateArithmeticSelf)

REG_KERNEL_CREATOR(PrimType_Abs, kNumberTypeFloat32, CreateArithmeticSelf)
REG_KERNEL_CREATOR(PrimType_Ceil, kNumberTypeFloat32, CreateArithmeticSelf)
REG_KERNEL_CREATOR(PrimType_Cos, kNumberTypeFloat32, CreateArithmeticSelf)
REG_KERNEL_CREATOR(PrimType_Erf, kNumberTypeFloat32, CreateArithmeticSelf)
REG_KERNEL_CREATOR(PrimType_Floor, kNumberTypeFloat32, CreateArithmeticSelf)
REG_KERNEL_CREATOR(PrimType_Log, kNumberTypeFloat32, CreateArithmeticSelf)
REG_KERNEL_CREATOR(PrimType_Log1p, kNumberTypeFloat32, CreateArithmeticSelf)
REG_KERNEL_CREATOR(PrimType_LogicalNot, kNumberTypeFloat32, CreateArithmeticSelf)
REG_KERNEL_CREATOR(PrimType_Neg, kNumberTypeFloat32, CreateArithmeticSelf)
REG_KERNEL_CREATOR(PrimType_Reciprocal, kNumberTypeFloat32, CreateArithmeticSelf)
REG_KERNEL_CREATOR(PrimType_Round, kNumberTypeFloat32, CreateArithmeticSelf)
REG_KERNEL_CREATOR(PrimType_Rsqrt, kNumberTypeFloat32, CreateArithmeticSelf)
REG_KERNEL_CREATOR(PrimType_Square, kNumberTypeFloat32, CreateArithmeticSelf)
REG_KERNEL_CREATOR(PrimType_Sqrt, kNumberTypeFloat32, CreateArithmeticSelf)
REG_KERNEL_CREATOR(PrimType_Sin, kNumberTypeFloat32, CreateArithmeticSelf)
REG_KERNEL_CREATOR(PrimType_IsFinite, kNumberTypeFloat32, CreateArithmeticSelf)

REG_KERNEL_CREATOR(PrimType_Abs, kNumberTypeFloat16, CreateArithmeticSelf)
REG_KERNEL_CREATOR(PrimType_Cos, kNumberTypeFloat16, CreateArithmeticSelf)
REG_KERNEL_CREATOR(PrimType_Log, kNumberTypeFloat16, CreateArithmeticSelf)
REG_KERNEL_CREATOR(PrimType_Square, kNumberTypeFloat16, CreateArithmeticSelf)
REG_KERNEL_CREATOR(PrimType_Sqrt, kNumberTypeFloat16, CreateArithmeticSelf)
REG_KERNEL_CREATOR(PrimType_Rsqrt, kNumberTypeFloat16, CreateArithmeticSelf)
REG_KERNEL_CREATOR(PrimType_Sin, kNumberTypeFloat16, CreateArithmeticSelf)
REG_KERNEL_CREATOR(PrimType_LogicalNot, kNumberTypeFloat16, CreateArithmeticSelf)
REG_KERNEL_CREATOR(PrimType_Floor, kNumberTypeFloat16, CreateArithmeticSelf)
REG_KERNEL_CREATOR(PrimType_Ceil, kNumberTypeFloat16, CreateArithmeticSelf)
REG_KERNEL_CREATOR(PrimType_Round, kNumberTypeFloat16, CreateArithmeticSelf)
REG_KERNEL_CREATOR(PrimType_Neg, kNumberTypeFloat16, CreateArithmeticSelf)
REG_KERNEL_CREATOR(PrimType_Reciprocal, kNumberTypeFloat16, CreateArithmeticSelf)
REG_KERNEL_CREATOR(PrimType_Erf, kNumberTypeFloat16, CreateArithmeticSelf)
