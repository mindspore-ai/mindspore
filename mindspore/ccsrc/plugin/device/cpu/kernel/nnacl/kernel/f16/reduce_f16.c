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

#include "nnacl/kernel/f16/reduce_f16.h"
#include "nnacl/fp16/reduce_fp16.h"
#include "nnacl/tensor_c_utils.h"
#include "nnacl/kernel/default_kernel_base.h"

typedef struct ReduceF16Compute {
  int type_;
  int (*f16_reducer_)(const int outer_size, const int inner_size, const int axis_size, const float16_t *src_data,
                      float16_t *dst_data, const int tid, const int thread_num);
} ReduceF16Compute;

typedef struct ReduceF16Struct {
  ReduceStruct reduce_;
  ReduceF16Compute compute_;
} ReduceF16Struct;

int CallReduceF16Unit(KernelBase *base, int task_id) {
  ReduceF16Struct *reduce_f16 = (ReduceF16Struct *)base;
  NNACL_CHECK_NULL_RETURN_ERR(reduce_f16->reduce_.src_data_);
  NNACL_CHECK_NULL_RETURN_ERR(reduce_f16->reduce_.src_data_);
  NNACL_CHECK_NULL_RETURN_ERR(reduce_f16->compute_.f16_reducer_);

  return reduce_f16->compute_.f16_reducer_(reduce_f16->reduce_.outer_size_, reduce_f16->reduce_.inner_size_,
                                           reduce_f16->reduce_.axis_size_,
                                           (const float16_t *)reduce_f16->reduce_.src_data_,
                                           (float16_t *)reduce_f16->reduce_.dst_data_, task_id, base->thread_nr_);
}

void InitialReduceF16KernelList(KernelBase *base) {
  ReduceF16Struct *reduce_f16 = (ReduceF16Struct *)base;
  ReduceParameter *param = (ReduceParameter *)(base->param_);

  ReduceF16Compute func_list[] = {{Reduce_Sum, ReduceSumFp16},   {Reduce_Mean, ReduceMeanFp16},
                                  {Reduce_Max, ReduceMaxFp16},   {Reduce_Min, ReduceMinFp16},
                                  {Reduce_Prod, ReduceProdFp16}, {Reduce_SumSquare, ReduceSumFp16},
                                  {Reduce_ASum, ReduceSumFp16},  {Reduce_L2, ReduceL2NormFp16}};

  size_t list_len = sizeof(func_list) / sizeof(ReduceF16Compute);
  for (size_t i = 0; i < list_len; ++i) {
    if (param->mode_ == func_list[i].type_) {
      reduce_f16->compute_ = func_list[i];
      return;
    }
  }
}

void HandleReduceF16ASumAndSumSquare(KernelBase *base) {
  TensorC *in_tensor = base->in_[FIRST_INPUT];
  NNACL_CHECK_NULL_RETURN_VOID(in_tensor);
  float16_t *data = (float16_t *)in_tensor->data_;
  NNACL_CHECK_NULL_RETURN_VOID(data);

  int num = GetElementNum(in_tensor);

  if (((ReduceParameter *)base->param_)->mode_ == Reduce_ASum) {
    for (int i = 0; i < num; ++i) {
      if (data[i] < 0.0f) {
        data[i] = 0.0f - data[i];
      }
    }
  }

  if (((ReduceParameter *)base->param_)->mode_ == Reduce_SumSquare) {
    for (int i = 0; i < num; ++i) {
      data[i] = data[i] * data[i];
    }
    return;
  }
}

int CalculateReduceF16CoeffOutput(KernelBase *base) {
  TensorC *out_tensor = base->out_[OUTPUT_INDEX];
  int num = GetElementNum(out_tensor);

  float16_t *out_data = (float16_t *)out_tensor->data_;
  for (int i = 0; i < num; ++i) {
    out_data[i] *= ((ReduceParameter *)base->param_)->coeff;
  }
  return NNACL_OK;
}

KernelBase *CreateReduceF16(OpParameter *param, int data_type) {
  ReduceF16Struct *reduce_f16 = (ReduceF16Struct *)malloc(sizeof(ReduceF16Struct));
  NNACL_MALLOC_CHECK_NULL_RETURN_NULL(reduce_f16);
  memset(reduce_f16, 0, sizeof(ReduceF16Struct));

  ReduceStruct *reduce = &reduce_f16->reduce_;
  reduce->data_type_ = data_type;
  reduce->base_.Release = DefaultRelease;
  reduce->base_.Prepare = ReducePrepare;
  reduce->base_.Resize = ReduceResize;
  reduce->base_.Compute = ReduceCompute;

  reduce->handle_sum_square_ = HandleReduceF16ASumAndSumSquare;
  reduce->calculate_coeff_ = CalculateReduceF16CoeffOutput;
  reduce->init_kernel_list_ = InitialReduceF16KernelList;
  reduce->call_uint_ = CallReduceF16Unit;

  return (KernelBase *)reduce_f16;
}

REG_KERNEL_CREATOR(PrimType_ReduceFusion, kNumberTypeFloat16, CreateReduceF16)
