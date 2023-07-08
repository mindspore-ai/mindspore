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

#include "nnacl/kernel/activation.h"
#include "nnacl/activation_parameter.h"
#include "nnacl/op_base.h"
#include "nnacl/fp32/activation_fp32.h"
#include "nnacl/kernel/default_kernel_base.h"
#include "nnacl/tensor_c_utils.h"
#ifdef ENABLE_FP16
#include "nnacl/fp16/activation_fp16.h"
#endif

typedef struct ActivationStruct {
  KernelBase base;
  int data_type_;
  ActType act_type_;
} ActivationStruct;

int ActivationResize(struct KernelBase *self) {
  ActivationStruct *activation = (ActivationStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(activation);
  self->thread_nr_ = self->UpdateThread(TC_TYPE(PrimType_Activation, activation->act_type_), 1, 1,
                                        GetElementNum(self->out_[0]), self->thread_nr_);
  return NNACL_OK;
}

int activation_fp32_run(ActivationStruct *activation, int task_id, int count, int stride) {
  ActivationParameter *param = (ActivationParameter *)activation->base.param_;
  float *input = activation->base.in_[0]->data_;
  float *output = activation->base.out_[0]->data_;
  NNACL_CHECK_NULL_RETURN_ERR(input);
  NNACL_CHECK_NULL_RETURN_ERR(output);

  switch (activation->act_type_) {
    case ActType_Relu:
      return Fp32Relu(input + task_id * stride, count, output + task_id * stride);
    case ActType_Relu6:
      return Fp32Relu6(input + task_id * stride, count, output + task_id * stride);
    case ActType_LeakyRelu:
      return LRelu(input + task_id * stride, count, output + task_id * stride, param->alpha_);
    case ActType_Sigmoid:
      return Sigmoid(input + task_id * stride, count, output + task_id * stride);
    case ActType_Tanh:
      return Tanh(input + task_id * stride, count, output + task_id * stride);
    case ActType_Swish:
      return Swish(input + task_id * stride, count, output + task_id * stride);
    case ActType_HSwish:
      return HSwish(input + task_id * stride, count, output + task_id * stride);
    case ActType_HSigmoid:
      return HSigmoid(input + task_id * stride, count, output + task_id * stride);
    case ActType_HardTanh:
      return HardTanh(input + task_id * stride, count, output + task_id * stride, param->min_val_, param->max_val_);
    case ActType_Gelu:
      return Gelu(input + task_id * stride, count, output + task_id * stride, param->approximate_);
    case ActType_Softplus:
      return Softplus(input + task_id * stride, count, output + task_id * stride);
    case ActType_Elu:
      return Elu(input + task_id * stride, count, output + task_id * stride, param->alpha_);
    default:
      return NNACL_ACTIVATION_TYPE_INVALID;
  }
}

int activation_int32_run(ActivationStruct *activation, int task_id, int count, int stride) {
  int32_t *input = activation->base.in_[0]->data_;
  int32_t *output = activation->base.out_[0]->data_;
  NNACL_CHECK_NULL_RETURN_ERR(input);
  NNACL_CHECK_NULL_RETURN_ERR(output);

  switch (activation->act_type_) {
    case ActType_Relu:
      return Int32Relu(input + task_id * stride, count, output + task_id * stride);
    default:
      return NNACL_ACTIVATION_TYPE_INVALID;
  }
}

int activation_fp16_run(ActivationStruct *activation, int task_id, int count, int stride) {
#ifdef ENABLE_FP16
  ActivationParameter *param = (ActivationParameter *)activation->base.param_;
  float16_t *input = activation->base.in_[0]->data_;
  float16_t *output = activation->base.out_[0]->data_;
  NNACL_CHECK_NULL_RETURN_ERR(input);
  NNACL_CHECK_NULL_RETURN_ERR(output);

  switch (activation->act_type_) {
    case ActType_Relu:
      return ReluFp16(input + stride * task_id, output + stride * task_id, count);
    case ActType_Relu6:
      return Relu6Fp16(input + stride * task_id, output + stride * task_id, count);
    case ActType_LeakyRelu:
      return LReluFp16(input + stride * task_id, output + stride * task_id, count, param->alpha_);
    case ActType_Sigmoid:
      return SigmoidFp16(input + stride * task_id, output + stride * task_id, count);
    case ActType_Tanh:
      return TanhFp16(input + stride * task_id, output + stride * task_id, count);
    case ActType_HSwish:
      return HSwishFp16(input + stride * task_id, output + stride * task_id, count);
    case ActType_Swish:
      return SwishFp16(input + stride * task_id, output + stride * task_id, count);
    case ActType_HSigmoid:
      return HSigmoidFp16(input + stride * task_id, output + stride * task_id, count);
    case ActType_HardTanh:
      return HardTanhFp16(input + stride * task_id, count, output + stride * task_id, param->min_val_, param->max_val_);
    case ActType_Gelu:
      return GeluFp16(input + stride * task_id, count, output + stride * task_id, true);
    case ActType_Softplus:
      return SoftplusFp16(input + stride * task_id, count, output + stride * task_id);
    case ActType_Elu:
      return EluFp16(input + stride * task_id, count, output + stride * task_id, param->alpha_);
    default:
      return NNACL_ACTIVATION_TYPE_INVALID;
  }
#endif
  return NNACL_DISABLE_FP16;
}

int ActivationImpl(void *cdata, int task_id, float l, float r) {
  ActivationStruct *activation = (ActivationStruct *)cdata;

  int ele_num = GetElementNum(activation->base.in_[0]);
  NNACL_CHECK_ZERO_RETURN_ERR(activation->base.thread_nr_);
  int stride = UP_DIV(ele_num, activation->base.thread_nr_);
  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(stride, task_id, NNACL_ERR);
  int count = MSMIN(stride, ele_num - stride * task_id);
  if (count <= 0) {
    return NNACL_OK;
  }
  NNACL_CHECK_FALSE(INT_MUL_OVERFLOW(stride, task_id), NNACL_ERR);

  switch (activation->data_type_) {
    case kNumberTypeFloat32:
      return activation_fp32_run(activation, task_id, count, stride);
    case kNumberTypeFloat16:
      return activation_fp16_run(activation, task_id, count, stride);
    case kNumberTypeInt32:
      return activation_int32_run(activation, task_id, count, stride);
    default:
      return NNACL_ACTIVATION_TYPE_INVALID;
  }
}

int ActivationCompute(struct KernelBase *self) {
  return self->env_->ParallelLaunch(self->env_->thread_pool_, ActivationImpl, self, self->thread_nr_);
}

KernelBase *CreateActivation(OpParameter *param, int data_type) {
  ActivationParameter *act = (ActivationParameter *)(param);

  int type = act->type_;
  if (data_type == kNumberTypeInt32) {
    if (type != ActType_Relu) {
      return NULL;
    }
  }

  if (data_type == kNumberTypeFloat32 || data_type == kNumberTypeFloat16) {
    if (type != ActType_Relu && type != ActType_Relu6 && type != ActType_LeakyRelu && type != ActType_Sigmoid &&
        type != ActType_Tanh && type != ActType_HSwish && type != ActType_Swish && type != ActType_HardTanh &&
        type != ActType_Gelu && type != ActType_HSigmoid && type != ActType_Softplus && type != ActType_Elu) {
      return NULL;
    }
  }

  ActivationStruct *activation = (ActivationStruct *)malloc(sizeof(ActivationStruct));
  NNACL_MALLOC_CHECK_NULL_RETURN_NULL(activation);
  memset(activation, 0, sizeof(ActivationStruct));

  activation->data_type_ = data_type;
  activation->act_type_ = act->type_;
  activation->base.Prepare = DefaultPrepare1In1Out;
  activation->base.Release = DefaultRelease;
  activation->base.Resize = ActivationResize;
  activation->base.Compute = ActivationCompute;
  return (KernelBase *)activation;
}

REG_KERNEL_CREATOR(PrimType_Activation, kNumberTypeFloat32, CreateActivation)
REG_KERNEL_CREATOR(PrimType_Activation, kNumberTypeFloat16, CreateActivation)
REG_KERNEL_CREATOR(PrimType_Activation, kNumberTypeUInt32, CreateActivation)
