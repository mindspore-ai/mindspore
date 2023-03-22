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

#include "nnacl/kernel.h"
#include "nnacl/tensor_c.h"
#include "nnacl/op_base.h"
#ifdef _MSC_VER
#include "nnacl/kernel/exp.h"
#include "nnacl/kernel/gather_d.h"
#include "nnacl/kernel/group_norm.h"
#include "nnacl/kernel/reshape.h"
#endif

static KernelCreator g_kernelCreatorRegistry[PrimType_MAX][16];
#define REGIST_DT(DataType) (DataType - kNumberTypeBegin - 1)

void RegKernelCreator(int opType, int dataType, KernelCreator creator) {
  g_kernelCreatorRegistry[opType][REGIST_DT(dataType)] = creator;
}

void Init_MSC_VER_kernels(void) {
#ifdef _MSC_VER
  /* VS env do not support automatic register
   * register here first time */
  static bool inited = false;
  if (inited == false) {
    g_kernelCreatorRegistry[PrimType_ExpFusion][REGIST_DT(kNumberTypeFloat32)] = CreateExp;
    g_kernelCreatorRegistry[PrimType_ExpFusion][REGIST_DT(kNumberTypeFloat16)] = CreateExp;
    g_kernelCreatorRegistry[PrimType_GatherD][REGIST_DT(kNumberTypeFloat32)] = CreateGatherD;
    g_kernelCreatorRegistry[PrimType_GatherD][REGIST_DT(kNumberTypeInt32)] = CreateGatherD;
    g_kernelCreatorRegistry[PrimType_GatherD][REGIST_DT(kNumberTypeFloat16)] = CreateGatherD;
    g_kernelCreatorRegistry[PrimType_GroupNormFusion][REGIST_DT(kNumberTypeFloat32)] = CreateGroupNorm;
    g_kernelCreatorRegistry[PrimType_Reshape][REGIST_DT(kNumberTypeInt32)] = CreateReshape;
    g_kernelCreatorRegistry[PrimType_Reshape][REGIST_DT(kNumberTypeFloat32)] = CreateReshape;
    g_kernelCreatorRegistry[PrimType_Reshape][REGIST_DT(kNumberTypeFloat16)] = CreateReshape;
    g_kernelCreatorRegistry[PrimType_Reshape][REGIST_DT(kNumberTypeBool)] = CreateReshape;
    g_kernelCreatorRegistry[PrimType_Flatten][REGIST_DT(kNumberTypeInt32)] = CreateReshape;
    g_kernelCreatorRegistry[PrimType_Flatten][REGIST_DT(kNumberTypeFloat16)] = CreateReshape;
    g_kernelCreatorRegistry[PrimType_Flatten][REGIST_DT(kNumberTypeFloat32)] = CreateReshape;
    g_kernelCreatorRegistry[PrimType_FlattenGrad][REGIST_DT(kNumberTypeFloat16)] = CreateReshape;
    g_kernelCreatorRegistry[PrimType_FlattenGrad][REGIST_DT(kNumberTypeFloat32)] = CreateReshape;
    g_kernelCreatorRegistry[PrimType_ExpandDims][REGIST_DT(kNumberTypeInt32)] = CreateReshape;
    g_kernelCreatorRegistry[PrimType_ExpandDims][REGIST_DT(kNumberTypeFloat16)] = CreateReshape;
    g_kernelCreatorRegistry[PrimType_ExpandDims][REGIST_DT(kNumberTypeFloat32)] = CreateReshape;
    g_kernelCreatorRegistry[PrimType_ExpandDims][REGIST_DT(kNumberTypeBool)] = CreateReshape;
    g_kernelCreatorRegistry[PrimType_ExpandDims][REGIST_DT(kNumberTypeInt8)] = CreateReshape;
    g_kernelCreatorRegistry[PrimType_Squeeze][REGIST_DT(kNumberTypeFloat32)] = CreateReshape;
    g_kernelCreatorRegistry[PrimType_Squeeze][REGIST_DT(kNumberTypeFloat16)] = CreateReshape;
    g_kernelCreatorRegistry[PrimType_Squeeze][REGIST_DT(kNumberTypeInt32)] = CreateReshape;
    g_kernelCreatorRegistry[PrimType_Squeeze][REGIST_DT(kNumberTypeBool)] = CreateReshape;
    g_kernelCreatorRegistry[PrimType_Unsqueeze][REGIST_DT(kNumberTypeFloat16)] = CreateReshape;
    g_kernelCreatorRegistry[PrimType_Unsqueeze][REGIST_DT(kNumberTypeFloat32)] = CreateReshape;
    g_kernelCreatorRegistry[PrimType_Unsqueeze][REGIST_DT(kNumberTypeInt32)] = CreateReshape;
    g_kernelCreatorRegistry[PrimType_Unsqueeze][REGIST_DT(kNumberTypeInt64)] = CreateReshape;
    g_kernelCreatorRegistry[PrimType_Unsqueeze][REGIST_DT(kNumberTypeBool)] = CreateReshape;
    inited = true;
  }
#endif
  return;
}

bool SupportKernelC(int opType, int dataType) {
  Init_MSC_VER_kernels();
  const int length = 16;
  if (REGIST_DT(dataType) < 0 || REGIST_DT(dataType) >= length) {
    return false;
  }
  KernelCreator creator = g_kernelCreatorRegistry[opType][REGIST_DT(dataType)];
  return creator != NULL;
}

int NnaclKernelInferShape(struct KernelBase *self) { return NNACL_OK; }

KernelBase *CreateKernel(OpParameter *param, TensorC *ins, size_t in_size, TensorC *outs, size_t out_size,
                         int data_type, ExecEnv *env) {
  Init_MSC_VER_kernels();
  KernelCreator creator = g_kernelCreatorRegistry[param->type_][REGIST_DT(data_type)];
  if (creator == NULL) {
    return NULL;
  }
  KernelBase *kernel_base = creator(param, data_type);
  kernel_base->infershape = NnaclKernelInferShape;
  kernel_base->env_ = env;
  kernel_base->param_ = param;
  kernel_base->thread_nr_ = param->thread_num_;
  kernel_base->in_ = ins;
  kernel_base->in_size_ = in_size;
  kernel_base->out_ = outs;
  kernel_base->out_size_ = out_size;
  return kernel_base;
}
