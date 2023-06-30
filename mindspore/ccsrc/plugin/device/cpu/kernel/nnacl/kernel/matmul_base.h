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

#ifndef NNACL_KERNEL_MATMUL_BASE_H_
#define NNACL_KERNEL_MATMUL_BASE_H_

#include "nnacl/op_base.h"
#include "nnacl/tensor_c.h"
#include "nnacl/matmul_parameter.h"
#include "nnacl/kernel/matmul_struct.h"

void MatmulBaseGetThreadCuttingPolicy(MatmulStruct *matmul);
void MatmulBaseFreeBatchOffset(MatmulStruct *matmul);
int MatmulBaseMallocBatchOffset(MatmulStruct *matmul);
int MatmulBaseInitParameter(MatmulStruct *matmul);
int MatmulBasePrepare(KernelBase *self);
int MatmulBaseResize(KernelBase *self);
int MatmulBaseRelease(KernelBase *self);

KernelBase *CreateMatmulBase();

#endif  // NNACL_KERNEL_MATMUL_BASE_H_
