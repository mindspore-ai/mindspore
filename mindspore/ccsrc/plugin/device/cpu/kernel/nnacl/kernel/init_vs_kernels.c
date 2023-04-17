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

#include "nnacl/kernel/init_vs_kernels.h"
#include "nnacl/kernel/activation.h"
#include "nnacl/kernel/biasadd.h"
#include "nnacl/kernel/exp.h"
#include "nnacl/kernel/fullconnection.h"
#include "nnacl/kernel/gather.h"
#include "nnacl/kernel/gather_d.h"
#include "nnacl/kernel/group_norm.h"
#include "nnacl/kernel/matmul.h"
#include "nnacl/kernel/reshape.h"
#include "nnacl/kernel/shape.h"
#include "nnacl/kernel/softmax.h"
#include "nnacl/kernel/tile.h"

void init_vs_kernels_a(KernelCreator **creators) {
  creators[PrimType_Activation][REGIST_DT(kNumberTypeFloat32)] = CreateActivation;
  creators[PrimType_Activation][REGIST_DT(kNumberTypeFloat16)] = CreateActivation;
  creators[PrimType_Activation][REGIST_DT(kNumberTypeUInt32)] = CreateActivation;
  creators[PrimType_BiasAdd][REGIST_DT(kNumberTypeFloat32)] = CreateBiasAdd;
  creators[PrimType_ExpFusion][REGIST_DT(kNumberTypeFloat32)] = CreateExp;
  creators[PrimType_ExpFusion][REGIST_DT(kNumberTypeFloat16)] = CreateExp;
  creators[PrimType_ExpandDims][REGIST_DT(kNumberTypeInt32)] = CreateReshape;
  creators[PrimType_ExpandDims][REGIST_DT(kNumberTypeFloat16)] = CreateReshape;
  creators[PrimType_ExpandDims][REGIST_DT(kNumberTypeFloat32)] = CreateReshape;
  creators[PrimType_ExpandDims][REGIST_DT(kNumberTypeBool)] = CreateReshape;
  creators[PrimType_ExpandDims][REGIST_DT(kNumberTypeInt8)] = CreateReshape;
  creators[PrimType_Flatten][REGIST_DT(kNumberTypeInt32)] = CreateReshape;
  creators[PrimType_Flatten][REGIST_DT(kNumberTypeFloat16)] = CreateReshape;
  creators[PrimType_Flatten][REGIST_DT(kNumberTypeFloat32)] = CreateReshape;
  creators[PrimType_FlattenGrad][REGIST_DT(kNumberTypeFloat16)] = CreateReshape;
  creators[PrimType_FlattenGrad][REGIST_DT(kNumberTypeFloat32)] = CreateReshape;
  creators[PrimType_FullConnection][REGIST_DT(kNumberTypeFloat32)] = CreateFullconnection;
  creators[PrimType_Gather][REGIST_DT(kNumberTypeFloat16)] = CreateGather;
  creators[PrimType_Gather][REGIST_DT(kNumberTypeFloat32)] = CreateGather;
  creators[PrimType_Gather][REGIST_DT(kNumberTypeInt32)] = CreateGather;
  creators[PrimType_Gather][REGIST_DT(kNumberTypeBool)] = CreateGather;
  creators[PrimType_GatherD][REGIST_DT(kNumberTypeFloat32)] = CreateGatherD;
  creators[PrimType_GatherD][REGIST_DT(kNumberTypeInt32)] = CreateGatherD;
  creators[PrimType_GatherD][REGIST_DT(kNumberTypeFloat16)] = CreateGatherD;
  creators[PrimType_GroupNormFusion][REGIST_DT(kNumberTypeFloat32)] = CreateGroupNorm;
}

void init_vs_kernels_i(KernelCreator **creators) {
  creators[PrimType_MatMulFusion][REGIST_DT(kNumberTypeFloat32)] = CreateMatmul;
}

void init_vs_kernels_r(KernelCreator **creators) {
  creators[PrimType_Reshape][REGIST_DT(kNumberTypeInt32)] = CreateReshape;
  creators[PrimType_Reshape][REGIST_DT(kNumberTypeFloat32)] = CreateReshape;
  creators[PrimType_Reshape][REGIST_DT(kNumberTypeFloat16)] = CreateReshape;
  creators[PrimType_Reshape][REGIST_DT(kNumberTypeBool)] = CreateReshape;
  creators[PrimType_Shape][REGIST_DT(kNumberTypeInt32)] = CreateShape;
  creators[PrimType_Shape][REGIST_DT(kNumberTypeBool)] = CreateShape;
  creators[PrimType_Shape][REGIST_DT(kNumberTypeFloat16)] = CreateShape;
  creators[PrimType_Shape][REGIST_DT(kNumberTypeFloat32)] = CreateShape;
  creators[PrimType_Shape][REGIST_DT(kNumberTypeInt8)] = CreateShape;
  creators[PrimType_Shape][REGIST_DT(kNumberTypeUInt8)] = CreateShape;
  creators[PrimType_Shape][REGIST_DT(kNumberTypeInt64)] = CreateShape;
  creators[PrimType_Softmax][REGIST_DT(kNumberTypeFloat16)] = CreateSoftmax;
  creators[PrimType_Softmax][REGIST_DT(kNumberTypeFloat32)] = CreateSoftmax;
  creators[PrimType_Squeeze][REGIST_DT(kNumberTypeFloat32)] = CreateReshape;
  creators[PrimType_Squeeze][REGIST_DT(kNumberTypeFloat16)] = CreateReshape;
  creators[PrimType_Squeeze][REGIST_DT(kNumberTypeInt32)] = CreateReshape;
  creators[PrimType_Squeeze][REGIST_DT(kNumberTypeBool)] = CreateReshape;
  creators[PrimType_TileFusion][REGIST_DT(kNumberTypeInt32)] = CreateTile;
  creators[PrimType_TileFusion][REGIST_DT(kNumberTypeFloat32)] = CreateTile;
  creators[PrimType_TileFusion][REGIST_DT(kNumberTypeFloat16)] = CreateTile;
  creators[PrimType_TileFusion][REGIST_DT(kNumberTypeBool)] = CreateTile;
  creators[PrimType_TileFusion][REGIST_DT(kNumberTypeUInt8)] = CreateTile;
  creators[PrimType_Unsqueeze][REGIST_DT(kNumberTypeFloat16)] = CreateReshape;
  creators[PrimType_Unsqueeze][REGIST_DT(kNumberTypeFloat32)] = CreateReshape;
  creators[PrimType_Unsqueeze][REGIST_DT(kNumberTypeInt32)] = CreateReshape;
  creators[PrimType_Unsqueeze][REGIST_DT(kNumberTypeInt64)] = CreateReshape;
  creators[PrimType_Unsqueeze][REGIST_DT(kNumberTypeBool)] = CreateReshape;
}

void init_vs_kernels(KernelCreator **creators) {
  init_vs_kernels_a(creators);
  init_vs_kernels_i(creators);
  init_vs_kernels_r(creators);
}
