/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "nnacl/experimental/ms_core.h"
#include "nnacl/op_base.h"
#include "nnacl/fp32/pack_fp32.h"
#include "nnacl/fp32/exp_fp32.h"
#include "nnacl/kernel/matmul_experimental.h"

void GetPostParameters(ActType act, float *min, float *max) {
#define RELU6_VALUE 6.0f
#define RELU_VALUE 0.0f
  *min = -FLT_MAX;
  *max = FLT_MAX;

  if (act == ActType_Relu) {
    *min = RELU_VALUE;
  }
  if (act == ActType_Relu6) {
    *min = RELU_VALUE;
    *max = RELU6_VALUE;
  }
  return;
}

void InitOptMatmulTile(int *row_tile, int *col_tile) {
  *row_tile = C12NUM;
  *col_tile = C8NUM;
}

void InitCore(CoreFuncs *funcs_) {
  funcs_->pack = C4NUM;
  funcs_->byte = sizeof(float);
  funcs_->ExpMatmulTile = InitExpMMFp32TileCount;
  funcs_->PackNcX = PackNCHWToNC4HW4Fp32;
  funcs_->UnPackNcX = PackNC4HW4ToNCHWFp32;
  funcs_->ExpMatmulPackIn = PackExpMatmulIn;
  funcs_->ExpMatmulBlock = ExpMatMulBlock;
  funcs_->ExpMatMulRemain = ExpMatmulRemain;
  funcs_->ExpFusion = ExpFusionFp32;
  funcs_->OptMatmulTile = InitOptMatmulTile;
  funcs_->PostParam = GetPostParameters;
}
