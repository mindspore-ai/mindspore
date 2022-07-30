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

#ifdef ENABLE_FP16
#include "nnacl/experimental/ms_core.h"
#include "nnacl/fp16/exp_fp16.h"
#include "nnacl/fp16/pack_fp16.h"
#include "nnacl/kernel/matmul_experimental.h"

void InitFp16Core(CoreFuncs *funcs_) {
  funcs_->pack = C8NUM;
  funcs_->byte = sizeof(float16_t);
  funcs_->ExpFusion = ExpFusionFp16;
  funcs_->ExpMatmulTile = InitExpMMFp16TileCount;
  funcs_->PackNcX = PackNCHWFp32ToNC8HW8Fp16;
  funcs_->UnPackNcX = PackNC8HW8ToNCHWFp16;
  funcs_->ExpMatmulPackIn = PackExpMatmulInFp16;
  funcs_->ExpMatmulBlock = ExpMatMulBlockFp16;
  funcs_->ExpMatMulRemain = ExpMatmulRemainFp16;
}
#endif
