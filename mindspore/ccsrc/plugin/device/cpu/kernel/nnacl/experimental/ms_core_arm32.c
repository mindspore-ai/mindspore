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

#ifdef ENABLE_ARM32
#include "nnacl/experimental/ms_core.h"
void InitOptMatmulTileArm32(int *row_tile, int *col_tile) {
  *row_tile = C12NUM;
  *col_tile = C4NUM;
}

void InitArm32Core(CoreFuncs *funcs_) {
  funcs_->pack = C4NUM;
  funcs_->byte = sizeof(float);
  funcs_->OptMatmulTile = InitOptMatmulTileArm32;
}
#endif
