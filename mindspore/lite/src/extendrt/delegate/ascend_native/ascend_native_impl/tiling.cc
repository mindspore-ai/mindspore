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

#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <string>
#include <iostream>
#include "mindspore/lite/src/extendrt/delegate/ascend_native/ascend_native_impl/tiling.h"
#include "mindspore/lite/src/extendrt/delegate/ascend_native/ascend_native_impl/utils.h"
#include "extendrt/delegate/ascend_native/ascend_native_impl/tiling_if.h"
#include "tiling/tiling_api.h"
#include "src/common/log.h"

using matmul_tiling::CubeFormat;
using matmul_tiling::DataType;
using matmul_tiling::MatmulApiTiling;
using matmul_tiling::MultiCoreMatmulTiling;
using matmul_tiling::TPosition;

namespace mindspore::ascend_native {
void PrepareMatmul(void **tile_data_d, void **tile_data_h, int m, int n, int k, bool ta, bool tb, bool bias, bool bmm,
                   void *stream, void *ctx) {
  int tile_size = tiling_size();
  *tile_data_h = malloc(tile_size);
  int row = m;
  int col = n;
  int deep = k;
  int core_num = (bmm) ? 1 : CUBE_CORE_NUM;
  buildTiling(core_num, row, col, deep, ta, tb, bias, *tile_data_h, tile_size);
  *tile_data_d = ascend_native::MallocCopy(*tile_data_h, tile_size, ctx);
}

void PrepareMatmulExtra(void **tile_data_d, MMExtra *tile_data_h, void *stream, void *ctx, uint32_t bmm_num,
                        uint32_t lda, uint32_t ldb, uint32_t ldc) {
  int tile_size = sizeof(MMExtra);
  buildMMExtra(tile_data_h, bmm_num, lda, ldb, ldc);
  *tile_data_d = ascend_native::MallocCopy(reinterpret_cast<void *>(tile_data_h), tile_size, ctx);
}

int buildTiling(int blockDim, int M, int N, int K, bool transposeA, bool transposeB, bool isBias, void *tilingPtr) {
  optiling::TCubeTiling tilingData;
  TPosition leftPos = TPosition::GM;
  CubeFormat leftFormat = CubeFormat::ND;
  DataType leftDtype = DataType::DT_FLOAT16;

  TPosition rightPos = TPosition::GM;
  CubeFormat rightFormat = CubeFormat::ND;
  DataType rightDtype = DataType::DT_FLOAT16;

  TPosition resPos = TPosition::GM;
  CubeFormat resFormat = CubeFormat::ND;
  DataType resDtype = DataType::DT_FLOAT16;

  TPosition biasPos = TPosition::GM;
  CubeFormat biasFormat = CubeFormat::ND;
  DataType biasDtype = DataType::DT_FLOAT16;

  tilingData.set_usedCoreNum(blockDim);
  if (blockDim == 1) {
    MatmulApiTiling tilingApi;
    tilingApi.SetAType(leftPos, leftFormat, leftDtype, transposeA);
    tilingApi.SetBType(rightPos, rightFormat, rightDtype, static_cast<bool>(transposeB));
    tilingApi.SetCType(resPos, resFormat, resDtype);
    tilingApi.SetBiasType(biasPos, biasFormat, biasDtype);
    tilingApi.SetShape(M, N, K);
    tilingApi.SetOrgShape(M, N, K);
    tilingApi.SetBias(isBias);
    tilingApi.SetBufferSpace(-1, -1, -1);
    tilingApi.SetDoubleBuffer(true, true, true, true);
    int64_t res = tilingApi.GetTiling(tilingData);
    if (res == -1) {
      MS_LOG(ERROR) << "gen tiling failed";
      return res;
    }
  } else {
    MultiCoreMatmulTiling tilingApi;
    tilingApi.SetDim(blockDim);
    tilingApi.SetAType(leftPos, leftFormat, leftDtype, transposeA);
    tilingApi.SetBType(rightPos, rightFormat, rightDtype, transposeB);
    tilingApi.SetCType(resPos, resFormat, resDtype);
    tilingApi.SetBiasType(biasPos, biasFormat, biasDtype);
    tilingApi.SetShape(M, N, K);
    tilingApi.SetOrgShape(M, N, K);

    tilingApi.SetBias(static_cast<bool>(isBias));
    tilingApi.SetBufferSpace(-1, -1, -1);
    tilingApi.SetDoubleBuffer(true, true, true, true);
    int64_t res = tilingApi.GetTiling(tilingData);
    if (res == -1) {
      MS_LOG(ERROR) << "gen tiling failed";
      return res;
    }
  }
  tilingData.SaveToBuffer(tilingPtr, tilingData.GetDataSize());
  return 0;
}

int tilingSize() {
  optiling::TCubeTiling tiling_data;
  return tiling_data.GetDataSize();
}

}  // namespace mindspore::ascend_native
