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

#define CLNTSRV
#include <string>
#include "extendrt/delegate/ascend_native/ascend_native_impl/utils.h"
#include "extendrt/delegate/ascend_native/ascend_native_impl/mix_core/matmul.h"
#include "tikcfw/kernel_operator.h"
#include "extendrt/delegate/ascend_native/ascend_native_impl/tiling_if.h"
#include "extendrt/delegate/ascend_native/ascend_native_impl/op_custom_gemm.h"
#include "acl/acl.h"

namespace mindspore::ascend_native {

__global__[mix] void mmfuse_custom_mix(bool ta, bool tb, GM_ADDR mat_a, GM_ADDR mat_b, GM_ADDR mat_c, GM_ADDR bias,
                                       GM_ADDR tiling_data, GM_ADDR ws, GM_ADDR extra) {
  constexpr int pipe = 1;
  if (!ta && !tb)
    kernel_mm_fuse_operator<half, false, false, KernelFuseNone<half, pipe>, pipe>(mat_a, mat_b, mat_c, bias,
                                                                                  tiling_data, ws, extra);
  else if (ta && !tb)
    kernel_mm_fuse_operator<half, true, false, KernelFuseNone<half, pipe>, pipe>(mat_a, mat_b, mat_c, bias, tiling_data,
                                                                                 ws, extra);
  else if (!ta && tb)
    kernel_mm_fuse_operator<half, false, true, KernelFuseNone<half, pipe>, pipe>(mat_a, mat_b, mat_c, bias, tiling_data,
                                                                                 ws, extra);
  else if (ta && tb)
    kernel_mm_fuse_operator<half, true, true, KernelFuseNone<half, pipe>, pipe>(mat_a, mat_b, mat_c, bias, tiling_data,
                                                                                ws, extra);
}

void MatmulMix(bool ta, bool tb, void *mat_a, void *mat_b, void *mat_c, void *bias, void *tiling_data_d,
               void *tiling_data_h, void *ws, bool bmm, void *extra, void *stream, void *ctx) {
  ascend_native::SetContext(ctx);
  auto tiling = reinterpret_cast<TCubeTiling *>(tiling_data_h);
  int core_num = (bmm) ? CUBE_CORE_NUM : tiling->usedCoreNum;
  mmfuse_custom_mix<<<core_num, nullptr, stream>>>(
    ta, tb, reinterpret_cast<GM_ADDR>(mat_a), reinterpret_cast<GM_ADDR>(mat_b), reinterpret_cast<GM_ADDR>(mat_c),
    reinterpret_cast<GM_ADDR>(bias), reinterpret_cast<GM_ADDR>(tiling_data_d), reinterpret_cast<GM_ADDR>(ws),
    reinterpret_cast<GM_ADDR>(extra));
}
}  // namespace mindspore::ascend_native
