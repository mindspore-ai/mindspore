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
#ifndef MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_NATIVE_ASCEND_NATIVE_IMPL_TILING_IF_H
#define MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_NATIVE_ASCEND_NATIVE_IMPL_TILING_IF_H
#include <vector>
#include "extendrt/delegate/ascend_native/ascend_native_impl/utils.h"
namespace mindspore::ascend_native {
int buildTiling(int blockDim, int M, int N, int K, bool transposeA, bool transposeB, bool isBias, void *tilingPtr,
                int size);
int tiling_size();
void buildMMExtra(mindspore::ascend_native::MMExtra *e, uint32_t bmm_num = 1, uint32_t lda = -1, uint32_t ldb = -1,
                  uint32_t ldc = -1);
}  // namespace mindspore::ascend_native
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_NATIVE_ASCEND_NATIVE_IMPL_TILING_IF_H
