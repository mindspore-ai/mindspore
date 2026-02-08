/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_OPS_INCLUDE_VIEW_CHUNK_STRIDES_CALC_H
#define MINDSPORE_OPS_INCLUDE_VIEW_CHUNK_STRIDES_CALC_H

#include <vector>
#include "view/view_strides_calculator.h"

namespace mindspore {
namespace ops {
OPS_API TensorStorageInfoPtrList ChunkStridesCalc(const std::vector<int64_t> &old_shape,
                                                  const std::vector<int64_t> &old_strides,
                                                  const TensorStorageInfoPtr &storage_info, const int64_t &chunks,
                                                  const int64_t &dim);

OPS_API TensorStorageInfoPtrList ChunkBasicTypeCalc(const mindspore::tensor::TensorPtr &input_tensor,
                                                    const int64_t &chunks, const int64_t &dim);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_OPS_INCLUDE_VIEW_CHUNK_STRIDES_CALC_H
