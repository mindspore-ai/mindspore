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
#ifndef MINDSPORE_OPS_INCLUDE_VIEW_NARROW_STRIDES_EXT_CALC_H
#define MINDSPORE_OPS_INCLUDE_VIEW_NARROW_STRIDES_EXT_CALC_H

#include <vector>
#include "view/view_strides_calculator.h"

namespace mindspore {
namespace ops {
OPS_API TensorStorageInfoPtrList NarrowStridesCalc(const std::vector<int64_t> &cur_shape,
                                                   const std::vector<int64_t> &cur_strides,
                                                   const TensorStorageInfoPtr &cur_storage_info, const int64_t &dim,
                                                   const int64_t &start, const int64_t &length);
OPS_API TensorStorageInfoPtrList NarrowBasicTypeCalc(const mindspore::tensor::TensorPtr &input_tensor,
                                                     const int64_t &dim, const int64_t &start, const int64_t &length);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_OPS_INCLUDE_VIEW_NARROW_STRIDES_EXT_CALC_H
