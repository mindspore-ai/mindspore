/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "ops/sparse_gather_v2.h"
#include <set>
#include <memory>
#include <algorithm>
#include "ops/op_utils.h"
#include "mindapi/src/helper.h"
#include "utils/check_convert_utils.h"
#include "ops/gather.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(SparseGatherV2, BaseOperator);
REGISTER_HOST_DEPENDS(kNameSparseGatherV2, {2});
REGISTER_PRIMITIVE_EVAL_IMPL(SparseGatherV2, prim::kPrimSparseGatherV2, GatherInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
