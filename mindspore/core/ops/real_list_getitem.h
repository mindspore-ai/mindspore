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

#ifndef MINDSPORE_CORE_OPS_REAL_LIST_GETITEM_H_
#define MINDSPORE_CORE_OPS_REAL_LIST_GETITEM_H_
#include "ops/base_operator.h"
#include "mindspore/core/ops/core_ops.h"

namespace mindspore {
namespace ops {
/// \brief RealListGetItem op is used to get list[index] value, list is a dynamic length list or index is variable
class MIND_API RealListGetItem : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(RealListGetItem);
  /// \brief Constructor.
  RealListGetItem() : BaseOperator(prim::kRealListGetItem) { InitIOName({"input", "index"}, {"output"}); }
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_REAL_LIST_GETITEM_H_
