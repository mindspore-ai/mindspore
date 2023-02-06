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

#ifndef MINDSPORE_CORE_OPS_LIST_SET_ITEM_H_
#define MINDSPORE_CORE_OPS_LIST_SET_ITEM_H_
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameListSetItem = "list_setitem";
/// \brief RealListSetItem op is used to set one item to the specific position in the list.
class MIND_API list_setitem : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(list_setitem);
  /// \brief Constructor.
  list_setitem() : BaseOperator(kNameListSetItem) {}
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_LIST_SET_ITEM_H_
