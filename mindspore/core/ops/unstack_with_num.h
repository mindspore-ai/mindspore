/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPS_UNSTACK_WITH_NUM_H_
#define MINDSPORE_CORE_OPS_UNSTACK_WITH_NUM_H_

#include <map>
#include <vector>
#include <string>
#include <memory>
#include <algorithm>

#include "ops/base_operator.h"
#include "mindapi/base/types.h"

namespace mindspore {
namespace ops {
constexpr auto kNameUnstackWithNum = "UnstackWithNum";
/// \brief UnstackWithNums tensor in specified axis with specified output number.
class MIND_API UnstackWithNum : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(UnstackWithNum);
  /// \brief Constructor.
  UnstackWithNum() : BaseOperator(kNameUnstackWithNum) { InitIOName({"x"}, {"y"}); }
  /// \brief Init.
  void Init(const int64_t num, const int64_t axis);
  /// \brief Set axis.
  void set_axis(const int64_t axis);
  /// \brief Get axis.
  ///
  /// \return axis.
  int64_t get_axis() const;
  /// \brief Set output num.
  void set_num(const int64_t num);
  /// \brief Get output num.
  ///
  /// \return output num.
  int64_t get_num() const;
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_UNSTACK_WITH_NUM_H_
