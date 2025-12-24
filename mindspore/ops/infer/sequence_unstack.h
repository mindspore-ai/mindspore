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

#ifndef MINDSPORE_CORE_OPS_SEQUENCE_UNSTACK_H_
#define MINDSPORE_CORE_OPS_SEQUENCE_UNSTACK_H_

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameSequenceUnstack = "SequenceUnstack";

class OPS_API SequenceUnstack : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(SequenceUnstack);
  /// \brief Constructor.
  SequenceUnstack() : BaseOperator(kNameSequenceUnstack) {}
  void Init(const int64_t axis = 0);
  void set_axis(const int64_t axis);
  int64_t get_axis() const;
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_SEQUENCE_UNSTACK_H_
