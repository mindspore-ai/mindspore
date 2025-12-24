/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPS_BROADCAST_H_
#define MINDSPORE_CORE_OPS_BROADCAST_H_
#include <memory>
#include <string>
#include <vector>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameBroadcast = "Broadcast";
/// \brief Broadcasts the tensor to the whole group. Refer to Python API @ref mindspore.ops.Broadcast for more details.
class OPS_API Broadcast : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(Broadcast);
  /// \brief Constructor.
  Broadcast() : BaseOperator(kNameBroadcast) {}
};
using PrimBroadcast = std::shared_ptr<Broadcast>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_BROADCAST_H_
