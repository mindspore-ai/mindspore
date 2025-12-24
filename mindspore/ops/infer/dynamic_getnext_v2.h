
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

#ifndef MINDSPORE_CORE_OPS_DYNAMIC_GETNEXT_V2_H_
#define MINDSPORE_CORE_OPS_DYNAMIC_GETNEXT_V2_H_
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameDynamicGetNextV2 = "DynamicGetNextV2";
/// \brief Returns the next element in the dataset queue.
/// Refer to Python API @ref mindspore.ops.DynamicGetNextV2 for more details.
class OPS_API DynamicGetNextV2 : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(DynamicGetNextV2);
  /// \brief Constructor.
  DynamicGetNextV2() : BaseOperator(kNameDynamicGetNextV2) {}
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.DynamicGetNextV2 for the inputs.
  void Init() const {}
};
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_DYNAMIC_GETNEXT_V2_H_
