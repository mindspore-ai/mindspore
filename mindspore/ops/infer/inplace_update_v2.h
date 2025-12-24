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

#ifndef MINDSPORE_CORE_OPS_INPLACE_UPDATE_V2_H_
#define MINDSPORE_CORE_OPS_INPLACE_UPDATE_V2_H_
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameInplaceUpdateV2 = "InplaceUpdateV2";
/// \brief InplaceUpdate operation. Refer to Python API @ref mindspore.ops.InplaceUpdateV2 for more details.
class OPS_API InplaceUpdateV2 : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(InplaceUpdateV2);
  /// \brief Constructor.
  InplaceUpdateV2() : BaseOperator(kNameInplaceUpdateV2) { InitIOName({"x", "indices", "v"}, {"y"}); }
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.InplaceUpdateV2 for the inputs.
  void Init() const {}
};

OPS_API abstract::AbstractBasePtr InplaceUpdateV2Infer(const abstract::AnalysisEnginePtr &,
                                                       const PrimitivePtr &primitive,
                                                       const std::vector<abstract::AbstractBasePtr> &input_args);
using PrimInplaceUpdateV2Ptr = std::shared_ptr<InplaceUpdateV2>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_INPLACE_UPDATE_V2_H_
