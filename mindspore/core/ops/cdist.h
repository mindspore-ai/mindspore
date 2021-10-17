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

#ifndef MINDSPORE_CORE_OPS_CDIST_H_
#define MINDSPORE_CORE_OPS_CDIST_H_
#include <memory>
#include <vector>
#include <algorithm>
#include <set>
#include <map>
#include <string>
#include "ops/primitive_c.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameCdist = "Cdist";
/// \brief Computes batched the p norm distance between each pair of the two collections of row vectors.
/// Refer to Python API @ref mindspore.ops.Cdist for more details.
class MS_CORE_API Cdist : public PrimitiveC {
 public:
  /// \brief Constructor.
  Cdist() : PrimitiveC(kNameCdist) { InitIOName({"input_x", "input_y"}, {"output"}); }
  /// \brief Destructor.
  ~Cdist() = default;
  MS_DECLARE_PARENT(Cdist, PrimitiveC);
};

AbstractBasePtr CdistInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                           const std::vector<AbstractBasePtr> &input_args);
using PrimCdistPtr = std::shared_ptr<Cdist>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_CDIST_H_
