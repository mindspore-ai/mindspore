/* copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPS_ACCUMULATENV2_H_
#define MINDSPORE_CORE_OPS_ACCUMULATENV2_H_
#include <memory>
#include <vector>

#include "abstract/abstract_value.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameAccumulateNV2 = "AccumulateNV2";
class MS_CORE_API AccumulateNV2 : public PrimitiveC {
 public:
  AccumulateNV2() : PrimitiveC(kNameAccumulateNV2) { InitIOName({"inputs"}, {"sum"}); }
  ~AccumulateNV2() = default;
  MS_DECLARE_PARENT(AccumulateNV2, PrimitiveC);
};
AbstractBasePtr AccumulateNV2Infer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args);
using PrimAccumulateNV2Ptr = std::shared_ptr<AccumulateNV2>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_ACCUMULATENV2_H_
