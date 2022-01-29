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

#ifndef MINDSPORE_CORE_OPS_BN_TRAINING_UPDATE_H_
#define MINDSPORE_CORE_OPS_BN_TRAINING_UPDATE_H_

#include <map>
#include <memory>
#include <vector>
#include <string>

#include "ops/primitive_c.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameBNTrainingUpdate = "BNTrainingUpdate";
class MS_CORE_API BNTrainingUpdate : public PrimitiveC {
 public:
  BNTrainingUpdate() : PrimitiveC(kNameBNTrainingUpdate) {
    InitIOName({"x", "sum", "square_sum", "scale", "b", "mean", "variance"},
               {"y", "running_mean", "running_variance", "save_mean", "save_inv_variance"});
  }
  ~BNTrainingUpdate() = default;
  MS_DECLARE_PARENT(BNTrainingUpdate, PrimitiveC);
};

AbstractBasePtr BNTrainingUpdateInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                      const std::vector<AbstractBasePtr> &input_args);

using kPrimBNTrainingUpdatePtr = std::shared_ptr<BNTrainingUpdate>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_BN_TRAINING_UPDATE_H_
