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

#ifndef MINDSPORE_LITE_SRC_PASS_FUSION_CONV1D_WEIGHT_EXPANDING_PASS_H_
#define MINDSPORE_LITE_SRC_PASS_FUSION_CONV1D_WEIGHT_EXPANDING_PASS_H_
#include <string>
#include "schema/inner/model_generated.h"
#include "tools/converter/converter_flags.h"
#include "backend/optimizer/common/pass.h"
#include "tools/optimizer/common/gllo_utils.h"

namespace mindspore::opt {
class Conv1DWeightExpandingPass : public Pass {
 public:
  Conv1DWeightExpandingPass() : Pass("conv1d_weight_expanding_pass") {}
  ~Conv1DWeightExpandingPass() override = default;
  bool Run(const FuncGraphPtr &graph) override;

 private:
  lite::STATUS ExpandFilterShape(const AnfNodePtr &weight_node, const schema::Format &format);
};
}  // namespace mindspore::opt
#endif  // MINDSPORE_LITE_SRC_PASS_FUSION_CONV1D_WEIGHT_EXPANDING_PASS_H_
