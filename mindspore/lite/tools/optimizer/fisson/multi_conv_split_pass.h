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

#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_FISSON_MULTI_CONV_SPLIT_PASS_H_
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_FISSON_MULTI_CONV_SPLIT_PASS_H_

#include <utility>
#include <vector>
#include <string>
#include <unordered_map>
#include "schema/model_generated.h"
#include "tools/optimizer/common/pattern_process_pass_extends.h"
#include "tools/optimizer/fisson/fisson_util.h"
#include "tools/optimizer/parallel/split_strategy.h"
#include "tools/optimizer/parallel/multi_node_split.h"

using mindspore::schema::PrimitiveType;
namespace mindspore {
namespace opt {

class MultiConvSplitPass : public LitePatternProcessPass {
 public:
  explicit MultiConvSplitPass(std::unordered_map<std::string, SplitStrategy> strategys, int32_t fmk_type = -1,
                              int32_t num = 3, bool multigraph = true)
      : LitePatternProcessPass("multi_conv_split", multigraph),
        strategys_(std::move(strategys)),
        fmk_type_(fmk_type),
        num_(num) {}
  ~MultiConvSplitPass() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;

 private:
  std::string IsMultiParallelConvNode(const AnfNodePtr &node) const;

  std::unordered_map<std::string, SplitStrategy> strategys_{};
  PrimitiveType primitive_type_{schema::PrimitiveType_NONE};
  int32_t fmk_type_{-1};
  int32_t num_{0};
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_FISSON_MULTI_CONV_SPLIT_PASS_H_
