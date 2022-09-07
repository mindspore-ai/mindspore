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

#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_FISSON_ELIMINATE_CONCAT_SPLIT_H_
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_FISSON_ELIMINATE_CONCAT_SPLIT_H_

#include "tools/optimizer/common/pattern_process_pass_extends.h"
#include "tools/optimizer/fisson/fisson_util.h"

namespace mindspore {
namespace opt {
class EliminateConcatSplit : public LitePatternProcessPass {
 public:
  explicit EliminateConcatSplit(bool multigraph = true)
      : LitePatternProcessPass("eliminate_concat_split", multigraph) {}
  ~EliminateConcatSplit() override = default;

 private:
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_FISSON_ELIMINATE_CONCAT_SPLIT_H_
