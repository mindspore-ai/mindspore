/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_CONST_FOLD_FOLD_WITH_INFERSHAPE_H_
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_CONST_FOLD_FOLD_WITH_INFERSHAPE_H_

#include <set>
#include <utility>
#include <memory>
#include "include/backend/optimizer/pass.h"
#include "include/registry/converter_context.h"
#include "tools/optimizer/graph/node_infershape.h"
#include "tools/optimizer/const_fold/fold_utils.h"

namespace mindspore {
namespace opt {
class ConstFoldWithInferShape : public Pass {
 public:
  explicit ConstFoldWithInferShape(converter::FmkType fmk_type = converter::kFmkTypeMs, bool train_flag = false)
      : Pass("ConstFoldWithInferShape"), fmk_type_(fmk_type), train_flag_(train_flag) {}
  ~ConstFoldWithInferShape() override = default;
  bool Run(const FuncGraphPtr &func_graph) override;

 private:
  int HandleCommonFold(const FuncGraphPtr &func_graph, std::set<FuncGraphPtr> *has_visited);
  bool CheckCanCommonFold(const CNodePtr &cnode) const;
  int HandleSpecialFold(const FuncGraphPtr &func_graph);
  bool CheckCanSpecialFold(const CNodePtr &cnode) const;
  converter::FmkType fmk_type_{converter::kFmkTypeMs};
  bool train_flag_{false};
  std::shared_ptr<ConstFoldProcessor> const_fold_processor_{nullptr};
  std::shared_ptr<NodeInferShape> node_infershape_{nullptr};
  FuncGraphManagerPtr manager_{nullptr};
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_CONST_FOLD_FOLD_WITH_INFERSHAPE_H_
