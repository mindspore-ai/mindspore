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

#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_ELIMINATE_REDUNDANT_CAST_PASS_H_
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_ELIMINATE_REDUNDANT_CAST_PASS_H_
#include <set>
#include "include/registry/converter_context.h"
#include "include/backend/optimizer/pass.h"

namespace mindspore::opt {
using mindspore::converter::FmkType;
class EliminateRedundantCastPass : public Pass {
 public:
  explicit EliminateRedundantCastPass(FmkType fmk_type = converter::kFmkTypeMs, bool train_flag = false)
      : Pass("eliminate_redundant_cast_pass"), fmk_type_(fmk_type), train_flag_(train_flag) {}
  ~EliminateRedundantCastPass() override = default;
  bool Run(const FuncGraphPtr &func_graph) override;

 protected:
  int RemoveCastOp(const AnfNodePtr &anf_node, const FuncGraphManagerPtr &manager);

 private:
  FmkType fmk_type_{converter::kFmkTypeMs};
  bool train_flag_{false};
  std::set<AnfNodePtr> remove_cnode_;
};
}  // namespace mindspore::opt
#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_ELIMINATE_REDUNDANT_CAST_PASS_H_
