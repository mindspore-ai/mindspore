/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_SPLIT_WITH_SIZE_OP_PASS_H_
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_SPLIT_WITH_SIZE_OP_PASS_H_
#include <set>
#include <string>
#include "include/registry/converter_context.h"
#include "include/backend/optimizer/pass.h"
#include "tools/optimizer/common/gllo_utils.h"

namespace mindspore::opt {
class SplitWithSizeOpPass : public Pass {
 public:
  SplitWithSizeOpPass() : Pass("split_with_size_op_pass") {}
  ~SplitWithSizeOpPass() override = default;
  bool Run(const FuncGraphPtr &func_graph) override;

 private:
  AnfNodePtr SplitWithSizeMapperToSplitV(const FuncGraphPtr &func_graph, const CNodePtr &cnode);
  STATUS RunSplitWithSizePass(const FuncGraphPtr &func_graph, const FuncGraphManagerPtr &manager);

 protected:
  const std::string kAttrNameNumSplit = "num_split";
  const std::string kAttrNameSizeSplits = "size_splits";
  const std::string kAttrNameSplitDim = "split_dim";
};
}  // namespace mindspore::opt
#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_SPLIT_WITH_SIZE_OP_PASS_H_
