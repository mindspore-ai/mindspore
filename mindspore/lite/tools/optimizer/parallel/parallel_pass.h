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

#include <memory>
#include <utility>
#include <set>
#include <string>
#include <unordered_map>
#include "ir/anf.h"
#include "tools/optimizer/common/node_pass_extends.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "tools/optimizer/parallel/split_strategy.h"
#include "tools/optimizer/parallel/operator_info.h"

#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_PARALLEL_PARALLEL_PASS_H_
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_PARALLEL_PARALLEL_PASS_H_

namespace mindspore {
namespace opt {
class ParallelPass : public opt::LiteNodePass {
 public:
  explicit ParallelPass(const std::unordered_map<std::string, SplitStrategy> &strategys, const int32_t fmk_type)
      : LiteNodePass("parallel_pass"), split_strategys_(strategys), fmk_type_(fmk_type) {}
  ~ParallelPass() override = default;
  AnfNodePtr Run(const FuncGraphPtr &func_graph, const AnfNodePtr &node) override;

 private:
  // to check this node whether support to parallel && split
  bool IsParallelCareNode(const AnfNodePtr &node);

  // set curr_node a new op_name with parallel symbol
  bool SetParallelOpName(const AnfNodePtr &node, std::string *parallel_name);

  // create a parallel operator from different scope_name
  OperatorInfoPtr CreateParallelOperator(const CNodePtr &cnode, const std::string &scope_name,
                                         const std::string &parallel_op_name);

 private:
  bool is_depth_wise_{false};
  std::string type_name_;
  std::unordered_map<std::string, SplitStrategy> split_strategys_;
  int32_t fmk_type_{};
};

}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_PARALLEL_PARALLEL_PASS_H_
