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
#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_OPTIMIZER_ADD_CAST_TO_NODE_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_OPTIMIZER_ADD_CAST_TO_NODE_H_

#include <utility>
#include <string>
#include <vector>
#include <map>
#include "include/backend/optimizer/optimizer.h"
#include "ops/auto_generate/gen_ops_primitive.h"

namespace mindspore {
namespace opt {
class AddCastToNode : public PatternProcessPass {
 public:
  explicit AddCastToNode(bool multi_graph = true) : PatternProcessPass("add_cast_to_node", multi_graph) {}
  ~AddCastToNode() override = default;
  const AnfNodePtr Process(const FuncGraphPtr &graph, const AnfNodePtr &node, const EquivPtr &) const override;

 private:
  const std::vector<TypeId> all_type = {kNumberTypeUInt8,  kNumberTypeUInt16, kNumberTypeUInt32,
                                        kNumberTypeUInt64, kNumberTypeInt8,   kNumberTypeInt16,
                                        kNumberTypeInt32,  kNumberTypeInt64,  kNumberTypeBool};
  std::map<std::string, std::pair<std::vector<TypeId>, TypeId>> prim_type_map = {
    {prim::kPrimReciprocal->name(), {all_type, kNumberTypeFloat32}}};
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_OPTIMIZER_ADD_CAST_TO_NODE_H_
