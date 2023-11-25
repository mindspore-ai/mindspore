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
#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_OPTIMIZER_GE_ADD_CAST_FOR_GE_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_OPTIMIZER_GE_ADD_CAST_FOR_GE_H_

#include <utility>
#include <string>
#include <vector>
#include <map>
#include "include/backend/optimizer/optimizer.h"
#include "ops/auto_generate/gen_ops_primitive.h"

namespace mindspore {
namespace opt {
class AddCastForGe : public PatternProcessPass {
 public:
  explicit AddCastForGe(bool multi_graph = true) : PatternProcessPass("add_cast_for_ge", multi_graph) {}
  ~AddCastForGe() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &graph, const AnfNodePtr &node, const EquivPtr &) const override;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_OPTIMIZER_GE_ADD_CAST_FOR_GE_H_
