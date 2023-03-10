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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_FORMAT_TYPE_CONVERT_CAST_FORMAT_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_FORMAT_TYPE_CONVERT_CAST_FORMAT_H_

#include <string>
#include <utility>
#include <memory>
#include <vector>

#include "utils/hash_map.h"
#include "include/backend/optimizer/optimizer.h"

namespace mindspore {
namespace opt {
class ConvertCastFormat : public PatternProcessPass {
 public:
  explicit ConvertCastFormat(bool multigraph = true) : PatternProcessPass("convert_cast_format", multigraph) {}
  ~ConvertCastFormat() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node, const EquivPtr &) const override;

 private:
  mindspore::HashMap<string, size_t> CalculateFormat(
    const std::shared_ptr<std::vector<std::pair<AnfNodePtr, int>>> &used_cast_node_list,
    const CNodePtr &cast_node) const;
  void ChangeCastFormat(const CNodePtr &cast_node, const FuncGraphPtr &func_graph) const;
  void SetCastFormat(const CNodePtr &cast_node, const string &format) const;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_FORMAT_TYPE_
