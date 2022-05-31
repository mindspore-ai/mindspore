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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_DPICO_PARSER_UNIFY_FORMAT_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_DPICO_PARSER_UNIFY_FORMAT_H_

#include <vector>
#include <memory>
#include <set>
#include <string>
#include <map>
#include "common/data_transpose_utils.h"
#include "common/anf_util.h"
#include "include/errorcode.h"

namespace mindspore {
namespace lite {
class UnifyFormatToNHWC {
 public:
  UnifyFormatToNHWC() = default;
  ~UnifyFormatToNHWC() = default;
  bool Run(const api::FuncGraphPtr &func_graph);

 private:
  STATUS InsertPostTransNode(const api::FuncGraphPtr &func_graph, const api::CNodePtr &cnode,
                             const std::vector<int> &perm);
  STATUS InsertPreTransNode(const api::FuncGraphPtr &func_graph, const api::CNodePtr &cnode,
                            const std::vector<int> &perm);
  STATUS GenNewInput(const api::FuncGraphPtr &func_graph, const api::CNodePtr &cnode, const std::vector<int> &perm,
                     bool before, size_t index = 0);
  bool BasicProcess(const api::FuncGraphPtr &func_graph, bool main_graph);
  void GetTransNodeFormatType(const api::CNodePtr &cnode, dpico::TransTypePair *trans_info);
  STATUS HandleGraphInput(const api::FuncGraphPtr &func_graph);
  STATUS HandleGraphNode(const api::FuncGraphPtr &func_graph, const api::CNodePtr &cnode);
  STATUS ConvWeightFormatTrans(const api::FuncGraphPtr &graph, std::set<api::AnfNodePtr> *has_visited);
  std::map<api::FuncGraphPtr, std::vector<api::AnfNodePtr>> sub_inputs_map_;
};
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_DPICO_PARSER_UNIFY_FORMAT_H_
