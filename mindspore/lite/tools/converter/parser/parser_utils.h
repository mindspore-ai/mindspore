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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_PARSER_UTILS_H
#define MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_PARSER_UTILS_H

#include <set>
#include <vector>
#include "include/registry/model_parser.h"
#include "ir/anf.h"
#include "ir/func_graph.h"
#include "src/common/log_adapter.h"
#include "schema/inner/model_generated.h"

namespace mindspore {
namespace lite {
void GetAllFuncGraph(const FuncGraphPtr &func_graph, std::set<FuncGraphPtr> *all_func_graphs);
int CommonAnfAdjust(const std::set<FuncGraphPtr> &all_func_graphs);
int GetTransposePerm(schema::Format src_format, schema::Format dst_format, std::vector<int> *perm);
int GetTransposePermSharing(schema::Format src_format, schema::Format dst_format, std::vector<int> *perm);
AnfNodePtr GetRealConvWeightNode(const FuncGraphPtr &graph, const CNodePtr &cnode, size_t index);
int UnifyConvWeightFormat(const FuncGraphPtr &graph, const CNodePtr &cnode, schema::Format src_format,
                          schema::Format dst_format, std::set<AnfNodePtr> *has_visited);
int UnifyVariableConvWeight(const FuncGraphPtr &graph, const AnfNodePtr &weight_node, schema::Format src_format,
                            schema::Format dst_format, std::set<AnfNodePtr> *has_visited);
int UnifyConstConvWeight(const FuncGraphPtr &graph, const AnfNodePtr &weight_node, schema::Format src_format,
                         schema::Format dst_format, std::set<AnfNodePtr> *has_visited);
int HandleConstConvWeightShared(const FuncGraphPtr &graph, const AnfNodePtr &weight_node, schema::Format src_format,
                                schema::Format dst_format, std::set<AnfNodePtr> *has_visited);

template <class T>
converter::ModelParser *LiteModelParserCreator() {
  auto *parser = new (std::nothrow) T();
  if (parser == nullptr) {
    MS_LOG(ERROR) << "new model parser failed";
    return nullptr;
  }
  return parser;
}
}  // namespace lite
}  // namespace mindspore

#endif
