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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_INSERT_QUANT_NODE_MANAGER_H
#define MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_INSERT_QUANT_NODE_MANAGER_H
#include <vector>
#include <set>
#include <string>
#include "include/errorcode.h"
#include "ir/anf.h"
#include "ir/dtype/type_id.h"
#include "ir/func_graph.h"
#include "tools/converter/quantizer/quantize_util.h"
#include "ops/dynamic_quant.h"

namespace mindspore::lite::quant {
class InsertQuantNodeManager {
 public:
  InsertQuantNodeManager() = default;

  ~InsertQuantNodeManager() = default;

  int InsertQuantDtypeCastNode(const FuncGraphPtr &graph);

  int InsertDynamicQuantNode(const FuncGraphPtr &graph, const std::set<PrimitivePtr> &support_dynamic_quant_ops,
                             const std::set<std::string> &skip_quant_node);

 private:
  ValueNodePtr NewQuantCastValueNode(int src_type, int dst_type, const std::vector<schema::QuantParamT> &quant_params);

  int InsertCastNode(const FuncGraphPtr &graph, const CNodePtr &cnode, size_t input_index, bool is_graph_input);

  int CheckDataType(const AnfNodePtr &input_node, TypeId check_type_id);

  int NewDynamicQuantNode(const FuncGraphPtr &graph, const CNodePtr &cnode);

  int MarkDynamicQuantize(const CNodePtr &cnode);

  int InsertDynamicQuantWithIndex(const FuncGraphPtr &graph, const CNodePtr &cnode, size_t index);

 private:
  TypeId dst_type_ = kNumberTypeInt8;
  bool symmetric_ = true;
};
}  // namespace mindspore::lite::quant
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_INSERT_QUANT_NODE_MANAGER_H
