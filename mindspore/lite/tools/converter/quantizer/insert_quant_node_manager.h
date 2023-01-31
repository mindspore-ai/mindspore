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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_INSERT_QUANT_NODE_MANAGER_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_INSERT_QUANT_NODE_MANAGER_H_
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

  int InsertDynamicQuantNode(const FuncGraphPtr &graph, const std::set<PrimitivePtr> &support_dynamic_quant_ops,
                             const std::set<std::string> &skip_quant_node);

  int InsertDequantNode(const FuncGraphPtr &graph);

  int InsertForwardCastNode(const FuncGraphPtr &graph, const CNodePtr &cnode, TypeId cast_dtype,
                            quant::QuantType curr_quant_type);

  int InsertCastNodeForFullQuant(const FuncGraphPtr &graph, const CNodePtr &cnode, TypeId cast_dtype,
                                 quant::QuantType curr_quant_type);

  int InsertBackwardCastNode(const FuncGraphPtr &graph, const CNodePtr &cnode, TypeId cast_dtype,
                             quant::QuantType curr_quant_type);

  int InsertQuantDtypeCastFlyNode(const FuncGraphPtr &func_graph, const CNodePtr &cnode, size_t input_index,
                                  TypeId src_dtype, TypeId dst_dtype, int axis);

  int InsertFSEDecodeNode(const FuncGraphPtr &func_graph, const CNodePtr &cnode, size_t input_index, TypeId dst_dtype);

  int InsertAscendQuantNode(const FuncGraphPtr &func_graph, const CNodePtr &cnode);

  int InsertAscendDeQuantNode(const FuncGraphPtr &func_graph, const CNodePtr &cnode);

 private:
  int CheckDataType(const AnfNodePtr &input_node, TypeId check_type_id) const;

  int NewDynamicQuantNode(const FuncGraphPtr &graph, const CNodePtr &cnode);

  int MarkDynamicQuantize(const CNodePtr &cnode);

  int InsertDynamicQuantWithIndex(const FuncGraphPtr &graph, const CNodePtr &cnode, size_t index);

  int SetCastNodeAbstract(const CNodePtr &cnode, const AnfNodePtr &input_node, const CNodePtr &cast_cnode);

  int InsertForwardQuantNode(const FuncGraphPtr &graph, const CNodePtr &cnode, TypeId cast_dtype, size_t index,
                             CastNodeType cast_node_type);

  int InsertBackwardDeQuantNode(const FuncGraphPtr &graph, const CNodePtr &cnode, TypeId cast_dtype, size_t index,
                                const AnfNodePtr &output_node);

  int InsertQuantDtypeCastNode(const FuncGraphPtr &graph, const CNodePtr &cnode, InsertDirection insert_direction,
                               TypeId cast_dtype, CastNodeType cast_node_type, size_t index,
                               const AnfNodePtr &output_node);

  int CreateFSEInputs(const FuncGraphPtr &func_graph, const AnfNodePtr &input_node, std::vector<AnfNodePtr> *op_inputs,
                      TypeId dst_dtype);

  ValueNodePtr NewQuantCastPrimitive(int src_type, int dst_type,
                                     const std::vector<schema::QuantParamT> &input_quant_params,
                                     const std::vector<schema::QuantParamT> &output_quant_params, int axis = 0,
                                     bool set_quant_flag = true);

  ValueNodePtr NewFSEDecodePrimitive(int dst_type, uint64_t curr_chunk, int64_t curr_chunk_index,
                                     int64_t curr_bit_count, int64_t table_log);

 private:
  TypeId dst_type_ = kNumberTypeInt8;
  bool symmetric_ = false;
};
}  // namespace mindspore::lite::quant
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_INSERT_QUANT_NODE_MANAGER_H_
