/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_TF_TF_MODEL_PARSER_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_TF_TF_MODEL_PARSER_H_

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <set>
#include <utility>
#include "proto/graph.pb.h"
#include "proto/node_def.pb.h"
#include "schema/inner/model_generated.h"
#include "securec/include/securec.h"
#include "tools/common/tensor_util.h"
#include "include/registry/model_parser.h"
#include "include/registry/model_parser_registry.h"
#include "ops/primitive_c.h"

namespace mindspore {
namespace lite {
class TFModelParser : public converter::ModelParser {
 public:
  TFModelParser() = default;
  ~TFModelParser() override = default;

  api::FuncGraphPtr Parse(const converter::ConverterParameters &flag) override;

  static int TF2AnfAdjust(const std::set<FuncGraphPtr> &all_func_graphs);

 private:
  static STATUS ConvertConstVariant(const tensorflow::TensorProto &tensor_proto, tensor::TensorPtr *tensor_info);
  STATUS ConvertConstTensor(const tensorflow::NodeDef &node_def, const tensorflow::AttrValue &attr_value,
                            const TypeId &type, const ParameterPtr &parameter, std::vector<int64_t> *shape_vector);
  STATUS SetInt64TensorInfo(const tensorflow::TensorProto &tensor_proto, tensor::TensorPtr *tensor_info,
                            const std::string &node_name);
  STATUS SetInt64TensorToInt64Tensor(const tensorflow::TensorProto &tensor_proto, tensor::TensorPtr *tensor_info);
  STATUS SetTensorInfoFromType(const tensorflow::TensorProto &tensor_proto, tensor::TensorPtr *tensor_info,
                               const std::string &node_name);
  STATUS ConvertParameter(const tensorflow::NodeDef &node, const ParameterPtr &parameter,
                          std::unordered_map<std::string, AnfNodePtr> *anf_node_map, bool root_graph = false);
  STATUS ConvertGraphInputsAndConsts(const std::vector<const tensorflow::NodeDef *> &tf_graph_nodes,
                                     const FuncGraphPtr &anf_graph,
                                     std::unordered_map<std::string, AnfNodePtr> *anf_node_map,
                                     bool root_graph = false);
  static STATUS ConvertInputNodes(const tensorflow::NodeDef &node_def, const std::vector<std::string> &input_names,
                                  const std::map<std::string, const tensorflow::NodeDef *> &tf_node_map,
                                  const std::unordered_map<std::string, AnfNodePtr> &anf_node_map,
                                  std::vector<AnfNodePtr> *inputs, std::vector<std::string> *input_name_not_found);
  static STATUS ConvertOutputTensor(const tensorflow::NodeDef &op, const CNodePtr &anf_node,
                                    std::unordered_map<std::string, AnfNodePtr> *anf_node_map,
                                    const FuncGraphPtr &anf_graph, int output_size);
  STATUS ConvertOps(const tensorflow::NodeDef &node_def,
                    const std::map<std::string, const tensorflow::NodeDef *> &tf_node_map,
                    const FuncGraphPtr &func_graph_ptr, std::unordered_map<std::string, AnfNodePtr> *anf_node_map);
  STATUS ResetAbstractTensorToInt64(const std::string &op_type, const std::vector<std::string> &input_names,
                                    const std::map<std::string, const tensorflow::NodeDef *> &tf_node_map,
                                    const std::unordered_map<std::string, AnfNodePtr> &anf_node_map);
  STATUS ProcessControlFlowOp(const CNodePtr &anf_node, const string &op_type, const tensorflow::NodeDef &node_def);

  std::set<std::string> GetAllNodeInputs();

  STATUS GetGraphOutputNames(std::vector<AnfNodePtr> *output_nodes);

  STATUS ConvertRootGraphOutputs();

  void UpdateMap(const CNodePtr &cnode, const FuncGraphPtr &sub_func_graph, const std::string &sub_graph_name);

  STATUS ConvertSubgraph();

  STATUS ConvertSubgraphInputs(std::map<std::string, const tensorflow::NodeDef *> *tf_sub_node_map,
                               std::unordered_map<std::string, AnfNodePtr> *anf_sub_node_map,
                               const tensorflow::FunctionDef &tf_sub_fuction, const CNodePtr &cnode,
                               const FuncGraphPtr &sub_func_graph);

  static STATUS ConvertSubgraphOutputs(std::map<std::string, const tensorflow::NodeDef *> *tf_sub_node_map,
                                       const std::unordered_map<std::string, AnfNodePtr> &anf_sub_node_map,
                                       const tensorflow::FunctionDef &tf_sub_fuction,
                                       const FuncGraphPtr &sub_func_graph);

  STATUS ControlFlowNodePostProcess(const std::map<CNodePtr, FuncGraphPtr> &first_func_map,
                                    const std::map<CNodePtr, FuncGraphPtr> &second_func_map);

  static STATUS MakeAnfGraphOutputs(const std::vector<AnfNodePtr> &output_nodes, const FuncGraphPtr &anf_graph);

  STATUS RecordNullInput(const CNodePtr &node, const std::vector<std::string> &input_name_not_found);

  STATUS ConnectNullInput();

  std::unique_ptr<tensorflow::GraphDef> tf_root_graph_;                     // tf root graph def
  std::map<std::string, const tensorflow::NodeDef *> tf_root_graph_nodes_;  // tf root graph node map
  std::vector<const tensorflow::NodeDef *> tf_root_graph_nodes_vec_;
  std::unordered_map<std::string, AnfNodePtr> anf_root_node_map_;
  std::vector<std::string> graph_input_names_;
  std::vector<std::string> graph_output_names_;
  std::map<std::string, AnfNodePtr> function_while_map_;  // tf function name->while_node_name
  std::map<std::string, AnfNodePtr> function_if_map_;     // tf function name->if_node
  std::vector<std::pair<CNodePtr, std::vector<std::string>>> nodes_with_null_input_{};
  std::vector<std::string> while_cond_branch_name_;
  std::vector<std::string> if_then_branch_name_;
  std::unordered_map<std::string, int> node_output_num_;
  std::map<CNodePtr, FuncGraphPtr> while_cond_map_, while_body_map_, if_then_map_, if_else_map_;
};
}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_TF_TF_MODEL_PARSER_H_
