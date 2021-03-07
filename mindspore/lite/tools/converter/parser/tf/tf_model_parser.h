/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_TF_MODEL_PARSER_H
#define MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_TF_MODEL_PARSER_H

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <utility>
#include "proto/graph.pb.h"
#include "proto/node_def.pb.h"
#include "schema/inner/model_generated.h"
#include "securec/include/securec.h"
#include "tools/common/tensor_util.h"
#include "tools/converter/model_parser.h"
#include "src/param_value_lite.h"

namespace mindspore {
namespace lite {
class TFModelParser : public ModelParser {
 public:
  TFModelParser() = default;
  ~TFModelParser() = default;

  FuncGraphPtr Parse(const std::string &modelFile, const std::string &weightFile, const QuantType &quantType);

 private:
  static STATUS ConvertConstVariant(const tensorflow::TensorProto &tensor_proto, const ParamValueLitePtr &param_value);
  STATUS ConvertConstTensor(const tensorflow::NodeDef &node_def, const tensorflow::AttrValue &attr_value,
                            const TypeId &type, const ParameterPtr &parameter, std::vector<int64_t> *shape_vector);
  static STATUS GetValueFromType(const tensorflow::TensorProto &tensor_proto,
                                 const tensorflow::TensorShapeProto &tensor_shape, ParamValueLitePtr param_value,
                                 const TypeId &type, int shape_size);
  STATUS ConvertParameter(const tensorflow::NodeDef &node, const ParameterPtr &parameter,
                          std::unordered_map<std::string, AnfNodePtr> *anf_node_map);
  STATUS ConvertGraphInputsAndConsts(const std::map<std::string, const tensorflow::NodeDef *> &tf_graph_nodes,
                                     const FuncGraphPtr &anf_graph,
                                     std::unordered_map<std::string, AnfNodePtr> *anf_node_map);
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
  STATUS ConvertRootGraphOutputs();

  STATUS ConvertSubgraph();

  STATUS ConvertSubgraphInputs(std::map<std::string, const tensorflow::NodeDef *> *tf_sub_node_map,
                               std::unordered_map<std::string, AnfNodePtr> *anf_sub_node_map,
                               const tensorflow::FunctionDef &tf_sub_fuction, CNodePtr cnode,
                               FuncGraphPtr sub_func_graph);

  static STATUS ConvertSubgraphOutputs(std::map<std::string, const tensorflow::NodeDef *> *tf_sub_node_map,
                                       const std::unordered_map<std::string, AnfNodePtr> &anf_sub_node_map,
                                       const tensorflow::FunctionDef &tf_sub_fuction, FuncGraphPtr sub_func_graph);

  STATUS ControlFlowNodePostProcess(const std::map<CNodePtr, FuncGraphPtr> &first_func_map,
                                    const std::map<CNodePtr, FuncGraphPtr> &second_func_map);

  static STATUS MakeAnfGraphOutputs(std::vector<AnfNodePtr> *output_nodes, const FuncGraphPtr &anf_graph);

  STATUS RecordNullInput(const CNodePtr &node, const std::vector<std::string> &input_name_not_found);

  STATUS ConnectNullInput();

  FuncGraphPtr anf_root_graph_;
  std::unique_ptr<tensorflow::GraphDef> tf_root_graph_;                     // tf root graph def
  std::map<std::string, const tensorflow::NodeDef *> tf_root_graph_nodes_;  // tf root graph node map
  std::unordered_map<std::string, AnfNodePtr> anf_root_node_map_;
  std::vector<std::string> graph_input_names_;
  std::vector<std::string> graph_output_names_;
  std::map<std::string, AnfNodePtr> function_while_map_;  // tf function name->while_node_name
  std::map<std::string, AnfNodePtr> function_if_map_;     // tf function name->if_node
  std::vector<std::pair<CNodePtr, std::vector<std::string>>> nodes_with_null_input_{};
};
}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_TF_MODEL_PARSER_H
