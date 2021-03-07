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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_ONNX_MODEL_PARSER_H
#define MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_ONNX_MODEL_PARSER_H

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <fcntl.h>
#include <unistd.h>
#include <string>
#include <vector>
#include <memory>
#include <set>
#include <unordered_map>
#include "securec/include/securec.h"
#include "tools/converter/model_parser.h"
#include "tools/converter/parser/onnx/onnx_node_parser_registry.h"
#include "proto/onnx.pb.h"
#include "src/param_value_lite.h"

namespace mindspore {
namespace lite {
class OnnxModelParser : public ModelParser {
 public:
  OnnxModelParser() = default;

  ~OnnxModelParser() override = default;

  FuncGraphPtr Parse(const std::string &model_file, const std::string &weight_file,
                     const QuantType &quant_type) override;
  static TypeId GetDataTypeFromOnnx(onnx::TensorProto_DataType onnx_type);
  static STATUS CopyOnnxTensorData(const onnx::TensorProto &onnx_const_value,
                                   const ParamValueLitePtr &param_value_lite);

 private:
  STATUS InitOriginModel(const std::string &model_file);
  STATUS ConvertNodes(const onnx::GraphProto &onnx_graph, const FuncGraphPtr &func_graph_ptr,
                      std::unordered_map<std::string, AnfNodePtr> *anf_nodes_map, std::vector<AnfNodePtr> *graph_inputs,
                      const std::string &root_node_name);
  STATUS ConvertOnnxGraph(const onnx::GraphProto &onnx_graph, const FuncGraphPtr &func_graph_ptr,
                          std::unordered_map<std::string, AnfNodePtr> *anf_nodes_map,
                          std::vector<AnfNodePtr> *graph_inputs, const std::string &root_node_name);
  static STATUS ConvertConstTensors(const onnx::GraphProto &onnx_graph, const FuncGraphPtr &func_graph_ptr,
                                    std::unordered_map<std::string, AnfNodePtr> *anf_nodes_map);
  static STATUS ConvertGraphInputs(const onnx::GraphProto &onnx_graph, const FuncGraphPtr &func_graph_ptr,
                                   std::unordered_map<std::string, AnfNodePtr> *nodes_map);
  static STATUS ConvertGraphOutputs(const onnx::GraphProto &onnx_graph, const FuncGraphPtr &func_graph_ptr,
                                    const std::unordered_map<std::string, AnfNodePtr> &anf_nodes_map);
  static STATUS BuildReturnNode(const FuncGraphPtr &func_graph_ptr, const std::vector<AnfNodePtr> &return_inputs);
  static STATUS BuildParameterNode(const ParameterPtr &parameter_node, const onnx::TensorProto &tensor);
  STATUS BuildParameterNodeForQuantParam(const void *data, const std::string &name, TypeId type);
  STATUS BuildCNode(const onnx::NodeProto &onnx_node, const FuncGraphPtr &func_graph_ptr,
                    std::unordered_map<std::string, AnfNodePtr> *anf_nodes_map, std::vector<AnfNodePtr> *graph_inputs,
                    ops::PrimitiveC *primitive_c, std::string loop_name);
  static STATUS BuildOpOutputs(const onnx::NodeProto &onnx_node, const FuncGraphPtr &func_graph_ptr,
                               std::unordered_map<std::string, AnfNodePtr> *anf_nodes_map, const CNodePtr &cnode);
  static STATUS ConvertSpecialOnnxNode(const onnx::NodeProto &onnx_node, const FuncGraphPtr &func_graph_ptr,
                                       std::unordered_map<std::string, AnfNodePtr> *anf_nodes_map,
                                       ops::PrimitiveC *primitive_c);
  static STATUS ConvertOnnxGemmNode(const onnx::NodeProto &onnx_node, const FuncGraphPtr &func_graph_ptr,
                                    std::unordered_map<std::string, AnfNodePtr> *anf_nodes_map,
                                    ops::PrimitiveC *primitive_c);
  static STATUS BuildCNodeForGemm(const onnx::NodeProto &onnx_node, const FuncGraphPtr &func_graph_ptr,
                                  std::unordered_map<std::string, AnfNodePtr> *anf_nodes_map,
                                  ops::PrimitiveC *primitive_c, const std::string &name);
  STATUS ConvertOpQuantParams(const onnx::NodeProto &onnx_node, ops::PrimitiveC *primitive_c);
  STATUS ParseQuantParam(const onnx::NodeProto &onnx_node);
  STATUS SetTensorQuantParam(const std::string &tensor_name, std::vector<QuantParamT> *quant_params);
  STATUS SetTensorQuantParamFromNode(const std::string &tensor_name, std::vector<QuantParamT> *quant_params);
  STATUS CopyTensorQuantParam(const std::string &tensor_name, QuantParamT *quant_param, bool scale_or_not);
  static bool IsSpecialOnnxNode(const onnx::NodeProto &onnx_node);
  STATUS ConvertLoopOnnxNode(const onnx::NodeProto &onnx_node,
                             std::unordered_map<std::string, AnfNodePtr> *anf_nodes_map,
                             const std::string &root_node_name);
  STATUS ConvertIfOnnxNode(const onnx::NodeProto &onnx_node, std::unordered_map<std::string, AnfNodePtr> *anf_nodes_map,
                           const std::string &root_node_name);
  STATUS AddTensorArrayEdge(const FuncGraphPtr &anf_graph, std::vector<AnfNodePtr> *return_new_inputs,
                            const std::string &loop_node_name, std::vector<AnfNodePtr> *body_graph_inputs,
                            int act_output_num);
  STATUS AddTensorListStackNode(const AnfNodePtr &root_while_node, const onnx::NodeProto &onnx_node, int act_output_num,
                                int body_output_size);
  static STATUS BuildCondGraph(const FuncGraphPtr &cond_graph, const AnfNodePtr &root_while_node, int inputs_num,
                               const std::string &cond_graph_name);
  STATUS ConvertIfSubgraph(const onnx::GraphProto &onnx_graph, const FuncGraphPtr &anf_graph,
                           const std::string &subgrah_name, const std::string &if_node_name,
                           const std::string &root_node_name);
  onnx::ModelProto onnx_model_;
  onnx::GraphProto onnx_root_graph_;
  std::vector<FuncGraphPtr> all_subgraphs_;
  std::unordered_map<std::string, AnfNodePtr> anf_nodes_map_;
  std::unordered_map<std::string, std::unordered_map<std::string, AnfNodePtr> *> control_nodes_map_;
  std::unordered_map<std::string, std::string> child_root_map_;  // for nest control flow node
  FuncGraphPtr anf_root_graph_ = nullptr;
};
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_ONNX_MODEL_PARSER_H
