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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_ONNX_ONNX_MODEL_PARSER_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_ONNX_ONNX_MODEL_PARSER_H_

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <fcntl.h>
#include <string>
#include <vector>
#include <memory>
#include <set>
#include <unordered_map>
#include "securec/include/securec.h"
#include "include/registry/model_parser.h"
#include "include/registry/model_parser_registry.h"
#include "tools/converter/parser/onnx/onnx_node_parser_registry.h"
#include "proto/onnx.pb.h"
#include "schema/inner/model_generated.h"

namespace mindspore {
namespace lite {
class OnnxModelParser : public converter::ModelParser {
 public:
  OnnxModelParser() = default;

  ~OnnxModelParser() override = default;

  api::FuncGraphPtr Parse(const converter::ConverterParameters &flag) override;

 private:
  STATUS InitOriginModel(const std::string &model_file);
  std::vector<int> SortOnnxNodeIndex(const onnx::GraphProto &onnx_graph);
  STATUS ConvertNodes(const onnx::GraphProto &onnx_graph, const FuncGraphPtr &func_graph_ptr,
                      std::unordered_map<std::string, AnfNodePtr> *anf_nodes_map, std::vector<AnfNodePtr> *graph_inputs,
                      const std::string &root_node_name);
  STATUS ConvertOnnxGraph(const onnx::GraphProto &onnx_graph, const FuncGraphPtr &func_graph_ptr,
                          std::unordered_map<std::string, AnfNodePtr> *anf_nodes_map,
                          std::vector<AnfNodePtr> *graph_inputs, const std::string &root_node_name);
  STATUS BuildParameterNodeForQuantParam(const void *data, const std::string &name, TypeId type);
  STATUS BuildCNode(const onnx::NodeProto &onnx_node, const FuncGraphPtr &func_graph_ptr,
                    std::unordered_map<std::string, AnfNodePtr> *anf_nodes_map, std::vector<AnfNodePtr> *graph_inputs,
                    ops::PrimitiveCPtr primitive_c, std::string loop_name);
  STATUS ConvertOpQuantParams(const onnx::NodeProto &onnx_node, ops::PrimitiveCPtr primitive_c);
  STATUS ParseQuantParam(const onnx::NodeProto &onnx_node);
  STATUS SetTensorQuantParam(const std::string &tensor_name, std::vector<schema::QuantParamT> *quant_params);
  STATUS SetTensorQuantParamFromNode(const std::string &tensor_name, std::vector<schema::QuantParamT> *quant_params);
  STATUS CopyTensorQuantParam(const std::string &tensor_name, schema::QuantParamT *quant_param, bool scale_or_not);
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
  FuncGraphPtr BuildBodyGraph(const onnx::NodeProto &loop_node, const onnx::GraphProto &subgraph_proto,
                              int *cond_graph_input_num);
  STATUS ConvertIfSubgraph(const onnx::GraphProto &onnx_graph, const FuncGraphPtr &anf_graph,
                           const std::string &subgrah_name, const std::string &if_node_name,
                           const std::string &root_node_name);

  onnx::ModelProto onnx_model_{};
  onnx::GraphProto onnx_root_graph_{};
  std::vector<FuncGraphPtr> all_subgraphs_{};
  std::unordered_map<std::string, AnfNodePtr> anf_nodes_map_{};
  std::unordered_map<std::string, std::unordered_map<std::string, AnfNodePtr> *> control_nodes_map_{};
  std::unordered_map<std::string, std::string> child_root_map_{};  // for nest control flow node
  std::string model_file_{};
  bool has_subgraph_ = false;
};
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_ONNX_ONNX_MODEL_PARSER_H_
