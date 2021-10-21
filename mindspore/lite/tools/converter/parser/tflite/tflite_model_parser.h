/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_TFLITE_MODEL_PARSER_H
#define MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_TFLITE_MODEL_PARSER_H

#include <string>
#include <unordered_map>
#include <memory>
#include <vector>
#include <set>
#include <map>
#include <utility>
#include "include/registry/model_parser.h"
#include "include/registry/model_parser_registry.h"
#include "tools/converter/parser/tflite/tflite_node_parser_registry.h"
#include "tools/common/tensor_util.h"

namespace mindspore {
namespace lite {
class TfliteModelParser : public converter::ModelParser {
 public:
  TfliteModelParser() = default;

  ~TfliteModelParser() override {
    if (tflite_model_buf_ != nullptr) {
      delete[] tflite_model_buf_;
      tflite_model_buf_ = nullptr;
    }
  }

  api::FuncGraphPtr Parse(const converter::ConverterParameters &flag) override;

  static int Tflite2AnfAdjust(const std::set<FuncGraphPtr> &all_func_graphs);

 private:
  std::unique_ptr<tflite::ModelT> tflite_model_;
  std::map<int, CNodePtr> control_flow_nodes_;
  std::map<CNodePtr, std::pair<FuncGraphPtr, FuncGraphPtr>> control_flow_map_;
  char *tflite_model_buf_ = nullptr;
  std::unique_ptr<tflite::ModelT> ReadTfliteModel(const std::string &model_path);
  STATUS ConvertConstTensor(const std::unique_ptr<tflite::TensorT> &tensor, const ParameterPtr &parameter,
                            const std::string &tensor_name, bool is_uint8_weight_quant);
  STATUS ConvertOps(const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph, const FuncGraphPtr &func_graph,
                    std::unordered_map<int, AnfNodePtr> *anf_node_map);
  STATUS ConvertGraphInputs(const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph, const FuncGraphPtr &func_graph,
                            std::unordered_map<int, AnfNodePtr> *anf_node_map);
  STATUS ConvertGraphOutputs(const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph, const FuncGraphPtr &func_graph,
                             std::unordered_map<int, AnfNodePtr> *anf_node_map);
  STATUS ConvertTfliteGraph();
  STATUS ProcessControlFlowOp(const std::unique_ptr<tflite::OperatorT> &op, const CNodePtr &anf_node,
                              const std::string &op_type);
  STATUS BuildSubFuncGraphMap(size_t subgraph_idx, const FuncGraphPtr &sub_func_graph,
                              const std::string &subgraph_name);
  STATUS ControlFlowNodePostProcess();
  void ConvertInputTensor(const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph, const FuncGraphPtr &func_graph,
                          const std::unique_ptr<tflite::OperatorT> &op, tflite::BuiltinOperator tflite_op_type,
                          std::unordered_map<int, AnfNodePtr> *anf_node_map, std::string op_name,
                          std::vector<AnfNodePtr> *op_inputs);
  static STATUS ConvertOutputTensor(const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph,
                                    const FuncGraphPtr &func_graph, const std::unique_ptr<tflite::OperatorT> &op,
                                    const CNodePtr &dst_cnode, std::unordered_map<int, AnfNodePtr> *anf_node_map);
  static STATUS ConvertOpQuantParams(const std::unique_ptr<tflite::OperatorT> &op,
                                     const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph,
                                     ops::PrimitiveC *primitive_c);
  static STATUS SetTensorQuantParam(const std::unique_ptr<tflite::TensorT> &tflite_tensor,
                                    std::vector<QuantParamT> *quant_params, int round_type = 1);

  STATUS TfliteModelVerify();
};
}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_TFLITE_MODEL_PARSER_H
