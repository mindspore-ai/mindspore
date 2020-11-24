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

#include <string>
#include <vector>
#include <memory>
#include <map>
#include <unordered_map>
#include "securec/include/securec.h"
#include "tools/common/tensor_util.h"
#include "tools/converter/model_parser.h"
#include "schema/inner/model_generated.h"
#include "proto/node_def.pb.h"
#include "proto/graph.pb.h"

namespace mindspore {
namespace lite {
class TFModelParser : public ModelParser {
 public:
  TFModelParser() = default;
  ~TFModelParser() = default;

  FuncGraphPtr Parse(const std::string &modelFile, const std::string &weightFile, const QuantType &quantType);

 protected:
  schema::MetaGraphT *ParseToFb(const std::string &modelFile, const std::string &weightFile,
                                const QuantType &quantType = QuantType_QUANT_NONE) override;

 private:
  AnfNodePtr GetAnfNode(const std::string &name);
  std::string GetOriginInputName(const tensorflow::NodeDef &node);
  STATUS ConvertConstTensor(const tensorflow::AttrValue &attr_value, const TypeId &type, const ParameterPtr &parameter,
                            std::vector<int64_t> *shape_vector);
  STATUS ConvertParameter(const tensorflow::NodeDef &node, const ParameterPtr &parameter);
  STATUS ConvertGraphInputsAndConsts();
  STATUS ConvertInputNodes(const tensorflow::NodeDef &node_def, const std::vector<std::string> &input_names,
                           std::vector<AnfNodePtr> *inputs);
  STATUS ConvertOutputTensor(const tensorflow::NodeDef &op, const CNodePtr &anf_node, int output_size);
  STATUS ConvertOps();
  STATUS ConvertGraphOutputs();

  FuncGraphPtr funcGraphPtr;
  std::unique_ptr<tensorflow::GraphDef> tf_graph_def;
  std::map<std::string, const tensorflow::NodeDef *> tf_node_map;
  std::unordered_map<std::string, AnfNodePtr> anf_node_map;
  std::vector<std::string> graph_input_names;
  std::vector<std::string> graph_output_names;
};
}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_TF_MODEL_PARSER_H
