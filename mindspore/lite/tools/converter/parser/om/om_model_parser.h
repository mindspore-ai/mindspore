/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_OM_OM_MODEL_PARSER_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_OM_OM_MODEL_PARSER_H_

#include <map>
#include <string>
#include <vector>
#include "ir/anf.h"
#include "ir/func_graph.h"
#include "securec/include/securec.h"
#include "include/registry/model_parser.h"
#include "include/registry/model_parser_registry.h"
#include "ops/custom.h"

namespace mindspore {
namespace lite {
struct InputInfo {
  std::string name = "OMCustomInput";
  std::vector<int64_t> shape = {};
  TypeId data_type = kNumberTypeFloat32;
};

struct CustomOutputInfo {
  std::string name = "CustomOutput";
  std::vector<int64_t> shape = {};
  TypeId data_type = kNumberTypeFloat32;
};

class OMModelParser : public converter::ModelParser {
 public:
  OMModelParser() = default;
  ~OMModelParser() override = default;

  api::FuncGraphPtr Parse(const converter::ConverterParameters &flag) override;

 private:
  bool CreateGraphInputs(const FuncGraphPtr &func_graph);
  bool CreateGraphOutputs(const FuncGraphPtr &func_graph);
  CNodePtr CreateMakeTupleGraphOutput(const FuncGraphPtr &func_graph, const CNodePtr &custom_node);
  bool SetCustomOutputs(const CNodePtr &custom_node);
  bool SetMultiOutputs(const CNodePtr &custom_node);

  bool ParseInputsAndOutputsInfo(const converter::ConverterParameters &flag);
  std::vector<std::string> ParseNames(const std::map<std::string, std::string> &attrs,
                                      const std::string &names_section);
  std::vector<std::vector<int64_t>> ParseShapes(const std::map<std::string, std::string> &attrs,
                                                const std::string &shapes_section);
  std::vector<TypeId> ParseDataTypes(const std::map<std::string, std::string> &attrs,
                                     const std::string &data_types_section);

 private:
  std::vector<InputInfo> inputs_info_;
  std::vector<CustomOutputInfo> custom_outputs_info_;
};
}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_OM_OM_MODEL_PARSER_H_
