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

#include <map>
#include <string>
#include <vector>
#include <memory>
#include <algorithm>
#include "tools/converter/parser/om/om_model_parser.h"
#include "src/common/mmap_utils.h"
#include "src/common/log_util.h"
#include "tools/common/string_util.h"
#include "tools/common/tensor_util.h"
#include "tools/common/custom_ascend_utils.h"
#include "tools/converter/parser/lite_model_parser_creator.h"
#include "ops/return.h"
#include "ops/tuple_get_item.h"
#include "mindspore/core/ops/sequence_ops.h"

namespace mindspore {
namespace lite {
constexpr size_t kOneOutputs = 1;
api::FuncGraphPtr OMModelParser::Parse(const converter::ConverterParameters &flag) {
  auto om_file = flag.model_file;
  if (om_file.empty()) {
    MS_LOG(ERROR) << "OM file path is invalid.";
    return nullptr;
  }
  size_t len;
  auto buf = ReadFileByMmap(om_file, &len);
  if (buf == nullptr) {
    MS_LOG(ERROR) << "OM model buf is null";
    return nullptr;
  }
  if (!ParseInputsAndOutputsInfo(flag)) {
    MS_LOG(ERROR) << "Parse inputs and outputs info failed.";
    return nullptr;
  }
  Buffer om_data(buf, len);
  auto graph = std::make_shared<FuncGraph>();
  MS_CHECK_TRUE_MSG(graph != nullptr, nullptr, "Create FuncGraph failed.");
  if (!CreateGraphInputs(graph)) {
    MS_LOG(ERROR) << "Create Graph inputs failed.";
    return nullptr;
  }
  if (!CreateGraphOutputs(graph)) {
    MS_LOG(ERROR) << "Create Graph outputs failed.";
    return nullptr;
  }
  if (!CustomAscendUtils::CreateCustomFuncGraph(graph, om_data, "ACL_om_data", {}, {})) {
    MS_LOG(ERROR) << "Create Custom FuncGraph failed.";
    return nullptr;
  }
  auto res_graph = api::MakeShared<api::FuncGraph>(graph);
  return res_graph;
}

bool OMModelParser::CreateGraphInputs(const FuncGraphPtr &func_graph) {
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "func_graph is null.";
    return false;
  }
  for (auto &input_info : inputs_info_) {
    auto parameter = func_graph->add_parameter();
    if (parameter == nullptr) {
      MS_LOG(ERROR) << "New parameter failed.";
      return false;
    }
    auto shape_vector = input_info.shape;
    auto data_type = input_info.data_type;
    auto name = input_info.name;

    auto abstract_tensor = CreateTensorAbstract(shape_vector, data_type);
    if (abstract_tensor == nullptr) {
      MS_LOG(ERROR) << "Create tensor abstract failed.";
      return false;
    }
    parameter->set_abstract(abstract_tensor);
    parameter->set_name(name);
  }
  return true;
}

bool OMModelParser::CreateGraphOutputs(const FuncGraphPtr &func_graph) {
  MS_CHECK_TRUE_MSG(func_graph != nullptr, false, "Get FuncGraph failed.");
  auto prim = std::make_shared<ops::Custom>();
  MS_CHECK_TRUE_MSG(prim != nullptr, false, "New Custom op failed.");
  prim->set_type("ACL_Fake");
  auto prim_c = prim->GetPrim();
  auto graph_input = func_graph->get_inputs();
  MS_CHECK_TRUE_MSG(!graph_input.empty(), false, "Graph input is empty.");
  auto custom_node = func_graph->NewCNode(prim_c, graph_input);
  MS_CHECK_TRUE_MSG(custom_node != nullptr, false, "Create custom node failed.");
  custom_node->set_fullname_with_scope("custom_fake");

  if (!SetCustomOutputs(custom_node)) {
    MS_LOG(ERROR) << "Set custom node outputs failed.";
    return false;
  }
  auto return_prim = std::make_shared<ops::Return>();
  if (return_prim == nullptr) {
    MS_LOG(ERROR) << "New Return failed.";
    return false;
  }
  auto return_prim_c = return_prim->GetPrim();
  if (return_prim_c == nullptr) {
    MS_LOG(ERROR) << "return prim_c is null.";
    return false;
  }
  CNodePtr output_node = nullptr;
  if (custom_outputs_info_.size() == kOneOutputs) {
    output_node = custom_node;
  } else {
    output_node = CreateMakeTupleGraphOutput(func_graph, custom_node);
  }
  if (output_node == nullptr) {
    MS_LOG(ERROR) << "Create Graph output node failed.";
    return false;
  }
  auto return_cnode = func_graph->NewCNode(return_prim_c, {output_node});
  if (return_cnode == nullptr) {
    MS_LOG(ERROR) << "New return cnode failed.";
    return false;
  }
  auto abstract = output_node->abstract();
  if (abstract == nullptr) {
    MS_LOG(ERROR) << "Custom node abstract is null.";
    return false;
  }
  return_cnode->set_abstract(abstract);
  return_cnode->set_fullname_with_scope("Return");
  func_graph->set_return(return_cnode);
  return true;
}

bool OMModelParser::SetCustomOutputs(const CNodePtr &custom_node) {
  if (custom_node == nullptr) {
    MS_LOG(ERROR) << "custom node is null";
    return false;
  }
  if (custom_outputs_info_.size() == kOneOutputs) {
    CustomOutputInfo custom_output_info = custom_outputs_info_.at(0);
    auto name = custom_output_info.name;
    auto dims = custom_output_info.shape;
    auto type = custom_output_info.data_type;
    auto abstract_tensor = CreateTensorAbstract(dims, type);
    if (abstract_tensor == nullptr) {
      MS_LOG(ERROR) << "Abstract tensor is null.";
      return false;
    }
    abstract_tensor->set_name(name);
    custom_node->set_abstract(abstract_tensor);
  } else {
    if (!SetMultiOutputs(custom_node)) {
      MS_LOG(ERROR) << "Set Multi Graph output failed.";
      return false;
    }
  }
  return true;
}

CNodePtr OMModelParser::CreateMakeTupleGraphOutput(const FuncGraphPtr &func_graph, const CNodePtr &custom_node) {
  std::vector<CNodePtr> node_list;
  for (size_t i = 0; i < custom_outputs_info_.size(); ++i) {
    auto tuple_get_item_prim_ptr = std::make_shared<ops::TupleGetItem>();
    if (tuple_get_item_prim_ptr == nullptr) {
      MS_LOG(ERROR) << "New TupleGetItem failed for output" << custom_outputs_info_.at(i).name;
      return nullptr;
    }
    auto tuple_get_item_prim_ptr_c = tuple_get_item_prim_ptr->GetPrim();
    auto tuple_get_item_prim = NewValueNode(tuple_get_item_prim_ptr_c);
    MS_CHECK_TRUE_MSG(tuple_get_item_prim != nullptr, nullptr, "item_prim is nullptr.");
    auto get_item_value = NewValueNode(MakeValue<int64_t>(i));
    MS_CHECK_TRUE_MSG(get_item_value != nullptr, nullptr, "item_value is nullptr.");
    AnfNodePtrList inputs{tuple_get_item_prim, custom_node, get_item_value};
    CNodePtr get_item_cnode = func_graph->NewCNode(inputs);
    if (get_item_cnode == nullptr) {
      MS_LOG(ERROR) << "New get item cnode failed for output " << custom_outputs_info_.at(i).name;
      return nullptr;
    }
    get_item_cnode->set_fullname_with_scope(custom_node->fullname_with_scope() + "_getitem_" + std::to_string(i));
    node_list.emplace_back(get_item_cnode);
  }
  auto make_tuple_val_node = NewValueNode(prim::kPrimMakeTuple);
  MS_CHECK_TRUE_MSG(make_tuple_val_node != nullptr, nullptr, "New make tuple val node failed.");
  AnfNodePtrList new_inputs = {make_tuple_val_node};
  new_inputs.insert(new_inputs.end(), node_list.begin(), node_list.end());
  auto make_tuple_cnode = func_graph->NewCNode(new_inputs);
  MS_CHECK_TRUE_MSG(make_tuple_cnode != nullptr, nullptr, "New make tuple cnode failed.");
  make_tuple_cnode->set_fullname_with_scope("return tuple");
  if (custom_node->abstract() == nullptr) {
    MS_LOG(ERROR) << "custom_node->abstract() is null";
    return nullptr;
  }
  make_tuple_cnode->set_abstract(custom_node->abstract());
  return make_tuple_cnode;
}

bool OMModelParser::SetMultiOutputs(const CNodePtr &custom_node) {
  if (custom_node == nullptr) {
    MS_LOG(ERROR) << "Custom node is null.";
    return false;
  }
  AbstractBasePtrList abstract_list;
  for (auto &custom_output_info : custom_outputs_info_) {
    auto abstract_tensor = CreateTensorAbstract(custom_output_info.shape, custom_output_info.data_type);
    if (abstract_tensor == nullptr) {
      MS_LOG(ERROR) << "Abstract tensor is null for output " << custom_output_info.name;
      return false;
    }
    abstract_tensor->set_name(custom_output_info.name);
    abstract_list.emplace_back(abstract_tensor);
  }
  custom_node->set_abstract(std::make_shared<abstract::AbstractTuple>(abstract_list));
  return true;
}

bool OMModelParser::ParseInputsAndOutputsInfo(const converter::ConverterParameters &flag) {
  if (flag.attrs.empty()) {
    MS_LOG(ERROR) << "Can not find info.";
    return false;
  }
  auto attrs = flag.attrs;
  auto input_names = ParseNames(attrs, "input_name_vector");
  auto input_shapes = ParseShapes(attrs, "input_shape_vector");
  auto input_data_types = ParseDataTypes(attrs, "input_data_type_vector");
  auto output_names = ParseNames(attrs, "output_name_vector");
  auto output_shapes = ParseShapes(attrs, "output_shape_vector");
  auto output_data_types = ParseDataTypes(attrs, "output_data_type_vector");
  auto input_size = input_names.size();
  if (input_size == 0 || input_shapes.size() != input_size || input_data_types.size() != input_size) {
    MS_LOG(ERROR) << "Invalid inputs info.";
    return false;
  }
  auto output_size = output_names.size();
  if (output_size == 0 || output_shapes.size() != output_size || output_data_types.size() != output_size) {
    MS_LOG(ERROR) << "Invalid outputs info.";
    return false;
  }
  for (size_t i = 0; i < input_size; i++) {
    InputInfo input_info;
    input_info.name = input_names[i];
    input_info.shape = input_shapes[i];
    input_info.data_type = input_data_types[i];
    inputs_info_.push_back(input_info);
  }
  for (size_t i = 0; i < output_size; i++) {
    CustomOutputInfo custom_output_info;
    custom_output_info.name = output_names[i];
    custom_output_info.shape = output_shapes[i];
    custom_output_info.data_type = output_data_types[i];
    custom_outputs_info_.push_back(custom_output_info);
  }
  return true;
}

std::vector<std::string> OMModelParser::ParseNames(const std::map<std::string, std::string> &attrs,
                                                   const std::string &names_section) {
  auto ret = attrs.find(names_section);
  if (ret == attrs.end()) {
    MS_LOG(ERROR) << "Invalid attrs: " << names_section;
    return {};
  }
  auto name_str = SplitStringToVector(ret->second, ':');
  return name_str;
}

std::vector<std::vector<int64_t>> OMModelParser::ParseShapes(const std::map<std::string, std::string> &attrs,
                                                             const std::string &shapes_section) {
  auto ret = attrs.find(shapes_section);
  if (ret == attrs.end()) {
    MS_LOG(ERROR) << "Invalid attrs: " << shapes_section;
    return {};
  }
  std::vector<std::vector<int64_t>> shapes;
  auto shape_str = SplitStringToVector(ret->second, ':');
  for (size_t i = 0; i < shape_str.size(); i++) {
    auto shape_str_split = SplitStringToVector(shape_str[i], ',');
    std::vector<int64_t> shape;
    std::transform(shape_str_split.begin(), shape_str_split.end(), std::back_inserter(shape), [](std::string dim_str) {
      int64_t val;
      ConvertStrToInt(dim_str, &val);
      return val;
    });
    shapes.push_back(shape);
  }
  return shapes;
}

std::vector<TypeId> OMModelParser::ParseDataTypes(const std::map<std::string, std::string> &attrs,
                                                  const std::string &data_types_section) {
  auto ret = attrs.find(data_types_section);
  if (ret == attrs.end()) {
    MS_LOG(ERROR) << "Invalid attrs: " << data_types_section;
    return {};
  }
  auto data_type_str = SplitStringToVector(ret->second, ':');
  const std::map<std::string, TypeId> supported_data_type_map = {
    {"FLOAT16", kNumberTypeFloat16}, {"FLOAT32", kNumberTypeFloat32}, {"FLOAT64", kNumberTypeFloat64},
    {"UINT8", kNumberTypeUInt8},     {"UINT16", kNumberTypeUInt16},   {"UINT32", kNumberTypeUInt32},
    {"UINT64", kNumberTypeUInt64},   {"INT8", kNumberTypeInt8},       {"INT16", kNumberTypeInt16},
    {"INT32", kNumberTypeInt32},     {"INT64", kNumberTypeInt64}};
  std::vector<TypeId> data_types;
  std::transform(
    data_type_str.begin(), data_type_str.end(), std::back_inserter(data_types),
    [&supported_data_type_map](const std::string &type_str) { return supported_data_type_map.at(type_str); });
  return data_types;
}

using mindspore::converter::kFmkTypeOM;
REG_MODEL_PARSER(kFmkTypeOM, LiteModelParserCreator<OMModelParser>)
}  // namespace lite
}  // namespace mindspore
