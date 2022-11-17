/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "src/om_generator.h"
#include <fcntl.h>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <unordered_map>
#include "ops/tuple_get_item.h"
#include "ops/custom.h"
#include "ops/make_tuple.h"
#include "common/anf_util.h"
#include "common/string_util.h"
#include "mapper/op_mapper_registry.h"
#include "common/op_attr.h"
#include "./pico_mapper_api.h"
#include "op/preprocess_operator.h"
#include "src/mapper_config_parser.h"
#include "src/graph_split_api.h"

namespace mindspore {
namespace dpico {
namespace {
const std::unordered_map<TypeId, std::string> kMapperSupportedTypes = {
  {kNumberTypeUInt8, "U8"},   {kNumberTypeInt8, "S8"},      {kNumberTypeInt16, "S16"},
  {kNumberTypeUInt16, "U16"}, {kNumberTypeFloat16, "FP16"}, {kNumberTypeFloat32, "FP32"}};
api::CNodePtrList GetOutputCNodes(const api::FuncGraphManagerPtr &manager, const api::AnfNodePtr &node) {
  MS_CHECK_TRUE_MSG(manager != nullptr && node != nullptr, {}, "obtain nullptr input parameter.");
  api::CNodePtrList output_cnodes;
  auto node_users = manager->GetUsers(node);
  if (node_users.size() == 1) {
    auto output_cnode = node_users.begin()->first->cast<api::CNodePtr>();
    if (output_cnode != nullptr) {
      (void)output_cnodes.emplace_back(output_cnode);
    }
  } else {
    std::map<int, api::CNodePtr> output_cnode_ptr_map;
    bool has_tuple_get_item = false;
    for (const auto &node_user : node_users) {
      auto output_cnode = node_user.first->cast<api::CNodePtr>();
      if (output_cnode != nullptr) {
        if (CheckPrimitiveType(output_cnode, api::MakeShared<ops::TupleGetItem>())) {
          has_tuple_get_item = true;
          auto last_input_idx = output_cnode->inputs().size() - 1;
          auto anode = output_cnode->input(last_input_idx);
          MS_CHECK_TRUE_MSG(anode != nullptr, {},
                            output_cnode->fullname_with_scope() << " input(" << last_input_idx << ") is nullptr.");
          auto value_node = anode->cast<api::ValueNodePtr>();
          MS_CHECK_TRUE_MSG(value_node != nullptr, {}, "value node is nullptr. " << anode->fullname_with_scope());
          auto value_ptr = value_node->value();
          MS_CHECK_TRUE_MSG(value_ptr != nullptr, {}, "value ptr is nullptr. " << anode->fullname_with_scope());
          auto num_str = value_ptr->ToString();
          MS_CHECK_TRUE_MSG(IsValidUnsignedNum(num_str), {}, "num_str must an unsigned int, which is invalid");
          auto index = stoi(num_str);
          MS_CHECK_TRUE_MSG(index >= 0, {}, "tuple_get_item index is invalid. " << index);
          output_cnode_ptr_map[index] = output_cnode;
        } else {
          (void)output_cnodes.emplace_back(output_cnode);
        }
      }
    }
    if (has_tuple_get_item) {
      (void)std::transform(output_cnode_ptr_map.begin(), output_cnode_ptr_map.end(), std::back_inserter(output_cnodes),
                           [](const std::pair<int, api::CNodePtr> &iter) { return iter.second; });
    }
  }
  return output_cnodes;
}

std::string GetOutNodesStr(const api::FuncGraphManagerPtr &manager, const Subgraph &sub_graph,
                           const std::unordered_map<std::string, std::string> &mapper_config) {
  std::string out_nodes_str;
  if (mapper_config.find(kOutNodes) != mapper_config.end()) {
    out_nodes_str = mapper_config.at(kOutNodes);
  }
  if (!out_nodes_str.empty() && out_nodes_str.back() != ';') {
    out_nodes_str.push_back(';');
  }
  auto subgraph_outputs = GetSubgraphOutputs(sub_graph, manager);
  api::AnfNodePtrList report_nodes;
  for (const auto &output : subgraph_outputs) {
    auto node_users = manager->GetUsers(output);
    for (const auto &node_user : node_users) {
      auto output_cnode = node_user.first->cast<api::CNodePtr>();
      if (output_cnode == nullptr) {
        continue;
      }
      if (std::find(sub_graph.cnodes.begin(), sub_graph.cnodes.end(), output_cnode) != sub_graph.cnodes.end()) {
        report_nodes.push_back(output);
      }
    }
  }
  out_nodes_str = std::accumulate(report_nodes.begin(), report_nodes.end(), out_nodes_str,
                                  [](const std::string &res, const api::AnfNodePtr &anf_node_ptr) {
                                    return res + anf_node_ptr->fullname_with_scope() + ":0;";
                                  });
  return out_nodes_str;
}
std::string GetInputTypeStr(const api::AnfNodePtrList &subgraph_inputs,
                            const std::unordered_map<std::string, std::string> &mapper_config) {
  std::string input_type_str;
  for (const auto &input : subgraph_inputs) {
    auto node_name = input->fullname_with_scope();
    if (CheckPrimitiveType(input, api::MakeShared<ops::Custom>())) {
      node_name = GetCustomOutputName(input);
      MS_CHECK_TRUE_MSG(!node_name.empty(), {}, "get custom node origin name failed." << input->fullname_with_scope());
    }
    auto abstract = GetAbstractFromAnfNode(input);
    MS_CHECK_TRUE_MSG(abstract != nullptr, "", "get abstract failed. " << input->fullname_with_scope());
    TypeId type_id;
    if (FetchTypeIdFromAbstract(abstract, &type_id) != RET_OK) {
      MS_LOG(ERROR) << "get type_id failed." << input->fullname_with_scope();
      return "";
    }
    std::string data_type;
    if (kMapperSupportedTypes.find(type_id) == kMapperSupportedTypes.end()) {
      MS_LOG(WARNING) << node_name << "'s data type " << dpico::TypeIdToString(type_id)
                      << " is unsupported by dpico, will set it to FP32";
      data_type = "FP32";
    } else {
      data_type = kMapperSupportedTypes.at(type_id);
    }
    input_type_str = node_name + ':' + data_type + ';';
  }
  if (!input_type_str.empty() && input_type_str.back() == ';') {
    input_type_str.pop_back();
  }
  return input_type_str;
}
STATUS ConfigImageList(const api::AnfNodePtrList &subgraph_inputs, std::ofstream *mapper_ofs) {
  MS_CHECK_TRUE_MSG(mapper_ofs != nullptr, RET_ERROR, "mapper_ofs is nullptr.");
  auto image_lists = MapperConfigParser::GetInstance()->GetImageLists();
  if (image_lists.empty()) {
    MS_LOG(ERROR) << "image_lists shouldn't be empty.";
    mapper_ofs->close();
    return RET_ERROR;
  }
  *mapper_ofs << kImageList << " ";

  for (const auto &input : subgraph_inputs) {
    auto node_name = input->fullname_with_scope();
    if (node_name.empty()) {
      MS_LOG(ERROR) << "graph input node name is empty.";
      mapper_ofs->close();
      return RET_ERROR;
    }
    if (CheckPrimitiveType(input, api::MakeShared<ops::Custom>())) {
      node_name = GetCustomOutputName(input);
      if (node_name.empty()) {
        MS_LOG(ERROR) << "get custom node origin name failed." << input->fullname_with_scope();
        mapper_ofs->close();
        return RET_ERROR;
      }
    }
    if (image_lists.find(node_name) == image_lists.end()) {
      MS_LOG(ERROR) << "can't find " << node_name << " in image_lists.";
      mapper_ofs->close();
      return RET_ERROR;
    }
    *mapper_ofs << node_name << ":" << image_lists[node_name] << ";";
  }
  *mapper_ofs << std::endl;
  return RET_OK;
}
}  // namespace

int OmGenerator::GenerateAippConfig(const std::string &aipp_cfg_path, const api::AnfNodePtrList &subgraph_inputs) {
  auto aipp_modules = MapperConfigParser::GetInstance()->GetAippModules();
  bool need_aipp_cfg = false;
  std::ofstream aipp_ofs;
  aipp_ofs.open(aipp_cfg_path, std::ios::out);
  MS_CHECK_TRUE_MSG(aipp_ofs.is_open(), RET_ERROR, "open file failed.");
  (void)aipp_ofs.precision(kNumPrecision);
  aipp_ofs << "aipp_op {" << std::endl;
  for (size_t i = 0; i < subgraph_inputs.size(); i++) {
    auto input = subgraph_inputs.at(i);
    auto node_name = input->fullname_with_scope();
    if (node_name.empty()) {
      MS_LOG(ERROR) << "graph input node name is empty.";
      aipp_ofs.close();
      return RET_ERROR;
    }
    if (aipp_modules.find(node_name) != aipp_modules.end()) {
      need_aipp_cfg = true;
      auto aipp_module = aipp_modules.at(node_name);
      aipp_ofs << kRelatedInputRank << ":" << i << std::endl;
      aipp_ofs << kInputFormat << ":" << aipp_module.input_format << std::endl;
      aipp_ofs << kModelFormat << ":" << aipp_module.model_format << std::endl;
      for (const auto &iter : aipp_module.mean_map) {
        aipp_ofs << kMeanChn << "_" << iter.first << ":" << aipp_module.mean_map.at(iter.first) << std::endl;
      }
      for (const auto &iter : aipp_module.val_map) {
        aipp_ofs << kVarReciChn << "_" << iter.first << ":" << aipp_module.val_map.at(iter.first) << std::endl;
      }
    }
  }
  aipp_ofs << "}" << std::endl;
  aipp_ofs.close();
  if (!need_aipp_cfg) {
    return RET_NO_CHANGE;
  }
  return RET_OK;
}

int OmGenerator::GenerateMapperConfig(const api::FuncGraphPtr &func_graph, const Subgraph &sub_graph, int custom_id,
                                      const std::string &mapper_cfg_path) {
  MS_CHECK_TRUE_MSG(func_graph != nullptr, RET_ERROR, "func_graph is nullptr.");
  auto manager = func_graph->manager();
  MS_CHECK_TRUE_MSG(manager != nullptr, RET_ERROR, "funcgraph manager is nullptr.");
  auto mapper_config = MapperConfigParser::GetInstance()->GetCommonConfig();
  MS_CHECK_TRUE_MSG(!mapper_config.empty(), RET_ERROR, "mapper config is empty.");
  auto subgraph_inputs = GetSubgraphInputs(sub_graph, func_graph);
  MS_CHECK_TRUE_MSG(!subgraph_inputs.empty(), RET_ERROR, "get subgraph inputs failed.");
  // generate mapper config file
  std::ofstream mapper_ofs;
  mapper_ofs.open(mapper_cfg_path, std::ios::out);
  MS_CHECK_TRUE_MSG(mapper_ofs.is_open(), RET_ERROR, "open file failed.");
  for (const auto &iter : mapper_config) {
    if (iter.first == kImageList || iter.first == kInsertOpConf || iter.first == kInstructionName ||
        iter.first == kOutNodes || iter.first == kInputType || iter.first == kInputShape || iter.first == kOutputType) {
      continue;
    }
    mapper_ofs << iter.first << " " << iter.second << std::endl;
  }
  mapper_ofs << kInstructionName << " " << MapperConfigParser::GetInstance()->GetOutputPath() << kCustomName
             << std::to_string(custom_id) << "/inst" << std::endl;

  // set subgraph inner cnode to report
  auto out_nodes_str = GetOutNodesStr(manager, sub_graph, mapper_config);
  if (!out_nodes_str.empty()) {
    mapper_ofs << kOutNodes << " " + out_nodes_str << std::endl;
  }

  // config [image_list]
  if (ConfigImageList(subgraph_inputs, &mapper_ofs) != RET_OK) {
    MS_LOG(ERROR) << "config [image_list] failed.";
    return RET_ERROR;
  }

  // config [input_type]
  auto input_type_str = GetInputTypeStr(subgraph_inputs, mapper_config);
  if (!input_type_str.empty()) {
    mapper_ofs << kInputType << " " + input_type_str << std::endl;
  }

  // generate aipp config file
  auto aipp_cfg_path =
    MapperConfigParser::GetInstance()->GetOutputPath() + kCustomName + std::to_string(custom_id) + "_aipp.cfg";
  auto status = GenerateAippConfig(aipp_cfg_path, subgraph_inputs);
  if (status == RET_ERROR) {
    MS_LOG(ERROR) << "generate aipp config file failed.";
    return RET_ERROR;
  } else if (status == RET_OK) {
    mapper_ofs << kInsertOpConf << " " << aipp_cfg_path << std::endl;
  }
  mapper_ofs.close();

  return RET_OK;
}

int OmGenerator::TransformSubGraphInputs(const api::AnfNodePtrList &inputs,
                                         std::vector<BaseOperatorPtr> *base_operators) {
  MS_CHECK_TRUE_MSG(!inputs.empty(), RET_ERROR, "subgraph inputs shouldn't be empty.");
  MS_CHECK_TRUE_MSG(base_operators != nullptr, RET_ERROR, "base_operators is nullptr.");
  for (const auto &input : inputs) {
    MS_CHECK_TRUE_MSG(input != nullptr, RET_ERROR, "input node is nullptr.");
    auto preprocess_operator = std::make_unique<mapper::PreprocessOperator>();
    MS_CHECK_TRUE_MSG(preprocess_operator != nullptr, RET_ERROR, "preprocess_operator is nullptr.");
    preprocess_operator->SetOpType(mapper::OpType::PREPROCESS);

    auto op_name = input->fullname_with_scope();
    if (CheckPrimitiveType(input, api::MakeShared<ops::Custom>())) {
      op_name = GetCustomOutputName(input);
      MS_CHECK_TRUE_MSG(!op_name.empty(), RET_ERROR,
                        "get custom node output name failed." << input->fullname_with_scope());
    }
    preprocess_operator->SetOpName(op_name);
    preprocess_operator->SetOutputNamesVec({op_name});

    ShapeVector shape_vector;
    MS_CHECK_TRUE_MSG(GetAnfNodeOutputShape(input, &shape_vector) == RET_OK, RET_ERROR,
                      "get " << input->fullname_with_scope() << " output shape failed.");
    std::vector<int32_t> dims;
    (void)std::transform(shape_vector.begin(), shape_vector.end(), std::back_inserter(dims),
                         [](const int64_t &value) { return static_cast<int32_t>(value); });
    preprocess_operator->SetShapeParamVec(dims);
    base_operators->push_back(std::move(preprocess_operator));
  }

  return RET_OK;
}

int OmGenerator::TransformSubGraphCNodes(const api::FuncGraphManagerPtr &manager, const api::CNodePtrList &cnodes,
                                         std::vector<BaseOperatorPtr> *base_operators) {
  MS_CHECK_TRUE_MSG(!cnodes.empty(), RET_ERROR, "subgraph inputs shouldn't be empty.");
  MS_CHECK_TRUE_MSG(base_operators != nullptr, RET_ERROR, "base_operators is nullptr.");
  for (const auto &cnode : cnodes) {
    MS_CHECK_TRUE_MSG(api::utils::isa<api::CNodePtr>(cnode), RET_ERROR, "cur node should be a cnode");
    auto primitive = api::GetValueNode<api::PrimitivePtr>(cnode->input(0));
    MS_CHECK_TRUE_MSG(primitive != nullptr, RET_ERROR,
                      "invalid anf node, which don't have primitive. " << cnode->fullname_with_scope());
    if (CheckPrimitiveType(cnode, api::MakeShared<ops::MakeTuple>()) ||
        CheckPrimitiveType(cnode, api::MakeShared<ops::TupleGetItem>())) {
      MS_LOG(DEBUG) << "MakeTuple and TupleGetItem don't need to transform.";
      continue;
    }

    auto output_cnodes = GetOutputCNodes(manager, cnode);
    MS_CHECK_TRUE_MSG(!output_cnodes.empty(), RET_ERROR, "output_cnodes is empty. " << cnode->fullname_with_scope());
    std::string op_type_name;
    if (GetPrimitiveType(cnode, &op_type_name) != RET_OK) {
      MS_LOG(ERROR) << "get cnode primitive type failed " << cnode->fullname_with_scope();
      return RET_ERROR;
    }
    auto op_mapper = OpMapperRegistry::GetInstance()->GetOpMapper(op_type_name);
    MS_CHECK_TRUE_MSG(op_mapper != nullptr, RET_ERROR, "op mapper is unsupported: " << op_type_name);
    if (op_mapper->Map(cnode, base_operators, primitive, output_cnodes) != RET_OK) {
      MS_LOG(ERROR) << "op mapper Map failed: " << op_type_name;
      return RET_ERROR;
    }
  }

  return RET_OK;
}

int OmGenerator::Run(const api::FuncGraphPtr &func_graph, const Subgraph &sub_graph, int custom_id,
                     mapper::ModelCoreInfo *om_model_info, bool use_origin_config) {
  MS_CHECK_TRUE_MSG(func_graph != nullptr && om_model_info != nullptr, RET_ERROR, "obtain nullptr input parameter.");
  auto manager = func_graph->manager();
  MS_CHECK_TRUE_MSG(manager != nullptr, RET_ERROR, "funcgraph manager is nullptr.");
  std::string mapper_cfg_path;
  if (!use_origin_config) {
    mapper_cfg_path =
      MapperConfigParser::GetInstance()->GetOutputPath() + kCustomName + std::to_string(custom_id) + "_mapper.cfg";
    if (GenerateMapperConfig(func_graph, sub_graph, custom_id, mapper_cfg_path) != RET_OK) {
      MS_LOG(ERROR) << "generate om config file failed.";
      return RET_ERROR;
    }
  } else {
    mapper_cfg_path = MapperConfigParser::GetInstance()->GetOriginConfigPath();
  }

  std::vector<BaseOperatorPtr> base_operators;
  auto subgraph_inputs = GetSubgraphInputs(sub_graph, func_graph);
  MS_CHECK_TRUE_MSG(!subgraph_inputs.empty(), RET_ERROR,
                    "get subgraph inputs failed. subgraph id is " << sub_graph.graph_id);
  if (TransformSubGraphInputs(subgraph_inputs, &base_operators) != RET_OK) {
    MS_LOG(ERROR) << "subgraph inputs transform failed.";
    return RET_ERROR;
  }

  if (TransformSubGraphCNodes(manager, sub_graph.cnodes, &base_operators) != RET_OK) {
    MS_LOG(ERROR) << "subgraph cnodes transform failed.";
    return RET_ERROR;
  }

  if (!mapper::GenerateModelBinary(mapper_cfg_path.c_str(), base_operators, om_model_info)) {
    MS_LOG(ERROR) << "Generate Model Binary failed.";
    return RET_ERROR;
  }

  return RET_OK;
}
}  // namespace dpico
}  // namespace mindspore
