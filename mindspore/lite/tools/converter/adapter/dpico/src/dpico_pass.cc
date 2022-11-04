/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "src/dpico_pass.h"
#include <memory>
#include <string>
#include <vector>
#include <map>
#include <unordered_set>
#include <algorithm>
#include <utility>
#include "ops/cast.h"
#include "ops/transpose.h"
#include "ops/return.h"
#include "ops/depend.h"
#include "common/format_utils.h"
#include "common/anf_util.h"
#include "common/string_util.h"
#include "common/op_attr.h"
#include "checker/op_checker.h"
#include "src/om_generator.h"
#include "include/registry/pass_registry.h"
#include "src/data_preprocessor.h"
#include "src/mapper_config_parser.h"
#include "include/registry/converter_context.h"
#include "src/calib_data_generator.h"
#include "src/custom_creator.h"

namespace mindspore {
namespace dpico {
namespace {
const size_t kMinimumNumbOfSegments = 1;
bool CheckInputDimSize(const api::CNodePtr &cnode) {
  for (size_t i = 0; i < cnode->inputs().size(); i++) {
    auto input_node = cnode->input(i);
    if (!api::utils::isa<api::CNodePtr>(input_node) &&
        (input_node->cast<api::ParameterPtr>() == nullptr || input_node->cast<api::ParameterPtr>()->has_default())) {
      continue;
    }
    ShapeVector shape_vector;
    if (GetInputShapeFromCNode(cnode, i, &shape_vector) == RET_OK) {
      if (shape_vector.size() <= 1 || shape_vector.size() > kDims4) {
        MS_LOG(DEBUG) << cnode->fullname_with_scope() << " input:" << i << " 's input shape size "
                      << shape_vector.size() << " should be in range [2, 4].";
        return false;
      }
    }
  }
  return true;
}

bool CheckOpHasInferred(const api::CNodePtr &cnode) {
  MS_ASSERT(cnode != nullptr);
  if (GetBoolAttr(cnode, kInferDone)) {
    return true;
  }
  if (!CheckPrimitiveType(cnode, api::MakeShared<ops::Transpose>())) {
    return false;
  }
  auto abstract = cnode->abstract();
  MS_CHECK_TRUE_MSG(abstract != nullptr, false, "abstract is nullptr.");
  ShapeVector shape;
  if (FetchShapeFromAbstract(abstract, &shape) != lite::RET_OK) {
    return false;
  }
  return !shape.empty() && std::all_of(shape.begin(), shape.end(), [](int64_t dim) { return dim > 0; });
}

STATUS AddDumpKernels(const api::FuncGraphPtr &func_graph, const Subgraph &subgraph, api::AnfNodePtrList *dump_kernels,
                      std::map<api::AnfNodePtr, std::pair<api::CNodePtr, int>> *param_to_cnode) {
  MS_ASSERT(func_graph != nullptr && dump_kernels != nullptr && param_to_cnode != nullptr);
  auto subgraph_inputs = GetSubgraphInputs(subgraph, func_graph);
  MS_CHECK_TRUE_MSG(!subgraph_inputs.empty(), RET_ERROR,
                    "get subgraph inputs failed. subgraph id is " << subgraph.graph_id);
  (void)dump_kernels->insert(dump_kernels->end(), subgraph_inputs.begin(), subgraph_inputs.end());
  bool is_main_graph =
    func_graph->get_attr(kIsMainGraph) != nullptr && api::GetValue<bool>(func_graph->get_attr(kIsMainGraph));
  if (is_main_graph) {
    return RET_OK;
  }
  for (const auto &node : subgraph_inputs) {
    if (!api::utils::isa<api::Parameter>(node)) {
      continue;
    }
    if (param_to_cnode->find(node) != param_to_cnode->end()) {
      continue;
    }
    for (const auto &inner_cnode : subgraph.cnodes) {
      MS_CHECK_TRUE_MSG(api::utils::isa<api::CNodePtr>(inner_cnode), RET_ERROR, "inner cnode is nullptr");
      auto cnode_inputs = inner_cnode->inputs();
      auto iter = std::find(cnode_inputs.begin(), cnode_inputs.end(), node);
      if (iter == cnode_inputs.end()) {
        continue;
      }
      (void)param_to_cnode->emplace(node, std::make_pair(inner_cnode, iter - cnode_inputs.begin() - 1));
    }
  }
  return RET_OK;
}
STATUS ModifyGraphInputDataType(const Subgraph &subgraph, const api::FuncGraphPtr &func_graph,
                                const std::shared_ptr<mapper::ModelCoreInfo> &om_model_info) {
  auto subgraph_inputs = GetSubgraphInputs(subgraph, func_graph);
  for (size_t i = 0; i < subgraph_inputs.size(); i++) {
    auto input = subgraph_inputs.at(i);
    auto input_node_name = input->fullname_with_scope();
    auto param = input->cast<api::ParameterPtr>();
    if (param != nullptr && !param->has_default()) {  // only for graph input parameter node
      auto param_abstract = param->abstract();
      MS_CHECK_TRUE_MSG(param_abstract != nullptr, RET_ERROR, "param_abstract is nullptr");
      auto abstractScalar = param_abstract->cast<api::AbstractTensorPtr>();
      MS_CHECK_TRUE_MSG(abstractScalar != nullptr, RET_ERROR, "abstractScalar is nullptr");
      auto element = abstractScalar->element();
      MS_CHECK_TRUE_MSG(element != nullptr, RET_ERROR, "element is nullptr");
      auto correspond_info_iter = std::find_if(om_model_info->inputInfos.begin(), om_model_info->inputInfos.end(),
                                               [input_node_name](const mapper::DataCoreInfo &data_core_info) {
                                                 auto om_input_name = RemoveSpecifiedChar(data_core_info.name, '\0');
                                                 return om_input_name == input_node_name;
                                               });
      MS_CHECK_TRUE_MSG(correspond_info_iter != om_model_info->inputInfos.end(), RET_ERROR,
                        "can't find \"" << input_node_name << "\" in om model input infos.");
      switch (correspond_info_iter->type) {
        case mapper::OpDataType::OP_DTYPE_S8:
          element->set_type(api::Type::GetType(kNumberTypeInt8));
          break;
        case mapper::OpDataType::OP_DTYPE_U8:
          element->set_type(api::Type::GetType(kNumberTypeUInt8));
          break;
        case mapper::OpDataType::OP_DTYPE_S16:
          element->set_type(api::Type::GetType(kNumberTypeInt16));
          break;
        case mapper::OpDataType::OP_DTYPE_U16:
          element->set_type(api::Type::GetType(kNumberTypeUInt16));
          break;
        case mapper::OpDataType::OP_DTYPE_F32:
          element->set_type(api::Type::GetType(kNumberTypeFloat32));
          break;
        default:
          MS_LOG(ERROR) << "current op type is unsupported. " << om_model_info->inputInfos.at(i).type;
          return RET_ERROR;
      }
    }
  }
  return RET_OK;
}
#ifdef Debug
void PrintUnsupportedOps(const std::map<std::string, std::vector<std::string>> &unsupported_ops,
                         size_t unsupported_ops_size, const api::FuncGraphPtr &func_graph) {
  if (!unsupported_ops.empty()) {
    if (func_graph->get_attr(kGraphName) != nullptr) {
      auto func_graph_name = api::GetValue<std::string>(func_graph->get_attr(kGraphName));
      MS_LOG(WARNING) << "func_graph: " << func_graph_name;
    }
    MS_LOG(WARNING) << "there are " << unsupported_ops_size << " unsupported ops in this net.";
    for (auto &unsupported_op : unsupported_ops) {
      MS_LOG(WARNING) << "type: " << unsupported_op.first
                      << " op_nums: " << std::to_string(unsupported_op.second.size());
      auto type_name_and_op_names = std::accumulate(
        unsupported_op.second.begin(), unsupported_op.second.end(), std::string{},
        [](const std::string &cur_str, const std::string &op_name) { return cur_str + " |" + op_name; });
      MS_LOG(WARNING) << type_name_and_op_names;
    }
  }
}
#endif
}  // namespace
STATUS DpicoPass::InitDpicoConfigInfo() {
  dpico_config_path_ = "./dpico.cfg";
  bool use_default_config = true;
  auto config_info = converter::ConverterContext::GetConfigInfo("dpico");
  if (!config_info.empty()) {
    if (config_info.find("dpico_config_path") != config_info.end()) {
      dpico_config_path_ = config_info.at("dpico_config_path");
      use_default_config = false;
    }
    if (config_info.find("save_temporary_files") != config_info.end()) {
      auto save_temp_file_str = config_info.at("save_temporary_files");
      if (save_temp_file_str == "on") {
        save_tmp_files_ = true;
      } else if (save_temp_file_str == "off") {
        save_tmp_files_ = false;
      } else {
        MS_LOG(WARNING) << "invalid [save_temporary_files] value, will consider it as off.";
        save_tmp_files_ = false;
      }
    }
  }
  if (use_default_config) {
    MS_LOG(WARNING)
      << R"(there is no "dpico_config_path" in the converter config file, will use the default value: "./dpico.cfg")";
  }
  if (AccessFile(dpico_config_path_, F_OK) != 0) {
    MS_LOG(ERROR) << "File not exist: " << dpico_config_path_;
    return RET_ERROR;
  }
  return RET_OK;
}

void DpicoPass::FetchFuncGraphs(const api::FuncGraphPtr &func_graph) {
  if (std::find(func_graphs_.begin(), func_graphs_.end(), func_graph) == func_graphs_.end()) {
    func_graphs_.push_back(func_graph);
  } else {
    return;
  }
  auto node_list = api::FuncGraph::TopoSort(func_graph->get_return());
  for (auto &node : node_list) {
    auto fg = api::GetValueNode<api::FuncGraphPtr>(node);
    if (fg != nullptr) {
      FetchFuncGraphs(fg);
    }
  }
}

STATUS DpicoPass::CheckDynamicInputShape(const api::FuncGraphPtr &func_graph) {
  MS_ASSERT(func_graph != nullptr);
  auto graph_inputs = func_graph->get_inputs();
  for (const auto &node : graph_inputs) {
    auto graph_input = node->cast<api::ParameterPtr>();
    MS_CHECK_TRUE_MSG(graph_input != nullptr, RET_ERROR, "graph_input is nullptr.");
    ShapeVector shape_vector;
    if (GetShapeVectorFromParameter(graph_input, &shape_vector) != RET_OK) {
      MS_LOG(ERROR) << "get shape vector failed. " << graph_input->fullname_with_scope();
      return RET_ERROR;
    }
    std::vector<size_t> indexes;
    for (size_t i = 0; i < shape_vector.size(); i++) {
      if (shape_vector.at(i) < 0) {
        indexes.push_back(i);
      }
    }
    if (!indexes.empty()) {
      MS_LOG(WARNING) << "dynamic graph input is unsupported by dpico.";
      return RET_NO_CHANGE;
    }
  }

  return RET_OK;
}

STATUS DpicoPass::MarkNodes(const api::FuncGraphPtr &func_graph) {
  MS_CHECK_TRUE_MSG(func_graph != nullptr, RET_ERROR, "func_graph is nullptr.");
  auto manager = api::FuncGraphManager::Manage(func_graph, true);
  MS_CHECK_TRUE_MSG(manager != nullptr, RET_ERROR, "manager is nullptr.");
  auto node_list = api::FuncGraph::TopoSort(func_graph->get_return());
  std::map<std::string, std::vector<std::string>> unsupported_ops;
  size_t unsupported_ops_size = 0;
  for (auto &node : node_list) {
    auto cnode = node->cast<api::CNodePtr>();
    if (cnode == nullptr) {
      continue;
    }
    auto primitive = api::GetValueNode<api::PrimitivePtr>(cnode->input(0));
    MS_CHECK_TRUE_MSG(primitive != nullptr, RET_ERROR, "primitive is nullptr:" << cnode->fullname_with_scope());
    std::string op_type_name;
    if (GetPrimitiveType(cnode, &op_type_name) != RET_OK) {
      MS_LOG(ERROR) << "get cnode primitive type failed:" << cnode->fullname_with_scope();
      return RET_ERROR;
    }

    // mark nodes
    bool is_supported = false;
    if (IsSpecialType(cnode)) {
      auto cnode_inputs = cnode->inputs();
      is_supported = CheckPrimitiveType(cnode, api::MakeShared<ops::Return>()) ||
                         CheckPrimitiveType(cnode, api::MakeShared<ops::Depend>())
                       ? false
                       : std::all_of(cnode_inputs.begin(), cnode_inputs.end(), [](const api::AnfNodePtr &node) {
                           return !api::utils::isa<api::CNode>(node) || GetBoolAttr(node, kIsMapperSupported);
                         });
    } else {
      auto op_checker = OpCheckerRegistry::GetInstance()->GetOpChecker(op_type_name);
      if (op_checker != nullptr) {
        auto node_users = manager->GetUsers(cnode);
        is_supported = CheckInputDimSize(cnode) &&
                       op_checker->Check(cnode, static_cast<int>(node_users.size()), mindspore::Format::NCHW);
      }
      is_supported = is_supported && CheckOpHasInferred(cnode);
      is_supported = is_supported && (CheckPrimitiveType(cnode, api::MakeShared<ops::Cast>())
                                        ? api::utils::isa<api::CNode>(cnode->input(1)) &&
                                            GetBoolAttr(cnode->input(1), kIsMapperSupported)
                                        : true);
    }
    if (!is_supported && !CheckPrimitiveType(cnode, api::MakeShared<ops::Return>())) {
      unsupported_ops[op_type_name].push_back(cnode->fullname_with_scope());
      unsupported_ops_size++;
    }
    (void)primitive->AddAttr(kIsMapperSupported, api::MakeValue<bool>(is_supported));
  }
#ifdef Debug
  PrintUnsupportedOps(unsupported_ops, unsupported_ops_size, func_graph);
#endif
  return RET_OK;
}

STATUS DpicoPass::ParseMapperConfig(const api::FuncGraphPtr &func_graph) {
  if (graph_split_info_.num_of_segments < kMinimumNumbOfSegments) {  // no segment
    MapperConfigParser::GetInstance()->SetOriginConfigFilePath(dpico_config_path_);
    return RET_OK;
  }

  std::vector<std::string> graph_input_names;
  auto inputs = func_graph->get_inputs();
  (void)std::transform(inputs.begin(), inputs.end(), std::back_inserter(graph_input_names),
                       [](const api::AnfNodePtr &anode) { return anode->fullname_with_scope(); });

  if (MapperConfigParser::GetInstance()->Parse(dpico_config_path_, graph_input_names) != RET_OK) {
    MS_LOG(ERROR) << "parse mapper config file failed.";
    return RET_ERROR;
  }

  return RET_OK;
}

STATUS DpicoPass::DataPrepare(const api::FuncGraphPtr &func_graph, bool *use_origin_config) {
  MS_CHECK_TRUE_MSG(func_graph != nullptr && use_origin_config != nullptr, RET_ERROR,
                    "obtain nullptr input parameter.");
  auto graph_inputs = func_graph->get_inputs();
  auto mapper_config = MapperConfigParser::GetInstance()->GetCommonConfig();
  if (graph_split_info_.num_of_segments >= kMinimumNumbOfSegments &&
      mapper_config.find(kImageList) != mapper_config.end()) {
    MS_LOG(WARNING) << "there are unsupported npu operators in this model, and it's going to generate calib set.";
    if (DataPreprocessor::GetInstance()->Run(graph_inputs) != RET_OK) {
      MS_LOG(ERROR) << "input data preprocess failed.";
      return RET_ERROR;
    }

    api::AnfNodePtrList dump_kernels;
    std::map<api::AnfNodePtr, std::pair<api::CNodePtr, int>> param_to_cnode;
    for (auto &graph : func_graphs_) {
      for (auto &subgraph : graph_split_info_.subgraphs_map[graph]) {
        if (!subgraph.is_supported) {
          continue;
        }
        if (AddDumpKernels(graph, subgraph, &dump_kernels, &param_to_cnode) != RET_OK) {
          MS_LOG(ERROR) << "add dump kernels failed.";
          return RET_ERROR;
        }
      }
    }
    if (param_to_cnode.empty() &&
        std::all_of(dump_kernels.begin(), dump_kernels.end(),
                    [](const api::AnfNodePtr &node) { return api::utils::isa<api::Parameter>(node); })) {
      MS_LOG(DEBUG) << "required tensors are all graph inputs, which do not need to dump data.";
      return RET_OK;
    }
    int dump_level = static_cast<int>(param_to_cnode.empty() ? kDumpOutput : kDumpInputOutput);
    auto calib_data_generator = std::make_shared<CalibDataGenerator>(dump_level, param_to_cnode);
    MS_CHECK_TRUE_MSG(calib_data_generator != nullptr, RET_ERROR, "new calib generator failed.");
    if (calib_data_generator->Run(graph_inputs, dump_kernels) != RET_OK &&
        mapper_config.find(kGfpqParamFile) == mapper_config.end()) {
      MS_LOG(ERROR) << "generate calib data failed.";
      return RET_ERROR;
    }
    *use_origin_config = false;
  }
  return RET_OK;
}

STATUS DpicoPass::ReplaceSubgraphWithCustom(const api::FuncGraphPtr &func_graph, bool use_origin_config) {
  MS_CHECK_TRUE_MSG(func_graph != nullptr, RET_ERROR, "func_graph is nullptr.");
  auto manager = api::FuncGraphManager::Manage(func_graph, true);
  MS_CHECK_TRUE_MSG(manager != nullptr, RET_ERROR, "manager is nullptr.");
  for (auto &subgraph : graph_split_info_.subgraphs_map[func_graph]) {
    if (!subgraph.is_supported) {
      continue;
    }
    // pre fill subgraph's outputs_format info, so that the replaced custom op do not need to consider.
    if (FillSubgraphOutputsFormat(&subgraph, func_graph) != RET_OK) {
      MS_LOG(ERROR) << "fill subgraph's outputs_format info failed.";
      return RET_ERROR;
    }
  }
  for (auto &subgraph : graph_split_info_.subgraphs_map[func_graph]) {
    if (!subgraph.is_supported) {
      continue;
    }
    // transform subgraph to om
    auto om_generator = std::make_shared<OmGenerator>();
    MS_CHECK_TRUE_MSG(om_generator != nullptr, RET_ERROR, "OmGenerator is nullptr.");
    auto om_model_info = std::make_shared<mapper::ModelCoreInfo>();
    MS_CHECK_TRUE_MSG(om_model_info != nullptr, RET_ERROR, "ModelCoreInfo is nullptr.");
    MS_CHECK_TRUE_MSG(custom_op_creator_ != nullptr, RET_ERROR, "custom_op_creator_ is nullptr.");
    if (om_generator->Run(func_graph, subgraph, custom_op_creator_->GetCustomId(), om_model_info.get(),
                          use_origin_config) != RET_OK) {
      MS_LOG(ERROR) << "current subgraph generate om failed.";
      return RET_ERROR;
    }
    MS_CHECK_TRUE_MSG(WriteOmBufferToFile(om_model_info, custom_op_creator_->GetCustomId()) == RET_OK, RET_ERROR,
                      "save om file failed. custom id is " << custom_op_creator_->GetCustomId());
    auto custom_cnode = custom_op_creator_->CreateCustomOp(func_graph, &subgraph, om_model_info);
    MS_CHECK_TRUE_MSG(custom_cnode != nullptr, RET_ERROR, "custom_cnode is nullptr.");
    if (ModifyGraphInputDataType(subgraph, func_graph, om_model_info) != RET_OK) {
      MS_LOG(ERROR) << "modify graph input data type failed";
      return RET_ERROR;
    }
  }
  return RET_OK;
}
STATUS DpicoPass::WriteOmBufferToFile(const std::shared_ptr<mapper::ModelCoreInfo> &om_model_info, size_t custom_id) {
  if (!save_tmp_files_) {
    MS_LOG(DEBUG) << "om file will not be generated.";
    return RET_OK;
  }
  auto tmp_file_path = MapperConfigParser::GetInstance()->GetOutputPath();
  if (tmp_file_path.empty()) {
    auto dir_pos = dpico_config_path_.find_last_of('/');
    tmp_file_path = dpico_config_path_.substr(0, dir_pos + 1) + "tmp/";
    if (AccessFile(tmp_file_path, F_OK) == 0) {
      MS_CHECK_TRUE_MSG(RemoveDir(tmp_file_path) == RET_OK, RET_ERROR, "rm dir failed. " << tmp_file_path);
    }
    MS_CHECK_TRUE_MSG(CreateDir(&tmp_file_path) == RET_OK, RET_ERROR, "create dir failed. " << tmp_file_path);
  }
  std::string output_om_path = tmp_file_path + "custom_" + std::to_string(custom_id) + ".om";
  MS_CHECK_TRUE_MSG(WriteToBin(output_om_path, om_model_info->modelBuffer, om_model_info->modelSize) == RET_OK,
                    RET_ERROR, "write om to file failed.");
  return RET_OK;
}

STATUS DpicoPass::RemoveTemporaryFiles() {
  if (save_tmp_files_) {
    MS_LOG(DEBUG) << "temporary files will not be removed.";
    return RET_OK;
  }
  auto tmp_file_path = MapperConfigParser::GetInstance()->GetOutputPath();
  if (tmp_file_path.empty()) {
    MS_LOG(DEBUG) << "there is no temporary file, no need to remove.";
    return RET_OK;
  }
  if (RemoveDir(tmp_file_path) != RET_OK) {
    MS_LOG(ERROR) << "remove temp files failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

bool DpicoPass::Execute(const api::FuncGraphPtr &func_graph) {
  MS_CHECK_TRUE_MSG(func_graph != nullptr, false, "func_graph is nullptr.");
  func_graph->set_attr(kIsMainGraph, api::MakeValue<bool>(true));
  FetchFuncGraphs(func_graph);
  auto status = CheckDynamicInputShape(func_graph);
  if (status == RET_NO_CHANGE) {
    MS_LOG(WARNING) << "Dynamic input shape is unsupported by dpico. All nodes will be CPU operator. "
                    << "Or, you can set the \"inputShape\" when converting.";
    return true;
  } else if (status != RET_OK) {
    MS_LOG(ERROR) << "check dynamic graph input failed.";
    return false;
  }
  std::vector<std::string> output_names;
  converter::ConverterContext::SetGraphOutputTensorNames(output_names);

  if (InitDpicoConfigInfo() != RET_OK) {
    MS_LOG(ERROR) << "get dpico config info from converter context failed.";
    return false;
  }

  for (auto &graph : func_graphs_) {
    if (MarkNodes(graph) != RET_OK) {
      MS_LOG(ERROR) << "mark graph nodes failed.";
      return false;
    }
  }

  if (GraphSplit(func_graphs_, &graph_split_info_) != RET_OK) {
    MS_LOG(ERROR) << "split subgraphs failed.";
    return false;
  }

  if (ParseMapperConfig(func_graph) != RET_OK) {
    MS_LOG(ERROR) << "parse mapper config failed.";
    return false;
  }

  bool use_origin_config = true;
  if (DataPrepare(func_graph, &use_origin_config) != RET_OK) {
    MS_LOG(ERROR) << "prepare data for mapper failed.";
    return false;
  }

  bool has_unsupported = graph_split_info_.num_of_segments >= kMinimumNumbOfSegments;
  custom_op_creator_ = std::make_shared<CustomOpCreator>(0, has_unsupported);
  MS_CHECK_TRUE_MSG(custom_op_creator_ != nullptr, false, "make a custom op creator failed.");
  for (auto &graph : func_graphs_) {
    if (ReplaceSubgraphWithCustom(graph, use_origin_config) != RET_OK) {
      MS_LOG(ERROR) << "replace subgraph with custom node failed.";
      return false;
    }
  }
  if (RemoveTemporaryFiles() != RET_OK) {
    MS_LOG(ERROR) << "remove temporarily generated files failed.";
    return false;
  }
  return true;
}
REG_PASS(DpicoPass, dpico::DpicoPass)
}  // namespace dpico
}  // namespace mindspore
namespace mindspore::registry {
const std::vector<std::string> schedule_pipe = {"ConstFoldPass",         "ToNCHWFormat", "DpicoPreprocessPass",
                                                "DecreaseTransposeAlgo", "DumpGraph",    "DpicoPass",
                                                "ToNHWCFormat"};
REG_SCHEDULED_PASS(POSITION_BEGIN, schedule_pipe)
}  // namespace mindspore::registry
