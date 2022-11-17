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

#include "src/custom_creator.h"
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <utility>
#include "include/api/format.h"
#include "ops/make_tuple.h"
#include "common/anf_util.h"
#include "common/op_attr.h"
#include "common/op_enum.h"
#include "common/string_util.h"
#include "src/mapper_config_parser.h"
#include "src/om_generator.h"
#include "src/graph_split_api.h"
#include "ops/tuple_get_item.h"
#include "third_party/securec/include/securec.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
namespace mindspore {
namespace dpico {
namespace {
const int kMaxRoiFrameCnt = 300;
const std::map<mapper::OpDataType, TypeId> kDataTypeMap = {
  {mapper::OpDataType::OP_DTYPE_S8, kNumberTypeInt8},     {mapper::OpDataType::OP_DTYPE_U8, kNumberTypeUInt8},
  {mapper::OpDataType::OP_DTYPE_S16, kNumberTypeInt16},   {mapper::OpDataType::OP_DTYPE_U16, kNumberTypeUInt16},
  {mapper::OpDataType::OP_DTYPE_S32, kNumberTypeInt32},   {mapper::OpDataType::OP_DTYPE_U32, kNumberTypeUInt32},
  {mapper::OpDataType::OP_DTYPE_F16, kNumberTypeFloat16}, {mapper::OpDataType::OP_DTYPE_F32, kNumberTypeFloat32},
};

int CheckOmDataCoreInfo(const mapper::DataCoreInfo &data_core_info) {
  MS_CHECK_TRUE_MSG(!data_core_info.name.empty(), RET_ERROR, "output name is empty.");
  MS_CHECK_TRUE_MSG(kDataTypeMap.find(data_core_info.type) != kDataTypeMap.end(), RET_ERROR,
                    "unsupported data type, op data type is " << data_core_info.type);
  MS_CHECK_TRUE_MSG(!data_core_info.shape.empty(), RET_ERROR,
                    "output shape shouldn't be empty. " << data_core_info.name);
  return RET_OK;
}

bool CheckInputCNodeSize(const api::CNodePtr &cnode, size_t *next_idx) {
  int target_valid_input_size = 1;
  for (size_t i = 1; i < cnode->inputs().size(); i++) {
    auto input_node = cnode->input(i);
    if (api::utils::isa<api::ParameterPtr>(cnode->input(i))) {
      auto param_node = input_node->cast<api::ParameterPtr>();
      if (param_node != nullptr && !param_node->has_default()) {  // graph input
        target_valid_input_size--;
      }
    } else if (api::utils::isa<api::CNodePtr>(input_node)) {
      *next_idx = i;
      target_valid_input_size--;
    }
  }
  return target_valid_input_size == 0;
}

bool IsCorrespondOutput(const api::AnfNodePtr &node, const std::string &target_name) {
  if (node->fullname_with_scope() == target_name) {
    return true;
  }
  auto cnode = node->cast<api::CNodePtr>();
  if (cnode == nullptr) {
    MS_LOG(INFO) << "cur node isn't a cnode, will stop recursive search. " << node->fullname_with_scope();
    return false;
  }
  MS_CHECK_TRUE_MSG(cnode->inputs().size() >= kInputIndex2, false,
                    node->fullname_with_scope() << " inputs size " << cnode->inputs().size() << " is invalid.");
  size_t next_idx = 1;
  if (!CheckInputCNodeSize(cnode, &next_idx)) {
    return false;
  } else {
    return IsCorrespondOutput(cnode->input(next_idx), target_name);
  }
}
}  // namespace

api::CNodePtr CustomOpCreator::CreateCustomOp(const api::FuncGraphPtr &func_graph, Subgraph *subgraph,
                                              const ModelCoreInfoPtr &om_model_info) {
  MS_CHECK_TRUE_MSG(func_graph != nullptr && subgraph != nullptr && om_model_info != nullptr, nullptr,
                    "obtain nullptr input parameter.");
  auto om_parameter = CreateOmParameter(func_graph, om_model_info);
  MS_CHECK_TRUE_MSG(om_parameter != nullptr, nullptr, "create om parameter failed.");
  if (SetSubgraphInputOutputDims(subgraph, func_graph, om_model_info) != RET_OK) {
    MS_LOG(ERROR) << "set subgraph input output dims failed.";
    return nullptr;
  }

  auto prim = api::MakeShared<ops::Custom>();
  MS_CHECK_TRUE_MSG(prim != nullptr, nullptr, "new Custom failed");
  prim->set_type("DPICO");

  // set op inputs
  auto subgraph_inputs = GetSubgraphInputs(*subgraph, func_graph);
  MS_CHECK_TRUE_MSG(!subgraph_inputs.empty(), nullptr,
                    "get subgraph inputs failed. subgraph id is " << subgraph->graph_id);
  auto custom_cnode = func_graph->NewCNode(prim, subgraph_inputs);
  MS_CHECK_TRUE_MSG(custom_cnode != nullptr, nullptr, "new cnode error");
  custom_cnode->set_fullname_with_scope(kCustomName + std::to_string(custom_id_));
  custom_cnode->add_input(om_parameter);

  // build op outputs && replace origin subgraph with custom node
  if (SetCustomOutputs(func_graph, subgraph, custom_cnode, om_model_info) != RET_OK) {
    MS_LOG(ERROR) << "set supported custom op outputs failed";
    return nullptr;
  }

  // set attr for custom op.
  if (SetCustomAttrs(*subgraph, func_graph, prim) != RET_OK) {
    MS_LOG(ERROR) << "set supported custom op attrs failed";
    return nullptr;
  }
  ++custom_id_;
  return custom_cnode;
}

api::ParameterPtr CustomOpCreator::CreateOmParameter(const api::FuncGraphPtr &func_graph,
                                                     const ModelCoreInfoPtr &om_model_info) {
  MS_CHECK_TRUE_MSG(func_graph != nullptr && om_model_info != nullptr, nullptr, "obtain nullptr input parameter.");
  MS_CHECK_TRUE_MSG(om_model_info->modelSize != 0, nullptr, "om model size equals 0.");
  auto om_parameter = func_graph->add_parameter();
  MS_CHECK_TRUE_MSG(om_parameter != nullptr, nullptr, "new parameter failed.");
  om_parameter->set_name("DPICO_om_data");
  ShapeVector shape_vector = {static_cast<int64_t>(om_model_info->modelSize)};
  auto abstract_tensor = api::MakeShared<api::AbstractTensor>(kNumberTypeUInt8, shape_vector);
  MS_CHECK_TRUE_MSG(abstract_tensor != nullptr, nullptr, "abstract_tensor is nullptr.");
  om_parameter->set_abstract(abstract_tensor);

  auto tensor_info =
    api::MakeShared<api::Tensor>(kNumberTypeUInt8, ShapeVector({static_cast<int64_t>(om_model_info->modelSize)}));
  MS_CHECK_TRUE_MSG(tensor_info != nullptr, nullptr, "tensor_info is nullptr.");
  auto tensor_data = tensor_info->data();
  MS_CHECK_TRUE_MSG(tensor_data != nullptr, nullptr, "new api::Tensor failed.");
  MS_CHECK_TRUE_MSG(tensor_info->Size() != 0, nullptr, "tensor size shouldn't be 0");
  if (memcpy_s(tensor_data, tensor_info->Size(), om_model_info->modelBuffer, om_model_info->modelSize) != EOK) {
    MS_LOG(ERROR) << "memcpy_s failed.";
    return nullptr;
  }
  om_parameter->set_default_param(tensor_info);

  return om_parameter;
}

STATUS CustomOpCreator::SetSubgraphInputOutputDims(Subgraph *subgraph, const api::FuncGraphPtr &func_graph,
                                                   const ModelCoreInfoPtr &om_model_info) {
  MS_CHECK_TRUE_MSG(subgraph != nullptr && func_graph != nullptr && om_model_info != nullptr, RET_ERROR,
                    "obtain nullptr input parameter.");
  auto subgraph_inputs = GetSubgraphInputs(*subgraph, func_graph);
  for (const auto &input_info : om_model_info->inputInfos) {
    if (CheckOmDataCoreInfo(input_info) != RET_OK) {
      MS_LOG(ERROR) << "om input info is invalid.";
      return RET_ERROR;
    }
    for (const auto &node : subgraph_inputs) {
      auto node_name = RemoveSpecifiedChar(node->fullname_with_scope(), '\0');
      if (CheckPrimitiveType(node, api::MakeShared<ops::Custom>())) {
        node_name = GetCustomOutputName(node);
        MS_CHECK_TRUE_MSG(!node_name.empty(), RET_ERROR,
                          "get custom node output name failed." << node->fullname_with_scope());
      }
      auto input_info_name = RemoveSpecifiedChar(input_info.name, '\0');
      if (node_name == input_info_name) {  // DetectionOutput network has extra input, will filter it.
        ShapeVector ori_shape_vector;
        if (GetAnfNodeOutputShape(node, &ori_shape_vector)) {
          MS_LOG(ERROR) << "get " << node->fullname_with_scope() << " output shape failed.";
          return RET_ERROR;
        }
        if (ori_shape_vector.size() != input_info.shape.size()) {
          MS_LOG(ERROR) << node_name << "'s input shape size " << ori_shape_vector.size()
                        << " is not equal to om input shape size " << input_info.shape.size();
          return RET_ERROR;
        }
        ShapeVector shape_vector(input_info.shape.begin(), input_info.shape.end());
        subgraph->inputs_dims.push_back(shape_vector);
        break;
      }
    }
  }
  MS_CHECK_TRUE_MSG(!subgraph->inputs_dims.empty(), RET_ERROR, "subgraph input dims shouldn't be empty.");
  for (const auto &output_info : om_model_info->outputInfos) {
    if (CheckOmDataCoreInfo(output_info) != RET_OK) {
      MS_LOG(ERROR) << "om output info is invalid.";
      return RET_ERROR;
    }
    ShapeVector shape_vector(output_info.shape.begin(), output_info.shape.end());
    subgraph->outputs_dims.push_back(shape_vector);
  }
  MS_CHECK_TRUE_MSG(!subgraph->outputs_dims.empty(), RET_ERROR, "subgraph output dims shouldn't be empty.");
  return RET_OK;
}

STATUS CustomOpCreator::SetCustomAttrs(const Subgraph &subgraph, const api::FuncGraphPtr &func_graph,
                                       const api::SharedPtr<ops::Custom> &prim) {
  MS_CHECK_TRUE_MSG(func_graph != nullptr && prim != nullptr, RET_ERROR, "obtain nullptr input parameter.");
  std::map<std::string, std::vector<uint8_t>> custom_attrs;
  // add "input_shape " attr
  std::string input_dims_attr_value_str;
  for (auto &item : subgraph.inputs_dims) {
    input_dims_attr_value_str += std::to_string(item.size()) + ",";
    for (auto &v : item) {
      input_dims_attr_value_str += std::to_string(v) + ",";
    }
  }
  std::vector<uint8_t> input_dims_attr_value(input_dims_attr_value_str.begin(), input_dims_attr_value_str.end());
  (void)custom_attrs.insert(std::make_pair(kInputsShape, input_dims_attr_value));

  // add "output_shape " attr
  std::string output_dims_attr_value_str;
  for (auto &item : subgraph.outputs_dims) {
    output_dims_attr_value_str += std::to_string(item.size()) + ",";
    for (auto &v : item) {
      output_dims_attr_value_str += std::to_string(v) + ",";
    }
  }
  std::vector<uint8_t> output_dims_attr_value(output_dims_attr_value_str.begin(), output_dims_attr_value_str.end());
  (void)custom_attrs.insert(std::make_pair(kOutputsShape, output_dims_attr_value));

  // add "outputs_format" attr
  std::string output_format_attr_str;
  for (auto &item : subgraph.outputs_format) {
    output_format_attr_str += std::to_string(item) + ",";
  }
  std::vector<uint8_t> output_format_attr_value(output_format_attr_str.begin(), output_format_attr_str.end());
  (void)custom_attrs.insert(std::make_pair(kOutputsFormat, output_format_attr_value));

  // add om net type attr
  auto om_net_type_str = std::to_string(static_cast<int>(subgraph.om_net_type));
  std::vector<uint8_t> om_net_type_value(om_net_type_str.begin(), om_net_type_str.end());
  (void)custom_attrs.insert(std::make_pair(kNetType, om_net_type_value));

  // add max_roi_fram_cnt attr
  if (subgraph.om_net_type == OmNetType::kRoi) {
    auto max_roi_frame_cnt_str = std::to_string(kMaxRoiFrameCnt);
    std::vector<uint8_t> max_roi_frame_cnt_value(max_roi_frame_cnt_str.begin(), max_roi_frame_cnt_str.end());
    (void)custom_attrs.insert(std::make_pair("max_roi_frame_cnt", max_roi_frame_cnt_value));
  }
  prim->set_attr(custom_attrs);
  return RET_OK;
}

STATUS CustomOpCreator::SetCustomOutputs(const api::FuncGraphPtr &func_graph, Subgraph *subgraph,
                                         const api::CNodePtr &custom_cnode, const ModelCoreInfoPtr &om_model_info) {
  MS_CHECK_TRUE_MSG(subgraph != nullptr, RET_ERROR, "subgraph is nullptr.");
  auto manager = func_graph->manager();
  MS_CHECK_TRUE_MSG(manager != nullptr, RET_ERROR, "funcgraph manager is nullptr.");
  auto subgraph_outputs = GetSubgraphOutputs(*subgraph, manager);
  MS_CHECK_TRUE_MSG(subgraph->outputs_format.size() == subgraph_outputs.size(), RET_ERROR,
                    "subgraph's outputs_format may be empty, which should be pre-determined.");
  if (om_model_info->outputInfos.size() < subgraph_outputs.size()) {
    MS_LOG(ERROR) << "om output info size:" << om_model_info->outputInfos.size()
                  << " is less than subgraph outputs size:" << subgraph_outputs.size();
    return RET_ERROR;
  }
  std::vector<std::string> custom_outputs_names;  // used for anf exporter
  if (om_model_info->outputInfos.size() == 1) {
    if (SetCustomSingleOutput(func_graph, subgraph, custom_cnode, om_model_info, &custom_outputs_names) != RET_OK) {
      MS_LOG(ERROR) << "set custom single output failed. " << custom_cnode->fullname_with_scope();
      return RET_ERROR;
    }
  } else {
    if (SetCustomMultiOutput(func_graph, subgraph, custom_cnode, om_model_info, &custom_outputs_names) != RET_OK) {
      MS_LOG(ERROR) << "set custom multi output failed. " << custom_cnode->fullname_with_scope();
      return RET_ERROR;
    }
  }
  custom_cnode->AddAttr(kOutputsNames, api::MakeValue(custom_outputs_names));
  return RET_OK;
}

STATUS CustomOpCreator::SetCustomSingleOutput(const api::FuncGraphPtr &func_graph, Subgraph *subgraph,
                                              const api::CNodePtr &custom_cnode,
                                              const std::shared_ptr<mapper::ModelCoreInfo> &om_model_info,
                                              std::vector<std::string> *custom_outputs_names) {
  auto manager = func_graph->manager();
  MS_CHECK_TRUE_MSG(manager != nullptr, RET_ERROR, "funcgraph manager is nullptr.");
  auto subgraph_outputs = GetSubgraphOutputs(*subgraph, manager);
  auto output_info = om_model_info->outputInfos.at(0);
  if (CheckOmDataCoreInfo(output_info) != RET_OK) {
    MS_LOG(ERROR) << "om output info is invalid.";
    return RET_ERROR;
  }
  ShapeVector shape_vector(output_info.shape.begin(), output_info.shape.end());
  auto abstract_tensor = CreateTensorAbstract(shape_vector, kDataTypeMap.at(output_info.type));
  MS_CHECK_TRUE_MSG(abstract_tensor != nullptr, RET_ERROR, "abstract_tensor is nullptr.");
  custom_cnode->set_abstract(abstract_tensor);
  auto output_name = RemoveSpecifiedChar(output_info.name, '\0');
  custom_outputs_names->push_back(output_name);
  subgraph->cnodes.clear();
  subgraph->cnodes.push_back(custom_cnode);
  if (!manager->Replace(subgraph_outputs.at(0), custom_cnode)) {
    MS_LOG(ERROR) << "replace node failed";
    return RET_ERROR;
  }
  auto image_lists = MapperConfigParser::GetInstance()->GetImageLists();
  auto origin_node_name = subgraph_outputs.at(0)->fullname_with_scope();
  if (image_lists.find(origin_node_name) != image_lists.end() &&
      origin_node_name != output_name) {  // custom op could be a supported subgraph's input
    (void)MapperConfigParser::GetInstance()->AddImageList(output_name, image_lists.at(origin_node_name));
  }
  return RET_OK;
}

STATUS CustomOpCreator::SetCustomMultiOutput(const api::FuncGraphPtr &func_graph, Subgraph *subgraph,
                                             const api::CNodePtr &custom_cnode,
                                             const std::shared_ptr<mapper::ModelCoreInfo> &om_model_info,
                                             std::vector<std::string> *custom_outputs_names) {
  auto manager = func_graph->manager();
  MS_CHECK_TRUE_MSG(manager != nullptr, RET_ERROR, "funcgraph manager is nullptr.");
  auto subgraph_outputs = GetSubgraphOutputs(*subgraph, manager);
  MS_ASSERT(subgraph->outputs_format.size() == subgraph_outputs.size());
  MS_ASSERT(om_model_info->outputInfos.size() >= subgraph_outputs.size());
  api::AbstractBasePtrList abstract_list;
  api::CNodePtrList subgraph_new_cnodes = {custom_cnode};
  auto output_formats = subgraph->outputs_format;
  subgraph->outputs_format.resize(om_model_info->outputInfos.size(), static_cast<int>(NCHW));
  size_t has_replace_num = 0;
  for (size_t i = 0; i < om_model_info->outputInfos.size(); i++) {
    auto output_info = om_model_info->outputInfos.at(i);
    if (CheckOmDataCoreInfo(output_info) != RET_OK) {
      MS_LOG(ERROR) << "om output info is invalid.";
      return RET_ERROR;
    }
    ShapeVector shape_vector(output_info.shape.begin(), output_info.shape.end());
    auto abstract_tensor = CreateTensorAbstract(shape_vector, kDataTypeMap.at(output_info.type));
    MS_CHECK_TRUE_MSG(abstract_tensor != nullptr, RET_ERROR, "abstract_tensor is nullptr.");
    (void)abstract_list.emplace_back(abstract_tensor);
    auto tuple_get_item_prim_ptr = api::MakeShared<ops::TupleGetItem>();
    MS_CHECK_TRUE_MSG(tuple_get_item_prim_ptr != nullptr, RET_ERROR, "tuple_get_item_prim_ptr is nullptr.");
    auto tuple_get_item_prim = NewValueNode(tuple_get_item_prim_ptr);
    auto get_item_value = NewValueNode(api::MakeValue<int64_t>(i));
    api::AnfNodePtrList inputs{tuple_get_item_prim, custom_cnode, get_item_value};
    api::CNodePtr get_item_cnode = func_graph->NewCNode(inputs);
    MS_CHECK_TRUE_MSG(get_item_cnode != nullptr, RET_ERROR, "get_item_cnode is nullptr.");
    get_item_cnode->set_fullname_with_scope(custom_cnode->fullname_with_scope() + "_getitem_" + std::to_string(i));
    auto output_name = RemoveSpecifiedChar(output_info.name, '\0');
    custom_outputs_names->push_back(output_name);
    subgraph_new_cnodes.push_back(get_item_cnode);
    if (has_unsupported_) {
      auto ori_node_iter = std::find_if(  // extra or inconsistent output will be found.
        subgraph_outputs.begin(), subgraph_outputs.end(),
        [output_name](const api::AnfNodePtr &anf_node) { return IsCorrespondOutput(anf_node, output_name); });
      if (ori_node_iter == subgraph_outputs.end()) {
        continue;
      }
      has_replace_num++;
      get_item_cnode->set_fullname_with_scope((*ori_node_iter)->fullname_with_scope());
      subgraph->outputs_format[i] = output_formats[ori_node_iter - subgraph_outputs.begin()];
      if (!manager->Replace(*ori_node_iter, get_item_cnode)) {
        MS_LOG(ERROR) << "replace node failed. " << get_item_cnode->fullname_with_scope();
        return RET_ERROR;
      }
      continue;
    }
    get_item_cnode->set_fullname_with_scope(output_name);
    // the whole network is not segmented
    if (i < subgraph_outputs.size()) {
      subgraph->outputs_format[i] = output_formats[i];
      if (!manager->Replace(subgraph_outputs[i], get_item_cnode)) {
        MS_LOG(ERROR) << "replace node failed. " << get_item_cnode->fullname_with_scope();
        return RET_ERROR;
      }
    } else {
      auto return_cnode = func_graph->get_return();
      if (CheckPrimitiveType(return_cnode->input(1), api::MakeShared<ops::MakeTuple>())) {
        manager->AddEdge(return_cnode->input(1), get_item_cnode);
      } else {
        manager->AddEdge(return_cnode, get_item_cnode);
      }
    }
    has_replace_num++;
  }
  if (has_replace_num < subgraph_outputs.size()) {
    MS_LOG(ERROR) << "origin outputs haven't been all replaced.";
    return RET_ERROR;
  }
  subgraph->cnodes = subgraph_new_cnodes;
  auto abstract_tuple = api::MakeShared<api::AbstractTuple>(abstract_list);
  MS_CHECK_TRUE_MSG(abstract_tuple != nullptr, RET_ERROR, "abstract_tuple is nullptr.");
  custom_cnode->set_abstract(abstract_tuple);
  return RET_OK;
}
}  // namespace dpico
}  // namespace mindspore
