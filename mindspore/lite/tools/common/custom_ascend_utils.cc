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
#include "mindspore/lite/tools/common/custom_ascend_utils.h"
#include "mindspore/lite/src/common/log_util.h"
#include "mindspore/core/utils/ms_utils_secure.h"
#include "mindspore/lite/tools/common/func_graph_utils.h"
#include "mindspore/core/ops/tuple_get_item.h"
#include "mindspore/lite/src/common/common.h"
#include "mindspore/lite/tools/optimizer/common/gllo_utils.h"

namespace mindspore {
namespace {
constexpr auto kCustomPrimTypeACL = "ACL";
constexpr auto kCustomNodeName = "custom_0";
constexpr auto kFuncType = "func_type";
constexpr auto kUniqueName = "uniq_name";
}  // namespace

ParameterPtr CustomAscendUtils::CreateOmParameter(const FuncGraphPtr &func_graph, const Buffer &om_data,
                                                  const std::string &graph_name) {
  MS_CHECK_TRUE_MSG(func_graph != nullptr, nullptr, "func_graph is nullptr.");
  ParameterPtr om_parameter = func_graph->add_parameter();
  MS_CHECK_TRUE_MSG(om_parameter != nullptr, nullptr, "om_parameter is nullptr.");
  om_parameter->set_name(graph_name);
  om_parameter->debug_info()->set_name(graph_name);

  auto type_ptr = TypeIdToType(kNumberTypeUInt8);
  MS_CHECK_TRUE_MSG(type_ptr != nullptr, nullptr, "type_ptr is nullptr.");
  ShapeVector shape_vector = {static_cast<int64_t>(om_data.DataSize())};
  auto abstract_tensor = std::make_shared<abstract::AbstractTensor>(type_ptr, shape_vector);
  MS_CHECK_TRUE_MSG(abstract_tensor != nullptr, nullptr, "abstract_tensor is nullptr.");
  om_parameter->set_abstract(abstract_tensor);

  auto param_value =
    std::make_shared<tensor::Tensor>(kNumberTypeUInt8, ShapeVector({static_cast<int64_t>(om_data.DataSize())}));
  MS_CHECK_TRUE_MSG(param_value != nullptr, nullptr, "param_value is nullptr.");
  auto tensor_data = param_value->data_c();
  MS_CHECK_TRUE_MSG(tensor_data != nullptr, nullptr, "New Tensor failed.");
  if (param_value->Size() < om_data.DataSize()) {
    MS_LOG(ERROR) << "Dst buff size  " << param_value->Size() << " should be greater than src buff size "
                  << om_data.DataSize();
    return nullptr;
  }
  if (common::huge_memcpy(reinterpret_cast<uint8_t *>(tensor_data), param_value->Size(),
                          reinterpret_cast<const uint8_t *>(om_data.Data()), om_data.DataSize()) != EOK) {
    MS_LOG(ERROR) << "Memcpy om data failed.";
    return nullptr;
  }
  om_parameter->set_default_param(param_value);
  return om_parameter;
}

bool CustomAscendUtils::SetCustomOutputs(const FuncGraphPtr &func_graph, const CNodePtr &custom_node) {
  if (outputs_.size() == 1) {
    auto abstract_tensor = FuncGraphUtils::GetAbstractFromNode(outputs_[0]);
    if (abstract_tensor == nullptr) {
      MS_LOG(ERROR) << "Abstract_tensor is nullptr.";
      return false;
    }
    auto abstract_tensor_clone = abstract_tensor->Clone();
    abstract_tensor_clone->set_name(abstract_tensor->name());
    custom_node->set_abstract(abstract_tensor_clone);
    return true;
  } else {
    AbstractBasePtrList abstract_list;
    for (size_t j = 0; j < outputs_.size(); j++) {
      auto abstract_tensor = FuncGraphUtils::GetAbstractFromNode(outputs_[j]);
      if (abstract_tensor == nullptr) {
        MS_LOG(ERROR) << "Abstract tensor is nullptr for output " << j;
        return false;
      }
      auto abstract_tensor_clone = abstract_tensor->Clone();
      abstract_tensor_clone->set_name(abstract_tensor->name());
      (void)abstract_list.emplace_back(abstract_tensor_clone);
    }
    custom_node->set_abstract(std::make_shared<abstract::AbstractTuple>(abstract_list));
  }
  return true;
}

void CustomAscendUtils::SetZeroValueRefDatas(const ops::PrimitiveCPtr &primc,
                                             const std::vector<std::pair<std::string, tensor::TensorPtr>> &ref_infos) {
  ValuePtrList value_ptr_list;
  for (const auto &item : ref_infos) {
    (void)value_ptr_list.emplace_back(MakeValue<std::string>(item.first));
    (void)value_ptr_list.emplace_back(MakeValue(static_cast<uint64_t>(item.second->data_type())));
    (void)value_ptr_list.emplace_back(MakeValue(item.second->shape_c()));
  }
  (void)primc->AddAttr(lite::kNameAttrZeroValRefDatas, MakeValue(value_ptr_list));
}

bool CustomAscendUtils::GetZeroValueRefDatas(const ops::PrimitiveCPtr &primc,
                                             std::vector<std::pair<std::string, tensor::TensorPtr>> *ref_infos) {
  auto attr = primc->GetAttr(lite::kNameAttrZeroValRefDatas);
  if (attr == nullptr) {
    MS_LOG(INFO) << "Not found attr " << lite::kNameAttrZeroValRefDatas << " in custom node";
    return true;
  }
  auto value_ptr_list = GetValue<ValuePtrList>(attr);
  constexpr size_t every_item_size = 3;
  if (value_ptr_list.size() % every_item_size != 0) {
    MS_LOG(ERROR) << "Attr " << lite::kNameAttrZeroValRefDatas << " item count should be multiply of 3, but got "
                  << value_ptr_list.size();
    return false;
  }
  for (size_t i = 0; i < value_ptr_list.size(); i += every_item_size) {
    auto param_name = GetValue<std::string>(value_ptr_list[i]);
    auto data_type = static_cast<TypeId>(GetValue<uint64_t>(value_ptr_list[i + 1]));
    auto param_shape = GetValue<ShapeVector>(value_ptr_list[i + 2]);
    auto tensor = std::make_shared<tensor::Tensor>(data_type, param_shape);
    ref_infos->push_back(std::make_pair(param_name, tensor));
  }
  return true;
}

CNodePtr CustomAscendUtils::CreateCustomNode(const FuncGraphPtr &func_graph, const ParameterPtr &om_parameter,
                                             const std::map<std::string, ValuePtr> &attr_map,
                                             const std::vector<std::string> &ref_datas) {
  MS_CHECK_TRUE_MSG(func_graph != nullptr, nullptr, "func_graph is nullptr.");
  auto prim = std::make_shared<mindspore::ops::Custom>();
  MS_CHECK_TRUE_MSG(prim != nullptr, nullptr, "New custom op failed.");
  prim->set_type(kCustomPrimTypeACL);
  auto prim_c = prim->GetPrim();
  auto graph_input = func_graph->get_inputs();
  std::vector<std::pair<std::string, tensor::TensorPtr>> zeor_val_ref_infos;
  for (auto &item : func_graph->parameters()) {
    if (item && item->cast<ParameterPtr>() != nullptr) {
      auto parameter = item->cast<ParameterPtr>();
      auto param_name = parameter->name();
      if (std::find(ref_datas.begin(), ref_datas.end(), param_name) != ref_datas.end()) {
        auto value = parameter->default_param();
        if (value == nullptr) {
          continue;
        }
        auto tensor = value->cast<std::shared_ptr<tensor::Tensor>>();
        if (tensor == nullptr) {
          continue;
        }
        if (IsParameterValueZero(tensor)) {
          zeor_val_ref_infos.push_back(std::make_pair(param_name, tensor));
        } else {
          graph_input.push_back(item);
        }
      }
    }
  }
  CNodePtr custom_node = func_graph->NewCNode(prim_c, graph_input);
  MS_CHECK_TRUE_MSG(custom_node != nullptr, nullptr, "Custom cnode failed.");
  custom_node->set_fullname_with_scope(kCustomNodeName);
  custom_node->add_input(om_parameter);

  if (!SetCustomOutputs(func_graph, custom_node)) {
    MS_LOG(ERROR) << "Set custom outputs failed.";
    return nullptr;
  }
  SetCustomAttrs(prim, attr_map);
  SetZeroValueRefDatas(prim_c, zeor_val_ref_infos);
  (void)prim->AddAttr(lite::kNameAttrRefDatas, api::MakeValue(ref_datas));
  return custom_node;
}

void CustomAscendUtils::SetCustomAttrs(const std::shared_ptr<ops::Custom> &prim,
                                       const std::map<std::string, ValuePtr> &attr_map) {
  std::string output_dim_str;
  for (const auto &item : outputs_) {
    auto shape = opt::GetAnfNodeOutputShape(item.first, item.second);
    output_dim_str += std::to_string(shape.size()) + ",";
    for (const auto &val : shape) {
      output_dim_str += std::to_string(val) + ",";
    }
  }
  std::vector<uint8_t> output_dim_char(output_dim_str.begin(), output_dim_str.end());
  std::map<std::string, std::vector<uint8_t>> attrs = {{lite::kOutputShapes, output_dim_char}};
  prim->set_attr(attrs);
  prim->AddAttr(kFuncType, api::MakeValue<std::string>("acl_build"));
  prim->AddAttr(kUniqueName, api::MakeValue<std::string>(lite::kNameCustomAscend));
  auto prim_c = prim->GetPrim();
  for (auto &attr : attr_map) {
    prim_c->AddAttr(attr.first, attr.second);
  }
}

CNodePtr CustomAscendUtils::CreateMakeTupleGraphOutput(const FuncGraphPtr &func_graph, const CNodePtr &custom_node) {
  std::vector<CNodePtr> node_list;
  for (size_t j = 0; j < outputs_.size(); ++j) {
    auto tuple_get_item_prim_ptr = std::make_shared<ops::TupleGetItem>();
    if (tuple_get_item_prim_ptr == nullptr) {
      MS_LOG(ERROR) << "New TupleGetItem failed for output " << j;
      return nullptr;
    }
    auto tuple_get_item_prim_ptr_c = tuple_get_item_prim_ptr->GetPrim();
    auto tuple_get_item_prim = NewValueNode(tuple_get_item_prim_ptr_c);
    MS_CHECK_TRUE_MSG(tuple_get_item_prim != nullptr, nullptr, "item_prim is nullptr.");
    auto get_item_value = NewValueNode(MakeValue<int64_t>(j));
    MS_CHECK_TRUE_MSG(get_item_value != nullptr, nullptr, "item_value is nullptr.");
    AnfNodePtrList inputs{tuple_get_item_prim, custom_node, get_item_value};
    CNodePtr get_item_cnode = func_graph->NewCNode(inputs);
    if (get_item_cnode == nullptr) {
      MS_LOG(ERROR) << "New get item cnode failed for output " << j;
      return nullptr;
    }
    get_item_cnode->set_fullname_with_scope(custom_node->fullname_with_scope() + "_getitem_" + std::to_string(j));
    node_list.emplace_back(get_item_cnode);
  }
  auto make_tuple_val_node = NewValueNode(prim::kPrimMakeTuple);
  MS_CHECK_TRUE_MSG(make_tuple_val_node != nullptr, nullptr, "New make tuple val node failed.");
  AnfNodePtrList new_inputs = {make_tuple_val_node};
  new_inputs.insert(new_inputs.end(), node_list.begin(), node_list.end());
  auto make_tuple_cnode = func_graph->NewCNode(new_inputs);
  MS_CHECK_TRUE_MSG(make_tuple_cnode != nullptr, nullptr, "New make tuple cnode failed.");
  return make_tuple_cnode;
}

bool CustomAscendUtils::ModifyGraphByCustomNode(const FuncGraphPtr &func_graph, const CNodePtr &custom_node) {
  auto manager = Manage(func_graph, true);
  if (manager == nullptr) {
    MS_LOG(ERROR) << "Manager is nullptr";
    return false;
  }
  AnfNodePtr return_input = func_graph->output();
  MS_CHECK_TRUE_MSG(return_input != nullptr, lite::RET_ERROR, "return input is nullptr.");
  if (outputs_.size() == 1) {
    if (!manager->Replace(return_input, custom_node)) {
      MS_LOG(ERROR) << "Replace node failed.";
      return false;
    }
  } else {
    auto make_tuple_node = CreateMakeTupleGraphOutput(func_graph, custom_node);
    MS_CHECK_TRUE_MSG(make_tuple_node != nullptr, lite::RET_ERROR, "Create make tuple cnode failed.");
    if (!manager->Replace(return_input, make_tuple_node)) {
      MS_LOG(ERROR) << "Replace node failed for outputs of graph.";
      return false;
    }
  }
  std::vector<AnfNodePtr> new_parameters;
  auto node_users = manager->node_users();
  for (auto &item : func_graph->parameters()) {
    auto parameter = item->cast<ParameterPtr>();
    if (!parameter) {
      continue;
    }
    if (!parameter->has_default()) {
      new_parameters.push_back(parameter);
    } else {
      auto users = node_users.find(item);
      if (!users->second.empty()) {
        new_parameters.push_back(item);
      }
    }
  }
  manager->SetParameters(func_graph, new_parameters);
  MS_LOG(DEBUG) << "Modify graph by custom node success.";
  return true;
}

bool CustomAscendUtils::IsParameterValueZero(const tensor::TensorPtr &tensor) {
  if (tensor == nullptr) {
    return false;
  }
  auto size = tensor->Size();
  auto count = size / sizeof(uint64_t);
  auto data_u8 = reinterpret_cast<uint8_t *>(tensor->data_c());
  if (data_u8 == nullptr) {
    return false;
  }
  auto data_u64 = reinterpret_cast<uint64_t *>(tensor->data_c());
  for (size_t i = 0; i < count; i++) {
    if (data_u64[i] != 0) {
      return false;
    }
  }
  for (size_t i = count * sizeof(uint64_t); i < size; i++) {
    if (data_u8[i] != 0) {
      return false;
    }
  }
  return true;
}

bool CustomAscendUtils::CreateCustomFuncGraph(const FuncGraphPtr &func_graph, const Buffer &model_cache,
                                              const std::string &graph_name,
                                              const std::map<std::string, ValuePtr> &attr_map,
                                              const std::vector<std::string> &ref_datas) {
  CustomAscendUtils utils;
  utils.outputs_ = opt::GetNodeInputs(func_graph->get_return());
  auto om_parameter = CreateOmParameter(func_graph, model_cache, graph_name);
  if (om_parameter == nullptr) {
    MS_LOG(ERROR) << "Create custom parameter failed";
    return false;
  }
  auto cnode = utils.CreateCustomNode(func_graph, om_parameter, attr_map, ref_datas);
  if (cnode == nullptr) {
    MS_LOG(ERROR) << "Create custom cnode failed";
    return false;
  }
  if (!utils.ModifyGraphByCustomNode(func_graph, cnode)) {
    return false;
  }
  return true;
}

CNodePtr CustomAscendUtils::GetCustomNode(const FuncGraphPtr &func_graph) {
  if (func_graph == nullptr) {
    return nullptr;
  }
  auto nodes = func_graph->TopoSort(func_graph->get_return());
  if (nodes.empty()) {
    MS_LOG(WARNING) << "There are no nodes in the graph";
    return nullptr;
  }
  CNodePtr custom_node = nullptr;
  size_t cnode_count = 0;
  for (auto &node : nodes) {
    auto cnode = node->cast<CNodePtr>();
    if (!cnode || !AnfUtils::IsRealKernel(cnode)) {
      continue;
    }
    std::string kernel_name = AnfUtils::GetCNodeName(cnode);
    if (kernel_name != lite::kNameCustomAscend) {
      return nullptr;
    }
    cnode_count += 1;
    if (cnode_count > 1) {
      MS_LOG(ERROR) << "Only support one " << lite::kNameCustomAscend << " node, but got " << kernel_name << ", node "
                    << cnode->fullname_with_scope();
      return nullptr;
    }
    auto inputs = cnode->inputs();
    if (inputs.size() < 1) {
      MS_LOG(ERROR) << "Custom node input count " << inputs.size() << " invalid";
      return nullptr;
    }
    custom_node = cnode;
  }
  return custom_node;
}

bool CustomAscendUtils::IsCustomFuncGraph(const FuncGraphPtr &func_graph) {
  return GetCustomNode(func_graph) != nullptr;
}

bool CustomAscendUtils::ParseCustomFuncGraph(const FuncGraphPtr &func_graph, tensor::TensorPtr *model_cache,
                                             std::string *graph_name, std::map<std::string, ValuePtr> *attr_map,
                                             std::vector<std::pair<std::string, tensor::TensorPtr>> *ref_datas) {
  MS_ERROR_IF_NULL_W_RET_VAL(func_graph, false);
  MS_ERROR_IF_NULL_W_RET_VAL(model_cache, false);
  MS_ERROR_IF_NULL_W_RET_VAL(graph_name, false);
  MS_ERROR_IF_NULL_W_RET_VAL(attr_map, false);
  MS_ERROR_IF_NULL_W_RET_VAL(ref_datas, false);
  auto custom_node = GetCustomNode(func_graph);
  if (custom_node == nullptr) {
    MS_LOG(ERROR) << "Cannot find Custom node, or other real node find in the graph";
    return false;
  }
  auto inputs = custom_node->inputs();
  if (inputs.size() < 1) {
    MS_LOG(ERROR) << "Custom node input count " << inputs.size() << " invalid";
    return false;
  }
  auto input_last = *inputs.rbegin();
  if (!input_last) {
    MS_LOG(ERROR) << "Custom node last input is nullptr";
    return false;
  }
  auto tensor = FuncGraphUtils::GetParameterConstValue(input_last);
  if (tensor == nullptr) {
    MS_LOG(ERROR) << "Failed to cast parameter value to Tensor";
    return false;
  }
  if (tensor->data_c() == nullptr || tensor->Size() == 0) {
    MS_LOG(ERROR) << "Custom node tensor data is empty";
    return false;
  }
  auto prim = GetValueNode<PrimitivePtr>(custom_node->input(0));
  if (!prim) {
    MS_LOG(ERROR) << "Primitive of cnode " << custom_node->fullname_with_scope() << " cannot be nullptr";
    return false;
  }
  for (auto &attr : prim->attrs()) {
    (*attr_map)[attr.first] = attr.second;
  }
  auto attr_ref_datas = prim->GetAttr(lite::kNameAttrRefDatas);
  if (attr_ref_datas) {
    auto ref_datas_names = GetValue<std::vector<std::string>>(attr_ref_datas);
    std::vector<std::pair<std::string, tensor::TensorPtr>> zero_val_ref_infos;
    if (!GetZeroValueRefDatas(prim, &zero_val_ref_infos)) {
      MS_LOG(ERROR) << "Failed to get zero value ref data";
      return false;
    }
    auto parameters = func_graph->parameters();
    std::vector<AnfNodePtr> new_parameters = func_graph->get_inputs();
    for (auto &ref_name : ref_datas_names) {
      auto it = std::find_if(zero_val_ref_infos.begin(), zero_val_ref_infos.end(),
                             [&ref_name](const auto &info) { return info.first == ref_name; });
      if (it != zero_val_ref_infos.end()) {
        ref_datas->push_back(std::make_pair(ref_name, it->second));
        continue;
      }
      auto p_it = std::find_if(parameters.begin(), parameters.end(),
                               [&ref_name](auto &item) { return item->fullname_with_scope() == ref_name; });
      if (p_it == parameters.end() || *p_it == nullptr) {
        MS_LOG(ERROR) << "Cannot find RefData parameter " << ref_name;
        return false;
      }
      auto ref_tensor = FuncGraphUtils::GetParameterConstValue(*p_it);
      if (ref_tensor == nullptr) {
        MS_LOG(ERROR) << "Failed to find tensor value for parameter " << ref_name;
        return false;
      }
      ref_datas->push_back(std::make_pair(ref_name, ref_tensor));
    }
  }
  *model_cache = tensor;
  *graph_name = input_last->fullname_with_scope();
  return true;
}
}  // namespace mindspore
