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

#include "backend/kernel_compiler/tbe/tbe_json/tbe_json_creator.h"
#include <memory>
#include <map>
#include <utility>
#include <algorithm>
#include "backend/session/anf_runtime_algorithm.h"
#include "backend/kernel_compiler/common_utils.h"
#include "backend/kernel_compiler/tbe/tbe_adapter.h"
#include "backend/kernel_compiler/tbe/tbe_convert_utils.h"
#include "backend/kernel_compiler/tbe/tbe_dynaminc_shape_util.h"
#include "utils/ms_context.h"
#include "runtime/dev.h"
#include "utils/ms_utils.h"
#include "utils/json_operation_utils.h"
#include "backend/kernel_compiler/tbe/tbe_json/tbe_json_utils.h"

namespace mindspore::kernel {

namespace {
std::unordered_map<std::string, TypeID> type_id_map = {{kVTypeInt, TypeID::kIntID},
                                                       {kVTypeInt64, TypeID::kInt64ID},
                                                       {kVTypeStr, TypeID::kStrID},
                                                       {kVTypeBool, TypeID::kBoolID},
                                                       {kVTypeFloat, TypeID::kFloatID},
                                                       {kVTypeListInt, TypeID::kListIntID},
                                                       {kVTypeListFloat, TypeID::kListFloatID},
                                                       {kVTypeListUInt64, TypeID::kListUInt64ID},
                                                       {kVTypeListListInt, TypeID::kListListIntID}};

bool ParseListIntValue(const mindspore::ValuePtr &value, std::vector<int64_t> *attr_value) {
  auto value_type = value->type();
  if (value_type == nullptr) {
    MS_LOG(ERROR) << "Value's type is null.";
    return false;
  }
  if (value_type->ToString() == kVTypeInt64) {
    attr_value->push_back(GetValue<int64_t>(value));
  } else {
    auto vec = value->isa<ValueTuple>() ? value->cast<ValueTuplePtr>()->value() : value->cast<ValueListPtr>()->value();
    if (!vec.empty()) {
      if (vec[0]->isa<Int32Imm>()) {
        std::vector<int32_t> attr_value_me = GetValue<std::vector<int32_t>>(value);
        (void)std::transform(attr_value_me.begin(), attr_value_me.end(), std::back_inserter(*attr_value),
                             [](const int &value) { return static_cast<int64_t>(value); });
      } else {
        *attr_value = GetValue<std::vector<int64_t>>(value);
      }
    }
  }
  return true;
}

bool ParseAttrValue(const std::string &type, const mindspore::ValuePtr &value, nlohmann::json *attr_obj) {
  MS_EXCEPTION_IF_NULL(attr_obj);
  if (value == nullptr) {
    MS_LOG(ERROR) << "Node's attr value is null.";
    return false;
  }
  auto result = type_id_map.find(type);
  if (result == type_id_map.end()) {
    MS_LOG(ERROR) << "Type: " << type << "not support";
    return false;
  }
  switch (result->second) {
    case TypeID::kIntID:
      (*attr_obj)[kJValue] = value->isa<Int32Imm>() ? GetValue<int>(value) : GetValue<int64_t>(value);
      break;
    case TypeID::kInt64ID:
      (*attr_obj)[kJValue] = GetValue<int64_t>(value);
      break;
    case TypeID::kStrID: {
      auto attr_value = GetValue<std::string>(value);
      (*attr_obj)[kJValue] = attr_value == kOpFormat_FRAC_Z ? kJOpFormat_FRACTAL_Z : attr_value;
      break;
    }
    case TypeID::kBoolID:
      (*attr_obj)[kJValue] = GetValue<bool>(value);
      break;
    case TypeID::kFloatID:
      (*attr_obj)[kJValue] = GetValue<float>(value);
      break;
    case TypeID::kListIntID: {
      std::vector<int64_t> attr_value;
      if (!ParseListIntValue(value, &attr_value)) {
        MS_LOG(ERROR) << "Parse list_value failed, maybe the input is a nullptr.";
        return false;
      }
      (*attr_obj)[kJValue] = attr_value;
      break;
    }
    case TypeID::kListFloatID: {
      auto value_type = value->type();
      if (value_type == nullptr) {
        MS_LOG(ERROR) << "Value's type is null.";
        return false;
      }
      (*attr_obj)[kJValue] = value_type->ToString() == kVTypeFloat ? std::vector<float>{GetValue<float>(value)}
                                                                   : GetValue<std::vector<float>>(value);
      break;
    }
    case TypeID::kListUInt64ID:
      (*attr_obj)[kJValue] = GetValue<std::vector<size_t>>(value);
      break;
    case TypeID::kListListIntID:
      (*attr_obj)[kJValue] = GetValue<std::vector<std::vector<int64_t>>>(value);
      break;
  }
  return true;
}

bool ParseAttrDefaultValue(const std::string &type, const std::string &value, nlohmann::json *attr_obj) {
  MS_EXCEPTION_IF_NULL(attr_obj);
  auto result = type_id_map.find(type);
  if (result == type_id_map.end()) {
    MS_LOG(ERROR) << "Type: " << type << "not support";
    return false;
  }
  switch (result->second) {
    case TypeID::kIntID:
      (*attr_obj)[kJValue] = std::stoi(value);
      break;
    case TypeID::kInt64ID:
      (*attr_obj)[kJValue] = std::stoll(value);
      break;
    case TypeID::kStrID:
      (*attr_obj)[kJValue] = value;
      break;
    case TypeID::kBoolID: {
      bool attr_value = false;
      std::istringstream(value) >> std::boolalpha >> attr_value;
      (*attr_obj)[kJValue] = attr_value;
      break;
    }
    case TypeID::kFloatID:
      (*attr_obj)[kJValue] = std::stof(value);
      break;
    case TypeID::kListIntID: {
      std::stringstream string_value(value);
      std::string list_elem;
      std::vector<int64_t> attrs_value;
      while (std::getline(string_value, list_elem, ',')) {
        attrs_value.push_back(std::stoi(list_elem));
      }
      (*attr_obj)[kJValue] = attrs_value;
      break;
    }
    default:
      MS_LOG(ERROR) << "Type: " << type << "not support";
      return false;
  }
  return true;
}

}  // namespace

bool TbeJsonCreator::GenComputeJson(const AnfNodePtr &anf_node, nlohmann::json *compute_json) {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_EXCEPTION_IF_NULL(compute_json);
  MS_LOG(DEBUG) << "Start.";

  if (!GenInputsJson(anf_node, compute_json)) {
    MS_LOG(ERROR) << "generate inputs json failed, node full name:" << anf_node->fullname_with_scope();
    return false;
  }
  if (!GenOutputsJson(anf_node, compute_json)) {
    MS_LOG(ERROR) << "generate outputs json failed, node full name:" << anf_node->fullname_with_scope();
    return false;
  }
  GenOutputDataDescJson(anf_node, compute_json);
  GenAttrsDescJson(anf_node, compute_json);
  GenComputeCommonJson(anf_node, compute_json);
  GenOtherJson(anf_node, compute_json);
  MS_LOG(DEBUG) << "End.";
  return true;
}

void TbeJsonCreator::GenFusionOpName(nlohmann::json *kernel_json, std::string prefix) {
  json_name_.clear();
  json_hash_ = GenJsonHash((*kernel_json));
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  json_name_ = std::move(prefix);
  auto device_id = context_ptr->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  for (auto node_json : (*kernel_json)[kJOpList]) {
    if (GetJsonValue<std::string>(node_json, kJType) != kJData) {
      json_name_.append(node_json[kJFuncName]);
      json_name_.append("_");
    }
  }
  json_name_ = json_name_ + std::to_string(json_hash_) + "_" + std::to_string(device_id);
  MS_LOG(DEBUG) << "Generate Json name: " << json_name_;
  (*kernel_json)[kJFusionOpName] = json_name_;
}

void TbeJsonCreator::DeleteDescName(nlohmann::json *desc_jsons) {
  for (auto &desc_json : (*desc_jsons)) {
    if (desc_json.is_array()) {
      for (auto &desc_item : desc_json) {
        desc_item.erase(kJName);
      }
    } else {
      desc_json.erase(kJName);
    }
  }
}

size_t TbeJsonCreator::GenJsonHash(nlohmann::json tbe_json) {
  auto &op_lists = tbe_json.at(kJOpList);
  for (auto &op : op_lists) {
    op.erase(kJName);
    op.erase(kJOriName);
    op.erase(kJPattern);
    DeleteDescName(&op.at(kJOutputDesc));
    if (op[kJType] != kJData) {
      DeleteDescName(&op.at(kJInputDesc));
    }
  }
  return std::hash<std::string>()(op_lists.dump());
}

void TbeJsonCreator::AddOpNameForComputeNode(nlohmann::json *kernel_json) {
  auto op_name = GetJsonValue<std::string>((*kernel_json), kJFusionOpName);
  for (auto &node_json : (*kernel_json).at(kJOpList)) {
    // compute node
    if (GetJsonValue<std::string>(node_json, kJType) != kJData) {
      node_json[kJOpName] = op_name;
    }
  }
}

bool TbeJsonCreator::GenAttrsJson(const AnfNodePtr &anf_node, const OpInfoPtr &op_info, nlohmann::json *attrs_json) {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_EXCEPTION_IF_NULL(op_info);
  MS_EXCEPTION_IF_NULL(attrs_json);
  auto attrs_ptr = op_info->attrs_ptr();
  if (!AttrsJsonPreProcessing(anf_node, &attrs_ptr, attrs_json)) {
    MS_LOG(EXCEPTION) << "PreProcessing node attr error, node: " << anf_node->fullname_with_scope();
  }

  std::string op_name = AnfAlgo::GetCNodeName(anf_node);
  auto primitive = AnfAlgo::GetCNodePrimitive(anf_node);
  MS_EXCEPTION_IF_NULL(primitive);
  for (const auto &attr_ptr : attrs_ptr) {
    std::string attr_name = attr_ptr->name();
    nlohmann::json attr_obj;
    attr_obj[kJName] = attr_name;
    if (primitive->GetAttr(attr_name) != nullptr) {
      if (!ParseAttrValue(attr_ptr->type(), primitive->GetAttr(attr_name), &attr_obj)) {
        MS_LOG(EXCEPTION) << "op [ " << op_info->op_name() << " ]'s attr [ " << attr_name << " ] generates failed";
      }
      attr_obj[kJValid] = true;
    } else {
      auto default_value = attr_ptr->default_value();
      if (!default_value.empty()) {
        if (!ParseAttrDefaultValue(attr_ptr->type(), default_value, &attr_obj)) {
          MS_LOG(EXCEPTION) << "op [ " << op_info->op_name() << " ]'s default attr [ " << attr_name
                            << " ] generates failed";
        }
        attr_obj[kJValid] = true;
      } else {
        MS_LOG(INFO) << "op " << op_name << "'s attr \"" << attr_name << "\" should have a default value.";
        if (!op_info->impl_path().empty() && attr_ptr->param_type() == kJParamRequred) {
          MS_LOG(EXCEPTION) << "Op name: " << op_info->op_name() << " attr: " << attr_name
                            << " is required, but not set.";
        } else {
          attr_obj[kJValid] = false;
        }
      }
    }
    (*attrs_json).push_back(attr_obj);
  }

  if (!AttrsJsonPostProcessing(anf_node, op_info, attrs_json)) {
    MS_LOG(EXCEPTION) << "PostProcessing node attr error, node: " << anf_node->fullname_with_scope();
  }
  return true;
}

bool TbeJsonCreator::GenAttrsDescJson(const AnfNodePtr &anf_node, nlohmann::json *compute_json) {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_EXCEPTION_IF_NULL(compute_json);
  auto cnode = anf_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto op_name = AnfAlgo::GetCNodeName(cnode);
  auto op_info_ptr = tbe::TbeDynamicShapeUtil::FindOp(op_name, cnode);
  nlohmann::json attrs_json;
  GenAttrsJson(cnode, op_info_ptr, &attrs_json);

  nlohmann::json attrs_desc;
  for (const auto &attr : attrs_json) {
    if (GetJsonValue<std::string>(attr, kJName) != kJIsRef && GetJsonValue<bool>(attr, kJValid)) {
      attrs_desc.push_back(attr.at(kJValue));
    }
  }
  if (!attrs_desc.empty()) {
    (*compute_json)[kJAttrDesc] = attrs_desc;
  }
  return true;
}

void TbeJsonCreator::GenComputeCommonJson(const AnfNodePtr &anf_node, nlohmann::json *compute_json) {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_EXCEPTION_IF_NULL(compute_json);
  auto cnode = anf_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto op_name = AnfAlgo::GetCNodeName(cnode);
  auto op_info_ptr = tbe::TbeDynamicShapeUtil::FindOp(op_name, cnode);
  auto func_name = op_info_ptr->kernel_name();
  (*compute_json)[kJFuncName] = func_name;
  auto python_module_path = op_info_ptr->impl_path();
  if (python_module_path.empty()) {
    python_module_path = kPyPath;
  }

  auto iter = tbe::opTypeAdapter.find(op_name);
  (*compute_json)[kJType] = (iter != tbe::opTypeAdapter.end()) ? iter->second : op_name;
  (*compute_json)[kJPyModulePath] = python_module_path;
  (*compute_json)[kJDynamicCompileStatic] = false;
  (*compute_json)[kJInt64Mode] = false;
  (*compute_json)[kJName] = cnode->fullname_with_scope();
  (*compute_json)[kJPattern] = kernel::GetFusionNameByType(AnfAlgo::GetFusionType(cnode));
  (*compute_json)[kJModuleName] = kJModuleNamePrefix + func_name;
}

// node_out_idx: node output index
// desc_output_idx: this index use to add json
void TbeJsonCreator::GenDescJson(const AnfNodePtr &anf_node, size_t node_out_idx, size_t desc_output_idx,
                                 nlohmann::json *output_desc) {
  MS_EXCEPTION_IF_NULL(anf_node);
  GenDesJsonCommon(output_desc);
  std::vector<int64_t> shape;
  std::vector<int64_t> ori_shape;
  AnfAlgo::GetRealDynamicShape(AnfAlgo::GetOutputDeviceShape(anf_node, node_out_idx), NOT_NULL(&shape));
  AnfAlgo::GetRealDynamicShape(AnfAlgo::GetOutputInferShape(anf_node, node_out_idx), NOT_NULL(&ori_shape));
  if (shape.empty()) {
    shape.emplace_back(1);
  }
  if (ori_shape.empty()) {
    ori_shape.emplace_back(1);
  }

  auto full_name = anf_node->fullname_with_scope();
  auto output_desc_name = node_out_idx > 0 ? (full_name + "_" + std::to_string(node_out_idx)) : full_name;

  // !! Note: format: only data node's output use it
  auto format = AnfAlgo::GetOutputFormat(anf_node, node_out_idx);
  format = tbe::TbeAdapter::FormatPass(format, ori_shape.size());
  auto def_format = TbeJsonUtils::IsNeedChangeDefaultFormat(anf_node) ? kOpFormat_NCDHW : kOpFormat_NCHW;
  format =
    (def_format == kOpFormat_NCDHW && k3DFormatSet.find(format) == k3DFormatSet.end()) ? kOpFormat_NCDHW : format;

  (*output_desc)[kJDataType] = tbe::TypeIdToString(AnfAlgo::GetOutputDeviceDataType(anf_node, node_out_idx));
  (*output_desc)[kJDtype] = GetJsonValue<std::string>(*output_desc, kJDataType);
  (*output_desc)[kJFormat] = format;
  (*output_desc)[kJOriFormat] = def_format;
  (*output_desc)[kJOriShape] = ori_shape;
  (*output_desc)[kJShape] = shape;
  (*output_desc)[kJName] = output_desc_name;
  // !! Note: output_index, only node's output use it
  (*output_desc)[kJOutputIndex] = desc_output_idx;
}

void TbeJsonCreator::GenDesJsonCommon(nlohmann::json *output_desc) {
  MS_EXCEPTION_IF_NULL(output_desc);
  (*output_desc)[kJL1AddrOffset] = 0;
  (*output_desc)[kJL1FusionType] = -1;
  (*output_desc)[kJL1WorkspaceSize] = -1;
  (*output_desc)[kJAddrType] = 0;
  (*output_desc)[kJSliceOffset] = nlohmann::json::array();
  (*output_desc)[kJSplitIndex] = 0;
  (*output_desc)[kJTotalShape] = nlohmann::json::array();
  (*output_desc)[kJValidShape] = nlohmann::json::array();
}
bool TbeJsonCreator::AttrsJsonPreProcessing(const AnfNodePtr &anf_node, std::vector<OpAttrPtr> *attrs_ptr,
                                            nlohmann::json *attrs_json) {
  tbe::TbeAdapter::CastAttrJsonPrePass(anf_node, attrs_ptr, attrs_json);
  return true;
}
bool TbeJsonCreator::GenOutputDataDescJson(const AnfNodePtr &anf_node, nlohmann::json *compute_json) {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_EXCEPTION_IF_NULL(compute_json);
  auto op_desc = AnfAlgo::GetOutputDataDesc(anf_node);
  // get output_data_desc from prebuild
  if (!op_desc.empty() && op_desc.at(0).find(kJListArgs) != op_desc.at(0).end()) {
    (*compute_json)[kJOutputDataDesc] = GetJsonValue<nlohmann::json>(op_desc.at(0), kJListArgs);
  } else {
    auto outputs_desc = GetJsonValue<std::vector<nlohmann::json>>(*compute_json, kJOutputDesc);
    std::vector<nlohmann::json> outputs_data_desc;
    for (auto output_desc : outputs_desc) {
      if (output_desc.find(kJOriShape) != output_desc.end()) {
        output_desc.erase(kJName);
        outputs_data_desc.push_back(output_desc);
      }
    }
    (*compute_json)[kJOutputDataDesc] = outputs_data_desc;
  }
  return true;
}

bool TbeJsonCreator::AttrsJsonPostProcessing(const AnfNodePtr &anf_node, const OpInfoPtr &op_info_ptr,
                                             nlohmann::json *attrs_json) {
  return true;
}
}  // namespace mindspore::kernel
