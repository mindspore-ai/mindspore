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

#include "plugin/device/ascend/kernel/tbe/tbe_adapter.h"

#include <map>
#include <unordered_set>
#include <string>
#include <memory>
#include <vector>
#include <set>
#include <algorithm>
#include <unordered_map>
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "kernel/oplib/opinfo.h"
#include "frontend/parallel/ops_info/ops_utils.h"
#include "plugin/device/ascend/kernel/tbe/tbe_dynamic_shape_util.h"
#include "plugin/device/ascend/kernel/tbe/tbe_json/tbe_json_utils.h"
#include "include/common/utils/json_operation_utils.h"
#include "utils/ms_context.h"
#include "include/common/utils/utils.h"

namespace mindspore {
namespace kernel {
namespace tbe {
namespace {
constexpr int kInvalid = -1;
constexpr int kFloat = 0;
constexpr int kFloat16 = 1;
constexpr int kInt8 = 2;
constexpr int kInt32 = 3;
constexpr int kUint8 = 4;
constexpr int kInt64 = 9;
constexpr int kUint64 = 10;
constexpr int kBool = 12;
constexpr int kbFloat = 27;
constexpr size_t kC0 = 16;
constexpr size_t kShapeIndex0 = 0;
constexpr size_t kShapeIndex1 = 1;
constexpr size_t kShapeIndex2 = 2;
constexpr size_t kShapeIndex3 = 3;
constexpr size_t kShapeIndex4 = 4;
int TypeStrToDstType(const std::string &type_str) {
  std::unordered_map<std::string, int> type_name_type_id_map = {
    {"Float", kFloat}, {"Float32", kFloat}, {"Float16", kFloat16}, {"Int8", kInt8}, {"Int32", kInt32},
    {"UInt8", kUint8}, {"Int64", kInt64},   {"UInt64", kUint64},   {"Bool", kBool}};
  auto iter = type_name_type_id_map.find(type_str);
  if (iter != type_name_type_id_map.end()) {
    return iter->second;
  } else {
    MS_LOG(INFO) << "Error type str is invailed: " << type_str;
  }
  return kInvalid;
}
}  // namespace

bool TbeAdapter::IsSpecialFusionComputeNode(const std::vector<mindspore::AnfNodePtr> &compute_nodes) {
  auto result = std::find_if(compute_nodes.begin(), compute_nodes.end(), [](const auto &it) {
    auto op_name = common::AnfAlgo::GetCNodeName(it);
    return (op_name == kConv2DBackpropInputDOpName || op_name == kConv2DOpName);
  });
  return result != compute_nodes.end();
}

bool TbeAdapter::GetSpecInputLayers(const std::string &op_name, const std::vector<mindspore::AnfNodePtr> &reorder_layer,
                                    std::map<const AnfNodePtr, FusionDataType> *spec_data_input) {
  if ((op_name == kReluGradV2OpName || op_name == kAddNOpName || op_name == kTensorAddOpName) &&
      reorder_layer.empty()) {
    MS_LOG(WARNING) << "Fusion error: node(" << op_name << " )'s input is null. ";
    return false;
  }
  if (op_name == kReluGradV2OpName) {
    (*spec_data_input)[reorder_layer[0]] = kFusionReLUGradV2;
  } else if (op_name == kAddNOpName) {
    for (const auto &it : reorder_layer) {
      (*spec_data_input)[it] = kFusionAddN;
    }
  } else if (op_name == kTensorAddOpName) {
    (*spec_data_input)[reorder_layer[0]] = kFusionAdd;
  }
  return true;
}

void TbeAdapter::FusionDescJsonPass(const AnfNodePtr &node, nlohmann::json *output_desc,
                                    const std::map<const AnfNodePtr, tbe::FusionDataType> &spec_data_input) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(output_desc);
  tbe::FusionDataType fusion_data_type =
    spec_data_input.find(node) != spec_data_input.end() ? spec_data_input.at(node) : tbe::kFusionNormal;
  std::vector<size_t> shape = (*output_desc)["shape"];
  if ((fusion_data_type == kFusionAddN || fusion_data_type == kFusionAdd) && shape.size() == kShape5dDims) {
    std::vector<size_t> spec_shape = {};
    spec_shape.emplace_back(shape[kShapeIndex0]);
    spec_shape.emplace_back(shape[kShapeIndex1]);
    spec_shape.emplace_back(shape[kShapeIndex2] * shape[kShapeIndex3]);
    spec_shape.emplace_back(shape[kShapeIndex4]);
    (*output_desc)["shape"] = spec_shape;
  } else if (fusion_data_type == kFusionReLUGradV2) {
    std::vector<size_t> spec_shape = {};
    spec_shape.emplace_back(shape[kShapeIndex0]);
    spec_shape.emplace_back(shape[kShapeIndex1]);
    spec_shape.emplace_back(shape[kShapeIndex2] * shape[kShapeIndex3]);
    spec_shape.emplace_back(kC0);
    (*output_desc)["shape"] = spec_shape;
    (*output_desc)["data_type"] = "bool";
  }
}

std::string TbeAdapter::GetRealOpType(const std::string &origin_type) {
  const std::map<std::string, std::string> buffer_fussion_op_map = {
    {parallel::DEPTHWISE_CONV2D_NATIVE, parallel::DEPTHWISE_CONV2D}};
  auto iter = buffer_fussion_op_map.find(origin_type);
  return (iter != buffer_fussion_op_map.end()) ? iter->second : origin_type;
}

std::string TbeAdapter::FormatPass(const std::string &format, const size_t &origin_shape_size) {
  if (format == kOpFormat_DEFAULT) {
    auto ms_context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(ms_context);
    if (ms_context->get_param<int>(MS_CTX_EXECUTION_MODE) == kPynativeMode) {
      return kOpFormat_NCHW;
    }
    return origin_shape_size == kNCHWShapeSize ? kOpFormat_NCHW : kOpFormat_ND;
  }
  return format;
}

bool TbeAdapter::GetSpecDataInput(const FusionScopeInfo &fusion_scope_info,
                                  std::map<const AnfNodePtr, tbe::FusionDataType> *spec_data_input) {
  MS_EXCEPTION_IF_NULL(spec_data_input);
  auto input_nodes = fusion_scope_info.input_nodes;
  auto compute_nodes = fusion_scope_info.compute_nodes;
  for (const auto &compute_node : compute_nodes) {
    MS_EXCEPTION_IF_NULL(compute_node);
    std::vector<mindspore::AnfNodePtr> layer = {};
    std::vector<mindspore::AnfNodePtr> reorder_layer = {};
    auto op_name = common::AnfAlgo::GetCNodeName(compute_node);
    auto ccompute_node = compute_node->cast<CNodePtr>();
    if (ccompute_node == nullptr) {
      MS_LOG(WARNING) << "Fusion error: fusion compute node must be cnode, but the node is "
                      << compute_node->DebugString();
      return false;
    }
    for (size_t i = 1; i < ccompute_node->inputs().size(); ++i) {
      auto input = ccompute_node->input(i);
      auto find_iter = std::find(input_nodes.begin(), input_nodes.end(), input);
      if (find_iter != input_nodes.end()) {
        layer.emplace_back((*find_iter));
      }
    }
    InputOrderPass<AnfNodePtr>(compute_node, layer, &reorder_layer);
    if (IsSpecialFusionComputeNode(compute_nodes)) {
      if (!GetSpecInputLayers(op_name, reorder_layer, spec_data_input)) {
        return false;
      }
    }
  }
  return true;
}
bool TbeAdapter::IsPlaceHolderInput(const AnfNodePtr &node, const OpIOInfoPtr &input_ptr) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(input_ptr);
  static std::set<std::string> node_set = {kDynamicRNNOpName, kDynamicGRUV2OpName};
  auto cnode_name = common::AnfAlgo::GetCNodeName(node);
  if (node_set.find(cnode_name) == node_set.end()) {
    return false;
  }
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (common::AnfAlgo::HasNodeAttr(kAttrPlaceHolderIndex, cnode)) {
    auto none_index = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(node, kAttrPlaceHolderIndex);
    return find(none_index.begin(), none_index.end(), input_ptr->index()) != none_index.end();
  } else {
    MS_LOG(EXCEPTION) << "Cnode: " << cnode_name << " doesn't has attribute placeholder_index.";
  }
}
void TbeAdapter::CastAttrJsonPrePass(const AnfNodePtr &anf_node, std::vector<OpAttrPtr> *op_info_attrs,
                                     nlohmann::json *attrs_json) {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_EXCEPTION_IF_NULL(attrs_json);
  if (common::AnfAlgo::GetCNodeName(anf_node) != kCastOpName) {
    return;
  }
  if (op_info_attrs->size() != 1) {
    MS_LOG(INFO) << "cast node should has dst_type attr";
    return;
  }
  auto attr_name = (*op_info_attrs)[0]->name();
  TensorTypePtr type_ptr;
  auto build_info = AnfAlgo::GetSelectKernelBuildInfo(anf_node);
  if (build_info) {
    type_ptr = std::make_shared<TensorType>(TypeIdToType(AnfAlgo::GetOutputDeviceDataType(anf_node, 0)));
  } else {
    // use infer shape during select kernel
    type_ptr = std::make_shared<TensorType>(TypeIdToType(common::AnfAlgo::GetOutputInferDataType(anf_node, 0)));
  }
  MS_EXCEPTION_IF_NULL(type_ptr);
  auto type_element = type_ptr->element();
  MS_EXCEPTION_IF_NULL(type_element);
  auto dtype = type_element->ToString();
  auto dst_type_value = TypeStrToDstType(dtype);
  nlohmann::json attr_obj;
  attr_obj[kJValue] = dst_type_value;
  attr_obj[kJValid] = true;
  attr_obj[kJDtype] = "int32";
  attr_obj[kJName] = attr_name;
  attrs_json->push_back(attr_obj);
  op_info_attrs->clear();
}

void TbeAdapter::CastAttrJsonPost(const AnfNodePtr &anf_node, nlohmann::json *attrs_json) {
  if (common::AnfAlgo::GetCNodeName(anf_node) != kCastOpName) {
    return;
  }
  std::map<int, std::string> dst_type_map{{0, "float32"}, {1, "float16"}, {2, "int8"}, {3, "int32"},
                                          {4, "uint8"},   {10, "uint64"}, {12, "bool"}};
  auto type_id = GetJsonValue<int>(attrs_json->at(0), kJValue);
  auto iter = dst_type_map.find(type_id);
  if (iter != dst_type_map.end()) {
    attrs_json->at(0)[kJValue] = iter->second;
  } else {
    MS_LOG(EXCEPTION) << "Invalid type: " << type_id;
  }
}
void TbeAdapter::LayerNormAttrJsonPost(const AnfNodePtr &anf_node, nlohmann::json *attrs_json) {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_EXCEPTION_IF_NULL(attrs_json);
  if (common::AnfAlgo::GetCNodeName(anf_node) == parallel::LAYER_NORM) {
    nlohmann::json new_attrs_json;
    for (auto &json_item : *attrs_json) {
      if (GetJsonValue<std::string>(json_item, kJName) == kAttrEpsilon) {
        continue;
      }
      new_attrs_json.push_back(json_item);
    }
    *attrs_json = new_attrs_json;
  }
}
}  // namespace tbe
}  // namespace kernel
}  // namespace mindspore
