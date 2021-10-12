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

#include "backend/kernel_compiler/tbe/tbe_adapter.h"

#include <map>
#include <unordered_set>
#include <string>
#include <memory>
#include <vector>
#include <set>
#include <algorithm>
#include <unordered_map>
#include "backend/session/anf_runtime_algorithm.h"
#include "backend/kernel_compiler/oplib/opinfo.h"
#include "frontend/parallel/ops_info/ops_utils.h"
#include "backend/kernel_compiler/tbe/tbe_dynaminc_shape_util.h"
#include "backend/kernel_compiler/tbe/tbe_json/tbe_json_utils.h"
#include "utils/json_operation_utils.h"
#include "utils/ms_context.h"

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
constexpr int kUint64 = 10;
constexpr int kBool = 12;
constexpr size_t kC0 = 16;
constexpr size_t kShapeIndex0 = 0;
constexpr size_t kShapeIndex1 = 1;
constexpr size_t kShapeIndex2 = 2;
constexpr size_t kShapeIndex3 = 3;
constexpr size_t kShapeIndex4 = 4;
int TypeStrToDstType(const std::string &type_str) {
  std::unordered_map<std::string, int> type_name_type_id_map = {
    {"Float", kFloat}, {"Float32", kFloat}, {"Float16", kFloat16}, {"Int8", kInt8},
    {"Int32", kInt32}, {"UInt8", kUint8},   {"UInt64", kUint64},   {"Bool", kBool}};
  auto iter = type_name_type_id_map.find(type_str);
  if (iter != type_name_type_id_map.end()) {
    return iter->second;
  } else {
    MS_LOG(INFO) << "Error type str is invailed: " << type_str;
  }
  return kInvalid;
}
}  // namespace
std::unordered_set<std::string> TbeAdapter::input_order_adjusted_ops_ = {kConv2DBackpropInputOpName,
                                                                         kConv2DBackpropFilterOpName,
                                                                         kLogSoftmaxGradOpName,
                                                                         kLayerNormGradOpName,
                                                                         kLayerNormXBackpropOpName,
                                                                         kLayerNormXBackpropV2OpName,
                                                                         kLayerNormBetaGammaBackpropOpName,
                                                                         kMinimumGradOpName,
                                                                         kMaximumGradOpName,
                                                                         kApplyCenteredRMSPropOpName};

std::map<std::string, FAttrsPass> TbeAdapter::build_json_attr_pass_map_ = {{"Cast", TbeAdapter::CastAttrJsonPass}};

bool TbeAdapter::DynamicInputAdjusted(const std::shared_ptr<AnfNode> &anf_node,
                                      std::vector<std::vector<nlohmann::json>> const &inputs_list,
                                      nlohmann::json *inputs_json) {
  if (!AnfAlgo::IsNodeDynamicShape(anf_node) && !AnfAlgo::IsDynamicShape(anf_node)) {
    return false;
  }
  auto op_name = AnfAlgo::GetCNodeName(anf_node);
  if (op_name == kConv2DBackpropInputOpName) {
    // process dynamic Conv2DBackpropInput, tbe kernel input is x, input_size and dout
    inputs_json->push_back(inputs_list[kIndex2]);
    inputs_json->push_back(inputs_list[kIndex1]);
    inputs_json->push_back(inputs_list[kIndex0]);
    return true;
  }
  if (op_name == kConv2DBackpropFilterOpName) {
    // process dynamic Conv2DBackpropFilter, tbe kernel input is filter_size, x and dout
    inputs_json->push_back(inputs_list[kIndex1]);
    inputs_json->push_back(inputs_list[kIndex2]);
    inputs_json->push_back(inputs_list[kIndex0]);
    return true;
  }
  return false;
}

void TbeAdapter::InputOrderPass(const std::shared_ptr<AnfNode> &anf_node,
                                std::vector<std::vector<nlohmann::json>> const &inputs_list,
                                nlohmann::json *inputs_json) {
  MS_EXCEPTION_IF_NULL(inputs_json);
  if (DynamicInputAdjusted(anf_node, inputs_list, inputs_json)) {
    return;
  }
  auto op_name = AnfAlgo::GetCNodeName(anf_node);
  if (input_order_adjusted_ops_.find(op_name) == input_order_adjusted_ops_.end()) {
    (void)std::copy(inputs_list.begin(), inputs_list.end(), std::back_inserter((*inputs_json)));
  } else {
    if (op_name == "MinimumGrad" || op_name == "MaximumGrad") {
      inputs_json->push_back(inputs_list[kIndex2]);
      inputs_json->push_back(inputs_list[kIndex0]);
      inputs_json->push_back(inputs_list[kIndex1]);
      for (size_t i = 3; i < inputs_list.size(); ++i) {
        inputs_json->push_back(inputs_list[i]);
      }
    } else if (op_name == "ApplyCenteredRMSProp") {
      // Parameter order of ApplyCenteredRMSProp's TBE implementation is different from python API, so map
      // TBE parameter to correspond python API parameter by latter's index using hardcode
      inputs_json->push_back(inputs_list[kIndex0]);
      inputs_json->push_back(inputs_list[kIndex1]);
      inputs_json->push_back(inputs_list[kIndex2]);
      inputs_json->push_back(inputs_list[kIndex3]);
      inputs_json->push_back(inputs_list[kIndex5]);
      inputs_json->push_back(inputs_list[kIndex6]);
      inputs_json->push_back(inputs_list[kIndex7]);
      inputs_json->push_back(inputs_list[kIndex8]);
      inputs_json->push_back(inputs_list[kIndex4]);
    } else {
      inputs_json->push_back(inputs_list[kIndex1]);
      inputs_json->push_back(inputs_list[kIndex0]);
      for (size_t i = 2; i < inputs_list.size(); ++i) {
        inputs_json->push_back(inputs_list[i]);
      }
    }
  }
}

void TbeAdapter::FusionInputOrderPass(const std::shared_ptr<AnfNode> &anf_node,
                                      const std::vector<nlohmann::json> &inputs_list,
                                      std::vector<nlohmann::json> *inputs_json) {
  MS_EXCEPTION_IF_NULL(inputs_json);
  if (DynamicInputAdjusted(anf_node, inputs_list, inputs_json)) {
    return;
  }
  auto op_name = AnfAlgo::GetCNodeName(anf_node);
  if (input_order_adjusted_ops_.find(op_name) == input_order_adjusted_ops_.end()) {
    (void)std::copy(inputs_list.begin(), inputs_list.end(), std::back_inserter((*inputs_json)));
  } else {
    if (op_name == "MinimumGrad" || op_name == "MaximumGrad") {
      inputs_json->emplace_back(inputs_list[2]);
      inputs_json->emplace_back(inputs_list[0]);
      inputs_json->emplace_back(inputs_list[1]);
      for (size_t i = 3; i < inputs_list.size(); ++i) {
        inputs_json->emplace_back(inputs_list[i]);
      }
    } else {
      inputs_json->emplace_back(inputs_list[1]);
      inputs_json->emplace_back(inputs_list[0]);
      for (size_t i = 2; i < inputs_list.size(); ++i) {
        inputs_json->emplace_back(inputs_list[i]);
      }
    }
  }
}

void TbeAdapter::FusionDataOrderPass(const std::string &op_name, const std::vector<AnfNodePtr> &data_layer,
                                     std::vector<AnfNodePtr> *reorder_data_layer) {
  MS_EXCEPTION_IF_NULL(reorder_data_layer);
  if (input_order_adjusted_ops_.find(op_name) == input_order_adjusted_ops_.end()) {
    (void)std::copy(data_layer.begin(), data_layer.end(), std::back_inserter((*reorder_data_layer)));
  } else {
    if (op_name == "MinimumGrad" || op_name == "MaximumGrad") {
      (void)reorder_data_layer->emplace_back(data_layer[kIndex2]);
      (void)reorder_data_layer->emplace_back(data_layer[kIndex0]);
      (void)reorder_data_layer->emplace_back(data_layer[kIndex1]);
      for (size_t i = 3; i < data_layer.size(); ++i) {
        (void)reorder_data_layer->emplace_back(data_layer[i]);
      }
    } else {
      (void)reorder_data_layer->emplace_back(data_layer[kIndex1]);
      (void)reorder_data_layer->emplace_back(data_layer[kIndex0]);
      for (size_t i = 2; i < data_layer.size(); ++i) {
        reorder_data_layer->emplace_back(data_layer[i]);
      }
    }
  }
}

bool TbeAdapter::RunAttrPass(const mindspore::AnfNodePtr &anf_node, const std::vector<OpAttrPtr> &op_info_attrs,
                             nlohmann::json *attrs_json) {
  MS_EXCEPTION_IF_NULL(attrs_json);
  auto cnode_name = AnfAlgo::GetCNodeName(anf_node);
  auto FPass = build_json_attr_pass_map_.find(cnode_name);
  if (FPass != build_json_attr_pass_map_.end()) {
    FPass->second(anf_node, op_info_attrs, attrs_json);
    return true;
  }
  return false;
}

void TbeAdapter::MaxiOrMinimumGradAttrJsonPass(const AnfNodePtr &anf_node,
                                               const std::vector<std::shared_ptr<OpAttr>> &op_info_attrs,
                                               nlohmann::json *attrs_json) {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_EXCEPTION_IF_NULL(attrs_json);
  auto attr_num = op_info_attrs.size();
  auto primitive = AnfAlgo::GetCNodePrimitive(anf_node);
  MS_EXCEPTION_IF_NULL(primitive);
  for (size_t i = 0; i < attr_num; i++) {
    nlohmann::json attr_obj;
    MS_EXCEPTION_IF_NULL(op_info_attrs[i]);
    std::string attr_name = op_info_attrs[i]->name();
    auto value = primitive->GetAttr(attr_name);
    if (value != nullptr) {
      bool attr_value = GetValue<bool>(value);
      attr_obj["value"] = attr_value;
      attr_obj["valid"] = true;
    } else {
      attr_obj["valid"] = false;
    }
    attr_obj["name"] = attr_name;
    attrs_json->push_back(attr_obj);
  }
  MS_LOG(INFO) << "MaxiOrMinimumGradAttrJsonPass done.";
}

void TbeAdapter::CastAttrJsonPass(const mindspore::AnfNodePtr &anf_node,
                                  const std::vector<std::shared_ptr<mindspore::kernel::OpAttr>> &op_info_attrs,
                                  nlohmann::json *attrs_json) {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_EXCEPTION_IF_NULL(attrs_json);
  if (op_info_attrs.size() != 1) {
    MS_LOG(INFO) << "cast node should has dst_type attr";
    return;
  }
  auto attr_name = op_info_attrs[0]->name();
  auto type_ptr = std::make_shared<TensorType>(TypeIdToType(AnfAlgo::GetOutputDeviceDataType(anf_node, 0)));
  MS_EXCEPTION_IF_NULL(type_ptr);
  auto type_element = type_ptr->element();
  MS_EXCEPTION_IF_NULL(type_element);
  auto dtype = type_element->ToString();
  auto dst_type_value = TypeStrToDstType(dtype);
  nlohmann::json attr_obj;
  attr_obj["value"] = dst_type_value;
  attr_obj["valid"] = true;
  attr_obj["name"] = attr_name;
  attrs_json->push_back(attr_obj);
}

void TbeAdapter::GenTopKV2IndicesTensorInfo(const std::shared_ptr<mindspore::AnfNode> &anf_node,
                                            size_t real_input_index, std::vector<nlohmann::json> *input_list,
                                            mindspore::kernel::kCreaterType creater_type) {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_EXCEPTION_IF_NULL(input_list);
  auto input_x_shape = AnfAlgo::GetOutputInferShape(anf_node, 0);
  size_t last_dim = input_x_shape[input_x_shape.size() - 1];
  std::vector<size_t> tensor_shape = {last_dim};
  std::vector<size_t> tensor_origin_shape = {last_dim};
  std::string tensor_format = AnfAlgo::GetInputFormat(anf_node, static_cast<const size_t &>(real_input_index));
  if (tensor_format == kOpFormat_DEFAULT) {
    tensor_format = kOpFormat_NCHW;
  }
  std::string tensor_origin_format = kOpFormat_NCHW;
  std::string tensor_dtype = "float16";
  nlohmann::json input_desc_json;
  input_desc_json["dtype"] = tensor_dtype;
  input_desc_json["name"] = AnfAlgo::GetCNodeName(anf_node);
  input_desc_json["ori_shape"] = tensor_origin_shape;
  input_desc_json["ori_format"] = tensor_origin_format;
  input_desc_json["shape"] = tensor_shape;
  if (creater_type == OP_SELECT_FORMAT) {
    input_desc_json["format"] = tensor_origin_format;
  } else {
    input_desc_json["format"] = tensor_format;
  }
  input_desc_json["valid"] = true;
  input_list->emplace_back(input_desc_json);
}

bool TbeAdapter::IsSpecialFusionComputeNode(const std::vector<mindspore::AnfNodePtr> &compute_nodes) {
  auto result = std::find_if(compute_nodes.begin(), compute_nodes.end(), [](const auto &it) {
    auto op_name = AnfAlgo::GetCNodeName(it);
    return (op_name == kConv2DBackpropInputOpName || op_name == kConv2DOpName);
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
  static std::map<std::string, std::string> buffer_fussion_op_map = {
    {parallel::DEPTHWISE_CONV2D_NATIVE, parallel::DEPTHWISE_CONV2D}};
  auto iter = buffer_fussion_op_map.find(origin_type);
  return (iter != buffer_fussion_op_map.end()) ? iter->second : origin_type;
}

std::string TbeAdapter::GetNodeFusionType(const mindspore::CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  auto node_type = AnfAlgo::GetCNodeName(cnode);
  static std::map<std::string, std::string> fusion_type_map = {{kConv2DOpName, "Convolution"},
                                                               {kBNTrainingReduceOpName, "bn_reduce"},
                                                               {kBNTrainingUpdateOpName, "bn_update"},
                                                               {kReluV2OpName, "ElemWise"},
                                                               {kTensorAddOpName, "ElemWise"},
                                                               {kConv2DBackpropInputOpName, "Conv2d_backprop_input"},
                                                               {kConv2DBackpropFilterOpName, "Conv2d_backprop_filter"},
                                                               {kDepthwiseConv2dNativeOpName, "DepthwiseConvolution"},
                                                               {kAddNOpName, "ElemWise"},
                                                               {kReluGradV2OpName, "ElemWise"},
                                                               {kRealDivOpName, "ElemWise"},
                                                               {kBiasAddOpName, "BiasAdd"}};
  auto find = fusion_type_map.find(node_type);
  if (find == fusion_type_map.end()) {
    MS_LOG(INFO) << "Fusion warning: get node fusion type failed from lists, origin node type: " << node_type;
    auto op_info = mindspore::kernel::tbe::TbeDynamicShapeUtil::FindOp(node_type, cnode);
    MS_EXCEPTION_IF_NULL(op_info);
    return op_info->fusion_type();
  } else {
    return find->second;
  }
}

std::string TbeAdapter::FormatPass(const std::string &format, const size_t &origin_shape_size) {
  if (format == kOpFormat_DEFAULT) {
    auto ms_context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(ms_context);
    if (ms_context->get_param<int>(MS_CTX_EXECUTION_MODE) == kPynativeMode) {
      return kOpFormat_NCHW;
    }
    return origin_shape_size == kNCHWShapeSize ? kOpFormat_NCHW : kOpFormat_ND;
  } else if (format == kOpFormat_FRAC_Z) {
    return kOpFormat_FRACTAL_Z;
  } else {
    return format;
  }
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
    auto op_name = AnfAlgo::GetCNodeName(compute_node);
    auto ccompute_node = compute_node->cast<CNodePtr>();
    if (ccompute_node == nullptr) {
      MS_LOG(WARNING) << "Fusion error: fusion compute node must be cnode, but the node is "
                      << ccompute_node->DebugString();
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
  auto cnode_name = AnfAlgo::GetCNodeName(node);
  if (node_set.find(cnode_name) == node_set.end()) {
    return false;
  }
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (AnfAlgo::HasNodeAttr("placeholder_index", cnode)) {
    auto none_index = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(node, "placeholder_index");
    return find(none_index.begin(), none_index.end(), input_ptr->index()) != none_index.end();
  } else {
    MS_LOG(EXCEPTION) << "Cnode: " << cnode_name << "doesn't has attribute placeholder_index.";
  }
}
void TbeAdapter::CastAttrJsonPrePass(const AnfNodePtr &anf_node, std::vector<OpAttrPtr> *op_info_attrs,
                                     nlohmann::json *attrs_json) {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_EXCEPTION_IF_NULL(attrs_json);
  if (AnfAlgo::GetCNodeName(anf_node) != kCastOpName) {
    return;
  }
  if (op_info_attrs->size() != 1) {
    MS_LOG(INFO) << "cast node should has dst_type attr";
    return;
  }
  auto attr_name = (*op_info_attrs)[0]->name();
  auto type_ptr = std::make_shared<TensorType>(TypeIdToType(AnfAlgo::GetOutputDeviceDataType(anf_node, 0)));
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
  if (AnfAlgo::GetCNodeName(anf_node) != kCastOpName) {
    return;
  }
  std::map<int, std::string> dst_type_map{{0, "float32"}, {1, "float16"}, {2, "int8"}, {3, "int32"},
                                          {4, "uint8"},   {10, "uint64"}, {12, "bool"}};
  auto type_id = GetJsonValue<int>(attrs_json->at(0), kJValue);
  auto iter = dst_type_map.find(type_id);
  if (iter != dst_type_map.end()) {
    attrs_json->at(0)[kJValue] = iter->second;
  } else {
    MS_LOG(EXCEPTION) << "Invalid type:" << type_id;
  }
}
void TbeAdapter::LayerNormAttrJsonPost(const AnfNodePtr &anf_node, nlohmann::json *attrs_json) {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_EXCEPTION_IF_NULL(attrs_json);
  if (AnfAlgo::GetCNodeName(anf_node) == parallel::LAYER_NORM) {
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
