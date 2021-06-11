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
#include <algorithm>

#include "backend/session/anf_runtime_algorithm.h"
#include "backend/kernel_compiler/oplib/opinfo.h"

namespace mindspore {
namespace kernel {
namespace tbe {
constexpr size_t INPUT0 = 0;
constexpr size_t INPUT1 = 1;
constexpr size_t INPUT2 = 2;
constexpr size_t INPUT3 = 3;
constexpr size_t INPUT4 = 4;
constexpr size_t INPUT5 = 5;
constexpr size_t INPUT6 = 6;
constexpr size_t INPUT7 = 7;
constexpr size_t INPUT8 = 8;
constexpr size_t kInputSize3 = 3;
constexpr size_t kInputSize2 = 2;
constexpr size_t kApplyCenteredRMSPropInputSize = 9;

std::unordered_set<std::string> input_order_adjusted_ops = {
  "Conv2DBackpropInput",        "Conv2DBackpropFilter", "LogSoftmaxGrad", "LayerNormGrad",       "LayerNormXBackprop",
  "LayerNormBetaGammaBackprop", "MinimumGrad",          "MaximumGrad",    "ApplyCenteredRMSProp"};

void TbeAdapter::InputOrderPass(const std::string &op_name, std::vector<std::vector<nlohmann::json>> const &inputs_list,
                                nlohmann::json *inputs_json) {
  MS_EXCEPTION_IF_NULL(inputs_json);
  if (input_order_adjusted_ops.find(op_name) == input_order_adjusted_ops.end()) {
    (void)std::copy(inputs_list.begin(), inputs_list.end(), std::back_inserter((*inputs_json)));
  } else {
    if (op_name == "MinimumGrad" || op_name == "MaximumGrad") {
      if (inputs_list.size() < kInputSize3) {
        MS_LOG(EXCEPTION) << op_name << "'s input size " << inputs_list.size() << " is less than " << kInputSize3;
      }
      inputs_json->push_back(inputs_list[INPUT2]);
      inputs_json->push_back(inputs_list[INPUT0]);
      inputs_json->push_back(inputs_list[INPUT1]);
      for (size_t i = 3; i < inputs_list.size(); ++i) {
        inputs_json->push_back(inputs_list[i]);
      }
    } else if (op_name == "ApplyCenteredRMSProp") {
      // Parameter order of ApplyCenteredRMSProp's TBE implementation is different from python API, so map
      // TBE parameter to correspond python API parameter by latter's index using hardcode
      if (inputs_list.size() < kApplyCenteredRMSPropInputSize) {
        MS_LOG(EXCEPTION) << op_name << "'s input size " << inputs_list.size() << " is less than "
                          << kApplyCenteredRMSPropInputSize;
      }
      inputs_json->push_back(inputs_list[INPUT0]);
      inputs_json->push_back(inputs_list[INPUT1]);
      inputs_json->push_back(inputs_list[INPUT2]);
      inputs_json->push_back(inputs_list[INPUT3]);
      inputs_json->push_back(inputs_list[INPUT5]);
      inputs_json->push_back(inputs_list[INPUT6]);
      inputs_json->push_back(inputs_list[INPUT7]);
      inputs_json->push_back(inputs_list[INPUT8]);
      inputs_json->push_back(inputs_list[INPUT4]);
    } else {
      if (inputs_list.size() < kInputSize2) {
        MS_LOG(EXCEPTION) << op_name << "'s input size " << inputs_list.size() << " is less than " << kInputSize2;
      }
      inputs_json->push_back(inputs_list[1]);
      inputs_json->push_back(inputs_list[0]);
      for (size_t i = 2; i < inputs_list.size(); ++i) {
        inputs_json->push_back(inputs_list[i]);
      }
    }
  }
}

void TbeAdapter::FusionInputOrderPass(const std::string &op_name, const std::vector<nlohmann::json> &inputs_list,
                                      std::vector<nlohmann::json> *inputs_json) {
  MS_EXCEPTION_IF_NULL(inputs_json);
  if (input_order_adjusted_ops.find(op_name) == input_order_adjusted_ops.end()) {
    (void)std::copy(inputs_list.begin(), inputs_list.end(), std::back_inserter((*inputs_json)));
  } else {
    if (op_name == "MinimumGrad" || op_name == "MaximumGrad") {
      if (inputs_list.size() < kInputSize3) {
        MS_LOG(EXCEPTION) << op_name << "'s input size " << inputs_list.size() << " is less than " << kInputSize3;
      }
      inputs_json->emplace_back(inputs_list[INPUT2]);
      inputs_json->emplace_back(inputs_list[INPUT0]);
      inputs_json->emplace_back(inputs_list[INPUT1]);
      for (size_t i = 3; i < inputs_list.size(); ++i) {
        inputs_json->emplace_back(inputs_list[i]);
      }
    } else {
      if (inputs_list.size() < kInputSize2) {
        MS_LOG(EXCEPTION) << op_name << "'s input size " << inputs_list.size() << " is less than " << kInputSize2;
      }
      inputs_json->emplace_back(inputs_list[INPUT1]);
      inputs_json->emplace_back(inputs_list[INPUT0]);
      for (size_t i = 2; i < inputs_list.size(); ++i) {
        inputs_json->emplace_back(inputs_list[i]);
      }
    }
  }
}

void TbeAdapter::FusionDataOrderPass(const std::string &op_name, const std::vector<AnfNodePtr> &data_layer,
                                     std::vector<AnfNodePtr> *reorder_data_layer) {
  MS_EXCEPTION_IF_NULL(reorder_data_layer);
  if (input_order_adjusted_ops.find(op_name) == input_order_adjusted_ops.end()) {
    (void)std::copy(data_layer.begin(), data_layer.end(), std::back_inserter((*reorder_data_layer)));
  } else {
    if (op_name == "MinimumGrad" || op_name == "MaximumGrad") {
      if (data_layer.size() < kInputSize3) {
        MS_LOG(EXCEPTION) << op_name << "'s input size " << data_layer.size() << " is less than " << kInputSize3;
      }
      (void)reorder_data_layer->emplace_back(data_layer[INPUT2]);
      (void)reorder_data_layer->emplace_back(data_layer[INPUT0]);
      (void)reorder_data_layer->emplace_back(data_layer[INPUT1]);
      for (size_t i = 3; i < data_layer.size(); ++i) {
        (void)reorder_data_layer->emplace_back(data_layer[i]);
      }
    } else {
      if (data_layer.size() < kInputSize2) {
        MS_LOG(EXCEPTION) << op_name << "'s input size " << data_layer.size() << " is less than " << kInputSize2;
      }
      (void)reorder_data_layer->emplace_back(data_layer[INPUT1]);
      (void)reorder_data_layer->emplace_back(data_layer[INPUT0]);
      for (size_t i = 2; i < data_layer.size(); ++i) {
        reorder_data_layer->emplace_back(data_layer[i]);
      }
    }
  }
}

std::map<std::string, FAttrsPass> TbeAdapter::build_json_attr_pass_map_ = {
  {"MaximumGrad", TbeAdapter::MaxiOrMinimumGradAttrJsonPass},
  {"MinimumGrad", TbeAdapter::MaxiOrMinimumGradAttrJsonPass},
  {"Cast", TbeAdapter::CastAttrJsonPass}};

bool TbeAdapter::RunAttrPass(const mindspore::AnfNodePtr &anf_node,
                             const std::vector<std::shared_ptr<mindspore::kernel::OpAttr>> &op_info_attrs,
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

static int TypeStrToDstType(const std::string &type_str) {
  constexpr int kInvalid = -1;
  constexpr int kFloat = 0;
  constexpr int kFloat16 = 1;
  constexpr int kInt8 = 2;
  constexpr int kInt32 = 3;
  constexpr int kUint8 = 4;
  constexpr int kUint64 = 10;
  constexpr int kBool = 12;
  if (type_str == "Float" || type_str == "Float32") {
    return kFloat;
  } else if (type_str == "Float16") {
    return kFloat16;
  } else if (type_str == "Int8") {
    return kInt8;
  } else if (type_str == "Int32") {
    return kInt32;
  } else if (type_str == "UInt8") {
    return kUint8;
  } else if (type_str == "UInt64") {
    return kUint64;
  } else if (type_str == "Bool") {
    return kBool;
  } else {
    MS_LOG(INFO) << "Error type str is invailed: " << type_str;
  }
  return kInvalid;
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
  if (input_x_shape.empty()) {
    MS_LOG(EXCEPTION) << AnfAlgo::GetCNodeName(anf_node) << "'s output infer shape is empty.";
  }
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
}  // namespace tbe
}  // namespace kernel
}  // namespace mindspore
