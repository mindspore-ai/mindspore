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
      inputs_json->push_back(inputs_list[2]);
      inputs_json->push_back(inputs_list[0]);
      inputs_json->push_back(inputs_list[1]);
      for (size_t i = 3; i < inputs_list.size(); ++i) {
        inputs_json->push_back(inputs_list[i]);
      }
    } else if (op_name == "ApplyCenteredRMSProp") {
      // Parameter order of ApplyCenteredRMSProp's TBE implementation is different from python API, so map
      // TBE parameter to correspond python API parameter by latter's index using hardcode
      inputs_json->push_back(inputs_list[0]);
      inputs_json->push_back(inputs_list[1]);
      inputs_json->push_back(inputs_list[2]);
      inputs_json->push_back(inputs_list[3]);
      inputs_json->push_back(inputs_list[5]);
      inputs_json->push_back(inputs_list[6]);
      inputs_json->push_back(inputs_list[7]);
      inputs_json->push_back(inputs_list[8]);
      inputs_json->push_back(inputs_list[4]);
    } else {
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
  if (input_order_adjusted_ops.find(op_name) == input_order_adjusted_ops.end()) {
    (void)std::copy(data_layer.begin(), data_layer.end(), std::back_inserter((*reorder_data_layer)));
  } else {
    if (op_name == "MinimumGrad" || op_name == "MaximumGrad") {
      reorder_data_layer->emplace_back(data_layer[2]);
      reorder_data_layer->emplace_back(data_layer[0]);
      reorder_data_layer->emplace_back(data_layer[1]);
      for (size_t i = 3; i < data_layer.size(); ++i) {
        reorder_data_layer->emplace_back(data_layer[i]);
      }
    } else {
      reorder_data_layer->emplace_back(data_layer[1]);
      reorder_data_layer->emplace_back(data_layer[0]);
      for (size_t i = 2; i < data_layer.size(); ++i) {
        reorder_data_layer->emplace_back(data_layer[i]);
      }
    }
  }
}

std::map<std::string, FAttrsPass> TbeAdapter::build_json_attr_pass_map_ = {
  {"MaximumGrad", TbeAdapter::MaximumGradAttrJsonPass},
  {"MinimumGrad", TbeAdapter::MinimumGradAttrJsonPass},
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

void TbeAdapter::MaximumGradAttrJsonPass(const mindspore::AnfNodePtr &anf_node,
                                         const std::vector<std::shared_ptr<mindspore::kernel::OpAttr>> &op_info_attrs,
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
  MS_LOG(INFO) << "MaximumGradAttrJsonPass done.";
}

void TbeAdapter::MinimumGradAttrJsonPass(const mindspore::AnfNodePtr &anf_node,
                                         const std::vector<std::shared_ptr<mindspore::kernel::OpAttr>> &op_info_attrs,
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
  MS_LOG(INFO) << "MinimumGradAttrJsonPass done.";
}

static int TypeStrToDstType(const std::string &type_str) {
  int ret = -1;
  if (type_str == "Float" || type_str == "Float32") {
    ret = 0;
  } else if (type_str == "Float16") {
    ret = 1;
  } else if (type_str == "Int8") {
    ret = 2;
  } else if (type_str == "Int32") {
    ret = 3;
  } else if (type_str == "UInt8") {
    ret = 4;
  } else if (type_str == "UInt64") {
    ret = 10;
  } else if (type_str == "Bool") {
    ret = 12;
  } else {
    MS_LOG(INFO) << "Error type str is invailed: " << type_str;
  }
  return ret;
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
}  // namespace tbe
}  // namespace kernel
}  // namespace mindspore
