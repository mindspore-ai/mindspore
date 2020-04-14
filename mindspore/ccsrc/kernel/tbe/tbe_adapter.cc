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

#include "kernel/tbe/tbe_adapter.h"

#include <map>
#include <unordered_set>
#include <string>
#include <memory>
#include <vector>
#include <algorithm>

#include "session/anf_runtime_algorithm.h"
#include "kernel/oplib/opinfo.h"

namespace mindspore {
namespace kernel {
namespace tbe {
static std::map<string, string> tbe_func_adapter_map = {
  {"re_lu6", "relu6"},
  {"re_lu6_grad", "relu6_grad"},
  {"re_lu", "relu"},
  {"tensor_add", "add"},
  {"reduce_mean", "reduce_mean_d"},
  {"reduce_max", "reduce_max_d"},
  {"reduce_min", "reduce_min_d"},
  {"conv2d_backprop_filter", "conv2d_backprop_filter_d"},
  {"conv2d_backprop_input", "conv2d_backprop_input_d"},
  {"top_kv2", "top_k"},
  {"scatter_nd", "scatter_nd_d"},
  {"tile", "tile_d"},
  {"gather_v2", "gather_v2_d"},
  {"batch_mat_mul", "batch_matmul"},
  {"b_n_training_reduce", "bn_training_reduce"},
  {"b_n_training_update", "bn_training_update"},
  {"b_n_training_reduce_grad", "bn_training_reduce_grad"},
  {"b_n_training_update_grad", "bn_training_update_grad"},
  {"n_pu_clear_float_status", "n_p_u_clear_float_status"},
  {"n_pu_get_float_status", "n_p_u_get_float_status"},
  {"n_pu_alloc_float_status", "n_p_u_alloc_float_status"},
  {"dropout_do_mask", "drop_out_do_mask"},
  {"strided_slice", "strided_slice_d"},
  {"strided_slice_grad", "strided_slice_grad_d"},
  {"transpose", "transpose_d"},
  {"unsorted_segment_sum", "unsorted_segment_sum_d"},
  {"concat", "concat_d"},
  {"slice", "slice_d"},
  {"reduce_sum", "reduce_sum_d"},
  {"one_hot", "one_hot_d"},
  {"sum", "reduce_sum_d"},
  {"lamb_next_mv_with_decay_v1", "lamb_next_m_v_with_decay_v1"},
  {"lamb_next_mv", "lamb_next_m_v"},
  {"split", "split_d"},
  {"resize_nearest_neighbor", "resize_nearest_neighbor_d"},
  {"resize_nearest_neighbor_grad", "resize_nearest_neighbor_grad_d"},
  {"pad", "pad_d"},
  {"adam", "apply_adam"}};

void TbeAdapter::NormalizeFuncName(std::string *func_name) {
  if (func_name == nullptr) {
    MS_LOG(EXCEPTION) << "func_name is null";
  }
  std::string name_tmp;
  bool sub_head = false;
  for (string::iterator iter = func_name->begin(); iter != func_name->end(); ++iter) {
    if (islower(*iter)) {
      sub_head = false;
    }
    if (isdigit(*iter)) {
      sub_head = true;
    }
    if (isupper(*iter) && iter != func_name->begin()) {
      if (!sub_head) {
        (void)name_tmp.insert(name_tmp.end(), '_');
        sub_head = true;
      } else {
        string::iterator iter_next = iter + 1;
        if (iter_next != func_name->end()) {
          if (islower(*iter_next)) {
            (void)name_tmp.insert(name_tmp.end(), '_');
          }
        }
      }
    }
    (void)name_tmp.insert(name_tmp.end(), *iter);
  }
  (void)transform(name_tmp.begin(), name_tmp.end(), name_tmp.begin(), ::tolower);
  *func_name = name_tmp;
  auto iter = tbe_func_adapter_map.find(*func_name);
  if (iter != tbe_func_adapter_map.end()) {
    MS_LOG(INFO) << "map actual op from me " << func_name << "to tbe op" << iter->second;
    *func_name = iter->second;
  }
}

void TbeAdapter::SetTbeAttrsForTransDataOp(const mindspore::AnfNodePtr &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  if (AnfAlgo::GetCNodeName(anf_node) == kTransDataOpName) {
    std::string input_format = AnfAlgo::GetInputFormat(anf_node, 0);
    std::string output_format = AnfAlgo::GetOutputFormat(anf_node, 0);
    if (input_format == kOpFormat_DEFAULT) {
      input_format = kOpFormat_NCHW;
    }
    if (output_format == kOpFormat_DEFAULT) {
      output_format = kOpFormat_NCHW;
    }
    AnfAlgo::SetNodeAttr("src_format", MakeValue(input_format), anf_node);
    AnfAlgo::SetNodeAttr("dst_format", MakeValue(output_format), anf_node);
  }
}

std::unordered_set<std::string> input_order_adjusted_ops = {
  "Conv2DBackpropInput", "Conv2DBackpropFilter",       "LogSoftmaxGrad", "LayerNormGrad",
  "LayerNormXBackprop",  "LayerNormBetaGammaBackprop", "MinimumGrad",    "MaximumGrad"};

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
    } else {
      inputs_json->push_back(inputs_list[1]);
      inputs_json->push_back(inputs_list[0]);
      for (size_t i = 2; i < inputs_list.size(); ++i) {
        inputs_json->push_back(inputs_list[i]);
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
  } else if (type_str == "Bool_") {
    ret = 12;
  } else {
    MS_EXCEPTION(ArgumentError) << "type str is invailed: " << type_str;
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
  MS_LOG(INFO) << "CastAttrJsonPass done.";
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
