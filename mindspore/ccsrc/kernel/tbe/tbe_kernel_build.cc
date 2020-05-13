/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "kernel/tbe/tbe_kernel_build.h"

#include <memory>
#include <map>
#include <algorithm>
#include <unordered_set>

#include "operator/ops.h"
#include "session/anf_runtime_algorithm.h"
#include "kernel/tbe/tbe_kernel_mod.h"
#include "kernel/tbe/tbe_adapter.h"
#include "kernel/tbe/tbe_python_funcs.h"
#include "kernel/tbe/tbe_convert_utils.h"
#include "kernel/tbe/tbe_utils.h"

namespace mindspore {
namespace kernel {
using mindspore::kernel::tbe::TbeAdapter;
using mindspore::kernel::tbe::TbeUtils;
constexpr auto kFusionOpList = "op_list";
constexpr auto kFusionKernelNamePrfix = "te_fusion";
constexpr auto kOptional = "optional_";
constexpr auto kOpFormat_FRACTAL_Z = "FRACTAL_Z";

std::string NormalizeFullScopeName(const string &full_scope_name) {
  // exp:Default/ReLU-op0 -->Default_ReLU_op0
  string normal_ret = full_scope_name;
  std::replace(normal_ret.begin(), normal_ret.end(), '/', '_');
  std::replace(normal_ret.begin(), normal_ret.end(), '-', '_');
  return normal_ret;
}

bool TbeKernelJsonCreator::GenTbeSingleKernelJson(const shared_ptr<mindspore::AnfNode> &anf_node,
                                                  nlohmann::json *kernel_json) {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_EXCEPTION_IF_NULL(kernel_json);
  std::string op_name = AnfAlgo::GetCNodeName(anf_node);
  auto op_info_ptr = mindspore::kernel::OpLib::FindOp(op_name, OpImplyType::kTBE);
  MS_EXCEPTION_IF_NULL(op_info_ptr);
  (*kernel_json)["platform"] = "TBE";
  (*kernel_json)["gen_model"] = "single";
  (*kernel_json)["impl_path"] = op_info_ptr->impl_path();
  nlohmann::json op_info_json;
  if (op_info_ptr->impl_path().empty()) {
    tbe::TbeAdapter::NormalizeFuncName(&op_name);
  } else {
    op_name = op_info_ptr->kernel_name();
  }
  op_info_json["name"] = op_name;
  // generate inputs json
  nlohmann::json inputs_json;
  if (!GenTbeInputsJson(anf_node, op_info_ptr, &inputs_json)) {
    MS_LOG(ERROR) << "Anf Node [" << op_name << "] generate inputs json failed";
    return false;
  }
  op_info_json["inputs"] = inputs_json;
  // generate outputs json
  nlohmann::json outputs_json;
  if (!GenTbeOutputsJson(anf_node, op_info_ptr, &outputs_json)) {
    MS_LOG(ERROR) << "Anf Node [" << op_name << "] generate outputs json failed";
    return false;
  }
  op_info_json["outputs"] = outputs_json;
  // generate attrs json
  nlohmann::json attrs_json;
  (void)GenTbeAttrJson(anf_node, op_info_ptr, &attrs_json);
  op_info_json["attrs"] = attrs_json;
  std::string json_str = op_info_json.dump();
  size_t hash_id = std::hash<std::string>()(json_str);
  json_name_ = op_name + "_" + std::to_string(hash_id);
  json_info_ = json_str;
  if (creater_type_ == PREBUILD) {
    op_info_json["kernel_name"] = NormalizeFullScopeName(anf_node->fullname_with_scope());
  } else {
    op_info_json["kernel_name"] = json_name_;
  }
  (*kernel_json)["op_info"] = op_info_json;
  if (creater_type_ == SINGLE_BUILD) {
    TbeUtils::SaveJsonInfo(json_name_, json_info_);
  }

  MS_LOG(INFO) << "Operate type:" << creater_type_ << ", full scope name is :" << anf_node->fullname_with_scope()
               << ", json info name is : " << json_name_ << ", kernel json:" << kernel_json->dump();

  return true;
}

bool TbeKernelJsonCreator::GenInputDescJson(const shared_ptr<AnfNode> &anf_node, size_t real_input_index, bool value,
                                            const shared_ptr<OpIOInfo> &input_ptr, const string &op_input_name,
                                            size_t input_i, vector<nlohmann::json> *input_list) {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_EXCEPTION_IF_NULL(input_ptr);
  MS_EXCEPTION_IF_NULL(input_list);
  std::string op_name = AnfAlgo::GetCNodeName(anf_node);
  if (input_ptr->name() == "input_indices" && op_name == kTopKOpName) {
    TbeAdapter::GenTopKV2IndicesTensorInfo(anf_node, real_input_index, input_list, creater_type_);
  } else {
    // dtype : float16
    auto tensor_dtype =
      std::make_shared<TensorType>(TypeIdToType(AnfAlgo::GetInputDeviceDataType(anf_node, real_input_index)));
    MS_EXCEPTION_IF_NULL(tensor_dtype);
    std::string dtype = tensor_dtype->element()->ToString();
    dtype = tbe::DtypeToString(dtype);

    // format
    std::string format = AnfAlgo::GetInputFormat(anf_node, real_input_index);
    if (format == kOpFormat_DEFAULT) {
      format = kOpFormat_NCHW;
    } else if (format == kOpFormat_FRAC_Z) {
      format = kOpFormat_FRACTAL_Z;
    }

    nlohmann::json input_desc_json;
    input_desc_json["dtype"] = dtype;
    input_desc_json["name"] = op_input_name + std::to_string(input_i);
    auto ori_shape = AnfAlgo::GetPrevNodeOutputInferShape(anf_node, real_input_index);
    if (ori_shape.empty()) {
      ori_shape.emplace_back(1);
    }
    input_desc_json["ori_shape"] = ori_shape;
    input_desc_json["ori_format"] = kOpFormat_NCHW;
    auto shape = AnfAlgo::GetInputDeviceShape(anf_node, real_input_index);
    if (shape.empty()) {
      shape.emplace_back(1);
    }
    if (creater_type_ == OP_SELECT_FORMAT || creater_type_ == CHECK_SUPPORTED) {
      input_desc_json["shape"] = ori_shape;
      input_desc_json["format"] = kOpFormat_NCHW;
    } else {
      input_desc_json["shape"] = shape;
      input_desc_json["format"] = format;
    }
    input_desc_json["valid"] = value;
    input_list->emplace_back(input_desc_json);
  }
  return true;
}

bool TbeKernelJsonCreator::GenInputList(const shared_ptr<AnfNode> &anf_node, size_t input_tensor_num,
                                        const shared_ptr<OpIOInfo> &input_ptr, size_t *real_input_index,
                                        string *op_input_name, vector<nlohmann::json> *input_list) {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_EXCEPTION_IF_NULL(input_ptr);
  MS_EXCEPTION_IF_NULL(real_input_index);
  MS_EXCEPTION_IF_NULL(op_input_name);
  MS_EXCEPTION_IF_NULL(input_list);
  std::string op_name = AnfAlgo::GetCNodeName(anf_node);
  auto primitive = AnfAlgo::GetCNodePrimitive(anf_node);
  size_t real_input_num = AnfAlgo::GetInputTensorNum(anf_node);
  bool value = true;
  for (size_t input_i = 0; input_i < input_tensor_num; input_i++) {
    if (*real_input_index >= real_input_num) {
      if (input_ptr->param_type() == "optional") {
        *op_input_name = input_ptr->name() + "_optional_";
        nlohmann::json input_desc_json;
        input_desc_json["valid"] = false;
        input_desc_json["name"] = *op_input_name + std::to_string(*real_input_index);
        input_list->emplace_back(input_desc_json);
        continue;
      }
      MS_LOG(ERROR) << "input num" << *real_input_index << "is not match op inputs";
      return false;
    }
    if (op_name == "BatchNorm") {
      if (input_ptr->name() == "mean" || input_ptr->name() == "variance") {
        auto attr = primitive->GetAttr("is_training");
        MS_EXCEPTION_IF_NULL(attr);
        bool is_training = GetValue<bool>(attr);
        MS_LOG(INFO) << "op_name" << op_name << ", tensor_name " << input_ptr->name() << ", is_training "
                     << is_training;
        if (is_training) {
          (*real_input_index)++;
          break;
        }
      }
    }
    bool ret = GenInputDescJson(anf_node, *real_input_index, value, input_ptr, *op_input_name, input_i, input_list);
    (*real_input_index)++;
    if (!ret) {
      return false;
    }
  }
  return true;
}

bool GetInputNameAndRealNum(const std::shared_ptr<AnfNode> &anf_node, const shared_ptr<OpIOInfo> &input_ptr,
                            size_t *dyn_input_index, size_t *input_num, std::string *op_input_name) {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_EXCEPTION_IF_NULL(input_ptr);
  MS_EXCEPTION_IF_NULL(dyn_input_index);
  MS_EXCEPTION_IF_NULL(input_num);
  MS_EXCEPTION_IF_NULL(op_input_name);
  auto primitive = AnfAlgo::GetCNodePrimitive(anf_node);
  // for dynamic input number, dyn_input_sizes has the info of dynamic input num for each input.
  std::vector<int> dyn_input_sizes;
  if (primitive->GetAttr(kAttrDynInputSizes) != nullptr) {
    dyn_input_sizes = GetValue<const std::vector<int>>(primitive->GetAttr(kAttrDynInputSizes));
  }

  if (input_ptr->param_type() == "dynamic") {
    if (*dyn_input_index >= dyn_input_sizes.size()) {
      MS_LOG(ERROR) << "dyn input index" << *dyn_input_index << "is over dyn input num" << dyn_input_sizes.size();
      return false;
    }
    *input_num = IntToSize(dyn_input_sizes[*dyn_input_index]);
    *op_input_name = input_ptr->name() + "_dynamic_";
    (*dyn_input_index)++;
    // if optional input is exist
  } else {
    *input_num = 1;
    *op_input_name = input_ptr->name() + "_";
  }
  return true;
}

bool TbeKernelJsonCreator::GenTbeInputsJson(const std::shared_ptr<AnfNode> &anf_node,
                                            const std::shared_ptr<OpInfo> &op_info, nlohmann::json *inputs_json) {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_EXCEPTION_IF_NULL(op_info);
  MS_EXCEPTION_IF_NULL(inputs_json);
  std::string op_name = AnfAlgo::GetCNodeName(anf_node);
  if (op_name == kAtomicAddrCleanOpName) {
    return true;
  }
  std::vector<std::shared_ptr<OpIOInfo>> inputs_ptr = op_info->inputs_ptr();
  if (inputs_ptr.empty()) {
    MS_LOG(INFO) << "Apply kernel " << op_name << "registration info has no input info";
    return true;
  }
  auto op_info_input_num = inputs_ptr.size();
  size_t dyn_input_index = 0;
  size_t real_input_index = 0;
  std::vector<std::vector<nlohmann::json>> inputs_list;
  for (size_t i = 0; i < op_info_input_num; i++) {
    size_t input_tensor_num;
    std::shared_ptr<OpIOInfo> input_ptr = inputs_ptr[i];
    std::string op_input_name;
    MS_EXCEPTION_IF_NULL(input_ptr);
    if (!GetInputNameAndRealNum(anf_node, input_ptr, &dyn_input_index, &input_tensor_num, &op_input_name)) {
      return false;
    }
    std::vector<nlohmann::json> input_list;
    if (!GenInputList(anf_node, input_tensor_num, input_ptr, &real_input_index, &op_input_name, &input_list)) {
      return false;
    }
    inputs_list.emplace_back(input_list);
  }

  TbeAdapter::InputOrderPass(op_name, inputs_list, inputs_json);
  return true;
}

bool TbeKernelJsonCreator::GenTbeOutputsJson(const std::shared_ptr<AnfNode> &anf_node,
                                             const std::shared_ptr<OpInfo> &op_info, nlohmann::json *outputs_json) {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_EXCEPTION_IF_NULL(op_info);
  MS_EXCEPTION_IF_NULL(outputs_json);
  auto op_name = AnfAlgo::GetCNodeName(anf_node);
  if (op_name == kAtomicAddrCleanOpName) {
    return true;
  }
  auto outputs_ptr = op_info->outputs_ptr();
  return GenOutputDescJson(anf_node, outputs_ptr, outputs_json);
}

bool TbeKernelJsonCreator::GenOutputDescJson(const shared_ptr<mindspore::AnfNode> &anf_node,
                                             const vector<shared_ptr<mindspore::kernel::OpIOInfo>> &outputs_ptr,
                                             nlohmann::json *outputs_json) {
  MS_EXCEPTION_IF_NULL(outputs_json);
  size_t output_idx = 0;
  auto op_name = AnfAlgo::GetCNodeName(anf_node);
  size_t real_output_num = AnfAlgo::GetOutputTensorNum(anf_node);

  for (const auto &output_ptr : outputs_ptr) {
    size_t output_obj_num = 0;
    if (output_ptr->param_type() == "required") {
      output_obj_num = 1;
    } else if (output_ptr->param_type() == "dynamic") {
      if (outputs_ptr.size() > 1) {
        MS_LOG(ERROR) << "Dynamic output is unsupported multi output!";
        return false;
      }
      output_obj_num = real_output_num;
    } else {
      if (output_idx >= real_output_num) {
        MS_LOG(INFO) << "op:" << op_name << ", output" << output_ptr->name() << " is optional, output is none.";
        std::vector<nlohmann::json> output_list;
        nlohmann::json output_obj;
        output_obj["name"] = output_ptr->name();
        output_obj["valid"] = false;
        output_list.emplace_back(output_obj);
        (*outputs_json).push_back(output_list);
        continue;
      } else {
        output_obj_num = 1;
      }
    }
    std::vector<nlohmann::json> output_list;
    GenOutputList(anf_node, output_obj_num, output_ptr, &output_idx, &output_list);
    (*outputs_json).push_back(output_list);
  }
  return true;
}

void TbeKernelJsonCreator::GenOutputList(const shared_ptr<AnfNode> &anf_node, const size_t &output_obj_num,
                                         const shared_ptr<OpIOInfo> &output_ptr, size_t *output_idx,
                                         vector<nlohmann::json> *output_list) {
  MS_EXCEPTION_IF_NULL(output_idx);
  MS_EXCEPTION_IF_NULL(output_list);
  for (size_t i = 0; i < output_obj_num; i++) {
    nlohmann::json output_obj;
    auto type_ptr = std::make_shared<TensorType>(TypeIdToType(AnfAlgo::GetOutputDeviceDataType(anf_node, *output_idx)));
    std::string dtype = type_ptr->element()->ToString();
    dtype = tbe::DtypeToString(dtype);
    std::string format = AnfAlgo::GetOutputFormat(anf_node, *output_idx);
    if (format == kOpFormat_DEFAULT) {
      format = kOpFormat_NCHW;
    } else if (format == kOpFormat_FRAC_Z) {
      format = kOpFormat_FRACTAL_Z;
    }
    std::vector<size_t> ori_shape;
    if (AnfAlgo::GetOutputInferShape(anf_node, *output_idx).empty()) {
      ori_shape.emplace_back(1);
    } else {
      ori_shape = AnfAlgo::GetOutputInferShape(anf_node, *output_idx);
    }
    output_obj["dtype"] = dtype;
    auto shape = AnfAlgo::GetOutputDeviceShape(anf_node, *output_idx);
    if (shape.empty()) {
      shape.emplace_back(1);
    }
    if (creater_type_ == OP_SELECT_FORMAT || creater_type_ == CHECK_SUPPORTED) {
      output_obj["shape"] = ori_shape;
      output_obj["format"] = kOpFormat_NCHW;
    } else {
      output_obj["shape"] = shape;
      output_obj["format"] = format;
    }
    output_obj["ori_shape"] = ori_shape;
    output_obj["ori_format"] = kOpFormat_NCHW;
    output_obj["name"] = output_ptr->name();
    output_obj["valid"] = true;

    output_list->emplace_back(output_obj);
    (*output_idx)++;
  }
}

bool TbeKernelJsonCreator::GenTbeAttrJson(const std::shared_ptr<AnfNode> &anf_node,
                                          const std::shared_ptr<OpInfo> &op_info, nlohmann::json *attrs_json) {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_EXCEPTION_IF_NULL(op_info);
  MS_EXCEPTION_IF_NULL(attrs_json);
  auto attrs_ptr = op_info->attrs_ptr();
  if (TbeAdapter::RunAttrPass(anf_node, attrs_ptr, attrs_json)) {
    return true;
  }
  auto primitive = AnfAlgo::GetCNodePrimitive(anf_node);
  MS_EXCEPTION_IF_NULL(primitive);
  for (const auto &attr_ptr : attrs_ptr) {
    std::string attr_name = attr_ptr->name();
    nlohmann::json attr_obj;
    attr_obj["name"] = attr_name;
    if (primitive->GetAttr(attr_name) != nullptr) {
      auto value = primitive->GetAttr(attr_name);
      std::string type = attr_ptr->type();
      ParseAttrValue(type, value, &attr_obj);
      attr_obj["valid"] = true;
    } else {
      if (op_info->impl_path().empty()) {
        attr_obj["valid"] = false;
      } else {
        if (attr_ptr->param_type() == "required" && creater_type_ == SINGLE_BUILD) {
          MS_LOG(EXCEPTION) << "op name: " << op_info->op_name() << " attr: " << attr_name
                            << " is required, but not set.";
        } else {
          attr_obj["valid"] = false;
        }
      }
    }
    (*attrs_json).push_back(attr_obj);
  }
  return true;
}

void TbeKernelJsonCreator::ParseAttrValue(const std::string &type, const mindspore::ValuePtr &value,
                                          nlohmann::json *attr_obj) {
  MS_EXCEPTION_IF_NULL(value);
  MS_EXCEPTION_IF_NULL(attr_obj);
  if (type == "int") {
    auto attr_value = GetValue<int>(value);
    (*attr_obj)["value"] = attr_value;
  } else if (type == "str") {
    auto attr_value = GetValue<std::string>(value);
    if (attr_value == kOpFormat_FRAC_Z) {
      attr_value = kOpFormat_FRACTAL_Z;
    }
    (*attr_obj)["value"] = attr_value;
  } else if (type == "bool") {
    auto attr_value = GetValue<bool>(value);
    (*attr_obj)["value"] = attr_value;
  } else if (type == "float") {
    auto attr_value = GetValue<float>(value);
    (*attr_obj)["value"] = attr_value;
  } else if (type == "listInt") {
    std::vector<int> attr_value;
    auto value_type = value->type();
    MS_EXCEPTION_IF_NULL(value_type);
    auto value_type_str = value_type->ToString();
    if (value_type_str == "Int32") {
      int data = GetValue<int>(value);
      attr_value.push_back(data);
    } else {
      attr_value = GetValue<std::vector<int>>(value);
    }
    (*attr_obj)["value"] = attr_value;
  } else if (type == "listListInt") {
    auto attr_value = GetValue<std::vector<std::vector<int>>>(value);
    (*attr_obj)["value"] = attr_value;
  } else {
    MS_LOG(EXCEPTION) << "type: " << type << "not support";
  }
}

bool TbeKernelBuild::GetIOSize(const nlohmann::json &kernel_json, std::vector<size_t> *input_size_list,
                               std::vector<size_t> *output_size_list) {
  if (input_size_list == nullptr || output_size_list == nullptr) {
    MS_LOG(ERROR) << "input size or output size is nullptr";
    return false;
  }
  input_size_list->clear();
  output_size_list->clear();
  for (size_t i = 0; i < kernel_json["op_info"]["inputs"].size(); i++) {
    for (size_t m = 0; m < kernel_json["op_info"]["inputs"][i].size(); m++) {
      size_t size_i = 1;
      if (kernel_json["op_info"]["inputs"][i][m]["valid"] == false) {
        std::string input_name = kernel_json["op_info"]["inputs"][i][m]["name"];
        MS_LOG(INFO) << "Input name:" << input_name << "is optional, valid is false.";
        continue;
      }
      for (const auto &j : kernel_json["op_info"]["inputs"][i][m]["shape"]) {
        size_i *= static_cast<size_t>(j);
      }
      std::string dtype = kernel_json["op_info"]["inputs"][i][m]["dtype"];
      size_t nbyte = tbe::GetDtypeNbyte(dtype);
      size_i *= nbyte;
      input_size_list->push_back(size_i);
    }
  }
  for (size_t i = 0; i < kernel_json["op_info"]["outputs"].size(); i++) {
    for (size_t m = 0; m < kernel_json["op_info"]["outputs"][i].size(); m++) {
      size_t size_i = 1;
      if (kernel_json["op_info"]["outputs"][i][m]["valid"] == false) {
        std::string output_name = kernel_json["op_info"]["outputs"][i][m]["name"];
        MS_LOG(INFO) << "Output name:" << output_name << " is optional, valid is false.";
        continue;
      }
      for (const auto &j : kernel_json["op_info"]["outputs"][i][m]["shape"]) {
        size_i *= static_cast<size_t>(j);
      }
      std::string dtype = kernel_json["op_info"]["outputs"][i][m]["dtype"];
      size_t nbyte = tbe::GetDtypeNbyte(dtype);
      size_i *= nbyte;
      output_size_list->push_back(size_i);
    }
  }
  return true;
}

bool TbeKernelBuild::GenFusionScopeJson(const vector<mindspore::AnfNodePtr> &input_nodes,
                                        const vector<mindspore::AnfNodePtr> &compute_nodes, nlohmann::json *fusion_str,
                                        std::string *fusion_kernel) {
  MS_EXCEPTION_IF_NULL(fusion_str);
  MS_EXCEPTION_IF_NULL(fusion_kernel);
  // get input layer info
  std::vector<std::vector<mindspore::AnfNodePtr>> input_layers;
  std::map<const AnfNodePtr, FusionDataType> spec_data_input;
  if (!GetInputLayers(input_nodes, compute_nodes, &input_layers, &spec_data_input)) {
    return false;
  }
  // gen fusion scopre_op jsom
  vector<nlohmann::json> compute_list;
  (*fusion_kernel) = kFusionKernelNamePrfix;
  // index: fusion build option input record, next one from 0
  static size_t index = 0;
  auto layer_iter = input_layers.begin();
  auto compute_op_iter = compute_nodes.begin();
  for (; compute_op_iter != compute_nodes.end(); ++compute_op_iter, ++layer_iter) {
    nlohmann::json compute_op_str;
    (void)GenFusionComputeJson(*compute_op_iter, &layer_iter, &compute_op_str, fusion_kernel, &index);
    compute_list.push_back(compute_op_str);
  }
  index = 0;
  // gen data input json
  vector<nlohmann::json> data_list;
  for (const auto &layer : input_layers) {
    for (const auto &data_input : layer) {
      nlohmann::json data_str;
      if (!GenFusionDataInputJson(data_input, spec_data_input, &data_str, &index)) {
        MS_LOG(INFO) << "Fusion error: gen fusion datainput json faild.";
        return false;
      }
      data_list.push_back(data_str);
    }
  }
  index = 0;
  data_list.insert(data_list.end(), compute_list.begin(), compute_list.end());
  (*fusion_str)[kFusionOpList] = data_list;
  return true;
}

void TbeKernelBuild::GenDescJson(const std::shared_ptr<mindspore::AnfNode> &anf_node, size_t node_out_idx,
                                 size_t desc_output_idx, nlohmann::json *output_desc, FusionDataType fusion_data_type) {
  std::string output_desc_name = anf_node->fullname_with_scope();
  if (node_out_idx > 0) {
    output_desc_name = output_desc_name + "_" + std::to_string(node_out_idx);
  }
  (*output_desc)["name"] = NormalizeFullScopeName(output_desc_name);
  auto type_id = AnfAlgo::GetOutputDeviceDataType(anf_node, node_out_idx);
  (*output_desc)["data_type"] = tbe::TypeIdToString(type_id);
  auto ori_shape = AnfAlgo::GetOutputInferShape(anf_node, node_out_idx);
  if (ori_shape.empty()) {
    ori_shape.emplace_back(1);
  }
  (*output_desc)["ori_shape"] = ori_shape;
  auto shape = AnfAlgo::GetOutputDeviceShape(anf_node, node_out_idx);
  if (shape.empty()) {
    shape.emplace_back(1);
  }
  (*output_desc)["shape"] = shape;
  auto format = AnfAlgo::GetOutputFormat(anf_node, node_out_idx);
  if (format == kOpFormat_DEFAULT) {
    format = ori_shape.size() == 4 ? kOpFormat_NCHW : kOpFormat_ND;
  }
  (*output_desc)["format"] = format;
  (*output_desc)["ori_format"] = kOpFormat_NCHW;
  (*output_desc)["output_index"] = desc_output_idx;
  if (fusion_data_type == kFusionAddN && format == kOpFormat_NC1HWC0) {
    std::vector<size_t> spec_shape = {};
    spec_shape.emplace_back(shape[0]);
    spec_shape.emplace_back(shape[1]);
    spec_shape.emplace_back(shape[2] * shape[3]);
    spec_shape.emplace_back(shape[4]);
    (*output_desc)["shape"] = spec_shape;
  } else if (fusion_data_type == kFusionReLUGradV2 && (*output_desc)["data_type"] == "uint8") {
    std::vector<size_t> spec_shape = {};
    spec_shape.emplace_back(shape[0]);
    spec_shape.emplace_back(shape[1]);
    spec_shape.emplace_back(shape[2] * shape[3]);
    spec_shape.emplace_back(16);
    (*output_desc)["shape"] = spec_shape;
    (*output_desc)["data_type"] = "bool";
  }
}

void TbeKernelBuild::GenReusedOutputDesc(const shared_ptr<mindspore::AnfNode> &anf_node, size_t index,
                                         size_t output_index, nlohmann::json *output_desc) {
  std::string output_desc_name = anf_node->fullname_with_scope() + "_" + std::to_string(index);
  (*output_desc)["name"] = NormalizeFullScopeName(output_desc_name);
  (*output_desc)["output_index"] = output_index;
  std::vector<size_t> shape;
  (*output_desc)["shape"] = shape;
}

bool TbeKernelBuild::GetSpecInputLayers(const std::string &op_name,
                                        const std::vector<mindspore::AnfNodePtr> &reorder_layer,
                                        std::map<const AnfNodePtr, FusionDataType> *spec_data_input) {
  if ((op_name == kReluGradV2OpName || op_name == kAddNOpName) && reorder_layer.empty()) {
    MS_LOG(INFO) << "Fusion error: node(" << op_name << " )'s input is null. ";
    return false;
  }
  MS_LOG(INFO) << "Fusion info: op_name: " << op_name << "input layer size: " << reorder_layer.size();
  if (op_name == kReluGradV2OpName) {
    (*spec_data_input)[reorder_layer[0]] = kFusionReLUGradV2;
  } else if (op_name == kAddNOpName) {
    for (const auto &it : reorder_layer) {
      (*spec_data_input)[it] = kFusionAddN;
    }
  }
  return true;
}

bool TbeKernelBuild::GetInputLayers(const std::vector<mindspore::AnfNodePtr> &input_nodes,
                                    const std::vector<mindspore::AnfNodePtr> &compute_nodes,
                                    std::vector<std::vector<mindspore::AnfNodePtr>> *input_layers,
                                    std::map<const AnfNodePtr, FusionDataType> *spec_data_input) {
  auto result = std::find_if(compute_nodes.begin(), compute_nodes.end(), [](const auto &it) {
    auto op_name = AnfAlgo::GetCNodeName(it);
    return op_name == kConv2DBackpropInputOpName;
  });
  bool need_spec = (result != compute_nodes.end());
  size_t input_size = 0;
  for (const auto &compute_node : compute_nodes) {
    std::vector<mindspore::AnfNodePtr> layer = {};
    std::vector<mindspore::AnfNodePtr> reorder_layer = {};
    MS_EXCEPTION_IF_NULL(compute_node);
    auto op_name = AnfAlgo::GetCNodeName(compute_node);
    auto ccompute_node = compute_node->cast<CNodePtr>();
    if (ccompute_node == nullptr) {
      MS_LOG(INFO) << "Fusion error: fusion compute node must be cnode";
      return false;
    }
    MS_LOG(INFO) << "Fusion info: compute name: " << compute_node->fullname_with_scope();
    for (size_t i = 1; i < ccompute_node->inputs().size(); ++i) {
      auto input = ccompute_node->input(i);
      auto find_iter = std::find(input_nodes.begin(), input_nodes.end(), input);
      if (find_iter != input_nodes.end()) {
        MS_LOG(INFO) << "Fusion info: add compute node's [" << i << "] input: " << input->fullname_with_scope();
        layer.emplace_back((*find_iter));
      } else {
        MS_LOG(INFO) << "Fusion warnig: this input [" << i << "] may be pre compute(" << input->fullname_with_scope()
                     << ") node's output.";
      }
    }
    TbeAdapter::FusionDataOrderPass(op_name, layer, &reorder_layer);
    if (need_spec) {
      MS_LOG(INFO) << "Fusion info: match conv2d backprop input + ... patten.";
      if (!GetSpecInputLayers(op_name, reorder_layer, spec_data_input)) {
        return false;
      }
    }
    input_size += reorder_layer.size();
    input_layers->emplace_back(reorder_layer);
  }
  if (input_nodes.size() != input_size) {
    MS_LOG(INFO) << "Fusion error: fusion scope error, layer input:" << input_size
                 << ", input_node:" << input_nodes.size();
    return false;
  }
  return true;
}

bool TbeKernelBuild::GenFusionDataInputJson(const std::shared_ptr<mindspore::AnfNode> &data_input,
                                            const std::map<const AnfNodePtr, FusionDataType> &spec_data_input,
                                            nlohmann::json *data_str, size_t *index) {
  MS_EXCEPTION_IF_NULL(data_str);
  MS_EXCEPTION_IF_NULL(index);
  std::vector<nlohmann::json> output_desc_list;
  if (!data_input) {
    MS_LOG(INFO) << "data input is optional node";
    auto name = std::string(kOptional) + std::to_string(*index);
    (*data_str)["name"] = name;
    nlohmann::json output_desc;
    output_desc["name"] = name;
    output_desc["shape"] = "NULL";
    output_desc_list.push_back(output_desc);
    (*index)++;
  } else {
    FusionDataType fusion_data_type = kFusionNormal;
    if (spec_data_input.find(data_input) != spec_data_input.end()) {
      fusion_data_type = spec_data_input.at(data_input);
    }
    auto kernel_idx = AnfAlgo::VisitKernel(data_input, 0);
    auto real_node = kernel_idx.first;
    size_t real_idx = kernel_idx.second;
    MS_LOG(INFO) << "real name " << real_node->fullname_with_scope() << " index:" << real_idx;
    // "output_desc"
    nlohmann::json output_desc;
    GenDescJson(real_node, real_idx, real_idx, &output_desc, fusion_data_type);
    output_desc_list.push_back(output_desc);
    (*data_str)["name"] = NormalizeFullScopeName(real_node->fullname_with_scope());
  }
  (*data_str)["output_desc"] = output_desc_list;
  (*data_str)["type"] = "Data";
  return true;
}

bool TbeKernelBuild::IsDynamicInput(const mindspore::CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  auto primitive = AnfAlgo::GetCNodePrimitive(cnode);
  MS_EXCEPTION_IF_NULL(primitive);
  // for dynamic input number, dyn_input_sizes has the info of dynamic input num for each input.
  bool ret = false;
  std::vector<int> dyn_input_sizes;
  auto dynamic_input_attr = primitive->GetAttr(kAttrDynInputSizes);
  if (dynamic_input_attr != nullptr) {
    dyn_input_sizes = GetValue<const std::vector<int>>(dynamic_input_attr);
    auto real_input_size = cnode->inputs().size() - 1;
    auto dyn_input_size = dyn_input_sizes.size();
    if (dyn_input_size != 1) {
      MS_LOG(INFO) << "Fusion error: fusion build not support dyn_input_sizes > 1";
      return ret;
    }
    if (IntToSize(dyn_input_sizes[0]) != real_input_size) {
      MS_LOG(INFO) << "Fusion error: dyn_input_size" << dyn_input_sizes[0] << "not equal real_input_size"
                   << real_input_size;
      return ret;
    }
    ret = true;
  }
  return ret;
}

size_t TbeKernelBuild::GetOptionalInput(const mindspore::CNodePtr &cnode, bool is_dynamic_input) {
  if (is_dynamic_input) {
    return 0;
  }
  MS_EXCEPTION_IF_NULL(cnode);
  auto node_name = AnfAlgo::GetCNodeName(cnode);
  auto op_info = OpLib::FindOp(node_name, kTBE);
  MS_EXCEPTION_IF_NULL(cnode);
  if (op_info->inputs_ptr().size() < (cnode->inputs().size() - 1)) {
    MS_EXCEPTION(ArgumentError) << "op info error, node name:" << cnode->fullname_with_scope();
  }
  return (op_info->inputs_ptr().size() + 1 - cnode->inputs().size());
}

std::string TbeKernelBuild::GetRealOpType(const std::string &origin_type) {
  static std::map<std::string, std::string> buffer_fussion_op_map = {{"DepthwiseConv2dNative", "DepthwiseConv2D"},
                                                                     {"TensorAdd", "Add"}};
  string result = origin_type;
  auto iter = buffer_fussion_op_map.find(origin_type);
  if (iter != buffer_fussion_op_map.end()) {
    result = iter->second;
  }
  return result;
}

bool TbeKernelBuild::GenFusionComputeInputJson(const mindspore::CNodePtr &cnode,
                                               std::vector<std::vector<mindspore::AnfNodePtr>>::iterator *layer_iter,
                                               std::vector<nlohmann::json> *input_desc_list, size_t *index) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(input_desc_list);
  std::vector<nlohmann::json> input_desc_list_tmp = {};
  bool is_dynamic_input = IsDynamicInput(cnode);
  for (size_t i = 1; i < cnode->inputs().size(); ++i) {
    auto input = cnode->input(i);
    auto kernel_idx = AnfAlgo::VisitKernel(input, 0);
    auto real_node = kernel_idx.first;
    size_t real_idx = kernel_idx.second;
    MS_LOG(INFO) << "real name" << real_node->fullname_with_scope() << "index:" << real_idx;
    nlohmann::json input_desc;
    GenDescJson(real_node, real_idx, real_idx, &input_desc);
    if (is_dynamic_input) {
      MS_LOG(INFO) << "node has dynamic input.";
      input_desc["dyn_index"] = (i - 1);
    }
    input_desc_list_tmp.emplace_back(input_desc);
  }
  size_t optional_num = GetOptionalInput(cnode, is_dynamic_input);
  if (optional_num > 0) {
    MS_LOG(INFO) << "node has optional input.";
    for (size_t i = 0; i < optional_num; ++i) {
      nlohmann::json optional_input_desc;
      optional_input_desc["name"] = std::string(kOptional) + std::to_string(*index);
      (*index)++;
      (*layer_iter)->emplace_back(nullptr);
      input_desc_list_tmp.emplace_back(optional_input_desc);
    }
  }
  auto op_name = AnfAlgo::GetCNodeName(cnode);
  TbeAdapter::FusionInputOrderPass(op_name, input_desc_list_tmp, input_desc_list);
  return true;
}

std::vector<size_t> TbeKernelBuild::GetDescOutputIndex(const std::vector<int> &output_used_nums) {
  std::vector<size_t> desc_output_index = {};
  for (size_t idx = 0; idx < output_used_nums.size(); ++idx) {
    auto output_use_num_item = output_used_nums[idx];
    MS_LOG(INFO) << "output used num[" << idx << "] = " << output_use_num_item;
    desc_output_index.emplace_back(idx);
    if (output_use_num_item > 1) {
      desc_output_index.emplace_back(idx);
    }
  }
  return desc_output_index;
}

bool TbeKernelBuild::GenFusionComputeOutputJson(const mindspore::CNodePtr &cnode,
                                                std::vector<nlohmann::json> *output_desc_list) {
  auto output_size = AnfAlgo::GetOutputTensorNum(cnode);
  if (AnfAlgo::HasNodeAttr(kAttrOutputUsedNum, cnode)) {
    auto output_used_nums = AnfAlgo::GetNodeAttr<std::vector<int>>(cnode, kAttrOutputUsedNum);
    MS_LOG(INFO) << "This node's output has been reused, node name: " << cnode->fullname_with_scope();
    if (output_used_nums.size() != output_size) {
      MS_LOG(INFO) << "Fusion error: output tenor num(" << output_size << ")"
                   << " is not match output used num(" << output_used_nums.size() << ")";
      return false;
    }
    auto desc_output_index = GetDescOutputIndex(output_used_nums);
    for (size_t i = 0; i < output_size; ++i) {
      MS_LOG(INFO) << "Fusion index: " << i << ", desc_output_index: " << desc_output_index[i];
      nlohmann::json output_desc;
      GenDescJson(cnode, i, desc_output_index[i], &output_desc);
      output_desc_list->emplace_back(output_desc);
    }
    for (size_t j = output_size; j < desc_output_index.size(); ++j) {
      MS_LOG(INFO) << "Fusion index: " << j << ", desc_output_index: " << desc_output_index[j];
      nlohmann::json output_desc;
      GenReusedOutputDesc(cnode, j, desc_output_index[j], &output_desc);
      output_desc_list->emplace_back(output_desc);
    }
  } else {
    for (size_t i = 0; i < output_size; ++i) {
      nlohmann::json output_desc;
      GenDescJson(cnode, i, i, &output_desc);
      output_desc_list->push_back(output_desc);
    }
  }
  return true;
}

bool TbeKernelBuild::GenFusionComputeJson(const mindspore::AnfNodePtr &compute_node,
                                          std::vector<std::vector<mindspore::AnfNodePtr>>::iterator *layer_iter,
                                          nlohmann::json *compute_op_str, std::string *fusion_kernel_name,
                                          size_t *index) {
  MS_EXCEPTION_IF_NULL(compute_node);
  auto cnode = compute_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  // gen input desc
  std::vector<nlohmann::json> input_desc_list;
  (void)GenFusionComputeInputJson(cnode, layer_iter, &input_desc_list, index);
  (*compute_op_str)["input_desc"] = input_desc_list;
  // gen output desc
  std::vector<nlohmann::json> output_desc_list;
  if (!GenFusionComputeOutputJson(cnode, &output_desc_list)) {
    MS_LOG(INFO) << "Fusion Error: gen fusion output desc faild, node full name: " << cnode->fullname_with_scope();
    return false;
  }
  (*compute_op_str)["output_desc"] = output_desc_list;
  // gen others
  auto origin_type = AnfAlgo::GetCNodeName(cnode);
  // replace special op type for buffer fusion op
  auto type = GetRealOpType(origin_type);
  (*compute_op_str)["type"] = type;
  tbe::TbeAdapter::NormalizeFuncName(&type);
  (*compute_op_str)["func_name"] = type;
  (*compute_op_str)["name"] = NormalizeFullScopeName(cnode->fullname_with_scope());
  (void)(*fusion_kernel_name).append("_");
  (void)(*fusion_kernel_name).append(type);
  return true;
}

size_t TbeKernelBuild::GetIOSizeImpl(const nlohmann::json &desc) {
  size_t ret = 1;
  for (const auto &shape_item : desc["shape"]) {
    ret *= static_cast<size_t>(shape_item);
  }
  std::string data_type = desc["data_type"];
  size_t nbyte = tbe::GetDtypeNbyte(data_type);
  ret *= nbyte;
  return ret;
}

bool TbeKernelBuild::GetIOSize(const nlohmann::json &fusion_op_list, const vector<mindspore::AnfNodePtr> &output_nodes,
                               std::vector<size_t> *input_size_list, std::vector<size_t> *output_size_list) {
  MS_EXCEPTION_IF_NULL(input_size_list);
  MS_EXCEPTION_IF_NULL(output_size_list);
  input_size_list->clear();
  output_size_list->clear();

  for (const auto &op : fusion_op_list) {
    if (op["type"] == "Data") {
      const auto &data_output_desc = op["output_desc"];
      for (const auto &data_output : data_output_desc) {
        if (data_output["shape"] == "NULL") {
          break;
        }
        auto ret = GetIOSizeImpl(data_output);
        input_size_list->push_back(ret);
        MS_LOG(INFO) << "Fusion info: scope input name： " << op["name"] << ", size: " << ret;
      }
    }
  }

  for (const auto &output_node : output_nodes) {
    auto kernel_idx = AnfAlgo::VisitKernel(output_node, 0);
    auto real_node = kernel_idx.first;
    size_t real_idx = kernel_idx.second;
    auto normal_name = NormalizeFullScopeName(real_node->fullname_with_scope());
    MS_LOG(INFO) << "Fusion info: real node name: " << normal_name << ", real output index: " << real_idx;
    for (const auto &op : fusion_op_list) {
      if (op["name"] == normal_name) {
        auto op_output_desces = op["output_desc"];
        if (output_node != real_node) {
          // tuple_get item
          MS_LOG(INFO) << "output is a tuple getitem node";
          auto output_desc = op_output_desces[real_idx];
          if (output_desc["shape"].empty()) {
            MS_LOG(INFO) << "Fusion error: output_desc's shape is empty. real_index " << real_idx;
            return false;
          }
          auto ret = GetIOSizeImpl(output_desc);
          output_size_list->push_back(ret);
          MS_LOG(INFO) << "Fusion info: scope output index： " << real_idx << ", size: " << ret;
        } else {
          for (const auto &output_desc : op_output_desces) {
            if (output_desc["shape"].empty()) {
              MS_LOG(INFO) << "Fusion info: output_desc's shape is empty, may be this node output";
              continue;
            }
            auto ret = GetIOSizeImpl(output_desc);
            output_size_list->push_back(ret);
            MS_LOG(INFO) << "Fusion info: scope output size: " << ret;
          }
        }
      }
    }
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
