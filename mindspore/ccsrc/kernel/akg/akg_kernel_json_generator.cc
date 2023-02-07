/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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

#include "kernel/akg/akg_kernel_json_generator.h"

#include <set>
#include <functional>
#include <algorithm>
#include "ir/func_graph.h"
#include "utils/anf_utils.h"
#include "utils/ms_context.h"
#include "kernel/oplib/oplib.h"
#include "common/graph_kernel/core/graph_builder.h"
#include "common/graph_kernel/core/graph_kernel_utils.h"
#include "runtime/hardware/device_context_manager.h"
#include "common/graph_kernel/graph_kernel_flags.h"

namespace mindspore::graphkernel {
using kernel::OpAttr;
using kernel::OpImplyType;
using kernel::OpInfo;
using kernel::OpIOInfo;
namespace {
constexpr int kCurrentInfoVersion = 1;
constexpr auto kAttrParallelDimInfoSize = 2;
constexpr auto kDebugStrDepth = 2;

std::vector<int64_t> GetDynInputSizes(const AnfNodePtr &anf_node) {
  std::vector<int64_t> dyn_input_sizes;
  auto primitive = GetCNodePrimitive(anf_node);
  MS_EXCEPTION_IF_NULL(primitive);
  if (primitive->HasAttr(kAttrDynInputSizes)) {
    dyn_input_sizes = GetValue<const std::vector<int64_t>>(primitive->GetAttr(kAttrDynInputSizes));
  }
  return dyn_input_sizes;
}

std::pair<AnfNodePtr, size_t> GetKernelInput(const AnfNodePtr &anf_node, size_t index) {
  MS_EXCEPTION_IF_NULL(anf_node);
  auto inputs_num = AnfUtils::GetInputTensorNum(anf_node);
  if (index >= inputs_num) {
    MS_EXCEPTION(ArgumentError) << "Input index " << index << " is out of range [0, " << inputs_num << ") in node ["
                                << anf_node->DebugString() << "]";
  }
  auto cnode = anf_node->cast<CNodePtr>();
  if (cnode == nullptr) {
    return AnfUtils::VisitKernel(anf_node, 0);
  } else {
    return AnfUtils::VisitKernel(cnode->input(index + 1), 0);
  }
}

std::vector<std::pair<AnfNodePtr, std::pair<size_t, size_t>>> GetInputIndex(const std::vector<AnfNodePtr> &node_list,
                                                                            const std::vector<AnfNodePtr> &input_list) {
  std::vector<std::pair<AnfNodePtr, std::pair<size_t, size_t>>> input_index;
  for (size_t i = 0; i < input_list.size(); ++i) {
    auto const &input = input_list[i];
    MS_EXCEPTION_IF_NULL(input);
    MS_EXCEPTION_IF_NULL(input->func_graph());
    auto mng = input->func_graph()->manager();
    MS_EXCEPTION_IF_NULL(mng);
    const NodeUsersMap &users = mng->node_users();
    auto input_users = users.find(input);
    if (input_users == users.end() || input_users->second.empty()) {
      MS_EXCEPTION(ArgumentError) << "Input [" << i << "][" << input->DebugString(kDebugStrDepth) << "] of ["
                                  << input->func_graph()->ToString() << "] has no users.";
    }
    bool found = false;
    for (auto const &input_user : input_users->second) {
      for (auto const &anf_node : node_list) {
        if (anf_node != input_user.first) {
          continue;
        }
        auto dyn_input_sizes = GetDynInputSizes(anf_node);
        if (dyn_input_sizes.empty()) {
          input_index.push_back(std::make_pair(anf_node, std::make_pair(IntToSize(input_user.second - 1), 0)));
          found = true;
          break;
        }
        int64_t used_as_idx = IntToLong(input_user.second - 1);
        int64_t accum_idx = 0;
        for (size_t dyn_i = 0; dyn_i < dyn_input_sizes.size(); ++dyn_i) {
          accum_idx += dyn_input_sizes[dyn_i];
          if (used_as_idx < accum_idx) {
            auto tmp_dyn_i = dyn_i;  // to evade pclint warning "for statement index variable modified in body."
            input_index.push_back(std::make_pair(
              anf_node, std::make_pair(tmp_dyn_i, LongToSize(used_as_idx - (accum_idx - dyn_input_sizes[dyn_i])))));
            found = true;
            break;
          }
        }
        if (found) {
          break;
        }
      }
      if (found) {
        break;
      }
    }
    if (found) {
      continue;
    }
    MS_EXCEPTION(ArgumentError) << "Input [" << i << "][" << input->DebugString(kDebugStrDepth) << "] of ["
                                << input->func_graph()->ToString() << "] found no related kernel info.";
  }
  return input_index;
}

std::vector<std::pair<AnfNodePtr, size_t>> GetOutputIndex(const std::vector<AnfNodePtr> &node_list,
                                                          const std::vector<AnfNodePtr> &input_list,
                                                          const std::vector<AnfNodePtr> &output_list) {
  std::vector<std::pair<AnfNodePtr, size_t>> output_index;
  for (size_t i = 0; i < output_list.size(); ++i) {
    bool found = false;
    auto const &output = output_list[i];
    MS_EXCEPTION_IF_NULL(output);
    auto pree_node = AnfUtils::VisitKernel(output, 0);
    auto pos = std::find(std::begin(node_list), std::end(node_list), pree_node.first);
    if (pos != std::end(node_list)) {
      (void)output_index.emplace_back(pree_node);
      continue;
    }
    auto ret = std::find(std::begin(input_list), std::end(input_list), pree_node.first);
    if (ret != std::end(input_list)) {
      (void)output_index.emplace_back(std::make_pair(pree_node.first, 0));
      found = true;
    }
    if (!found) {
      MS_EXCEPTION(ArgumentError) << "Output [" << i << "][" << output->DebugString(kDebugStrDepth) << "] of ["
                                  << output->func_graph()->ToString() << "] found no related kernel info.";
    }
  }
  return output_index;
}

class OpInfoExtractor {
 public:
  OpInfoExtractor() = default;
  ~OpInfoExtractor() = default;
  OpInfoPtr Run(const AnfNodePtr &anf_node) {
    MS_EXCEPTION_IF_NULL(anf_node);
    cnode_ = anf_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode_);
    auto op_info = std::make_shared<OpInfo>();
    op_info->set_op_name(AnfUtils::GetCNodeName(cnode_));
    op_info->set_imply_type(OpImplyType::kImplyAKG);
    ExtractInputs(op_info);
    ExtractOutputs(op_info);
    ExtractAttrs(op_info);
    return op_info;
  }

 private:
  void ExtractInputs(const OpInfoPtr &op_info) const {
    auto dyn_input_sizes = GetDynInputSizes(cnode_);
    if (dyn_input_sizes.empty()) {
      for (size_t i = 1; i < cnode_->size(); i++) {
        auto io_info = std::make_shared<OpIOInfo>();
        io_info->set_name("input_" + std::to_string(i - 1));
        op_info->add_inputs_ptr(io_info);
      }
    } else {
      for (size_t i = 0; i < dyn_input_sizes.size(); i++) {
        auto io_info = std::make_shared<OpIOInfo>();
        io_info->set_name("input_" + std::to_string(i));
        io_info->set_param_type("dynamic");
        op_info->add_inputs_ptr(io_info);
      }
    }
  }

  void ExtractOutputs(const OpInfoPtr &op_info) const {
    size_t output_tensor_num = AnfUtils::GetOutputTensorNum(cnode_);
    for (size_t i = 0; i < output_tensor_num; i++) {
      auto io_info = std::make_shared<OpIOInfo>();
      io_info->set_name("output_" + std::to_string(i));
      op_info->add_outputs_ptr(io_info);
    }
  }

  bool ExcludeAttr(const std::string &name) const {
    const std::set<std::string> black_list = {"IsFeatureMapInputList", "IsFeatureMapOutput", kAttrOutputNames,
                                              kAttrInputNames, "is_load"};
    return black_list.count(name) != 0;
  }

  void ExtractAttrs(const OpInfoPtr &op_info) {
    auto prim = GetCNodePrimitive(cnode_);
    if (prim == nullptr) {
      return;
    }
    for (const auto &[name, v] : prim->attrs()) {
      if (ExcludeAttr(name)) {
        continue;
      }
      auto op_attr = std::make_shared<OpAttr>();
      op_attr->set_name(name);
      op_attr->set_param_type("required");
      // Only support the following types in op json.
      if (v->isa<Int32Imm>() || v->isa<Int64Imm>()) {
        op_attr->set_type("int");
      } else if (v->isa<FP32Imm>() || v->isa<FP64Imm>()) {
        op_attr->set_type("float");
      } else if (v->isa<BoolImm>()) {
        op_attr->set_type("bool");
      } else if (v->isa<StringImm>()) {
        op_attr->set_type("str");
      } else if (v->isa<Type>()) {
        // convert the TypeId to string
        op_attr->set_type("str");
      } else if (v->isa<ValueSequence>()) {
        const auto &vec = v->cast<ValueSequencePtr>()->value();
        if (vec.empty()) {
          op_attr->set_type("listInt");
        } else if (vec[0]->isa<Int32Imm>() || vec[0]->isa<Int64Imm>()) {
          op_attr->set_type("listInt");
        } else if (vec[0]->isa<StringImm>()) {
          op_attr->set_type("listStr");
        }
      }
      if (op_attr->type().empty()) {
        MS_LOG(DEBUG) << "Unknown type, ignore attr: " << name;
        continue;
      }
      op_info->add_attrs_ptr(op_attr);
    }
  }

  CNodePtr cnode_;
};
}  // namespace

bool AkgKernelJsonGenerator::GetInputTensorValue(const AnfNodePtr &anf_node, size_t input_idx,
                                                 nlohmann::json *node_json) const {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_EXCEPTION_IF_NULL(node_json);
  auto cnode = anf_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (input_idx + 1 >= cnode->size()) {
    MS_EXCEPTION(ArgumentError) << "Input index " << input_idx << " is out of range [0, " << cnode->inputs().size()
                                << ") in node [" << cnode->DebugString() << "]";
  }

  auto input_node = cnode->input(input_idx + 1);
  if (!IsValueNode<tensor::Tensor>(input_node)) {
    return false;
  }

  auto tensor = GetValueNode<tensor::TensorPtr>(input_node);
  if (tensor == nullptr) {
    MS_LOG(DEBUG) << "Value of input node is nullptr, op: [" << input_node->DebugString() << "]";
    return false;
  }

  auto type_id = tensor->data_type();
  auto *data = tensor->data_c();
  MS_EXCEPTION_IF_NULL(data);
  if (tensor->DataSize() > 1) {
    // not const tensor.
    MS_LOG(WARNING) << "Not take value of tensor whose datasize greater than 1, ["
                    << input_node->DebugString(kDebugStrDepth) << "]";
    return false;
  }

  if (type_id == kFloat64->type_id()) {
    (*node_json)["value"] = static_cast<double *>(data)[0];
  } else if (type_id == kFloat32->type_id()) {
    (*node_json)["value"] = static_cast<float *>(data)[0];
  } else if (type_id == kFloat16->type_id()) {
    float16 *val = static_cast<float16 *>(data);
    (*node_json)["value"] = static_cast<float>(val[0]);
  } else if (type_id == kUInt64->type_id()) {
    (*node_json)["value"] = static_cast<uint64_t *>(data)[0];
  } else if (type_id == kUInt32->type_id()) {
    (*node_json)["value"] = static_cast<uint32_t *>(data)[0];
  } else if (type_id == kUInt16->type_id()) {
    (*node_json)["value"] = static_cast<uint16_t *>(data)[0];
  } else if (type_id == kUInt8->type_id()) {
    (*node_json)["value"] = static_cast<uint8_t *>(data)[0];
  } else if (type_id == kInt64->type_id()) {
    (*node_json)["value"] = static_cast<int64_t *>(data)[0];
  } else if (type_id == kInt32->type_id()) {
    (*node_json)["value"] = static_cast<int32_t *>(data)[0];
  } else if (type_id == kInt16->type_id()) {
    (*node_json)["value"] = static_cast<int16_t *>(data)[0];
  } else if (type_id == kInt8->type_id()) {
    (*node_json)["value"] = static_cast<int8_t *>(data)[0];
  } else if (type_id == kBool->type_id()) {
    (*node_json)["value"] = static_cast<bool *>(data)[0];
  } else {
    MS_LOG(EXCEPTION) << "Fail to parse the input value of [" << cnode->DebugString() << "], the input index is "
                      << input_idx << ", because the value type: " << TypeIdToString(type_id, true)
                      << " is not in supported list: [float64, float32, float16, uint64, uint32, uint16, uint8, int64, "
                         "int32, int16, int8, bool].";
  }
  return true;
}

bool AkgKernelJsonGenerator::CreateInputDescJson(const AnfNodePtr &anf_node, const OpInfoPtr &op_info,
                                                 nlohmann::json *inputs_json) {
  // for dynamic input number, dyn_input_sizes has the info of dynamic input num for each input.
  auto inputs_ptr = op_info->inputs_ptr();
  if (inputs_ptr.empty()) {
    MS_LOG(ERROR) << "Kernel [" << anf_node->fullname_with_scope() << "] info has no input info";
    return false;
  }

  // for dynamic input number, dyn_input_sizes has the info of dynamic input num for each input.
  auto dyn_input_sizes = GetDynInputSizes(anf_node);
  size_t real_input_index = 0;
  for (size_t i = 0; i < inputs_ptr.size(); i++) {
    auto input_ptr = inputs_ptr[i];
    if (input_ptr == nullptr) {
      MS_LOG(ERROR) << "Kernel [" << anf_node->fullname_with_scope() << "] input[" << i << "] is nullptr";
      return false;
    }

    size_t input_tensor_num = dyn_input_sizes.empty() ? 1 : LongToSize(dyn_input_sizes[i]);
    std::vector<nlohmann::json> input_list;
    for (size_t input_i = 0; input_i < input_tensor_num; input_i++) {
      auto type_id = this->cb_->GetInputType(anf_node, real_input_index);
      std::string dtype = TypeIdToString(type_id, true);
      if (dtype.empty()) {
        MS_LOG(ERROR) << "Op [" << anf_node->fullname_with_scope() << "] input [" << real_input_index
                      << "] data type is null. ";
        return false;
      }
      nlohmann::json input_desc_json;
      input_desc_json[kJsonKeyDataType] = dtype;
      input_desc_json[kJsonKeyFormat] = this->cb_->GetInputFormat(anf_node, real_input_index);
      input_desc_json[kJsonKeyName] = input_ptr->name();
      input_desc_json[kJsonKeyTensorName] = "input_" + std::to_string(GetInputTensorIdxInc(anf_node, real_input_index));
      auto input_shape = this->cb_->GetInputShape(anf_node, real_input_index);
      if (!is_basic_op_ && GetInputTensorValue(anf_node, real_input_index, &input_desc_json)) {
        MS_LOG(DEBUG) << "Pick single value [" << input_desc_json[kJsonKeyValue] << "] from input[" << real_input_index
                      << "] of node [" << anf_node->DebugString(kDebugStrDepth);
        input_shape.clear();
      }
      if (input_shape.empty()) {
        input_shape.push_back(1);
      }
      input_desc_json[kJsonKeyShape] = input_shape;
      (void)input_list.emplace_back(input_desc_json);
      real_input_index++;
    }
    (void)inputs_json->emplace_back(input_list);
  }
  return true;
}

bool AkgKernelJsonGenerator::CreateOutputDescJson(const AnfNodePtr &anf_node, const OpInfoPtr &op_info,
                                                  nlohmann::json *outputs_json) {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_EXCEPTION_IF_NULL(op_info);
  MS_EXCEPTION_IF_NULL(outputs_json);
  size_t output_tensor_num = AnfUtils::GetOutputTensorNum(anf_node);

  auto outputs = op_info->outputs_ptr();
  for (size_t i = 0; i < output_tensor_num; i++) {
    nlohmann::json output_json;
    auto type_id = this->cb_->GetOutputType(anf_node, i);
    std::string dtype = TypeIdToString(type_id, true);
    if (dtype.empty()) {
      MS_LOG(ERROR) << "Op [" << anf_node->fullname_with_scope() << "] output [" << i << "] data type is null. ";
      return false;
    }

    std::string output_name = outputs[i]->name();
    output_json[kJsonKeyDataType] = dtype;
    output_json[kJsonKeyFormat] = this->cb_->GetOutputFormat(anf_node, i);
    output_json[kJsonKeyName] = output_name;
    output_json[kJsonKeyTensorName] = "output_" + std::to_string(i) + "_" + std::to_string(GetOutputTensorIdxInc());
    auto output_shape = this->cb_->GetOutputShape(anf_node, i);
    if (output_shape.empty()) {
      output_shape.push_back(1);
    }
    output_json[kJsonKeyShape] = output_shape;
    outputs_json->push_back(output_json);
  }
  return true;
}

void AkgKernelJsonGenerator::GetAttrJson(const AnfNodePtr &anf_node, const std::vector<int64_t> &dyn_input_sizes,
                                         const OpAttrPtr &op_attr, nlohmann::json *attr_json,
                                         const ValuePtr &attr_value) {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_EXCEPTION_IF_NULL(op_attr);
  MS_EXCEPTION_IF_NULL(attr_json);

  auto get_int_value = [](const ValuePtr &value) -> int {
    return value->isa<Int64Imm>() ? static_cast<int>(GetValue<int64_t>(value)) : GetValue<int>(value);
  };
  std::string type = op_attr->type();
  (*attr_json)[kJsonKeyDataType] = type;
  if (type == "int") {
    (*attr_json)[kJsonKeyValue] = get_int_value(attr_value);
  } else if (type == "str") {
    if (attr_value->isa<Type>()) {
      (*attr_json)[kJsonKeyValue] = TypeIdToString(attr_value->cast<TypePtr>()->type_id(), true);
    } else {
      (*attr_json)[kJsonKeyValue] = GetValue<std::string>(attr_value);
    }
  } else if (type == "bool") {
    (*attr_json)[kJsonKeyValue] = GetValue<bool>(attr_value);
  } else if (type == "float") {
    (*attr_json)[kJsonKeyValue] = GetValue<float>(attr_value);
  } else if (type == "listInt") {
    std::vector<int> list_int;
    const auto &vals = attr_value->cast<ValueSequencePtr>()->value();
    (void)std::transform(vals.begin(), vals.end(), std::back_inserter(list_int), get_int_value);
    (*attr_json)[kJsonKeyValue] = list_int;
  } else if (type == "listStr") {
    std::vector<std::string> data_format;
    if (op_attr->name() == kJsonKeyDataformat) {
      size_t tensor_args_num =
        !dyn_input_sizes.empty() ? dyn_input_sizes.size() : AnfUtils::GetInputTensorNum(anf_node);
      for (size_t format_i = 0; format_i < tensor_args_num; format_i++) {
        auto input_format = this->cb_->GetInputFormat(anf_node, format_i);
        data_format.push_back(input_format);
      }
    } else {
      data_format = GetValue<std::vector<std::string>>(attr_value);
    }
    (*attr_json)[kJsonKeyValue] = data_format;
  } else {
    MS_LOG(WARNING) << "Invalid attr " << op_attr->name() << " found in node " << anf_node->fullname_with_scope()
                    << ", because its type: " << type
                    << " is not in supported list: [str, int, bool, float, listInt, listStr].";
  }
}

bool AkgKernelJsonGenerator::CreateAttrDescJson(const AnfNodePtr &anf_node, const OpInfoPtr &op_info,
                                                nlohmann::json *attrs_json) {
  auto attrs = op_info->attrs_ptr();
  if (attrs.empty()) {
    MS_LOG(DEBUG) << "Apply kernel [" << anf_node->fullname_with_scope() << "] op info attrs is empty";
    return true;
  }
  auto dyn_input_sizes = GetDynInputSizes(anf_node);
  auto primitive = GetCNodePrimitive(anf_node);

  // create input name list for "x_shape" in attr with "x" in primitive.
  auto inputs = op_info->inputs_ptr();
  std::map<std::string, size_t> op_info_shape_name;
  for (size_t i = 0; i < inputs.size(); i++) {
    op_info_shape_name[inputs[i]->name() + "_shape"] = i;
  }

  for (const auto &op_attr : attrs) {
    nlohmann::json attr_json;
    ValuePtr attr_value = primitive->GetAttr(op_attr->name());
    if (attr_value == nullptr && op_attr->name() != kJsonKeyDataformat) {
      if (op_attr->param_type() != "required") {
        continue;
      }
      // match "x_shape" in attr with "x" in primitive.
      auto find_item = op_info_shape_name.find(op_attr->name());
      if (find_item != op_info_shape_name.end()) {
        if (!dyn_input_sizes.empty()) {
          if (find_item->second >= dyn_input_sizes.size() - 1) {
            MS_LOG(EXCEPTION) << "dyn_input_sizes list index " << find_item->second << " is out of range [0, "
                              << dyn_input_sizes.size() - 1 << ") in node [" << anf_node->fullname_with_scope() << "]";
            return false;
          }
          size_t tensor_idx = LongToSize(std::accumulate(&dyn_input_sizes[0], &dyn_input_sizes[find_item->second], 0));
          for (int64_t input_i = 0; input_i < dyn_input_sizes[find_item->second]; input_i++) {
            attr_json[kJsonKeyValue] = this->cb_->GetInputInferShape(anf_node, tensor_idx);
            attr_json[kJsonKeyName] = op_attr->name();
            attrs_json->push_back(attr_json);
            tensor_idx++;
          }
        } else {
          attr_json[kJsonKeyValue] = this->cb_->GetInputInferShape(anf_node, find_item->second);
          attr_json[kJsonKeyName] = op_attr->name();
          attrs_json->push_back(attr_json);
        }
      } else {
        MS_LOG(ERROR) << "Can not find attr '" << op_attr->name() << "' in node [" << anf_node->fullname_with_scope()
                      << "]";
        return false;
      }
    } else {
      GetAttrJson(anf_node, dyn_input_sizes, op_attr, &attr_json, attr_value);
      attr_json[kJsonKeyName] = op_attr->name();
      attrs_json->push_back(attr_json);
    }
  }
  return true;
}

size_t AkgKernelJsonGenerator::GetInputTensorIdxInc(const AnfNodePtr &anf_node, size_t input_idx) {
  MS_EXCEPTION_IF_NULL(anf_node);
  auto cnode = anf_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (input_idx + 1 >= cnode->inputs().size()) {
    MS_EXCEPTION(ArgumentError) << "Input index " << input_idx << " is out of range [0, " << cnode->inputs().size()
                                << ") in node [" << cnode->DebugString() << "]";
  }

  auto input_node = cnode->input(input_idx + 1);
  if (input_tensor_idx_.find(input_node) == input_tensor_idx_.end()) {
    size_t index = input_tensor_idx_.size();
    input_tensor_idx_[input_node] = index;
  }

  return input_tensor_idx_[input_node];
}

size_t AkgKernelJsonGenerator::GetOutputTensorIdxInc() {
  size_t idx = output_tensor_idx_++;
  return idx;
}

std::string AkgKernelJsonGenerator::GetTensorName(const nlohmann::json &node_json, const std::string &tag,
                                                  const std::pair<size_t, size_t> &position) const {
  if (node_json.count(tag) == 0) {
    MS_LOG(ERROR) << "Node [" << node_json.dump() << "] has no key [" << tag << "].";
    return "";
  }

  auto const &tag_desc = node_json[tag];
  nlohmann::json first_index;
  if (tag == kJsonKeyOutputDesc) {
    first_index = tag_desc;
  } else if (!tag_desc.is_array() || tag_desc.size() <= position.first) {
    MS_LOG(ERROR) << "Access index is out of range: "
                  << " trying to access index " << position.first << " of node: " << tag_desc.dump();
    return "";
  } else {
    first_index = tag_desc[position.first];
  }

  if (!first_index.is_array() || first_index.size() <= position.second) {
    MS_LOG(ERROR) << "Access index is out of range: "
                  << " trying to access index " << position.second << " of node: " << first_index.dump();
    return "";
  }
  auto const &second_index = first_index[position.second];
  if (second_index.count(kJsonKeyTensorName) == 0) {
    MS_LOG(ERROR) << "Node [" << second_index.dump() << "] has no key [" << kJsonKeyTensorName << "].";
    return "";
  }

  return second_index[kJsonKeyTensorName];
}

void AkgKernelJsonGenerator::SetTensorName(const std::string &tag, const std::string &new_name,
                                           const std::pair<size_t, size_t> &position, nlohmann::json *node_json) const {
  MS_EXCEPTION_IF_NULL(node_json);
  if (node_json->count(tag) == 0) {
    MS_LOG(ERROR) << "Node [" << node_json->dump() << "] has no key [" << tag << "].";
    return;
  }

  nlohmann::json *tag_desc = &((*node_json)[tag]);
  nlohmann::json *first_index;
  if (tag == kJsonKeyOutputDesc) {
    first_index = tag_desc;
  } else if (!tag_desc->is_array() || tag_desc->size() <= position.first) {
    MS_LOG(ERROR) << "Access index is out of range: "
                  << " trying to access index " << position.first << " of node: " << tag_desc->dump();
    return;
  } else {
    first_index = &((*tag_desc)[position.first]);
  }

  if (!first_index->is_array() || first_index->size() <= position.second) {
    MS_LOG(ERROR) << "Access index is out of range: "
                  << " trying to access index " << position.second << " of node: " << first_index->dump();
    return;
  }
  nlohmann::json *second_index = &((*first_index)[position.second]);
  if (second_index->count(kJsonKeyTensorName) == 0) {
    MS_LOG(ERROR) << "Node [" << second_index->dump() << "] has no key [" << kJsonKeyTensorName << "].";
    return;
  }
  (*second_index)[kJsonKeyTensorName] = new_name;
  return;
}

void AkgKernelJsonGenerator::SaveNodeAddress(const AnfNodePtr &anf_node, nlohmann::json *node_json) {
  if (dump_option_.save_ptr_address) {
    std::ostringstream get_the_address;
    get_the_address << anf_node.get();
    auto address = get_the_address.str();
    (*node_json)[kJsonKeyPtrAddress] = address;
    address_node_map_[address] = anf_node;
  }
}

OpInfoPtr AkgKernelJsonGenerator::ExtractOpInfo(const AnfNodePtr &anf_node) const {
  if (dump_option_.extract_opinfo_from_anfnode) {
    OpInfoExtractor e;
    return e.Run(anf_node);
  } else {
#ifdef MSLITE_ENABLE_GRAPH_KERNEL
    MS_LOG(EXCEPTION) << "OpLib is not supported.";
#else
    return kernel::OpLib::FindOp(AnfUtils::GetCNodeName(anf_node), OpImplyType::kImplyAKG);
#endif
  }
}

std::string AkgKernelJsonGenerator::GetProcessorByTarget() const {
  auto target = cb_->GetTargetFromContext();
  if (target == kGPUDevice) {
    return "cuda";
  }
  if (target == kAscendDevice) {
    return "aicore";
  }
  return "cpu";
}

bool AkgKernelJsonGenerator::GenerateSingleKernelJson(const AnfNodePtr &anf_node, nlohmann::json *node_json) {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_EXCEPTION_IF_NULL(node_json);
  OpInfoPtr op_info = ExtractOpInfo(anf_node);
  MS_EXCEPTION_IF_NULL(op_info);

  // get basic params from currentNodeOpDesc
  std::string op_name;
  if (IsPrimitiveCNode(anf_node, prim::kPrimCustom)) {
    auto primitive = GetCNodePrimitive(anf_node);
    MS_EXCEPTION_IF_NULL(primitive);
    op_name = primitive->name();
  } else {
    op_name = op_info->op_name();
  }
  if (all_ops_name_.empty()) {
    all_ops_name_ = op_name;
  } else {
    static_cast<void>(all_ops_name_.append("_").append(op_name));
  }
  (*node_json)[kJsonKeyName] = op_name;
  (*node_json)[kJsonKeyImplPath] = op_info->impl_path();
  SaveNodeAddress(anf_node, node_json);

  // input desc
  nlohmann::json inputs_json;
  if (!CreateInputDescJson(anf_node, op_info, &inputs_json)) {
    MS_LOG(ERROR) << "Create input desc json failed, op[" << anf_node->fullname_with_scope() << "].";
    return false;
  }
  (*node_json)[kJsonKeyInputDesc] = inputs_json;
  MS_LOG(DEBUG) << "Akg create input desc json success.";

  // output desc
  nlohmann::json outputs_json;
  if (!CreateOutputDescJson(anf_node, op_info, &outputs_json)) {
    MS_LOG(ERROR) << "Create output desc json failed, op[" << anf_node->fullname_with_scope() << "].";
    return false;
  }
  (*node_json)[kJsonKeyOutputDesc] = outputs_json;
  MS_LOG(DEBUG) << "Akg create output desc json success.";

  // attribute desc
  nlohmann::json attrs_json;
  if (!CreateAttrDescJson(anf_node, op_info, &attrs_json)) {
    MS_LOG(ERROR) << "Create attr desc json failed, op[" << anf_node->fullname_with_scope() << "].";
    return false;
  }
  (*node_json)[kJsonKeyAttr] = attrs_json;
  return true;
}

size_t AkgKernelJsonGenerator::GetTensorSize(const nlohmann::json &node_json) const {
  const ShapeVector &shape = node_json[kJsonKeyShape];
  const std::string &dtype = node_json[kJsonKeyDataType];
  auto type_ptr = StringToType(dtype);
  MS_EXCEPTION_IF_NULL(type_ptr);
  auto num_ptr = type_ptr->cast<NumberPtr>();
  MS_EXCEPTION_IF_NULL(num_ptr);
  size_t nbyte = IntToSize(num_ptr->nbits() / static_cast<int>(BitsNum::eBits8));
  return std::accumulate(shape.begin(), shape.end(), nbyte, std::multiplies<size_t>());
}

void AkgKernelJsonGenerator::GetIOSize(const nlohmann::json &node_json, std::vector<size_t> *input_size,
                                       std::vector<size_t> *output_size) const {
  input_size->clear();
  output_size->clear();
  for (size_t i = 0; i < node_json[kJsonKeyInputDesc].size(); i++) {
    for (size_t m = 0; m < node_json[kJsonKeyInputDesc][i].size(); m++) {
      input_size->push_back(GetTensorSize(node_json[kJsonKeyInputDesc][i][m]));
    }
  }
  for (size_t i = 0; i < node_json[kJsonKeyOutputDesc].size(); i++) {
    output_size->push_back(GetTensorSize(node_json[kJsonKeyOutputDesc][i]));
  }
}

size_t AkgKernelJsonGenerator::GenHashId(const std::string &info) const {
  if (!dump_option_.save_ptr_address) {
    return std::hash<std::string>()(info);
  }
  // gen hash id without node address
  // the format is like {"ptr_address":"0x12345678"}
  std::string key = std::string("\"") + kJsonKeyPtrAddress + "\"";
  std::ostringstream result;
  size_t begin = 0;
  size_t pos;
  while ((pos = info.find(key, begin)) != std::string::npos) {
    result << info.substr(begin, pos - begin);
    // skip the address
    auto addr_begin = info.find('\"', pos + key.size());
    auto addr_end = info.find('\"', addr_begin + 1);
    begin = addr_end + 1;
  }
  result << info.substr(begin);
  return std::hash<std::string>()(result.str());
}

bool AkgKernelJsonGenerator::CollectJson(const AnfNodePtr &anf_node, nlohmann::json *kernel_json) {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_EXCEPTION_IF_NULL(kernel_json);
  std::string op_name = AnfUtils::GetCNodeName(anf_node);
  MS_LOG(DEBUG) << "Akg start generate kernel json desc, full scope name is : " << anf_node->fullname_with_scope();
  is_basic_op_ = true;
  if (!GenerateSingleKernelJson(anf_node, kernel_json)) {
    MS_LOG(ERROR) << "Op[" << anf_node->fullname_with_scope() << "] create single kernel json failed.";
    return false;
  }
  if (dump_option_.get_target_info) {
    TargetInfoSetter::Set(kernel_json);
  }
  (*kernel_json)[kJsonKeyProcess] = GetProcessorByTarget();
  (*kernel_json)[kJsonKeyVersion] = kCurrentInfoVersion;

  // gen hash id with the above info.
  size_t hash_id = GenHashId(kernel_json->dump());
  kernel_name_ = op_name + "_" + std::to_string(hash_id);
  if (dump_option_.gen_kernel_name_only) {
    return true;
  }
  (*kernel_json)[kJsonKeyId] = 0;  // unused key
  (*kernel_json)[kJsonKeyOp] = kernel_name_;
  const auto &flags = GraphKernelFlags::GetInstance();
  (*kernel_json)[kJsonKeyPlatform] = flags.kernel_generator;
  (*kernel_json)[kJsonKeyComposite] = false;

  GetIOSize(*kernel_json, &input_size_list_, &output_size_list_);

  MS_LOG(DEBUG) << "Akg create kernel json desc success, full scope name is : " << anf_node->fullname_with_scope()
                << ", json info name is : " << kernel_name_;
  return true;
}

void AkgKernelJsonGenerator::GenStitchJson(const std::vector<AnfNodePtr> &anf_nodes,
                                           std::map<AnfNodePtr, nlohmann::json> *node_json_map,
                                           nlohmann::json *kernel_json) const {
  std::vector<std::string> stitchs;
  for (auto const &anf_node : anf_nodes) {
    auto prim = GetCNodePrimitive(anf_node);
    MS_EXCEPTION_IF_NULL(prim);
    auto stitch_attr = prim->GetAttr(kAttrStitch);
    if (stitch_attr != nullptr && GetValue<std::string>(stitch_attr) == "common") {
      auto name = GetTensorName((*node_json_map)[anf_node], kJsonKeyOutputDesc, {0, 0});
      if (std::find(stitchs.begin(), stitchs.end(), name) == stitchs.end()) {
        (void)stitchs.emplace_back(name);
      }
    }
  }
  if (!stitchs.empty()) {
    std::vector<nlohmann::json> v;
    for (auto &s : stitchs) {
      std::vector<std::string> t(1, s);
      (void)v.emplace_back(std::move(t));
    }
    nlohmann::json stitch_json;
    stitch_json[kJsonKeyStitchOp] = v;
    (*kernel_json)[kJsonKeyBufferStitch] = stitch_json;
  }
}

void AkgKernelJsonGenerator::GenKernelName(const FuncGraphPtr &fg, size_t hash_id, nlohmann::json *kernel_json) {
  MS_EXCEPTION_IF_NULL(fg);
  // the final kernel name has a hash_id, and may has a "_more" suffix.
  // total len is up to about 105, file name (with ".info") is up to 110.
  constexpr size_t name_len_limited = 80;
  kernel_name_ = "Fused_";
  auto attr_val = fg->get_attr(FUNC_GRAPH_ATTR_GRAPH_KERNEL);
  std::string ops_name = (attr_val != nullptr) ? GetValue<std::string>(attr_val) : all_ops_name_;
  if (ops_name.size() > name_len_limited) {
    (*kernel_json)[kJsonKeyOpFullName] = kernel_name_ + ops_name;
    auto suffix_pos = ops_name.find_last_of("_");
    if (suffix_pos != std::string::npos && ops_name.size() - suffix_pos < name_len_limited) {
      ops_name =
        ops_name.substr(0, name_len_limited - (ops_name.size() - suffix_pos)) + "_more" + ops_name.substr(suffix_pos);
    } else {
      ops_name = ops_name.substr(0, name_len_limited) + "_more";
    }
  }
  (void)kernel_name_.append(ops_name).append("_");
  (void)kernel_name_.append(std::to_string(hash_id));
}

bool AkgKernelJsonGenerator::CollectFusedJson(const std::vector<AnfNodePtr> &anf_nodes,
                                              const std::vector<AnfNodePtr> &input_list,
                                              const std::vector<AnfNodePtr> &output_list, nlohmann::json *kernel_json) {
  if (anf_nodes.empty()) {
    MS_LOG(ERROR) << "anf_nodes list is empty";
    return false;
  }
  MS_LOG(DEBUG) << "Fusion nodes: [" << output_list.size() << "], input_list: [" << anf_nodes.size()
                << "], output_list: [" << input_list.size() << "].";
  std::map<AnfNodePtr, nlohmann::json> node_json_map;
  is_basic_op_ = false;
  dump_option_.extract_opinfo_from_anfnode = true;  // always extract from anfnode for composite ops.
  if (!GenSingleJsons(anf_nodes, &node_json_map)) {
    return false;
  }

  UpdateTensorName(anf_nodes, &node_json_map);

  std::vector<nlohmann::json> node_json_desc;
  (void)std::transform(anf_nodes.begin(), anf_nodes.end(), std::back_inserter(node_json_desc),
                       [&node_json_map](const AnfNodePtr &anf_node) { return node_json_map[anf_node]; });
  (*kernel_json)[kJsonKeyOpDesc] = node_json_desc;

  auto inputs_json = CreateInputsJson(anf_nodes, input_list, node_json_map);
  (*kernel_json)[kJsonKeyInputDesc] = inputs_json;
  (*kernel_json)[kJsonKeyOutputDesc] =
    CreateOutputsJson(anf_nodes, input_list, output_list, inputs_json, node_json_map);

  // Add parallel fusion information.
  GenParallelJson(anf_nodes, input_list, output_list, node_json_map, kernel_json);
  GenStitchJson(anf_nodes, &node_json_map, kernel_json);
  if (dump_option_.get_target_info) {
    TargetInfoSetter::Set(kernel_json);
  }
  (*kernel_json)[kJsonKeyProcess] = GetProcessorByTarget();
  (*kernel_json)[kJsonKeyVersion] = kCurrentInfoVersion;
  auto fg = anf_nodes[0]->func_graph();
  MS_EXCEPTION_IF_NULL(fg);
  if (fg->has_attr("dynamic_input_index")) {
    (*kernel_json)[kJsonKeyDynamicInputIndex] = GetValue<std::string>(fg->get_attr("dynamic_input_index"));
  }

  // gen hash id with the above info.
  size_t hash_id = GenHashId(kernel_json->dump());
  GenKernelName(fg, hash_id, kernel_json);
  if (dump_option_.gen_kernel_name_only) {
    return true;
  }
  (*kernel_json)[kJsonKeyId] = 0;  // unused key
  (*kernel_json)[kJsonKeyOp] = kernel_name_;
  const auto &flags = GraphKernelFlags::GetInstance();
  (*kernel_json)[kJsonKeyPlatform] = flags.kernel_generator;
  (*kernel_json)[kJsonKeyComposite] = true;
  (*kernel_json)[kJsonKeyCompositeGraph] = fg->ToString();
  if (fg->has_attr(kAttrNodeName)) {
    (*kernel_json)[kJsonKeyNodeName] = GetValue<std::string>(fg->get_attr(kAttrNodeName));
  }

  GetIOSize(*kernel_json, &input_size_list_, &output_size_list_);

  return true;
}

bool AkgKernelJsonGenerator::GenSingleJsons(const std::vector<AnfNodePtr> &anf_nodes,
                                            std::map<AnfNodePtr, nlohmann::json> *node_json_map) {
  for (auto const &anf_node : anf_nodes) {
    nlohmann::json node_json;
    if (!GenerateSingleKernelJson(anf_node, &node_json)) {
      MS_LOG(ERROR) << "Op [" << anf_node->fullname_with_scope() << "] create single kernel json failed.";
      return false;
    }

    auto primitive = GetCNodePrimitive(anf_node);
    MS_EXCEPTION_IF_NULL(primitive);

    (*node_json_map)[anf_node] = node_json;
  }
  return true;
}

void AkgKernelJsonGenerator::UpdateTensorName(const std::vector<AnfNodePtr> &anf_nodes,
                                              std::map<AnfNodePtr, nlohmann::json> *node_json_map) const {
  for (auto const &anf_node : anf_nodes) {
    auto dyn_input_sizes = GetDynInputSizes(anf_node);
    bool is_dynamic_input = !dyn_input_sizes.empty();
    size_t input_num = is_dynamic_input ? dyn_input_sizes.size() : AnfUtils::GetInputTensorNum(anf_node);
    size_t real_input_index = 0;
    for (size_t i = 0; i < input_num; ++i) {
      size_t input_tensor_num = is_dynamic_input ? LongToSize(dyn_input_sizes[i]) : 1;
      for (size_t j = 0; j < input_tensor_num; ++j) {
        auto tmp_input = GetKernelInput(anf_node, real_input_index);
        auto tmpi = i;
        auto tmpj = j;  // use tmpi and tmpj to evade pclint warning "for statement index variable modified in body."
        std::string tensor_name =
          GetTensorName((*node_json_map)[anf_node], kJsonKeyInputDesc, std::make_pair(tmpi, tmpj));
        if (node_json_map->find(tmp_input.first) != node_json_map->end()) {
          std::string new_tensor_name =
            GetTensorName((*node_json_map)[tmp_input.first], kJsonKeyOutputDesc, std::make_pair(0, tmp_input.second));
          SetTensorName(kJsonKeyInputDesc, new_tensor_name, std::make_pair(tmpi, tmpj), &((*node_json_map)[anf_node]));
          MS_LOG(DEBUG) << "Update [" << real_input_index << "] input [" << tensor_name << "] of ["
                        << anf_node->fullname_with_scope() << "] to [" << tmp_input.second << "] output ["
                        << new_tensor_name << "] of [" << tmp_input.first->fullname_with_scope() << "].";
        } else {
          MS_LOG(DEBUG) << "[" << real_input_index << "] input " << tensor_name << "] of ["
                        << anf_node->fullname_with_scope() << "] is out input.";
        }
        real_input_index++;
      }
    }
  }
}

nlohmann::json AkgKernelJsonGenerator::CreateInputsJson(const std::vector<AnfNodePtr> &anf_nodes,
                                                        const std::vector<AnfNodePtr> &input_list,
                                                        const std::map<AnfNodePtr, nlohmann::json> &node_json_map) {
  nlohmann::json inputs_json;
  auto input_index = GetInputIndex(anf_nodes, input_list);
  for (size_t i = 0; i < input_index.size(); ++i) {
    auto tmp_input = input_index[i];
    auto type_id = this->cb_->GetInputType(tmp_input.first, tmp_input.second.first);
    std::string dtype = TypeIdToString(type_id, true);
    nlohmann::json input_desc_json;
    input_desc_json[kJsonKeyTensorName] =
      GetTensorName(node_json_map.at(tmp_input.first), kJsonKeyInputDesc, tmp_input.second);
    input_desc_json[kJsonKeyDataType] = dtype;
    input_desc_json[kJsonKeyFormat] = this->cb_->GetInputFormat(tmp_input.first, tmp_input.second.first);
    auto input_shape = this->cb_->GetInputShape(tmp_input.first, tmp_input.second.first);
    if (input_shape.empty()) {
      input_shape.push_back(1);
    }
    input_desc_json[kJsonKeyShape] = input_shape;
    (void)inputs_json.emplace_back(std::vector<nlohmann::json>{input_desc_json});
  }
  return inputs_json;
}

void AkgKernelJsonGenerator::GenParallelJson(const std::vector<AnfNodePtr> &anf_nodes,
                                             const std::vector<AnfNodePtr> &input_list,
                                             const std::vector<AnfNodePtr> &output_list,
                                             const std::map<AnfNodePtr, nlohmann::json> &node_json_map,
                                             nlohmann::json *kernel_json) const {
  std::map<size_t, std::pair<size_t, std::vector<std::string>>> sub_graphs_info;
  std::string fusion_type;
  std::vector<std::vector<int>> type_info;

  auto output_index = GetOutputIndex(anf_nodes, input_list, output_list);
  for (size_t i = 0; i < output_index.size(); ++i) {
    auto [tmp_output, tmp_output_index] = output_index[i];
    bool found = std::any_of(input_list.cbegin(), input_list.cend(),
                             [&tmp_output](const AnfNodePtr &in) { return tmp_output == in; });
    if (!found) {
      auto tcnode = tmp_output->cast<CNodePtr>();
      if (tcnode == nullptr) {
        return;
      }
      auto prim = GetCNodePrimitive(tcnode);
      MS_EXCEPTION_IF_NULL(prim);
      // Get dim info.
      if (prim->HasAttr(kAttrParallelDimInfo)) {
        auto info = GetValue<std::vector<size_t>>(prim->GetAttr(kAttrParallelDimInfo));
        auto info_size = info.size();
        if (info_size != kAttrParallelDimInfoSize) {
          MS_LOG(EXCEPTION) << "The size of attr " << kAttrParallelDimInfo << " in node ["
                            << tcnode->fullname_with_scope() << "] should be " << kAttrParallelDimInfoSize
                            << ", but got " << info_size;
        }
        auto tensor_name =
          GetTensorName(node_json_map.at(tmp_output), kJsonKeyOutputDesc, std::make_pair(0, tmp_output_index));
        sub_graphs_info[info[0]].second.push_back(tensor_name);
        sub_graphs_info[info[0]].first = info[1];
      }
      // Get fusion type.
      if (prim->HasAttr(kAttrParallelFusionType)) {
        fusion_type = GetValue<std::string>(prim->GetAttr(kAttrParallelFusionType));
      }
      // Get fusion type info.
      if (prim->HasAttr(kAttrParallelTypeInfo)) {
        type_info = GetValue<std::vector<std::vector<int>>>(prim->GetAttr(kAttrParallelTypeInfo));
      }
    }
  }

  if (!sub_graphs_info.empty()) {
    nlohmann::json parallel_fusion_json;
    parallel_fusion_json[kJsonKeyFusionType] = fusion_type;
    parallel_fusion_json[kJsonKeyTypeInfo] = type_info;
    std::vector<std::vector<std::string>> sgraphs;
    std::vector<size_t> cnums;
    (void)std::for_each(
      sub_graphs_info.cbegin(), sub_graphs_info.cend(),
      [&sgraphs, &cnums](const std::pair<size_t, std::pair<size_t, std::vector<std::string>>> &sg_info) {
        sgraphs.push_back(sg_info.second.second);
        cnums.push_back(sg_info.second.first);
      });
    parallel_fusion_json[kJsonKeySubGraph] = sgraphs;
    parallel_fusion_json[kJsonKeyCoreNum] = cnums;

    (*kernel_json)[kJsonKeyParallelFusion] = parallel_fusion_json;
  }
}

nlohmann::json AkgKernelJsonGenerator::CreateOutputsJson(const std::vector<AnfNodePtr> &anf_nodes,
                                                         const std::vector<AnfNodePtr> &input_list,
                                                         const std::vector<AnfNodePtr> &output_list,
                                                         const nlohmann::json &inputs_json,
                                                         const std::map<AnfNodePtr, nlohmann::json> &node_json_map) {
  nlohmann::json outputs_json;
  auto output_index = GetOutputIndex(anf_nodes, input_list, output_list);
  for (size_t i = 0; i < output_index.size(); ++i) {
    auto tmp_output = output_index[i];
    bool found = false;
    nlohmann::json output_desc_json;
    for (size_t input_i = 0; input_i < input_list.size(); ++input_i) {
      if (tmp_output.first == input_list[input_i]) {
        output_desc_json = inputs_json[input_i][0];
        found = true;
        break;
      }
    }
    if (!found) {
      auto type_id = this->cb_->GetOutputType(tmp_output.first, tmp_output.second);
      std::string dtype = TypeIdToString(type_id, true);
      output_desc_json[kJsonKeyTensorName] =
        GetTensorName(node_json_map.at(tmp_output.first), kJsonKeyOutputDesc, std::make_pair(0, tmp_output.second));
      output_desc_json[kJsonKeyDataType] = dtype;
      output_desc_json[kJsonKeyFormat] = this->cb_->GetOutputFormat(tmp_output.first, tmp_output.second);
      auto output_shape = this->cb_->GetOutputShape(tmp_output.first, tmp_output.second);
      if (output_shape.empty()) {
        output_shape.push_back(1);
      }
      output_desc_json[kJsonKeyShape] = output_shape;
    }
    (void)outputs_json.emplace_back(output_desc_json);
  }
  return outputs_json;
}

bool AkgKernelJsonGenerator::CollectJson(const AnfNodePtr &anf_node) {
  kernel_json_ = nlohmann::json();
  return CollectJson(anf_node, &kernel_json_);
}

bool AkgKernelJsonGenerator::CollectFusedJson(const std::vector<AnfNodePtr> &anf_nodes,
                                              const std::vector<AnfNodePtr> &input_list,
                                              const std::vector<AnfNodePtr> &output_list) {
  kernel_json_ = nlohmann::json();
  return CollectFusedJson(anf_nodes, input_list, output_list, &kernel_json_);
}

bool AkgKernelJsonGenerator::CollectFusedJsonWithSingleKernel(const CNodePtr &c_node) {
  kernel_json_ = nlohmann::json();
  std::vector<AnfNodePtr> node_list, input_list, output_list;
  FuncGraphPtr fg = std::get<0>(BuildGraphFromNodes({c_node}));
  FuncGraphManagerPtr mng = GkUtils::GetFuncGraphManager(fg);
  auto out_cnode = fg->output()->cast<CNodePtr>();
  if (out_cnode == nullptr) {
    MS_LOG(ERROR) << "Wrong graph generated for kernel [" << c_node->fullname_with_scope()
                  << "], output cnode is a null pointer";
    return false;
  }
  // check all inputs in the cnodes: if it is a valuenode, replace it by a parameter
  std::set<AnfNodePtr> value_nodes;
  auto &inputs = out_cnode->inputs();
  for (size_t i = 1; i < inputs.size(); ++i) {
    const auto &tnode = inputs[i];
    auto tensor = GetValueNode(tnode);
    if (tensor) {
      (void)value_nodes.insert(tnode);
    }
  }

  for (const auto &vnode : value_nodes) {
    auto parameter = fg->add_parameter();
    parameter->set_abstract(vnode->abstract());
    parameter->set_kernel_info(vnode->kernel_info_ptr());
    (void)mng->Replace(vnode, parameter);
  }

  // add new parameter for the same inputs
  std::set<AnfNodePtr> inputs_set;
  bool changed = false;
  for (size_t i = 1; i < out_cnode->size(); i++) {
    auto inp = out_cnode->input(i);
    if (inputs_set.count(inp) == 0) {
      (void)inputs_set.insert(inp);
    } else {
      auto p = fg->add_parameter();
      p->set_abstract(inp->abstract());
      p->set_kernel_info(inp->kernel_info_ptr());
      out_cnode->set_input(i, p);
      changed = true;
    }
  }
  if (changed) {
    GkUtils::UpdateFuncGraphManager(mng, fg);
  }

  node_list.push_back(out_cnode);
  (void)input_list.insert(input_list.cbegin(), out_cnode->inputs().cbegin() + 1, out_cnode->inputs().cend());
  auto output_num = static_cast<int64_t>(AnfUtils::GetOutputTensorNum(out_cnode));
  if (output_num > 1) {
    for (int64_t idx = 0; idx < output_num; idx++) {
      auto gt =
        out_cnode->func_graph()->NewCNode({NewValueNode(prim::kPrimTupleGetItem), out_cnode, NewValueNode(idx)});
      (void)output_list.emplace_back(std::move(gt));
    }
  } else {
    output_list.push_back(out_cnode);
  }
  return CollectFusedJson(node_list, input_list, output_list, &kernel_json_);
}

namespace {
bool GetCpuInfo(nlohmann::json *target_info) {
  const auto &flags = GraphKernelFlags::GetInstance();
  std::string target_os = flags.target_os;
  std::string arch = flags.cpu_arch;
  std::string feature = flags.cpu_feature;
  std::string type = flags.cpu_type;
  std::set<std::string> valid_os = {"linux", "windows"};
  // arch: <{supported-features}, default-feature>
  std::map<std::string, std::pair<std::set<std::string>, std::string>> valid_features = {
    {"arm", {{"neon"}, "neon"}},
    {"aarch64", {{"neon"}, "neon"}},
    {"x86_64", {{"sse", "avx", "avx512"}, "avx"}},
  };
  std::set<std::string> valid_cpu_types = {"core-avx2", "skylake-avx512", "core-avx-i", "haswell", "skylake"};

  if (valid_os.count(target_os) == 0) {
    MS_LOG(WARNING) << "GraphKernelFlag: unsupported \"target_os\": " << target_os;
    target_os = "linux";
  }
  if (valid_features.count(arch) == 0) {
    if (!arch.empty()) {
      MS_LOG(WARNING) << "GraphKernelFlag: unsupported \"cpu_arch\": " << arch;
    }
#if defined(__arm__)
    arch = "arm";
#elif defined(__aarch64__)
    arch = "aarch64";
#else
    arch = "x86_64";
#endif
  }

  auto &features = valid_features[arch];
  if (features.first.count(feature) == 0) {
    if (!feature.empty()) {
      MS_LOG(WARNING) << "GraphKernelFlag: unsupported \"cpu_feature\": " << feature;
    }
    feature = features.second;
  }

  if (valid_cpu_types.count(type) == 0) {
    if (!type.empty()) {
      MS_LOG(WARNING) << "GraphKernelFlag: unsupported \"cpu_type\": " << type;
      type = "";
    }
    if (feature == "avx512") {
      type = "skylake-avx512";
    } else if (feature == "avx") {
      type = "core-avx2";
    }
  }

  (*target_info)[kJsonKeySystem] = target_os;
  (*target_info)[kJsonKeyCpuArch] = arch;
  (*target_info)[kJsonKeyCpuFeature] = feature;
  if (!type.empty()) {
    (*target_info)[kJsonKeyCpuType] = type;
  }
  return true;
}
}  // namespace

void TargetInfoSetter::GetTargetInfo() {
  auto target = Callback::Instance()->GetTargetFromContext();
#ifndef MSLITE_ENABLE_GRAPH_KERNEL
  if (target == kGPUDevice) {
    const auto &device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
      {kGPUDevice, MsContext::GetInstance()->get_param<uint32_t>(MS_CTX_DEVICE_ID)});
    MS_EXCEPTION_IF_NULL(device_context);
    auto deprecated_ptr = device_context->GetDeprecatedInterface();
    MS_EXCEPTION_IF_NULL(deprecated_ptr);
    auto major_version = deprecated_ptr->GetGPUCapabilityMajor();
    auto minor_version = deprecated_ptr->GetGPUCapabilityMinor();
    auto sm_count = deprecated_ptr->GetGPUMultiProcessorCount();
    if (major_version == -1 || minor_version == -1 || sm_count == -1) {
      has_info_ = false;
    } else {
      has_info_ = true;
      (target_info_)[kJsonKeyComputeCapability] = std::to_string(major_version) + "." + std::to_string(minor_version);
      (target_info_)[kJsonKeySmCount] = sm_count;
    }
    return;
  }
#endif
  if (target == kCPUDevice) {
    has_info_ = GetCpuInfo(&target_info_);
    return;
  }
}

void TargetInfoSetter::SetTargetInfo(nlohmann::json *kernel_info) const {
  if (has_info_) {
    (*kernel_info)[kJsonKeyTargetInfo] = target_info_;
  }
}
}  // namespace mindspore::graphkernel
