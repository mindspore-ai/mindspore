/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "backend/kernel_compiler/akg/akg_kernel_json_generator.h"

#include <algorithm>
#include <functional>
#include <map>
#include <set>
#include <sstream>
#include <tuple>
#include "backend/kernel_compiler/akg/akg_kernel_attrs_process.h"
#include "backend/kernel_compiler/common_utils.h"
#include "backend/kernel_compiler/oplib/oplib.h"
#include "backend/session/anf_runtime_algorithm.h"

namespace mindspore {
namespace kernel {
namespace {
std::vector<int> GetDynInputSize(const AnfNodePtr &anf_node) {
  std::vector<int> dyn_input_sizes;
  auto primitive = AnfAlgo::GetCNodePrimitive(anf_node);
  MS_EXCEPTION_IF_NULL(primitive);
  if (primitive->HasAttr(kAttrDynInputSizes)) {
    std::vector<int64_t> dyn_input_sizes_me =
      GetValue<const std::vector<int64_t>>(primitive->GetAttr(kAttrDynInputSizes));
    (void)std::transform(dyn_input_sizes_me.begin(), dyn_input_sizes_me.end(), std::back_inserter(dyn_input_sizes),
                         [](const int64_t &value) { return static_cast<int>(value); });
  }
  return dyn_input_sizes;
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
    op_info->set_op_name(AnfAlgo::GetCNodeName(cnode_));
    op_info->set_imply_type(OpImplyType::kAKG);
    ExtractInputs(op_info);
    ExtractOutputs(op_info);
    ExtractAttrs(op_info);
    return op_info;
  }

 private:
  void ExtractInputs(const OpInfoPtr &op_info) {
    auto dyn_input_sizes = GetDynInputSize(cnode_);
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

  void ExtractOutputs(const OpInfoPtr &op_info) {
    size_t output_tensor_num = AnfAlgo::GetOutputTensorNum(cnode_);
    for (size_t i = 0; i < output_tensor_num; i++) {
      auto io_info = std::make_shared<OpIOInfo>();
      io_info->set_name("output_" + std::to_string(i));
      op_info->add_outputs_ptr(io_info);
    }
  }

  bool ExcludeAttr(const std::string &name) {
    const std::set<std::string> black_list = {"IsFeatureMapInputList", "IsFeatureMapOutput", kAttrOutputNames,
                                              kAttrInputNames};
    return black_list.count(name) != 0;
  }

  void ExtractAttrs(const OpInfoPtr &op_info) {
    auto prim = GetCNodePrimitive(cnode_);
    if (prim == nullptr) return;
    for (const auto &[name, v] : prim->attrs()) {
      if (ExcludeAttr(name)) continue;
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
      } else if (v->isa<ValueList>() || v->isa<ValueTuple>()) {
        auto vec = v->isa<ValueList>() ? v->cast<ValueListPtr>()->value() : v->cast<ValueTuplePtr>()->value();
        if (vec.empty()) {
          op_attr->set_type("listInt");
        } else if (vec[0]->isa<Int32Imm>() || vec[0]->isa<Int64Imm>()) {
          op_attr->set_type("listInt");
        } else if (vec[0]->isa<StringImm>()) {
          op_attr->set_type("listStr");
        }
      }
      if (op_attr->type().empty()) {
        MS_LOG(DEBUG) << "Unknown type, ignore attr " << name;
        continue;
      }
      op_info->add_attrs_ptr(op_attr);
    }
  }

  CNodePtr cnode_;
};
}  // namespace

int AkgKernelJsonGenerator::op_cnt_ = 0;
std::mutex AkgKernelJsonGenerator::op_cnt_mtx_;

int AkgKernelJsonGenerator::GetOpCntInc() {
  op_cnt_mtx_.lock();
  int cnt = op_cnt_++;
  op_cnt_mtx_.unlock();
  return cnt;
}

inline TypeId AkgKernelJsonGenerator::GetInputDataType(const AnfNodePtr &anf_node, size_t real_index) {
  return dump_option_.is_before_select_kernel ? AnfAlgo::GetPrevNodeOutputInferDataType(anf_node, real_index)
                                              : AnfAlgo::GetInputDeviceDataType(anf_node, real_index);
}

inline std::vector<size_t> AkgKernelJsonGenerator::GetInputShape(const AnfNodePtr &anf_node, size_t real_index) {
  return dump_option_.is_before_select_kernel ? AnfAlgo::GetPrevNodeOutputInferShape(anf_node, real_index)
                                              : AnfAlgo::GetInputDeviceShape(anf_node, real_index);
}

inline std::string AkgKernelJsonGenerator::GetInputFormat(const AnfNodePtr &anf_node, size_t real_index) {
  return dump_option_.is_before_select_kernel ? kOpFormat_DEFAULT : AnfAlgo::GetInputFormat(anf_node, real_index);
}

inline TypeId AkgKernelJsonGenerator::GetOutputDataType(const AnfNodePtr &anf_node, size_t index) {
  return dump_option_.is_before_select_kernel ? AnfAlgo::GetOutputInferDataType(anf_node, index)
                                              : AnfAlgo::GetOutputDeviceDataType(anf_node, index);
}

inline std::vector<size_t> AkgKernelJsonGenerator::GetOutputShape(const AnfNodePtr &anf_node, size_t index) {
  return dump_option_.is_before_select_kernel ? AnfAlgo::GetOutputInferShape(anf_node, index)
                                              : AnfAlgo::GetOutputDeviceShape(anf_node, index);
}

inline std::string AkgKernelJsonGenerator::GetOutputFormat(const AnfNodePtr &anf_node, size_t index) {
  return dump_option_.is_before_select_kernel ? kOpFormat_DEFAULT : AnfAlgo::GetOutputFormat(anf_node, index);
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
  auto dyn_input_sizes = GetDynInputSize(anf_node);
  size_t real_input_index = 0;
  for (size_t i = 0; i < inputs_ptr.size(); i++) {
    auto input_ptr = inputs_ptr[i];
    if (input_ptr == nullptr) {
      MS_LOG(ERROR) << "Kernel [" << anf_node->fullname_with_scope() << "] input[" << i << "] is nullptr";
      return false;
    }

    size_t input_tensor_num = dyn_input_sizes.empty() ? 1 : IntToSize(dyn_input_sizes[i]);
    std::vector<nlohmann::json> input_list;
    for (size_t input_i = 0; input_i < input_tensor_num; input_i++) {
      auto type_id = this->GetInputDataType(anf_node, real_input_index);
      std::string dtype = TypeId2String(type_id, dump_option_.is_before_select_kernel);
      if (dtype.empty()) {
        MS_LOG(ERROR) << "Op [" << anf_node->fullname_with_scope() << "] input [" << real_input_index
                      << "] data type is null. ";
        return false;
      }
      nlohmann::json input_desc_json;
      input_desc_json[kJsonKeyDataType] = dtype;
      input_desc_json[kJsonKeyFormat] = this->GetInputFormat(anf_node, real_input_index);
      input_desc_json[kJsonKeyName] = input_ptr->name();
      input_desc_json[kJsonKeyTensorName] = "input_" + std::to_string(GetInputTensorIdxInc(anf_node, real_input_index));
      auto input_shape = this->GetInputShape(anf_node, real_input_index);
      if (!is_basic_op_ && GetInputTensorValue(anf_node, real_input_index, &input_desc_json)) {
        MS_LOG(DEBUG) << "Take input[" << real_input_index << "] of [" << anf_node->DebugString(2)
                      << "] as const tensor, shape: [" << Vector2Str(input_shape)
                      << "], value: " << input_desc_json[kJsonKeyValue];
        input_shape.clear();
      }
      if (input_shape.empty()) {
        input_shape.push_back(1);
      }
      input_desc_json[kJsonKeyShape] = input_shape;
      input_list.emplace_back(input_desc_json);
      real_input_index++;
    }
    inputs_json->emplace_back(input_list);
  }
  return true;
}

bool AkgKernelJsonGenerator::CreateOutputDescJson(const AnfNodePtr &anf_node, const OpInfoPtr &op_info,
                                                  nlohmann::json *outputs_json) {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_EXCEPTION_IF_NULL(op_info);
  MS_EXCEPTION_IF_NULL(outputs_json);
  size_t output_tensor_num = AnfAlgo::GetOutputTensorNum(anf_node);

  auto outputs = op_info->outputs_ptr();
  for (size_t i = 0; i < output_tensor_num; i++) {
    nlohmann::json output_json;
    auto type_id = this->GetOutputDataType(anf_node, i);
    std::string dtype = TypeId2String(type_id, dump_option_.is_before_select_kernel);
    if (dtype.empty()) {
      MS_LOG(ERROR) << "Op [" << anf_node->fullname_with_scope() << "] output [" << i << "] data type is null. ";
      return false;
    }

    std::string output_name = outputs[i]->name();
    output_json[kJsonKeyDataType] = dtype;
    output_json[kJsonKeyFormat] = this->GetOutputFormat(anf_node, i);
    output_json[kJsonKeyName] = output_name;
    output_json[kJsonKeyTensorName] = "output_" + std::to_string(i) + "_" + std::to_string(GetOutputTensorIdxInc());
    auto output_shape = this->GetOutputShape(anf_node, i);
    if (output_shape.empty()) {
      output_shape.push_back(1);
    }
    output_json[kJsonKeyShape] = output_shape;
    outputs_json->push_back(output_json);
  }
  return true;
}

void AkgKernelJsonGenerator::GetAttrJson(const AnfNodePtr &anf_node, const std::vector<int> &dyn_input_sizes,
                                         const OpAttrPtr &op_attr, nlohmann::json *attr_json,
                                         const ValuePtr &attr_value) {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_EXCEPTION_IF_NULL(op_attr);
  MS_EXCEPTION_IF_NULL(attr_json);
  std::string type = op_attr->type();
  (*attr_json)[kJsonKeyDataType] = type;
  if (type == "int") {
    (*attr_json)[kJsonKeyValue] = static_cast<int>(GetValue<int64_t>(attr_value));
  } else if (type == "str") {
    (*attr_json)[kJsonKeyValue] = GetValue<std::string>(attr_value);
  } else if (type == "bool") {
    (*attr_json)[kJsonKeyValue] = GetValue<bool>(attr_value);
  } else if (type == "float") {
    (*attr_json)[kJsonKeyValue] = GetValue<float>(attr_value);
  } else if (type == "listInt") {
    std::vector<int> list_int;
    std::vector<int64_t> list_int_me = GetValue<std::vector<int64_t>>(attr_value);
    (void)std::transform(list_int_me.begin(), list_int_me.end(), std::back_inserter(list_int),
                         [](const int64_t &value) { return static_cast<int>(value); });
    (*attr_json)[kJsonKeyValue] = list_int;
  } else if (type == "listStr") {
    std::vector<std::string> data_format;
    if (op_attr->name() == kArgDataformat) {
      size_t tensor_args_num = !dyn_input_sizes.empty() ? dyn_input_sizes.size() : AnfAlgo::GetInputTensorNum(anf_node);
      for (size_t format_i = 0; format_i < tensor_args_num; format_i++) {
        auto input_format = this->GetInputFormat(anf_node, format_i);
        data_format.push_back(input_format);
      }
    } else {
      data_format = GetValue<std::vector<std::string>>(attr_value);
    }
    (*attr_json)[kJsonKeyValue] = data_format;
  } else {
    MS_LOG(WARNING) << "No valid json value for attr type: " << type;
  }
}

bool AkgKernelJsonGenerator::CreateAttrDescJson(const AnfNodePtr &anf_node, const OpInfoPtr &op_info,
                                                nlohmann::json *attrs_json) {
  auto attrs = op_info->attrs_ptr();
  if (attrs.empty()) {
    MS_LOG(DEBUG) << "Apply kernel [" << anf_node->fullname_with_scope() << "] op info attrs is empty";
    return true;
  }
  auto dyn_input_sizes = GetDynInputSize(anf_node);
  auto primitive = AnfAlgo::GetCNodePrimitive(anf_node);

  // create input name list for "x_shape" in attr with "x" in primitive.
  auto inputs = op_info->inputs_ptr();
  std::map<std::string, size_t> op_info_shape_name;
  for (size_t i = 0; i < inputs.size(); i++) {
    op_info_shape_name[inputs[i]->name() + "_shape"] = i;
  }

  for (const auto &op_attr : attrs) {
    nlohmann::json attr_json;
    ValuePtr attr_value = primitive->GetAttr(op_attr->name());
    if (attr_value == nullptr && op_attr->name() != kArgDataformat) {
      if (op_attr->param_type() != "required") continue;
      // match "x_shape" in attr with "x" in primitive.
      auto find_item = op_info_shape_name.find(op_attr->name());
      if (find_item != op_info_shape_name.end()) {
        if (!dyn_input_sizes.empty()) {
          if (find_item->second >= dyn_input_sizes.size() - 1) {
            MS_LOG(EXCEPTION) << "dyn_input_sizes list index:" << find_item->second
                              << " is out of range:" << dyn_input_sizes.size() - 1 << ".";
            return false;
          }
          size_t tensor_idx = IntToSize(std::accumulate(&dyn_input_sizes[0], &dyn_input_sizes[find_item->second], 0));
          for (int input_i = 0; input_i < dyn_input_sizes[find_item->second]; input_i++) {
            attr_json[kJsonKeyValue] = AnfAlgo::GetPrevNodeOutputInferShape(anf_node, tensor_idx);
            attr_json[kJsonKeyName] = op_attr->name();
            attrs_json->push_back(attr_json);
            tensor_idx++;
          }
        } else {
          attr_json[kJsonKeyValue] = AnfAlgo::GetPrevNodeOutputInferShape(anf_node, find_item->second);
          attr_json[kJsonKeyName] = op_attr->name();
          attrs_json->push_back(attr_json);
        }
      } else {
        MS_LOG(ERROR) << "op [" << anf_node->fullname_with_scope() << "] should have attr :" << op_attr->name();
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
    MS_EXCEPTION(ArgumentError) << "input_idx [" << input_idx << "] is out of index of inputs of ["
                                << cnode->inputs().size() - 1 << "][" << cnode->DebugString() << "]";
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
                                                  const std::pair<size_t, size_t> &position) {
  if (node_json.count(tag) == 0) {
    MS_LOG(ERROR) << "Node [" << node_json.dump() << "] has no key [" << tag << "].";
    return "";
  }

  auto const &tag_desc = node_json[tag];
  nlohmann::json first_index;
  if (tag == kJsonKeyOutputDesc) {
    first_index = tag_desc;
  } else if (!tag_desc.is_array() || tag_desc.size() <= position.first) {
    MS_LOG(ERROR) << "Node [" << tag_desc.dump() << "] has no enough value [" << position.first << "].";
    return "";
  } else {
    first_index = tag_desc[position.first];
  }

  if (!first_index.is_array() || first_index.size() <= position.second) {
    MS_LOG(ERROR) << "Node [" << first_index.dump() << "] has no enough value [" << position.second << "].";
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
                                           const std::pair<size_t, size_t> &position, nlohmann::json *node_json) {
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
    MS_LOG(ERROR) << "Node [" << tag_desc->dump() << "] has no enough value [" << position.first << "].";
    return;
  } else {
    first_index = &((*tag_desc)[position.first]);
  }

  if (!first_index->is_array() || first_index->size() <= position.second) {
    MS_LOG(ERROR) << "Node [" << first_index->dump() << "] has no enough value [" << position.second << "].";
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

OpInfoPtr AkgKernelJsonGenerator::ExtractOpInfo(const AnfNodePtr &anf_node) {
  if (dump_option_.extract_opinfo_from_anfnode) {
    return OpInfoExtractor().Run(anf_node);
  } else {
    return mindspore::kernel::OpLib::FindOp(AnfAlgo::GetCNodeName(anf_node), OpImplyType::kAKG);
  }
}

bool AkgKernelJsonGenerator::GenerateSingleKernelJson(const AnfNodePtr &anf_node, nlohmann::json *node_json) {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_EXCEPTION_IF_NULL(node_json);
  OpInfoPtr op_info = ExtractOpInfo(anf_node);
  MS_EXCEPTION_IF_NULL(op_info);

  // get basic params from currentNodeOpDesc
  (*node_json)[kJsonKeyName] = op_info->op_name();
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

bool AkgKernelJsonGenerator::GetIOSize(const nlohmann::json &node_json, std::vector<size_t> *input_size,
                                       std::vector<size_t> *output_size) {
  if (input_size == nullptr || output_size == nullptr) {
    MS_LOG(ERROR) << "input size or output size is nullptr";
    return false;
  }
  input_size->clear();
  output_size->clear();

  for (size_t i = 0; i < node_json[kJsonKeyInputDesc].size(); i++) {
    for (size_t m = 0; m < node_json[kJsonKeyInputDesc][i].size(); m++) {
      std::string dtype = node_json[kJsonKeyInputDesc][i][m][kJsonKeyDataType];
      size_t nbyte = GetDtypeNbyte(dtype);
      size_t size_i =
        std::accumulate(node_json[kJsonKeyInputDesc][i][m][kJsonKeyShape].begin(),
                        node_json[kJsonKeyInputDesc][i][m][kJsonKeyShape].end(), nbyte, std::multiplies<size_t>());
      input_size->push_back(size_i);
    }
  }

  for (size_t i = 0; i < node_json[kJsonKeyOutputDesc].size(); i++) {
    std::string dtype = node_json[kJsonKeyOutputDesc][i][kJsonKeyDataType];
    size_t nbyte = GetDtypeNbyte(dtype);
    size_t size_i =
      std::accumulate(node_json[kJsonKeyOutputDesc][i][kJsonKeyShape].begin(),
                      node_json[kJsonKeyOutputDesc][i][kJsonKeyShape].end(), nbyte, std::multiplies<size_t>());
    output_size->push_back(size_i);
  }

  return true;
}

bool AkgKernelJsonGenerator::CollectJson(const AnfNodePtr &anf_node, nlohmann::json *kernel_json) {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_EXCEPTION_IF_NULL(kernel_json);
  std::string op_name = AnfAlgo::GetCNodeName(anf_node);
  MS_LOG(DEBUG) << "Akg start generate kernel json desc, full scope name is : " << anf_node->fullname_with_scope();
  SetAkgKernelAttrs(anf_node);
  is_basic_op_ = true;
  if (!GenerateSingleKernelJson(anf_node, kernel_json)) {
    MS_LOG(ERROR) << "Op[" << anf_node->fullname_with_scope() << "] create single kernel json failed.";
    return false;
  }

  size_t hash_id = std::hash<std::string>()(kernel_json->dump());
  kernel_name_ = op_name + "_";
  (void)kernel_name_.append(std::to_string(hash_id));
  (*kernel_json)[kJsonKeyId] = GetOpCntInc();
  (*kernel_json)[kJsonKeyOp] = kernel_name_;
  (*kernel_json)[kJsonKeyPlatform] = "AKG";
  (*kernel_json)[kJsonKeyProcess] = GetProcessorStr(anf_node);
  (*kernel_json)[kJsonKeyComposite] = false;

  if (!GetIOSize(*kernel_json, &input_size_list_, &output_size_list_)) {
    MS_LOG(ERROR) << "Cal mem size failed.";
    return false;
  }

  MS_LOG(DEBUG) << "Akg create kernel json desc success, full scope name is : " << anf_node->fullname_with_scope()
                << ", json info name is : " << kernel_name_;
  return true;
}

void AkgKernelJsonGenerator::GenStitchJson(const std::vector<AnfNodePtr> &anf_nodes,
                                           std::map<AnfNodePtr, nlohmann::json> *node_json_map,
                                           nlohmann::json *kernel_json) {
  std::vector<std::string> stitchs;
  for (auto const &anf_node : anf_nodes) {
    if (AnfAlgo::HasNodeAttr(kAttrStitch, anf_node->cast<CNodePtr>()) &&
        AnfAlgo::GetNodeAttr<std::string>(anf_node, kAttrStitch) == "common") {
      auto name = GetTensorName((*node_json_map)[anf_node], kJsonKeyOutputDesc, {0, 0});
      if (std::find(stitchs.begin(), stitchs.end(), name) == stitchs.end()) {
        stitchs.emplace_back(name);
      }
    }
  }
  if (!stitchs.empty()) {
    std::vector<nlohmann::json> v;
    for (auto &s : stitchs) {
      std::vector<std::string> t;
      t.emplace_back(s);
      v.emplace_back(t);
    }
    nlohmann::json stitch_json;
    stitch_json[kJsonKeyStitchOp] = v;
    (*kernel_json)[kJsonKeyBufferStitch] = stitch_json;
  }
}
bool AkgKernelJsonGenerator::CollectFusedJson(const std::vector<AnfNodePtr> &anf_nodes,
                                              const std::vector<AnfNodePtr> &input_list,
                                              const std::vector<AnfNodePtr> &output_list, nlohmann::json *kernel_json) {
  if (anf_nodes.empty()) {
    MS_LOG(ERROR) << "Invalid input size, anf_nodes [" << anf_nodes.size() << "], input_list [" << input_list.size()
                  << "].";
    return false;
  }
  MS_LOG(DEBUG) << "Fusion nodes: [" << output_list.size() << "], input_list: [" << anf_nodes.size()
                << "], output_list: [" << input_list.size() << "].";
  std::map<AnfNodePtr, nlohmann::json> node_json_map;
  is_basic_op_ = false;
  dump_option_.extract_opinfo_from_anfnode = true;  // always extract from anfnode for composite ops.
  if (!GenSingleJsons(anf_nodes, &node_json_map)) return false;

  UpdateTensorName(anf_nodes, &node_json_map);

  std::vector<nlohmann::json> node_json_desc;
  std::transform(anf_nodes.begin(), anf_nodes.end(), std::back_inserter(node_json_desc),
                 [&node_json_map](const AnfNodePtr &anf_node) { return node_json_map[anf_node]; });
  (*kernel_json)[kJsonKeyOpDesc] = node_json_desc;

  auto inputs_json = CreateInputsJson(anf_nodes, input_list, node_json_map);
  (*kernel_json)[kJsonKeyInputDesc] = inputs_json;
  (*kernel_json)[kJsonKeyOutputDesc] =
    CreateOutputsJson(anf_nodes, input_list, output_list, inputs_json, node_json_map);

  // Add parallel fusion information.
  GenParallelJson(anf_nodes, input_list, output_list, node_json_map, kernel_json);

  size_t hash_id = std::hash<std::string>()(kernel_json->dump());
  kernel_name_ = "Fused_";
  auto fg = anf_nodes[0]->func_graph();
  MS_EXCEPTION_IF_NULL(fg);
  auto attr_val = fg->get_attr(FUNC_GRAPH_ATTR_GRAPH_KERNEL);
  constexpr size_t name_len_limited = 80;
  if (attr_val != nullptr) {
    auto fg_name = GetValue<std::string>(attr_val);
    if (fg_name.size() > name_len_limited) {
      (*kernel_json)[kJsonKeyOpFullName] = kernel_name_ + fg_name;
      auto suffix_pos = fg_name.find_last_of("_");
      fg_name =
        fg_name.substr(0, name_len_limited - fg_name.size() + suffix_pos) + "_more" + fg_name.substr(suffix_pos);
    }
    static_cast<void>(kernel_name_.append(fg_name).append("_"));
  }
  static_cast<void>(kernel_name_.append(std::to_string(hash_id)));
  (*kernel_json)[kJsonKeyId] = GetOpCntInc();
  (*kernel_json)[kJsonKeyOp] = kernel_name_;
  (*kernel_json)[kJsonKeyPlatform] = "AKG";
  (*kernel_json)[kJsonKeyProcess] = GetProcessorStr(anf_nodes[0]);
  (*kernel_json)[kJsonKeyComposite] = true;
  (*kernel_json)[kJsonKeyCompositeGraph] = fg->ToString() + "." + fg->debug_info()->get_id();

  GenStitchJson(anf_nodes, &node_json_map, kernel_json);

  if (!GetIOSize(*kernel_json, &input_size_list_, &output_size_list_)) {
    MS_LOG(ERROR) << "Cal mem size failed.";
    return false;
  }

  return true;
}

bool AkgKernelJsonGenerator::GenSingleJsons(const std::vector<AnfNodePtr> &anf_nodes,
                                            std::map<AnfNodePtr, nlohmann::json> *node_json_map) {
  for (auto const &anf_node : anf_nodes) {
    MS_EXCEPTION_IF_NULL(anf_node);
    if (!AnfAlgo::IsRealKernel(anf_node)) {
      MS_LOG(ERROR) << "Invalid anf node to build [" << anf_node->fullname_with_scope() << "].";
      return false;
    }
    SetAkgKernelAttrs(anf_node);

    nlohmann::json node_json;
    if (!GenerateSingleKernelJson(anf_node, &node_json)) {
      MS_LOG(ERROR) << "Op [" << anf_node->fullname_with_scope() << "] create single kernel json failed.";
      return false;
    }

    auto primitive = AnfAlgo::GetCNodePrimitive(anf_node);
    MS_EXCEPTION_IF_NULL(primitive);

    if (primitive->GetAttr("fusion") != nullptr) {
      node_json["fusion"] = primitive->GetAttr("fusion")->ToString();
    }

    (*node_json_map)[anf_node] = node_json;
  }
  return true;
}

void AkgKernelJsonGenerator::UpdateTensorName(const std::vector<AnfNodePtr> &anf_nodes,
                                              std::map<AnfNodePtr, nlohmann::json> *node_json_map) {
  for (auto const &anf_node : anf_nodes) {
    auto dyn_input_sizes = GetDynInputSize(anf_node);
    bool is_dynamic_input = !dyn_input_sizes.empty();
    size_t input_num = is_dynamic_input ? dyn_input_sizes.size() : AnfAlgo::GetInputTensorNum(anf_node);
    size_t real_input_index = 0;
    for (size_t i = 0; i < input_num; ++i) {
      size_t input_tensor_num = is_dynamic_input ? IntToSize(dyn_input_sizes[i]) : 1;
      for (size_t j = 0; j < input_tensor_num; ++j) {
        auto tmp_input = GetKernelInput(anf_node, real_input_index);
        std::string tensor_name = GetTensorName((*node_json_map)[anf_node], kJsonKeyInputDesc, std::make_pair(i, j));
        if (node_json_map->find(tmp_input.first) != node_json_map->end()) {
          std::string new_tensor_name =
            GetTensorName((*node_json_map)[tmp_input.first], kJsonKeyOutputDesc, std::make_pair(0, tmp_input.second));
          SetTensorName(kJsonKeyInputDesc, new_tensor_name, std::make_pair(i, j), &((*node_json_map)[anf_node]));
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
    auto type_id = this->GetInputDataType(tmp_input.first, tmp_input.second.first);
    std::string dtype = TypeId2String(type_id, dump_option_.is_before_select_kernel);
    nlohmann::json input_desc_json;
    input_desc_json[kJsonKeyTensorName] =
      GetTensorName(node_json_map.at(tmp_input.first), kJsonKeyInputDesc, tmp_input.second);
    input_desc_json[kJsonKeyDataType] = dtype;
    input_desc_json[kJsonKeyFormat] = this->GetInputFormat(tmp_input.first, tmp_input.second.first);
    auto input_shape = this->GetInputShape(tmp_input.first, tmp_input.second.first);
    if (input_shape.empty()) {
      input_shape.push_back(1);
    }
    input_desc_json[kJsonKeyShape] = input_shape;
    inputs_json.emplace_back(std::vector<nlohmann::json>{input_desc_json});
  }
  return inputs_json;
}

void AkgKernelJsonGenerator::GenParallelJson(const std::vector<AnfNodePtr> &anf_nodes,
                                             const std::vector<AnfNodePtr> &input_list,
                                             const std::vector<AnfNodePtr> &output_list,
                                             const std::map<AnfNodePtr, nlohmann::json> &node_json_map,
                                             nlohmann::json *kernel_json) {
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
      // Get dim info.
      if (AnfAlgo::HasNodeAttr(kAttrParallelDimInfo, tcnode)) {
        auto info = AnfAlgo::GetNodeAttr<std::vector<size_t>>(tcnode, kAttrParallelDimInfo);
        if (info.size() != 2) {
          MS_LOG(EXCEPTION) << "Parallel dim info is invalid!";
        }
        auto tensor_name =
          GetTensorName(node_json_map.at(tmp_output), kJsonKeyOutputDesc, std::make_pair(0, tmp_output_index));
        sub_graphs_info[info[0]].second.push_back(tensor_name);
        sub_graphs_info[info[0]].first = info[1];
      }
      // Get fusion type.
      if (AnfAlgo::HasNodeAttr(kAttrParallelFusionType, tcnode)) {
        fusion_type = AnfAlgo::GetNodeAttr<std::string>(tcnode, kAttrParallelFusionType);
      }
      // Get fusion type info.
      if (AnfAlgo::HasNodeAttr(kAttrParallelTypeInfo, tcnode)) {
        type_info = AnfAlgo::GetNodeAttr<std::vector<std::vector<int>>>(tcnode, kAttrParallelTypeInfo);
      }
    }
  }

  if (!sub_graphs_info.empty()) {
    auto processor = GetProcessorStr(anf_nodes[0]);
    if (processor != kProcessorCuda) {
      MS_LOG(EXCEPTION) << "Parallel fusion not support " << processor << " now.";
    }

    nlohmann::json parallel_fusion_json;
    parallel_fusion_json[kJsonKeyFusionType] = fusion_type;
    parallel_fusion_json[kJsonKeyTypeInfo] = type_info;
    std::vector<std::vector<std::string>> sgraphs;
    std::vector<size_t> cnums;
    std::for_each(sub_graphs_info.cbegin(), sub_graphs_info.cend(),
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
      auto type_id = this->GetOutputDataType(tmp_output.first, tmp_output.second);
      std::string dtype = TypeId2String(type_id, dump_option_.is_before_select_kernel);
      output_desc_json[kJsonKeyTensorName] =
        GetTensorName(node_json_map.at(tmp_output.first), kJsonKeyOutputDesc, std::make_pair(0, tmp_output.second));
      output_desc_json[kJsonKeyDataType] = dtype;
      output_desc_json[kJsonKeyFormat] = this->GetOutputFormat(tmp_output.first, tmp_output.second);
      auto output_shape = this->GetOutputShape(tmp_output.first, tmp_output.second);
      if (output_shape.empty()) {
        output_shape.push_back(1);
      }
      output_desc_json[kJsonKeyShape] = output_shape;
    }
    outputs_json.emplace_back(output_desc_json);
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
}  // namespace kernel
}  // namespace mindspore
