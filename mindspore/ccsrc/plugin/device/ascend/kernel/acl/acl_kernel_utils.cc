/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "plugin/device/ascend/kernel/acl/acl_kernel_utils.h"
#include <string>
#include <map>
#include <functional>
#include "ir/value.h"
#include "include/common/utils/utils.h"
#include "include/common/utils/anfalgo.h"
#include "kernel/common_utils.h"
#include "backend/common/session/anf_runtime_algorithm.h"

namespace mindspore {
namespace kernel {
namespace {
static const std::map<TypeId, aclDataType> kMsTypeToAclType = {
  {kNumberTypeBool, ACL_BOOL},     {kNumberTypeInt, ACL_INT32},     {kNumberTypeInt8, ACL_INT8},
  {kNumberTypeInt16, ACL_INT16},   {kNumberTypeInt32, ACL_INT32},   {kNumberTypeInt64, ACL_INT64},
  {kNumberTypeUInt, ACL_UINT32},   {kNumberTypeUInt8, ACL_UINT8},   {kNumberTypeUInt16, ACL_UINT16},
  {kNumberTypeUInt32, ACL_UINT32}, {kNumberTypeUInt64, ACL_UINT64}, {kNumberTypeFloat16, ACL_FLOAT16},
  {kNumberTypeFloat, ACL_FLOAT},   {kNumberTypeFloat32, ACL_FLOAT}, {kNumberTypeFloat64, ACL_DOUBLE}};

static const std::map<std::string, aclFormat> kMsFormatToAclFormat = {{kOpFormat_DEFAULT, ACL_FORMAT_NCHW},
                                                                      {kOpFormat_NCHW, ACL_FORMAT_NCHW},
                                                                      {kOpFormat_NHWC, ACL_FORMAT_NHWC},
                                                                      {kOpFormat_ND, ACL_FORMAT_ND},
                                                                      {kOpFormat_NC1HWC0, ACL_FORMAT_NC1HWC0},
                                                                      {kOpFormat_FRACTAL_Z, ACL_FORMAT_FRACTAL_Z},
                                                                      {kOpFormat_NC1HWC0_C04, ACL_FORMAT_NC1HWC0_C04},
                                                                      {kOpFormat_NDHWC, ACL_FORMAT_NDHWC},
                                                                      {kOpFormat_FRAC_NZ, ACL_FORMAT_FRACTAL_NZ},
                                                                      {kOpFormat_NCDHW, ACL_FORMAT_NCDHW},
                                                                      {kOpFormat_NDC1HWC0, ACL_FORMAT_NDC1HWC0}};

static const std::map<std::string, aclFormat> kMsSpecOriginFormat = {{"BatchMatMul", ACL_FORMAT_ND},
                                                                     {"MatMul", ACL_FORMAT_ND}};

std::map<std::string, std::string> GetConvertAttr(const std::string &op_type) {
  std::map<std::string, std::string> attrs;
  static const std::map<std::string, std::map<std::string, std::string>> op_type_map = {
    {"Conv2D", {{"pad_list", "pads"}, {"dilation", "dilations"}, {"stride", "strides"}, {"format", "data_format"}}},
    {"Conv2DBackpropInput",
     {{"pad_list", "pads"}, {"dilation", "dilations"}, {"stride", "strides"}, {"format", "data_format"}}},
    {"Conv2DBackpropFilter",
     {{"pad_list", "pads"}, {"dilation", "dilations"}, {"stride", "strides"}, {"format", "data_format"}}},
    {"BatchMatMul", {{"transpose_x1", "adj_x1"}, {"transpose_x2", "adj_x2"}}}};
  auto iter = op_type_map.find(op_type);
  return iter == op_type_map.end() ? attrs : iter->second;
}
}  // namespace

AclOpDesc::AclOpDesc(const std::string &op_type) {
  op_type_ = op_type;
  acl_attr_ = aclopCreateAttr();
}

AclOpDesc::~AclOpDesc() {
  aclopDestroyAttr(acl_attr_);
  for (auto *input_desc : input_tensor_desc_) {
    if (input_desc != nullptr) {
      aclDestroyTensorDesc(input_desc);
    }
  }
  for (auto *output_desc : output_tensor_desc_) {
    if (output_desc != nullptr) {
      aclDestroyTensorDesc(output_desc);
    }
  }
  for (auto *input_data : input_tensor_data_) {
    aclDestroyDataBuffer(input_data);
  }
  for (auto *output_data : output_tensor_data_) {
    aclDestroyDataBuffer(output_data);
  }
}

void AclOpDesc::AddInputTensor(const AnfNodePtr &anf_node, const size_t input_num,
                               const std::vector<AddressPtr> &inputs, const std::vector<size_t> &input_size_list,
                               const std::string &op_type) {
  MS_EXCEPTION_IF_NULL(anf_node);
  for (size_t i = 0; i < input_num; ++i) {
    auto real_index = AnfAlgo::GetInputIndexInGraph(anf_node, i);
    if (real_index >= input_num) {
      MS_LOG(EXCEPTION) << "Error index for current node:" << anf_node->fullname_with_scope() << " and index is "
                        << real_index;
    }
    auto ori_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(anf_node, real_index);

    auto input_shape = AnfAlgo::GetInputDeviceShape(anf_node, real_index);
    auto input_type = AnfAlgo::GetInputDeviceDataType(anf_node, real_index);
    auto input_format = AnfAlgo::GetInputFormat(anf_node, real_index);

    auto acl_type = AclUtils::ConvertTypeIdToAclType(input_type);
    auto acl_format = AclUtils::ConvertFormatToAclFormat(input_format);
    auto ori_iter = kMsSpecOriginFormat.find(op_type);
    auto ori_format = (ori_iter == kMsSpecOriginFormat.end()) ? ACL_FORMAT_NCHW : ori_iter->second;

    auto input_desc = aclCreateTensorDesc(acl_type, ori_shape.size(), ori_shape.data(), ori_format);
    MS_EXCEPTION_IF_NULL(input_desc);
    if (aclSetTensorShape(input_desc, input_shape.size(), input_shape.data())) {
      MS_LOG(EXCEPTION) << "Acl set tensor shape failed!";
    }
    if (aclSetTensorFormat(input_desc, acl_format)) {
      MS_LOG(EXCEPTION) << "Acl set tensor format failed!";
    }
    auto input_data = aclCreateDataBuffer(inputs[i]->addr, input_size_list[real_index]);
    MS_EXCEPTION_IF_NULL(input_data);
    (void)input_tensor_desc_.emplace_back(input_desc);
    (void)input_tensor_data_.emplace_back(input_data);
  }
}

void AclOpDesc::AddOutputTensor(const AnfNodePtr &anf_node, const size_t output_num,
                                const std::vector<AddressPtr> &outputs, const std::vector<size_t> &output_size_list,
                                const std::string &op_type) {
  MS_EXCEPTION_IF_NULL(anf_node);
  for (size_t i = 0; i < output_num; ++i) {
    auto ori_shape = common::AnfAlgo::GetOutputInferShape(anf_node, i);

    auto output_shape = AnfAlgo::GetOutputDeviceShape(anf_node, i);
    auto output_type = AnfAlgo::GetOutputDeviceDataType(anf_node, i);
    auto output_format = AnfAlgo::GetOutputFormat(anf_node, i);

    auto acl_type = AclUtils::ConvertTypeIdToAclType(output_type);
    auto acl_format = AclUtils::ConvertFormatToAclFormat(output_format);
    auto ori_iter = kMsSpecOriginFormat.find(op_type);
    auto ori_format = (ori_iter == kMsSpecOriginFormat.end()) ? ACL_FORMAT_NCHW : ori_iter->second;

    auto output_desc = aclCreateTensorDesc(acl_type, ori_shape.size(), ori_shape.data(), ori_format);
    MS_EXCEPTION_IF_NULL(output_desc);
    if (aclSetTensorShape(output_desc, output_shape.size(), output_shape.data())) {
      MS_LOG(EXCEPTION) << "Acl set tensor shape failed!";
    }
    if (aclSetTensorFormat(output_desc, acl_format)) {
      MS_LOG(EXCEPTION) << "Acl set tensor format failed!";
    }

    auto output_data = aclCreateDataBuffer(outputs[i]->addr, output_size_list[i]);
    MS_EXCEPTION_IF_NULL(output_data);
    (void)output_tensor_desc_.emplace_back(output_desc);
    (void)output_tensor_data_.emplace_back(output_data);
  }
}

void AclOpDesc::AddTensorAttr(const std::string &attr_name, const ValuePtr &value, const std::string &op_type) {
  MS_EXCEPTION_IF_NULL(value);
  if (acl_attr_ == nullptr) {
    MS_LOG(EXCEPTION) << "Acl attr create failed!";
  }

  aclError ret = 0;
  auto to_convert_attr = GetConvertAttr(op_type);
  auto new_name = attr_name;
  auto iter = to_convert_attr.find(new_name);
  if (iter != to_convert_attr.end()) {
    new_name = iter->second;
  }

  if (value->isa<BoolImm>()) {
    ret = aclopSetAttrBool(acl_attr_, new_name.c_str(), GetValue<bool>(value));
  } else if (value->isa<Int64Imm>()) {
    ret = aclopSetAttrInt(acl_attr_, new_name.c_str(), GetValue<int64_t>(value));
  } else if (value->isa<FP32Imm>()) {
    ret = aclopSetAttrFloat(acl_attr_, new_name.c_str(), GetValue<float>(value));
  } else if (value->isa<StringImm>()) {
    ret = aclopSetAttrString(acl_attr_, new_name.c_str(), GetValue<std::string>(value).c_str());
  } else if (value->isa<ValueSequence>()) {
    SetListAttr(new_name, value);
  } else {
    MS_LOG(INFO) << "Currently not support to Add the attr '" << new_name << "' with value: " << value->ToString()
                 << ", perhaps you should add more supported type.";
  }

  if (ret) {
    MS_LOG(EXCEPTION) << "Set node attr '" << attr_name << "' with value: " << value->ToString() << " failed!";
  }
}

void AclOpDesc::SetListAttr(const std::string &attr_name, const ValuePtr &value) {
  const auto &value_sequence = value->cast<ValueSequencePtr>()->value();
  if (value_sequence.size() <= 0) {
    return;
  }

  aclError ret = 0;
  auto val = value_sequence[0];
  if (val->isa<BoolImm>()) {
    auto value_list = GetValue<std::vector<uint8_t>>(value);
    ret = aclopSetAttrListBool(acl_attr_, attr_name.c_str(), value_list.size(), value_list.data());
  } else if (val->isa<Int64Imm>()) {
    auto value_list = GetValue<std::vector<int64_t>>(value);
    ret = aclopSetAttrListInt(acl_attr_, attr_name.c_str(), value_list.size(), value_list.data());
  } else if (val->isa<FP32Imm>()) {
    auto value_list = GetValue<std::vector<float>>(value);
    ret = aclopSetAttrListFloat(acl_attr_, attr_name.c_str(), value_list.size(), value_list.data());
  } else if (val->isa<StringImm>()) {
    auto value_list = GetValue<std::vector<std::string>>(value);
    ret = aclopSetAttrListString(acl_attr_, attr_name.c_str(), value_list.size(),
                                 reinterpret_cast<const char **>(value_list.data()));
  } else {
    MS_LOG(INFO) << "Currently not support to Add the attr '" << attr_name << "' with value: " << value->ToString()
                 << ", perhaps you should add more supported type.";
  }

  if (ret) {
    MS_LOG(EXCEPTION) << "Set node attr '" << attr_name << "' with value: " << value->ToString() << " failed!";
  }
}

aclDataType AclUtils::ConvertTypeIdToAclType(const TypeId &type_id) {
  auto iter = kMsTypeToAclType.find(type_id);
  if (iter == kMsTypeToAclType.end()) {
    MS_LOG(EXCEPTION) << "Unsupported op data type:" << type_id << " when convert to acl data type";
  }
  return iter->second;
}

aclFormat AclUtils::ConvertFormatToAclFormat(const std::string &format) {
  auto iter = kMsFormatToAclFormat.find(format);
  if (iter == kMsFormatToAclFormat.end()) {
    MS_LOG(EXCEPTION) << "Unsupported op format:" << format << " when convert to acl data type";
  }
  return iter->second;
}
}  // namespace kernel
}  // namespace mindspore
