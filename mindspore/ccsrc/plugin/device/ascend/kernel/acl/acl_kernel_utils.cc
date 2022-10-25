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
#include <set>
#include <functional>
#include <algorithm>
#include "ir/value.h"
#include "include/common/utils/utils.h"
#include "include/common/utils/anfalgo.h"
#include "kernel/common_utils.h"
#include "backend/common/session/anf_runtime_algorithm.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kMaxAttrToInputSize = 1024;

static const std::map<::ge::DataType, aclDataType> kMsTypeToAclType = {
  {::ge::DT_BOOL, ACL_BOOL},       {::ge::DT_INT8, ACL_INT8},     {::ge::DT_INT16, ACL_INT16},
  {::ge::DT_INT32, ACL_INT32},     {::ge::DT_INT64, ACL_INT64},   {::ge::DT_UINT8, ACL_UINT8},
  {::ge::DT_UINT16, ACL_UINT16},   {::ge::DT_UINT32, ACL_UINT32}, {::ge::DT_UINT64, ACL_UINT64},
  {::ge::DT_FLOAT16, ACL_FLOAT16}, {::ge::DT_FLOAT, ACL_FLOAT},   {::ge::DT_DOUBLE, ACL_DOUBLE},
  {::ge::DT_STRING, ACL_STRING}};

static const std::map<::ge::Format, aclFormat> kMsFormatToAclFormat = {
  {::ge::FORMAT_NCHW, ACL_FORMAT_NCHW},
  {::ge::FORMAT_NHWC, ACL_FORMAT_NHWC},
  {::ge::FORMAT_ND, ACL_FORMAT_ND},
  {::ge::FORMAT_FRACTAL_Z_3D, ACL_FRACTAL_Z_3D},
  {::ge::FORMAT_NC1HWC0, ACL_FORMAT_NC1HWC0},
  {::ge::FORMAT_FRACTAL_Z, ACL_FORMAT_FRACTAL_Z},
  {::ge::FORMAT_NC1HWC0_C04, ACL_FORMAT_NC1HWC0_C04},
  {::ge::FORMAT_NDHWC, ACL_FORMAT_NDHWC},
  {::ge::FORMAT_FRACTAL_NZ, ACL_FORMAT_FRACTAL_NZ},
  {::ge::FORMAT_NCDHW, ACL_FORMAT_NCDHW},
  {::ge::FORMAT_NDC1HWC0, ACL_FORMAT_NDC1HWC0}};

static const std::map<std::string, aclFormat> kMsSpecOriginFormat = {{"BatchMatMul", ACL_FORMAT_ND},
                                                                     {"MatMul", ACL_FORMAT_ND}};
}  // namespace

AclOpDesc::AclOpDesc(const std::string &op_type) {
  op_type_ = op_type;
  acl_attr_ = aclopCreateAttr();
  auto ret = aclrtMallocHost(&attr_to_input_, kMaxAttrToInputSize);
  if (ret != ACL_SUCCESS) {
    MS_LOG(WARNING) << "Malloc acl attr memory failed! error info:" << ret;
  }
}

AclOpDesc::~AclOpDesc() {
  aclopDestroyAttr(acl_attr_);
  if (attr_to_input_ != nullptr) {
    aclrtFreeHost(attr_to_input_);
    attr_to_input_ = nullptr;
  }
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

aclTensorDesc *AclOpDesc::CreateTensorDesc(const GeTensorDescPtr &tensor_desc) {
  MS_EXCEPTION_IF_NULL(tensor_desc);
  auto ori_shape = tensor_desc->GetOriginShape().GetDims();
  auto dev_shape = tensor_desc->GetShape().GetDims();
  auto dev_type = tensor_desc->GetDataType();
  auto dev_format = tensor_desc->GetFormat();

  auto acl_type = AclUtils::ConvertTypeIdToAclType(dev_type);
  auto acl_format = AclUtils::ConvertFormatToAclFormat(dev_format);

  auto ori_format = tensor_desc->GetOriginFormat();
  auto acl_ori_format = AclUtils::ConvertFormatToAclFormat(ori_format);
  auto ori_iter = kMsSpecOriginFormat.find(op_type_);
  acl_ori_format = (ori_iter == kMsSpecOriginFormat.end()) ? acl_ori_format : ori_iter->second;

  auto acl_desc = aclCreateTensorDesc(acl_type, ori_shape.size(), ori_shape.data(), acl_ori_format);
  MS_EXCEPTION_IF_NULL(acl_desc);
  if (aclSetTensorShape(acl_desc, dev_shape.size(), dev_shape.data())) {
    MS_LOG(EXCEPTION) << "Acl set tensor shape failed!";
  }
  if (aclSetTensorFormat(acl_desc, acl_format)) {
    MS_LOG(EXCEPTION) << "Acl set tensor format failed!";
  }
  return acl_desc;
}

aclDataBuffer *AclOpDesc::CreateDataBuf(const AddressPtr &address, const size_t op_size) {
  MS_EXCEPTION_IF_NULL(address);
  auto data_buf = aclCreateDataBuffer(address->addr, op_size);
  MS_EXCEPTION_IF_NULL(data_buf);
  return data_buf;
}

void AclOpDesc::AddTensorDesc(const std::vector<GeTensorDescPtr> &inputs, const std::vector<GeTensorDescPtr> &outputs) {
  (void)std::transform(inputs.begin(), inputs.end(), std::back_inserter(input_tensor_desc_),
                       [this](const GeTensorDescPtr &desc) { return CreateTensorDesc(desc); });
  (void)std::transform(outputs.begin(), outputs.end(), std::back_inserter(output_tensor_desc_),
                       [this](const GeTensorDescPtr &desc) { return CreateTensorDesc(desc); });
}

void AclOpDesc::AddDataBuf(const std::vector<AddressPtr> &inputs, const std::vector<size_t> &input_size_list,
                           const std::vector<AddressPtr> &outputs, const std::vector<size_t> &output_size_list) {
  for (size_t i = 0; i < input_size_list.size(); ++i) {
    auto data_buf = CreateDataBuf(inputs[i], input_size_list[i]);
    (void)input_tensor_data_.emplace_back(data_buf);
  }
  for (size_t i = 0; i < output_size_list.size(); ++i) {
    auto data_buf = CreateDataBuf(outputs[i], output_size_list[i]);
    (void)output_tensor_data_.emplace_back(data_buf);
  }
}

void AclOpDesc::AddTensorAttr(const std::string &attr_name, const ValuePtr &value) {
  MS_EXCEPTION_IF_NULL(value);
  if (acl_attr_ == nullptr) {
    MS_LOG(EXCEPTION) << "Acl attr create failed!";
  }

  aclError ret = 0;
  if (value->isa<BoolImm>()) {
    ret = aclopSetAttrBool(acl_attr_, attr_name.c_str(), GetValue<bool>(value));
  } else if (value->isa<Int64Imm>()) {
    ret = aclopSetAttrInt(acl_attr_, attr_name.c_str(), GetValue<int64_t>(value));
  } else if (value->isa<Int32Imm>()) {
    ret = aclopSetAttrInt(acl_attr_, attr_name.c_str(), IntToLong(GetValue<int>(value)));
  } else if (value->isa<FP32Imm>()) {
    ret = aclopSetAttrFloat(acl_attr_, attr_name.c_str(), GetValue<float>(value));
  } else if (value->isa<StringImm>()) {
    ret = aclopSetAttrString(acl_attr_, attr_name.c_str(), GetValue<std::string>(value).c_str());
  } else if (value->isa<ValueSequence>()) {
    SetListAttr(attr_name, value);
  } else {
    MS_LOG(INFO) << "Currently not support to Add the attr '" << attr_name << "' with value: " << value->ToString()
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
  } else if (val->isa<Int32Imm>()) {
    auto value_list = GetValue<std::vector<int>>(value);
    std::vector<int64_t> value_list_int64;
    std::transform(value_list.begin(), value_list.end(), std::back_inserter(value_list_int64),
                   [](const int val) { return IntToLong(val); });
    ret = aclopSetAttrListInt(acl_attr_, attr_name.c_str(), value_list_int64.size(), value_list_int64.data());
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

bool AclOpDesc::SelectConversionDataType(const ValuePtr &value, const unsigned int index) {
  MS_EXCEPTION_IF_NULL(value);
  if (value->isa<Int64Imm>()) {
    AddConstDescAndBuf<int64_t>(GetValue<int64_t>(value), kNumberTypeInt64, index);
  } else if (value->isa<Int32Imm>()) {
    AddConstDescAndBuf<int>(GetValue<int>(value), kNumberTypeInt, index);
  } else if (value->isa<FP32Imm>()) {
    AddConstDescAndBuf<float>(GetValue<float>(value), kNumberTypeFloat32, index);
  } else if (value->isa<FP64Imm>()) {
    AddConstDescAndBuf<double>(GetValue<double>(value), kNumberTypeFloat64, index);
  } else if (value->isa<BoolImm>()) {
    AddConstDescAndBuf<bool>(GetValue<bool>(value), kNumberTypeBool, index);
  } else if (value->isa<ValueSequence>()) {
    const auto &value_sequence = value->cast<ValueSequencePtr>()->value();
    if (value_sequence.size() <= 0) {
      return false;
    }
    auto val = value_sequence[0];
    if (val->isa<Int64Imm>()) {
      auto vec = GetValue<std::vector<int64_t>>(value);
      AddConstDescAndBuf<std::vector<int64_t>>(vec, kNumberTypeInt64, index);
    } else if (val->isa<Int32Imm>()) {
      auto vec = GetValue<std::vector<int>>(value);
      AddConstDescAndBuf<std::vector<int>>(vec, kNumberTypeInt, index);
    } else if (val->isa<FP32Imm>()) {
      auto vec = GetValue<std::vector<float>>(value);
      AddConstDescAndBuf<std::vector<float>>(vec, kNumberTypeFloat32, index);
    } else {
      return false;
    }
  } else {
    return false;
  }
  return true;
}

template <typename T>
void AclOpDesc::AddConstDescAndBuf(const T &val, const TypeId type, const size_t index) {
  size_t real_size = 0;
  size_t shape_size = 1;
  aclError ret;
  void *current_addr = static_cast<void *>(static_cast<int *>(attr_to_input_) + attr_data_offset_ / sizeof(int));
  if constexpr (is_vector<T>) {
    real_size = sizeof(typename T::value_type) * val.size();
    shape_size = val.size();
    ret = aclrtMemcpy(current_addr, kMaxAttrToInputSize - attr_data_offset_, val.data(), real_size,
                      ACL_MEMCPY_HOST_TO_HOST);
  } else {
    real_size = sizeof(T);
    ret = aclrtMemcpy(current_addr, kMaxAttrToInputSize - attr_data_offset_, &val, real_size, ACL_MEMCPY_HOST_TO_HOST);
  }
  if (ret != ACL_SUCCESS) {
    MS_LOG(EXCEPTION) << "When convert attr to input, offset is " << attr_data_offset_ << " and current size is "
                      << real_size << ", please increase limit of kMaxAttrToInputSize.";
    return;
  }
  attr_data_offset_ += real_size;

  ShapeVector shape = {SizeToLong(shape_size)};
  auto tensor_desc =
    CreateTensorDesc(GeOpConvertor::GetTensorDesc(shape, type, kOpFormat_DEFAULT, shape, kOpFormat_DEFAULT));

  // Set host memory flag to const input.
  ret = aclSetTensorPlaceMent(tensor_desc, ACL_MEMTYPE_HOST);
  if (ret != ACL_SUCCESS) {
    MS_LOG(EXCEPTION) << "Set host memory flag failed! ret = " << ret;
    return;
  }
  auto data_buf = aclCreateDataBuffer(current_addr, real_size);
  input_tensor_desc_.insert(input_tensor_desc_.begin() + index - 1, tensor_desc);
  input_tensor_data_.insert(input_tensor_data_.begin() + index - 1, data_buf);
}

void AclOpDesc::AddConstInputTensor(const AnfNodePtr &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  auto cnode = anf_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto prim = GetValueNode<PrimitivePtr>(cnode->inputs()[0]);
  MS_EXCEPTION_IF_NULL(prim);

  const auto &add_index_info = GeOpConvertor::GetNeedAddInput(anf_node, true);
  for (const auto &[attr_name, index] : add_index_info) {
    auto value = prim->GetAttr(attr_name);
    if (value == nullptr) {
      MS_LOG(WARNING) << "Attr name " << attr_name << " isn't in current node, please check adaptor's attr name!";
      continue;
    }
    if (!SelectConversionDataType(value, index)) {
      MS_LOG(INFO) << "Currently not support to Add the attr '" << attr_name << "' with value: " << value->ToString()
                   << ", perhaps you should add more supported type.";
    }
  }
}

aclDataType AclUtils::ConvertTypeIdToAclType(const ::ge::DataType &type) {
  auto iter = kMsTypeToAclType.find(type);
  if (iter == kMsTypeToAclType.end()) {
    MS_LOG(EXCEPTION) << "Unsupported op data type:" << type << " when convert to acl data type";
  }
  return iter->second;
}

aclFormat AclUtils::ConvertFormatToAclFormat(const ::ge::Format &format) {
  auto iter = kMsFormatToAclFormat.find(format);
  if (iter == kMsFormatToAclFormat.end()) {
    MS_LOG(EXCEPTION) << "Unsupported op format:" << format << " when convert to acl data type";
  }
  return iter->second;
}

bool AclUtils::UpdateTensorDesc(const AnfNodePtr &anf_node, std::vector<GeTensorDescPtr> *inputs,
                                std::vector<GeTensorDescPtr> *outputs) {
  MS_EXCEPTION_IF_NULL(anf_node);
  const auto &new_inputs = GetInputTensorDesc(anf_node);
  if (new_inputs.size() != inputs->size()) {
    return false;
  }
  *inputs = new_inputs;
  const auto &new_outputs = GetOutputTensorDesc(anf_node);
  if (new_outputs.size() != outputs->size()) {
    return false;
  }
  *outputs = new_outputs;
  return true;
}

std::vector<GeTensorDescPtr> AclUtils::GetInputTensorDesc(const AnfNodePtr &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  size_t input_num = common::AnfAlgo::GetInputTensorNum(anf_node);
  const auto &remove_index = GeOpConvertor::GetNeedRemoveInput(anf_node, true);
  std::vector<GeTensorDescPtr> res;
  std::set<size_t> already_add_index;
  for (size_t i = 0; i < input_num; ++i) {
    if (remove_index.count(i + 1) != 0) {
      MS_LOG(INFO) << "Current node's input " << (i + 1) << " need convert to attr!";
      continue;
    }
    auto index = AnfAlgo::GetInputGraphIdxByKernelIdx(anf_node, i);
    if (index >= input_num) {
      MS_LOG(EXCEPTION) << "Error real index:" << index;
    }
    (void)already_add_index.insert(i + 1);
    auto ori_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(anf_node, index);
    auto input_shape = AnfAlgo::GetInputDeviceShape(anf_node, index);
    auto input_type = AnfAlgo::GetInputDeviceDataType(anf_node, index);
    auto input_format = AnfAlgo::GetInputFormat(anf_node, index);
    auto input_desc = GeOpConvertor::GetTensorDesc(input_shape, input_type, input_format, ori_shape, kOpFormat_DEFAULT);
    MS_EXCEPTION_IF_NULL(input_desc);
    (void)res.emplace_back(input_desc);
  }

  const auto &add_index_info = GeOpConvertor::GetNeedAddInput(anf_node, true);
  for (const auto &[attr_name, index] : add_index_info) {
    if (already_add_index.count(index) != 0) {
      MS_LOG(EXCEPTION) << "Current node's input " << index
                        << " is convert from attr, but already set input, please check adaptor of attr " << attr_name;
    }
  }
  return res;
}

std::vector<GeTensorDescPtr> AclUtils::GetOutputTensorDesc(const AnfNodePtr &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  size_t output_num = common::AnfAlgo::GetOutputTensorNum(anf_node);
  std::vector<GeTensorDescPtr> res;
  for (size_t i = 0; i < output_num; ++i) {
    auto ori_shape = common::AnfAlgo::GetOutputInferShape(anf_node, i);
    auto output_shape = AnfAlgo::GetOutputDeviceShape(anf_node, i);
    auto output_type = AnfAlgo::GetOutputDeviceDataType(anf_node, i);
    auto output_format = AnfAlgo::GetOutputFormat(anf_node, i);
    auto output_desc =
      GeOpConvertor::GetTensorDesc(output_shape, output_type, output_format, ori_shape, kOpFormat_DEFAULT);
    MS_EXCEPTION_IF_NULL(output_desc);
    (void)res.emplace_back(output_desc);
  }
  return res;
}
}  // namespace kernel
}  // namespace mindspore
