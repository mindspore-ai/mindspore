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
  {::ge::FORMAT_NCHW, ACL_FORMAT_NCHW},           {::ge::FORMAT_HWCN, ACL_FORMAT_NCHW},
  {::ge::FORMAT_NHWC, ACL_FORMAT_NHWC},           {::ge::FORMAT_ND, ACL_FORMAT_ND},
  {::ge::FORMAT_FRACTAL_Z_3D, ACL_FRACTAL_Z_3D},  {::ge::FORMAT_NC1HWC0, ACL_FORMAT_NC1HWC0},
  {::ge::FORMAT_FRACTAL_Z, ACL_FORMAT_FRACTAL_Z}, {::ge::FORMAT_NC1HWC0_C04, ACL_FORMAT_NC1HWC0_C04},
  {::ge::FORMAT_NDHWC, ACL_FORMAT_NDHWC},         {::ge::FORMAT_FRACTAL_NZ, ACL_FORMAT_FRACTAL_NZ},
  {::ge::FORMAT_NCDHW, ACL_FORMAT_NCDHW},         {::ge::FORMAT_NDC1HWC0, ACL_FORMAT_NDC1HWC0}};

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

template <typename T>
void AclOpDesc::CallFunc(const T &val, const TypeId type, const std::string &attr_name, const ProcessAttrMode &mode) {
  if (mode == SET_ACL_ATTR) {
    CallAclAttrFunc<T>(val, type, attr_name);
  } else {
    AddConstDescAndBuf<T>(val, type, attr_name);
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
  if (!dev_shape.empty() && aclSetTensorShape(acl_desc, dev_shape.size(), dev_shape.data())) {
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
                       [this](const GeTensorDescPtr &desc) -> aclTensorDesc * {
                         if (desc->GetDataType() == GeDataType::DT_UNDEFINED) {
                           return nullptr;
                         } else {
                           return CreateTensorDesc(desc);
                         }
                       });
  (void)std::transform(outputs.begin(), outputs.end(), std::back_inserter(output_tensor_desc_),
                       [this](const GeTensorDescPtr &desc) { return CreateTensorDesc(desc); });
}

void AclOpDesc::AddDataBuf(const std::vector<AddressPtr> &inputs, const std::vector<size_t> &input_size_list,
                           const std::vector<AddressPtr> &outputs, const std::vector<size_t> &output_size_list) {
  for (size_t i = 0; i < input_size_list.size(); ++i) {
    if (input_size_list[i] == SIZE_MAX) {
      auto null_desc = aclCreateTensorDesc(ACL_DT_UNDEFINED, 0, nullptr, ACL_FORMAT_UNDEFINED);
      auto null_buf = aclCreateDataBuffer(nullptr, 0);
      input_tensor_desc_[i] = null_desc;
      (void)input_tensor_data_.emplace_back(null_buf);
      continue;
    }
    if (input_size_list[i] == 0) {
      (void)input_tensor_data_.emplace_back(nullptr);
      continue;
    }
    auto data_buf = CreateDataBuf(inputs[i], input_size_list[i]);
    (void)input_tensor_data_.emplace_back(data_buf);
  }
  for (size_t i = 0; i < output_size_list.size(); ++i) {
    auto data_buf = CreateDataBuf(outputs[i], output_size_list[i]);
    (void)output_tensor_data_.emplace_back(data_buf);
  }

  (void)input_tensor_desc_.erase(std::remove_if(input_tensor_desc_.begin(), input_tensor_desc_.end(),
                                                [](aclTensorDesc *desc) { return desc == nullptr; }),
                                 input_tensor_desc_.end());
  (void)input_tensor_data_.erase(std::remove_if(input_tensor_data_.begin(), input_tensor_data_.end(),
                                                [](aclDataBuffer *buf) { return buf == nullptr; }),
                                 input_tensor_data_.end());
}

void AclOpDesc::ProcessAclAttrs(const std::string &attr_name, const ValuePtr &value, const ProcessAttrMode &mode) {
  if (value == nullptr) {
    MS_LOG(INFO) << "Attr: " << attr_name << " has no value, skip!";
    return;
  }

  if (value->isa<BoolImm>()) {
    CallFunc<bool>(GetValue<bool>(value), kNumberTypeBool, attr_name, mode);
  } else if (value->isa<Int64Imm>()) {
    CallFunc<int64_t>(GetValue<int64_t>(value), kNumberTypeInt64, attr_name, mode);
  } else if (value->isa<Int32Imm>()) {
    CallFunc<int>(GetValue<int>(value), kNumberTypeInt64, attr_name, mode);
  } else if (value->isa<FP32Imm>()) {
    CallFunc<float>(GetValue<float>(value), kNumberTypeFloat32, attr_name, mode);
  } else if (value->isa<StringImm>()) {
    CallFunc<std::string>(GetValue<std::string>(value), kObjectTypeString, attr_name, mode);
  } else if (value->isa<ValueSequence>()) {
    GetListAttr(attr_name, value, mode);
  } else {
    MS_LOG(INFO) << "Currently not support to Add the attr '" << attr_name << "' with value: " << value->ToString()
                 << ", perhaps you should add more supported type.";
  }
}

std::vector<std::vector<int64_t>> AclOpDesc::GetListListAttrBool(const std::string &attr_name,
                                                                 const ValuePtrList &value_sequence) {
  std::vector<std::vector<int64_t>> value_lists;
  for (const auto &val : value_sequence) {
    if (!val->isa<ValueSequence>()) {
      continue;
    }
    auto uint8_value = GetValue<std::vector<uint8_t>>(val);
    std::vector<int64_t> tmp;
    (void)std::transform(uint8_value.begin(), uint8_value.end(), std::back_inserter(tmp),
                         [](uint8_t num) { return static_cast<int64_t>(num); });
    (void)value_lists.emplace_back(tmp);
  }
  return value_lists;
}

std::vector<std::vector<int64_t>> AclOpDesc::GetListListAttrInt(const std::string &attr_name,
                                                                const ValuePtrList &value_sequence) {
  std::vector<std::vector<int64_t>> value_lists;
  for (const auto &val : value_sequence) {
    if (!val->isa<ValueSequence>()) {
      continue;
    }
    (void)value_lists.emplace_back(GetValue<std::vector<int64_t>>(val));
  }
  return value_lists;
}

std::vector<std::vector<int64_t>> AclOpDesc::GetListListAttrFloat(const std::string &attr_name,
                                                                  const ValuePtrList &value_sequence) {
  std::vector<std::vector<int64_t>> value_lists;
  for (const auto &val : value_sequence) {
    if (!val->isa<ValueSequence>()) {
      continue;
    }
    auto float_value = GetValue<std::vector<float>>(val);
    std::vector<int64_t> tmp;
    (void)std::transform(float_value.begin(), float_value.end(), std::back_inserter(tmp), FloatToLong);
    (void)value_lists.emplace_back(tmp);
  }
  return value_lists;
}

void AclOpDesc::GetListListAttr(const std::string &attr_name, const ValuePtr &value, const ProcessAttrMode &mode) {
  const auto &value_sequence = value->cast<ValueSequencePtr>()->value();
  if (value_sequence.size() <= 0) {
    return;
  }
  auto val = value_sequence[0];
  std::vector<std::vector<int64_t>> value_list;
  if (val->isa<ValueSequence>()) {
    const auto &sub_value_sequence = val->cast<ValueSequencePtr>()->value();
    auto sub_val = sub_value_sequence[0];
    if (sub_val->isa<BoolImm>()) {
      value_list = GetListListAttrBool(attr_name, value_sequence);
    } else if (sub_val->isa<Int64Imm>()) {
      value_list = GetListListAttrInt(attr_name, value_sequence);
    } else if (sub_val->isa<FP32Imm>()) {
      value_list = GetListListAttrFloat(attr_name, value_sequence);
    } else {
      MS_LOG(INFO) << "Currently not support to Add the attr '" << attr_name << "' with value: " << value->ToString()
                   << ", perhaps you should add more supported type.";
      return;
    }
    CallFunc<std::vector<std::vector<int64_t>>>(value_list, kNumberTypeInt64, attr_name, mode);
  }
}

void AclOpDesc::GetListAttr(const std::string &attr_name, const ValuePtr &value, const ProcessAttrMode &mode) {
  MS_EXCEPTION_IF_NULL(value);
  const auto &value_sequence = value->cast<ValueSequencePtr>()->value();
  if (value_sequence.size() <= 0) {
    std::vector<int64_t> empty_vec = {};
    CallFunc<std::vector<int64_t>>(empty_vec, kNumberTypeInt64, attr_name, mode);
    return;
  }

  auto val = value_sequence[0];
  if (val->isa<BoolImm>()) {
    CallFunc<std::vector<uint8_t>>(GetValue<std::vector<uint8_t>>(value), kNumberTypeBool, attr_name, mode);
  } else if (val->isa<Int64Imm>()) {
    CallFunc<std::vector<int64_t>>(GetValue<std::vector<int64_t>>(value), kNumberTypeInt64, attr_name, mode);
  } else if (val->isa<Int32Imm>()) {
    auto value_list = GetValue<std::vector<int>>(value);
    std::vector<int64_t> value_list_int64;
    std::transform(value_list.begin(), value_list.end(), std::back_inserter(value_list_int64),
                   [](const int val) { return IntToLong(val); });
    CallFunc<std::vector<int64_t>>(value_list_int64, kNumberTypeInt64, attr_name, mode);
  } else if (val->isa<FP32Imm>()) {
    auto value_list = GetValue<std::vector<float>>(value);
    CallFunc<std::vector<float>>(value_list, kNumberTypeFloat32, attr_name, mode);
  } else if (val->isa<StringImm>()) {
    auto value_list = GetValue<std::vector<std::string>>(value);
    CallFunc<std::vector<std::string>>(value_list, kObjectTypeString, attr_name, mode);
  } else if (val->isa<ValueSequence>()) {
    GetListListAttr(attr_name, value, mode);
  } else {
    MS_LOG(INFO) << "Currently not support to Add the attr '" << attr_name << "' with value: " << value->ToString()
                 << ", perhaps you should add more supported type.";
  }
}

template <typename T>
void AclOpDesc::AddConstDescAndBuf(const T &val, const TypeId type, const std::string &attr_name) {
  size_t real_size = 0;
  bool is_empty_vec = false;
  ShapeVector shape;
  aclError ret = ACL_SUCCESS;
  std::vector<int> value_list_int;
  TypeId new_type = type;
  void *current_addr = static_cast<void *>(static_cast<int *>(attr_to_input_) + attr_data_offset_ / sizeof(int));
  if constexpr (is_vector<T>) {
    if constexpr (is_vector<typename T::value_type>) {
      if (!val.empty() && !val[0].empty()) {
        real_size = sizeof(typename T::value_type::value_type) * val.size() * val[0].size();
        shape.push_back(SizeToLong(val.size()));
        shape.push_back(SizeToLong(val[0].size()));
      }
    } else {
      constexpr size_t min_byte = 1;
      real_size = std::max(sizeof(typename T::value_type) * val.size(), min_byte);
      if constexpr (std::is_same_v<T, std::vector<int64_t>>) {
        std::transform(val.begin(), val.end(), std::back_inserter(value_list_int),
                       [](const int64_t val) { return LongToInt(val); });
        real_size = std::max(sizeof(int) * value_list_int.size(), min_byte);
        new_type = kNumberTypeInt32;
      }
      shape.push_back(SizeToLong(val.size()));
    }
    if (!value_list_int.empty()) {
      ret = aclrtMemcpy(current_addr, kMaxAttrToInputSize - attr_data_offset_, value_list_int.data(), real_size,
                        ACL_MEMCPY_HOST_TO_HOST);
    } else if (!val.empty()) {
      ret = aclrtMemcpy(current_addr, kMaxAttrToInputSize - attr_data_offset_, val.data(), real_size,
                        ACL_MEMCPY_HOST_TO_HOST);
    } else {
      is_empty_vec = true;
    }
  } else {
    real_size = sizeof(T);
    shape.push_back(1);
    ret = aclrtMemcpy(current_addr, kMaxAttrToInputSize - attr_data_offset_, &val, real_size, ACL_MEMCPY_HOST_TO_HOST);
  }
  if (ret != ACL_SUCCESS) {
    MS_LOG(EXCEPTION) << "When convert attr to input, offset is " << attr_data_offset_ << " and current size is "
                      << real_size << ", please increase limit of kMaxAttrToInputSize.";
    return;
  }
  attr_data_offset_ += real_size;

  aclTensorDesc *tensor_desc = nullptr;
  if (is_empty_vec) {
    std::vector<int64_t> empty_shape = {0};
    tensor_desc = aclCreateTensorDesc(ACL_INT64, 0, empty_shape.data(), ACL_FORMAT_ND);
  } else {
    tensor_desc =
      CreateTensorDesc(GeOpConvertor::GetTensorDesc(shape, new_type, kOpFormat_DEFAULT, shape, kOpFormat_DEFAULT));
  }

  // Set host memory flag to const input.
  ret = aclSetTensorPlaceMent(tensor_desc, ACL_MEMTYPE_HOST);
  if (ret != ACL_SUCCESS) {
    MS_LOG(EXCEPTION) << "Set host memory flag failed! ret = " << ret;
    return;
  }
  auto data_buf = aclCreateDataBuffer(current_addr, real_size);
  auto index = attr_to_input_maps_[attr_name];
  input_tensor_desc_.insert(input_tensor_desc_.begin() + index - 1, tensor_desc);
  input_tensor_data_.insert(input_tensor_data_.begin() + index - 1, data_buf);
}

aclError AclOpDesc::AclSetAttrListListInt(const std::string &attr_name,
                                          const std::vector<std::vector<int64_t>> &value_list) {
  auto list_size = value_list.size();
  int64_t *values[list_size];
  std::vector<int> num_values;
  for (size_t i = 0; i < list_size; i++) {
    values[i] = const_cast<int64_t *>(value_list[i].data());
    (void)num_values.emplace_back(SizeToInt(value_list[i].size()));
  }
  aclError ret = aclopSetAttrListListInt(acl_attr_, attr_name.c_str(), list_size, num_values.data(), values);
  return ret;
}

aclError AclOpDesc::AclSetAttrListString(const std::string &attr_name, const std::vector<std::string> &value_list) {
  std::vector<const char *> convert_list;
  std::transform(value_list.begin(), value_list.end(), std::back_inserter(convert_list),
                 [](const std::string &s) { return s.c_str(); });
  aclError ret = aclopSetAttrListString(acl_attr_, attr_name.c_str(), value_list.size(), convert_list.data());
  return ret;
}

template <typename T>
void AclOpDesc::CallAclAttrFunc(const T &val, const TypeId type, const std::string &attr_name) {
  if (acl_attr_ == nullptr) {
    MS_LOG(EXCEPTION) << "Acl attr create failed!";
  }

  aclError ret = ACL_SUCCESS;
  if constexpr (is_vector<T>) {
    if constexpr (is_vector<typename T::value_type>) {
      ret = AclSetAttrListListInt(attr_name, val);
    } else {
      if constexpr (std::is_same_v<T, std::vector<uint8_t>>) {
        ret = aclopSetAttrListBool(acl_attr_, attr_name.c_str(), val.size(), val.data());
      } else if constexpr (std::is_same_v<T, std::vector<int64_t>>) {
        ret = aclopSetAttrListInt(acl_attr_, attr_name.c_str(), val.size(), val.data());
      } else if constexpr (std::is_same_v<T, std::vector<float>>) {
        ret = aclopSetAttrListFloat(acl_attr_, attr_name.c_str(), val.size(), val.data());
      } else if constexpr (std::is_same_v<T, std::vector<std::string>>) {
        ret = AclSetAttrListString(attr_name, val);
      } else {
        MS_LOG(EXCEPTION) << "Currently not support to Add the attr '" << attr_name << "' with value tyep: " << type
                          << ", perhaps you should add more supported type.";
      }
    }
  } else {
    if constexpr (std::is_same_v<T, bool>) {
      ret = aclopSetAttrBool(acl_attr_, attr_name.c_str(), val);
    } else if constexpr (std::is_same_v<T, int64_t>) {
      ret = aclopSetAttrInt(acl_attr_, attr_name.c_str(), val);
    } else if constexpr (std::is_same_v<T, float>) {
      ret = aclopSetAttrFloat(acl_attr_, attr_name.c_str(), val);
    } else if constexpr (std::is_same_v<T, std::string>) {
      ret = aclopSetAttrString(acl_attr_, attr_name.c_str(), val.c_str());
    } else {
      MS_LOG(EXCEPTION) << "Currently not support to Add the attr '" << attr_name << "' with value tyep: " << type
                        << ", perhaps you should add more supported type.";
    }
  }

  if (ret) {
    MS_LOG(EXCEPTION) << "Set node attr '" << attr_name << "' with value tyep: " << type << " failed!";
  }
}

void AclOpDesc::AddConstInputTensor(const AnfNodePtr &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  auto cnode = anf_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto prim = GetValueNode<PrimitivePtr>(cnode->inputs()[0]);
  MS_EXCEPTION_IF_NULL(prim);

  attr_to_input_maps_ = GeOpConvertor::GetNeedAddInput(anf_node, true);
  for (const auto &[attr_name, index] : attr_to_input_maps_) {
    auto value = prim->GetAttr(attr_name);
    if (value == nullptr) {
      MS_LOG(WARNING) << "Attr name " << attr_name
                      << " isn't in current node, please check adaptor's attr name and index:" << index;
      continue;
    }
    ProcessAclAttrs(attr_name, value, SET_ACL_INPUT);
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
    MS_LOG(EXCEPTION) << "Unsupported op format:" << format << " when convert to acl format";
  }
  return iter->second;
}

bool AclUtils::UpdateTensorDesc(const AnfNodePtr &anf_node, std::vector<GeTensorDescPtr> *inputs,
                                std::vector<GeTensorDescPtr> *outputs) {
  MS_EXCEPTION_IF_NULL(anf_node);
  const auto &new_inputs = GetInputTensorDesc(anf_node);
  if (new_inputs.size() != inputs->size()) {
    MS_LOG(ERROR) << "Error match size between " << new_inputs.size() << " with " << inputs->size();
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

  auto cnode = anf_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  std::vector<size_t> useless_input_lists;
  if (common::AnfAlgo::HasNodeAttr(kAttrUselessInput, cnode)) {
    useless_input_lists = common::AnfAlgo::GetNodeAttr<std::vector<size_t>>(cnode, kAttrUselessInput);
  }

  std::vector<GeTensorDescPtr> res;
  std::set<size_t> already_add_index;
  for (size_t i = 0; i < input_num; ++i) {
    auto index = AnfAlgo::GetInputGraphIdxByKernelIdx(anf_node, i);
    if (index >= input_num) {
      MS_LOG(EXCEPTION) << "Error real index:" << index;
    }
    if (remove_index.count(index + 1) != 0) {
      MS_LOG(INFO) << "Current node's input " << (index + 1) << " need convert to attr or useless input!";
      continue;
    }
    if (std::find(useless_input_lists.begin(), useless_input_lists.end(), i) != useless_input_lists.end()) {
      MS_LOG(INFO) << "Current node's input " << i << " is useless!";
      auto desc = std::make_shared<GeTensorDesc>(GeShape(), GeFormat::FORMAT_ND, GeDataType::DT_UNDEFINED);
      MS_EXCEPTION_IF_NULL(desc);
      (void)res.emplace_back(desc);
      continue;
    }
    (void)already_add_index.insert(index + 1);
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
      MS_LOG(WARNING) << "Current node's input " << index
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
