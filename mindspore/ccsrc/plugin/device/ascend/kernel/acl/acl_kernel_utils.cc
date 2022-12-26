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
#include <utility>
#include <algorithm>
#include <iterator>
#include <optional>
#include <unordered_map>

#include "ir/value.h"
#include "include/common/utils/utils.h"
#include "include/common/utils/anfalgo.h"
#include "runtime/device/ms_device_shape_transfer.h"
#include "runtime/pynative/op_runtime_info.h"
#include "include/backend/anf_runtime_algorithm.h"

#include "plugin/device/ascend/optimizer/ascend_helper.h"
#include "acl/acl_rt.h"
#include "include/robin_hood.h"
#include "ir/primitive.h"
#include "ir/scalar.h"
#include "kernel/oplib/super_bar.h"
#include "ops/base_operator.h"
#include "ops/core_ops.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kMaxAttrToInputSize = 1024;
constexpr size_t KFormatLimitNumber = 2;
constexpr auto kParamDynamic = "dynamic";

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

static const std::unordered_map<std::string, std::vector<std::string>> kMsNeedPad = {
  {kTransDataOpName, {}},
  {kBNTrainingReduceOpName, {"", kOpFormat_NCHW}},
  {kBNTrainingUpdateOpName, {"", kOpFormat_NCHW}},
  {kBNTrainingReduceGradOpName, {"", kOpFormat_NCHW}},
  {kBNTrainingUpdateGradOpName, {"", kOpFormat_NCHW}},
  {kBNInferOpName, {"", kOpFormat_NCHW}},
  {kStridedSliceGradOpName, {"", kOpFormat_NCHW}},
  {kTensorMoveOpName, {"", kOpFormat_NCHW}},
  {kBiasAddOpName, {kOpFormat_NCHW, kOpFormat_NCHW}},
  {kBiasAddGradOpName, {kOpFormat_NCHW, kOpFormat_NCHW}}};

static const std::map<std::string, std::vector<int>> kInputOrders = {
  // op_name: {graph_id to kernel_id} . -1 means the the graph id is useless in acl kernel
  {prim::kPrimOneHotD->name(), {0, 2, 3}},
  {prim::kPrimAvgPool->name(), {0, -1, -1}},
  {prim::kPrimMaximumGrad->name(), {1, 2, 0}},
  {prim::kPrimMinimumGrad->name(), {1, 2, 0}},
  {prim::kPrimInplaceUpdateD->name(), {0, 2}},
  {prim::kPrimDeformableOffsets->name(), {0, 1, -1}},
  {prim::kPrimSplitD->name(), {1}},
  {prim::kPrimInplaceAddD->name(), {0, 2}}};

static const std::map<std::string, std::vector<int>> kOutputOrders = {
  // op_name: {graph_id to kernel_id} . -1 means the the graph id is useless in acl kernel
  {prim::kPrimApplyMomentum->name(), {0, -1}},
  {prim::kPrimApplyFtrlD->name(), {0, -1, -1}},
  {prim::kPrimSparseApplyFtrlV2D->name(), {0, -1, -1}},
  {prim::kPrimApplyMomentumD->name(), {0, -1}},
  {prim::kPrimApplyFtrl->name(), {0, -1, -1}}};
}  // namespace

AclOpDesc::AclOpDesc(const std::string &op_type, const AnfNodePtr &anf_node_ptr) {
  op_type_ = op_type;
  acl_attr_ = aclopCreateAttr();
  auto ret = aclrtMallocHost(&attr_to_input_, kMaxAttrToInputSize);
  if (ret != ACL_SUCCESS) {
    MS_LOG(WARNING) << "Malloc acl attr memory failed! error info:" << ret;
  }
  anf_node_ = anf_node_ptr;
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
  auto name = tensor_desc->GetName();

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
  if (!name.empty()) {
    aclSetTensorDescName(acl_desc, name.c_str());
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
                         if (desc == nullptr) {
                           return nullptr;
                         } else {
                           return CreateTensorDesc(desc);
                         }
                       });
  (void)std::transform(outputs.begin(), outputs.end(), std::back_inserter(output_tensor_desc_),
                       [this](const GeTensorDescPtr &desc) -> aclTensorDesc * {
                         if (desc == nullptr) {
                           return nullptr;
                         } else {
                           return CreateTensorDesc(desc);
                         }
                       });
}

void AclOpDesc::AddDataBuf(const std::vector<AddressPtr> &inputs, const std::vector<size_t> &input_size_list,
                           const std::vector<AddressPtr> &outputs, const std::vector<size_t> &output_size_list,
                           const std::vector<std::string> &input_names, const std::vector<std::string> &output_names,
                           const std::map<int, tensor::TensorPtr> &const_input_list) {
  auto node = anf_node_.lock();
  MS_EXCEPTION_IF_NULL(node);
  input_tensor_data_.clear();
  input_tensor_data_.resize(input_names.size(), nullptr);
  for (size_t i = 0; i < inputs.size(); i++) {
    auto idx = AclUtils::GetInputKernelIdxByGraphIdx(node, i);
    if (idx < 0) {
      continue;
    }
    if (idx >= SizeToInt(input_size_list.size())) {
      MS_LOG(EXCEPTION) << "Invalid index: " << idx << ", index in anf node: " << i
                        << ", node:" << node->fullname_with_scope();
    }
    if (input_size_list[idx] == kSizeMax) {
      if (input_tensor_desc_[idx] != nullptr || common::AnfAlgo::IsNoneInput(node, i)) {
        CreateNullAclTensor(idx, true);
      }
      continue;
    }
    input_tensor_data_[idx] = CreateDataBuf(inputs[i], input_size_list[idx]);
    if (const_input_list.find(idx) != const_input_list.end()) {
      const auto &tensor = const_input_list.at(idx);
      auto const_ret = aclSetTensorConst(input_tensor_desc_[idx], tensor->data_c(), tensor->Size());
      if (const_ret != ACL_SUCCESS) {
        MS_LOG(EXCEPTION) << "ACL set tensor const failed!";
      }
    }
  }

  output_tensor_data_.clear();
  output_tensor_data_.resize(output_names.size(), aclCreateDataBuffer(nullptr, 0));
  for (size_t i = 0; i < outputs.size(); ++i) {
    auto idx = AclUtils::GetOutputKernelIdxByGraphIdx(node, i);
    if (idx < 0) {
      continue;
    }
    if (idx >= SizeToInt(output_size_list.size())) {
      MS_LOG(EXCEPTION) << "Invalid output index: " << idx << ", node:" << node->fullname_with_scope();
    }
    if (output_size_list[idx] == kSizeMax) {
      CreateNullAclTensor(idx, false);
      continue;
    }
    output_tensor_data_[idx] = CreateDataBuf(outputs[i], output_size_list[idx]);
  }
}

void AclOpDesc::CreateNullAclTensor(const size_t idx, const bool is_input) {
  auto null_desc = aclCreateTensorDesc(ACL_DT_UNDEFINED, 0, nullptr, ACL_FORMAT_UNDEFINED);
  auto null_buf = aclCreateDataBuffer(nullptr, 0);
  if (is_input) {
    input_tensor_desc_[idx] = null_desc;
    input_tensor_data_[idx] = null_buf;
    return;
  }
  output_tensor_desc_[idx] = null_desc;
  output_tensor_data_[idx] = null_buf;
}

void AclOpDesc::ClearNullTensor() {
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

void AclOpDesc::ListListToListInt(const std::vector<std::vector<int64_t>> &value_list,
                                  std::vector<int64_t> *array_list) {
  MS_EXCEPTION_IF_NULL(array_list);
  auto list_size = value_list.size();
  for (size_t i = 0; i < list_size; i++) {
    (void)array_list->insert(array_list->end(), value_list[i].begin(), value_list[i].end());
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
  std::vector<int64_t> value_list_long;
  TypeId new_type = type;
  void *current_addr = static_cast<void *>(static_cast<int *>(attr_to_input_) + attr_data_offset_ / sizeof(int));
  if constexpr (is_vector<T>) {
    if constexpr (is_vector<typename T::value_type>) {
      if (!val.empty() && !val[0].empty()) {
        real_size = sizeof(typename T::value_type::value_type) * val.size() * val[0].size();
        shape.push_back(SizeToLong(val.size()));
        shape.push_back(SizeToLong(val[0].size()));
        if constexpr (std::is_same_v<T, std::vector<std::vector<int64_t>>>) {
          ListListToListInt(val, &value_list_long);
        }
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
    } else if (!value_list_long.empty()) {
      ret = aclrtMemcpy(current_addr, kMaxAttrToInputSize - attr_data_offset_, value_list_long.data(), real_size,
                        ACL_MEMCPY_HOST_TO_HOST);
    } else if (!val.empty()) {
      ret = aclrtMemcpy(current_addr, kMaxAttrToInputSize - attr_data_offset_, val.data(), real_size,
                        ACL_MEMCPY_HOST_TO_HOST);
    } else {
      is_empty_vec = true;
    }
  } else {
    real_size = sizeof(T);
    if constexpr (std::is_same_v<T, int64_t>) {
      new_type = kNumberTypeInt32;
      real_size = sizeof(int32_t);
    }
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

  // Set by input name
  auto node = anf_node_.lock();
  MS_EXCEPTION_IF_NULL(node);
  const auto &attr_to_input_maps = GeOpConvertor::GetNeedAddInput(node, true);
  auto to_input_name = attr_to_input_maps.at(attr_name);
  const auto &input_names = kernel::AclUtils::GetOpInputAnchorNames(node);
  auto iter = std::find(input_names.begin(), input_names.end(), to_input_name);
  if (iter == input_names.end()) {
    MS_LOG(EXCEPTION) << "Error input name of " << to_input_name;
  }
  size_t index = LongToSize(iter - input_names.begin());
  if (index >= input_tensor_desc_.size() || index >= input_tensor_data_.size()) {
    MS_LOG(EXCEPTION) << "Index exceed the input tensor desc size, maybe it is a dynamic input index: " << index
                      << ", node:" << node->fullname_with_scope() << ", all anchors: " << input_names;
  }
  aclSetTensorDescName(tensor_desc, to_input_name.c_str());
  input_tensor_desc_[index] = tensor_desc;
  input_tensor_data_[index] = data_buf;
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

int AclUtils::GetInputKernelIdxByGraphIdx(const AnfNodePtr &node, size_t ori_idx) {
  MS_EXCEPTION_IF_NULL(node);
  auto op_type = GeOpConvertor::GetOpType(node, true);
  auto op_name = common::AnfAlgo::GetCNodeName(node);
  auto kernel_type = AnfAlgo::GetKernelType(node);
  if (kernel_type != TBE_KERNEL && kernel_type != ACL_KERNEL) {
    return SizeToInt(ori_idx);
  }
  auto item = kInputOrders.find(op_name);
  if (item != kInputOrders.end()) {
    return item->second[ori_idx];
  }
  auto orders = kernel::SuperBar::GetGraphIdxToKernelIdx(op_type);
  if (!orders.has_value()) {
    return SizeToInt(ori_idx);
  }
  const auto &input_orders = orders.value();
  auto iter = input_orders.find(ori_idx);
  if (iter != input_orders.end()) {
    return SizeToInt(iter->second);
  }
  MS_LOG(EXCEPTION) << "Get input order failed,input idx: " << ori_idx << ", node: " << node->fullname_with_scope();
}

int AclUtils::GetOutputKernelIdxByGraphIdx(const AnfNodePtr &node, size_t ori_idx) {
  MS_EXCEPTION_IF_NULL(node);
  auto op_type = GeOpConvertor::GetOpType(node, true);
  auto kernel_type = AnfAlgo::GetKernelType(node);
  if (kernel_type != TBE_KERNEL && kernel_type != ACL_KERNEL) {
    return SizeToInt(ori_idx);
  }
  auto iter = kOutputOrders.find(op_type);
  if (iter != kOutputOrders.end()) {
    return iter->second[ori_idx];
  }
  return SizeToInt(ori_idx);
}

int AclUtils::GetInputGraphIdxByKernelIdx(const AnfNodePtr &node, size_t ori_idx) {
  MS_EXCEPTION_IF_NULL(node);
  auto op_type = GeOpConvertor::GetOpType(node, true);
  auto kernel_type = AnfAlgo::GetKernelType(node);
  if (kernel_type != TBE_KERNEL && kernel_type != ACL_KERNEL) {
    return SizeToInt(ori_idx);
  }
  auto orders = kernel::SuperBar::GetKernelIdxToGraphIdx(op_type);
  if (!orders.has_value()) {
    return SizeToInt(ori_idx);
  }
  const auto &input_orders = orders.value();
  auto iter = input_orders.find(ori_idx);
  if (iter == input_orders.end()) {
    MS_LOG(EXCEPTION) << "Get input order failed,input idx: " << ori_idx << ", node: " << node->fullname_with_scope();
  }
  return SizeToInt(iter->second);
}

std::vector<std::string> AclUtils::GetOpInputAnchorNames(const AnfNodePtr &node) {
  auto adapter_input_info = GeOpConvertor::GetAclInputNames(node);
  auto dynamic_input_info = GeOpConvertor::GetAclDynamicInputNames(node);
  std::vector<std::string> names;
  size_t dynamic_input_index = 0;
  for (const auto &[index, name] : adapter_input_info) {
    auto real_idx = AclUtils::GetInputGraphIdxByKernelIdx(node, index - 1) + 1;
    if (adapter_input_info.find(real_idx) == adapter_input_info.end()) {
      MS_LOG(EXCEPTION) << "Invalid index: " << real_idx << ", name: " << name
                        << ", node: " << node->fullname_with_scope();
    }
    if (dynamic_input_info.find(real_idx) != dynamic_input_info.end()) {
      // means the real idx is dynamic input
      auto cnode = node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      if (!common::AnfAlgo::HasNodeAttr(kAttrDynInputSizes, cnode)) {
        MS_LOG(EXCEPTION) << "Node has no attr: " << kAttrDynInputSizes << ", " << node->fullname_with_scope();
      }
      auto dynamic_inputs_list = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(node, kAttrDynInputSizes);
      if (IntToSize(real_idx - 1) >= dynamic_inputs_list.size()) {
        MS_LOG(EXCEPTION) << "Invalid index: " << real_idx << ", node: " << node->fullname_with_scope();
      }
      for (size_t i = 0; i < LongToSize(dynamic_inputs_list[real_idx - 1]); i++) {
        (void)names.emplace_back(adapter_input_info[real_idx] + std::to_string(i));
      }
    } else {
      (void)names.emplace_back(adapter_input_info[real_idx]);
    }
    dynamic_input_index++;
  }
  return names;
}

std::vector<std::string> AclUtils::GetOpOutputAnchorNames(const AnfNodePtr &node) {
  auto adapter_output_info = GeOpConvertor::GetAclOutputNames(node);
  auto dynamic_output_info = GeOpConvertor::GetAclDynamicOutputNames(node);
  std::vector<std::string> names;
  for (const auto &[index, name] : adapter_output_info) {
    if (dynamic_output_info.find(index) != dynamic_output_info.end()) {
      auto real_outputs_size = AnfAlgo::GetOutputTensorNum(node);
      for (size_t i = 0; i < real_outputs_size; i++) {
        (void)names.emplace_back(name + std::to_string(i));
      }
      continue;
    }
    (void)names.emplace_back(name);
  }
  return names;
}

void AclUtils::UpdateShape(const AnfNodePtr &node, ShapeVector *shape, std::string *format) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(shape);
  MS_EXCEPTION_IF_NULL(format);
  auto node_name = common::AnfAlgo::GetCNodeName(node);
  if (kMsNeedPad.count(node_name) == 0 || shape->size() >= kDim4) {
    return;
  }
  const auto &default_format_str = kMsNeedPad.at(node_name);
  if (!default_format_str.empty()) {
    if (default_format_str.size() != KFormatLimitNumber) {
      MS_LOG(EXCEPTION) << "kMsNeedPad's node name:" << node_name << "'s size is invalid";
    }
    auto update_format = default_format_str.at(1);
    std::string format_pad = (shape->size() < kDim2) ? default_format_str.at(0) : update_format;
    *shape = trans::PaddingShape(*shape, *format, format_pad);
    *format = update_format.empty() ? kOpFormat_ND : update_format;
  } else if (!IsOneOfNoPaddingFormat(*format)) {
    *shape = trans::PaddingShape(*shape, *format);
  }
  return;
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
  const auto &input_names = AclUtils::GetOpInputAnchorNames(anf_node);
  std::vector<GeTensorDescPtr> res(input_names.size(), nullptr);
  for (size_t i = 0; i < input_num; ++i) {
    auto index = AclUtils::GetInputKernelIdxByGraphIdx(anf_node, i);
    if (index < 0) {
      // if index less than 0, means the input "i" is useless in acl kernel.
      continue;
    }

    auto [input, idx] = common::AnfAlgo::GetPrevNodeOutput(anf_node, i);
    auto op_runtime_info = input->user_data<runtime::OpRuntimeInfo>();
    auto input_type =
      (op_runtime_info == nullptr) ? AnfAlgo::GetOutputDeviceDataType(input, idx) : op_runtime_info->output_type(idx);
    if (input_type == kMetaTypeNone) {
      continue;
    }
    auto ori_shape = (op_runtime_info == nullptr) ? common::AnfAlgo::GetOutputInferShape(input, idx)
                                                  : op_runtime_info->output_infer_shape(idx);
    auto input_shape = (op_runtime_info == nullptr) ? AnfAlgo::GetOutputDeviceShape(input, idx)
                                                    : op_runtime_info->output_device_shape(idx);

    auto input_format =
      (op_runtime_info == nullptr) ? AnfAlgo::GetOutputFormat(input, idx) : op_runtime_info->output_format(idx);
    auto ori_format = IsOneOf3DFormat(input_format) ? kOpFormat_NCDHW : kOpFormat_DEFAULT;
    if (!opt::NeedInsertTransData(ori_shape, input_format)) {
      MS_LOG_DEBUG << "Set format of " << anf_node->fullname_with_scope() << " to origin format";
      input_shape = ori_shape;
      input_format = ori_format;
    }
    UpdateShape(anf_node, &ori_shape, &input_format);
    auto input_desc = GeOpConvertor::GetTensorDesc(input_shape, input_type, input_format, ori_shape, ori_format);
    MS_EXCEPTION_IF_NULL(input_desc);
    input_desc->SetName(input_names[index]);
    res[index] = input_desc;
  }
  return res;
}

std::vector<GeTensorDescPtr> AclUtils::GetOutputTensorDesc(const AnfNodePtr &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  size_t output_num = AnfAlgo::GetOutputTensorNum(anf_node);
  const auto &output_names = AclUtils::GetOpOutputAnchorNames(anf_node);
  std::vector<GeTensorDescPtr> res(output_names.size(), nullptr);
  auto op_runtime_info = anf_node->user_data<runtime::OpRuntimeInfo>();

  for (size_t i = 0; i < output_num; ++i) {
    auto index = AclUtils::GetOutputKernelIdxByGraphIdx(anf_node, i);
    if (index < 0) {
      continue;
    }
    auto output_type =
      (op_runtime_info == nullptr) ? AnfAlgo::GetOutputDeviceDataType(anf_node, i) : op_runtime_info->output_type(i);
    if (output_type == kMetaTypeNone) {
      continue;
    }
    auto ori_shape = (op_runtime_info == nullptr) ? common::AnfAlgo::GetOutputInferShape(anf_node, i)
                                                  : op_runtime_info->output_infer_shape(i);
    auto output_shape = (op_runtime_info == nullptr) ? AnfAlgo::GetOutputDeviceShape(anf_node, i)
                                                     : op_runtime_info->output_device_shape(i);
    auto output_format =
      (op_runtime_info == nullptr) ? AnfAlgo::GetOutputFormat(anf_node, i) : op_runtime_info->output_format(i);
    auto ori_format = IsOneOf3DFormat(output_format) ? kOpFormat_NCDHW : kOpFormat_DEFAULT;
    if (!opt::NeedInsertTransData(ori_shape, output_format)) {
      MS_LOG_DEBUG << "Set format of " << anf_node->fullname_with_scope() << " to origin format";
      output_shape = ori_shape;
      output_format = ori_format;
    }
    UpdateShape(anf_node, &ori_shape, &output_format);
    auto output_desc = GeOpConvertor::GetTensorDesc(output_shape, output_type, output_format, ori_shape, ori_format);
    MS_EXCEPTION_IF_NULL(output_desc);
    output_desc->SetName(output_names[index]);
    res[index] = output_desc;
  }
  return res;
}
}  // namespace kernel
}  // namespace mindspore
