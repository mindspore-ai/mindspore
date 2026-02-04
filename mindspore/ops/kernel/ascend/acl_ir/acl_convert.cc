/**
 * Copyright 2023-2024 Huawei Technologies Co., Ltd
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

#include "kernel/ascend/acl_ir/acl_convert.h"
#include <fstream>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <limits>
#include <algorithm>
#include "kernel/ascend/acl_ir/acl_adapter_info.h"
#include "kernel/ascend/acl_ir/acl_helper.h"
#include "ops_utils/op_utils.h"
#include "device_address/device_address.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "kernel/ascend/acl_ir/op_api_util.h"
#include "plugin/ascend/res_manager/symbol_interface/acl_base_symbol.h"
#include "plugin/ascend/res_manager/symbol_interface/acl_rt_symbol.h"
#include "plugin/ascend/res_manager/symbol_interface/symbol_utils.h"

namespace mindspore::device::ascend {
namespace {
static const std::map<TypeId, aclDataType> kDataTypeToAclDataTypeTable = {{kNumberTypeBool, ACL_BOOL},
                                                                          {kNumberTypeInt, ACL_INT32},
                                                                          {kNumberTypeInt8, ACL_INT8},
                                                                          {kNumberTypeInt16, ACL_INT16},
                                                                          {kNumberTypeInt32, ACL_INT32},
                                                                          {kNumberTypeInt64, ACL_INT64},
                                                                          {kNumberTypeUInt, ACL_UINT32},
                                                                          {kNumberTypeUInt8, ACL_UINT8},
                                                                          {kNumberTypeUInt16, ACL_UINT16},
                                                                          {kNumberTypeUInt32, ACL_UINT32},
                                                                          {kNumberTypeUInt64, ACL_UINT64},
                                                                          {kNumberTypeFloat, ACL_FLOAT},
                                                                          {kNumberTypeFloat16, ACL_FLOAT16},
                                                                          {kNumberTypeFloat32, ACL_FLOAT},
                                                                          {kNumberTypeFloat64, ACL_DOUBLE},
                                                                          {kNumberTypeBFloat16, ACL_BF16},
#ifdef EXPERIMENT_A5
                                                                          {kNumberTypeFloat8E4M3FN, ACL_FLOAT8_E4M3FN},
                                                                          {kNumberTypeFloat8E5M2, ACL_FLOAT8_E5M2},
                                                                          {kNumberTypeHiFloat8, ACL_HIFLOAT8},
#endif
                                                                          {kNumberTypeDouble, ACL_DOUBLE},
                                                                          {kNumberTypeComplex, ACL_DT_UNDEFINED},
                                                                          {kNumberTypeComplex64, ACL_COMPLEX64},
                                                                          {kNumberTypeComplex128, ACL_COMPLEX128},
                                                                          {kNumberTypeInt4, ACL_INT4},
                                                                          {kNumberTypeGLUInt, ACL_DT_UNDEFINED}};

static const std::map<std::string, aclFormat> kMsFormatToAclFormat = {{kOpFormat_NCHW, ACL_FORMAT_NCHW},
                                                                      {kOpFormat_NHWC, ACL_FORMAT_NHWC},
                                                                      {kOpFormat_ND, ACL_FORMAT_ND},
                                                                      {kOpFormat_DEFAULT, ACL_FORMAT_ND},
                                                                      {kOpFormat_NC1HWC0, ACL_FORMAT_NC1HWC0},
                                                                      {kOpFormat_NDC1HWC0, ACL_FORMAT_NDC1HWC0},
                                                                      {kOpFormat_FRAC_Z, ACL_FORMAT_FRACTAL_Z},
                                                                      {kOpFormat_FRAC_NZ, ACL_FORMAT_FRACTAL_NZ},
                                                                      {kOpFormat_FRACTAL_Z_3D, ACL_FRACTAL_Z_3D},
                                                                      {kOpFormat_NCDHW, ACL_FORMAT_NCDHW},
                                                                      {kOpFormat_NDHWC, ACL_FORMAT_NDHWC}};

static const std::map<aclDataType, std::string> kAclDatatypeToStr = {
  {ACL_FLOAT, "float"},
  {ACL_FLOAT16, "float16"},
  {ACL_INT8, "int8"},
  {ACL_INT32, "int32"},
  {ACL_UINT8, "uint8"},
  {ACL_INT16, "int16"},
  {ACL_UINT16, "uint16"},
  {ACL_UINT32, "uint32"},
  {ACL_INT64, "int64"},
  {ACL_UINT64, "uint64"},
  {ACL_DOUBLE, "double"},
  {ACL_BOOL, "bool"},
  {ACL_STRING, "string"},
  {ACL_COMPLEX64, "complex64"},
  {ACL_COMPLEX128, "complex128"},
  {ACL_BF16, "bf16"},
#ifdef EXPERIMENT_A5
  {ACL_HIFLOAT8, "hifloat8"},
  {ACL_FLOAT8_E5M2, "float8_e5m2"},
  {ACL_FLOAT8_E4M3FN, "float8_e4m3fn"},
#endif
};

static const std::map<aclFormat, std::string> kAclFormatToStr = {
  {ACL_FORMAT_NCHW, "NCHW"},       {ACL_FORMAT_NHWC, "NHWC"},           {ACL_FORMAT_ND, "ND"},
  {ACL_FORMAT_NC1HWC0, "NC1HWC0"}, {ACL_FORMAT_FRACTAL_Z, "FRACTAL_Z"}, {ACL_FORMAT_NC1HWC0_C04, "NC1HWC0_C04"},
  {ACL_FORMAT_HWCN, "HWCN"},       {ACL_FORMAT_NDHWC, "NDHWC"},         {ACL_FORMAT_FRACTAL_NZ, "FRACTAL_NZ"},
  {ACL_FORMAT_NCDHW, "NCDHW"},     {ACL_FORMAT_NDC1HWC0, "NDC1HWC0"},   {ACL_FRACTAL_Z_3D, "FRACTAL_Z_3D"}};

std::string aclDatatypeToStr(aclDataType type) {
  auto iter = kAclDatatypeToStr.find(type);
  if (iter != kAclDatatypeToStr.end()) {
    return iter->second;
  }
  return "undefined";
}

std::string aclFormatToStr(aclFormat fmt) {
  auto iter = kAclFormatToStr.find(fmt);
  if (iter != kAclFormatToStr.end()) {
    return iter->second;
  }
  return "undefined";
}

template <typename T>
inline std::string VectorToString(const std::vector<T> &values) {
  std::stringstream ss;
  for (auto iter = values.begin(); iter != values.end(); ++iter) {
    ss << *iter;
    if (iter != values.end() - 1) {
      ss << ", ";
    }
  }
  return ss.str();
}

std::string AclTensorDescString(const AclDumpString &desc) {
  std::stringstream ss;
  ss << "[TensorDesc] ";
  ss << "Name = " << desc.tensor_name;
  ss << ", DataType = " << desc.data_type;
  ss << ", Origin Format = " << desc.ori_format;
  ss << ", Origin Shape = " << desc.ori_shape;
  ss << ", Device Format = " << desc.dev_format;
  ss << ", Device Shape = " << desc.dev_shape;
  ss << ", Tensor Type = ";
  if (desc.tensor_type == AclDumpString::TensorType::kDeviceTensor) {
    ss << "Device Tensor";
  } else if (desc.tensor_type == AclDumpString::TensorType::kNullTensor) {
    ss << "Null Tensor";
  } else {
    ss << "Host Tensor";
  }
  return ss.str();
}

void DumpAclString(const aclDataType data_type, const ShapeVector &ori_shape, const ShapeVector &dev_shape,
                   const aclFormat ori_format, const aclFormat dev_format, AclDumpString *dump) {
  if (dump == nullptr) {
    return;
  }
  dump->data_type = aclDatatypeToStr(data_type);
  dump->ori_format = aclFormatToStr(ori_format);
  dump->dev_format = aclFormatToStr(dev_format);
  dump->ori_shape = VectorToString(ori_shape);
  dump->dev_shape = VectorToString(dev_shape);
  dump->tensor_type = AclDumpString::TensorType::kDeviceTensor;
  if (ori_format == ACL_FORMAT_UNDEFINED && dev_format == ACL_FORMAT_UNDEFINED) {
    dump->tensor_type = AclDumpString::TensorType::kNullTensor;
  } else if (dev_format == ACL_FORMAT_UNDEFINED) {
    dump->tensor_type = AclDumpString::TensorType::kHostTensor;
  }
}

std::variant<KernelTensor *, AclHostInfoPtr> GetInputParam(const std::vector<KernelTensor *> &inputs,
                                                           const AclInputToHost &input_on_host, size_t ms_real_idx) {
  auto host_input = input_on_host.get(ms_real_idx);
  if (host_input != nullptr) {
    return host_input;
  }
  if (ms_real_idx >= inputs.size()) {
    MS_LOG(EXCEPTION) << "Failed to find input " << ms_real_idx << " large than " << inputs.size();
  }
  return inputs[ms_real_idx];
}

static const char kDynamicN[] = "N";
AddressPtr UpdateKernelTensorAddress(KernelTensor *tensor, size_t split_num, size_t index) {
  MS_EXCEPTION_IF_NULL(tensor);
  auto offset = tensor->size() / split_num;
  auto ori_device = reinterpret_cast<uint8_t *>(tensor->device_ptr());
  return std::make_shared<mindspore::kernel::Address>(reinterpret_cast<void *>(ori_device + offset * index), offset);
}

size_t GetTupleSize(const KernelTensor *tensor) {
  MS_EXCEPTION_IF_NULL(tensor);
  if (tensor->type_id() == kObjectTypeTuple || tensor->type_id() == kObjectTypeList) {
    auto shape = tensor->GetShapeVector();
    if (shape.empty()) {
      MS_LOG(EXCEPTION) << "Current tensor is a tuple of tensor, but get a empty shape!";
    }
    if (shape[kIndex0] <= 0) {
      MS_LOG(EXCEPTION) << shape << " is an invalid shape, please check op infer!";
    }
    return static_cast<size_t>(shape[kIndex0]);
  }
  return 1;
}

static std::once_flag kAclopStaticListInit;
static std::unordered_set<std::string> kAclopStaticList;

bool ReadStatciAclOp(const std::string &op_name, bool is_dynamic) {
  static auto enable_static_env = common::GetEnv("MS_DEV_STATIC_ACL_OP");
  if (enable_static_env == "1") {
    return false;
  }

  static auto read_config = !enable_static_env.empty() && enable_static_env != "0";
  if (read_config) {
    std::call_once(kAclopStaticListInit, []() {
      std::ifstream in_file(enable_static_env);
      if (!in_file.is_open()) {
        MS_LOG(WARNING) << "MS_DEV_STATIC_ACL_OP set path:" << enable_static_env << " is invalid.";
        return;
      }
      std::string line;
      while (getline(in_file, line)) {
        kAclopStaticList.insert(line);
      }
      in_file.close();
    });

    if (kAclopStaticList.count(op_name) != 0) {
      return false;
    }
  }

  return is_dynamic;
}

std::vector<int64_t> GetDynamicInputSize(const PrimitivePtr &prim) {
  MS_EXCEPTION_IF_NULL(prim);
  return prim->HasAttr(kAttrDynInputSizes) ? GetValue<std::vector<int64_t>>(prim->GetAttr(kAttrDynInputSizes))
                                           : std::vector<int64_t>();
}
}  // namespace

template <typename ConvertType>
void AttrHelper<ConvertType>::ConvertValueToDstType(const ValuePtr &value, const TypeId src_type) {
  MS_EXCEPTION_IF_NULL(value);
  auto sub_converter = static_cast<ConvertType *>(this);
  switch (src_type) {
    case kNumberTypeInt32: {
      sub_converter->ConvertValue(value, AttrDeclType<int32_t>());
      break;
    }
    case kNumberTypeInt64: {
      sub_converter->ConvertValue(value, AttrDeclType<int64_t>());
      break;
    }
    case kNumberTypeFloat32: {
      sub_converter->ConvertValue(value, AttrDeclType<float>());
      break;
    }
    case kNumberTypeFloat64: {
      sub_converter->ConvertValue(value, AttrDeclType<double>());
      break;
    }
    default: {
      MS_LOG(EXCEPTION) << "Unsupported type: " << src_type;
    }
  }
}

template <typename ConvertType>
template <typename T>
void AttrHelper<ConvertType>::ConvertValueToRealType(const ValuePtr &value, const std::string &attr_name,
                                                     T trans_struct) {
  MS_EXCEPTION_IF_NULL(value);
  attr_name_ = attr_name;

  auto sub_converter = static_cast<ConvertType *>(this);
  // Set datatype
  if (value->isa<Scalar>()) {
    if constexpr (std::is_same<T, TensorParams *>::value) {
      auto scalar_type = value->type();
      MS_EXCEPTION_IF_NULL(scalar_type);
      TypeId scalar_type_id = scalar_type->type_id();
      trans_struct->ori_shape = {};
      trans_struct->dev_shape = {};
      trans_struct->data_type = scalar_type_id;
    }
  }

  if (value->isa<BoolImm>()) {
    sub_converter->ConvertValue(value, AttrDeclType<bool>(), trans_struct);
  } else if (value->isa<Int64Imm>()) {
    sub_converter->ConvertValue(value, AttrDeclType<int64_t>(), trans_struct);
  } else if (value->isa<Int32Imm>()) {
    sub_converter->ConvertValue(value, AttrDeclType<int32_t>(), trans_struct);
  } else if (value->isa<FP32Imm>()) {
    sub_converter->ConvertValue(value, AttrDeclType<float>(), trans_struct);
  } else if (value->isa<StringImm>()) {
    sub_converter->ConvertValue(value, AttrDeclType<std::string>(), trans_struct);
  } else if (value->isa<GeDataTypeImm>()) {
    sub_converter->ConvertValue(value, AttrDeclType<::ge::DataType>(), trans_struct);
  } else if (value->isa<ValueSequence>()) {
    ConvertListAttr(value, trans_struct);
  } else {
    MS_LOG(EXCEPTION) << "Currently not support to Add the attr '" << attr_name << "' with value: " << value->ToString()
                      << ", perhaps you should add more supported type.";
  }
}

template <typename ConvertType>
template <typename T>
void AttrHelper<ConvertType>::ConvertListAttr(const ValuePtr &value, T trans_struct) {
  MS_EXCEPTION_IF_NULL(value);
  MS_EXCEPTION_IF_NULL(value->cast<ValueSequencePtr>());
  const auto &value_sequence = value->cast<ValueSequencePtr>()->value();
  ShapeVector shape;
  TypePtr type_ptr = nullptr;
  bool is_ge_datatype = false;
  auto sub_converter = static_cast<ConvertType *>(this);
  GetValueSequenceDataTypeAndShape(value_sequence, &type_ptr, &shape, &is_ge_datatype);
  if (is_ge_datatype) {
    sub_converter->ConvertValue(value, AttrDeclType<std::vector<::ge::DataType>>(), trans_struct);
    return;
  }
  if (type_ptr == nullptr) {
    return;
  }
  TypeId type_id = type_ptr->type_id();
  if constexpr (std::is_same<T, TensorParams *>::value) {
    trans_struct->data_type = type_id;
    trans_struct->ori_shape = shape;
    trans_struct->dev_shape = shape;
  }

  if (shape.size() > 1) {
    if (type_id == TypeId::kNumberTypeInt64) {
      sub_converter->ConvertValue(value, AttrDeclType<std::vector<std::vector<int64_t>>>(), shape, trans_struct);
    } else {
      MS_LOG(EXCEPTION) << "Currently not support to convert input with value: " << value->ToString()
                        << ", perhaps you should add more supported type: " << TypeIdToString(type_id);
    }
  } else {
    if (type_id == TypeId::kNumberTypeBool) {
      sub_converter->ConvertValue(value, AttrDeclType<std::vector<uint8_t>>(), trans_struct);
    } else if (type_id == TypeId::kNumberTypeFloat) {
      sub_converter->ConvertValue(value, AttrDeclType<std::vector<float>>(), trans_struct);
    } else if (type_id == TypeId::kNumberTypeFloat32) {
      sub_converter->ConvertValue(value, AttrDeclType<std::vector<float>>(), trans_struct);
    } else if (type_id == TypeId::kNumberTypeInt32) {
      sub_converter->ConvertValue(value, AttrDeclType<std::vector<int32_t>>(), trans_struct);
    } else if (type_id == TypeId::kNumberTypeInt64) {
      sub_converter->ConvertValue(value, AttrDeclType<std::vector<int64_t>>(), trans_struct);
    } else if (type_id == TypeId::kObjectTypeString) {
      sub_converter->ConvertValue(value, AttrDeclType<std::vector<std::string>>(), trans_struct);
    } else {
      MS_LOG(EXCEPTION) << "Currently not support to convert input with value: " << value->ToString()
                        << ", perhaps you should add more supported type: " << TypeIdToString(type_id);
    }
  }
}

template <typename ConvertType>
void AttrHelper<ConvertType>::GetValueSequenceDataTypeAndShape(const ValuePtrList &value_sequence, TypePtr *data_type,
                                                               ShapeVector *shape, bool *is_ge_datatype) {
  MS_EXCEPTION_IF_NULL(data_type);
  MS_EXCEPTION_IF_NULL(shape);
  MS_EXCEPTION_IF_NULL(is_ge_datatype);
  if (value_sequence.size() == 0) {
    MS_LOG(WARNING) << "value sequence is empty, failed to get data type";
    return;
  }
  (void)shape->push_back(value_sequence.size());
  auto val = value_sequence[0];
  MS_EXCEPTION_IF_NULL(val);
  if (val->isa<GeDataTypeImm>()) {
    *is_ge_datatype = true;
    return;
  }
  if (val->isa<Scalar>()) {
    *data_type = val->type();
  }
  if (val->isa<ValueSequence>()) {
    const auto &sub_sequence = val->cast<ValueSequencePtr>()->value();
    GetValueSequenceDataTypeAndShape(sub_sequence, data_type, shape, is_ge_datatype);
  }
}

void AclConverter::ConvertValueDependToHostInput(const std::string &kernel_name,
                                                 const std::vector<KernelTensor *> &inputs,
                                                 const std::vector<TensorParams> &input_params,
                                                 const std::set<int64_t> &value_depend_args) {
  MS_LOG(DEBUG) << "Start convert value_depend to acl host_input";
  auto info = GeAdapterManager::GetInstance().GetInfo(kernel_name, true);
  MS_EXCEPTION_IF_NULL(info);
  for (const auto ms_proto_idx : value_depend_args) {
    auto idx_convert_iter = ms_and_ge_inputs_idx_info_.find(ms_proto_idx);
    if (idx_convert_iter == ms_and_ge_inputs_idx_info_.end()) {
      MS_LOG(DEBUG) << kernel_name << " input(" << ms_proto_idx << ") is invalid.";
      continue;
    }
    size_t ms_real_idx = idx_convert_iter->second.ms_real_idx[kIndex0];
    MS_LOG(DEBUG) << "ms adapter proto index is " << ms_proto_idx << " real index is " << ms_real_idx;
    const auto &input = inputs[ms_real_idx];
    const auto &param = input_params[ms_real_idx];
    MS_EXCEPTION_IF_NULL(input);
    auto type_id = input->dtype_id();
    AclHostInfoPtr acl_host_input;
    bool is_const = input->IsConstValue();
    if (input->size() == 0) {
      acl_host_input = std::make_shared<AclHostInfo>(nullptr, 0, type_id, is_const);
    } else if (!device::ascend::AclHelper::IsInputDtypeSupport(kernel_name, param.data_type, ms_proto_idx) &&
               param.data_type != kMetaTypeNone) {
      ValueDependToInputConverter value_convert;
      auto cast_map = value_convert.GetValueDependCastMap();
      auto iter = cast_map.find(param.data_type);
      if (iter == cast_map.end()) {
        MS_LOG(INTERNAL_EXCEPTION) << kernel_name << " input(" << ms_proto_idx
                                   << ") data type not support and can not add Cast.";
      }
      if (!device::ascend::AclHelper::IsInputDtypeSupport(kernel_name, iter->second, ms_proto_idx)) {
        MS_LOG(INTERNAL_EXCEPTION) << kernel_name << " input(" << ms_proto_idx << ") data type not support.";
      }
      auto value_ptr = input->GetValue();
      MS_EXCEPTION_IF_NULL(value_ptr);
      value_convert.ConvertValueToDstType(value_ptr, param.data_type);
      host_save_list_[ms_real_idx] = std::move(value_convert.GetData());
      acl_host_input = std::make_shared<AclHostInfo>(host_save_list_[ms_real_idx].data(),
                                                     host_save_list_[ms_real_idx].size(), iter->second, is_const);
    } else {
      acl_host_input =
        std::make_shared<AclHostInfo>(const_cast<void *>(input->GetValuePtr()), input->size(), type_id, is_const);
    }
    input_on_host_.emplace(ms_real_idx, acl_host_input);
  }
}

bool AclConverter::IsNeedSkipExecute(const std::string &kernel_name, const std::vector<KernelTensor *> &inputs,
                                     const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  auto opinfo = GeAdapterManager::GetInstance().GetInfo(kernel_name, true);
  MS_EXCEPTION_IF_NULL(opinfo);
  auto op_type = opinfo->op_type();

  if (!AclAdapterManager::GetInstance().CheckAclAdapter(op_type)) {
    return false;
  }
  auto acl_info = AclAdapterManager::GetInstance().GetOpInfo(op_type);
  if (acl_info.input_check_selector() != nullptr) {
    auto func = acl_info.input_check_selector();
    auto is_real_skip = func(inputs);
    if (is_real_skip) {
      if (inputs.empty() || outputs.empty()) {
        MS_LOG(EXCEPTION) << "Skip node [" << kernel_name << "] should have at least one input and output, but got "
                          << inputs.size() << " input and " << outputs.size() << " output.";
      }
      MS_EXCEPTION_IF_NULL(inputs[0]);
      MS_EXCEPTION_IF_NULL(outputs[0]);
      if (inputs[0]->size() == 0) {
        return true;
      }
      aclError status =
        CALL_ASCEND_API(aclrtMemcpyAsync, outputs[0]->device_ptr(), outputs[0]->size(), inputs[0]->device_ptr(),
                        inputs[0]->size(), ACL_MEMCPY_DEVICE_TO_DEVICE, stream_ptr);
      if (status != ACL_SUCCESS) {
        MS_LOG(EXCEPTION) << "MemCpyAsync op aclrtMemcpyAsync failed, ret:" << status
                          << " destMax:" << outputs[0]->size() << " count:" << inputs[0]->size();
      }
      return true;
    }
  }
  return false;
}

void AclConverter::ConvertToAclInput(const PrimitivePtr &prim, const std::vector<KernelTensor *> &inputs,
                                     const std::vector<TensorParams> &input_params) {
  MS_EXCEPTION_IF_NULL(prim);
  auto info = GeAdapterManager::GetInstance().GetInfo(prim->name(), true);
  MS_EXCEPTION_IF_NULL(info);
  auto flags = info->GetInputMappingFlags();
  if (flags & GeTensorInfo::kEmptyParam) {
    return;
  }

  // Special const input.
  bool set_const = false;
  if (AclAdapterManager::GetInstance().CheckAclAdapter(info->op_type())) {
    set_const = AclAdapterManager::GetInstance().GetOpInfo(info->op_type()).is_const_input();
  }

  for (auto &[ms_idx, input_info] : ms_and_ge_inputs_idx_info_) {
    auto opt_ge_input_info = info->GetOptGeInputByMsInputIndex(ms_idx);
    if (!opt_ge_input_info.has_value()) {
      continue;
    }
    auto &ge_input_info = opt_ge_input_info.value();

    AclDumpString dump_str;
    AclDumpString *dump_str_pointer = device::ascend::AclHelper::IsPrintDebugString() ? &dump_str : nullptr;

    size_t ge_idx_len = input_info.ge_real_idx.size();
    size_t ms_idx_len = input_info.ms_real_idx.size();
    if (ge_idx_len >= ms_idx_len) {
      if (ge_idx_len > ms_idx_len) {
        // Split ms tensor to dynamic ge tensor.
        auto ms_real_idx = input_info.ms_real_idx[kIndex0];
        auto ms_tensor = inputs[ms_real_idx];
        auto new_input_param = input_params[ms_real_idx];
        new_input_param.ori_shape.erase(new_input_param.ori_shape.begin());
        new_input_param.dev_shape.erase(new_input_param.dev_shape.begin());
        for (size_t i = 0; i < ge_idx_len; ++i) {
          std::string arg_name = (ge_input_info.type == Ms2GeParamInfo::DYNAMIC ? ge_input_info.name + std::to_string(i)
                                                                                : ge_input_info.name);
          auto new_address = UpdateKernelTensorAddress(ms_tensor, ge_idx_len, i);
          auto [acl_desc, acl_data] =
            ConvertTensorToAclDesc(new_address, new_input_param, arg_name, dump_str_pointer, true);
          runner_.SetInput(input_info.ge_real_idx[i], acl_desc, acl_data);
          if (device::ascend::AclHelper::IsPrintDebugString()) {
            input_str_[input_info.ge_real_idx[i]] = dump_str;
          }
        }
        AttrConverter attr_converter;
        attr_converter.ConvertValueToRealType(MakeValue(static_cast<int64_t>(ge_idx_len)), kDynamicN, this);
      } else {
        // Convert ms tensor to ge tensor one by one.
        for (size_t i = 0; i < ge_idx_len; ++i) {
          auto ms_real_idx = input_info.ms_real_idx[i];
          auto ge_real_idx = input_info.ge_real_idx[i];
          auto input_param = GetInputParam(inputs, input_on_host_, ms_real_idx);
          std::string arg_name = (ge_input_info.type == Ms2GeParamInfo::DYNAMIC ? ge_input_info.name + std::to_string(i)
                                                                                : ge_input_info.name);
          if (std::holds_alternative<KernelTensor *>(input_param)) {
            auto [acl_desc, acl_data] = ConvertTensorToAclDesc(
              std::get<KernelTensor *>(input_param), input_params[ms_real_idx], arg_name, dump_str_pointer, true);
            runner_.SetInput(ge_real_idx, acl_desc, acl_data);
          } else {
            auto &host_input = std::get<AclHostInfoPtr>(input_param);
            auto [acl_desc, acl_data] =
              ConvertTensorToAclDesc(host_input, input_params[ms_real_idx], arg_name, dump_str_pointer, true);
            if (set_const && host_input->is_const) {
              (void)CALL_ASCEND_API(aclSetTensorConst, acl_desc, host_input->host_addr, host_input->size);
            }
            runner_.SetInput(ge_real_idx, acl_desc, acl_data);
          }
          if (device::ascend::AclHelper::IsPrintDebugString()) {
            input_str_[ge_real_idx] = dump_str;
          }
        }
      }
    }
  }

  // fill null optional input in the range of inputs with placeholder
  runner_.FillOptInputWithPlaceHolder();
}

void AclConverter::ConvertToAclOutput(const PrimitivePtr &prim, const std::vector<KernelTensor *> &outputs,
                                      const std::vector<TensorParams> &output_params) {
  // Get output real index
  MS_EXCEPTION_IF_NULL(prim);
  auto info = GeAdapterManager::GetInstance().GetInfo(prim->name(), true);
  MS_EXCEPTION_IF_NULL(info);
  auto flags = info->GetOutputMappingFlags();

  if ((flags & GeTensorInfo::kEmptyParam) != 0) {
    return;
  }

  size_t num_max_outputs = ((flags & GeTensorInfo::kDynamicParam) ? outputs.size() : info->GetNumOutputsOfMsOpProto());
  std::vector<int64_t> dyn_output_sizes = {};

  if (prim->HasAttr(kAttrDynOutputSizes)) {
    dyn_output_sizes = GetValue<std::vector<int64_t>>(prim->GetAttr(kAttrDynOutputSizes));
    if (dyn_output_sizes.empty()) {
      MS_LOG(EXCEPTION) << "Attribute " << kAttrDynOutputSizes << " of primitive " << prim->name() << " is "
                        << dyn_output_sizes << ", of which size is empty";
    }
  }

  if (flags & GeTensorInfo::kMultiDynParam) {
    num_max_outputs = info->GetNumOutputsOfMsOpProto();
    for (auto &[ms_idx, ge_info] : info->GetMs2GeOutputMap()) {
      if (ge_info.type != Ms2GeParamInfo::DYNAMIC) {
        continue;
      }
      dyn_outputs_map_[ms_idx] = (dyn_output_sizes[ms_idx] > 0 ? dyn_output_sizes[ms_idx] : 1);
      num_max_outputs += dyn_outputs_map_[ms_idx] - 1;
    }
  }

  // pre-allocate output buffer
  if (device::ascend::AclHelper::IsPrintDebugString()) {
    output_str_.clear();
    output_str_.resize(num_max_outputs);
  }
  runner_.ResizeOpOutputs(num_max_outputs);

  if (flags & GeTensorInfo::kMultiDynParam) {
    ConvertOutputsMultiDynParams(prim, outputs, info);
  } else {
    ConvertOutputsNormal(prim, outputs, info);
  }

  for (auto &[ms_idx, ms_output_info] : outputs_idx_convert_map_) {
    AclDumpString dump_str;
    AclDumpString *dump_str_pointer = device::ascend::AclHelper::IsPrintDebugString() ? &dump_str : nullptr;

    for (size_t i = 0; i < ms_output_info.folded_size; i++) {
      auto opt_ge_output_info = info->GetOptGeOutputByMsOutputIndex(ms_idx);
      size_t ms_real_idx = ms_output_info.real_offset + i;
      // mindpore op contains extra output parameters, e.g. ApplyAdagradV2, ApplyAdam
      if (!opt_ge_output_info.has_value()) {
        MS_LOG(DEBUG) << "Not found matched GE input for mindspore input idx:" << ms_idx << " of primitive "
                      << prim->name();
        ms_real_idx += 1;
        continue;
      }
      auto ge_output_info = opt_ge_output_info.value();
      std::string arg_name = (ge_output_info.type == Ms2GeParamInfo::DYNAMIC ? ge_output_info.name + std::to_string(i)
                                                                             : ge_output_info.name);
      size_t acl_real_output_idx = ms_output_info.ge_offset + i;
      MS_LOG(DEBUG) << "Fill acl real output " << acl_real_output_idx << " use ms real output " << ms_real_idx;
      auto [acl_desc, acl_data] =
        ConvertTensorToAclDesc(outputs[ms_real_idx], output_params[ms_real_idx], arg_name, dump_str_pointer, false);
      runner_.SetOutput(acl_real_output_idx, acl_desc, acl_data);
      if (device::ascend::AclHelper::IsPrintDebugString()) {
        output_str_[acl_real_output_idx] = dump_str;
      }
      ms_real_idx += 1;
    }
  }
}

void AclConverter::ConvertOutputsMultiDynParams(const PrimitivePtr &prim, const std::vector<KernelTensor *> &outputs,
                                                const GeAdapterInfoPtr &info) {
  // Calculate GE output proto index to MS real output index and number of folded outputs
  // NOTE: here we use an ordered map to sort ge output indices ascendly
  std::map<uint32_t, MsOutputInfo> ge2ms_real_output_map;
  size_t ms_idx = 0;
  size_t offset = 0;  // offset in real outputs corresponding to output proto index
  while (offset < outputs.size()) {
    size_t num_folded_outputs = (dyn_outputs_map_.count(ms_idx) > 0 ? dyn_outputs_map_[ms_idx] : 1);
    auto opt_ge_output_info = info->GetOptGeOutputByMsOutputIndex(ms_idx);
    if (!opt_ge_output_info.has_value()) {
      MS_LOG(DEBUG) << "Not found matched GE input for mindspore input idx:" << ms_idx << " of primitive "
                    << prim->name();
      continue;
    }
    auto ge_idx = opt_ge_output_info.value().index;
    ge2ms_real_output_map[ge_idx] = MsOutputInfo{ms_idx, offset, num_folded_outputs, 0};
    offset += num_folded_outputs;
    ms_idx += 1;
  }

  if (offset != outputs.size()) {
    MS_LOG(EXCEPTION) << "Number of real outputs is " << outputs.size() << ", which is not equal to expected size "
                      << offset << " of primitive " << prim->name();
  }

  // construct acl outputs by ge output indices increasingly
  size_t ge_start_idx = 0;
  size_t ge_index_cnt = 0;
  for (auto &[ge_idx, ms_info] : ge2ms_real_output_map) {
    if (ge_idx != ge_index_cnt++) {
      MS_LOG(EXCEPTION) << "There is no mindspore output corresponded to ge output index" << ge_index_cnt
                        << " of primitive " << prim->name();
    }
    ms_info.ge_offset = ge_start_idx;
    outputs_idx_convert_map_[ms_info.proto_index] = ms_info;
    ge_start_idx += ms_info.folded_size;
  }
}

void AclConverter::ConvertOutputsNormal(const PrimitivePtr &prim, const std::vector<KernelTensor *> &outputs,
                                        const GeAdapterInfoPtr &info) {
  size_t ms_real_idx = 0;
  size_t num_real_outputs = outputs.size();
  auto flags = info->GetOutputMappingFlags();
  size_t num_folded_outputs =
    ((flags & GeTensorInfo::kDynamicParam) ? outputs.size() - info->GetNumOutputsOfMsOpProto() + 1 : 0);
  // NOTE: num of real outputs params may larger than 'info->GetNumOutputsOfMsOpProto()', e.g. ApplyAdagradV2, ApplyAdam
  for (size_t ms_idx = 0; ms_idx <= info->GetNumOutputsOfMsOpProto(); ++ms_idx) {
    if (ms_real_idx >= num_real_outputs) {
      break;
    }

    auto opt_ge_output_info = info->GetOptGeOutputByMsOutputIndex(ms_idx);
    if (!opt_ge_output_info.has_value()) {
      MS_LOG(DEBUG) << "Not found matched GE input for mindspore input idx:" << ms_idx << " of primitive "
                    << prim->name();
      ms_real_idx += 1;
      continue;
    }
    auto ge_output_info = opt_ge_output_info.value();
    size_t count = (ge_output_info.type == Ms2GeParamInfo::DYNAMIC ? num_folded_outputs : 1);
    size_t ge_start_idx = (ge_output_info.is_after_dynamic ? ge_output_info.index + count - 1 : ge_output_info.index);
    MsOutputInfo ms_output_info = {ms_idx, ms_real_idx, count, ge_start_idx};
    outputs_idx_convert_map_[ms_idx] = ms_output_info;
    ms_real_idx += count;
  }
}

void AclConverter::ConvertAttrToAclInput(const mindspore::HashMap<std::string, ValuePtr> &attrs,
                                         const std::string &kernel_name, std::vector<TensorParams> *input_params) {
  MS_LOG(DEBUG) << "Start convert attr to acl input";
  MS_EXCEPTION_IF_NULL(input_params);
  auto info = GeAdapterManager::GetInstance().GetInfo(kernel_name, true);
  MS_EXCEPTION_IF_NULL(info);
  for (const auto &[input_idx, ms_attr_name] : info->attr_input_map()) {
    auto iter = attrs.find(ms_attr_name);
    if (iter == attrs.end()) {
      MS_LOG(DEBUG) << "Not found attr " << ms_attr_name << " for primitive " << kernel_name << ", ignore it.";
      continue;
    }

    auto opt_ge_input_info = info->GetOptGeInputByMsInputIndex(input_idx);
    // mindpore input mapped to GE attribute
    if (!opt_ge_input_info.has_value()) {
      MS_LOG(DEBUG) << "Not found matched GE input for mindspore input idx:" << input_idx << " of primitive "
                    << kernel_name;
      continue;
    }

    auto &ge_input_info = opt_ge_input_info.value();
    if (ge_input_info.type == Ms2GeParamInfo::DYNAMIC) {
      MS_LOG(EXCEPTION) << "Mindspore attribute " << ms_attr_name << " mapped to a dynamic GE input";
    }

    if (ms_and_ge_inputs_idx_info_.find(input_idx) == ms_and_ge_inputs_idx_info_.end()) {
      MS_LOG(EXCEPTION) << kernel_name << " can't convert " << ms_attr_name << " attr to input index " << input_idx;
    }
    auto ms_real_idx = ms_and_ge_inputs_idx_info_[input_idx].ms_real_idx[kIndex0];

    AttrToInputConverter attr_coverter;
    TensorParams new_params;
    attr_coverter.ConvertValueToRealType(iter->second, ms_attr_name, &new_params);
    host_save_list_[ms_real_idx] = std::move(attr_coverter.GetData());

    auto acl_host_input = std::make_shared<AclHostInfo>(
      host_save_list_[ms_real_idx].data(), host_save_list_[ms_real_idx].size(), new_params.data_type, true);
    input_on_host_.emplace(ms_real_idx, acl_host_input);
    if (ms_real_idx >= input_params->size()) {
      input_params->resize(ms_real_idx + 1);
      (*input_params)[ms_real_idx] = new_params;
    }
    MS_LOG(DEBUG) << "Fill acl real input " << ms_real_idx << " with attribute " << ms_attr_name << " of primitive "
                  << kernel_name;
  }
  MS_LOG(DEBUG) << "Convert attr to acl input over";
}

std::string AclConverter::GetFormatFromInputAttrMap(const std::vector<KernelTensor *> &inputs,
                                                    const std::string &kernel_name) {
  MS_LOG(DEBUG) << "Start GetFormatFromInputAttrMap";
  std::string format;
  auto info = GeAdapterManager::GetInstance().GetInfo(kernel_name, true);
  MS_EXCEPTION_IF_NULL(info);
  for (const auto &[input_idx, attr_name] : info->input_attr_map()) {
    if (attr_name == kAttrDataFormat) {
      // adapter dyn_num input
      size_t ms_real_idx = ms_and_ge_inputs_idx_info_[input_idx].ms_real_idx[kIndex0];
      MS_LOG(DEBUG) << "Operator " << kernel_name << " converts input " << input_idx << " to attribute " << attr_name;
      if (ms_real_idx >= inputs.size()) {
        MS_LOG(DEBUG) << "Operator " << kernel_name << " index " << ms_real_idx
                      << " is out of range of inputs, size of which is " << inputs.size() << ", ignore it.";
        continue;
      }
      MS_EXCEPTION_IF_NULL(inputs[ms_real_idx]);
      auto format_enum = GetScalarValue<int64_t>(inputs[ms_real_idx]->GetValue());
      format = FormatEnumToString(static_cast<mindspore::Format>(format_enum.value()));
    }
  }
  return format;
}

void AclConverter::ConvertInputToAclAttr(const std::vector<KernelTensor *> &inputs, const std::string &kernel_name) {
  MS_LOG(DEBUG) << "Start convert input to acl attr";
  auto info = GeAdapterManager::GetInstance().GetInfo(kernel_name, true);
  MS_EXCEPTION_IF_NULL(info);
  for (const auto &[input_idx, attr_name] : info->input_attr_map()) {
    if (ms_and_ge_inputs_idx_info_.find(input_idx) == ms_and_ge_inputs_idx_info_.end()) {
      continue;
    }
    // adapter dyn_num input
    size_t ms_real_idx = ms_and_ge_inputs_idx_info_[input_idx].ms_real_idx[kIndex0];
    MS_LOG(DEBUG) << "Operator " << kernel_name << " converts input " << input_idx << " to attribute " << attr_name;
    if (ms_real_idx >= inputs.size()) {
      MS_LOG(DEBUG) << "Operator " << kernel_name << " index " << ms_real_idx
                    << " is out of range of inputs, size of which is " << inputs.size() << ", ignore it.";
      continue;
    }
    MS_EXCEPTION_IF_NULL(inputs[ms_real_idx]);
    ValuePtr attr_value = inputs[ms_real_idx]->GetValue();
    if (attr_value->isa<None>()) {
      MS_LOG(DEBUG) << "Input " << ms_real_idx << " of operator " << kernel_name << " is None, ignore it.";
      continue;
    }
    info->GetGeAttrValueByMsInputValue(input_idx + 1, &attr_value);

    AttrConverter attr_coverter;
    attr_coverter.ConvertValueToRealType(attr_value, attr_name, this);
  }
  MS_LOG(DEBUG) << "Convert input to acl attr over";
}

void AclConverter::ConvertToAclAttr(const mindspore::HashMap<std::string, ValuePtr> &attrs,
                                    const std::string &prim_name, std::vector<std::string> *ms_attr_str) {
  MS_LOG(DEBUG) << "Start convert mindspore attr to acl attr";
  auto info = GeAdapterManager::GetInstance().GetInfo(prim_name, true);
  MS_EXCEPTION_IF_NULL(info);
  auto &ms_ge_attr_map = info->attr_map();

  for (const auto &[ms_attr_name, ge_attr_name] : ms_ge_attr_map) {
    ValuePtr attr_value = nullptr;
    if (attrs.count(ms_attr_name) != 0) {
      attr_value = attrs.at(ms_attr_name);
    }
    info->GetGeAttrValueByMsAttrValue(ms_attr_name, &attr_value);

    // Dump Info
    if (ms_attr_str != nullptr) {
      std::stringstream ss;
      ss << "attr name: " << ms_attr_name << ", value: " << attr_value->ToString();
      (void)ms_attr_str->emplace_back(ss.str());
    }

    AttrConverter attr_coverter;
    attr_coverter.ConvertValueToRealType(attr_value, ge_attr_name, this);
  }
  MS_LOG(DEBUG) << "convert mindspore attr to acl attr over";
}

void AclConverter::ConvertToAclOpType(const std::string &prim_name) {
  auto info = GeAdapterManager::GetInstance().GetInfo(prim_name, true);
  MS_EXCEPTION_IF_NULL(info);
  auto op_type = info->op_type();
  runner_.SetName(op_type);
}

void AclConverter::ConvertMsIdxToGeIdx(const PrimitivePtr &prim, const std::vector<KernelTensor *> &inputs) {
  if (is_create_mapping_) {
    return;
  }

  MS_EXCEPTION_IF_NULL(prim);
  auto prim_name = prim->name();
  auto info = GeAdapterManager::GetInstance().GetInfo(prim_name, true);
  MS_EXCEPTION_IF_NULL(info);
  auto mapping_flags = info->GetInputMappingFlags();

  ms_and_ge_inputs_sort_info_.clear();
  ms_and_ge_inputs_idx_info_.clear();
  auto dyn_input_sizes = GetDynamicInputSize(prim);

  size_t ms_real_idx = 0;
  size_t attr_offset = 0;
  size_t num_real_inputs = inputs.size();
  bool dynamic_tuple_flag = false;
  std::unordered_map<size_t, size_t> idx_dyn_input_num_map;
  for (int ms_idx = 0; ms_idx <= info->GetMaxMsProtoIndexOfInputMap(); ++ms_idx) {
    // Input to attr.
    auto opt_ge_input_info = info->GetOptGeInputByMsInputIndex(ms_idx);
    if (!opt_ge_input_info.has_value()) {
      if (info->input_attr_map().find(ms_idx) != info->input_attr_map().end()) {
        ms_and_ge_inputs_sort_info_.emplace(
          std::make_pair(static_cast<size_t>(ms_idx), std::numeric_limits<size_t>::max()),
          std::make_pair(std::vector<size_t>{ms_real_idx}, std::vector<size_t>{}));
        ms_real_idx++;
      }
      continue;
    }
    auto &ge_input_info = opt_ge_input_info.value();

    // Attr to input.
    auto attr_iter = info->attr_input_map().find(ms_idx);
    if ((attr_iter != info->attr_input_map().end()) && (prim->attrs().count(attr_iter->second) > 0)) {
      ms_and_ge_inputs_sort_info_.emplace(std::make_pair(static_cast<size_t>(ms_idx), ge_input_info.index),
                                          std::make_pair(std::vector<size_t>{ms_real_idx}, std::vector<size_t>{0}));
      ms_real_idx++;
      attr_offset++;
      continue;
    }

    size_t input_idx = ms_real_idx - attr_offset;
    if (input_idx >= num_real_inputs) {
      break;
    }

    if (mapping_flags & (GeTensorInfo::kDynamicParam | GeTensorInfo::kMultiDynParam)) {
      if (dyn_input_sizes.empty()) {
        // Convert tuple/list input index to ge index.
        size_t ge_input_num = 1;
        if (ge_input_info.type == Ms2GeParamInfo::DYNAMIC) {
          auto tensor = inputs[input_idx];
          ge_input_num = GetTupleSize(tensor);
        }

        std::vector<size_t> ge_index(ge_input_num);
        ms_and_ge_inputs_sort_info_.emplace(std::make_pair(static_cast<size_t>(ms_idx), ge_input_info.index),
                                            std::make_pair(std::vector<size_t>{input_idx}, ge_index));
        ms_real_idx++;
        dynamic_tuple_flag = true;
      } else {
        // Convert dynamic ms index to ge index.
        size_t ms_input_num = 1;
        if (ge_input_info.type == Ms2GeParamInfo::DYNAMIC) {
          if (dyn_input_sizes.size() <= static_cast<size_t>(ms_idx)) {
            MS_LOG(EXCEPTION) << "The size of dyn_input_sizes is " << dyn_input_sizes.size()
                              << ", but ms proto index is " << ms_idx << ", please check!";
          }
          ms_input_num = dyn_input_sizes[ms_idx];
        }
        std::vector<size_t> ms_index(ms_input_num);
        std::iota(ms_index.begin(), ms_index.end(), input_idx);
        ms_and_ge_inputs_sort_info_.emplace(std::make_pair(static_cast<size_t>(ms_idx), ge_input_info.index),
                                            std::make_pair(ms_index, ms_index));
        idx_dyn_input_num_map[ms_idx] = ms_input_num;
        ms_real_idx += ms_input_num;
      }
    } else {
      ms_and_ge_inputs_sort_info_.emplace(std::make_pair(static_cast<size_t>(ms_idx), ge_input_info.index),
                                          std::make_pair(std::vector<size_t>{input_idx}, std::vector<size_t>{0}));
      ms_real_idx++;
    }
  }
  for (const auto &input_attr : info->input_attr_map()) {
    auto input_idx = static_cast<size_t>(input_attr.first);
    if (input_idx >= num_real_inputs) {
      break;
    }
    size_t dyn_input_offset =
      std::accumulate(idx_dyn_input_num_map.begin(), idx_dyn_input_num_map.end(), 0,
                      [input_idx](size_t value, const std::unordered_map<size_t, size_t>::value_type &idx_num) {
                        return idx_num.first < input_idx ? value + idx_num.second - 1 : value;
                      });

    if (static_cast<int>(input_idx) > info->GetMaxMsProtoIndexOfInputMap()) {
      ms_and_ge_inputs_sort_info_.emplace(
        std::make_pair(input_idx, std::numeric_limits<size_t>::max()),
        std::make_pair(std::vector<size_t>{input_idx + dyn_input_offset}, std::vector<size_t>{}));
    }
  }

  if (!dynamic_tuple_flag) {
    is_create_mapping_ = true;
  }
  GenerateRealGeIdx();
}

void AclConverter::GenerateRealGeIdx() {
  size_t max_ge_number = 0;
  for (auto &[adapter_idx, mapping_info] : ms_and_ge_inputs_sort_info_) {
    auto block_size = mapping_info.second.size();
    if (block_size == 1) {
      mapping_info.second = {max_ge_number};
    } else if (block_size > 1) {
      std::vector<size_t> ge_real_index(block_size);
      std::iota(ge_real_index.begin(), ge_real_index.end(), max_ge_number);
      mapping_info.second = ge_real_index;
    }
    max_ge_number += block_size;
    MS_LOG(DEBUG) << "Dynamic info:[ms_adapter_idx] " << adapter_idx.first << ",[ge_adapter_idx] " << adapter_idx.second
                  << ",[ms_real_idx] " << mapping_info.first << ",[ge_real_idx] " << mapping_info.second;
  }
  std::transform(ms_and_ge_inputs_sort_info_.begin(), ms_and_ge_inputs_sort_info_.end(),
                 std::inserter(ms_and_ge_inputs_idx_info_, ms_and_ge_inputs_idx_info_.begin()), [](const auto &pair) {
                   return std::make_pair(pair.first.first,
                                         MsInputIdxToGe{pair.first.second, pair.second.first, pair.second.second});
                 });

  if (device::ascend::AclHelper::IsPrintDebugString()) {
    input_str_.clear();
    input_str_.resize(max_ge_number);
  }
  host_save_list_.clear();
  host_save_list_.resize(max_ge_number);
  runner_.ResizeOpInputs(max_ge_number);
}

aclDataType AclConverter::ConvertType(TypeId type) {
  if (type == kMetaTypeNone || type == kTypeUnknown) {
    return ACL_DT_UNDEFINED;
  }
  if (type == kObjectTypeString) {
    return ACL_STRING;
  }
  if (type <= kNumberTypeBegin || type >= kNumberTypeEnd) {
    MS_LOG(EXCEPTION) << "Invalid datatype:" << type;
  }
  auto iter = kDataTypeToAclDataTypeTable.find(type);
  if (iter == kDataTypeToAclDataTypeTable.end()) {
    MS_LOG(EXCEPTION) << "Invalid datatype:" << type;
  }
  auto acl_type = iter->second;
  if (acl_type == ACL_DT_UNDEFINED) {
    MS_LOG(EXCEPTION) << "Invalid datatype:" << type;
  }
  return acl_type;
}

aclFormat AclConverter::ConvertFormat(const std::string &format) {
  auto iter = kMsFormatToAclFormat.find(format);
  if (iter == kMsFormatToAclFormat.end()) {
    MS_LOG(EXCEPTION) << "Invalid format:" << format;
  }
  return iter->second;
}

std::pair<aclTensorDesc *, aclDataBuffer *> AclConverter::ConvertTensorToAclDesc(const AddressPtr &address,
                                                                                 const TensorParams &params,
                                                                                 const std::string &desc_name,
                                                                                 AclDumpString *dump_str,
                                                                                 bool is_input) const {
  AclTensorDescMaker tensor;
  if (dump_str != nullptr) {
    dump_str->tensor_name = desc_name;
  }

  // Create desc.
  aclTensorDesc *acl_desc = nullptr;
  if (params.data_type == kMetaTypeNone) {
    acl_desc = tensor.Create(ACL_DT_UNDEFINED, ACL_FORMAT_UNDEFINED)
                 .SetTensorPlaceMent(ACL_MEMTYPE_HOST_COMPILE_INDEPENDENT)
                 .SetName(desc_name)
                 .Get();
    DumpAclString(ACL_DT_UNDEFINED, params.ori_shape, params.dev_shape, ACL_FORMAT_UNDEFINED, ACL_FORMAT_UNDEFINED,
                  dump_str);
  } else {
    auto acl_data_type = ConvertType(params.data_type);
    auto acl_ori_format = ConvertFormat(params.ori_format);
    auto acl_dev_format = ConvertFormat(params.dev_format);
    acl_desc = tensor.Create(acl_data_type, params.ori_shape, acl_ori_format)
                 .SetShape(params.dev_shape)
                 .SetFormat(acl_dev_format)
                 .SetName(desc_name)
                 .Get();
    DumpAclString(acl_data_type, params.ori_shape, params.dev_shape, acl_ori_format, acl_dev_format, dump_str);
  }
  MS_EXCEPTION_IF_NULL(acl_desc);

  // Create buf.
  MS_EXCEPTION_IF_NULL(address);
  auto buffer_maker = std::make_shared<AclTensorBufferMaker>(address->addr, address->size, params.data_type, is_input);
  auto acl_data = buffer_maker->Get();
  MS_EXCEPTION_IF_NULL(acl_data);

  return std::make_pair(acl_desc, acl_data);
}

std::pair<aclTensorDesc *, aclDataBuffer *> AclConverter::ConvertTensorToAclDesc(const KernelTensor *ori_tensor,
                                                                                 const TensorParams &params,
                                                                                 const std::string &desc_name,
                                                                                 AclDumpString *dump_str,
                                                                                 bool is_input) const {
  AclTensorDescMaker tensor;
  if (dump_str != nullptr) {
    dump_str->tensor_name = desc_name;
  }

  // Create desc.
  aclTensorDesc *acl_desc = nullptr;
  if (params.data_type == kMetaTypeNone) {
    acl_desc = tensor.Create(ACL_DT_UNDEFINED, ACL_FORMAT_UNDEFINED)
                 .SetTensorPlaceMent(ACL_MEMTYPE_HOST_COMPILE_INDEPENDENT)
                 .SetName(desc_name)
                 .Get();
    DumpAclString(ACL_DT_UNDEFINED, params.ori_shape, params.dev_shape, ACL_FORMAT_UNDEFINED, ACL_FORMAT_UNDEFINED,
                  dump_str);
  } else {
    auto acl_data_type = ConvertType(params.data_type);
    auto acl_ori_format = ConvertFormat(params.ori_format);
    auto acl_dev_format = ConvertFormat(params.dev_format);
    acl_desc = tensor.Create(acl_data_type, params.ori_shape, acl_ori_format)
                 .SetShape(params.dev_shape)
                 .SetFormat(acl_dev_format)
                 .SetName(desc_name)
                 .Get();
    DumpAclString(acl_data_type, params.ori_shape, params.dev_shape, acl_ori_format, acl_dev_format, dump_str);
  }
  MS_EXCEPTION_IF_NULL(acl_desc);

  // Create buf.
  MS_EXCEPTION_IF_NULL(ori_tensor);
  auto buffer_maker =
    std::make_shared<AclTensorBufferMaker>(ori_tensor->device_ptr(), ori_tensor->size(), params.data_type, is_input);
  auto acl_data = buffer_maker->Get();
  MS_EXCEPTION_IF_NULL(acl_data);

  return std::make_pair(acl_desc, acl_data);
}

std::pair<aclTensorDesc *, aclDataBuffer *> AclConverter::ConvertTensorToAclDesc(const AclHostInfoPtr &acl_host_info,
                                                                                 const TensorParams &params,
                                                                                 const std::string &desc_name,
                                                                                 AclDumpString *dump_str,
                                                                                 bool is_input) const {
  MS_EXCEPTION_IF_NULL(acl_host_info);
  AclTensorDescMaker tensor;
  if (dump_str != nullptr) {
    dump_str->tensor_name = desc_name;
  }

  // Create desc.
  aclTensorDesc *acl_desc = nullptr;
  if (params.data_type == kMetaTypeNone) {
    acl_desc = tensor.Create(ACL_DT_UNDEFINED, ACL_FORMAT_UNDEFINED)
                 .SetTensorPlaceMent(ACL_MEMTYPE_HOST_COMPILE_INDEPENDENT)
                 .SetName(desc_name)
                 .Get();
    DumpAclString(ACL_DT_UNDEFINED, params.ori_shape, params.dev_shape, ACL_FORMAT_UNDEFINED, ACL_FORMAT_UNDEFINED,
                  dump_str);
  } else {
    auto acl_data_type = ConvertType(acl_host_info->dtype_id);
    auto acl_ori_format = ConvertFormat(params.ori_format);
    acl_desc = tensor.Create(acl_data_type, params.ori_shape, acl_ori_format)
                 .SetTensorPlaceMent(ACL_MEMTYPE_HOST)
                 .SetName(desc_name)
                 .Get();
    DumpAclString(acl_data_type, params.ori_shape, params.dev_shape, acl_ori_format, ACL_FORMAT_UNDEFINED, dump_str);
  }
  MS_EXCEPTION_IF_NULL(acl_desc);

  // Create buf.
  auto buffer_maker = std::make_shared<AclTensorBufferMaker>(acl_host_info->host_addr, acl_host_info->size,
                                                             acl_host_info->dtype_id, is_input);
  auto acl_data = buffer_maker->Get();
  MS_EXCEPTION_IF_NULL(acl_data);

  return std::make_pair(acl_desc, acl_data);
}

std::pair<aclTensorDesc *, aclDataBuffer *> AclConverter::ConvertTensorToAclDesc(const tensor::TensorPtr &host_tensor,
                                                                                 const TensorParams &params,
                                                                                 const std::string &desc_name,
                                                                                 AclDumpString *dump_str) const {
  AclTensorDescMaker tensor;
  if (dump_str != nullptr) {
    dump_str->tensor_name = desc_name;
  }

  // Create desc.
  aclTensorDesc *acl_desc = nullptr;
  if (params.data_type == kMetaTypeNone) {
    acl_desc = tensor.Create(ACL_DT_UNDEFINED, ACL_FORMAT_UNDEFINED)
                 .SetTensorPlaceMent(ACL_MEMTYPE_HOST_COMPILE_INDEPENDENT)
                 .SetName(desc_name)
                 .Get();
    DumpAclString(ACL_DT_UNDEFINED, params.ori_shape, params.dev_shape, ACL_FORMAT_UNDEFINED, ACL_FORMAT_UNDEFINED,
                  dump_str);
  } else {
    MS_EXCEPTION_IF_NULL(host_tensor);
    auto acl_data_type = ConvertType(host_tensor->data_type());
    auto acl_ori_format = ConvertFormat(params.ori_format);
    acl_desc = tensor.Create(acl_data_type, params.ori_shape, acl_ori_format)
                 .SetTensorPlaceMent(ACL_MEMTYPE_HOST)
                 .SetName(desc_name)
                 .Get();
    DumpAclString(acl_data_type, params.ori_shape, params.dev_shape, acl_ori_format, ACL_FORMAT_UNDEFINED, dump_str);
  }
  MS_EXCEPTION_IF_NULL(acl_desc);

  // Create buf.
  auto buffer_maker = std::make_shared<AclTensorBufferMaker>(host_tensor);
  auto acl_data = buffer_maker->Get();
  MS_EXCEPTION_IF_NULL(acl_data);

  return std::make_pair(acl_desc, acl_data);
}

template <typename T>
void AclConverter::AclRunnerAddAttr(const std::string &attrName, T value) {
  runner_.AddAttr(attrName, value);
  if (device::ascend::AclHelper::IsPrintDebugString()) {
    std::stringstream ss;
    ss << ", value: " << value;
    attr_map_str_[attrName] = ss.str();
    MS_LOG(DEBUG) << "set acl attr:" << attrName << " value:" << value;
  }
}

std::string AclConverter::DebugString() const {
  if (!device::ascend::AclHelper::IsPrintDebugString()) {
    return "";
  }
  std::stringstream ss;
  ss << "[AclLaunchInfo]OpType:" << runner_.GetName() << std::endl;
  for (size_t i = 0; i < runner_.GetNumRealInputs(); ++i) {
    ss << "InputDesc[" << i << "]:";
    ss << AclTensorDescString(input_str_[i]) << std::endl;
  }
  for (auto iter = attr_map_str_.begin(); iter != attr_map_str_.end(); iter++) {
    ss << "Attr name : " << iter->first << iter->second << std::endl;
  }
  for (size_t i = 0; i < runner_.GetNumRealOutputs(); ++i) {
    ss << "OutputDesc[" << i << "]:";
    ss << AclTensorDescString(output_str_[i]) << std::endl;
  }
  return ss.str();
}

void AclConverter::ProcessRunnerSpecialInfo(const std::string &prim_name,
                                            const std::vector<TensorParams> &output_params, bool is_dynamic) {
  auto opinfo = GeAdapterManager::GetInstance().GetInfo(prim_name, true);
  MS_EXCEPTION_IF_NULL(opinfo);
  auto op_type = opinfo->op_type();
  if (!AclAdapterManager::GetInstance().CheckAclAdapter(op_type)) {
    is_dynamic_ = ReadStatciAclOp(prim_name, is_dynamic);
    precision_mode_ = DEFAULT_MODE;
    return;
  }
  auto info = AclAdapterManager::GetInstance().GetOpInfo(op_type);

  // Set need retrieve output shape flag.
  is_need_retrieve_output_shape_ = info.is_need_retrieve_output_shape();

  // Set dynamic or static compile mode.
  is_dynamic_ = info.is_dynamic(is_dynamic);

  // Set acl precision mode
  precision_mode_ = info.precision_mode();
}

void AclConverter::SetRunnerSpecialInfo() {
  if (is_dynamic_) {
    runner_.SetDynamicMode();
  } else {
    runner_.SetStaticMode();
  }
  runner_.SetPrecisionMode(precision_mode_);
}

void AclConverter::Run(void *stream_ptr) { runner_.Run(stream_ptr, is_need_retrieve_output_shape_); }

void AclConverter::Reset() {
  runner_.Reset();
  if (device::ascend::AclHelper::IsPrintDebugString()) {
    output_str_.clear();
  }
  input_on_host_.clear();
}
}  // namespace  mindspore::device::ascend
