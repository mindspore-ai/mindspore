/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_TRANSFORM_ACL_IR_ACL_CONVERT_H_
#define MINDSPORE_CCSRC_TRANSFORM_ACL_IR_ACL_CONVERT_H_

#include <map>
#include <set>
#include <string>
#include <vector>
#include <utility>
#include <memory>
#include <algorithm>
#include "transform/acl_ir/acl_utils.h"
#include "transform/graph_ir/op_adapter_util.h"
#include "kernel/kernel.h"

namespace mindspore {
namespace transform {
using AddressPtr = kernel::AddressPtr;
using KernelTensor = mindspore::kernel::KernelTensor;

typedef enum { SET_ACL_ATTR, SET_ACL_ATTR_TO_INPUT } AttrConvertMode;
template <typename T>
struct AttrDeclType {
  using type = T;
};

struct TensorParams {
  TypeId data_type{kTypeUnknown};
  ShapeVector ori_shape{};
  ShapeVector dev_shape{};
  std::string ori_format{kOpFormat_DEFAULT};
  std::string dev_format{kOpFormat_DEFAULT};
  size_t type_size;
  bool is_default;
  bool all_fp32;
};

struct AclDumpString {
  std::string tensor_name;
  std::string data_type;
  std::string ori_shape;
  std::string dev_shape;
  std::string ori_format;
  std::string dev_format;
  enum class TensorType {
    kDeviceTensor = 0,
    kNullTensor = 1,
    kHostTensor = 2,
  } tensor_type;
};

// record the relationship between input in prototype and corresponding real inputs
struct MsInputInfo {
  // input index in operator prototype
  size_t proto_index;
  // for dynamic input, it is offset of the first element of dynamic input in real inputs; for normal input, it is the
  // offset of the input in real inputs
  size_t real_offset;
  // for dynamic input, it is the number of real inputs corresponding to `proto_index`; for normal input, it is const
  // value 1
  size_t folded_size;
};

using SetInputFunc =
  std::function<void(const MsInputInfo &ms_input_info, size_t ge_offset, const Ms2GeParamInfo &ge_input_info)>;

class AclConverter {
 public:
  void ConvertToAclOpType(const std::string &prim_name);
  void ResizeAclOpInputs(const PrimitivePtr &prim);
  void ConvertToAclInput(const PrimitivePtr &prim, const AclInputToHost &host_inputs,
                         const std::vector<KernelTensor *> &inputs, const std::vector<TensorParams> &input_params);
  void ConvertToAclOutput(const std::string &kernel_name, const std::vector<KernelTensor *> &outputs,
                          const std::vector<TensorParams> &output_params);

  void ConvertValueDependToHostInput(const std::string &kernel_name, const std::vector<KernelTensor *> &inputs,
                                     const std::vector<TensorParams> &input_params,
                                     const std::set<int64_t> &value_depend_args, AclInputToHost *inputs_on_host);

  void ConvertAttrToAclInput(const mindspore::HashMap<std::string, ValuePtr> &attrs, const std::string &kernel_name,
                             AclInputToHost *inputs_on_host);
  void ConvertInputToAclAttr(const std::vector<KernelTensor *> &inputs, const std::string &kernel_name);
  void ConvertInputToAclAttr(const AclInputToHost &inputs, const std::string &kernel_name);
  void ConvertToAclAttr(const mindspore::HashMap<std::string, ValuePtr> &attrs, const std::string &prim_name,
                        std::vector<std::string> *ms_attr_str);
  void ProcessRunnerSpecialInfo(const std::string &prim_name, const std::vector<TensorParams> &output_params);
  void SetRunnerSpecialInfo();

  bool is_need_retrieve_output_shape() const { return is_need_retrieve_output_shape_; }

  std::string DebugString() const;

  void Run(void *stream_ptr);

  std::vector<std::vector<int64_t>> SyncData() { return runner_.SyncData(); }

  void Reset();

  static aclDataType ConvertType(TypeId type);
  static aclFormat ConvertFormat(const std::string &format);
  std::string GetFormatFromInputAttrMap(const std::vector<KernelTensor *> &inputs, const std::string &kernel_name);

  static std::pair<aclTensorDesc *, aclDataBuffer *> CreateTensorDesc(const tensor::TensorPtr &tensor,
                                                                      const ShapeVector &dev_shape,
                                                                      const std::string &dev_format,
                                                                      const std::string &desc_name);

 private:
  friend class AttrConverter;
  std::pair<aclTensorDesc *, aclDataBuffer *> ConvertTensorToAclDesc(const AddressPtr &address,
                                                                     const TensorParams &params,
                                                                     const std::string &desc_name,
                                                                     AclDumpString *dump_str) const;
  std::pair<aclTensorDesc *, aclDataBuffer *> ConvertTensorToAclDesc(const tensor::TensorPtr &host_tensor,
                                                                     const TensorParams &params,
                                                                     const std::string &desc_name,
                                                                     AclDumpString *dump_str) const;

  template <typename T>
  void AclRunnerAddAttr(const std::string &attrName, T value);

  // convert acl inputs for operator with at most one dynamic parameter, general case
  void ConvertInputsNormal(const PrimitivePtr &prim, const std::vector<KernelTensor *> &inputs,
                           const GeAdapterInfoPtr &info, const SetInputFunc &convert_one_input);

  // convert acl inputs for operator with more than one dynamic parameters, rare case
  void ConvertInputsMutiDynParams(const PrimitivePtr &prim, const std::vector<KernelTensor *> &inputs,
                                  const GeAdapterInfoPtr &info, const SetInputFunc &convert_one_input);

  AclRunner runner_;

  std::vector<AclDumpString> input_str_;
  std::vector<AclDumpString> output_str_;
  std::vector<std::string> attr_map_str_;
  // number of folded inputs of dynamic input, only used for op with only one dynamic input
  size_t num_folded_inputs_ = 0;
  bool is_dynamic_ = false;
  AclPrecisionMode precision_mode_ = FORCE_FP32;
  bool is_need_retrieve_output_shape_ = false;
  // Fields for op containing multiple dynamic inputs, since operators with more than one dynamic inputs are rare, for
  // speed reason, we process this case separately.
  // Map for recording [MindSpore op input proto index of dynamic input] to [its number of folded inputs]
  // NOTE: here the map MUST be an ordered map to sort the input indices ascendly.
  std::map<size_t, size_t> dyn_inputs_map_;
};

using AclConverterPtr = std::shared_ptr<AclConverter>;

template <typename ConvertType>
class AttrHelper {
 public:
  template <typename T>
  void ConvertValueToRealType(const ValuePtr &value, const std::string &attr_name, T trans_struct);
  template <typename T>
  void ConvertListAttr(const ValuePtr &value, T trans_struct);

  void GetValueSequenceDataTypeAndShape(const ValuePtrList &value_sequence, TypePtr *data_type, ShapeVector *shape,
                                        bool *is_ge_datatype);

  void ConvertValueToDstType(const ValuePtr &value, const TypeId src_type);

  template <typename T>
  void ConvertValueSequenceToList(const ValuePtr &value, std::vector<T> *array_list) const {
    MS_EXCEPTION_IF_NULL(array_list);
    const auto &value_sequence = value->cast<ValueSequencePtr>()->value();
    auto val = value_sequence[0];
    if (val->isa<Scalar>()) {
      auto list_value = GetValue<std::vector<T>>(value);
      array_list->insert(array_list->end(), list_value.begin(), list_value.end());
    }
    if (val->isa<ValueSequence>()) {
      for (size_t i = 0; i < value_sequence.size(); i++) {
        ConvertValueSequenceToList(value_sequence[i], array_list);
      }
    }
  }

  std::string attr_name_;
};

class AttrConverter : public AttrHelper<AttrConverter> {
 public:
  template <typename T>
  void ConvertValue(const ValuePtr &value, const AttrDeclType<T> &, AclConverter *acl_converter) {
    auto real_val = GetValue<T>(value);
    MS_EXCEPTION_IF_NULL(acl_converter);
    acl_converter->AclRunnerAddAttr(attr_name_, real_val);
  }

  void ConvertValue(const ValuePtr &value, const AttrDeclType<int32_t> &, AclConverter *acl_converter) {
    auto real_val = static_cast<int64_t>(GetValue<int32_t>(value));
    MS_EXCEPTION_IF_NULL(acl_converter);
    acl_converter->AclRunnerAddAttr(attr_name_, real_val);
  }

  template <typename T>
  void ConvertValue(const ValuePtr &value, const AttrDeclType<std::vector<T>> &, AclConverter *acl_converter) {
    std::vector<T> array_list;
    ConvertValueSequenceToList(value, &array_list);
    MS_EXCEPTION_IF_NULL(acl_converter);
    acl_converter->AclRunnerAddAttr(attr_name_, array_list);
  }

  void ConvertValue(const ValuePtr &value, const AttrDeclType<std::vector<::ge::DataType>> &,
                    AclConverter *acl_converter) {
    std::vector<::ge::DataType> data;
    if (!value->isa<ValueTuple>() && !value->isa<ValueList>()) {
      MS_LOG(EXCEPTION) << "value must be sequence, but got " << value->ToString();
    }
    auto vec = value->isa<ValueTuple>() ? value->cast<ValueTuplePtr>()->value() : value->cast<ValueListPtr>()->value();
    (void)std::transform(vec.begin(), vec.end(), std::back_inserter(data),
                         [](const ValuePtr &it) { return it->cast<GeDataTypeImmPtr>()->value(); });
    MS_EXCEPTION_IF_NULL(acl_converter);
    acl_converter->AclRunnerAddAttr(attr_name_, data);
  }

  void ConvertValue(const ValuePtr &value, const AttrDeclType<std::vector<int32_t>> &, AclConverter *acl_converter) {
    std::vector<int32_t> array_list;
    ConvertValueSequenceToList(value, &array_list);
    std::vector<int64_t> array_list_int64;
    (void)std::transform(array_list.begin(), array_list.end(), std::back_inserter(array_list_int64),
                         [](const int val) { return IntToLong(val); });
    MS_EXCEPTION_IF_NULL(acl_converter);
    acl_converter->AclRunnerAddAttr(attr_name_, array_list_int64);
  }

  template <typename T>
  void ConvertValue(const ValuePtr &value, const AttrDeclType<std::vector<std::vector<T>>> &, const ShapeVector &shape,
                    AclConverter *acl_converter) {
    std::vector<T> array_list;
    ConvertValueSequenceToList(value, &array_list);
    std::vector<std::vector<int64_t>> array_list_int64(shape[0], std::vector<int64_t>(shape[1], 0));
    for (int64_t i = 0; i < shape[0]; ++i) {
      array_list_int64[i].assign(array_list.begin() + i * shape[1], array_list.begin() + (i + 1) * shape[1]);
    }
    MS_EXCEPTION_IF_NULL(acl_converter);
    acl_converter->AclRunnerAddAttr(attr_name_, array_list_int64);
  }
};

class ValueDependToInputConverter : public AttrHelper<ValueDependToInputConverter> {
 public:
  const tensor::TensorPtr &GetTensor() const { return tensor_; }
  const std::map<TypeId, TypeId> &GetValueDependCastMap() {
    static const std::map<TypeId, TypeId> kValueDependCastMap = {{kNumberTypeInt32, kNumberTypeInt64},
                                                                 {kNumberTypeInt64, kNumberTypeInt32},
                                                                 {kNumberTypeFloat32, kNumberTypeFloat64},
                                                                 {kNumberTypeFloat64, kNumberTypeFloat32}};
    return kValueDependCastMap;
  }

  void ConvertValue(const ValuePtr &value, const AttrDeclType<int64_t> &) {
    MS_EXCEPTION_IF_NULL(value);
    std::vector<int32_t> vec;
    auto ori_vec = ops::GetArrayValue<int64_t>(value).value().ToVector();
    (void)std::transform(ori_vec.begin(), ori_vec.end(), std::back_inserter(vec),
                         [](const auto &v) { return static_cast<int32_t>(v); });
    tensor_ = std::make_shared<tensor::Tensor>(vec);
    MS_EXCEPTION_IF_NULL(tensor_);
  }

  void ConvertValue(const ValuePtr &value, const AttrDeclType<int32_t> &) {
    MS_EXCEPTION_IF_NULL(value);
    std::vector<int64_t> vec;
    auto ori_vec = ops::GetArrayValue<int32_t>(value).value().ToVector();
    (void)std::transform(ori_vec.begin(), ori_vec.end(), std::back_inserter(vec),
                         [](const auto &v) { return static_cast<int64_t>(v); });
    tensor_ = std::make_shared<tensor::Tensor>(vec);
    MS_EXCEPTION_IF_NULL(tensor_);
  }

  void ConvertValue(const ValuePtr &value, const AttrDeclType<float> &) {
    MS_EXCEPTION_IF_NULL(value);
    std::vector<double> vec;
    auto ori_vec = ops::GetArrayValue<float>(value).value().ToVector();
    (void)std::transform(ori_vec.begin(), ori_vec.end(), std::back_inserter(vec),
                         [](const auto &v) { return static_cast<double>(v); });
    tensor_ = std::make_shared<tensor::Tensor>(vec);
    MS_EXCEPTION_IF_NULL(tensor_);
  }

  void ConvertValue(const ValuePtr &value, const AttrDeclType<double> &) {
    MS_EXCEPTION_IF_NULL(value);
    std::vector<float> vec;
    auto ori_vec = ops::GetArrayValue<double>(value).value().ToVector();
    (void)std::transform(ori_vec.begin(), ori_vec.end(), std::back_inserter(vec),
                         [](const auto &v) { return static_cast<float>(v); });
    tensor_ = std::make_shared<tensor::Tensor>(vec);
    MS_EXCEPTION_IF_NULL(tensor_);
  }

  tensor::TensorPtr tensor_;
};

class AttrToInputConverter : public AttrHelper<AttrToInputConverter> {
 public:
  const tensor::TensorPtr &GetTensor() const { return tensor_; }

  template <typename T>
  void ConvertValue(const ValuePtr &value, const AttrDeclType<T> &, TensorParams *) {
    tensor_ = std::make_shared<tensor::Tensor>(GetValue<T>(value));
    MS_EXCEPTION_IF_NULL(tensor_);
  }

  void ConvertValue(const ValuePtr &value, const AttrDeclType<std::string> &, TensorParams *) {
    MS_LOG(EXCEPTION) << "Unsupported convert from string to input.";
  }

  void ConvertValue(const ValuePtr &value, const AttrDeclType<::ge::DataType> &, TensorParams *) {
    MS_LOG(EXCEPTION) << "Unsupported convert from ::ge::DataType to input.";
  }

  void ConvertValue(const ValuePtr &value, const AttrDeclType<int64_t> &, TensorParams *param) {
    MS_EXCEPTION_IF_NULL(param);
    auto real_val = static_cast<int32_t>(GetValue<int64_t>(value));
    tensor_ = std::make_shared<tensor::Tensor>(real_val);
    param->data_type = TypeId::kNumberTypeInt32;
    MS_EXCEPTION_IF_NULL(tensor_);
  }

  template <typename T>
  void ConvertValue(const ValuePtr &value, const AttrDeclType<std::vector<T>> &, TensorParams *) {
    std::vector<T> array_list;
    ConvertValueSequenceToList(value, &array_list);
    tensor_ = std::make_shared<tensor::Tensor>(array_list);
    MS_EXCEPTION_IF_NULL(tensor_);
  }

  void ConvertValue(const ValuePtr &value, const AttrDeclType<std::vector<::ge::DataType>> &, TensorParams *) {
    MS_LOG(EXCEPTION) << "Unsupported convert from vector<::ge::DataType> to input.";
  }

  void ConvertValue(const ValuePtr &value, const AttrDeclType<std::vector<std::string>> &, TensorParams *) {
    MS_LOG(EXCEPTION) << "Unsupported convert from list_string to input.";
  }

  void ConvertValue(const ValuePtr &value, const AttrDeclType<std::vector<float>> &, TensorParams *) {
    MS_LOG(EXCEPTION) << "Unsupported convert from list_float to input.";
  }

  void ConvertValue(const ValuePtr &value, const AttrDeclType<std::vector<uint8_t>> &, TensorParams *param) {
    MS_EXCEPTION_IF_NULL(param);
    std::vector<uint8_t> array_list;
    ConvertValueSequenceToList(value, &array_list);
    std::vector<int32_t> array_list_int32;
    (void)std::transform(array_list.begin(), array_list.end(), std::back_inserter(array_list_int32),
                         [](const uint8_t val) { return static_cast<int32_t>(val); });
    tensor_ = std::make_shared<tensor::Tensor>(array_list_int32);
    param->data_type = TypeId::kNumberTypeInt32;
    MS_EXCEPTION_IF_NULL(tensor_);
  }

  void ConvertValue(const ValuePtr &value, const AttrDeclType<std::vector<int64_t>> &, TensorParams *param) {
    MS_EXCEPTION_IF_NULL(param);
    std::vector<int64_t> array_list;
    ConvertValueSequenceToList(value, &array_list);
    std::vector<int32_t> array_list_int32;
    (void)std::transform(array_list.begin(), array_list.end(), std::back_inserter(array_list_int32),
                         [](const int64_t val) { return LongToInt(val); });
    tensor_ = std::make_shared<tensor::Tensor>(array_list_int32);
    param->data_type = TypeId::kNumberTypeInt32;
    MS_EXCEPTION_IF_NULL(tensor_);
  }

  template <typename T>
  void ConvertValue(const ValuePtr &value, const AttrDeclType<std::vector<std::vector<T>>> &, const ShapeVector &shape,
                    TensorParams *) {
    std::vector<T> array_list;
    ConvertValueSequenceToList(value, &array_list);
    tensor_ = std::make_shared<tensor::Tensor>(kNumberTypeInt64, shape);
    MS_EXCEPTION_IF_NULL(tensor_);
    auto data_ptr = tensor_->data_c();
    MS_EXCEPTION_IF_NULL(data_ptr);
    auto size = static_cast<size_t>(tensor_->data().nbytes());
    if (memcpy_s(data_ptr, size, array_list.data(), size) != EOK) {
      MS_LOG(EXCEPTION) << "memcpy of listlistint failed in convert acl attr.";
    }
  }

  tensor::TensorPtr tensor_;
};
}  // namespace transform
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_TRANSFORM_ACL_IR_ACL_CONVERT_H_
