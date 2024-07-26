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
  // for dynamic input, it is offset of the first element of dynamic input in ge inputs; for normal input, it is the
  // offset of the input in ge inputs
  size_t ge_offset;
};

struct MsInputIdxToGe {
  size_t ge_adapter_idx;
  std::vector<size_t> ms_real_idx;
  std::vector<size_t> ge_real_idx;
};

class CompareGeIdx {
 public:
  bool operator()(const std::pair<size_t, size_t> &a, const std::pair<size_t, size_t> &b) const {
    return a.second == b.second ? a.first < b.first : a.second < b.second;
  }
};

class AclConverter {
 public:
  void ConvertToAclOpType(const std::string &prim_name);
  void ResizeAclOpInputs(const PrimitivePtr &prim, const std::vector<KernelTensor *> &inputs);
  void ConvertInputMsIndexToAclIndex(const PrimitivePtr &prim, const std::vector<KernelTensor *> &inputs);
  void ConvertToAclInput(const PrimitivePtr &prim, const std::vector<KernelTensor *> &inputs,
                         const std::vector<TensorParams> &input_params);
  void ConvertToAclOutput(const std::string &kernel_name, const std::vector<KernelTensor *> &outputs,
                          const std::vector<TensorParams> &output_params);
  bool IsNeedSkipExecute(const std::string &kernel_name, const std::vector<KernelTensor *> &inputs,
                         const std::vector<KernelTensor *> &outputs, void *stream_ptr);

  void ConvertValueDependToHostInput(const std::string &kernel_name, const std::vector<KernelTensor *> &inputs,
                                     const std::vector<TensorParams> &input_params,
                                     const std::set<int64_t> &value_depend_args);

  // NOTE: Attribute kAttrDynInputSizes is a vector<int64_t> with its element value -1 for dynamic input, and
  // number of foled real inputs for dynamic input. For example a op with 5 inputs, of whicch the input 1 and 2
  // are dynamic inputs, the attribute `dyn_input_size` of it may be the value as below: ms_proto_index : |  0 | 1
  // | 2 |  3 |  4 | dyn_input_sizes: | -1 |  2 |  5 | -1 | -1 |
  void ConvertMsIdxToGeIdx(const PrimitivePtr &prim, const std::vector<KernelTensor *> &inputs);

  void ConvertAttrToAclInput(const mindspore::HashMap<std::string, ValuePtr> &attrs, const std::string &kernel_name,
                             std::vector<TensorParams> *input_params);
  void ConvertInputToAclAttr(const std::vector<KernelTensor *> &inputs, const std::string &kernel_name);
  void ConvertToAclAttr(const mindspore::HashMap<std::string, ValuePtr> &attrs, const std::string &prim_name,
                        std::vector<std::string> *ms_attr_str);
  void ProcessRunnerSpecialInfo(const std::string &prim_name, const std::vector<TensorParams> &output_params,
                                bool is_dynamic);
  void SetRunnerSpecialInfo();

  bool is_need_retrieve_output_shape() const { return is_need_retrieve_output_shape_; }

  std::string DebugString() const;

  void Run(void *stream_ptr);

  AclRunner &runner() { return runner_; }

  std::vector<std::vector<int64_t>> SyncData() { return runner_.SyncData(); }

  void Reset();

  static aclDataType ConvertType(TypeId type);
  static aclFormat ConvertFormat(const std::string &format);
  std::string GetFormatFromInputAttrMap(const std::vector<KernelTensor *> &inputs, const std::string &kernel_name);

  std::pair<aclTensorDesc *, aclDataBuffer *> ConvertTensorToAclDesc(const AddressPtr &address,
                                                                     const TensorParams &params,
                                                                     const std::string &desc_name,
                                                                     AclDumpString *dump_str, bool is_input) const;
  std::pair<aclTensorDesc *, aclDataBuffer *> ConvertTensorToAclDesc(const KernelTensor *ori_tensor,
                                                                     const TensorParams &params,
                                                                     const std::string &desc_name,
                                                                     AclDumpString *dump_str, bool is_input) const;
  std::pair<aclTensorDesc *, aclDataBuffer *> ConvertTensorToAclDesc(const AclHostInfoPtr &address,
                                                                     const TensorParams &params,
                                                                     const std::string &desc_name,
                                                                     AclDumpString *dump_str, bool is_input) const;
  std::pair<aclTensorDesc *, aclDataBuffer *> ConvertTensorToAclDesc(const tensor::TensorPtr &host_tensor,
                                                                     const TensorParams &params,
                                                                     const std::string &desc_name,
                                                                     AclDumpString *dump_str) const;

 private:
  friend class AttrConverter;

  void GenerateRealGeIdx();

  template <typename T>
  void AclRunnerAddAttr(const std::string &attrName, T value);

  // convert acl inputs for operator with at most one dynamic parameter, general case
  void ConvertInputsNormal(const PrimitivePtr &prim, const std::vector<KernelTensor *> &inputs,
                           const GeAdapterInfoPtr &info);

  // convert acl inputs for operator with more than one dynamic parameters, rare case
  void ConvertInputsMutiDynParams(const PrimitivePtr &prim, const std::vector<KernelTensor *> &inputs,
                                  const GeAdapterInfoPtr &info);

  AclRunner runner_;
  AclInputToHost input_on_host_;
  std::vector<std::vector<uint8_t>> host_save_list_;

  std::vector<AclDumpString> input_str_;
  std::vector<AclDumpString> output_str_;
  std::map<std::string, std::string> attr_map_str_;

  bool is_create_mapping_ = false;

  bool is_dynamic_ = false;
  AclPrecisionMode precision_mode_ = DEFAULT_MODE;
  bool is_need_retrieve_output_shape_ = false;

  // Fields for op containing multiple dynamic inputs, since operators with more than one dynamic inputs are rare, for
  // speed reason, we process this case separately.
  // Map for recording [MindSpore op input proto index of dynamic input] to [its number of folded inputs]
  // NOTE: here the map MUST be an ordered map to sort the input indices ascendly.
  std::map<size_t, size_t> dyn_inputs_map_;
  std::map<size_t, MsInputInfo> inputs_idx_convert_map_;
  // number of folded inputs of dynamic input, only used for op with only one dynamic input
  std::pair<size_t, size_t> num_folded_inputs_{SIZE_MAX, 1};

  std::map<std::pair<size_t, size_t>, std::pair<std::vector<size_t>, std::vector<size_t>>, CompareGeIdx>
    ms_and_ge_inputs_sort_info_;
  std::map<size_t, MsInputIdxToGe> ms_and_ge_inputs_idx_info_;
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
  void ConvertValueSequenceToList(const ValuePtr &value, std::vector<T> *array_list,
                                  ShapeVector *shape = nullptr) const {
    MS_EXCEPTION_IF_NULL(array_list);
    const auto &value_sequence = value->cast<ValueSequencePtr>()->value();
    if (value_sequence.size() == 0) {
      return;
    }
    auto val = value_sequence[0];
    if (val->isa<Scalar>()) {
      auto list_value = GetValue<std::vector<T>>(value);
      array_list->insert(array_list->end(), list_value.begin(), list_value.end());
      if (shape != nullptr) {
        (void)shape->emplace_back(list_value.size());
      }
    }
    if (val->isa<ValueSequence>()) {
      for (size_t i = 0; i < value_sequence.size(); i++) {
        ConvertValueSequenceToList(value_sequence[i], array_list, shape);
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
    ShapeVector offset_shape;
    ConvertValueSequenceToList(value, &array_list, &offset_shape);
    std::vector<std::vector<int64_t>> array_list_int64(shape[0]);
    size_t offset = 0;
    for (int64_t i = 0; i < shape[0]; ++i) {
      array_list_int64[i].assign(array_list.begin() + offset, array_list.begin() + offset + offset_shape[i]);
      offset += offset_shape[i];
    }
    MS_EXCEPTION_IF_NULL(acl_converter);
    acl_converter->AclRunnerAddAttr(attr_name_, array_list_int64);
  }
};

class ValueDependToInputConverter : public AttrHelper<ValueDependToInputConverter> {
 public:
  const std::vector<uint8_t> GetData() const { return data_; }
  const std::map<TypeId, TypeId> &GetValueDependCastMap() {
    static const std::map<TypeId, TypeId> kValueDependCastMap = {{kNumberTypeInt32, kNumberTypeInt64},
                                                                 {kNumberTypeInt64, kNumberTypeInt32},
                                                                 {kNumberTypeFloat32, kNumberTypeFloat64},
                                                                 {kNumberTypeFloat64, kNumberTypeFloat32}};
    return kValueDependCastMap;
  }

  void ConvertValue(const ValuePtr &value, const AttrDeclType<int64_t> &) {
    MS_EXCEPTION_IF_NULL(value);
    auto ori_vec = ops::GetArrayValue<int64_t>(value).value().ToVector();
    data_.resize(ori_vec.size() * sizeof(int32_t));
    int32_t *data_ptr = reinterpret_cast<int32_t *>(data_.data());
    MS_EXCEPTION_IF_NULL(data_ptr);
    for (size_t i = 0; i < ori_vec.size(); ++i) {
      data_ptr[i] = static_cast<int32_t>(ori_vec[i]);
    }
  }

  void ConvertValue(const ValuePtr &value, const AttrDeclType<int32_t> &) {
    MS_EXCEPTION_IF_NULL(value);
    auto ori_vec = ops::GetArrayValue<int32_t>(value).value().ToVector();
    data_.resize(ori_vec.size() * sizeof(int64_t));
    int64_t *data_ptr = reinterpret_cast<int64_t *>(data_.data());
    MS_EXCEPTION_IF_NULL(data_ptr);
    for (size_t i = 0; i < ori_vec.size(); ++i) {
      data_ptr[i] = static_cast<int64_t>(ori_vec[i]);
    }
  }

  void ConvertValue(const ValuePtr &value, const AttrDeclType<float> &) {
    MS_EXCEPTION_IF_NULL(value);
    auto ori_vec = ops::GetArrayValue<float>(value).value().ToVector();
    data_.resize(ori_vec.size() * sizeof(double));
    double *data_ptr = reinterpret_cast<double *>(data_.data());
    MS_EXCEPTION_IF_NULL(data_ptr);
    for (size_t i = 0; i < ori_vec.size(); ++i) {
      data_ptr[i] = static_cast<double>(ori_vec[i]);
    }
  }

  void ConvertValue(const ValuePtr &value, const AttrDeclType<double> &) {
    MS_EXCEPTION_IF_NULL(value);
    auto ori_vec = ops::GetArrayValue<double>(value).value().ToVector();
    data_.resize(ori_vec.size() * sizeof(float));
    float *data_ptr = reinterpret_cast<float *>(data_.data());
    MS_EXCEPTION_IF_NULL(data_ptr);
    for (size_t i = 0; i < ori_vec.size(); ++i) {
      data_ptr[i] = static_cast<float>(ori_vec[i]);
    }
  }

  std::vector<uint8_t> data_;
};

class AttrToInputConverter : public AttrHelper<AttrToInputConverter> {
 public:
  const std::vector<uint8_t> GetData() const { return data_; }

  template <typename T>
  void ConvertValue(const ValuePtr &value, const AttrDeclType<T> &, TensorParams *) {
    auto tensor = std::make_shared<tensor::Tensor>(GetValue<T>(value));
    auto tensor_data_ptr = tensor->data_c();
    auto size = static_cast<size_t>(tensor->data().nbytes());
    data_.resize(size);
    if (memcpy_s(data_.data(), size, tensor_data_ptr, size) != EOK) {
      MS_LOG(EXCEPTION) << "memcpy of listlistint failed in attr convert to acl input.";
    }
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
    data_.resize(sizeof(int32_t));
    int32_t *data_ptr = reinterpret_cast<int32_t *>(data_.data());
    MS_EXCEPTION_IF_NULL(data_ptr);
    data_ptr[0] = static_cast<int32_t>(real_val);
    param->data_type = TypeId::kNumberTypeInt32;
  }

  template <typename T>
  void ConvertValue(const ValuePtr &value, const AttrDeclType<std::vector<T>> &, TensorParams *) {
    std::vector<T> array_list;
    ConvertValueSequenceToList(value, &array_list);
    data_.resize(array_list.size() * sizeof(T));
    T *data_ptr = reinterpret_cast<T *>(data_.data());
    MS_EXCEPTION_IF_NULL(data_ptr);
    for (size_t i = 0; i < array_list.size(); ++i) {
      data_ptr[i] = static_cast<T>(array_list[i]);
    }
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
    data_.resize(array_list.size() * sizeof(int32_t));
    int32_t *data_ptr = reinterpret_cast<int32_t *>(data_.data());
    MS_EXCEPTION_IF_NULL(data_ptr);
    for (size_t i = 0; i < array_list.size(); ++i) {
      data_ptr[i] = static_cast<int32_t>(array_list[i]);
    }
    param->data_type = TypeId::kNumberTypeInt32;
  }

  void ConvertValue(const ValuePtr &value, const AttrDeclType<std::vector<int64_t>> &, TensorParams *param) {
    MS_EXCEPTION_IF_NULL(param);
    std::vector<int64_t> array_list;
    ConvertValueSequenceToList(value, &array_list);
    data_.resize(array_list.size() * sizeof(int32_t));
    int32_t *data_ptr = reinterpret_cast<int32_t *>(data_.data());
    MS_EXCEPTION_IF_NULL(data_ptr);
    for (size_t i = 0; i < array_list.size(); ++i) {
      data_ptr[i] = static_cast<int32_t>(array_list[i]);
    }
    param->data_type = TypeId::kNumberTypeInt32;
  }

  template <typename T>
  void ConvertValue(const ValuePtr &value, const AttrDeclType<std::vector<std::vector<T>>> &, const ShapeVector &shape,
                    TensorParams *) {
    std::vector<T> array_list;
    ConvertValueSequenceToList(value, &array_list);
    data_.resize(array_list.size() * sizeof(T));
    T *data_ptr = reinterpret_cast<T *>(data_.data());
    MS_EXCEPTION_IF_NULL(data_ptr);
    for (size_t i = 0; i < array_list.size(); ++i) {
      data_ptr[i] = static_cast<T>(array_list[i]);
    }
  }

  std::vector<uint8_t> data_;
};
}  // namespace transform
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_TRANSFORM_ACL_IR_ACL_CONVERT_H_
