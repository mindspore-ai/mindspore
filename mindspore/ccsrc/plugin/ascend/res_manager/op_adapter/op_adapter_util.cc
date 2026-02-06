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

#include "plugin/ascend/res_manager/op_adapter/op_adapter_util.h"

#include <string>
#include <vector>
#include <algorithm>

#include "include/utils/utils.h"
#include "utils/check_convert_utils.h"
#include "plugin/ascend/res_manager/op_adapter/op_adapter_base.h"
#include "plugin/ascend/res_manager/op_adapter/transform_util.h"
#include "ir/kernel_tensor_value.h"
#include "ops_utils/op_utils.h"
#include "plugin/ascend/res_manager/op_adapter/op_adapter.h"
#include "plugin/ascend/res_manager/op_adapter/op_adapter_map.h"
#include "primitive/lite_ops.h"
#include "primitive/framework_ops.h"
#include "primitive/auto_generate/gen_ops_primitive_p.h"
#include "primitive/auto_generate/gen_ops_primitive_s.h"

namespace mindspore {
namespace {
constexpr size_t kPartialCNodeValue = 1;
}  // namespace

GeDataTypeImm::GeDataTypeImm() : IntegerImm(kInt32), v_(::ge::DataType::DT_FLOAT) {}
GeDataTypeImm::GeDataTypeImm(::ge::DataType v) : IntegerImm(kInt32), v_(v) {
  hash_ = hash_combine({tid(), std::hash<int>{}(v_)});
}
bool GeDataTypeImm::operator==(const Value &other) const {
  if (other.isa<GeDataTypeImm>()) {
    auto &other_ = static_cast<const GeDataTypeImm &>(other);
    return *this == other_;
  } else {
    return false;
  }
}
bool GeDataTypeImm::operator==(const GeDataTypeImm &other) const { return v_ == other.v_; }
std::string GeDataTypeImm::DumpText() const {
  std::ostringstream oss;
  oss << "GeDataType(" << int(v_) << ")";
  return oss.str();
}

namespace device::ascend {
inline GeTensor ConvertAnyUtilTensor(const ValuePtr &value, const AnyTraits<mindspore::tensor::Tensor> &, bool is_ref) {
  // To-DO the format may read from ME tensor
  MS_EXCEPTION_IF_NULL(value);
  auto me_tensor = value->cast<MeTensorPtr>();
  auto ge_tensor = TransformUtil::ConvertTensor(me_tensor, kOpFormat_ND, !is_ref);
  return ge_tensor == nullptr ? GeTensor() : *ge_tensor;
}
GeTensor ConvertAnyUtilWithRef(const ValuePtr &value, const AnyTraits<mindspore::tensor::Tensor> &traits) {
  return ConvertAnyUtilTensor(value, traits, true);
}

GeTensor ConvertAnyUtil(const ValuePtr &value, const AnyTraits<mindspore::tensor::Tensor> &traits) {
  return ConvertAnyUtilTensor(value, traits, false);
}

std::vector<int64_t> ConvertAnyUtil(const ValuePtr &value, const std::string &name,
                                    const AnyTraits<std::vector<int64_t>>) {
  MS_EXCEPTION_IF_NULL(value);
  std::vector<int64_t> list;
  if (name == "pad") {
    if (!value->isa<ValueSequence>()) {
      MS_LOG(EXCEPTION) << "Value should be ValueTuple, but got" << value->type_name();
    }
    auto vec = value->cast<ValueSequencePtr>();
    list.resize(vec->value().size() + 2);
    list[0] = 1;
    list[1] = 1;
    (void)std::transform(vec->value().begin(), vec->value().end(), list.begin() + 2,
                         [](const ValuePtr &val) { return GetValueWithCheck<int64_t>(val); });
  } else {
    int64_t data = GetValueWithCheck<int64_t>(value);
    int size = 2;  // 2 int in list
    list = TransformUtil::ConvertIntToList(data, size);
  }

  return list;
}

std::string ConvertAnyUtil(const ValuePtr &value, const AnyTraits<std::vector<int64_t>>, const AnyTraits<std::string>) {
  MS_EXCEPTION_IF_NULL(value);
  auto vec = value->cast<ValueTuplePtr>();
  if (vec == nullptr) {
    MS_LOG(EXCEPTION) << "not ValueTuplePtr";
  }
  std::ostringstream buffer;
  int i = 0;
  for (auto &it : vec->value()) {
    if (i != 0) {
      buffer << ",";
    }
    buffer << GetValueWithCheck<int64_t>(it);
    i++;
  }
  return buffer.str();
}

std::vector<float> ConvertAnyUtil(const ValuePtr &value, const AnyTraits<std::vector<float>>, const AnyTraits<float>) {
  MS_EXCEPTION_IF_NULL(value);
  auto vec = value->cast<ValueTuplePtr>();
  if (vec == nullptr) {
    MS_LOG(EXCEPTION) << "not ValueTuplePtr";
  }
  std::vector<float> list;
  list.resize(vec->value().size());
  (void)std::transform(vec->value().begin(), vec->value().end(), list.begin(),
                       [](const ValuePtr &val) { return GetValueWithCheck<float>(val); });
  return list;
}

std::vector<int64_t> ConvertAnyUtil(const ValuePtr &value, const std::string &format,
                                    const AnyTraits<std::vector<int64_t>>, const AnyTraits<int64_t>) {
  MS_EXCEPTION_IF_NULL(value);
  auto vec = value->cast<ValueTuplePtr>();
  if (vec == nullptr) {
    MS_LOG(EXCEPTION) << "not ValueTuplePtr";
  }
  std::vector<int64_t> list;
  list.resize(vec->value().size());
  (void)std::transform(vec->value().begin(), vec->value().end(), list.begin(),
                       [](const ValuePtr &val) { return GetValueWithCheck<int64_t>(val); });
  if (format == kOpFormat_NHWC) {
    if (list.size() < 4) {
      MS_LOG(EXCEPTION) << "The size of list is less than 4";
    } else {
      int64_t temp = list[1];
      list[1] = list[2];
      list[2] = list[3];
      list[3] = temp;
    }
  }
  return list;
}

GeDataType ConvertAnyUtil(const ValuePtr &value, const AnyTraits<GEType>) {
  MS_EXCEPTION_IF_NULL(value);
  TypeId me_type;
  if (value->isa<Type>()) {
    auto type = value->cast<TypePtr>();
    MS_EXCEPTION_IF_NULL(type);
    me_type = type->type_id();
    if (kObjectTypeTensorType == me_type) {
      me_type = dyn_cast<TensorType>(type)->element()->type_id();
    }
  } else if (value->isa<Int32Imm>()) {
    // type id
    me_type = static_cast<TypeId>(GetValue<int32_t>(value));
  } else if (value->isa<UInt64Imm>()) {
    // type id
    me_type = static_cast<TypeId>(GetValue<uint64_t>(value));
  } else if (value->isa<Int64Imm>()) {
    // type id
    me_type = static_cast<TypeId>(GetValue<int64_t>(value));
  } else if (value->isa<KernelTensorValue>()) {
    // type id
    auto value_opt = GetScalarValue<int64_t>(value);
    me_type = static_cast<TypeId>(value_opt.value());
  } else {
    MS_LOG(EXCEPTION) << "error convert Value to TypePtr for value: " << value->ToString()
                      << ", type: " << value->type_name() << ", value should be a Typeptr or TypeId";
  }
  return TransformUtil::ConvertDataType(me_type);
}

std::vector<GeDataType> ConvertAnyUtil(const ValuePtr &value, const AnyTraits<std::vector<GEType>>) {
  MS_EXCEPTION_IF_NULL(value);
  std::vector<GeDataType> data;
  if (!value->isa<ValueTuple>() && !value->isa<ValueList>()) {
    MS_LOG(WARNING) << "error convert Value to vector for value: " << value->ToString()
                    << ", type: " << value->type_name() << ", value should be a tuple or list";
    data.emplace_back(ConvertAnyUtil(value, AnyTraits<GEType>()));
    return data;
  }
  auto vec = value->isa<ValueTuple>() ? value->cast<ValueTuplePtr>()->value() : value->cast<ValueListPtr>()->value();
  std::transform(vec.begin(), vec.end(), std::back_inserter(data),
                 [](const ValuePtr &it) { return ConvertAnyUtil(it, AnyTraits<GEType>()); });
  return data;
}

std::string ConvertAnyUtil(const ValuePtr &value, const AnyTraits<GEDataFormat>) {
  MS_EXCEPTION_IF_NULL(value);
  if (value->isa<StringImm>()) {
    return GetValue<std::string>(value);
  }
  int64_t format_id = GetCastIntegralValue<int64_t>(value);
  return GEDataFormat::ConvertEnumToString(format_id);
}

std::string ConvertAnyUtil(const ValuePtr &value, const AnyTraits<GEPadMod>) {
  MS_EXCEPTION_IF_NULL(value);
  if (value->isa<StringImm>()) {
    return GetValue<std::string>(value);
  }
  int64_t pad_id = GetCastIntegralValue<int64_t>(value);
  return GEPadMod::ConvertEnumToString(pad_id);
}

std::string ConvertAnyUtil(const ValuePtr &value, const AnyTraits<GEReduction>) {
  MS_EXCEPTION_IF_NULL(value);
  if (value->isa<StringImm>()) {
    return GetValue<std::string>(value);
  }
  int64_t reduction_id = GetCastIntegralValue<int64_t>(value);
  return GEReduction::ConvertEnumToString(reduction_id);
}

std::string ConvertAnyUtil(const ValuePtr &value, const AnyTraits<AscendQuantRoundMode>) {
  MS_EXCEPTION_IF_NULL(value);
  if (value->isa<StringImm>()) {
    return GetValue<std::string>(value);
  }
  int64_t round_mode_id = GetCastIntegralValue<int64_t>(value);
  return AscendQuantRoundMode::ConvertEnumToString(round_mode_id);
}

std::string ConvertAnyUtil(const ValuePtr &value, const AnyTraits<FASInputLayoutMode>) {
  MS_EXCEPTION_IF_NULL(value);
  if (value->isa<StringImm>()) {
    return GetValue<std::string>(value);
  }
  int64_t input_layout_id = GetCastIntegralValue<int64_t>(value);
  return FASInputLayoutMode::ConvertEnumToString(input_layout_id);
}

std::string ConvertAnyUtil(const ValuePtr &value, const AnyTraits<FFNActivationMode>) {
  MS_EXCEPTION_IF_NULL(value);
  if (value->isa<StringImm>()) {
    return GetValue<std::string>(value);
  }
  int64_t activation_id = GetCastIntegralValue<int64_t>(value);
  return FFNActivationMode::ConvertEnumToString(activation_id);
}

std::string ConvertAnyUtil(const ValuePtr &value, const AnyTraits<ScatterReduceMode>) {
  MS_EXCEPTION_IF_NULL(value);
  if (value->isa<StringImm>()) {
    return GetValue<std::string>(value);
  }
  int64_t reduce_id = GetCastIntegralValue<int64_t>(value);
  return ScatterReduceMode::ConvertEnumToString(reduce_id);
}

std::string ConvertAnyUtil(const ValuePtr &value, const AnyTraits<GECoordinateTransformMode>) {
  MS_EXCEPTION_IF_NULL(value);
  if (value->isa<StringImm>()) {
    return GetValue<std::string>(value);
  }
  int64_t mode_id = GetCastIntegralValue<int64_t>(value);
  return GECoordinateTransformMode::ConvertEnumToString(mode_id);
}

std::string ConvertAnyUtil(const ValuePtr &value, const AnyTraits<GERotatedIouMode>) {
  MS_EXCEPTION_IF_NULL(value);
  int64_t mode_id = GetCastIntegralValue<int64_t>(value);
  return GERotatedIouMode::ConvertEnumToString(mode_id);
}

std::string ConvertAnyUtil(const ValuePtr &value, const AnyTraits<GEEnumToStr>,
                           const std::vector<std::string> &enum_string) {
  MS_EXCEPTION_IF_NULL(value);

  if (value->isa<StringImm>()) {
    return GetValue<std::string>(value);
  }
  int64_t id = GetCastIntegralValue<int64_t>(value);
  if (id < 0 || id >= static_cast<int64_t>(enum_string.size())) {
    MS_LOG(EXCEPTION) << "Invalid enum id " << id;
    return "";
  }
  return enum_string[id];
}

template <typename T1, typename T2>
GeTensor NestedVectorToTensorImpl(const ValuePtrList &vec, const TypeId &type) {
  const auto &vec_item =
    vec[0]->isa<ValueTuple>() ? vec[0]->cast<ValueTuplePtr>()->value() : vec[0]->cast<ValueListPtr>()->value();
  size_t attr_size1 = vec.size();
  size_t attr_size2 = vec_item.size();
  std::vector<T1> attr_list;
  for (const auto &item : vec) {
    auto value_list = GetValueWithCheck<std::vector<T1>>(item);
    (void)std::copy(value_list.begin(), value_list.end(), std::back_inserter(attr_list));
  }
  auto attr_value = MakeValue(attr_list);
  auto data = ConvertAnyUtil(attr_value, AnyTraits<T1>(), AnyTraits<std::vector<T2>>());
  auto desc =
    TransformUtil::GetGeTensorDesc({static_cast<int>(attr_size1), static_cast<int>(attr_size2)}, type, kOpFormat_NCHW);
  if (desc == nullptr) {
    MS_LOG(EXCEPTION) << "Update conversion descriptor failed!";
  }
  return GeTensor(*desc, reinterpret_cast<uint8_t *>(data.data()), data.size() * sizeof(T2));
}

GeTensor NestedVectorToTensor(const ValuePtr &value) {
  MS_EXCEPTION_IF_NULL(value);
  const auto &vec =
    value->isa<ValueTuple>() ? value->cast<ValueTuplePtr>()->value() : value->cast<ValueListPtr>()->value();
  const auto &vec_item =
    vec[0]->isa<ValueTuple>() ? vec[0]->cast<ValueTuplePtr>()->value() : vec[0]->cast<ValueListPtr>()->value();
  if (vec_item.empty()) {
    MS_LOG(WARNING) << "Convert a none nested tuple to an empty ge tensor";
    return GeTensor(GeTensorDesc(::ge::Shape({0})));
  }
  MS_EXCEPTION_IF_NULL(vec_item[0]);
  TypeId type;
  if (vec_item[0]->isa<Int32Imm>()) {
    type = kNumberTypeInt32;
    return NestedVectorToTensorImpl<int32_t, int32_t>(vec, type);
  } else if (vec_item[0]->isa<Int64Imm>()) {
    type = kNumberTypeInt64;
    return NestedVectorToTensorImpl<int64_t, int64_t>(vec, type);
  } else if (vec_item[0]->isa<FP32Imm>()) {
    type = kNumberTypeFloat32;
    return NestedVectorToTensorImpl<float, float>(vec, type);
  } else if (vec_item[0]->isa<BoolImm>()) {
    type = kNumberTypeBool;
    return NestedVectorToTensorImpl<bool, uint8_t>(vec, type);
  } else {
    MS_LOG(EXCEPTION) << "Unsupported data type of nested tuple or list elements: " << vec_item[0]->type_name();
  }
}

template <typename T1, typename T2>
GeTensor VectorToTensorImpl(const ValuePtr &value, const TypeId &type) {
  const auto &vec =
    value->isa<ValueTuple>() ? value->cast<ValueTuplePtr>()->value() : value->cast<ValueListPtr>()->value();
  auto data = ConvertAnyUtil(value, AnyTraits<T1>(), AnyTraits<std::vector<T2>>());
  auto format = vec.size() == kDim4 ? kOpFormat_NCHW : kOpFormat_ND;
  auto desc = TransformUtil::GetGeTensorDesc({static_cast<int>(vec.size())}, type, format);
  if (desc == nullptr) {
    MS_LOG(EXCEPTION) << "Update conversion descriptor failed!";
  }
  return GeTensor(*desc, reinterpret_cast<uint8_t *>(data.data()), data.size() * sizeof(T2));
}

GeTensor VectorToTensorUtil(const ValuePtr &value) {
  MS_EXCEPTION_IF_NULL(value);
  auto vec = value->isa<ValueTuple>() ? value->cast<ValueTuplePtr>()->value() : value->cast<ValueListPtr>()->value();
  if (vec.empty()) {
    MS_LOG(INFO) << "Convert a none tuple to an empty ge tensor";
    return GeTensor(GeTensorDesc(::ge::Shape({0}), ::ge::FORMAT_ND, ::ge::DT_INT64));
  }
  MS_EXCEPTION_IF_NULL(vec[0]);
  TypeId type;
  if (vec[0]->isa<Int32Imm>()) {
    MS_LOG(INFO) << "convert value to tensor with data type = Int32";
    type = kNumberTypeInt32;
    return VectorToTensorImpl<int32_t, int32_t>(value, type);
  } else if (vec[0]->isa<Int64Imm>()) {
    MS_LOG(INFO) << "convert value to tensor with data type = Int64";
    type = kNumberTypeInt64;
    return VectorToTensorImpl<int64_t, int64_t>(value, type);
  } else if (vec[0]->isa<FP32Imm>()) {
    MS_LOG(INFO) << "convert value to tensor with data type = Float32";
    type = kNumberTypeFloat32;
    return VectorToTensorImpl<float, float>(value, type);
  } else if (vec[0]->isa<BoolImm>()) {
    MS_LOG(INFO) << "convert value to tensor with data type = Bool";
    type = kNumberTypeBool;
    return VectorToTensorImpl<bool, uint8_t>(value, type);
  } else if (vec[0]->isa<ValueTuple>() || vec[0]->isa<ValueList>()) {
    // convert nested tuple or list to ge tensor, supported two dims
    MS_LOG(INFO) << "Convert nested tuple or list to ge tensor.";
    return NestedVectorToTensor(value);
  } else {
    MS_LOG(EXCEPTION) << "Unsupported data type of tuple or list elements: " << vec[0]->type_name();
  }
}

GeTensor ConvertAnyUtil(const ValuePtr &value, const AnyTraits<ValueAny>) {
  MS_EXCEPTION_IF_NULL(value);
  if (value->isa<MeTensor>()) {
    // convert me tensor to ge tensor
    return ConvertAnyUtil(value, AnyTraits<MeTensor>());
  } else if (value->isa<ValueList>() || value->isa<ValueTuple>()) {
    return VectorToTensorUtil(value);
  } else if (value->isa<Int32Imm>()) {
    // convert scalar Int to GeTensor
    MS_LOG(DEBUG) << "convert scalar to tensor with data type = Int32";
    GeTensorDesc desc(GeShape(), ::ge::FORMAT_ND, ::ge::DT_INT32);
    auto v = GetValue<int32_t>(value);
    desc.SetRealDimCnt(0);
    return GeTensor(desc, reinterpret_cast<uint8_t *>(&v), sizeof(int32_t));
  } else if (value->isa<UInt32Imm>()) {
    // convert scalar UInt to GeTensor
    MS_LOG(DEBUG) << "Convert scalar to tensor with data type = UInt32";
    GeTensorDesc desc(GeShape(), ::ge::FORMAT_ND, ::ge::DT_UINT32);
    auto v = GetValue<uint32_t>(value);
    desc.SetRealDimCnt(0);
    return GeTensor(desc, reinterpret_cast<uint8_t *>(&v), sizeof(uint32_t));
  } else if (value->isa<Int64Imm>()) {
    // convert scalar Int64 to GeTensor
    MS_LOG(DEBUG) << "convert scalar to tensor with data type = Int64";
    GeTensorDesc desc(GeShape(), ::ge::FORMAT_ND, ::ge::DT_INT64);
    auto v = GetValue<int64_t>(value);
    desc.SetRealDimCnt(0);
    return GeTensor(desc, reinterpret_cast<uint8_t *>(&v), sizeof(int64_t));
  } else if (value->isa<FP32Imm>()) {
    // convert scalar FP32 to GeTensor
    MS_LOG(DEBUG) << "convert scalar to tensor with data type = FP32";
    GeTensorDesc desc(GeShape(), ::ge::FORMAT_ND, ::ge::DT_FLOAT);
    auto v = GetValue<float>(value);
    desc.SetRealDimCnt(0);
    return GeTensor(desc, reinterpret_cast<uint8_t *>(&v), sizeof(float));
  } else if (value->isa<BoolImm>()) {
    // convert scalar FP32 to GeTensor
    MS_LOG(DEBUG) << "convert scalar to tensor with data type = Bool";
    GeTensorDesc desc(GeShape(), ::ge::FORMAT_ND, ::ge::DT_BOOL);
    auto v = GetValue<bool>(value);
    desc.SetRealDimCnt(0);
    return GeTensor(desc, reinterpret_cast<uint8_t *>(&v), sizeof(bool));
  } else if (value->isa<StringImm>()) {
    // convert String to GeTensor
    MS_LOG(DEBUG) << "convert string to tensor with data type = String";
    std::string v = GetValue<std::string>(value);
    std::vector<int64_t> ge_shape;
    GeShape shape(ge_shape);
    GeTensorDesc desc(shape, ::ge::FORMAT_ND, ::ge::DT_STRING);
    GeTensor str_tensor(desc);
    (void)str_tensor.SetData(v);
    return str_tensor;
  } else {
    MS_LOG(DEBUG) << "Unsupported value type: " << value->type_name()
                  << " to convert to tensor. Value: " << value->ToString();
  }
  return GeTensor();
}

enum class CustomOpTypeEnum { kUnKnown, kAkg, kTbe, kAiCpu };

CustomOpTypeEnum GetCustomOpTypeDetailEnum(const PrimitivePtr &prim) {
  if (prim == nullptr) {
    return CustomOpTypeEnum::kUnKnown;
  }
  auto type = prim->GetAttr(kAttrType);
  if (type != nullptr && GetValue<std::string>(type) == "GraphKernel") {
    return CustomOpTypeEnum::kAkg;
  }
  auto func_type = prim->GetAttr(kAttrFuncType);
  if (func_type != nullptr) {
    auto func_type_value = GetValue<std::string>(func_type);
    if (func_type_value == kCustomTypeTbe) {
      return CustomOpTypeEnum::kTbe;
    } else if (func_type_value == kCustomTypeAICPU) {
      return CustomOpTypeEnum::kAiCpu;
    }
  }
  return CustomOpTypeEnum::kUnKnown;
}

bool IsCustomPrim(const PrimitivePtr &prim) {
  if (prim == nullptr) {
    return false;
  }
  auto detail_type = GetCustomOpTypeDetailEnum(prim);
  if (prim->name() == "Custom") {
    if (detail_type == CustomOpTypeEnum::kAkg || detail_type == CustomOpTypeEnum::kTbe ||
        detail_type == CustomOpTypeEnum::kAiCpu) {
      return true;
    } else {
      auto value = prim->GetAttr(kAttrType);
      std::string op_type = "";
      if (value != nullptr) {
        op_type = GetValue<std::string>(value);
      }
      auto adpt = device::ascend::FindAdapter(op_type, false);
      if (adpt != nullptr) {
        prim->set_name(op_type);
        MS_LOG(INFO) << "Origin prim name is Custom. Because the adapter can be found, change prim name to prim type: "
                     << op_type << ". This prim is not Custom now.";
        return false;
      } else {
        MS_LOG(INFO) << "Origin prim name is Custom and the adapter can not be found.";
        return true;
      }
    }
  } else {
    return false;
  }
}

bool IsNoNeedConstantFoldCNode(const PrimitivePtr &prim) {
  // ON_THE_FLY Quantization node dont need constant folding.
  MS_EXCEPTION_IF_NULL(prim);
  return prim->GetAttr("no_need_constant_folding") != nullptr;
}

bool IsCustomCNode(const AnfNodePtr &anf) {
  if (anf == nullptr) {
    return false;
  }
  auto node = anf->cast<CNodePtr>();
  if (node == nullptr) {
    return false;
  }
  if (node->inputs().empty()) {
    MS_LOG(EXCEPTION) << "Length of node inputs is empty";
  }
  MS_EXCEPTION_IF_NULL(node->inputs()[0]);
  if (!node->inputs()[0]->isa<ValueNode>()) {
    return false;
  }
  auto cus_prim = GetValueNode<PrimitivePtr>(node->inputs()[0]);
  if (cus_prim == nullptr) {
    return false;
  }

  return IsCustomPrim(cus_prim);
}

bool IsPartialSuccNode(const AnfNodePtr node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    return false;
  }
  auto cnode = node->cast<CNodePtr>();
  if (!cnode->inputs().empty()) {
    for (size_t i = 0; i < cnode->size(); i++) {
      if (IsPartialCNode(cnode->input(i))) {
        return true;
      }
    }
  }
  return false;
}

bool IsPartialCNode(const AnfNodePtr node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    return false;
  }
  auto cnode = node->cast<CNodePtr>();
  if (GetCNodeFuncName(cnode) == prim::kPrimPartial->name()) {
    return true;
  }
  return false;
}

bool IsWhileNode(const AnfNodePtr &node) {
  if (!node->isa<CNode>()) {
    return false;
  }
  auto graph = node->func_graph();
  MS_EXCEPTION_IF_NULL(graph);
  bool in_kg = graph->type_name() == kKernelGraphTypeName;
  auto cnode = node->cast<CNodePtr>();
  ValueNodePtr graph_node = nullptr;
  if (in_kg && IsPrimitiveCNode(node, prim::kPrimCall) && cnode->input(1)->isa<ValueNode>()) {
    graph_node = cnode->input(1)->cast<ValueNodePtr>();
  }
  if (!in_kg) {
    if (IsPrimitiveCNode(cnode->input(0), prim::kPrimPartial)) {
      auto partial_node = cnode->input(0)->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(partial_node);
      auto graph_node_input = partial_node->input(1);
      MS_EXCEPTION_IF_NULL(graph_node_input);
      graph_node = graph_node_input->cast<ValueNodePtr>();
    } else if (cnode->input(0)->cast<ValueNodePtr>()) {
      graph_node = cnode->input(0)->cast<ValueNodePtr>();
    }
  }
  if (graph_node == nullptr) {
    return false;
  }

  auto graph_node_value = graph_node->value();
  MS_EXCEPTION_IF_NULL(graph_node_value);
  if (!graph_node_value->isa<FuncGraph>()) {
    return false;
  }
  auto cond_graph = graph_node_value->cast<FuncGraphPtr>();
  MS_EXCEPTION_IF_NULL(cond_graph);
  if (!cond_graph->recursive()) {
    return false;
  }
  const auto &cond_set = cond_graph->nodes();
  for (auto beg = cond_set.begin(); beg != cond_set.end(); ++beg) {
    if (!((*beg)->isa<CNode>())) {
      continue;
    }
    auto c_beg = (*beg)->cast<CNodePtr>();
    if (IsPrimitiveCNode(c_beg, prim::kPrimSwitch)) {
      auto func_graph = node->func_graph();
      MS_LOG(DEBUG) << "There is while node: " << node->ToString() << " in graph: " << func_graph->ToString();
      return true;
    }
  }
  return false;
}

bool IsCallNode(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    return false;
  }
  auto graph = node->func_graph();
  MS_EXCEPTION_IF_NULL(graph);
  bool in_kg = graph->type_name() == kKernelGraphTypeName;
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (in_kg && IsPrimitiveCNode(node, prim::kPrimCall) && cnode->input(1) != nullptr &&
      cnode->input(1)->isa<ValueNode>()) {
    return true;
  }
  return false;
}

bool CheckSwitchBranch(const AnfNodePtr &node) {
  AnfNodePtr value_node = nullptr;
  if (IsPartialCNode(node)) {
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    value_node = cnode->input(kPartialCNodeValue);
  } else if (IsValueNode<FuncGraph>(node)) {
    value_node = node;
  } else {
    return false;
  }
  auto graph = GetValueNode<FuncGraphPtr>(value_node);
  MS_EXCEPTION_IF_NULL(graph);
  if (graph->recursive()) {
    return false;
  }
  return true;
}

bool IsIfNode(const AnfNodePtr &node) {
  if (!node->isa<CNode>()) {
    return false;
  }
  auto graph = node->func_graph();
  MS_EXCEPTION_IF_NULL(graph);
  bool in_kg = graph->type_name() == kKernelGraphTypeName;
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  CNodePtr switch_node = nullptr;
  if (in_kg && IsPrimitiveCNode(cnode, prim::kPrimSwitch)) {
    switch_node = cnode;
  } else if (!in_kg && IsPrimitiveCNode(cnode->input(0), prim::kPrimSwitch)) {
    switch_node = cnode->input(0)->cast<CNodePtr>();
  } else if (!in_kg && IsPrimitiveCNode(cnode, prim::kPrimIf)) {
    switch_node = cnode;
  } else {
    return false;
  }
  auto true_branch = switch_node->input(kSwitchTrueBranchIndex);
  MS_EXCEPTION_IF_NULL(true_branch);
  auto false_branch = switch_node->input(kSwitchFalseBranchIndex);
  MS_EXCEPTION_IF_NULL(false_branch);

  if (!CheckSwitchBranch(switch_node->input(kSwitchTrueBranchIndex))) {
    return false;
  }
  auto func_graph = node->func_graph();
  MS_LOG(DEBUG) << "There is if node: " << node->ToString() << " in graph: " << func_graph->ToString();
  return true;
}

std::string GetCNodeTargetFuncName(const CNodePtr cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  if (IsCaseNode(cnode)) {
    return string(kNameCase);
  }
  if (IsWhileNode(cnode)) {
    return string(kNameWhile);
  }
  if (IsIfNode(cnode)) {
    return string(kNameIf);
  }
  if (IsCallNode(cnode)) {
    return string(kNamePartitionedCall);
  }
  return GetCNodeFuncName(cnode);
}

bool IsCaseNode(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    return false;
  }
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto graph = node->func_graph();
  MS_EXCEPTION_IF_NULL(graph);
  bool in_kg = graph->type_name() == kKernelGraphTypeName;
  if (in_kg && IsPrimitiveCNode(cnode, prim::kPrimSwitchLayer)) {
    return true;
  }
  if (!in_kg && IsPrimitiveCNode(cnode->input(0), prim::kPrimSwitchLayer)) {
    return true;
  }
  return false;
}

bool ConvertCheck(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->cast<CNodePtr>() || !AnfUtils::IsRealKernel(node)) {
    return true;
  }
  PrimitivePtr prim = common::AnfAlgo::GetCNodePrimitive(node);
  MS_EXCEPTION_IF_NULL(prim);
  auto &adapter_map = device::ascend::OpAdapterMap::get();
  return adapter_map.find(prim->name()) != adapter_map.end();
}

bool DynamicShapeSupportCheck(const AnfNodePtr &node, bool train) {
  auto adpt = device::ascend::FindAdapter(node, train);
  MS_EXCEPTION_IF_NULL(adpt);
  return adpt->GetDynamicShapeSupport();
}

bool SinkGraphCheck(const AnfNodePtr &node, bool train) {
  PrimitivePtr prim = common::AnfAlgo::GetCNodePrimitive(node);
  MS_EXCEPTION_IF_NULL(prim);
  auto adpt = device::ascend::FindAdapter(prim->name(), train);
  MS_EXCEPTION_IF_NULL(adpt);
  auto input_attr_map = adpt->getInputAttrMap();
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto input_size = cnode->size();
  for (auto &it : input_attr_map) {
    if (it.first >= input_size) {
      continue;
    }
    if (!cnode->input(it.first)->isa<ValueNode>()) {
      MS_LOG(DEBUG) << node->fullname_with_scope() << " inputs[" << it.first << "] is not a ValueNode";
      return false;
    }
  }
  auto input_map = adpt->getInputMap();
  for (auto &it : input_map) {
    if (static_cast<size_t>(it.first) >= input_size) {
      continue;
    }
    auto abs = cnode->input(it.first)->abstract();
    MS_EXCEPTION_IF_NULL(abs);
    if (abs->isa<abstract::AbstractAny>()) {
      MS_LOG(DEBUG) << node->fullname_with_scope() << " inputs[" << it.first << "] is a AbstractAny";
      return false;
    }
  }
  return true;
}
}  // namespace device::ascend
}  // namespace mindspore
