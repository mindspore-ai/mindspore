/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#include "utils/check_convert_utils.h"

#include <utility>
#include <vector>
#include <algorithm>
#include <typeinfo>
#include <functional>

#include "abstract/abstract_value.h"
#include "ops/op_utils.h"
#include "ir/dtype/type.h"
#include "ir/dtype/tensor_type.h"
#include "ir/dtype.h"
#include "utils/ms_context.h"

namespace mindspore {
static std::map<std::string, int64_t> DataFormatToEnumMap = {
  {"NCHW", Format::NCHW},   {"NHWC", Format::NHWC},     {"NHWC4", Format::NHWC4},
  {"HWKC", Format::HWKC},   {"HWCK", Format::HWCK},     {"KCHW", Format::KCHW},
  {"CKHW", Format::CKHW},   {"KHWC", Format::KHWC},     {"CHWK", Format::CHWK},
  {"HW", Format::HW},       {"HW4", Format::HW4},       {"NC", Format::NC},
  {"NC4", Format::NC4},     {"NC4HW4", Format::NC4HW4}, {"NUM_OF_FORMAT", Format::NUM_OF_FORMAT},
  {"NCDHW", Format::NCDHW}, {"NWC", Format::NWC},       {"NCW", Format::NCW},
};

static std::map<int64_t, std::string> DataFormatToStrMap = {
  {Format::NCHW, "NCHW"},   {Format::NHWC, "NHWC"},     {Format::NHWC4, "NHWC4"},
  {Format::HWKC, "HWKC"},   {Format::HWCK, "HWCK"},     {Format::KCHW, "KCHW"},
  {Format::CKHW, "CKHW"},   {Format::KHWC, "KHWC"},     {Format::CHWK, "CHWK"},
  {Format::HW, "HW"},       {Format::HW4, "HW4"},       {Format::NC, "NC"},
  {Format::NC4, "NC4"},     {Format::NC4HW4, "NC4HW4"}, {Format::NUM_OF_FORMAT, "NUM_OF_FORMAT"},
  {Format::NCDHW, "NCDHW"}, {Format::NWC, "NWC"},       {Format::NCW, "NCW"},
};

static std::map<std::string, int64_t> ReductionToEnumMap = {
  {"sum", Reduction::REDUCTION_SUM},
  {"mean", Reduction::MEAN},
  {"none", Reduction::NONE},
};

static std::map<int64_t, std::string> ReductionToStrMap = {
  {Reduction::REDUCTION_SUM, "sum"},
  {Reduction::MEAN, "mean"},
  {Reduction::NONE, "none"},
};

static std::map<std::string, int64_t> PadModToEnumMap = {
  {"pad", PadMode::PAD},
  {"same", PadMode::SAME},
  {"valid", PadMode::VALID},
};

static std::map<int64_t, std::string> PadModToStrMap = {
  {PadMode::PAD, "pad"},
  {PadMode::SAME, "same"},
  {PadMode::VALID, "valid"},
};

static std::map<std::string, int64_t> PadModToEnumUpperMap = {
  {"PAD", PadMode::PAD},
  {"SAME", PadMode::SAME},
  {"VALID", PadMode::VALID},
};

static std::map<int64_t, std::string> PadModToStrUpperMap = {
  {PadMode::PAD, "PAD"},
  {PadMode::SAME, "SAME"},
  {PadMode::VALID, "VALID"},
};

AttrConverterPair DataFormatConverter(DataFormatToEnumMap, DataFormatToStrMap);
AttrConverterPair PadModeConverter(PadModToEnumMap, PadModToStrMap);
AttrConverterPair PadModeUpperConverter(PadModToEnumUpperMap, PadModToStrUpperMap);
AttrConverterPair ReductionConverter(ReductionToEnumMap, ReductionToStrMap);

static std::map<std::string, AttrConverterPair> FormatAndPadAttrMap = {
  {ops::kFormat, DataFormatConverter},
  {ops::kPadMode, PadModeConverter},
};

static std::map<std::string, AttrConverterPair> FormatAndPadUpperAttrMap = {
  {ops::kFormat, DataFormatConverter},
  {ops::kPadMode, PadModeUpperConverter},
};

static std::map<std::string, AttrConverterPair> DataFormatMap = {
  {ops::kFormat, DataFormatConverter},
};

static std::map<std::string, AttrConverterPair> ReductionMap = {
  {ops::kReduction, ReductionConverter},
};

static std::map<std::string, std::map<std::string, AttrConverterPair>> PrimAttrConvertMap = {
  {"Conv2D", FormatAndPadAttrMap},
  {"Conv2DTranspose", FormatAndPadUpperAttrMap},
  {"Conv2DBackpropInput", FormatAndPadUpperAttrMap},
  {"Conv2DBackpropFilter", FormatAndPadUpperAttrMap},
  {"Conv3D", FormatAndPadAttrMap},
  {"Conv3DBackpropInput", FormatAndPadAttrMap},
  {"Conv3DBackpropFilter", FormatAndPadAttrMap},
  {"Conv3DTranspose", DataFormatMap},
  {"DepthwiseConv2dNative", FormatAndPadAttrMap},
  {"DepthwiseConv2dNativeBackpropInput", FormatAndPadAttrMap},
  {"DepthwiseConv2dNativeBackpropFilter", FormatAndPadAttrMap},
  {"AvgPool", FormatAndPadUpperAttrMap},
  {"MaxPool", FormatAndPadUpperAttrMap},
  {"MaxPoolWithArgmax", FormatAndPadUpperAttrMap},
  {"AvgPoolGrad", FormatAndPadUpperAttrMap},
  {"AvgPoolGradVm", FormatAndPadUpperAttrMap},
  {"AvgPoolGradGpu", FormatAndPadUpperAttrMap},
  {"AvgPoolGradCpu", FormatAndPadUpperAttrMap},
  {"MaxPoolGrad", FormatAndPadUpperAttrMap},
  {"MaxPoolGradGrad", FormatAndPadUpperAttrMap},
  {"MaxPoolGradWithArgmax", FormatAndPadUpperAttrMap},
  {"MaxPoolGradGradWithArgmax", FormatAndPadUpperAttrMap},
  {"BatchNorm", DataFormatMap},
  {"BatchNormGrad", DataFormatMap},
  {"BiasAdd", DataFormatMap},
  {"BiasAddGrad", DataFormatMap},
  {"BinaryCrossEntropy", ReductionMap},
  {"BinaryCrossEntropyGrad", ReductionMap},
  {"NLLLoss", ReductionMap},
  {"DepthToSpace", DataFormatMap},
  {"Pooling", DataFormatMap},
  {"Deconvolution", DataFormatMap},
  {"AvgPoolV2", DataFormatMap},
  {"MaxPoolV3", DataFormatMap},
  {"FusedBatchNorm", DataFormatMap}};

bool CheckAndConvertUtils::GetDataFormatEnumValue(const ValuePtr &value, int64_t *enum_value) {
  MS_EXCEPTION_IF_NULL(value);
  if (value->isa<StringImm>()) {
    auto attr_value_str = GetValue<std::string>(value);
    if (DataFormatToEnumMap.find(attr_value_str) == DataFormatToEnumMap.end()) {
      MS_LOG(DEBUG) << "The data format " << attr_value_str << " not be converted to enum.";
      return false;
    }
    *enum_value = DataFormatToEnumMap[attr_value_str];
    return true;
  } else {
    *enum_value = GetValue<int64_t>(value);
    return true;
  }
}

void CheckAndConvertUtils::GetPadModEnumValue(const ValuePtr &value, int64_t *enum_value, bool is_upper) {
  MS_EXCEPTION_IF_NULL(value);
  if (value->isa<StringImm>()) {
    auto attr_value_str = GetValue<std::string>(value);

    std::map<std::string, int64_t> pad_map = PadModToEnumMap;
    if (is_upper) {
      pad_map = PadModToEnumUpperMap;
    }
    if (pad_map.find(attr_value_str) == pad_map.end()) {
      MS_LOG(EXCEPTION) << "Invalid pad mode " << attr_value_str << " use pad, valid or same";
    }
    *enum_value = pad_map[attr_value_str];
  } else {
    *enum_value = GetValue<int64_t>(value);
  }
}

void CheckAndConvertUtils::GetReductionEnumValue(const ValuePtr &value, int64_t *enum_value) {
  MS_EXCEPTION_IF_NULL(value);
  if (value->isa<StringImm>()) {
    auto attr_value_str = GetValue<std::string>(value);

    std::map<std::string, int64_t> pad_map = ReductionToEnumMap;
    if (pad_map.find(attr_value_str) == pad_map.end()) {
      MS_LOG(EXCEPTION) << "Invalid pad mode " << attr_value_str << " use pad, valid or same";
    }
    *enum_value = pad_map[attr_value_str];
  } else {
    *enum_value = GetValue<int64_t>(value);
  }
}

AttrConverterPair CheckAndConvertUtils::GetAttrConvertPair(const std::string &op_type, const std::string &attr_name) {
  AttrConverterPair attr_pair;
  if (op_type.empty() || attr_name.empty()) {
    return attr_pair;
  }
  auto op_attr_map_it = PrimAttrConvertMap.find(op_type);
  if (op_attr_map_it == PrimAttrConvertMap.end()) {
    return attr_pair;
  }
  auto attr_pair_it = op_attr_map_it->second.find(attr_name);
  if (attr_pair_it == op_attr_map_it->second.end()) {
    return attr_pair;
  }

  return attr_pair_it->second;
}

bool CheckAndConvertUtils::ConvertAttrValueToInt(const std::string &op_type, const std::string &attr_name,
                                                 ValuePtr *const value) {
  if (value == nullptr || *value == nullptr) {
    MS_LOG(DEBUG) << "value of attr " << op_type << attr_name << " is nullptr.";
    return false;
  }
  if (!(*value)->isa<StringImm>()) {
    return false;
  }
  auto attr_map_pair = GetAttrConvertPair(op_type, attr_name);
  if (attr_map_pair.first.size() == 0) {
    return false;
  }

  std::string real_value = std::dynamic_pointer_cast<StringImm>(*value)->value();
  bool do_convert = false;
  if (attr_map_pair.first.find(real_value) != attr_map_pair.first.end()) {
    do_convert = true;
  }
  if (!do_convert) {
    transform(real_value.begin(), real_value.end(), real_value.begin(), ::toupper);
    if (attr_map_pair.first.find(real_value) != attr_map_pair.first.end()) {
      do_convert = true;
    }
  }
  if (!do_convert) {
    transform(real_value.begin(), real_value.end(), real_value.begin(), ::tolower);
    if (attr_map_pair.first.find(real_value) == attr_map_pair.first.end()) {
      MS_LOG(DEBUG) << "Can not convert " << op_type << " attr " << attr_name << ": " << real_value << " to int";
      return false;
    }
  }
  *value = MakeValue<int64_t>(attr_map_pair.first[real_value]);
  MS_LOG(DEBUG) << "convert str to int, name: " << op_type << ", attr: " << attr_name;
  return true;
}

bool CheckAndConvertUtils::ConvertAttrValueToString(const std::string &op_type, const std::string &attr_name,
                                                    ValuePtr *const value) {
  if (value == nullptr || *value == nullptr) {
    MS_LOG(DEBUG) << "value is nullptr! op_type = " << op_type << ", attr_name = " << attr_name;
    return false;
  }
  if (!(*value)->isa<Int64Imm>()) {
    return false;
  }
  auto attr_map_pair = GetAttrConvertPair(op_type, attr_name);
  if (attr_map_pair.second.size() == 0) {
    return false;
  }

  int64_t real_value = std::dynamic_pointer_cast<Int64Imm>(*value)->value();
  if (attr_map_pair.second.find(real_value) == attr_map_pair.second.end()) {
    MS_LOG(DEBUG) << "Can not convert " << op_type << " attr " << attr_name << ": " << real_value << " to string";
    return false;
  }
  *value = MakeValue<std::string>(attr_map_pair.second[real_value]);
  MS_LOG(DEBUG) << "convert int to str, name: " << op_type << ", attr: " << attr_name;
  return true;
}

void CheckAndConvertUtils::ConvertAttrValueInExport(const std::string &op_type, const std::string &attr_name,
                                                    ValuePtr *const value) {
  if (value == nullptr || *value == nullptr) {
    MS_LOG(DEBUG) << "value is nullptr! op_type = " << op_type << ", attr_name = " << attr_name;
    return;
  }
  // convert enum to string
  ConvertAttrValueToString(op_type, attr_name, value);
}

void CheckAndConvertUtils::ConvertAttrValueInLoad(const std::string &op_type, const std::string &attr_name,
                                                  ValuePtr *const value) {
  if (value == nullptr || *value == nullptr) {
    MS_LOG(DEBUG) << "value is nullptr! op_type = " << op_type << ", attr_name = " << attr_name;
    return;
  }
  // convert string to enum
  ConvertAttrValueToInt(op_type, attr_name, value);
}

namespace {
typedef std::map<std::string, std::function<ValuePtr(ValuePtr)>> AttrFunction;

ValuePtr L2NormalizeAttrConversion(ValuePtr attr) {
  if (attr->isa<Int64Imm>()) {
    return attr;
  }
  auto attr_value = GetValue<std::vector<int64_t>>(attr);
  return MakeValue(attr_value[0]);
}

std::map<std::string, AttrFunction> kIrAttrToOpAttr = {{"L2Normalize", {{"axis", L2NormalizeAttrConversion}}},
                                                       {"L2NormalizeGrad", {{"axis", L2NormalizeAttrConversion}}}};
}  // namespace

std::vector<int64_t> CheckAndConvertUtils::CheckPositiveVector(const std::string &arg_name,
                                                               const std::vector<int64_t> &arg_value,
                                                               const std::string &prim_name) {
  std::ostringstream buffer;
  buffer << "The primitive[" << prim_name << "]'s attribute[" << arg_name
         << "] should be a vector with all positive item. but got [";
  if (std::any_of(arg_value.begin(), arg_value.end(), [](int64_t item) { return item < 0; })) {
    for (auto item : arg_value) {
      buffer << item << ", ";
    }
    buffer << "].";
    MS_EXCEPTION(ValueError) << buffer.str();
  }

  return arg_value;
}

std::string CheckAndConvertUtils::CheckString(const std::string &arg_name, const std::string &arg_value,
                                              const std::set<std::string> &check_list, const std::string &prim_name) {
  if (check_list.find(arg_value) != check_list.end()) {
    return arg_value;
  }
  std::ostringstream buffer;
  buffer << "The primitive[" << prim_name << "]'s attribute[" << arg_name << "]";
  if (check_list.size() == 1) {
    buffer << " must be \"" << (*check_list.begin()) << "\",but got \"" << arg_value << "\".";
    MS_EXCEPTION(ValueError) << buffer.str();
  }
  buffer << " should be a element of {";
  for (const auto &item : check_list) {
    buffer << "\"" << item << "\", ";
  }
  buffer << "}"
         << ",but got \"" << arg_value << "\""
         << ".";
  MS_EXCEPTION(ValueError) << buffer.str();
}

int64_t CheckAndConvertUtils::CheckInteger(const std::string &arg_name, int64_t arg_value, CompareEnum compare_operator,
                                           int64_t match_value, const std::string &prim_name) {
  auto iter = kCompareMap<float>.find(compare_operator);
  if (iter == kCompareMap<float>.end()) {
    MS_EXCEPTION(NotExistsError) << "compare_operator " << compare_operator << " cannot find in the compare map";
  }
  if (iter->second(arg_value, match_value)) {
    return arg_value;
  }
  std::ostringstream buffer;
  if (prim_name.empty()) {
    buffer << "The argument[" << arg_name << "] must ";
  } else {
    buffer << "The primitive[" << prim_name << "]'s " << arg_name << " must ";
  }
  auto iter_to_string = kCompareToString.find(compare_operator);
  if (iter_to_string == kCompareToString.end()) {
    MS_EXCEPTION(NotExistsError) << "compare_operator " << compare_operator << " cannot find in the compare string map";
  }
  buffer << iter_to_string->second << match_value << ", but got " << arg_value << ".";
  MS_EXCEPTION(ValueError) << buffer.str();
}

void CheckAndConvertUtils::CheckInputArgs(const std::vector<AbstractBasePtr> &input_args,
                                          const CompareEnum compare_operator, const int64_t match_value,
                                          const std::string &prim_name) {
  (void)CheckInteger("input number", SizeToLong(input_args.size()), compare_operator, match_value, prim_name);
  for (size_t index = 0; index < input_args.size(); index++) {
    if (input_args[index] == nullptr) {
      MS_EXCEPTION(ValueError) << "The " << index << "'s input of " << prim_name << " is nullptr.";
    }
  }
}

TypePtr CheckAndConvertUtils::GetInputTensorType(const std::vector<AbstractBasePtr> &input_args, const size_t index,
                                                 const std::string &prim_name) {
  if (input_args.size() <= index) {
    MS_EXCEPTION(ValueError) << "The primitive[" << prim_name << "]'s input index[" << index
                             << "] is out of the input number " << input_args.size();
  }
  auto input_arg = input_args[index];
  if (input_arg == nullptr) {
    MS_EXCEPTION(ValueError) << "The primitive[" << prim_name << "]'s input index[" << index << "] is nullptr.";
  }
  auto base_type = input_arg->BuildType();
  MS_EXCEPTION_IF_NULL(base_type);
  if (!base_type->isa<TensorType>()) {
    MS_EXCEPTION(ValueError) << "The primitive[" << prim_name << "]'s input index[" << index << "] is not a tensor.";
  }
  auto tensor_type = base_type->cast<TensorTypePtr>();
  MS_EXCEPTION_IF_NULL(tensor_type);
  auto type = tensor_type->element();
  MS_EXCEPTION_IF_NULL(type);
  return type;
}

ShapeMap CheckAndConvertUtils::ConvertShapePtrToShapeMap(const BaseShapePtr &shape) {
  MS_EXCEPTION_IF_NULL(shape);
  if (!shape->isa<abstract::Shape>()) {
    return std::map<std::string, std::vector<int64_t>>();
  }
  auto shape_element = shape->cast<abstract::ShapePtr>();
  MS_EXCEPTION_IF_NULL(shape_element);
  ShapeMap shape_map;
  shape_map[kShape] = shape_element->shape();
  shape_map[kMinShape] = shape_element->min_shape();
  shape_map[kMaxShape] = shape_element->max_shape();
  return shape_map;
}

abstract::ShapePtr CheckAndConvertUtils::GetTensorInputShape(const std::string &prim_name,
                                                             const std::vector<AbstractBasePtr> &input_args,
                                                             int64_t index) {
  auto abstract = CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, LongToSize(index));
  MS_EXCEPTION_IF_NULL(abstract);
  auto base_shape = abstract->BuildShape();
  MS_EXCEPTION_IF_NULL(base_shape);
  if (!base_shape->isa<abstract::Shape>()) {
    MS_LOG(EXCEPTION) << prim_name << " can not get shape for input " << index;
  }
  auto shape = base_shape->cast<abstract::ShapePtr>();
  MS_EXCEPTION_IF_NULL(shape);
  return shape;
}

void CheckAndConvertUtils::Check(const string &arg_name, int64_t arg_value, CompareEnum compare_type, const string &,
                                 int64_t value, const string &prim_name, ExceptionType) {
  auto iter = kCompareMap<float>.find(compare_type);
  if (iter == kCompareMap<float>.end()) {
    MS_EXCEPTION(NotExistsError) << "the compare type :" << compare_type << " is not in the compare map";
  }
  if (iter->second(arg_value, value)) {
    return;
  }
  std::ostringstream buffer;
  if (prim_name.empty()) {
    buffer << "The attribute[" << arg_name << "] must ";
  } else {
    buffer << "The primitive[" << prim_name << "]'s attribute[" << arg_name << "] must ";
  }
  auto iter_to_string = kCompareToString.find(compare_type);
  if (iter_to_string == kCompareToString.end()) {
    MS_EXCEPTION(NotExistsError) << "compare_operator " << compare_type << " cannot find in the compare string map";
  }
  buffer << iter_to_string->second << value << ", but got " << arg_value << ".";
  MS_EXCEPTION(ValueError) << buffer.str();
}

TypePtr CheckAndConvertUtils::CheckTensorTypeSame(const std::map<std::string, TypePtr> &types,
                                                  const std::set<TypePtr> &check_list, const std::string &prim_name) {
  if (types.empty()) {
    MS_EXCEPTION(ArgumentError) << "Trying to use the function to check a empty types map!";
  }
  for (const auto &item : types) {
    auto type = item.second;
    MS_EXCEPTION_IF_NULL(type);
    if (!type->isa<TensorType>()) {
      std::ostringstream buffer;
      buffer << "The primitive[" << prim_name << "]'s input arguments must be all tensor.\n";
      if (!check_list.empty()) {
        buffer << "Valid type list: {";
        for (auto const &valid_type : check_list) {
          if (valid_type->isa<TensorType>()) {
            buffer << valid_type->ToString() << ", ";
            break;
          }
          buffer << "Tensor[" << valid_type << "]"
                 << ", ";
        }
        buffer << "}.\n";
      }
      for (const auto &type_info : types) {
        buffer << "input argument[" << type_info.first << "]"
               << ":" << type_info.second->ToString() << "\n";
      }
      MS_EXCEPTION(TypeError) << buffer.str();
    }
  }
  auto check_type = _CheckTypeSame(types, prim_name, false);
  std::string input_names = "";
  for (const auto &item : types) {
    (void)input_names.append(item.first);
    (void)input_names.append(", ");
  }
  return CheckSubClass(input_names, check_type, check_list, prim_name);
}

TypePtr CheckAndConvertUtils::CheckTensorTypeValid(const std::string &type_name, const TypePtr &type,
                                                   const std::set<TypePtr> &check_list, const std::string &prim_name) {
  MS_EXCEPTION_IF_NULL(type);
  if (!type->isa<TensorType>()) {
    MS_EXCEPTION(TypeError) << "The Primitive[" << prim_name << "] input argument[" << type_name
                            << "] must be a Tensor but got " << type->ToString() << ".";
  }
  auto tensor_type = type->cast<TensorTypePtr>();
  auto element = tensor_type->element();
  MS_EXCEPTION_IF_NULL(element);
  for (const TypePtr &item : check_list) {
    if (item->isa<TensorType>()) {
      auto item_tensor_type = item->cast<TensorTypePtr>();
      if (item_tensor_type->element() == nullptr) {
        return element;
      }
    }
  }
  return CheckSubClass(type_name, type, check_list, prim_name);
}

ShapeVector CheckAndConvertUtils::CheckTensorIntValue(const std::string &type_name, const ValuePtr &value,
                                                      const std::string &prim_name) {
  if (value == nullptr) {
    MS_EXCEPTION(ValueError) << "The primitive[" << prim_name << "] input argument[" << type_name
                             << "] value is nullptr.";
  }
  ShapeVector tensor_value;
  if (!value->isa<tensor::Tensor>()) {
    MS_EXCEPTION(ValueError) << "The primitive[" << prim_name << "] input argument[" << type_name
                             << "] must be a tensor,but got " << value->ToString();
  }
  auto input_tensor = value->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(input_tensor);
  size_t data_size = LongToSize(input_tensor->DataSize());
  auto tensor_type = input_tensor->Dtype();
  if (tensor_type->type_id() == kNumberTypeInt32) {
    auto data_c = reinterpret_cast<int *>(input_tensor->data_c());
    MS_EXCEPTION_IF_NULL(data_c);
    for (size_t i = 0; i < data_size; i++) {
      tensor_value.push_back(static_cast<int64_t>(*data_c));
      ++data_c;
    }
  } else if (tensor_type->type_id() == kNumberTypeInt64) {
    auto tensor_data = reinterpret_cast<int64_t *>(input_tensor->data_c());
    MS_EXCEPTION_IF_NULL(tensor_data);
    tensor_value = {tensor_data, tensor_data + data_size};
  } else {
    MS_EXCEPTION(TypeError) << "The primitive[" << prim_name << "] input argument[" << type_name
                            << "] must be a Tensor[Int64] or Tensor[Int32] type,but got " << value->ToString();
  }
  return tensor_value;
}

TypePtr CheckAndConvertUtils::CheckSubClass(const std::string &type_name, const TypePtr &type,
                                            const std::set<TypePtr> &template_types, const std::string &prim_name) {
  auto check_type = type;
  bool ok = std::any_of(template_types.begin(), template_types.end(), [check_type](const TypePtr &accept) -> bool {
    return IsIdentidityOrSubclass(check_type, accept);
  });
  if (ok) {
    return check_type;
  }
  if (type->isa<TensorType>()) {
    auto tensor_type = type->cast<TensorTypePtr>();
    check_type = tensor_type->element();
  }
  ok = std::any_of(template_types.begin(), template_types.end(),
                   [check_type](const TypePtr &accept) -> bool { return IsIdentidityOrSubclass(check_type, accept); });
  if (ok) {
    return check_type;
  } else {
    std::ostringstream buffer;
    buffer << "Primitive[" << prim_name << "]'s input argument[" << type_name << "] must be a type of ";
    buffer << GetErrorTypeString(template_types, type) << ", but got " << type->ToString();
    buffer << ".";
    MS_EXCEPTION(TypeError) << buffer.str();
  }
}

TypePtr CheckAndConvertUtils::CheckScalarOrTensorTypesSame(const std::map<std::string, TypePtr> &args,
                                                           const std::set<TypePtr> &valid_values,
                                                           const std::string &prim_name, const bool allow_mix) {
  auto arg_ = _CheckTypeSame(args, prim_name, allow_mix);
  return CheckTypeValid(args.begin()->first, arg_, valid_values, prim_name);
}

TypePtr CheckAndConvertUtils::_CheckTypeSame(const std::map<std::string, TypePtr> &args, const std::string &prim_name,
                                             const bool allow_mix) {
  if (args.empty()) {
    MS_EXCEPTION(ArgumentError) << "Trying to use the function to check a empty types map!";
  }
  std::ostringstream buffer;
  TypePtr return_type = args.begin()->second;
  buffer << "The primitive[" << prim_name << "]";
  bool tensor_flag = return_type->isa<TensorType>();
  std::set<TypeId> types_id;
  for (const auto &elem : args) {
    auto type = elem.second;
    MS_EXCEPTION_IF_NULL(type);
    if (!allow_mix) {
      // input must be all tensor or all other type
      if ((tensor_flag && !type->isa<TensorType>()) || (!tensor_flag && type->isa<TensorType>())) {
        buffer << "'s "
               << "input type must be same.\n";
        for (const auto &error_elem : args) {
          buffer << "input argument[" << error_elem.first << "]:" << error_elem.second->ToString() << "\n";
        }
        MS_EXCEPTION(TypeError) << buffer.str();
      }
    }
    if (type->isa<TensorType>()) {
      auto tensor_type = type->cast<TensorTypePtr>();
      auto element = tensor_type->element();
      MS_EXCEPTION_IF_NULL(element);
      if (!allow_mix) {
        return_type = element;
      } else {
        return_type = tensor_type;
      }
      (void)types_id.emplace(element->type_id());
    } else {
      (void)types_id.emplace(type->type_id());
    }
    if (types_id.size() > 1) {
      buffer << "'s input type must be same.\n";
      for (const auto &item : args) {
        buffer << "name:[" << item.first << "]:" << item.second->ToString() << ".\n";
      }
      MS_EXCEPTION(TypeError) << buffer.str();
    }
  }
  return return_type->DeepCopy();
}

TypePtr CheckAndConvertUtils::CheckTypeValid(const std::string &arg_name, const TypePtr &arg_type,
                                             const std::set<TypePtr> &valid_type, const std::string &prim_name) {
  if (valid_type.empty()) {
    MS_EXCEPTION(ArgumentError) << "Trying to use the function to check a empty valid_type!";
  }
  MS_EXCEPTION_IF_NULL(arg_type);
  if (arg_type->isa<TensorType>()) {
    return CheckTensorTypeValid(arg_name, arg_type, valid_type, prim_name);
  }
  return CheckSubClass(arg_name, arg_type, valid_type, prim_name);
}

bool CheckAndConvertUtils::CheckIrAttrtoOpAttr(const std::string &op_type, const std::string &attr_name,
                                               ValuePtr *const value) {
  if (*value == nullptr) {
    MS_LOG(DEBUG) << "value is nullptr! op_type = " << op_type << ", attr_name = " << attr_name;
    return false;
  }
  if (op_type.empty() || attr_name.empty()) {
    return false;
  }
  auto op_map = kIrAttrToOpAttr.find(op_type);
  if (op_map == kIrAttrToOpAttr.end()) {
    return false;
  }
  auto attr_func = op_map->second.find(attr_name);
  if (attr_func == op_map->second.end()) {
    return false;
  }
  *value = attr_func->second(*value);
  MS_LOG(DEBUG) << "convert ir attr to op attr, name: " << op_type << ", attr: " << attr_name;
  return true;
}

void CheckAndConvertUtils::CheckSummaryParam(const AbstractBasePtr &name, const AbstractBasePtr &value,
                                             const std::string &class_name) {
  MS_EXCEPTION_IF_NULL(name);
  MS_EXCEPTION_IF_NULL(value);
  CheckMode(class_name);
  (void)CheckTypeValid("name", name->BuildType(), {kString}, class_name);
  auto s = GetValue<std::string>(name->BuildValue());
  if (s.empty()) {
    MS_EXCEPTION(ValueError) << "The primitive[" << class_name << "]'s input argument[name] "
                             << " cannot be an empty string.";
  }
  (void)CheckTypeValid("value", value->BuildType(), {kTensorType}, class_name);
}

void CheckAndConvertUtils::CheckMode(const std::string &class_name) {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (ms_context->get_param<int>(MS_CTX_EXECUTION_MODE) == kPynativeMode) {
    MS_EXCEPTION(NotSupportError) << "The primitive[" << class_name << "] does not support PyNativeMode.\n"
                                  << "Please convert the mode to GraphMode";
  }
}

std::vector<int64_t> CheckAndConvertUtils::CheckAttrIntOrTupleInt(const std::string &arg_name, const ValuePtr &attr,
                                                                  const std::string &prim_name) {
  std::vector<int64_t> result;
  bool is_correct = false;
  MS_EXCEPTION_IF_NULL(attr);
  if (attr->isa<ValueTuple>()) {
    std::vector<ValuePtr> attr_vec = attr->cast<ValueTuplePtr>()->value();
    is_correct = std::all_of(attr_vec.begin(), attr_vec.end(), [&result](const ValuePtr &e) -> bool {
      MS_EXCEPTION_IF_NULL(e);
      if (e->isa<Int64Imm>()) {
        (void)result.emplace_back(GetValue<int64_t>(e));
        return true;
      }
      return false;
    });
  } else {
    if (attr->isa<Int64Imm>()) {
      is_correct = true;
      int64_t attr_val = attr->cast<Int64ImmPtr>()->value();
      result.push_back(attr_val);
    }
  }
  if (!is_correct) {
    MS_EXCEPTION(TypeError) << "The primitive[" << prim_name << "]'s attribute[" << arg_name
                            << "] must be a Int or a tuple with all Int elements, but got " << attr->ToString();
  }
  return result;
}

std::vector<int64_t> CheckAndConvertUtils::CheckAttrTupleInt(const std::string &arg_name, const ValuePtr &attr,
                                                             const std::string &prim_name) {
  std::vector<int64_t> result;
  MS_EXCEPTION_IF_NULL(attr);
  if (attr->isa<ValueTuple>()) {
    std::vector<ValuePtr> attr_vec = attr->cast<ValueTuplePtr>()->value();
    (void)std::transform(
      attr_vec.begin(), attr_vec.end(), std::back_inserter(result), [=](const ValuePtr &e) -> int64_t {
        if (!e->isa<Int64Imm>()) {
          MS_EXCEPTION(TypeError) << "The primitive[" << prim_name << "]'s attribute[" << arg_name
                                  << "] must be a tuple with all Int elements, but got " << attr->ToString();
        }
        return GetValue<int64_t>(e);
      });
  } else {
    MS_EXCEPTION(TypeError) << "The primitive[" << prim_name << "]'s attribute[" << arg_name
                            << "] must be a tuple with all Int elements, but got " << attr->ToString() << ".";
  }
  return result;
}

void CheckAndConvertUtils::CheckMinMaxShape(const ShapeVector &shape, ShapeVector *min_shape, ShapeVector *max_shape) {
  *min_shape = (*min_shape).empty() ? shape : *min_shape;
  *max_shape = (*max_shape).empty() ? shape : *max_shape;
}

int64_t CheckAndConvertUtils::GetAndCheckFormat(const ValuePtr &value) {
  int64_t data_format;
  bool result = CheckAndConvertUtils::GetDataFormatEnumValue(value, &data_format);
  if (!result || (data_format != Format::NHWC && data_format != Format::NCHW && data_format != Format::NCDHW)) {
    MS_LOG(EXCEPTION) << "data format is invalid, only support NCHW, NHWC and NCDHW";
  }
  return data_format;
}
size_t CheckAndConvertUtils::GetRemoveMonadAbsNum(const AbstractBasePtrList &abs_list) {
  size_t remove_monad_count = abs_list.size();
  for (const auto &item : abs_list) {
    if (item->isa<abstract::AbstractMonad>()) {
      --remove_monad_count;
    }
  }

  for (size_t i = 0; i < remove_monad_count; ++i) {
    if (abs_list[i]->isa<abstract::AbstractMonad>()) {
      MS_EXCEPTION(UnknownError) << "The monad inputs of the node must at last of the node inputs.";
    }
  }
  return remove_monad_count;
}

bool CheckAndConvertUtils::HasDynamicShapeInput(const AbstractBasePtrList &abs_list) {
  for (const auto &item : abs_list) {
    MS_EXCEPTION_IF_NULL(item);
    auto shape = item->BuildShape();
    if (shape->IsDynamic()) {
      return true;
    }
  }
  return false;
}

std::string CheckAndConvertUtils::GetErrorTypeString(const std::set<TypePtr> &check_list, const TypePtr &check_type) {
  std::ostringstream buffer;
  buffer << "{";
  // got tensor type list
  for (const auto &item : check_list) {
    if (item->isa<TensorType>()) {
      buffer << item->ToString();
      buffer << ", ";
      continue;
    }
    buffer << "Tensor[" << item->ToString() << "], ";
  }
  if (check_type->isa<TensorType>()) {
    buffer << "}";
    return buffer.str();
  }
  // got python type
  std::set<std::string> type_string;
  for (const auto &item : check_list) {
    if (item->isa<Float>()) {
      (void)type_string.emplace("Float");
    }
    if (item->isa<Int>()) {
      (void)type_string.emplace("Int");
    }
    if (item->isa<Bool>()) {
      (void)type_string.emplace("Bool");
    }
    if (item->isa<UInt>()) {
      (void)type_string.emplace("UInt");
    }
  }
  for (const auto &item : type_string) {
    buffer << item << ",";
  }
  buffer << "}";
  return buffer.str();
}
}  // namespace mindspore
