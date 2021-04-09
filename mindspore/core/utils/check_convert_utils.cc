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
};

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
  return false;
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

void ConvertTargetAttr(const std::string &attr_name, ValuePtr *const value) {
  if (attr_name == "primitive_target") {
    auto target_value = GetValue<std::string>(*value);
    if (target_value == "CPU") {
      *value = MakeValue<std::string>("host");
    } else {
      MS_LOG(EXCEPTION) << "The primitive_target only support CPU when export, but got " << target_value;
    }
  }
}

void RestoreTargetAttr(const std::string &attr_name, ValuePtr *const value) {
  if (attr_name == "primitive_target") {
    auto target_value = GetValue<std::string>(*value);
    // compatible with exported model
    if (target_value == "CPU") {
      return;
    }
    if (target_value == "host") {
      *value = MakeValue<std::string>("CPU");
    } else {
      MS_LOG(EXCEPTION) << "Invalid primitive_target value: " << target_value;
    }
  }
}

void CheckAndConvertUtils::ConvertAttrValueInExport(const std::string &op_type, const std::string &attr_name,
                                                    ValuePtr *const value) {
  if (value == nullptr || *value == nullptr) {
    MS_LOG(DEBUG) << "value is nullptr! op_type = " << op_type << ", attr_name = " << attr_name;
    return;
  }
  // convert enum to string
  ConvertAttrValueToString(op_type, attr_name, value);
  // set cpu target as host
  ConvertTargetAttr(attr_name, value);
}

void CheckAndConvertUtils::ConvertAttrValueInLoad(const std::string &op_type, const std::string &attr_name,
                                                  ValuePtr *const value) {
  if (value == nullptr || *value == nullptr) {
    MS_LOG(DEBUG) << "value is nullptr! op_type = " << op_type << ", attr_name = " << attr_name;
    return;
  }
  // convert string to enum
  ConvertAttrValueToInt(op_type, attr_name, value);
  // restore target as CPU
  RestoreTargetAttr(attr_name, value);
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

bool CheckAndConvertUtils::IsEqualVector(const std::vector<int64_t> &vec_1, const std::vector<int64_t> &vec_2) {
  if (vec_1.size() != vec_2.size()) {
    return false;
  }
  for (size_t index = 0; index < vec_1.size(); ++index) {
    if (vec_1[index] != vec_2[index]) {
      return false;
    }
  }
  return true;
}

std::vector<int64_t> CheckAndConvertUtils::CheckPositiveVector(const std::string &arg_name,
                                                               const std::vector<int64_t> &arg_value,
                                                               const std::string &prim_name) {
  for (auto item : arg_value) {
    if (item < 0) {
      MS_EXCEPTION(ValueError) << "For " << prim_name << " attr " << arg_name << " should be a positive vector";
    }
  }
  return arg_value;
}

std::string CheckAndConvertUtils::CheckString(const std::string &arg_name, const std::string &arg_value,
                                              const std::set<std::string> &check_list, const std::string &prim_name) {
  if (check_list.find(arg_value) != check_list.end()) {
    return arg_value;
  }
  std::ostringstream buffer;
  buffer << "For " << prim_name << " the " << arg_name << " should be str and must be ";
  if (check_list.size() == 1) {
    buffer << (*check_list.begin()) << "but got " << arg_value;
    MS_EXCEPTION(ValueError) << buffer.str();
  }
  buffer << "one of {";
  for (const auto &item : check_list) {
    buffer << item << " ,";
  }
  buffer << " }"
         << " but got " << arg_value;
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
    buffer << "The ";
  } else {
    buffer << "For " << prim_name << " the ";
  }
  buffer << arg_name << " must ";
  auto iter_to_string = kCompareToString.find(compare_operator);
  if (iter_to_string == kCompareToString.end()) {
    MS_EXCEPTION(NotExistsError) << "compare_operator " << compare_operator << " cannot find in the compare string map";
  }
  buffer << iter_to_string->second << match_value << " , but got " << arg_value;
  MS_EXCEPTION(ValueError) << buffer.str();
}

std::vector<int64_t> CheckAndConvertUtils::ConvertShapePtrToShape(const std::string &arg_name,
                                                                  const BaseShapePtr &shape,
                                                                  const std::string &prim_name) {
  MS_EXCEPTION_IF_NULL(shape);
  if (!shape->isa<abstract::Shape>()) {
    return std::vector<int64_t>();
  }
  auto shape_element = shape->cast<abstract::ShapePtr>();
  MS_EXCEPTION_IF_NULL(shape_element);
  return shape_element->shape();
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

void CheckAndConvertUtils::Check(const string &arg_name, int64_t arg_value, CompareEnum compare_type,
                                 const string &value_name, int64_t value, const string &prim_name,
                                 ExceptionType exception_type) {
  auto iter = kCompareMap<float>.find(compare_type);
  if (iter == kCompareMap<float>.end()) {
    MS_EXCEPTION(NotExistsError) << "the compare type :" << compare_type << " is not in the compare map";
  }
  if (iter->second(arg_value, value)) {
    return;
  }
  std::ostringstream buffer;
  if (prim_name.empty()) {
    buffer << "The ";
  } else {
    buffer << "For " << prim_name << " the ";
  }
  auto iter_to_string = kCompareToString.find(compare_type);
  if (iter_to_string == kCompareToString.end()) {
    MS_EXCEPTION(NotExistsError) << "compare_operator " << compare_type << " cannot find in the compare string map";
  }
  MS_EXCEPTION(exception_type) << buffer.str() << arg_name << " should be " << iter_to_string->second << value
                               << " but got " << arg_value;
}

TypePtr CheckAndConvertUtils::CheckTensorTypeSame(const std::map<std::string, TypePtr> &types,
                                                  const std::set<TypePtr> &check_list, const std::string &prim_name) {
  if (types.empty()) {
    MS_EXCEPTION(ArgumentError) << "Trying to use the function to check a empty types map!";
  }
  auto type = types.begin()->second;
  MS_EXCEPTION_IF_NULL(type);
  if (!type->isa<TensorType>()) {
    MS_EXCEPTION(TypeError) << "The " << prim_name << "'s " << types.begin()->first
                            << " input must be a tensor but got " << type->ToString();
  }
  TypePtr check_type = _CheckTypeSame(types, prim_name, false);
  return CheckTypeValid(types.begin()->first, check_type, check_list, prim_name);
}

TypePtr CheckAndConvertUtils::CheckTensorTypeValid(const std::string &type_name, const TypePtr &type,
                                                   const std::set<TypePtr> &check_list, const std::string &prim_name) {
  MS_EXCEPTION_IF_NULL(type);
  if (!type->isa<TensorType>()) {
    MS_EXCEPTION(TypeError) << "The " << prim_name << "'s " << type_name << " input must be tensor type but got "
                            << type->ToString();
  }
  auto tensor_type = type->cast<TensorTypePtr>();
  MS_EXCEPTION_IF_NULL(tensor_type);
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
  return CheckSubClass(type_name, element, check_list, prim_name);
}

TypePtr CheckAndConvertUtils::CheckSubClass(const std::string &type_name, const TypePtr &type_,
                                            const std::set<TypePtr> &template_types, const std::string &prim_name) {
  bool ok = std::any_of(template_types.begin(), template_types.end(),
                        [type_](const TypePtr &accept) -> bool { return IsIdentidityOrSubclass(type_, accept); });
  if (ok) {
    return type_;
  } else {
    std::string type_str = type_->ToString();
    std::ostringstream buffer;
    buffer << "For '" << prim_name << "', the type of `" << type_name << "` should be subclass of ";
    for (const auto &template_type : template_types) {
      buffer << template_type->ToString() << ",";
    }
    buffer << " but got " << type_str << ".";
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
  TypePtr return_type = nullptr;
  buffer << "For " << prim_name;
  auto first_type = args.begin()->second;
  MS_EXCEPTION_IF_NULL(first_type);
  bool tensor_flag = first_type->isa<TensorType>();
  std::set<TypeId> types_id;
  for (const auto &elem : args) {
    auto type = elem.second;
    MS_EXCEPTION_IF_NULL(type);
    if (!allow_mix) {
      // input must be all tensor or all other type
      if (tensor_flag ^ type->isa<TensorType>()) {
        buffer << "For " << prim_name << "'s "
               << "type is not same";
        for (const auto &error_elem : args) {
          buffer << " [ name :" << error_elem.first << ", type : " << error_elem.second->ToString() << "]";
        }
        MS_EXCEPTION(TypeError) << buffer.str();
      }
    }
    if (type->isa<TensorType>()) {
      auto tensor_type = type->cast<TensorTypePtr>();
      MS_EXCEPTION_IF_NULL(tensor_type);
      auto element = tensor_type->element();
      return_type = element->DeepCopy();
      MS_EXCEPTION_IF_NULL(element);
      types_id.emplace(element->type_id());
    } else {
      types_id.emplace(type->type_id());
      return_type = type->DeepCopy();
    }
    if (types_id.size() > 1) {
      buffer << "'s input type is not same : ";
      for (const auto &item : args) {
        buffer << "[ name : " << item.first << " ,type : " << item.second->ToString() << "]";
      }
      MS_EXCEPTION(TypeError) << buffer.str();
    }
  }
  return return_type;
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
  CheckTypeValid("name", name->BuildType(), {kString}, class_name);
  auto s = GetValue<std::string>(name->BuildValue());
  if (s.empty()) {
    MS_EXCEPTION(ValueError) << "For 'name' the value should by valid string in " << class_name
                             << ", but got an empty string.";
  }
  CheckTypeValid("value", value->BuildType(), {kTensorType}, class_name);
}

void CheckAndConvertUtils::CheckMode(const std::string &class_name) {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (ms_context->get_param<int>(MS_CTX_EXECUTION_MODE) == kPynativeMode) {
    MS_EXCEPTION(NotSupportError) << class_name << "operator does not support PyNative mode.";
  }
}

std::vector<int64_t> CheckAndConvertUtils::CheckAttrIntOrTupleInt(const std::string &arg_name, const ValuePtr &attr,
                                                                  const std::string &prim_name) {
  std::vector<int64_t> result;
  MS_EXCEPTION_IF_NULL(attr);
  if (attr->isa<ValueTuple>()) {
    std::vector<ValuePtr> attr_vec = attr->cast<ValueTuplePtr>()->value();
    (void)std::transform(
      attr_vec.begin(), attr_vec.end(), std::back_inserter(result), [=](const ValuePtr &e) -> int64_t {
        if (!e->isa<Int64Imm>()) {
          MS_EXCEPTION(TypeError) << "For " << prim_name << ", the type of" << arg_name << " must be Int64";
        }
        return GetValue<int64_t>(e);
      });
  } else {
    if (!attr->isa<Int64Imm>()) {
      MS_EXCEPTION(TypeError) << "For " << prim_name << ", the type of" << arg_name << " must be Int64";
    }
    int64_t attr_val = attr->cast<Int64ImmPtr>()->value();
    result.push_back(attr_val);
  }
  return result;
}
}  // namespace mindspore
