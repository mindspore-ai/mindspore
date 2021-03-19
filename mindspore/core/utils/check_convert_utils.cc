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

namespace mindspore {
static std::map<std::string, int64_t> DataFormatToEnumMap = {
  {"NCHW", Format::NCHW},   {"NHWC", Format::NHWC},     {"NHWC4", Format::NHWC4},
  {"HWKC", Format::HWKC},   {"HWCK", Format::HWCK},     {"KCHW", Format::KCHW},
  {"CKHW", Format::CKHW},   {"KHWC", Format::KHWC},     {"CHWK", Format::CHWK},
  {"HW", Format::HW},       {"HW4", Format::HW4},       {"NC", Format::NC},
  {"NC4", Format::NC4},     {"NC4HW4", Format::NC4HW4}, {"NUM_OF_FORMAT", Format::NUM_OF_FORMAT},
  {"NCDHW", Format::NCDHW},
};

static std::map<int64_t, std::string> DataFormatToStrMap = {
  {Format::NCHW, "NCHW"},   {Format::NHWC, "NHWC"},     {Format::NHWC4, "NHWC4"},
  {Format::HWKC, "HWKC"},   {Format::HWCK, "HWCK"},     {Format::KCHW, "KCHW"},
  {Format::CKHW, "CKHW"},   {Format::KHWC, "KHWC"},     {Format::CHWK, "CHWK"},
  {Format::HW, "HW"},       {Format::HW4, "HW4"},       {Format::NC, "NC"},
  {Format::NC4, "NC4"},     {Format::NC4HW4, "NC4HW4"}, {Format::NUM_OF_FORMAT, "NUM_OF_FORMAT"},
  {Format::NCDHW, "NCDHW"},
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
    MS_LOG(INFO) << "value is nullptr! op_type = " << op_type << ", attr_name = " << attr_name;
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
    MS_LOG(INFO) << "value is nullptr! op_type = " << op_type << ", attr_name = " << attr_name;
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
    MS_LOG(INFO) << "value is nullptr! op_type = " << op_type << ", attr_name = " << attr_name;
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
                                                               const std::string &prim_name, bool allow_four,
                                                               bool ret_four) {
  auto raise_message = [allow_four, prim_name, arg_value, arg_name]() -> void {
    std::ostringstream buffer;
    buffer << "For " << prim_name << " attr " << arg_name << " should be a positive vector of size two ";
    if (allow_four) {
      buffer << "or four ";
    }
    buffer << " positive int64_t numbers , but got [";
    for (auto item : arg_value) {
      buffer << item << ",";
    }
    buffer << "]";
    MS_EXCEPTION(ValueError) << buffer.str();
  };
  for (auto item : arg_value) {
    if (item < 0) {
      raise_message();
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
    MS_EXCEPTION(ValueError) << "The " << arg_name << "'s shape is " << shape->ToString()
                             << "should be a common shape!";
  }
  auto shape_element = shape->cast<abstract::ShapePtr>();
  MS_EXCEPTION_IF_NULL(shape_element);
  return shape_element->shape();
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

TypeId CheckAndConvertUtils::CheckTensorTypeSame(const std::map<std::string, TypePtr> &types,
                                                 const std::set<TypeId> &check_list, const std::string &prim_name) {
  if (types.empty()) {
    MS_EXCEPTION(ArgumentError) << "Trying to use the function to check a empty types map!";
  }
  std::set<TypeId> types_id;
  std::ostringstream buffer;
  buffer << "For " << prim_name;
  for (const auto &type : types) {
    MS_EXCEPTION_IF_NULL(type.second);
    if (!type.second->isa<TensorType>()) {
      MS_EXCEPTION(TypeError) << "The " << prim_name << "'s" << type.first << " input must be tensor type but got "
                              << type.second->ToString();
    }
    auto tensor_type = type.second->cast<TensorTypePtr>();
    MS_EXCEPTION_IF_NULL(tensor_type);
    auto element = tensor_type->element();
    MS_EXCEPTION_IF_NULL(element);
    types_id.emplace(element->type_id());
  }
  if (types_id.size() > 1) {
    buffer << "'s input type is not same : ";
    for (const auto &item : types) {
      buffer << "[ name : " << item.first << " ,type : " << item.second->ToString() << "]";
    }
    MS_EXCEPTION(TypeError) << buffer.str();
  }
  if (check_list.find(*types_id.begin()) == check_list.end()) {
    buffer << " type of ";
    for (const auto &elem : types) {
      buffer << elem.first << " should be in [";
      for (auto type_elem : check_list) {
        buffer << TypeIdToType(type_elem)->ToString() << " ,";
      }
      buffer << "] , but got " << types.begin()->second->ToString();
    }
    MS_EXCEPTION(TypeError) << buffer.str();
  }
  return *types_id.begin();
}

void CheckAndConvertUtils::CheckTensorTypeValid(const std::string &type_name, const TypePtr type,
                                                const std::set<TypeId> &check_list, const std::string &prim_name) {
  MS_EXCEPTION_IF_NULL(type);
  if (!type->isa<TensorType>()) {
    MS_EXCEPTION(TypeError) << "The " << prim_name << "'s " << type_name << " input must be tensor type but got "
                            << type->ToString();
  }
  auto tensor_type = type->cast<TensorTypePtr>();
  MS_EXCEPTION_IF_NULL(tensor_type);
  auto element = tensor_type->element();
  MS_EXCEPTION_IF_NULL(element);
  std::ostringstream buffer;
  if (check_list.find(element->type_id()) == check_list.end()) {
    buffer << "type of " << type_name << " should be in [";
    for (auto type_elem : check_list) {
      buffer << TypeIdToType(type_elem)->ToString() << " ,";
    }
    buffer << "], but got " << type->ToString();
    MS_EXCEPTION(TypeError) << buffer.str();
  }
}

void CheckAndConvertUtils::CheckSubClass(const std::string &type_name, const TypePtr type_,
                                         const std::set<TypePtr> &template_types, const std::string &prim_name) {
  MS_EXCEPTION_IF_NULL(type_);
  bool hit = false;
  for (auto template_type : template_types) {
    if (type_->isa<Type>()) {
      if (IsIdentidityOrSubclass(type_, template_type)) {
        hit = true;
        break;
      }
    } else if (type_->type_id() == template_type->type_id()) {
      hit = true;
      break;
    }
  }
  if (!hit) {
    std::string type_str = type_->ToString();
    std::ostringstream buffer;
    buffer << "For '" << prim_name << "', the type of `" << type_name << "` should be subclass of ";
    for (auto template_type : template_types) {
      buffer << template_type->ToString() << ",";
    }
    buffer << " but got " << type_str << ".";
    MS_EXCEPTION(TypeError) << buffer.str();
  }
}

void CheckAndConvertUtils::CheckScalarOrTensorTypesSame(const std::map<std::string, TypePtr> &args,
                                                        const std::set<TypeId> &valid_values,
                                                        const std::string &prim_name, const bool allow_mix) {
  std::vector<std::map<std::string, TypePtr>> check_results;
  for (auto &iter : args) {
    std::map<std::string, TypePtr> arg = {{iter.first, iter.second}};
    check_results.push_back(_CheckArgumentType(arg, valid_values, prim_name));
  }

  std::map<std::string, TypePtr> &arg_ = check_results[0];
  int64_t size = check_results.size();
  for (int64_t it = 1; it != size; it++) {
    arg_ = _CheckTypeSame(arg_, check_results[it], prim_name, allow_mix);
  }
}

std::map<std::string, TypePtr> CheckAndConvertUtils::_CheckArgumentType(const std::map<std::string, TypePtr> &arg,
                                                                        const std::set<TypeId> &valid_values,
                                                                        const std::string &prim_name) {
  std::string arg_key = arg.begin()->first;
  TypePtr arg_val = arg.begin()->second;

  if (arg_val->isa<TensorType>()) {
    auto arg_val_ = std::static_pointer_cast<TensorType>(arg_val);
    arg_val = arg_val_->element();
  }

  auto it = valid_values.find(arg_val->type_id());
  if (it == valid_values.end()) {
    std::ostringstream buffer;
    buffer << "For '" << prim_name << "' , the `" << arg_key << "` should be in { ";
    for (auto valid_value : valid_values) {
      buffer << TypeIdToType(valid_value)->ToString() << ",";
    }
    buffer << " },";
    buffer << "but `" << arg_key << "`"
           << "is" << arg_val->ToString() << ".";
    MS_EXCEPTION(TypeError) << buffer.str();
  }
  return arg;
}

std::map<std::string, TypePtr> CheckAndConvertUtils::_CheckTypeSame(const std::map<std::string, TypePtr> &arg1,
                                                                    const std::map<std::string, TypePtr> &arg2,
                                                                    const std::string &prim_name,
                                                                    const bool allow_mix) {
  std::string arg1_name = arg1.begin()->first;
  TypePtr arg1_type = arg1.begin()->second;
  std::string arg2_name = arg2.begin()->first;
  TypePtr arg2_type = arg2.begin()->second;
  bool except_flag = false;

  if (arg1_type->isa<TensorType>() && arg2_type->isa<TensorType>()) {
    arg1_type = std::static_pointer_cast<TensorType>(arg1_type)->element();
    arg2_type = std::static_pointer_cast<TensorType>(arg2_type)->element();
  } else if (allow_mix) {
    arg1_type = arg1_type->isa<TensorType>() ? std::static_pointer_cast<TensorType>(arg1_type)->element() : arg1_type;
    arg2_type = arg2_type->isa<TensorType>() ? std::static_pointer_cast<TensorType>(arg2_type)->element() : arg2_type;
  } else {
    except_flag = true;
  }

  if (except_flag || arg1_type->type_id() != arg2_type->type_id()) {
    std::ostringstream buffer;
    buffer << "For '" << prim_name << "'"
           << "type of "
           << "`" << arg2_name << "` should be same as "
           << "`" << arg1_name << "`,";
    buffer << "but `" << arg1_name << "` is " << arg1_type->ToString() << "and `" << arg2_name << "` is "
           << arg2_type->ToString() << ".";
    MS_EXCEPTION(TypeError) << buffer.str();
  }
  return arg1;
}

TypeId CheckAndConvertUtils::CheckTypeSame(const std::string &arg_name, const TypePtr arg_type,
                                           const std::set<TypeId> &valid_type, const std::string &prim_name) {
  if (valid_type.empty()) {
    MS_EXCEPTION(ArgumentError) << "Trying to use the function to check a empty valid_type!";
  }
  // std::set<TypeId> types_id;
  std::ostringstream buffer;
  TypeId arg_type_;
  arg_type_ = arg_type->isa<TensorType>() ? std::static_pointer_cast<TensorType>(arg_type)->generic_type_id()
                                          : arg_type->type_id();

  auto it = valid_type.find(arg_type_);
  if (it == valid_type.end()) {
    buffer << "For" << prim_name << ", the '" << arg_name << "' should be {' one of '" << valid_type.size() << "'}";
    for (auto type : valid_type) {
      buffer << "{" << TypeIdLabel(type);
    }
    buffer << "},";
    buffer << "but got " << arg_type->ToString() << ".";
    MS_EXCEPTION(TypeError) << buffer.str();
  }
  return arg_type_;
}

bool CheckAndConvertUtils::CheckIrAttrtoOpAttr(const std::string &op_type, const std::string &attr_name,
                                               ValuePtr *const value) {
  if (*value == nullptr) {
    MS_LOG(INFO) << "value is nullptr! op_type = " << op_type << ", attr_name = " << attr_name;
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
}  // namespace mindspore
