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
#include <set>
#include <map>
#include <iterator>

#include "abstract/abstract_value.h"
#include "ir/dtype/type.h"
#include "ir/dtype/tensor_type.h"
#include "ir/dtype.h"
#include "utils/ms_context.h"
#include "ir/primitive.h"
#include "ir/scalar.h"
#include "ir/tensor.h"
#include "ir/value.h"
#include "mindapi/base/format.h"
#include "mindapi/base/type_id.h"
#include "mindapi/base/types.h"
#include "ops/op_name.h"
#include "utils/convert_utils_base.h"

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
  // this should be removed, cause some op change "PAD" to "CALCULATED" in python.
  {"CALCULATED", PadMode::PAD},
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
  {"MaxPoolV1", FormatAndPadUpperAttrMap},
  {"MaxPool", FormatAndPadUpperAttrMap},
  {"MaxPoolWithArgmax", FormatAndPadUpperAttrMap},
  {"AvgPoolGrad", FormatAndPadUpperAttrMap},
  {"AvgPoolGradVm", FormatAndPadUpperAttrMap},
  {"AvgPoolGradGpu", FormatAndPadUpperAttrMap},
  {"AvgPoolGradCpu", FormatAndPadUpperAttrMap},
  {"AvgPoolV1", FormatAndPadUpperAttrMap},
  {"AvgPoolGradV1", FormatAndPadUpperAttrMap},
  {"MaxPoolGrad", FormatAndPadUpperAttrMap},
  {"MaxPoolGradV1", FormatAndPadUpperAttrMap},
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
  {"NLLLossGrad", ReductionMap},
  {"DepthToSpace", DataFormatMap},
  {"Pooling", DataFormatMap},
  {"Deconvolution", DataFormatMap},
  {"AvgPoolV2", DataFormatMap},
  {"MaxPoolV3", DataFormatMap},
  {"FusedBatchNorm", DataFormatMap}};

bool CheckAndConvertUtils::GetDataFormatEnumValue(const ValuePtr &value, int64_t *enum_value) {
  MS_EXCEPTION_IF_NULL(value);
  if (!value->isa<StringImm>()) {
    *enum_value = GetValue<int64_t>(value);
    return true;
  }
  auto attr_value_str = GetValue<std::string>(value);
  auto iter = DataFormatToEnumMap.find(attr_value_str);
  if (iter == DataFormatToEnumMap.end()) {
    MS_LOG(DEBUG) << "The data format " << attr_value_str << " not be converted to enum.";
    return false;
  }
  *enum_value = iter->second;
  return true;
}

void CheckAndConvertUtils::GetPadModEnumValue(const ValuePtr &value, int64_t *enum_value, bool is_upper) {
  MS_EXCEPTION_IF_NULL(value);
  if (!value->isa<StringImm>()) {
    *enum_value = GetValue<int64_t>(value);
    return;
  }
  auto attr_value_str = GetValue<std::string>(value);

  if (is_upper) {
    auto iter = PadModToEnumUpperMap.find(attr_value_str);
    if (iter == PadModToEnumUpperMap.end()) {
      MS_LOG(EXCEPTION) << "Invalid pad mode " << attr_value_str << " use pad, valid or same";
    }
    *enum_value = iter->second;
    return;
  }
  auto iter = PadModToEnumMap.find(attr_value_str);
  if (iter == PadModToEnumMap.end()) {
    MS_LOG(EXCEPTION) << "Invalid pad mode " << attr_value_str << " use pad, valid or same";
  }
  *enum_value = iter->second;
}

void CheckAndConvertUtils::GetReductionEnumValue(const ValuePtr &value, int64_t *enum_value) {
  MS_EXCEPTION_IF_NULL(value);
  if (!value->isa<StringImm>()) {
    *enum_value = GetValue<int64_t>(value);
    return;
  }
  auto attr_value_str = GetValue<std::string>(value);
  auto iter = ReductionToEnumMap.find(attr_value_str);
  if (iter == ReductionToEnumMap.end()) {
    MS_LOG(EXCEPTION) << "Invalid pad mode " << attr_value_str << " use pad, valid or same";
  }
  *enum_value = iter->second;
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
  if (attr_map_pair.first.empty()) {
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
  if (attr_map_pair.second.empty()) {
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

void CheckAndConvertUtils::GetFormatStringVal(const PrimitivePtr &prim, std::string *format) {
  if (prim == nullptr || format == nullptr) {
    MS_LOG(DEBUG) << "Prim or format is nullptr.";
    return;
  }
  auto value_ptr = prim->GetAttr(ops::kFormat);
  if (value_ptr == nullptr) {
    MS_LOG(DEBUG) << "Val is nullptr! op type = " << prim->name();
    return;
  }
  int64_t data_format;
  bool result = CheckAndConvertUtils::GetDataFormatEnumValue(value_ptr, &data_format);
  if (result) {
    if (DataFormatToStrMap.find(data_format) != DataFormatToStrMap.end()) {
      *format = DataFormatToStrMap.at(data_format);
    }
  }
}

size_t CheckAndConvertUtils::CheckAbstractShapeSame(const std::vector<AbstractBasePtr> &abs_list) {
  if (abs_list.size() <= 1) {
    return 0;
  }
  const auto &first_elem_abs = abs_list[0];
  MS_EXCEPTION_IF_NULL(first_elem_abs);
  auto abs1_shape = first_elem_abs->BuildShape();
  MS_EXCEPTION_IF_NULL(abs1_shape);
  for (size_t i = 0; i < abs_list.size(); ++i) {
    MS_EXCEPTION_IF_NULL(abs_list[i]);
    auto abs2_shape = abs_list[i]->BuildShape();
    MS_EXCEPTION_IF_NULL(abs2_shape);
    if (*abs1_shape != *abs2_shape) {
      MS_LOG(ERROR) << "Abstract shapes are not same, shape1:" << abs1_shape->ToString()
                    << ", shape2:" << abs2_shape->ToString();
      return i;
    }
  }
  return 0;
}

// For example,
// TensorType(element type is float16) and TensorType(element type is int32) are not same,
// TupleType(elements num is 3) and TupleType(elements num is 4) are same.
size_t CheckAndConvertUtils::CheckAbstractTypeSame(const std::vector<AbstractBasePtr> &abs_list) {
  if (abs_list.size() <= 1) {
    return 0;
  }
  const auto &first_elem_abs = abs_list[0];
  MS_EXCEPTION_IF_NULL(first_elem_abs);
  auto abs1_type = first_elem_abs->BuildType();
  MS_EXCEPTION_IF_NULL(abs1_type);
  for (size_t i = 1; i < abs_list.size(); ++i) {
    MS_EXCEPTION_IF_NULL(abs_list[i]);
    auto abs2_type = abs_list[i]->BuildType();
    MS_EXCEPTION_IF_NULL(abs2_type);
    if (!(*abs1_type == *abs2_type)) {
      MS_LOG(ERROR) << "Abstract types are not same, type1:" << abs1_type->ToString()
                    << ", type2:" << abs2_type->ToString();
      return i;
    }
  }
  return 0;
}

void CheckAndConvertUtils::CheckAbstractTypeAndShapeSame(const std::vector<AbstractBasePtr> &abs_list,
                                                         const std::string &precondition_log,
                                                         const std::string &standard_abs_description,
                                                         const std::string &differ_abs_description) {
  if (abs_list.size() <= 1) {
    return;
  }
  auto differ_index = CheckAndConvertUtils::CheckAbstractTypeSame(abs_list);
  if (differ_index == 0) {
    differ_index = CheckAndConvertUtils::CheckAbstractShapeSame(abs_list);
  }
  if (differ_index != 0) {
    auto log_info1 = standard_abs_description.empty() ? "sequence[0] item" : standard_abs_description;
    auto log_info2 =
      differ_abs_description.empty() ? "sequence[" + std::to_string(differ_index) + "] item" : differ_abs_description;
    MS_EXCEPTION(TypeError) << precondition_log << ", the " << log_info1 << " abstract '" << abs_list[0]->ToString()
                            << "' is not same with the " << log_info2 << " abstract '"
                            << abs_list[differ_index]->ToString() << "'.";
  }
}

abstract::AbstractSequencePtr CheckAndConvertUtils::BroadenAllSequenceElements(
  const abstract::AbstractSequencePtr &sequence) {
  MS_EXCEPTION_IF_NULL(sequence);
  const auto &elements = sequence->elements();
  AbstractBasePtrList new_elements;
  for (auto element : elements) {
    AbstractBasePtr new_element = nullptr;
    if (element->isa<abstract::AbstractSequence>()) {
      new_element = BroadenAllSequenceElements(element->cast<abstract::AbstractSequencePtr>());
    } else {
      auto tmp_element = element->Clone();
      if (element->isa<abstract::AbstractScalar>()) {
        tmp_element->cast<abstract::AbstractScalarPtr>()->set_is_variable(true);
      }
      new_element = tmp_element->Broaden();
    }
    new_elements.push_back(new_element);
  }
  if (sequence->isa<abstract::AbstractList>()) {
    return std::make_shared<abstract::AbstractList>(new_elements, sequence->sequence_nodes());
  }
  return std::make_shared<abstract::AbstractTuple>(new_elements, sequence->sequence_nodes());
}

bool CheckAndConvertUtils::CheckValueSame(const ValuePtr &value_1, const ValuePtr &value_2) {
  MS_EXCEPTION_IF_NULL(value_1);
  MS_EXCEPTION_IF_NULL(value_2);
  if (!value_1->IsSameTypeId(value_2->tid())) {
    return false;
  }
  if (value_1->isa<tensor::Tensor>()) {
    auto list_tensor_value = value_2->cast_ptr<tensor::Tensor>();
    return value_1->cast_ptr<tensor::Tensor>()->ValueEqual(*list_tensor_value);
  }
  return *value_1 == *value_2;
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
inline bool CheckType(const TypePtr &check_type, const std::set<TypePtr> &template_types) {
  return std::any_of(template_types.begin(), template_types.end(), [&check_type](const TypePtr &accept) -> bool {
    return IsIdentidityOrSubclass(check_type, accept);
  });
}
}  // namespace

std::string CheckAndConvertUtils::CheckString(const std::string &arg_name, const std::string &arg_value,
                                              const std::set<std::string> &check_list, const std::string &prim_name) {
  if (check_list.find(arg_value) != check_list.end()) {
    return arg_value;
  }
  std::ostringstream buffer;
  buffer << "For primitive[" << prim_name << "], the attribute[" << arg_name << "]";
  if (check_list.size() == 1) {
    buffer << " must be \"" << (*check_list.begin()) << "\", but got \"" << arg_value << "\".";
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
    buffer << "For primitive[" << prim_name << "], the " << arg_name << " must ";
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

ShapeMap CheckAndConvertUtils::ConvertShapePtrToShapeMap(const BaseShapePtr &shape) {
  MS_EXCEPTION_IF_NULL(shape);
  if (!shape->isa<abstract::Shape>()) {
    return std::map<std::string, std::vector<int64_t>>();
  }
  auto shape_element = shape->cast<abstract::ShapePtr>();
  MS_EXCEPTION_IF_NULL(shape_element);
  ShapeMap shape_map;
  shape_map[kShape] = shape_element->shape();
  shape_map[kMaxShape] = shape_element->max_shape();
  return shape_map;
}

abstract::ShapePtr CheckAndConvertUtils::GetTensorInputShape(const std::string &prim_name,
                                                             const std::vector<AbstractBasePtr> &input_args,
                                                             size_t index) {
  auto abstract = CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, index);
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

TypePtr CheckAndConvertUtils::GetTensorInputType(const std::string &prim_name,
                                                 const std::vector<AbstractBasePtr> &input_args, size_t index) {
  if (input_args.size() <= index) {
    MS_EXCEPTION(ValueError) << "For " << prim_name << ", the index " << index << " is out of the input number "
                             << input_args.size();
  }
  auto input_arg = input_args[index];
  if (input_arg == nullptr) {
    MS_EXCEPTION(ValueError) << "The " << index << "'s input of " << prim_name << " is nullptr.";
  }
  auto base_type = input_arg->BuildType();
  MS_EXCEPTION_IF_NULL(base_type);
  if (!base_type->isa<TensorType>()) {
    MS_EXCEPTION(TypeError) << "The " << index << "'s input type of " << prim_name << " is not Tensor.";
  }
  auto tensor_type = base_type->cast<TensorTypePtr>();
  MS_EXCEPTION_IF_NULL(tensor_type);
  auto type = tensor_type->element();
  MS_EXCEPTION_IF_NULL(type);
  return type;
}

void CheckAndConvertUtils::Check(const string &arg_name, int64_t arg_value, CompareEnum compare_type, int64_t value,
                                 const string &prim_name, ExceptionType) {
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
    buffer << "For primitive[" << prim_name << "], the attribute[" << arg_name << "] must ";
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
  // Check Input type is tensor type
  for (const auto &item : types) {
    auto type = item.second;
    MS_EXCEPTION_IF_NULL(type);
    if (!type->isa<TensorType>()) {
      size_t i = 1;
      std::ostringstream buffer;
      buffer << "The primitive[" << prim_name << "]'s input arguments[";
      for (const auto &item_type : types) {
        buffer << item_type.first;
        if (i < types.size()) {
          buffer << ", ";
          ++i;
        }
      }
      i = 1;
      buffer << "] must be all tensor and those type must be same.";
      for (const auto &type_info : types) {
        if (!type_info.second->isa<TensorType>()) {
          buffer << " But got input argument[" << type_info.first << "]"
                 << ":" << type_info.second->ToString() << "\n";
        }
      }
      if (!check_list.empty()) {
        buffer << "Valid type list: {";
        std::set<string> order_set;
        for (auto const &valid_type : check_list) {
          if (valid_type->isa<TensorType>()) {
            (void)order_set.emplace(valid_type->ToString());
            break;
          } else {
            (void)order_set.emplace("Tensor[" + valid_type->ToString() + "]");
          }
        }
        for (auto const &error_item : order_set) {
          buffer << error_item;
          if (error_item != *(--order_set.end())) {
            buffer << ", ";
          }
        }
        buffer << "}.";
      }
      MS_EXCEPTION(TypeError) << buffer.str();
    }
  }
  (void)_CheckTypeSame(types, prim_name, false);
  return CheckTensorSubClass(types.begin()->first, types.begin()->second, check_list, prim_name);
}

TypePtr CheckAndConvertUtils::CheckMathBinaryOpTensorType(const std::map<std::string, TypePtr> &types,
                                                          const std::set<TypePtr> &check_list,
                                                          const std::string &prim_name) {
  constexpr size_t n = 2;
  if (types.size() != n) {
    MS_EXCEPTION(ArgumentError) << "For primitive[" << prim_name << "], the size of types to check must be " << n
                                << ", but got " << types.size();
  }
  // Check Input type is tensor type
  std::vector<TypeId> type_ids;
  std::vector<TypePtr> type_ptr;
  bool has_complex = false;
  for (const auto &item : types) {
    MS_EXCEPTION_IF_NULL(item.second);
    if (!item.second->isa<TensorType>()) {
      MS_EXCEPTION(TypeError) << "The primitive[" << prim_name << "]'s input arguments[" << item.first
                              << "] must be Tensor, but got " << item.second->ToString();
    }
    auto tensor_type = item.second->cast<TensorTypePtr>();
    MS_EXCEPTION_IF_NULL(tensor_type);
    auto element = tensor_type->element();
    MS_EXCEPTION_IF_NULL(element);
    auto type_id = element->type_id();
    if (!has_complex && (type_id == kNumberTypeComplex64 || type_id == kNumberTypeComplex128)) {
      has_complex = true;
    }
    type_ids.push_back(type_id);
    type_ptr.push_back(item.second);
  }
  // Deal with complex data type
  if (has_complex) {
    static std::map<std::pair<TypeId, TypeId>, TypeId> type_infer_dict = {
      {{kNumberTypeComplex64, kNumberTypeComplex64}, kNumberTypeComplex64},
      {{kNumberTypeComplex64, kNumberTypeFloat32}, kNumberTypeComplex64},
      {{kNumberTypeFloat32, kNumberTypeComplex64}, kNumberTypeComplex64},
      {{kNumberTypeComplex128, kNumberTypeComplex128}, kNumberTypeComplex128},
      {{kNumberTypeComplex128, kNumberTypeFloat64}, kNumberTypeComplex128},
      {{kNumberTypeFloat64, kNumberTypeComplex128}, kNumberTypeComplex128}};
    std::pair<TypeId, TypeId> type_info(type_ids[0], type_ids[1]);
    auto iter = type_infer_dict.find(type_info);
    if (iter != type_infer_dict.end()) {
      return type_ids[0] == iter->second ? type_ptr[0] : type_ptr[1];
    }
    std::ostringstream buffer;
    buffer << "For primitive[" << prim_name << "], complex math binary op expecting Tensor";
    for (const auto &items : type_infer_dict) {
      buffer << "[" << TypeIdToString(items.first.first) << ", " << TypeIdToString(items.first.second) << "], ";
    }
    buffer << "but got Tensor[" << TypeIdToString(type_ids[0]) << ", " << TypeIdToString(type_ids[1]) << "]";
    MS_EXCEPTION(TypeError) << buffer.str();
  }
  // Deal with non-complex data type
  if (type_ids[0] != type_ids[1]) {
    MS_EXCEPTION(TypeError) << "For primitive[" << prim_name
                            << "], the input arguments must have same data type, but got Tensor["
                            << TypeIdToString(type_ids[0]) << "] and Tensor[" << TypeIdToString(type_ids[1]) << "]";
  }
  (void)CheckTensorSubClass(types.begin()->first, types.begin()->second, check_list, prim_name);
  return types.begin()->second;
}

ShapeVector CheckAndConvertUtils::CheckTensorShapeSame(const std::map<std::string, BaseShapePtr> &shapes,
                                                       const std::vector<int64_t> &check_shape,
                                                       const std::string &prim_name) {
  if (shapes.empty()) {
    MS_EXCEPTION(ArgumentError) << "Trying to use the function to check a empty shapes map!";
  }
  for (const auto &shape : shapes) {
    auto _shape_ptr_ = shape.second;
    MS_EXCEPTION_IF_NULL(_shape_ptr_);
    auto _shape_ = ConvertShapePtrToShapeMap(_shape_ptr_)[kShape];
    (void)CheckPositiveVectorExcludeZero(shape.first, _shape_, prim_name);
    if (!ShapeVectorIsSame(_shape_, check_shape)) {
      std::ostringstream buffer;
      buffer << "The primitive[" << prim_name << "]'s input arguments " << shape.first << " shape should equal to "
             << ShapeVectorToStr(check_shape) << ", but get the real shape " << ShapeVectorToStr(_shape_) << ".";
      MS_EXCEPTION(ValueError) << buffer.str();
    }
  }
  return check_shape;
}

TypePtr CheckAndConvertUtils::CheckTensorTypeValid(const std::string &type_name, const TypePtr &type,
                                                   const std::set<TypePtr> &check_list, const std::string &prim_name) {
  // note that the return type might be different from input type
  MS_EXCEPTION_IF_NULL(type);
  if (!type->isa<TensorType>()) {
    MS_EXCEPTION(TypeError) << "For Primitive[" << prim_name << "], the input argument[" << type_name
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
  return CheckTensorSubClass(type_name, type, check_list, prim_name);
}

TypePtr CheckAndConvertUtils::CheckSparseTensorTypeValid(const std::string &type_name, const TypePtr &type,
                                                         const std::set<TypePtr> &, const std::string &prim_name) {
  MS_EXCEPTION_IF_NULL(type);
  if (!type->isa<SparseTensorType>()) {
    MS_EXCEPTION(TypeError) << "For Primitive[" << prim_name << "], the input argument[" << type_name
                            << "] must be a CSRTensor or COOTensor, but got " << type->ToString() << ".";
  } else {
    auto sparse_type = type->cast<SparseTensorTypePtr>();
    if (sparse_type != nullptr) {
      return sparse_type->element_type();
    }
    MS_EXCEPTION(TypeError) << "For Primitive[" << prim_name << "], the input argument[" << type_name
                            << "] cast to SparseTensorTypePtr failed! Get type : " << type->ToString() << ".";
  }
}

ShapeVector CheckAndConvertUtils::CheckTensorIntValue(const std::string &type_name, const ValuePtr &value,
                                                      const std::string &prim_name) {
  if (value == nullptr) {
    MS_EXCEPTION(ValueError) << "For primitive[" << prim_name << "], the input argument[" << type_name
                             << "] value is nullptr.";
  }
  ShapeVector tensor_value;
  if (!value->isa<tensor::Tensor>()) {
    MS_EXCEPTION(ValueError) << "For primitive[" << prim_name << "], the input argument[" << type_name
                             << "] must be a tensor, but got " << value->ToString();
  }
  auto input_tensor = value->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(input_tensor);
  size_t data_size = input_tensor->DataSize();
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
  } else if (tensor_type->type_id() == kNumberTypeUInt32) {
    auto tensor_data = reinterpret_cast<uint32_t *>(input_tensor->data_c());
    MS_EXCEPTION_IF_NULL(tensor_data);
    tensor_value = {tensor_data, tensor_data + data_size};
  } else if (tensor_type->type_id() == kNumberTypeUInt64) {
    auto tensor_data = reinterpret_cast<uint64_t *>(input_tensor->data_c());
    MS_EXCEPTION_IF_NULL(tensor_data);
    tensor_value = {tensor_data, tensor_data + data_size};
  } else {
    MS_EXCEPTION(TypeError) << "For primitive[" << prim_name << "], the input argument[" << type_name
                            << "] must be a Tensor[Int64] or Tensor[Int32]"
                            << " or Tensor[UInt64] or Tensor[UInt32] type, but got " << value->ToString();
  }
  return tensor_value;
}

TypePtr CheckAndConvertUtils::CheckTensorSubClass(const string &type_name, const TypePtr &type,
                                                  const std::set<TypePtr> &template_types, const string &prim_name,
                                                  bool is_mix) {
  MS_EXCEPTION_IF_NULL(type);
  auto real_type = type;
  if (type->isa<TensorType>()) {
    auto tensor_type = type->cast<TensorTypePtr>();
    real_type = tensor_type->element();
  }
  if (CheckType(real_type, template_types)) {
    return real_type;
  }
  std::ostringstream buffer;
  buffer << "For primitive[" << prim_name << "], the input argument[" << type_name << "] must be a type of {";
  std::set<string> order_set;

  if (is_mix) {
    for (const auto &item : template_types) {
      (void)order_set.emplace(item->ToString());
    }
  }

  for (const auto &item : template_types) {
    if (item->isa<TensorType>()) {
      (void)order_set.emplace(item->ToString());
      continue;
    }
    (void)order_set.emplace("Tensor[" + item->ToString() + "]");
  }

  for (const auto &item : order_set) {
    buffer << item;
    if (item != *(--order_set.end())) {
      buffer << ", ";
    }
  }
  buffer << "}, but got " << type->ToString();
  buffer << ".";
  MS_EXCEPTION(TypeError) << buffer.str();
}

TypePtr CheckAndConvertUtils::CheckSubClass(const std::string &type_name, const TypePtr &type,
                                            const std::set<TypePtr> &template_types, const std::string &prim_name) {
  if (CheckType(type, template_types)) {
    return type;
  }
  std::ostringstream buffer;
  buffer << "For primitive[" << prim_name << "], the input argument[" << type_name << "] must be a type of {";
  std::set<string> order_set;
  for (const auto &item : template_types) {
    (void)order_set.emplace(item->ToString());
  }
  for (const auto &item : order_set) {
    buffer << item;
    if (item != *(--order_set.end())) {
      buffer << ", ";
    }
  }
  buffer << "}, but got " << type->ToString();
  buffer << ".";
  MS_EXCEPTION(TypeError) << buffer.str();
}

TypePtr CheckAndConvertUtils::CheckScalarOrTensorTypesSame(const std::map<std::string, TypePtr> &args,
                                                           const std::set<TypePtr> &valid_values,
                                                           const std::string &prim_name, bool allow_mix) {
  (void)_CheckTypeSame(args, prim_name, allow_mix);
  return CheckTensorSubClass(args.begin()->first, args.begin()->second, valid_values, prim_name, true);
}

TypePtr CheckAndConvertUtils::_CheckTypeSame(const std::map<std::string, TypePtr> &args, const std::string &prim_name,
                                             const bool allow_mix) {
  if (args.empty()) {
    MS_EXCEPTION(ArgumentError) << "Trying to use the function to check a empty types map!";
  }
  std::ostringstream buffer;
  TypePtr return_type = args.begin()->second;
  buffer << "For primitive[" << prim_name << "], the ";
  bool tensor_flag = return_type->isa<TensorType>();
  std::set<TypeId> types_id;
  for (const auto &elem : args) {
    auto type = elem.second;
    MS_EXCEPTION_IF_NULL(type);
    if (!allow_mix) {
      // input must be all tensor or all other type
      if ((tensor_flag && !type->isa<TensorType>()) || (!tensor_flag && type->isa<TensorType>())) {
        buffer << "input type must be same.\n";
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
      return_type = element;
      (void)types_id.emplace(element->type_id());
    } else {
      if (return_type->isa<TensorType>()) {
        return_type = type;
      }
      (void)types_id.emplace(type->type_id());
    }
    if (types_id.size() > 1) {
      buffer << "input type must be same.\n";
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
    MS_EXCEPTION(ValueError) << "For primitive[" << class_name << "], the input argument[name]"
                             << " cannot be an empty string.";
  }
  (void)CheckTypeValid("value", value->BuildType(), {kTensorType}, class_name);
}

void CheckAndConvertUtils::CheckMode(const std::string &class_name) {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (ms_context->get_param<int>(MS_CTX_EXECUTION_MODE) == kPynativeMode) {
    MS_EXCEPTION(NotSupportError) << "The primitive[" << class_name << "] does not support PyNativeMode.\n"
                                  << "Please convert the mode to GraphMode.";
  }
}

std::vector<int64_t> CheckAndConvertUtils::CheckIntOrTupleInt(const std::string &arg_name, const ValuePtr &attr,
                                                              const std::string &prim_name) {
  std::vector<int64_t> result;
  bool is_correct = false;
  MS_EXCEPTION_IF_NULL(attr);
  if (attr->isa<ValueTuple>() || attr->isa<ValueList>()) {
    auto attr_vec =
      attr->isa<ValueTuple>() ? attr->cast<ValueTuplePtr>()->value() : attr->cast<ValueListPtr>()->value();
    if (!attr_vec.size()) {
      return result;
    }
    is_correct = std::all_of(attr_vec.begin(), attr_vec.end(), [&result](const ValuePtr &e) -> bool {
      MS_EXCEPTION_IF_NULL(e);
      if (e->isa<Int64Imm>()) {
        (void)result.emplace_back(GetValue<int64_t>(e));
        return true;
      } else if (e->isa<Int32Imm>()) {
        (void)result.emplace_back(GetValue<int32_t>(e));
        return true;
      }
      return false;
    });
  } else {
    if (attr->isa<Int64Imm>()) {
      is_correct = true;
      int64_t attr_val = attr->cast<Int64ImmPtr>()->value();
      result.push_back(attr_val);
    } else if (attr->isa<Int32Imm>()) {
      is_correct = true;
      int64_t attr_val = attr->cast<Int32ImmPtr>()->value();
      result.push_back(attr_val);
    }
  }
  if (!is_correct) {
    MS_EXCEPTION(TypeError) << "For primitive[" << prim_name << "], the " << arg_name
                            << " must be one of ['int', 'tuple', 'list'] with all Int elements, but got "
                            << attr->ToString();
  }
  return result;
}

std::vector<int64_t> CheckAndConvertUtils::CheckTupleInt(const std::string &arg_name, const ValuePtr &attr,
                                                         const std::string &prim_name) {
  std::vector<int64_t> result;
  MS_EXCEPTION_IF_NULL(attr);
  if (attr->isa<ValueTuple>()) {
    std::vector<ValuePtr> attr_vec = attr->cast<ValueTuplePtr>()->value();
    (void)std::transform(
      attr_vec.begin(), attr_vec.end(), std::back_inserter(result), [=](const ValuePtr &e) -> int64_t {
        if (!e->isa<Int64Imm>()) {
          MS_EXCEPTION(TypeError) << "For primitive[" << prim_name << "], the " << arg_name
                                  << " must be a tuple with all Int elements, but got " << attr->type_name();
        }
        return GetValue<int64_t>(e);
      });
  } else {
    MS_EXCEPTION(TypeError) << "For primitive[" << prim_name << "], the " << arg_name
                            << " must be a tuple with all Int elements, but got " << attr->type_name() << ".";
  }
  return result;
}

std::vector<int64_t> CheckAndConvertUtils::CheckListInt(const std::string &arg_name, const ValuePtr &attr,
                                                        const std::string &prim_name) {
  std::vector<int64_t> result;
  MS_EXCEPTION_IF_NULL(attr);
  if (attr->isa<ValueList>()) {
    std::vector<ValuePtr> attr_vec = attr->cast<ValueListPtr>()->value();
    (void)std::transform(
      attr_vec.begin(), attr_vec.end(), std::back_inserter(result), [=](const ValuePtr &e) -> int64_t {
        if (!e->isa<Int64Imm>()) {
          MS_EXCEPTION(TypeError) << "For primitive[" << prim_name << "], the " << arg_name
                                  << " must be a list with all Int elements, but got " << attr->ToString();
        }
        return GetValue<int64_t>(e);
      });
  } else {
    MS_EXCEPTION(TypeError) << "For primitive[" << prim_name << "], the " << arg_name
                            << " must be a list with all Int elements, but got " << attr->ToString() << ".";
  }
  return result;
}

int64_t CheckAndConvertUtils::GetAndCheckFormat(const ValuePtr &value) {
  int64_t data_format;
  bool result = CheckAndConvertUtils::GetDataFormatEnumValue(value, &data_format);
  if (!result ||
      (data_format != static_cast<int64_t>(Format::NHWC) && data_format != static_cast<int64_t>(Format::NCHW) &&
       data_format != static_cast<int64_t>(Format::NCDHW))) {
    MS_LOG(EXCEPTION) << "data format value " << data_format << " is invalid, only support NCHW, NHWC and NCDHW";
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
}  // namespace mindspore
