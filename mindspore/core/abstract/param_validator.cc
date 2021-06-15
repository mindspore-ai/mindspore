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

#include "abstract/param_validator.h"

#include <algorithm>
#include <set>
#include <string>
#include <sstream>
#include <memory>
#include "utils/symbolic.h"
#include "abstract/utils.h"

namespace mindspore {
namespace abstract {
#define ABSTRACT_REPORT_NAME_DEC(abstract) constexpr char ReportNameTraits<Abstract##abstract>::name[];

ABSTRACT_REPORT_NAME_DEC(Tensor)
ABSTRACT_REPORT_NAME_DEC(Tuple)
ABSTRACT_REPORT_NAME_DEC(Scalar)
ABSTRACT_REPORT_NAME_DEC(List)
ABSTRACT_REPORT_NAME_DEC(Dictionary)
ABSTRACT_REPORT_NAME_DEC(Slice)
ABSTRACT_REPORT_NAME_DEC(Function)
ABSTRACT_REPORT_NAME_DEC(Type)
ABSTRACT_REPORT_NAME_DEC(KeywordArg)
ABSTRACT_REPORT_NAME_DEC(Class)

TypePtr CheckType(TypePtr type, const TypePtrList &accepts, const std::string &error_message_prefix) {
  bool ok = std::any_of(accepts.begin(), accepts.end(),
                        [type](const TypePtr &accept) -> bool { return IsIdentidityOrSubclass(type, accept); });
  if (ok) {
    return type;
  } else {
    MS_LOG(EXCEPTION) << error_message_prefix << accepts << " but is " << type->ToString();
  }
}

TypePtr CheckTensorDType(const AbstractTensorPtr &tensor, const TypePtrList &accepts,
                         const std::string &error_message_prefix) {
  MS_EXCEPTION_IF_NULL(tensor);
  TypePtr type = tensor->BuildType();
  MS_EXCEPTION_IF_NULL(type);
  if (!type->isa<TensorType>()) {
    MS_LOG(EXCEPTION) << error_message_prefix << "requires Tensor but got " << type->ToString();
  }
  auto elem = tensor->element();
  MS_EXCEPTION_IF_NULL(elem);
  TypePtr ele_type = elem->BuildType();
  if (ele_type == nullptr) {
    MS_LOG(EXCEPTION) << "Abstract tensor element type nullptr";
  }
  return CheckType(ele_type, accepts, error_message_prefix);
}

TypePtr CheckTensorsDTypeSame(const AbstractTensorPtrList &tensor_list, const TypePtrList &accepts,
                              const std::string &error_message_prefix) {
  if (tensor_list.empty()) {
    MS_LOG(EXCEPTION) << "Array list is empty";
  }

  auto sample_tensor = tensor_list[0];
  MS_EXCEPTION_IF_NULL(sample_tensor);
  auto sample_elem = sample_tensor->element();
  MS_EXCEPTION_IF_NULL(sample_elem);
  TypePtr sample_type = sample_elem->BuildType();
  MS_EXCEPTION_IF_NULL(sample_type);
  std::ostringstream loginfoBuffer;
  loginfoBuffer << "same type, got";
  // Check if other elements have the same type with the first element.
  for (size_t index = 1; index < tensor_list.size(); ++index) {
    MS_EXCEPTION_IF_NULL(tensor_list[index]);
    auto elem = tensor_list[index]->element();
    MS_EXCEPTION_IF_NULL(elem);
    auto a_type = elem->BuildType();
    MS_EXCEPTION_IF_NULL(a_type);
    loginfoBuffer << " " << a_type->ToString();
    if (sample_type->type_id() != a_type->type_id()) {
      MS_LOG(EXCEPTION) << "Expected type " << sample_type->ToString() << ", but got " << a_type->ToString()
                        << ", index " << index;
    }
  }
  MS_LOG(DEBUG) << error_message_prefix << loginfoBuffer.str();
  return CheckTensorDType(sample_tensor, accepts, error_message_prefix);
}

TypePtr CheckScalarType(const AbstractScalarPtr &scalar, const TypePtrList &accepts,
                        const std::string &error_message_prefix) {
  if (scalar == nullptr) {
    MS_LOG(EXCEPTION) << "Scalar nullptr";
  }
  auto type = scalar->BuildType();
  if (type == nullptr) {
    MS_LOG(EXCEPTION) << "Scalar value nullptr";
  }

  return CheckType(type, accepts, error_message_prefix);
}

ShapePtr CheckShapeSame(const std::string &op, const AbstractTensorPtr &tensor_base, const AbstractTensorPtr &tensor) {
  MS_EXCEPTION_IF_NULL(tensor_base);
  ShapePtr shape_base = tensor_base->shape();
  MS_EXCEPTION_IF_NULL(shape_base);
  MS_EXCEPTION_IF_NULL(tensor);
  ShapePtr shape = tensor->shape();
  MS_EXCEPTION_IF_NULL(shape);
  if (*shape != *shape_base) {
    MS_LOG(EXCEPTION) << op << " evaluator first arg shape " << shape->ToString()
                      << " are not consistent with second arg shape " << shape_base->ToString();
  }
  return shape_base;
}

TypePtr CheckDtypeSame(const std::string &op, const AbstractTensorPtr &tensor_base, const AbstractTensorPtr &tensor) {
  MS_EXCEPTION_IF_NULL(tensor_base);
  auto base_elem = tensor_base->element();
  MS_EXCEPTION_IF_NULL(base_elem);
  TypePtr type_base = base_elem->BuildType();
  MS_EXCEPTION_IF_NULL(tensor);
  auto tensor_elem = tensor->element();
  MS_EXCEPTION_IF_NULL(tensor_elem);
  TypePtr type = tensor_elem->BuildType();
  MS_EXCEPTION_IF_NULL(type_base);
  MS_EXCEPTION_IF_NULL(type);
  if (*type != *type_base) {
    MS_LOG(EXCEPTION) << op << " evaluator first arg dtype " << type_base->ToString()
                      << " are not consistent with second arg dtype " << type->ToString();
  }
  return type_base;
}

int64_t CheckAxis(const std::string &op, const ValuePtr &axis, int64_t minimum, int64_t max) {
  if (axis == nullptr) {
    MS_LOG(EXCEPTION) << op << " evaluator axis is null";
  }
  if (!axis->isa<Int64Imm>()) {
    MS_LOG(EXCEPTION) << op << " evaluator axis should be int64_t, but got " << axis->type_name();
  }
  int64_t axis_value = GetValue<int64_t>(axis);
  if (axis_value > max || axis_value < minimum) {
    MS_LOG(EXCEPTION) << op << " evaluator axis value should be in the range [" << minimum << ", " << max
                      << "], but get " << axis_value;
  }
  return axis_value;
}
void CheckArgsSize(const std::string &op, const mindspore::abstract::AbstractBasePtrList &args_spec_list,
                   size_t size_expect) {
  if (args_spec_list.size() != size_expect) {
    MS_LOG(EXCEPTION) << op << " input args size should be " << size_expect << ", but got " << args_spec_list.size();
  }

  for (size_t i = 0; i < size_expect; i++) {
    MS_EXCEPTION_IF_NULL(args_spec_list[i]);
  }
}

void CheckShapeAllPositive(const std::string &op, const ShapeVector &shape) {
  for (size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] < 0) {
      MS_LOG(EXCEPTION) << op << " shape element [" << i << "] must be positive integer, but got " << shape[i];
    }
  }
}

void CheckShapeAnyAndPositive(const std::string &op, const ShapeVector &shape) {
  for (size_t i = 0; i < shape.size(); ++i) {
    if ((shape[i] < 0) && (shape[i] != Shape::SHP_ANY)) {
      MS_EXCEPTION(ValueError) << op << " shape element [" << i << "] must be positive integer or SHP_ANY, but got "
                               << shape[i];
    }
  }
}

int64_t CheckAttrPositiveInt64(const std::string &op, const ValuePtr &attr, const std::string &attr_name) {
  MS_EXCEPTION_IF_NULL(attr);
  auto int64_value = attr->cast<Int64ImmPtr>();
  MS_EXCEPTION_IF_NULL(int64_value);
  int64_t attr_val = int64_value->value();
  if (attr_val <= 0) {
    MS_LOG(EXCEPTION) << op << " invalid " << attr_name << " value: " << attr_val << ", should be greater then 0";
  }
  return attr_val;
}

std::vector<int64_t> CheckAttrIntOrTuple(const std::string &op, const ValuePtr &attr, const size_t start_idx,
                                         const size_t num_element) {
  std::vector<int64_t> result;
  MS_EXCEPTION_IF_NULL(attr);
  if (attr->isa<ValueTuple>()) {
    auto tuple_attr = attr->cast<ValueTuplePtr>();
    MS_EXCEPTION_IF_NULL(tuple_attr);
    std::vector<ValuePtr> attr_vec = tuple_attr->value();
    if (start_idx > attr_vec.size() || start_idx + num_element > attr_vec.size()) {
      MS_EXCEPTION(IndexError) << op << " attr index is out of range, attr size is " << attr_vec.size()
                               << "but start idx got" << start_idx << " num element " << num_element;
    }
    auto it_start = attr_vec.begin() + start_idx;
    (void)std::transform(it_start, it_start + num_element, std::back_inserter(result),
                         [](const ValuePtr &e) -> int64_t { return GetValue<int64_t>(e); });
  } else {
    auto int64_imm = attr->cast<Int64ImmPtr>();
    MS_EXCEPTION_IF_NULL(int64_imm);
    int64_t attr_val = int64_imm->value();
    (void)result.insert(result.begin(), num_element, attr_val);
  }
  return result;
}

std::string CheckAttrStringSet(const std::string &op, const ValuePtr &attr, const std::string &attr_name,
                               const std::set<std::string> &val_set) {
  MS_EXCEPTION_IF_NULL(attr);
  auto string_attr = attr->cast<StringImmPtr>();
  MS_EXCEPTION_IF_NULL(string_attr);
  std::string attr_val = string_attr->value();
  if (val_set.find(attr_val) == val_set.end()) {
    std::ostringstream buffer;
    bool f_begin = true;
    buffer << "{";
    for (auto &x : val_set) {
      if (!f_begin) {
        buffer << ", ";
      } else {
        f_begin = false;
      }
      buffer << x;
    }
    buffer << "}";
    MS_LOG(EXCEPTION) << op << "Unsupported " << attr_name << ": " << attr_val << ". use " << buffer.str();
  }
  return attr_val;
}

void CheckRequiredArgsSize(const std::string &op, const mindspore::abstract::AbstractBasePtrList &args_spec_list,
                           size_t size_expect) {
  if (args_spec_list.size() < size_expect) {
    MS_LOG(EXCEPTION) << op << " required input args size " << size_expect << ", but got " << args_spec_list.size();
  }
  for (size_t i = 0; i < size_expect; i++) {
    MS_EXCEPTION_IF_NULL(args_spec_list[i]);
  }
}

}  // namespace abstract
}  // namespace mindspore
