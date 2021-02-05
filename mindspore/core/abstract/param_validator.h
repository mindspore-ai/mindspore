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

#ifndef MINDSPORE_CORE_ABSTRACT_PARAM_VALIDATOR_H_
#define MINDSPORE_CORE_ABSTRACT_PARAM_VALIDATOR_H_

#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include "abstract/abstract_value.h"
#include "abstract/utils.h"
#include "utils/any.h"
#include "ir/primitive.h"

namespace mindspore {
namespace abstract {
// check if variable's type is an instance of any of accepts or of a subclass of it.
TypePtr CheckType(TypePtr type, const TypePtrList &accepts, const std::string &error_message_prefix);

TypePtr CheckTensorDType(const AbstractTensorPtr &tensor, const TypePtrList &accepts,
                         const std::string &error_message_prefix);

TypePtr CheckTensorsDTypeSame(const AbstractTensorPtrList &tensor_list, const TypePtrList &accepts,
                              const std::string &error_message_prefix);

TypePtr CheckScalarType(const AbstractScalarPtr &scalar, const TypePtrList &accepts,
                        const std::string &error_message_prefix);

ShapePtr CheckShapeSame(const std::string &op, const AbstractTensorPtr &tensor_base, const AbstractTensorPtr &tensor);

TypePtr CheckDtypeSame(const std::string &op, const AbstractTensorPtr &tensor_base, const AbstractTensorPtr &tensor);

int64_t CheckAxis(const std::string &op, const ValuePtr &axis, int64_t min, int64_t max);

void CheckArgsSize(const std::string &op, const AbstractBasePtrList &args_spec_list, size_t size_expect);

void CheckShapeAllPositive(const std::string &op, const ShapeVector &shape);

void CheckShapeAnyAndPositive(const std::string &op, const ShapeVector &shape);

int64_t CheckAttrPositiveInt64(const std::string &op, const ValuePtr &attr, const std::string &attr_name);

std::vector<int64_t> CheckAttrIntOrTuple(const std::string &op, const ValuePtr &attr, const size_t start_idx,
                                         const size_t num_element);

std::string CheckAttrStringSet(const std::string &op, const ValuePtr &attr, const std::string &attr_name,
                               const std::set<std::string> &val_set);

void CheckRequiredArgsSize(const std::string &op, const AbstractBasePtrList &args_spec_list, size_t size_expect);

template <typename T>
struct ReportNameTraits {};

#define ABSTRACT_REPORT_NAME_TRAITS(abstract)   \
  template <>                                   \
  struct ReportNameTraits<Abstract##abstract> { \
    static constexpr char name[] = #abstract;   \
  };
ABSTRACT_REPORT_NAME_TRAITS(Tensor)
ABSTRACT_REPORT_NAME_TRAITS(Tuple)
ABSTRACT_REPORT_NAME_TRAITS(Scalar)
ABSTRACT_REPORT_NAME_TRAITS(List)
ABSTRACT_REPORT_NAME_TRAITS(Dictionary)
ABSTRACT_REPORT_NAME_TRAITS(Slice)
ABSTRACT_REPORT_NAME_TRAITS(Function)
ABSTRACT_REPORT_NAME_TRAITS(Type)
ABSTRACT_REPORT_NAME_TRAITS(KeywordArg)
ABSTRACT_REPORT_NAME_TRAITS(Class)
ABSTRACT_REPORT_NAME_TRAITS(RowTensor)
ABSTRACT_REPORT_NAME_TRAITS(SparseTensor)
ABSTRACT_REPORT_NAME_TRAITS(Sequeue)

template <typename T>
std::shared_ptr<T> CheckArg(const std::string &op, const AbstractBasePtrList &args_spec_list, size_t index) {
  if (index >= args_spec_list.size()) {
    MS_EXCEPTION(ValueError) << op << " evaluator args list index out of bound, size " << args_spec_list.size()
                             << ", index " << index;
  }
  auto arg = dyn_cast<T>(args_spec_list[index]);
  if (arg == nullptr) {
    MS_EXCEPTION(TypeError) << "Operator " << op << " input[" << index << "] should be " << ReportNameTraits<T>::name
                            << ", but got " << args_spec_list[index]->BuildType()->ToString() << ".";
  }
  return arg;
}

// check if each element in args_spec is type T, and can be joined.
template <typename T>
void CheckArgsSpec(const AbstractBasePtrList &args_list) {
  for (const auto &arg : args_list) {
    if (!arg->isa<T>()) {
      MS_EXCEPTION(TypeError) << "Expected type " << ReportNameTraits<T>::name << ", but got "
                              << arg->BuildType()->ToString() << ".";
    }
  }
  (void)AbstractJoin(args_list);
}
}  // namespace abstract
}  // namespace mindspore

#endif  // MINDSPORE_CORE_ABSTRACT_PARAM_VALIDATOR_H_
