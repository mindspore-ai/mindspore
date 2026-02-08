/**
 * Copyright 2019-2025 Huawei Technologies Co., Ltd
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
#include <cassert>
#include "abstract/abstract_value.h"
#include "mindapi/base/macros.h"

namespace mindspore {
namespace abstract {
MS_CORE_API TypePtr CheckTensorDType(const AbstractBasePtr &tensor, const TypePtrList &accepts,
                                     const std::string &error_message_prefix);

MS_CORE_API TypePtr CheckTensorsDTypeSame(const AbstractTensorPtrList &tensor_list, const TypePtrList &accepts,
                                          const std::string &error_message_prefix);

MS_CORE_API TypePtr CheckScalarType(const AbstractScalarPtr &scalar, const TypePtrList &accepts,
                                    const std::string &error_message_prefix);

MS_CORE_API void CheckShapeSame(const std::string &op, const AbstractTensorPtr &tensor_base,
                                const AbstractTensorPtr &tensor);

MS_CORE_API void CheckShapeSame(const std::string &op, const AbstractBasePtr &tensor_base,
                                const AbstractBasePtr &tensor);

MS_CORE_API inline void CheckDtypeSame(const std::string &op, const TypePtr &type1, const TypePtr &type2) {
  if (*type1 != *type2) {
    MS_EXCEPTION(TypeError) << "For '" << op << "', the dtype of two args should be same, but the dtype of first arg "
                            << type1->ToString() << " is not consistent with the dtype of second arg "
                            << type2->ToString();
  }
}

MS_CORE_API TypePtr CheckDtypeSame(const std::string &op, const AbstractTensorPtr &tensor_base,
                                   const AbstractTensorPtr &tensor);

MS_CORE_API TypePtr CheckDtypeSame(const std::string &op, const AbstractBasePtr &tensor_base,
                                   const AbstractBasePtr &tensor);

MS_CORE_API int64_t CheckAxis(const std::string &op, const std::string &args_name, const ValuePtr &axis, int64_t min,
                              int64_t max, const std::string &rank_name);

MS_CORE_API void CheckArgsSize(const std::string &op, const AbstractBasePtrList &args_abs_list, size_t size_expect);

MS_CORE_API void CheckShapeAnyAndPositive(const std::string &op, const ShapeVector &shape);

MS_CORE_API void CheckRequiredArgsSize(const std::string &op, const AbstractBasePtrList &args_abs_list,
                                       size_t size_expect);

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
ABSTRACT_REPORT_NAME_TRAITS(RowTensor)
ABSTRACT_REPORT_NAME_TRAITS(COOTensor)
ABSTRACT_REPORT_NAME_TRAITS(CSRTensor)
ABSTRACT_REPORT_NAME_TRAITS(MapTensor)
ABSTRACT_REPORT_NAME_TRAITS(Sequence)

template <typename T>
std::shared_ptr<T> CheckArg(const std::string &op, const AbstractBasePtrList &args_abs_list, size_t index) {
  if (index >= args_abs_list.size()) {
    MS_EXCEPTION(ValueError) << op << " evaluator args list index out of bound, size " << args_abs_list.size()
                             << ", index " << index;
  }
  auto arg = dyn_cast<T>(args_abs_list[index]);
  if (arg == nullptr) {
    MS_EXCEPTION(TypeError) << "For \'" << op << "\', input[" << index << "] should be " << ReportNameTraits<T>::name
                            << ", but got " << args_abs_list[index]->BuildType()->ToString() << ".";
  }
  return arg;
}
}  // namespace abstract
}  // namespace mindspore

#endif  // MINDSPORE_CORE_ABSTRACT_PARAM_VALIDATOR_H_
