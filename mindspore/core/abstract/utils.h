/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
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

#ifndef MINDSPORE_CORE_ABSTRACT_UTILS_H_
#define MINDSPORE_CORE_ABSTRACT_UTILS_H_

#include <vector>
#include <utility>
#include <memory>
#include <string>
#include "abstract/abstract_value.h"
#include "utils/any.h"
#include "utils/misc.h"
#include "utils/shape_utils.h"

namespace mindspore {
namespace abstract {
ValuePtr ValueJoin(const ValuePtr &value1, const ValuePtr &value2);
TypePtr TypeJoin(const TypePtr &type1, const TypePtr &type2);
ShapePtr ShapeJoin(const ShapePtr &shape1, const ShapePtr &shape2);

AbstractBasePtr AbstractJoin(const AbstractBasePtrList &args_spec_list);
AbstractBasePtrList AbstractJoin(const AbstractBasePtrList &spec1, const AbstractBasePtrList &spec2);

// Return an abstract value for the sensitivity of x.
// The sensitivity of a function is an Env
// The sensitivity of J(x) is x
// else self.Clone;
AbstractBasePtr SensitivityTransform(const AbstractBasePtr &spec);

TypePtr CheckTypeList(const TypePtr &predicate, const TypePtrList &args_type_list);

bool CheckType(const TypePtr &expected_type, const TypePtr &x);

int64_t GetPositiveAxis(int64_t axis_value, size_t increment);

ShapeVector BroadcastShape(ShapeVector shpx, ShapeVector shpy);

MS_CORE_API size_t TypeIdSize(const TypeId data_type);
size_t ShapeSize(const std::vector<size_t> &shape);

// Get broadcasted shape for binary element-wise operation
ShapePtr GetBroadcastShape(const std::string &op, const AbstractTensorPtr &tensor_x, const AbstractTensorPtr &tensor_y);

// Check dynamic shape routine
void CheckMinMaxShape(const ShapeVector &shape, ShapeVector *min_shape, ShapeVector *max_shape);

// Get 3rd argument for UnsortedSegmentOps' inferImpl function
int64_t GetUnsortedSegmentOpScalarArg(const AbstractBasePtrList &args_spec_list, const std::string &op_name);
AbstractBasePtr MakeAbstract(const BaseShapePtr &base_shape, const TypePtr &type);
AbstractBasePtr MakeMonadAbstract(const MonadTypePtr &type);
AbstractBasePtr MakeAbstractTensor(const ShapePtr &shape, const TypePtr &type);
}  // namespace abstract
}  // namespace mindspore
#endif  // MINDSPORE_CORE_ABSTRACT_UTILS_H_
