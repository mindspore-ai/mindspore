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

#include "ops/ops_func_impl/shape.h"

#include <algorithm>
#include <vector>
#include <memory>

#include "abstract/abstract_value.h"
#include "ops/ops_frontend_func_impl.h"
#include "abstract/dshape.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype.h"
#include "utils/shape_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr ShapeFuncImpl::InferShape(const PrimitivePtr &primitive,
                                       const std::vector<AbstractBasePtr> &input_args) const {
  auto shape = input_args[kIndex0]->GetShape()->GetShapeVector();
  return std::make_shared<abstract::TupleShape>(abstract::BaseShapePtrList(shape.size(), abstract::kNoShape));
}

TypePtr ShapeFuncImpl::InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const {
  auto shape = input_args[kIndex0]->GetShape()->GetShapeVector();
  return std::make_shared<Tuple>(TypePtrList(shape.size(), kInt64));
}

}  // namespace ops
}  // namespace mindspore
