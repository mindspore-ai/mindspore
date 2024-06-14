/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "ops/ops_func_impl/searchsorted.h"

#include <complex>
#include <memory>
#include <vector>
#include <map>

#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "base/float16.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/dtype/type.h"
#include "ir/primitive.h"
#include "ir/tensor.h"
#include "mindapi/base/type_id.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/framework_ops.h"
#include "ops/op_name.h"
#include "ops/op_utils.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"

namespace mindspore {
namespace ops {
namespace {
constexpr size_t kSearchSortedInputNum = 5;
}

bool CheckDimsMatched(const ShapeVector &sequence, const ShapeVector &values) {
  if (sequence.size() != values.size()) {
    return false;
  }

  for (size_t dim = 0; dim < sequence.size() - 1; ++dim) {
    if (sequence[dim] != values[dim]) {
      return false;
    }
  }

  return true;
}

abstract::BaseShapePtr SearchSortedFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                        const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  (void)CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(input_args.size()), kGreaterEqual,
                                           kSearchSortedInputNum, prim_name);
  auto sequence_shape_ptr = input_args[kInputIndex0]->GetShape();
  auto sequence_shape = sequence_shape_ptr->GetShapeVector();

  auto values_shape_ptr = input_args[kInputIndex1]->GetShape();
  auto values_shape = values_shape_ptr->GetShapeVector();

  if (MS_UNLIKELY(IsDynamicRank(sequence_shape)) || MS_UNLIKELY(IsDynamicRank(values_shape))) {
    return std::make_shared<abstract::TensorShape>(ShapeVector{abstract::TensorShape::kShapeRankAny});
  }

  auto dtype_opt = GetScalarValue<int64_t>(input_args[kInputIndex3]->GetValue());
  auto right_opt = GetScalarValue<bool>(input_args[kInputIndex4]->GetValue());

  if (!dtype_opt.has_value() || !right_opt.has_value()) {
    return std::make_shared<abstract::TensorShape>(ShapeVector{abstract::TensorShape::kShapeRankAny});
  }

  std::map<std::string, TypePtr> types;
  MS_EXCEPTION_IF_NULL(input_args[kInputIndex2]);
  TypePtr sorter_type = input_args[kInputIndex2]->GetType();

  if (!sorter_type->isa<TypeNone>()) {
    (void)types.emplace("sorter", sorter_type);
    (void)CheckAndConvertUtils::CheckTensorTypeSame(types, {kInt64}, prim_name);
  }

  if (sequence_shape.empty()) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                             << "', empty shape is not permitted in 'sorted_sequence' inputs. "
                             << "The shape of 'sorted_sequence': " << sequence_shape_ptr->ToString() << ".";
  }

  if (values_shape.empty() && sequence_shape.size() != 1) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                             << "', 'values' can be a scalar only when the dimension of 'sorted_sequence' is 1. "
                             << " but we got shape of 'sorted_sequence': " << sequence_shape_ptr->ToString() << ".";
  }

  if (!(IsDynamic(sequence_shape) || IsDynamic(values_shape)) && sequence_shape.size() != 1 &&
      !CheckDimsMatched(sequence_shape, values_shape)) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', the 'sorted_sequence' must be 1 dimensional or "
                             << "all dimensions except the last dimension of 'sorted_sequence' "
                             << "must be the same as all dimensions except the last dimension of 'values', "
                             << "but got shape of 'sorted_sequence': " << sequence_shape_ptr->ToString()
                             << " and shape of 'values': " << values_shape_ptr->ToString() << ".";
  }
  (void)CheckAndConvertUtils::CheckArgsType(prim_name, input_args, kInputIndex0, kObjectTypeTensorType);
  (void)CheckAndConvertUtils::CheckArgsType(prim_name, input_args, kInputIndex1, kObjectTypeTensorType);
  MS_EXCEPTION_IF_NULL(values_shape_ptr);
  auto shape_element = values_shape_ptr->cast<abstract::ShapePtr>();
  MS_EXCEPTION_IF_NULL(shape_element);
  return shape_element;
}

TypePtr SearchSortedFuncImpl::InferType(const PrimitivePtr &primitive,
                                        const std::vector<AbstractBasePtr> &input_args) const {
  auto dtype_ptr = GetScalarValue<int64_t>(input_args[kInputIndex3]->GetValue());
  MS_CHECK_VALUE(dtype_ptr.has_value(), primitive->name() + " error: dtype input should has valid value.");
  auto type = TypeIdToType(static_cast<TypeId>(dtype_ptr.value()));
  MS_CHECK_VALUE(type == kInt32 || type == kInt64, primitive->name() + " error: dtype should be " + kInt32->ToString() +
                                                     " or " + kInt64->ToString() + " but got " + type->ToString());

  return std::make_shared<TensorType>(type);
}
}  // namespace ops
}  // namespace mindspore
