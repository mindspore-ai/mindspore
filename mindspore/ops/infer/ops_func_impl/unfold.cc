/**
 * Copyright 2026 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */
#include "infer/ops_func_impl/unfold.h"
#include <algorithm>
#include "ops/ops_func_impl/simple_infer.h"
#include "ops_utils/op_constants.h"
#include "primitive/auto_generate/gen_ops_primitive_u.h"

namespace mindspore {
namespace ops {
ShapeArray UnfoldFuncImpl::InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  auto input_shape = input_infos[kIndex0]->GetShape();
  if (input_infos[kIndex0]->IsDynamicRank()) {
    return {{abstract::TensorShape::kShapeRankAny}};
  }
  auto dimension_opt = input_infos[kIndex1]->GetScalarValue<int64_t>();
  auto size_opt = input_infos[kIndex2]->GetScalarValue<int64_t>();
  auto step_opt = input_infos[kIndex3]->GetScalarValue<int64_t>();
  if (!dimension_opt.has_value() || !size_opt.has_value() || !step_opt.has_value()) {
    return {{abstract::TensorShape::kShapeDimAny}};
  }
  auto dimension = dimension_opt.value();
  const auto rank = SizeToLong(input_shape.size());
  if (dimension < 0) {
    dimension += rank;
  }
  if (dimension < 0 || dimension >= rank) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', dimension " << dimension_opt.value()
                             << " is out of range for a tensor with " << rank << " dimensions.";
  }
  const auto size = size_opt.value();
  const auto step = step_opt.value();
  if (size < 0 || step <= 0) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', size must be non-negative and step positive.";
  }
  const auto dimension_size = input_shape[LongToSize(dimension)];
  if (dimension_size >= 0) {
    if (size > dimension_size) {
      MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', size " << size << " exceeds dimension size "
                               << dimension_size << ".";
    }
    input_shape[LongToSize(dimension)] = (dimension_size - size) / step + 1;
  }
  input_shape.push_back(size);
  return {input_shape};
}

std::vector<TypeId> UnfoldFuncImpl::InferType(const PrimitivePtr &, const InferInfoPtrList &input_infos) const {
  return {input_infos[kIndex0]->GetType()};
}

REGISTER_SIMPLE_INFER(kNameUnfold, UnfoldFuncImpl)
}  // namespace ops
}  // namespace mindspore
