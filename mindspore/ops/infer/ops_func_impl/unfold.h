/**
 * Copyright 2026 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */
#ifndef MINDSPORE_CORE_OPS_OPS_FUNC_IMPL_UNFOLD_H_
#define MINDSPORE_CORE_OPS_OPS_FUNC_IMPL_UNFOLD_H_

#include <vector>
#include "ops/ops_func_impl/op_func_impl.h"

namespace mindspore {
namespace ops {
class OPS_API UnfoldFuncImpl : public OpFuncImpl {
 public:
  ShapeArray InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override;
  std::vector<TypeId> InferType(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override;
  bool GeneralInferRegistered() const override { return true; }
};
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_OPS_FUNC_IMPL_UNFOLD_H_
