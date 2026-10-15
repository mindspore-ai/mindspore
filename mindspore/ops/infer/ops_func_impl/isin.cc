/**
 * Copyright 2026 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */
#include "infer/ops_func_impl/isin.h"
#include "ops/ops_func_impl/simple_infer.h"
#include "ops_utils/op_constants.h"
#include "primitive/auto_generate/gen_ops_primitive_i.h"

namespace mindspore {
namespace ops {
ShapeArray IsinFuncImpl::InferShape(const PrimitivePtr &, const InferInfoPtrList &input_infos) const {
  return {input_infos[kInputIndex0]->GetShape()};
}

std::vector<TypeId> IsinFuncImpl::InferType(const PrimitivePtr &, const InferInfoPtrList &) const {
  return {kNumberTypeBool};
}

REGISTER_SIMPLE_INFER(kNameIsin, IsinFuncImpl)
}  // namespace ops
}  // namespace mindspore
