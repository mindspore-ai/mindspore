/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "ops/sequence_add_offset.h"

#include <vector>
#include <string>
#include <memory>

#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "include/common/utils/utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
AbstractBasePtr SequenceAddOffsetInferInner(const PrimitivePtr &primitive,
                                            const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  constexpr size_t input_len = 2;
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, input_len, prim_name);
  abstract::AbstractBasePtrList abs{};
  abs.push_back(std::make_shared<abstract::AbstractScalar>(kAnyValue, kInt64));
  abs.push_back(std::make_shared<abstract::AbstractScalar>(kAnyValue, kInt64));
  auto ret = std::make_shared<abstract::AbstractTuple>(abs);
  return ret;
}

MIND_API_OPERATOR_IMPL(SequenceAddOffset, BaseOperator);

class SequenceAddOffsetInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return SequenceAddOffsetInferInner(primitive, input_args)->BuildShape();
  }

  TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const override {
    return SequenceAddOffsetInferInner(prim, input_args)->BuildType();
  }

  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return SequenceAddOffsetInferInner(primitive, input_args);
  }
};
REGISTER_PRIMITIVE_OP_INFER_IMPL(SequenceAddOffset, prim::kPrimSequenceAddOffset, SequenceAddOffsetInfer, true);
}  // namespace ops
}  // namespace mindspore
