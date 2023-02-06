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

#include "ops/tuple_setitem.h"

#include <string>
#include <vector>

#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "include/common/utils/utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
class TupleSetItemInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return InferSequenceSetItem<abstract::AbstractTuple>(primitive, input_args)->BuildShape();
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return InferSequenceSetItem<abstract::AbstractTuple>(primitive, input_args)->BuildType();
  }

  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return InferSequenceSetItem<abstract::AbstractTuple>(primitive, input_args);
  }
};
MIND_API_OPERATOR_IMPL(tuple_setitem, BaseOperator);
REGISTER_PRIMITIVE_OP_INFER_IMPL(tuple_setitem, prim::kPrimTupleSetItem, TupleSetItemInfer, false);
}  // namespace ops
}  // namespace mindspore
