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

#include "ops/normalize_slice.h"

#include <vector>
#include <string>
#include <algorithm>
#include <memory>

#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "include/common/utils/utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
AbstractBasePtr NormalizeSliceInferInner(const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const std::string op_name = primitive->name();
  const size_t inputs_size = 5;
  CheckArgsSize(op_name, input_args, inputs_size);
  (void)std::for_each(input_args.begin(), input_args.end(), [op_name](const AbstractBasePtr &abs) {
    if (abs->isa<abstract::AbstractScalar>()) {
      if (abs->BuildType()->type_id() != kNumberTypeInt64) {
        MS_EXCEPTION(TypeError) << "The type of input of the MakeSlice operator must be int64 bot got "
                                << abs->ToString();
      }
    }
    if (abs->isa<abstract::AbstractTensor>()) {
      (void)CheckAndConvertUtils::CheckTensorTypeValid("slice_index", abs->BuildType(), std::set{kInt64}, op_name);
    }
  });
  auto abs_any = std::make_shared<abstract::AbstractScalar>(kValueAny, kInt64);
  auto abs_tensor =
    std::make_shared<abstract::AbstractTensor>(abs_any, std::make_shared<abstract::Shape>(std::vector<int64_t>{1}));
  auto output =
    std::make_shared<abstract::AbstractTuple>(abstract::AbstractBasePtrList{abs_tensor, abs_tensor, abs_tensor});
  return output;
}
MIND_API_OPERATOR_IMPL(NormalizeSlice, BaseOperator);
class NormalizeSliceInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return NormalizeSliceInferInner(primitive, input_args)->BuildShape();
  }

  TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const override {
    return NormalizeSliceInferInner(prim, input_args)->BuildType();
  }

  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return NormalizeSliceInferInner(primitive, input_args);
  }
};
REGISTER_PRIMITIVE_OP_INFER_IMPL(NormalizeSlice, prim::kPrimNormalizeSlice, NormalizeSliceInfer, false);
}  // namespace ops
}  // namespace mindspore
