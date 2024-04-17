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

#include <memory>
#include <vector>

#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/primitive_infer_map.h"
#include "ir/anf.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/src/helper.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "ops/framework_ops.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
class MIND_API EnvironCreateInfer : public abstract::OpInferBase {
 public:
  // This is used for backend infer by kernel tensor.
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    // Inputs: a tensor
    return abstract::kNoShape;
  }

  // This is used for backend infer by kernel tensor.
  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    const size_t size_expected = 2;
    CheckArgsSize(primitive->name(), input_args, size_expected);
    return std::make_shared<EnvType>();
  }

  // This is used for frontend infer by abstract. If MakeAbstract support make env type abstract, InferShapeAndType can
  // be deleted.
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    // args: None.
    CheckArgsSize(primitive->name(), input_args, 0);
    static const AbstractBasePtr abs_env =
      std::make_shared<abstract::AbstractScalar>(kValueAny, std::make_shared<EnvType>());
    return abs_env;
  }
};

class MIND_API EnvironCreate : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(EnvironCreate);
  /// \brief Constructor.
  EnvironCreate() : BaseOperator("EnvironCreate") {}
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(EnvironCreate, prim::kPrimEnvironCreate, EnvironCreateInfer, false);
}  // namespace ops
}  // namespace mindspore
