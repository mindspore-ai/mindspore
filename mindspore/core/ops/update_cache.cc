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

#include "abstract/dshape.h"
#include "abstract/ops/primitive_infer_map.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/src/helper.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/log_adapter.h"
#include "ops/array_ops.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
class MIND_API UpdateCacheInfer : public abstract::OpInferBase {
 public:
  // This is used for backend infer by kernel tensor.
  BaseShapePtr InferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &) const override {
    ShapeVector shape{1};
    return std::make_shared<abstract::TensorShape>(shape);
  }

  // This is used for backend infer by kernel tensor.
  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    const std::string op_name = primitive->name();
    auto input_x = CheckAndConvertUtils::CheckArgsType(op_name, input_args, 0, kObjectTypeTensorType);
    MS_EXCEPTION_IF_NULL(input_x->GetType());
    return input_x->GetType()->Clone();
  }
};

class MIND_API UpdateCache : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(UpdateCache);
  /// \brief Constructor.
  UpdateCache() : BaseOperator("UpdateCache") {}
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(UpdateCache, prim::kPrimUpdateCache, UpdateCacheInfer, false);
}  // namespace ops
}  // namespace mindspore
