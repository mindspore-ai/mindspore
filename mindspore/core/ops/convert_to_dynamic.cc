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

#include "ops/convert_to_dynamic.h"

#include <memory>
#include <vector>

#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/param_validator.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/primitive.h"
#include "ir/scalar.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/core_ops.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
constexpr auto kIsDynamicRank = "is_dynamic_rank";

void CheckConvertToDynamicRankArgs(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const int64_t kInputNum = 1;
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, kInputNum, prim_name);
}
}  // namespace

void ConvertToDynamic::set_is_dynamic_rank(const bool is_dynamic_rank) {
  (void)this->AddAttr(kIsDynamicRank, api::MakeValue(is_dynamic_rank));
}
bool ConvertToDynamic::get_is_dynamic_rank() const { return GetValue<bool>(GetAttr(kIsDynamicRank)); }
MIND_API_OPERATOR_IMPL(ConvertToDynamic, BaseOperator);

class ConvertToDynamicRankInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    CheckConvertToDynamicRankArgs(primitive, input_args);

    auto is_dynamic_rank_value_ptr = primitive->GetAttr(kIsDynamicRank);
    MS_EXCEPTION_IF_NULL(is_dynamic_rank_value_ptr);
    if (!is_dynamic_rank_value_ptr->isa<BoolImm>()) {
      MS_LOG(EXCEPTION) << "For '" << primitive->name() << "', 'is_dynamic_rank' must be Bool.";
    }
    bool is_dynamic_rank = GetValue<bool>(is_dynamic_rank_value_ptr);

    const std::string &op_name = primitive->name();
    auto input = abstract::CheckArg<abstract::AbstractTensor>(op_name, input_args, 0);
    MS_EXCEPTION_IF_NULL(input);
    auto input_shape = input->shape()->shape();
    if (IsDynamic(input_shape)) {
      MS_LOG(WARNING) << "It already dynamic case, input shape: " << input_shape;
    }

    ShapeVector inferred_shape;
    if (is_dynamic_rank) {
      const int64_t kUnkonwnRank = -2;
      inferred_shape = ShapeVector{kUnkonwnRank};
    } else {
      if (IsDynamicRank(input_shape)) {
        MS_LOG(WARNING) << "Do not convert dynamic rank to dynamic shape!";
        return std::make_shared<abstract::Shape>(input_shape);
      }
      int32_t input_rank = SizeToInt(input_shape.size());
      inferred_shape = ShapeVector(input_rank, abstract::Shape::kShapeDimAny);
    }
    return std::make_shared<abstract::Shape>(inferred_shape);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    CheckConvertToDynamicRankArgs(primitive, input_args);
    MS_EXCEPTION_IF_NULL(input_args[0]);
    return input_args[0]->BuildType();
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(ConvertToDynamic, prim::kPrimConvertToDynamic, ConvertToDynamicRankInfer, false);
}  // namespace ops
}  // namespace mindspore
