/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "ops/softmax_cross_entropy_with_logits.h"

#include <set>
#include <utility>
#include <map>
#include <type_traits>
#include <vector>

#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/container.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "mindapi/base/shape_vector.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
class SoftmaxCrossEntropyWithLogitsInfer : public abstract::OpInferBase {
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto prim_name = primitive->name();
    const int64_t kInputNum = 2;
    CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, kInputNum, prim_name);
    auto logits_shape = input_args[0]->BuildShape();
    auto label_shape = input_args[1]->BuildShape();
    auto logits_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(logits_shape)[kShape];
    auto label_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(label_shape)[kShape];
    const int64_t input_rank = 2;
    if (IsDynamicRank(logits_map) || IsDynamicRank(label_map)) {
      auto ds_shape_ptr = std::make_shared<abstract::Shape>(std::vector<int64_t>{abstract::Shape::kShapeRankAny});
      return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{ds_shape_ptr, ds_shape_ptr});
    }
    (void)CheckAndConvertUtils::CheckInteger("dimension of logits", SizeToLong(logits_map.size()), kEqual, input_rank,
                                             prim_name);
    (void)CheckAndConvertUtils::CheckInteger("dimension of labels", SizeToLong(label_map.size()), kEqual, input_rank,
                                             prim_name);
    auto logits_shape_ptr = logits_shape->cast<abstract::ShapePtr>();
    auto label_shape_ptr = label_shape->cast<abstract::ShapePtr>();
    // logits and label must have the same shape when is not dynamic
    if (!logits_shape_ptr->IsDynamic() && !label_shape_ptr->IsDynamic()) {
      if (*logits_shape != *label_shape) {
        MS_EXCEPTION(ValueError)
          << "For '" << prim_name
          << "', evaluator arg 'label' shape must be consistent with 'logits' shape, but got 'label' shape: "
          << label_shape->ToString() << ", 'logits' shape: " << logits_shape->ToString() << ".";
      }
    }
    auto logits_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(logits_shape);
    auto logits_shp = logits_shape_map[kShape];
    ShapeVector loss_shape = {logits_shp[0]};
    abstract::ShapePtr loss_shape_ptr = std::make_shared<abstract::Shape>(loss_shape);
    return std::make_shared<abstract::TupleShape>(
      std::vector<abstract::BaseShapePtr>{loss_shape_ptr, logits_shape_ptr});
  }
  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    const int64_t kInputNum = 2;
    CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, kInputNum, primitive->name());
    auto logits_type = input_args[0]->BuildType();
    auto label_type = input_args[1]->BuildType();
    const std::set<TypePtr> valid_types = {kFloat16, kFloat32};
    std::map<std::string, TypePtr> args;
    (void)args.insert(std::make_pair("logits_type", logits_type));
    (void)args.insert(std::make_pair("label_type", label_type));
    auto type = CheckAndConvertUtils::CheckTensorTypeSame(args, valid_types, primitive->name());
    return std::make_shared<Tuple>(std::vector<TypePtr>{type, type});
  }
};
MIND_API_OPERATOR_IMPL(SoftmaxCrossEntropyWithLogits, BaseOperator);
REGISTER_PRIMITIVE_OP_INFER_IMPL(SoftmaxCrossEntropyWithLogits, prim::kPrimSoftmaxCrossEntropyWithLogits,
                                 SoftmaxCrossEntropyWithLogitsInfer, false);
}  // namespace ops
}  // namespace mindspore
