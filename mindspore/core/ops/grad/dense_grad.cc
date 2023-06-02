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

#include "ops/grad/dense_grad.h"

#include <memory>
#include <set>
#include <string>
#include <vector>

#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype.h"
#include "ir/dtype/tensor_type.h"
#include "ir/dtype/type.h"
#include "ir/primitive.h"
#include "mindapi/base/format.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/nn_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/ms_context.h"
#include "utils/shape_utils.h"

namespace mindspore {
namespace ops {
constexpr size_t kDenseGradIndex0 = 0;
constexpr size_t kDenseGradIndex1 = 1;
constexpr size_t kDenseGradIndex2 = 2;
constexpr int64_t kDenseGradInputNum = 3;

MIND_API_OPERATOR_IMPL(DenseGrad, BaseOperator);
class DenseGradInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto prim_name = primitive->name();
    auto size = SizeToLong(input_args.size());
    (void)CheckAndConvertUtils::CheckInteger("input numbers", size, kEqual, kDenseGradInputNum, prim_name);
    for (const auto &item : input_args) {
      MS_EXCEPTION_IF_NULL(item);
    }
    auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kDenseGradIndex0]->BuildShape())[kShape];
    auto w_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kDenseGradIndex1]->BuildShape())[kShape];
    auto dx_shape_ptr = std::make_shared<abstract::Shape>(x_shape);
    auto dw_shape_ptr = std::make_shared<abstract::Shape>(w_shape);
    ShapeVector b_shape = {w_shape[kDenseGradIndex0]};
    auto db_shape_ptr = std::make_shared<abstract::Shape>(b_shape);
    auto shape_tuple_ptr = std::vector<abstract::BaseShapePtr>{dx_shape_ptr, dw_shape_ptr, db_shape_ptr};
    return std::make_shared<abstract::TupleShape>(shape_tuple_ptr);
  }

  TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(prim);
    auto prim_name = prim->name();
    MS_EXCEPTION_IF_NULL(input_args[kDenseGradIndex0]);
    auto x_type_map = input_args[kDenseGradIndex0]->BuildType();
    MS_EXCEPTION_IF_NULL(x_type_map);
    auto x_type = x_type_map->cast<TensorTypePtr>();
    MS_EXCEPTION_IF_NULL(x_type);
    MS_EXCEPTION_IF_NULL(input_args[kDenseGradIndex1]);
    auto w_type_map = input_args[kDenseGradIndex1]->BuildType();
    MS_EXCEPTION_IF_NULL(w_type_map);
    auto w_type = w_type_map->cast<TensorTypePtr>();
    MS_EXCEPTION_IF_NULL(w_type);
    MS_EXCEPTION_IF_NULL(input_args[kDenseGradIndex2]);
    auto dout_type_map = input_args[kDenseGradIndex2]->BuildType();
    MS_EXCEPTION_IF_NULL(dout_type_map);
    auto dout_type = dout_type_map->cast<TensorTypePtr>();
    MS_EXCEPTION_IF_NULL(dout_type);
    std::set<TypePtr> valid_type = {kTensorType};
    auto output_type = CheckAndConvertUtils::CheckTensorTypeValid("x", x_type, valid_type, prim_name);
    return std::make_shared<Tuple>(std::vector<TypePtr>{output_type, output_type, output_type});
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(DenseGrad, prim::kPrimDenseGrad, DenseGradInfer, false);
}  // namespace ops
}  // namespace mindspore
