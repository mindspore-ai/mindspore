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
#include <string>
#include <vector>

#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/dtype/type.h"
#include "ir/primitive.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/base/type_id.h"
#include "mindapi/ir/value.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/nn_ops.h"
#include "ops/dense.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"

namespace mindspore {
namespace ops {
constexpr size_t kDenseIndex0 = 0;
constexpr size_t kDenseIndex1 = 1;
constexpr size_t kDenseIndex2 = 2;
void Dense::Init(bool has_bias) { set_has_bias(has_bias); }

void Dense::set_has_bias(bool has_bias) { (void)AddAttr(kHasBias, api::MakeValue(has_bias)); }

bool Dense::get_has_bias() const {
  auto value_ptr = GetAttr(kHasBias);
  return GetValue<bool>(value_ptr);
}

MIND_API_OPERATOR_IMPL(Dense, BaseOperator);
class DenseInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    constexpr auto kInputNum = 3;
    const std::string op_name = primitive->name();
    (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kGreaterEqual, kInputNum,
                                             op_name);
    auto x = CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(op_name, input_args, 0);
    MS_EXCEPTION_IF_NULL(x);
    MS_EXCEPTION_IF_NULL(x->shape());
    auto w = CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(op_name, input_args, 1);
    MS_EXCEPTION_IF_NULL(w);
    MS_EXCEPTION_IF_NULL(w->shape());
    auto x_shp = x->shape()->shape();
    auto w_shp = w->shape()->shape();
    if (IsDynamicRank(x_shp) || IsDynamicRank(w_shp)) {
      ShapeVector ret_shape{abstract::Shape::kShapeRankAny};
      return std::make_shared<abstract::Shape>(ret_shape);
    }

    const size_t W_SHAPE_SIZE = 2;
    if (w_shp.size() != W_SHAPE_SIZE) {
      MS_EXCEPTION(ValueError) << "The size of w should be equal to 2.";
    }
    if (x_shp.size() < W_SHAPE_SIZE) {
      MS_EXCEPTION(ValueError) << "The size of x should be larger than 2.";
    }

    ValuePtr has_bias_ptr = primitive->GetAttr("has_bias");
    bool has_bias = GetValue<bool>(has_bias_ptr);
    if (has_bias) {
      auto b = CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(op_name, input_args, 2);
      MS_EXCEPTION_IF_NULL(b);
      MS_EXCEPTION_IF_NULL(b->shape());
      auto b_shp = b->shape()->shape();
      if (IsDynamicRank(b_shp)) {
        ShapeVector ret_shape{abstract::Shape::kShapeRankAny};
        return std::make_shared<abstract::Shape>(ret_shape);
      }
      const size_t B_SHAPE_SIZE = 1;
      if (b_shp.size() != B_SHAPE_SIZE) {
        MS_EXCEPTION(ValueError) << "The size of b should be equal to 1.";
      }
    }

    auto x_col = x_shp[x_shp.size() - 1];
    auto w_row = w_shp[1];
    if (x_col != w_row && x_col >= 0 && w_row >= 0) {
      MS_EXCEPTION(ValueError) << "Dense shape error, got x_col: " << x_col << ", w_row: " << w_row
                               << ". In Dense x_col and w_row should be equal.";
    }

    ShapeVector ret_shape;
    ret_shape.assign(x_shp.begin(), x_shp.end() - 1);
    ret_shape.push_back(w_shp[0]);
    return std::make_shared<abstract::Shape>(ret_shape);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    auto op_name = primitive->name();
    auto x = CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(op_name, input_args, kDenseIndex0);
    auto w = CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(op_name, input_args, kDenseIndex1);
    TypePtr x_type = x->element()->GetTypeTrack();
    TypePtr w_type = w->element()->GetTypeTrack();
    auto x_type_id = x_type->type_id();
    auto w_type_id = w_type->type_id();
    if (x_type_id != w_type_id) {
      MS_EXCEPTION(TypeError) << "The type of `x` and `w` must be same, but got " << x_type_id << " and " << w_type_id;
    }
    if (x_type_id != TypeId::kNumberTypeFloat16 && x_type_id != TypeId::kNumberTypeFloat32 &&
        x_type_id != TypeId::kNumberTypeFloat64) {
      MS_EXCEPTION(TypeError) << "The type of `x` must be float16, float32 or float64, but got " << x_type_id;
    }
    ValuePtr has_bias_ptr = primitive->GetAttr("has_bias");
    bool has_bias = GetValue<bool>(has_bias_ptr);
    if (has_bias) {
      auto b = CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(op_name, input_args, kDenseIndex2);
      TypePtr b_type = b->element()->GetTypeTrack();
      auto b_type_id = b_type->type_id();
      if (x_type_id != b_type_id) {
        MS_EXCEPTION(TypeError) << "The type of `x` and `b` must be same, but got " << x_type_id << " and "
                                << b_type_id;
      }
    }
    if (primitive->HasAttr("cast_type")) {
      auto out_type = primitive->GetAttr("cast_type");
      MS_EXCEPTION_IF_NULL(out_type);
      if (!out_type->isa<Type>()) {
        MS_EXCEPTION(ValueError) << "Dense cast_type must be a `Type`";
      }
      x_type = out_type->cast<TypePtr>();
    }
    return x_type;
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(Dense, prim::kPrimDense, DenseInfer, false);
}  // namespace ops
}  // namespace mindspore
