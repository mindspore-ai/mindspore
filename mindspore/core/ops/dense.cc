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
    constexpr auto kInputNum = 2;
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
    ShapeVector ret_shape;
    if (IsDynamicRank(x_shp) || IsDynamicRank(w_shp)) {
      ret_shape.push_back(abstract::Shape::kShapeRankAny);
      return std::make_shared<abstract::Shape>(ret_shape);
    }

    bool has_bias = false;
    const size_t kZero = 0;
    const size_t kOne = 1;
    const size_t kTwo = 2;
    const size_t kThree = 3;
    if (input_args.size() == kThree) {
      auto b = dyn_cast<abstract::AbstractTensor>(input_args[kDenseIndex2]);
      has_bias = b != nullptr;
    }
    if (w_shp.size() == kOne) {
      const auto kDimW = " if the dim of w is 1.";
      if (x_shp.size() != kOne) {
        MS_EXCEPTION(ValueError) << "The dim of x should be equal to 1" << kDimW;
      }
      if (x_shp[0] != w_shp[0]) {
        MS_EXCEPTION(ValueError) << "The value of x.shape[0] should be equal to w.shape[0]" << kDimW;
      }
      if (has_bias) {
        auto b = dyn_cast<abstract::AbstractTensor>(input_args[kDenseIndex2]);
        MS_EXCEPTION_IF_NULL(b->shape());
        auto b_shp = b->shape()->shape();
        if (b_shp.size() != kZero) {
          MS_EXCEPTION(ValueError) << "The dim of b should be equal to 0" << kDimW;
        }
      }
      return std::make_shared<abstract::Shape>(ret_shape);
    }

    const auto kDimW = " if the dim of w is 2.";
    if (w_shp.size() != kTwo) {
      MS_EXCEPTION(ValueError) << "The dim of w should be equal to 1 or 2.";
    }
    if (x_shp.size() < kTwo) {
      MS_EXCEPTION(ValueError) << "The dim of x should be larger than 1" << kDimW;
    }

    if (has_bias) {
      auto b = dyn_cast<abstract::AbstractTensor>(input_args[kDenseIndex2]);
      MS_EXCEPTION_IF_NULL(b->shape());
      auto b_shp = b->shape()->shape();
      if (b_shp.size() != kZero && b_shp.size() != kOne) {
        MS_EXCEPTION(ValueError) << "The dim of b should be equal to 0 or 1" << kDimW;
      }
    }
    (void)primitive->SetAttrs({{"has_bias", MakeValue(has_bias)}});

    auto x_col = x_shp[x_shp.size() - 1];
    auto w_row = w_shp[1];
    if (x_col != -1 && w_row != -1 && x_col != w_row && x_col >= 0 && w_row >= 0) {
      MS_EXCEPTION(ValueError) << "Dense shape error, got x_col: " << x_col << ", w_row: " << w_row
                               << ". In Dense x_col and w_row should be equal." << kDimW;
    }

    ret_shape.assign(x_shp.begin(), x_shp.end() - 1);
    ret_shape.push_back(w_shp[0]);
    return std::make_shared<abstract::Shape>(ret_shape);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    auto op_name = primitive->name();
    const std::set valid_types = {kUInt8,   kInt8,    kInt16,   kInt32,     kInt64,
                                  kFloat16, kFloat32, kFloat64, kComplex64, kComplex128};
    std::map<std::string, TypePtr> types;
    (void)types.emplace("x", input_args[kDenseIndex0]->BuildType());
    (void)types.emplace("w", input_args[kDenseIndex1]->BuildType());
    (void)CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, op_name);

    bool has_bias = false;
    const size_t kThree = 3;
    if (input_args.size() == kThree) {
      auto b = dyn_cast<abstract::AbstractTensor>(input_args[kDenseIndex2]);
      has_bias = b != nullptr;
      if (has_bias) {
        (void)types.emplace("b", input_args[kDenseIndex2]->BuildType());
        (void)CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, op_name);
      }
    }
    (void)primitive->SetAttrs({{"has_bias", MakeValue(has_bias)}});
    return input_args[kDenseIndex0]->BuildType();
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(Dense, prim::kPrimDense, DenseInfer, false);
}  // namespace ops
}  // namespace mindspore
