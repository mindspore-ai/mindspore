/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "ops/svd.h"

#include <algorithm>
#include <set>
#include <vector>

#include "mindapi/ir/type.h"
#include "utils/check_convert_utils.h"
#include "utils/anf_utils.h"
#include "ops/op_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::BaseShapePtr SvdInferShape(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  auto compute_uv = GetValue<bool>(prim->GetAttr(kAttrComputeUV));
  auto full_matrices = GetValue<bool>(prim->GetAttr(kAttrFullMatrices));

  auto a_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  if (IsDynamicRank(a_shape) || IsDynamic(a_shape)) {
    ShapeVector dyn_shape{abstract::Shape::kShapeRankAny};
    std::vector<abstract::BaseShapePtr> shape_tuple;
    (void)shape_tuple.emplace_back(std::make_shared<abstract::Shape>(dyn_shape));
    (void)shape_tuple.emplace_back(std::make_shared<abstract::Shape>(dyn_shape));
    (void)shape_tuple.emplace_back(std::make_shared<abstract::Shape>(dyn_shape));
    return std::make_shared<abstract::TupleShape>(shape_tuple);
  }
  auto ndim = a_shape.size();
  (void)CheckAndConvertUtils::CheckInteger("ndim", SizeToLong(ndim), kGreaterEqual, kSizeTwo, prim->name());
  auto m = a_shape[ndim - kIndexTwo];
  auto n = a_shape[ndim - kIndexOne];
  auto p = std::min(m, n);

  auto s_shape = ShapeVector(a_shape.begin(), a_shape.end() - SizeToLong(kIndexOne));
  s_shape[s_shape.size() - kIndexOne] = p;
  auto u_shape = ShapeVector(a_shape.begin(), a_shape.end());
  auto v_shape = ShapeVector(a_shape.begin(), a_shape.end());
  if (compute_uv) {
    if (full_matrices) {
      u_shape[u_shape.size() - kIndexTwo] = m;
      u_shape[u_shape.size() - kIndexOne] = m;
      v_shape[v_shape.size() - kIndexTwo] = n;
      v_shape[v_shape.size() - kIndexOne] = n;
    } else {
      u_shape[u_shape.size() - kIndexTwo] = m;
      u_shape[u_shape.size() - kIndexOne] = p;
      v_shape[v_shape.size() - kIndexTwo] = n;
      v_shape[v_shape.size() - kIndexOne] = p;
    }
  } else {
    u_shape = {1};
    v_shape = {1};
  }

  std::vector<abstract::BaseShapePtr> shape_tuple;
  (void)shape_tuple.emplace_back(std::make_shared<abstract::Shape>(s_shape));
  (void)shape_tuple.emplace_back(std::make_shared<abstract::Shape>(u_shape));
  (void)shape_tuple.emplace_back(std::make_shared<abstract::Shape>(v_shape));
  return std::make_shared<abstract::TupleShape>(shape_tuple);
}

TypePtr SvdInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  auto infer_type = input_args[kInputIndex0]->BuildType();
  MS_EXCEPTION_IF_NULL(infer_type);
  const std::set<TypePtr> valid_types = {kFloat32, kFloat64};
  auto type = CheckAndConvertUtils::CheckTensorTypeValid("a", infer_type, valid_types, prim->name());

  std::vector<TypePtr> type_tuple = {type, type, type};
  return std::make_shared<Tuple>(type_tuple);
}
}  // namespace

void Svd::Init(const bool full_matrices, const bool compute_uv) {
  set_full_matrices(full_matrices);
  set_compute_uv(compute_uv);
}

void Svd::set_full_matrices(const bool full_matrices) {
  (void)this->AddAttr(kAttrFullMatrices, api::MakeValue(full_matrices));
}

void Svd::set_compute_uv(const bool compute_uv) { (void)this->AddAttr(kAttrComputeUV, api::MakeValue(compute_uv)); }

bool Svd::full_matrices() const { return GetValue<bool>(GetAttr(kAttrFullMatrices)); }
bool Svd::compute_uv() const { return GetValue<bool>(GetAttr(kAttrComputeUV)); }

MIND_API_OPERATOR_IMPL(Svd, BaseOperator);
AbstractBasePtr SvdInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                         const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kInputsNum = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputsNum, primitive->name());
  auto shape = SvdInferShape(primitive, input_args);
  auto type = SvdInferType(primitive, input_args);
  return abstract::MakeAbstract(shape, type);
}

// AG means auto generated
class MIND_API AGSvdInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return SvdInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return SvdInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return SvdInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(Svd, prim::kPrimSvd, AGSvdInfer, false);
}  // namespace ops
}  // namespace mindspore
