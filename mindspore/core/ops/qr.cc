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
#include "ops/qr.h"

#include <algorithm>
#include <map>
#include <set>

#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/container.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::TupleShapePtr QrInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kDimLeastNum = 2;
  const int64_t kDimPenultimateNum = 2;
  auto x_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape());
  auto x_shape = x_shape_map[kShape];
  // support dynamic rank
  if (IsDynamicRank(x_shape)) {
    auto unknow_rank_ptr = std::make_shared<abstract::Shape>(std::vector<int64_t>{abstract::Shape::kShapeRankAny});
    return std::make_shared<abstract::TupleShape>(
      std::vector<abstract::BaseShapePtr>{unknow_rank_ptr, unknow_rank_ptr});
  }
  // support dynamic shape
  if (IsDynamic(x_shape)) {
    ShapeVector shape_out;
    for (size_t i = 0; i < x_shape.size(); ++i) {
      shape_out.push_back(abstract::Shape::kShapeDimAny);
    }
    auto unknow_shape_ptr = std::make_shared<abstract::Shape>(shape_out);
    return std::make_shared<abstract::TupleShape>(
      std::vector<abstract::BaseShapePtr>{unknow_shape_ptr, unknow_shape_ptr});
  }

  (void)CheckAndConvertUtils::CheckInteger("rank of argument[x]", SizeToLong(x_shape.size()), kGreaterEqual,
                                           kDimLeastNum, primitive->name());

  bool full_matrices_attr = GetValue<bool>(primitive->GetAttr("full_matrices"));
  std::vector<int64_t> out_q_dims(x_shape.begin(), x_shape.end());
  std::vector<int64_t> out_r_dims(x_shape.begin(), x_shape.end());
  if (full_matrices_attr) {
    out_q_dims[out_q_dims.size() - 1] = out_q_dims[out_q_dims.size() - kDimPenultimateNum];
  } else {
    auto p = std::min(x_shape[x_shape.size() - kDimPenultimateNum], x_shape[x_shape.size() - 1]);
    out_q_dims[out_q_dims.size() - 1] = p;
    out_r_dims[out_r_dims.size() - kDimPenultimateNum] = p;
  }
  abstract::ShapePtr q_shape = std::make_shared<abstract::Shape>(out_q_dims);
  abstract::ShapePtr r_shape = std::make_shared<abstract::Shape>(out_r_dims);
  return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{q_shape, r_shape});
}

TuplePtr QrInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kFloat64, kComplex64, kComplex128};
  auto out_type =
    CheckAndConvertUtils::CheckTensorTypeValid("x", input_args[0]->BuildType(), valid_types, prim->name());
  return std::make_shared<Tuple>(std::vector<TypePtr>{out_type, out_type});
}
}  // namespace

MIND_API_OPERATOR_IMPL(Qr, BaseOperator);

void Qr::Init(const bool full_matrices) { this->set_full_matrices(full_matrices); }

void Qr::set_full_matrices(const bool full_matrices) {
  (void)this->AddAttr("full_matrices", api::MakeValue(full_matrices));
}

bool Qr::get_full_matrices() const {
  auto value_ptr = GetAttr("full_matrices");
  return GetValue<bool>(value_ptr);
}

AbstractBasePtr QrInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                        const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto infer_type = QrInferType(primitive, input_args);
  auto infer_shape = QrInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

// AG means auto generated
class MIND_API AGQrInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return QrInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return QrInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return QrInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(Qr, prim::kPrimQr, AGQrInfer, false);
}  // namespace ops
}  // namespace mindspore
