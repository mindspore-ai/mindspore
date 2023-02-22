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

#include <map>
#include <memory>

#include "ops/grad/dynamic_rnn_grad.h"
#include "utils/check_convert_utils.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/container.h"
#include "ir/dtype/tensor_type.h"
#include "ir/dtype/type.h"
#include "ir/primitive.h"
#include "mindapi/base/shape_vector.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
constexpr int64_t kDynRNNGradIdx0 = 0;
constexpr int64_t kDynRNNGradIdx1 = 1;
constexpr int64_t kDynRNNGradIdx2 = 2;
constexpr int64_t kDynRNNGradIdx3 = 3;
constexpr int64_t kDynRNNGradIdx4 = 4;
constexpr int64_t kDynRNNGradIdx5 = 5;
constexpr int64_t kDynRNNGradIdx6 = 6;
constexpr int64_t kDynRNNGradIdx7 = 7;
constexpr int64_t kDynRNNGradIdx8 = 8;
constexpr int64_t kDynRNNGradIdx9 = 9;
constexpr int64_t kDynRNNGradIdx10 = 10;
constexpr int64_t kDynRNNGradIdx11 = 11;
constexpr int64_t kDynRNNGradIdx12 = 12;
constexpr int64_t kDynRNNGradIdx13 = 13;
constexpr int64_t kDynRNNGradIdx14 = 14;
constexpr int64_t kDynRNNGradIdx15 = 15;
constexpr int64_t kNum4 = 4;

abstract::TupleShapePtr DynamicRNNGradInferDynamicShape(const std::vector<AbstractBasePtr> &input_args) {
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kDynRNNGradIdx0]->BuildShape())[kShape];
  auto w_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kDynRNNGradIdx1]->BuildShape())[kShape];
  auto dh_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kDynRNNGradIdx9]->BuildShape())[kShape];
  auto dc_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kDynRNNGradIdx10]->BuildShape())[kShape];
  ShapeVector dw_out_shape_dyn;
  ShapeVector db_out_shape_dyn;
  ShapeVector dx_out_shape_dyn;
  ShapeVector dh_prev_out_shape_dyn;
  ShapeVector dc_prev_out_shape_dyn;
  for (size_t i = 0; i < w_shape.size(); ++i) {
    dw_out_shape_dyn.push_back(abstract::Shape::kShapeDimAny);
  }
  db_out_shape_dyn.push_back(abstract::Shape::kShapeDimAny);
  for (size_t i = 0; i < x_shape.size(); ++i) {
    dx_out_shape_dyn.push_back(abstract::Shape::kShapeDimAny);
  }
  for (size_t i = 0; i < dh_shape.size(); ++i) {
    dh_prev_out_shape_dyn.push_back(abstract::Shape::kShapeDimAny);
  }
  for (size_t i = 0; i < dc_shape.size(); ++i) {
    dc_prev_out_shape_dyn.push_back(abstract::Shape::kShapeDimAny);
  }
  abstract::ShapePtr dw_out_shape_dyn_ptr = std::make_shared<abstract::Shape>(dw_out_shape_dyn);
  abstract::ShapePtr db_out_shape_dyn_ptr = std::make_shared<abstract::Shape>(db_out_shape_dyn);
  abstract::ShapePtr dx_out_shape_dyn_ptr = std::make_shared<abstract::Shape>(dx_out_shape_dyn);
  abstract::ShapePtr dh_prev_out_shape_dyn_ptr = std::make_shared<abstract::Shape>(dh_prev_out_shape_dyn);
  abstract::ShapePtr dc_prev_out_shape_dyn_ptr = std::make_shared<abstract::Shape>(dc_prev_out_shape_dyn);
  return std::make_shared<abstract::TupleShape>(
    std::vector<abstract::BaseShapePtr>{dw_out_shape_dyn_ptr, db_out_shape_dyn_ptr, dx_out_shape_dyn_ptr,
                                        dh_prev_out_shape_dyn_ptr, dc_prev_out_shape_dyn_ptr});
}

void DynamicRNNGradShapeCheck(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto op_name = primitive->name();
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kDynRNNGradIdx0]->BuildShape())[kShape];
  auto w_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kDynRNNGradIdx1]->BuildShape())[kShape];
  auto b_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kDynRNNGradIdx2]->BuildShape())[kShape];
  const int64_t x_shape_size = 3;
  if (x_shape.size() != x_shape_size) {
    MS_EXCEPTION(ValueError) << "For '" << op_name << "', input 'x' size must be 3, but got " << x_shape.size() << ".";
  }
  int64_t hidden_size = w_shape[w_shape.size() - 1] / kNum4;
  if (w_shape[w_shape.size() - 1] % kNum4 != 0) {
    MS_EXCEPTION(ValueError) << "For '" << op_name << "', w_shape[-1] should multiple of 4, now is "
                             << w_shape[w_shape.size() - 1] << ".";
  }
  if (w_shape[0] != x_shape[kDynRNNGradIdx2] + hidden_size) {
    MS_EXCEPTION(ValueError) << "For '" << op_name
                             << "', w_shape[0] should equal to input_size + hidden_size, but gets " << w_shape[0]
                             << ".";
  }
  if (b_shape[0] != w_shape[1]) {
    MS_EXCEPTION(ValueError) << "For '" << op_name << "', b_shape[0] should equal to w_shape[1], but gets "
                             << b_shape[0] << ".";
  }
}

abstract::TupleShapePtr DynamicRNNGradInferShape(const PrimitivePtr &primitive,
                                                 const std::vector<AbstractBasePtr> &input_args) {
  auto op_name = primitive->name();
  auto x_shape_ptr = input_args[kDynRNNGradIdx0]->BuildShape();
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(x_shape_ptr)[kShape];
  auto w_shape_ptr = input_args[kDynRNNGradIdx1]->BuildShape();
  auto w_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kDynRNNGradIdx1]->BuildShape())[kShape];
  auto y_shape_ptr = input_args[kDynRNNGradIdx3]->BuildShape();
  auto h_shape_ptr = input_args[kDynRNNGradIdx6]->BuildShape();
  auto c_shape_ptr = input_args[kDynRNNGradIdx7]->BuildShape();
  auto dy_shape_ptr = input_args[kDynRNNGradIdx8]->BuildShape();
  auto dh_shape_ptr = input_args[kDynRNNGradIdx9]->BuildShape();
  auto dh_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(dh_shape_ptr)[kShape];
  auto dc_shape_ptr = input_args[kDynRNNGradIdx10]->BuildShape();
  auto dc_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(dc_shape_ptr)[kShape];
  auto i_shape_ptr = input_args[kDynRNNGradIdx11]->BuildShape();
  auto j_shape_ptr = input_args[kDynRNNGradIdx12]->BuildShape();
  auto f_shape_ptr = input_args[kDynRNNGradIdx13]->BuildShape();
  auto o_shape_ptr = input_args[kDynRNNGradIdx14]->BuildShape();
  auto tanhct_shape_ptr = input_args[kDynRNNGradIdx15]->BuildShape();
  if (IsDynamic(x_shape) || IsDynamic(w_shape) || IsDynamic(dh_shape) || IsDynamic(dc_shape)) {
    return DynamicRNNGradInferDynamicShape(input_args);
  }
  DynamicRNNGradShapeCheck(primitive, input_args);
  int64_t hidden_size = w_shape[w_shape.size() - 1] / kNum4;
  std::vector<int64_t> valid_shape{x_shape[kDynRNNGradIdx0], x_shape[kDynRNNGradIdx1], hidden_size};
  const std::map<std::string, BaseShapePtr> shapes = {
    {"y_shape", y_shape_ptr}, {"h_shape", h_shape_ptr},           {"c_shape", c_shape_ptr},
    {"i_shape", i_shape_ptr}, {"j_shape", j_shape_ptr},           {"f_shape", f_shape_ptr},
    {"o_shape", o_shape_ptr}, {"tanhct_shape", tanhct_shape_ptr}, {"dy_shape", dy_shape_ptr}};
  (void)CheckAndConvertUtils::CheckTensorShapeSame(shapes, valid_shape, op_name);
  std::vector<int64_t> valid_shape_d{x_shape[kDynRNNGradIdx1], hidden_size};
  const std::map<std::string, BaseShapePtr> shapes_d = {{"dh_shape", dh_shape_ptr}, {"dc_shape", dc_shape_ptr}};
  (void)CheckAndConvertUtils::CheckTensorShapeSame(shapes_d, valid_shape_d, op_name);
  std::vector<int64_t> db_out_shape{w_shape[1]};
  abstract::ShapePtr db_out_shape_ptr = std::make_shared<abstract::Shape>(db_out_shape);
  return std::make_shared<abstract::TupleShape>(
    std::vector<abstract::BaseShapePtr>{w_shape_ptr, db_out_shape_ptr, x_shape_ptr, dh_shape_ptr, dc_shape_ptr});
}

TuplePtr DynamicRNNGradInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto op_name = primitive->name();
  auto x_dtype = input_args[0]->BuildType();
  MS_EXCEPTION_IF_NULL(x_dtype);
  if (!x_dtype->isa<TensorType>()) {
    MS_EXCEPTION(TypeError) << "For '" << op_name << "', input must be a Tensor, but got: " << x_dtype->ToString()
                            << ".";
  }
  return std::make_shared<Tuple>(std::vector<TypePtr>{x_dtype, x_dtype, x_dtype, x_dtype, x_dtype});
}
}  // namespace

MIND_API_OPERATOR_IMPL(DynamicRNNGrad, BaseOperator);

AbstractBasePtr DynamicRNNGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 16;
  CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, input_num, primitive->name());
  auto type = DynamicRNNGradInferType(primitive, input_args);
  auto shape = DynamicRNNGradInferShape(primitive, input_args);
  return abstract::MakeAbstract(shape, type);
}

// AG means auto generated
class MIND_API AGDynamicRNNGradInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return DynamicRNNGradInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return DynamicRNNGradInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return DynamicRNNGradInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(DynamicRNNGrad, prim::kPrimDynamicRNNGrad, AGDynamicRNNGradInfer, false);
}  // namespace ops
}  // namespace mindspore
