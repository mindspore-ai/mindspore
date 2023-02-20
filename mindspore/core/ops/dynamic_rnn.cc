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
#include <set>
#include "ops/dynamic_rnn.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
constexpr int64_t kDynRnnIdx0 = 0;
constexpr int64_t kDynRnnIdx1 = 1;
constexpr int64_t kDynRnnIdx2 = 2;
constexpr int64_t kDynRnnIdx3 = 3;
constexpr int64_t kDynRnnIdx4 = 4;
constexpr int64_t kDynRnnIdx5 = 5;
constexpr int64_t kDynamicRnnShapeX = 3;
constexpr int64_t kDynamicRnnShapeW = 2;
constexpr int64_t kDynamicRnnShapeB = 1;
constexpr int64_t kDynamicRnnShapeH = 3;
constexpr int64_t kDynamicRnnShapeC = 3;
constexpr int64_t kDynRnnNum4 = 4;
constexpr int64_t kDynRnnInputNum = 6;

abstract::TupleShapePtr DynamicRNNInferDynamicShape(const std::vector<AbstractBasePtr> &input_args) {
  const int64_t y_shape_num = 3;
  ShapeVector y_shape_dyn;
  for (size_t i = 0; i < y_shape_num; ++i) {
    y_shape_dyn.push_back(abstract::Shape::kShapeDimAny);
  }
  abstract::ShapePtr y_shape_dyn_ptr = std::make_shared<abstract::Shape>(y_shape_dyn);
  return std::make_shared<abstract::TupleShape>(
    std::vector<abstract::BaseShapePtr>{y_shape_dyn_ptr, y_shape_dyn_ptr, y_shape_dyn_ptr, y_shape_dyn_ptr,
                                        y_shape_dyn_ptr, y_shape_dyn_ptr, y_shape_dyn_ptr, y_shape_dyn_ptr});
}

void DynamicRNNShapeCheck(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto op_name = primitive->name();
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kDynRnnIdx0]->BuildShape())[kShape];
  auto w_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kDynRnnIdx1]->BuildShape())[kShape];
  auto b_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kDynRnnIdx2]->BuildShape())[kShape];
  (void)CheckAndConvertUtils::CheckInteger("x_shape", SizeToLong(x_shape.size()), kEqual, kDynamicRnnShapeX, op_name);
  (void)CheckAndConvertUtils::CheckInteger("w_shape", SizeToLong(w_shape.size()), kEqual, kDynamicRnnShapeW, op_name);
  (void)CheckAndConvertUtils::CheckInteger("b_shape", SizeToLong(b_shape.size()), kEqual, kDynamicRnnShapeB, op_name);
  int64_t input_size = x_shape[kDynRnnIdx2];
  int64_t hidden_size = w_shape[w_shape.size() - 1] / kDynRnnNum4;
  if (w_shape[w_shape.size() - 1] % kDynRnnNum4 != 0) {
    MS_EXCEPTION(ValueError) << "For '" << op_name << "', w_shape[-1] should multiple of 4, now is "
                             << w_shape[w_shape.size() - 1] << ".";
  }
  if (w_shape[0] != input_size + hidden_size) {
    MS_EXCEPTION(ValueError) << "For '" << op_name
                             << "', w_shape[0] should equal to input_size + hidden_size, but gets " << w_shape[0]
                             << ".";
  }
  if (b_shape[0] != w_shape[1]) {
    MS_EXCEPTION(ValueError) << "For '" << op_name << "', b_shape[0] should equal to w_shape[1], but gets "
                             << b_shape[0] << ".";
  }

  if (input_args.size() > kDynRnnIdx3) {
    auto seq_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kDynRnnIdx3]->BuildShape())[kShape];
    if (seq_shape.size() != 0) {
      MS_EXCEPTION(ValueError) << "For '" << op_name << "', input 'seq' shape must be 0, but got " << seq_shape.size()
                               << ".";
    }
  }
  if (input_args.size() > kDynRnnIdx4) {
    int64_t batch_size = x_shape[kDynRnnIdx1];
    auto h_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kDynRnnIdx4]->BuildShape())[kShape];
    (void)CheckAndConvertUtils::CheckInteger("h_shape", SizeToLong(h_shape.size()), kEqual, kDynamicRnnShapeH, op_name);
    (void)CheckAndConvertUtils::CheckInteger("h_shape[0]", h_shape[kDynRnnIdx0], kEqual, (int64_t)1, op_name);
    (void)CheckAndConvertUtils::CheckInteger("h_shape[1]", h_shape[kDynRnnIdx1], kEqual, (int64_t)batch_size, op_name);
    (void)CheckAndConvertUtils::CheckInteger("h_shape[2]", h_shape[kDynRnnIdx2], kEqual, (int64_t)hidden_size, op_name);
    if (input_args.size() > kDynRnnIdx5) {
      auto c_shape_ptr = input_args[kDynRnnIdx5]->BuildShape();
      auto c_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kDynRnnIdx5]->BuildShape())[kShape];
      (void)CheckAndConvertUtils::CheckInteger("c_shape", SizeToLong(c_shape.size()), kEqual, kDynamicRnnShapeC,
                                               op_name);
      const std::map<std::string, BaseShapePtr> shapes = {{"c_shape", c_shape_ptr}};
      (void)CheckAndConvertUtils::CheckTensorShapeSame(shapes, h_shape, op_name);
    }
  }
}

abstract::TupleShapePtr DynamicRNNInferShape(const PrimitivePtr &primitive,
                                             const std::vector<AbstractBasePtr> &input_args) {
  CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, kDynRnnInputNum, primitive->name());
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kDynRnnIdx0]->BuildShape())[kShape];
  auto w_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kDynRnnIdx1]->BuildShape())[kShape];
  std::vector<ValuePtr> placeholder_index = {MakeValue((int64_t)3)};
  primitive->AddAttr("placeholder_index", MakeValue(placeholder_index));
  if (IsDynamicRank(x_shape) || IsDynamicRank(w_shape)) {
    return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{
      std::make_shared<abstract::Shape>(ShapeVector{abstract::Shape::kShapeRankAny}),
      std::make_shared<abstract::Shape>(ShapeVector{abstract::Shape::kShapeRankAny}),
      std::make_shared<abstract::Shape>(ShapeVector{abstract::Shape::kShapeRankAny}),
      std::make_shared<abstract::Shape>(ShapeVector{abstract::Shape::kShapeRankAny}),
      std::make_shared<abstract::Shape>(ShapeVector{abstract::Shape::kShapeRankAny}),
      std::make_shared<abstract::Shape>(ShapeVector{abstract::Shape::kShapeRankAny}),
      std::make_shared<abstract::Shape>(ShapeVector{abstract::Shape::kShapeRankAny}),
      std::make_shared<abstract::Shape>(ShapeVector{abstract::Shape::kShapeRankAny})});
  }
  if (IsDynamic(x_shape) || IsDynamic(w_shape)) {
    return DynamicRNNInferDynamicShape(input_args);
  }
  DynamicRNNShapeCheck(primitive, input_args);
  int64_t num_step = x_shape[kDynRnnIdx0];
  int64_t batch_size = x_shape[kDynRnnIdx1];
  int64_t input_size = x_shape[kDynRnnIdx2];
  int64_t hidden_size = w_shape[w_shape.size() - 1] / kDynRnnNum4;
  primitive->AddAttr("input_size", MakeValue(input_size));
  primitive->AddAttr("hidden_size", MakeValue(hidden_size));
  std::vector<int64_t> y_shape{num_step, batch_size, hidden_size};
  abstract::ShapePtr y_shape_ptr = std::make_shared<abstract::Shape>(y_shape);
  return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{
    y_shape_ptr, y_shape_ptr, y_shape_ptr, y_shape_ptr, y_shape_ptr, y_shape_ptr, y_shape_ptr, y_shape_ptr});
}

TuplePtr DynamicRNNInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, kDynRnnInputNum, primitive->name());
  auto op_name = primitive->name();
  auto x_dtype = input_args[kDynRnnIdx0]->BuildType();
  auto w_dtype = input_args[kDynRnnIdx1]->BuildType();
  auto b_dtype = input_args[kDynRnnIdx2]->BuildType();
  auto h_dtype = input_args[kDynRnnIdx4]->BuildType();
  auto c_dtype = input_args[kDynRnnIdx5]->BuildType();
  auto seq_type = input_args[kDynRnnIdx3]->BuildType();
  if (seq_type->type_id() != kMetaTypeNone) {
    MS_EXCEPTION(ValueError) << "For '" << op_name << "' seq is not None, please check seq's type";
  }
  std::set<TypePtr> float16_set = {kFloat16};
  MS_EXCEPTION_IF_NULL(x_dtype);
  MS_EXCEPTION_IF_NULL(w_dtype);
  MS_EXCEPTION_IF_NULL(h_dtype);
  MS_EXCEPTION_IF_NULL(c_dtype);
  std::map<std::string, TypePtr> types;
  types.emplace("x", x_dtype);
  types.emplace("w", w_dtype);
  types.emplace("h", h_dtype);
  types.emplace("c", c_dtype);
  (void)CheckAndConvertUtils::CheckScalarOrTensorTypesSame(types, float16_set, op_name, true);
  const std::set<TypePtr> valid_b_types = {kFloat16, kFloat32};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("b", b_dtype, valid_b_types, op_name);
  return std::make_shared<Tuple>(
    std::vector<TypePtr>{b_dtype, x_dtype, b_dtype, b_dtype, b_dtype, b_dtype, b_dtype, b_dtype});
}
}  // namespace

MIND_API_OPERATOR_IMPL(DynamicRNN, BaseOperator);

AbstractBasePtr DynamicRNNInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto type = DynamicRNNInferType(primitive, input_args);
  auto shape = DynamicRNNInferShape(primitive, input_args);
  return abstract::MakeAbstract(shape, type);
}

// AG means auto generated
class MIND_API AGDynamicRNNInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return DynamicRNNInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return DynamicRNNInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return DynamicRNNInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(DynamicRNN, prim::kPrimDynamicRNN, AGDynamicRNNInfer, false);
}  // namespace ops
}  // namespace mindspore
