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

#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "abstract/abstract_value.h"
#include "ops/dynamic_gru_v2_grad.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/dtype/container.h"
#include "ir/dtype/number.h"
#include "ir/dtype/type.h"
#include "ir/primitive.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/base/type_id.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
void DynamicGRUV2GradCheckShapeValue(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args,
                                     const int64_t &num_proj) {
  auto prim_name = primitive->name();
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  auto winput_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
  auto whidden_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->BuildShape())[kShape];
  auto y_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex3]->BuildShape())[kShape];
  auto init_h_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex4]->BuildShape())[kShape];
  auto h_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex5]->BuildShape())[kShape];
  auto dy_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex6]->BuildShape())[kShape];
  auto dh_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex7]->BuildShape())[kShape];
  auto update_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex8]->BuildShape())[kShape];
  auto reset_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex9]->BuildShape())[kShape];
  auto new_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex10]->BuildShape())[kShape];
  auto hnew_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex11]->BuildShape())[kShape];

  std::vector<ShapeVector> all_shapes = {x_shape,  winput_shape, whidden_shape, y_shape,     init_h_shape, h_shape,
                                         dy_shape, dh_shape,     update_shape,  reset_shape, new_shape,    hnew_shape};
  auto is_dynamic = std::any_of(all_shapes.begin(), all_shapes.end(), IsDynamic);
  if (is_dynamic) {
    return;
  }

  int64_t num_step = x_shape[0];
  int64_t batch_size = x_shape[1];
  int64_t input_size = x_shape[2];
  int64_t hidden_size = whidden_shape[0];

  auto winput_shape_ptr = input_args[kInputIndex1]->BuildShape();
  auto whidden_shape_ptr = input_args[kInputIndex2]->BuildShape();
  auto y_shape_ptr = input_args[kInputIndex3]->BuildShape();
  auto init_h_shape_ptr = input_args[kInputIndex4]->BuildShape();
  auto h_shape_ptr = input_args[kInputIndex5]->BuildShape();
  auto dy_shape_ptr = input_args[kInputIndex6]->BuildShape();
  auto dh_shape_ptr = input_args[kInputIndex7]->BuildShape();
  auto update_shape_ptr = input_args[kInputIndex8]->BuildShape();
  auto reset_shape_ptr = input_args[kInputIndex9]->BuildShape();
  auto new_shape_ptr = input_args[kInputIndex10]->BuildShape();
  auto hnew_shape_ptr = input_args[kInputIndex11]->BuildShape();

  (void)CheckAndConvertUtils::CheckTensorShapeSame({{"weight input shape", winput_shape_ptr}},
                                                   std::vector<int64_t>{input_size, 3 * hidden_size}, prim_name);
  (void)CheckAndConvertUtils::CheckTensorShapeSame({{"weight hidden shape", whidden_shape_ptr}},
                                                   std::vector<int64_t>{hidden_size, 3 * hidden_size}, prim_name);
  (void)CheckAndConvertUtils::CheckTensorShapeSame({{"init h shape", init_h_shape_ptr}},
                                                   std::vector<int64_t>{batch_size, hidden_size}, prim_name);
  (void)CheckAndConvertUtils::CheckTensorShapeSame({{"dh shape", dh_shape_ptr}},
                                                   std::vector<int64_t>{batch_size, hidden_size}, prim_name);

  std::vector<int64_t> valid_y_shape;
  valid_y_shape.emplace_back(num_step);
  valid_y_shape.emplace_back(batch_size);
  const int64_t kNumZero = 0;
  if (num_proj > kNumZero) {
    valid_y_shape.emplace_back(std::min(hidden_size, num_proj));
  } else {
    valid_y_shape.emplace_back(hidden_size);
  }
  (void)CheckAndConvertUtils::CheckTensorShapeSame({{"y shape", y_shape_ptr}}, valid_y_shape, prim_name);

  std::map<std::string, BaseShapePtr> check_shapes = {
    {"h shape", h_shape_ptr},         {"dy shape", dy_shape_ptr},   {"update shape", update_shape_ptr},
    {"reset shape", reset_shape_ptr}, {"new shape", new_shape_ptr}, {"hnew shape", hnew_shape_ptr}};
  std::vector<int64_t> valid_shape = {num_step, batch_size, hidden_size};
  (void)CheckAndConvertUtils::CheckTensorShapeSame(check_shapes, valid_shape, prim_name);

  if (input_args.size() >= kInputIndex13 && input_args[kInputIndex12]->BuildType()->type_id() != kMetaTypeNone) {
    auto seq_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex12]->BuildShape())[kShape];
    auto seq_shape_ptr = input_args[kInputIndex12]->BuildShape();
    if (!IsDynamic(seq_shape)) {
      (void)CheckAndConvertUtils::CheckTensorShapeSame({{"seq shape", seq_shape_ptr}}, std::vector<int64_t>{batch_size},
                                                       prim_name);
    }
  }
}

abstract::TupleShapePtr DynamicGRUV2GradInferShape(const PrimitivePtr &primitive,
                                                   const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  auto winput_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
  auto whidden_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->BuildShape())[kShape];
  auto y_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex3]->BuildShape())[kShape];

  int64_t num_proj = 0;
  if (primitive->HasAttr(kNumProj)) {
    num_proj = GetValue<int64_t>(primitive->GetAttr(kNumProj));
  }

  std::vector<ShapeVector> check_shapes = {x_shape, winput_shape, whidden_shape, y_shape};
  auto is_dynamic_rank = std::any_of(check_shapes.begin(), check_shapes.end(), IsDynamicRank);

  const size_t kNumTwo = 2;
  const size_t kNumThree = 3;
  if (!is_dynamic_rank) {
    (void)CheckAndConvertUtils::CheckInteger("x shape rank", x_shape.size(), kEqual, kNumThree, prim_name);
    (void)CheckAndConvertUtils::CheckInteger("weight input shape rank", winput_shape.size(), kEqual, kNumTwo,
                                             prim_name);
    (void)CheckAndConvertUtils::CheckInteger("weight hidden shape rank", whidden_shape.size(), kEqual, kNumTwo,
                                             prim_name);
    (void)CheckAndConvertUtils::CheckInteger("y shape rank", y_shape.size(), kEqual, kNumThree, prim_name);
  }
  DynamicGRUV2GradCheckShapeValue(primitive, input_args, num_proj);

  int64_t num_step = -1;
  int64_t batch_size = -1;
  int64_t input_size = -1;
  int64_t hidden_size = -1;
  int64_t hidden_size_three = -1;
  if (!(IsDynamic(x_shape) || IsDynamic(whidden_shape))) {
    num_step = x_shape[kInputIndex0];
    batch_size = x_shape[kInputIndex1];
    input_size = x_shape[kInputIndex2];
    hidden_size = whidden_shape[kInputIndex0];
    hidden_size_three = whidden_shape[kInputIndex1];
  }

  ShapeVector dx_shape = {num_step, batch_size, input_size};
  ShapeVector dh_shape = {batch_size, hidden_size};
  ShapeVector dwinput_shape = {input_size, hidden_size_three};
  ShapeVector dwhidden_shape = {hidden_size, hidden_size_three};
  ShapeVector db_shape = {hidden_size_three};

  auto db_shape_ptr = std::make_shared<abstract::Shape>(db_shape);
  auto dh_shape_ptr = std::make_shared<abstract::Shape>(dh_shape);
  auto dx_shape_ptr = std::make_shared<abstract::Shape>(dx_shape);
  auto dwinput_shape_ptr = std::make_shared<abstract::Shape>(dwinput_shape);
  auto dwhidden_shape_ptr = std::make_shared<abstract::Shape>(dwhidden_shape);

  return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{
    dwinput_shape_ptr, dwhidden_shape_ptr, db_shape_ptr, db_shape_ptr, dx_shape_ptr, dh_shape_ptr});
}

TuplePtr DynamicGRUV2GradInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32};
  auto x_dtype = input_args[kInputIndex0]->BuildType();
  auto winput_dtype = input_args[kInputIndex1]->BuildType();
  auto whidden_dtype = input_args[kInputIndex2]->BuildType();
  auto y_dtype = input_args[kInputIndex3]->BuildType();
  auto init_h_dtype = input_args[kInputIndex4]->BuildType();
  auto h_dtype = input_args[kInputIndex5]->BuildType();
  auto dy_dtype = input_args[kInputIndex6]->BuildType();
  auto dh_dtype = input_args[kInputIndex7]->BuildType();
  auto update_dtype = input_args[kInputIndex8]->BuildType();
  auto reset_dtype = input_args[kInputIndex9]->BuildType();
  auto new_dtype = input_args[kInputIndex10]->BuildType();
  auto hnew_dtype = input_args[kInputIndex11]->BuildType();

  std::map<std::string, TypePtr> check_types = {
    {"y_dtype", y_dtype},           {"h_dtype", h_dtype},         {"dy_dtype", dy_dtype},   {"dh_dtype", dh_dtype},
    {"update_dtype", update_dtype}, {"reset_dtype", reset_dtype}, {"new_dtype", new_dtype}, {"hnew_dtype", hnew_dtype}};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x_dtype", x_dtype, valid_types, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("winput_dtype", winput_dtype, valid_types, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("whidden_dtype", whidden_dtype, valid_types, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("init_h_dtype", init_h_dtype, valid_types, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeSame(check_types, valid_types, prim_name);
  if (input_args.size() >= kInputIndex13 && input_args[kInputIndex12]->BuildType()->type_id() != kMetaTypeNone) {
    auto seq_dtype = input_args[kInputIndex12]->BuildType();
    (void)CheckAndConvertUtils::CheckTensorTypeValid("seq_dtype", seq_dtype, valid_types, prim_name);
  }
  if (input_args.size() >= kInputIndex14 && input_args[kInputIndex13]->BuildType()->type_id() != kMetaTypeNone) {
    auto mask_dtype = input_args[kInputIndex13]->BuildType();
    (void)CheckAndConvertUtils::CheckTensorTypeValid("mask_dtype", mask_dtype, valid_types, prim_name);
  }

  return std::make_shared<Tuple>(std::vector<TypePtr>{x_dtype, x_dtype, x_dtype, x_dtype, x_dtype, x_dtype});
}
}  // namespace

AbstractBasePtr DynamicGRUV2GradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                      const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const int64_t MinInputNum = 12;
  (void)CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, MinInputNum, prim_name);
  auto types = DynamicGRUV2GradInferType(primitive, input_args);
  auto shapes = DynamicGRUV2GradInferShape(primitive, input_args);
  return abstract::MakeAbstract(shapes, types);
}

MIND_API_OPERATOR_IMPL(DynamicGRUV2Grad, BaseOperator);

// AG means auto generated
class MIND_API AGDynamicGRUV2GradInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return DynamicGRUV2GradInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return DynamicGRUV2GradInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return DynamicGRUV2GradInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(DynamicGRUV2Grad, prim::kPrimDynamicGRUV2Grad, AGDynamicGRUV2GradInfer, false);
}  // namespace ops
}  // namespace mindspore
