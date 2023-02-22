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

#include "ops/dynamic_gru_v2.h"

#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include <utility>

#include "abstract/abstract_value.h"
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
#include "ir/value.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/base/type_id.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::TupleShapePtr DynamicGRUV2InferShape(const PrimitivePtr &primitive,
                                               const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  auto winput_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
  auto whidden_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->BuildShape())[kShape];
  auto h_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex6]->BuildShape())[kShape];

  auto winput_shape_ptr = input_args[kInputIndex1]->BuildShape();
  auto whidden_shape_ptr = input_args[kInputIndex2]->BuildShape();
  auto h_shape_ptr = input_args[kInputIndex6]->BuildShape();

  std::vector<ShapeVector> all_shapes = {x_shape, winput_shape, whidden_shape, h_shape};
  auto is_dynamic_rank = std::any_of(all_shapes.begin(), all_shapes.end(), IsDynamicRank);
  auto is_dynamic = std::any_of(all_shapes.begin(), all_shapes.end(), IsDynamic);

  int64_t num_proj = 0;
  if (primitive->HasAttr(kNumProj)) {
    num_proj = GetValue<int64_t>(primitive->GetAttr(kNumProj));
  }

  const size_t kNumTwo = 2;
  const size_t kNumThree = 3;
  if (!is_dynamic_rank) {
    (void)CheckAndConvertUtils::CheckInteger("x shape rank", x_shape.size(), kEqual, kNumThree, prim_name);
    (void)CheckAndConvertUtils::CheckInteger("weight input shape rank", winput_shape.size(), kEqual, kNumTwo,
                                             prim_name);
    (void)CheckAndConvertUtils::CheckInteger("weight hidden shape rank", whidden_shape.size(), kEqual, kNumTwo,
                                             prim_name);
    (void)CheckAndConvertUtils::CheckInteger("h shape rank", h_shape.size(), kEqual, kNumTwo, prim_name);
  }

  std::map<size_t, bool> placeholder_map = {{3, true}, {4, true}, {5, true}};
  if (!is_dynamic) {
    int64_t batch_size = x_shape[kInputIndex1];
    int64_t input_size = x_shape[kInputIndex2];
    int64_t hidden_size = whidden_shape[kInputIndex0];

    (void)CheckAndConvertUtils::CheckTensorShapeSame({{"weight input shape", winput_shape_ptr}},
                                                     std::vector<int64_t>{input_size, 3 * hidden_size}, prim_name);
    (void)CheckAndConvertUtils::CheckTensorShapeSame({{"weight hidden shape", whidden_shape_ptr}},
                                                     std::vector<int64_t>{hidden_size, 3 * hidden_size}, prim_name);
    (void)CheckAndConvertUtils::CheckTensorShapeSame({{"init h shape", h_shape_ptr}},
                                                     std::vector<int64_t>{batch_size, hidden_size}, prim_name);

    std::vector<int64_t> valid_shape = {3 * hidden_size};
    if (input_args[kInputIndex3]->BuildType()->type_id() != kMetaTypeNone) {
      auto binput_shape =
        CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex3]->BuildShape())[kShape];
      auto binput_shape_ptr = input_args[kInputIndex3]->BuildShape();
      if (!IsDynamic(binput_shape)) {
        (void)CheckAndConvertUtils::CheckTensorShapeSame({{"binput_shape", binput_shape_ptr}}, valid_shape, prim_name);
        placeholder_map[kInputIndex3] = false;
      }
    }

    if (input_args[kInputIndex4]->BuildType()->type_id() != kMetaTypeNone) {
      auto bhidden_shape =
        CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex4]->BuildShape())[kShape];
      auto bhidden_shape_ptr = input_args[kInputIndex4]->BuildShape();
      if (!IsDynamic(bhidden_shape)) {
        (void)CheckAndConvertUtils::CheckTensorShapeSame({{"bhidden_shape", bhidden_shape_ptr}}, valid_shape,
                                                         prim_name);
        placeholder_map[kInputIndex4] = false;
      }
    }

    if (input_args[kInputIndex5]->BuildType()->type_id() != kMetaTypeNone) {
      auto seq_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex5]->BuildShape())[kShape];
      MS_EXCEPTION(ValueError) << "For '" << prim_name << "', the dimension of 'seq_length' must be None, but got "
                               << seq_shape << ".";
    }
  }

  std::vector<int64_t> placeholder{};
  for (auto iter = placeholder_map.begin(); iter != placeholder_map.end(); iter++) {
    if (iter->second) {
      placeholder.emplace_back(static_cast<int64_t>(iter->first));
    }
  }
  (void)primitive->AddAttr("placeholder_index", MakeValue<std::vector<int64_t>>(placeholder));

  ShapeVector y_shape = {-1, -1, -1};
  ShapeVector out_shape = {-1, -1, -1};
  const int64_t kNumZero = 0;
  if (!(IsDynamic(x_shape) || IsDynamic(whidden_shape))) {
    y_shape[kInputIndex0] = x_shape[kInputIndex0];
    y_shape[kInputIndex1] = x_shape[kInputIndex1];
    y_shape[kInputIndex2] =
      num_proj > kNumZero ? std::min(num_proj, whidden_shape[kInputIndex0]) : whidden_shape[kInputIndex0];
    out_shape[kInputIndex0] = x_shape[kInputIndex0];
    out_shape[kInputIndex1] = x_shape[kInputIndex1];
    out_shape[kInputIndex2] = whidden_shape[kInputIndex0];
  }
  auto y_shape_ptr = std::make_shared<abstract::Shape>(y_shape);
  auto out_shape_ptr = std::make_shared<abstract::Shape>(out_shape);
  return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{
    y_shape_ptr, out_shape_ptr, out_shape_ptr, out_shape_ptr, out_shape_ptr, out_shape_ptr});
}

TuplePtr DynamicGRUV2InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto x_dtype = input_args[kInputIndex0]->BuildType();
  auto winput_dtype = input_args[kInputIndex1]->BuildType();
  auto whidden_dtype = input_args[kInputIndex2]->BuildType();
  auto h_dtype = input_args[kInputIndex6]->BuildType();

  std::map<std::string, TypePtr> check_types = {
    {"x_dtype", x_dtype}, {"winput_dtype", winput_dtype}, {"whidden_dtype", whidden_dtype}};
  (void)CheckAndConvertUtils::CheckTensorTypeSame(check_types, {kFloat16}, prim_name);

  const std::set<TypePtr> valid_types = {kFloat16, kFloat32};
  std::map<std::string, TypePtr> check_types_h;
  check_types_h.insert({"init_h", h_dtype});
  if (input_args[kInputIndex3]->BuildType()->type_id() != kMetaTypeNone) {
    auto binput_dtype = input_args[kInputIndex3]->BuildType();
    check_types_h.insert({"bias_input", binput_dtype});
  }
  if (input_args[kInputIndex4]->BuildType()->type_id() != kMetaTypeNone) {
    auto bhidden_dtype = input_args[kInputIndex4]->BuildType();
    check_types_h.insert({"bias_hidden", bhidden_dtype});
  }
  (void)CheckAndConvertUtils::CheckTensorTypeSame(check_types_h, valid_types, prim_name);

  return std::make_shared<Tuple>(std::vector<TypePtr>{h_dtype, h_dtype, h_dtype, h_dtype, h_dtype, h_dtype});
}
}  // namespace

AbstractBasePtr DynamicGRUV2Infer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                  const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  for (auto &input : input_args) {
    MS_EXCEPTION_IF_NULL(input);
  }
  const size_t kInputNum = 7;
  (void)CheckAndConvertUtils::CheckInteger("Input Num", input_args.size(), kEqual, kInputNum, prim_name);
  auto types = DynamicGRUV2InferType(primitive, input_args);
  auto shapes = DynamicGRUV2InferShape(primitive, input_args);
  return abstract::MakeAbstract(shapes, types);
}

MIND_API_OPERATOR_IMPL(DynamicGRUV2, BaseOperator);

// AG means auto generated
class MIND_API AGDynamicGRUV2Infer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return DynamicGRUV2InferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return DynamicGRUV2InferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return DynamicGRUV2Infer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(DynamicGRUV2, prim::kPrimDynamicGRUV2, AGDynamicGRUV2Infer, false);
}  // namespace ops
}  // namespace mindspore
