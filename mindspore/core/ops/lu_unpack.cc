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

#include "ops/lu_unpack.h"

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
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
constexpr size_t kLuUnpackInputsNum = 2;
constexpr size_t kLuDataRank = 2;
constexpr size_t kLuPivotsRank = 1;
constexpr size_t kFirstDim = 1;
constexpr size_t kSecondDim = 2;
abstract::TupleShapePtr LuUnpackInferShape(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  int64_t input_size = static_cast<int64_t>(input_args.size());
  (void)CheckAndConvertUtils::CheckInteger("input number", input_size, kEqual, kLuUnpackInputsNum, prim_name);
  for (const auto &i : input_args) {
    MS_EXCEPTION_IF_NULL(i);
  }
  auto LU_data_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape());
  auto LU_data_shape = LU_data_shape_map[kShape];
  auto LU_pivots_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->BuildShape());
  auto LU_pivots_shape = LU_pivots_shape_map[kShape];

  if (IsDynamicRank(LU_data_shape)) {
    auto output_shpe = std::make_shared<abstract::Shape>(LU_data_shape);
    return std::make_shared<abstract::TupleShape>(
      std::vector<abstract::BaseShapePtr>{output_shpe, output_shpe, output_shpe});
  }

  if (LU_data_shape.size() < kLuDataRank) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "',"
                             << " the dimension of LU_data must be greater than or equal to 2, but got: "
                             << LU_data_shape.size() << ".";
  }

  if (!IsDynamicShape(LU_data_shape) && !IsDynamic(LU_pivots_shape)) {
    if (LU_pivots_shape.size() < kLuPivotsRank) {
      MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "',"
                               << " the dimension of LU_pivots must be greater than or equal to 1, but got: "
                               << LU_pivots_shape.size() << ".";
    }
    if (LU_pivots_shape[LU_pivots_shape.size() - 1] !=
        std::min(LU_data_shape[LU_data_shape.size() - kFirstDim], LU_data_shape[LU_data_shape.size() - kSecondDim])) {
      MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', "
                               << "the last dimension of LU_pivots must be the same as the minimum value of the last "
                                  "two dimensions of LU_data,"
                               << " but got lu_pivots': " << LU_pivots_shape[LU_pivots_shape.size() - 1]
                               << ", the minimum value of the last two dimensions of LU_data: "
                               << std::min(LU_data_shape[LU_data_shape.size() - kFirstDim],
                                           LU_data_shape[LU_data_shape.size() - kSecondDim])
                               << ".";
    }
    for (size_t i = 0; i < LU_pivots_shape.size() - 1; i++) {
      if (LU_data_shape[i] != LU_pivots_shape[i]) {
        MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "',"
                                 << " the batch dimensions of LU_data's does not match LU_pivots's batch dimensions.";
      }
    }
  }

  auto pivots_vector = LU_data_shape;
  auto L_vector = LU_data_shape;
  auto U_vector = LU_data_shape;
  if (LU_data_shape[LU_data_shape.size() - kSecondDim] != LU_data_shape[LU_data_shape.size() - kFirstDim]) {
    pivots_vector[LU_data_shape.size() - kFirstDim] = LU_data_shape[LU_data_shape.size() - kSecondDim];
  }
  // L_shape
  if (LU_data_shape[LU_data_shape.size() - kSecondDim] < LU_data_shape[LU_data_shape.size() - kFirstDim]) {
    L_vector[LU_data_shape.size() - kFirstDim] = LU_data_shape[LU_data_shape.size() - kSecondDim];
  }
  // U_shape
  if (LU_data_shape[LU_data_shape.size() - kSecondDim] > LU_data_shape[LU_data_shape.size() - kFirstDim]) {
    U_vector[LU_data_shape.size() - kSecondDim] = LU_data_shape[LU_data_shape.size() - kFirstDim];
  }
  auto pivots_vector_output = std::make_shared<abstract::Shape>(pivots_vector);
  auto L_vector_output = std::make_shared<abstract::Shape>(L_vector);
  auto U_vector_output = std::make_shared<abstract::Shape>(U_vector);
  return std::make_shared<abstract::TupleShape>(
    std::vector<abstract::BaseShapePtr>{pivots_vector_output, L_vector_output, U_vector_output});
}

TuplePtr LuUnpackInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = prim->name();
  MS_EXCEPTION_IF_NULL(prim);
  int64_t input_size = static_cast<int64_t>(input_args.size());
  (void)CheckAndConvertUtils::CheckInteger("input number", input_size, kEqual, kLuUnpackInputsNum, prim_name);
  for (const auto &i : input_args) {
    MS_EXCEPTION_IF_NULL(i);
  }
  auto LU_data_type = input_args[0]->BuildType();
  auto LU_pivots_type = input_args[1]->BuildType();

  const std::set<TypePtr> valid_types_1 = {kFloat64, kFloat32, kFloat16, kInt64, kInt32, kInt16, kInt8, kUInt8};
  const std::set<TypePtr> valid_types_2 = {kInt64, kInt32, kInt16, kInt8, kUInt8};
  std::map<std::string, TypePtr> input_types;
  (void)input_types.emplace("LU_data", LU_data_type);
  (void)input_types.emplace("LU_pivots", LU_pivots_type);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("LU_data", input_args[0]->BuildType(), valid_types_1, prim->name());
  (void)CheckAndConvertUtils::CheckTensorTypeValid("LU_pivots", input_args[1]->BuildType(), valid_types_2,
                                                   prim->name());
  return std::make_shared<Tuple>(std::vector<TypePtr>{LU_data_type, LU_data_type, LU_data_type});
}
}  // namespace

AbstractBasePtr LuUnpackInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                              const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto infer_type = LuUnpackInferType(primitive, input_args);
  auto infer_shape = LuUnpackInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

MIND_API_OPERATOR_IMPL(LuUnpack, BaseOperator);

// AG means auto generated
class MIND_API AGLuUnpackInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return LuUnpackInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return LuUnpackInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return LuUnpackInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(LuUnpack, prim::kPrimLuUnpack, AGLuUnpackInfer, false);
}  // namespace ops
}  // namespace mindspore
