/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "ops/split_v.h"
#include <set>

#include "ops/op_utils.h"
#include "utils/tensor_construct_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "utils/check_convert_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::TupleShapePtr SplitVInferShape(const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  (void)CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(input_args.size()), kEqual, 1L, prim_name);
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  auto x_rank = SizeToLong(x_shape.size());
  (void)CheckAndConvertUtils::CheckInteger("x_rank", x_rank, kGreaterEqual, 1, prim_name);
  auto split_dim = GetValue<int64_t>(primitive->GetAttr("split_dim"));
  CheckAndConvertUtils::CheckInRange("split_dim", split_dim, kIncludeLeft, {-x_rank, x_rank}, prim_name);
  if (split_dim < 0) {
    split_dim += x_rank;
  }
  auto shape_of_split_dim = x_shape[LongToSize(split_dim)];
  auto num_split = GetValue<int64_t>(primitive->GetAttr("num_split"));
  (void)CheckAndConvertUtils::CheckInteger("num_split", num_split, kGreaterEqual, 1, prim_name);
  auto size_splits = GetValue<std::vector<int64_t>>(primitive->GetAttr(kSizeSplits));
  CheckAndConvertUtils::Check("num_split", num_split, kEqual, SizeToLong(size_splits.size()), prim_name);
  auto default_idx = std::find(size_splits.begin(), size_splits.end(), -1);
  if (default_idx == size_splits.end()) {
    int64_t sum_of_size_splits = 0;
    for (int64_t i = 0; i < num_split; i++) {
      CheckAndConvertUtils::CheckInRange("elements of size_splits", size_splits[LongToSize(i)], kIncludeBoth,
                                         {0, shape_of_split_dim}, prim_name);
      sum_of_size_splits += size_splits[LongToSize(i)];
    }
    CheckAndConvertUtils::Check("sum of size_splits", sum_of_size_splits, kEqual, shape_of_split_dim, prim_name);
  } else {
    (void)size_splits.erase(default_idx);
    auto excessive_default_idx = std::find(size_splits.begin(), size_splits.end(), -1);
    if (excessive_default_idx != size_splits.end()) {
      MS_EXCEPTION(ValueError) << "For '" << prim_name
                               << "', 'size_splits' default value can contain only one -1, but got more than one.";
    } else {
      int64_t sum_of_size_splits = 0;
      for (int64_t i = 0; i < num_split - 1; i++) {
        CheckAndConvertUtils::CheckInRange("elements of size_splits", size_splits[LongToSize(i)], kIncludeBoth,
                                           {0, shape_of_split_dim}, prim_name);
        sum_of_size_splits += size_splits[LongToSize(i)];
      }
      auto default_value = shape_of_split_dim - sum_of_size_splits;
      (void)size_splits.insert(default_idx, default_value);
    }
  }
  std::vector<abstract::BaseShapePtr> shape_tuple;
  for (int64_t i = 0; i < num_split; i++) {
    auto shape = x_shape;
    shape[LongToSize(split_dim)] = size_splits[LongToSize(i)];
    abstract::ShapePtr out_shape = std::make_shared<abstract::Shape>(shape);
    shape_tuple.push_back(out_shape);
  }
  return std::make_shared<abstract::TupleShape>(shape_tuple);
}

TuplePtr SplitVInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto num_split = GetValue<int64_t>(prim->GetAttr("num_split"));
  auto infer_type = input_args[0]->BuildType();
  MS_EXCEPTION_IF_NULL(infer_type);
  const std::set<TypePtr> valid_types = {kInt8,   kInt16,  kInt32,  kInt64,   kUInt8,
                                         kUInt16, kUInt32, kUInt64, kFloat16, kFloat32};
  auto type = CheckAndConvertUtils::CheckTensorTypeValid("input_x", infer_type, valid_types, prim->name());
  std::vector<TypePtr> type_tuple;
  for (int64_t i = 0; i < num_split; i++) {
    type_tuple.push_back(type);
  }
  return std::make_shared<Tuple>(type_tuple);
}
}  // namespace

MIND_API_OPERATOR_IMPL(SplitV, BaseOperator);
AbstractBasePtr SplitVInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                            const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kSplitVInputsNum = 1;
  (void)CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kSplitVInputsNum, primitive->name());
  auto infertype = SplitVInferType(primitive, input_args);
  auto infershape = SplitVInferShape(primitive, input_args);
  return abstract::MakeAbstract(infershape, infertype);
}

// AG means auto generated
class MIND_API AGSplitVInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return SplitVInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return SplitVInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return SplitVInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(SplitV, prim::kPrimSplitV, AGSplitVInfer, false);
}  // namespace ops
}  // namespace mindspore
