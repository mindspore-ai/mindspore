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

#include "ops/sparse_segment_mean.h"

#include <functional>
#include <memory>
#include <numeric>
#include <set>

#include "abstract/ops/primitive_infer_map.h"
#include "ops/op_name.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr SparseSegmentMeanInferShape(const PrimitivePtr &prim,
                                               const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = prim->name();
  auto x_shape_ptr = input_args[kInputIndex0]->BuildShape();
  auto indices_shape_ptr = input_args[kInputIndex1]->BuildShape();
  auto segment_ids_shape_ptr = input_args[kInputIndex2]->BuildShape();
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(x_shape_ptr)[kShape];
  auto indices_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(indices_shape_ptr)[kShape];
  auto segment_ids_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(segment_ids_shape_ptr)[kShape];

  int64_t batch_rank = 0;
  if (prim->HasAttr(kBatchRank)) {
    auto batch_rank_ptr = prim->GetAttr(kBatchRank);
    batch_rank = GetValue<int64_t>(batch_rank_ptr);
  }

  if (x_shape_ptr->IsDimUnknown()) {
    constexpr int64_t unknown_shape = -2;
    return std::make_shared<abstract::Shape>(ShapeVector{unknown_shape});
  }

  constexpr int64_t number_one = 1;
  (void)CheckAndConvertUtils::CheckInteger("rank of 'x'", SizeToLong(x_shape.size()), kGreaterEqual,
                                           batch_rank + number_one, prim_name);
  (void)CheckAndConvertUtils::CheckInteger("rank of 'indices'", SizeToLong(indices_shape.size()), kEqual,
                                           batch_rank + number_one, prim_name);
  (void)CheckAndConvertUtils::CheckInteger("rank of 'segment_ids'", SizeToLong(segment_ids_shape.size()), kEqual,
                                           batch_rank + number_one, prim_name);
  if (!indices_shape_ptr->IsDynamic() && !segment_ids_shape_ptr->IsDynamic() &&
      indices_shape[kInputIndex0] != segment_ids_shape[kInputIndex0]) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name
                             << "', the size of 'indices' and 'segment_ids' must be the same, but got "
                             << indices_shape[kInputIndex0] << " vs " << segment_ids_shape[kInputIndex0] << ".";
  }
  ShapeVector out_shape = x_shape;
  if (!input_args[kInputIndex2]->BuildValue()->isa<tensor::Tensor>()) {
    // The real output shape relies on the last value of 'segment_ids', we have already added dependency map.
    // The framework ensures the `else` branch will be executed, so min/max shape are not necessary to set.
    out_shape[LongToSize(batch_rank)] = abstract::Shape::kShapeDimAny;
    return std::make_shared<abstract::Shape>(out_shape);
  } else {
    auto segment_ids = input_args[kInputIndex2]->cast<abstract::AbstractTensorPtr>();
    MS_EXCEPTION_IF_NULL(segment_ids);
    auto segment_ids_value = segment_ids->BuildValue();
    MS_EXCEPTION_IF_NULL(segment_ids_value);
    auto segment_ids_tensor = segment_ids_value->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(segment_ids_tensor);
    auto segment_ids_size = segment_ids_tensor->DataSize();
    auto expect_size = std::accumulate(segment_ids_shape.begin(), segment_ids_shape.end(), 1, std::multiplies{});
    MS_EXCEPTION_IF_CHECK_FAIL(segment_ids_size == LongToSize(expect_size),
                               "For '" + prim_name + "', something unexpected happened.");

    // Get last segment id and plus one as length of first dim of output.
    auto segment_ids_ptr = segment_ids_tensor->data_c();
    int64_t segment_num = 0;
    if (segment_ids_tensor->Dtype() == kInt32) {
      segment_num = IntToLong(*(reinterpret_cast<int *>(segment_ids_ptr) + segment_ids_size - 1) + 1);
    } else if (segment_ids_tensor->Dtype() == kInt64) {
      segment_num = *(reinterpret_cast<int64_t *>(segment_ids_ptr) + segment_ids_size - 1) + 1;
    }
    if (segment_num <= 0) {
      MS_LOG(EXCEPTION) << "For '" << prim_name << "', the input 'segment_ids' must be non-negative.";
    }
    out_shape[LongToSize(batch_rank)] = segment_num;
    return std::make_shared<abstract::Shape>(out_shape);
  }
}

TypePtr SparseSegmentMeanInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  auto x_type = input_args[kInputIndex0]->BuildType();
  auto indices_type = input_args[kInputIndex1]->BuildType();
  auto segment_ids_type = input_args[kInputIndex2]->BuildType();
  const std::set<TypePtr> valid_data_types = {kFloat16, kFloat32, kFloat64};
  const std::set<TypePtr> valid_index_types = {kInt32, kInt64};

  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", x_type, valid_data_types, prim->name());
  (void)CheckAndConvertUtils::CheckTensorTypeSame({{"indices", indices_type}, {"segment_ids", segment_ids_type}},
                                                  valid_index_types, prim->name());
  return x_type;
}
}  // namespace

AbstractBasePtr SparseSegmentMeanInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &prim,
                                       const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  constexpr int inputs_num = 3;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, inputs_num, prim->name());
  auto infer_type = SparseSegmentMeanInferType(prim, input_args);
  auto infer_shape = SparseSegmentMeanInferShape(prim, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

MIND_API_OPERATOR_IMPL(SparseSegmentMean, BaseOperator);

// AG means auto generated
class MIND_API AGSparseSegmentMeanInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return SparseSegmentMeanInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return SparseSegmentMeanInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return SparseSegmentMeanInfer(engine, primitive, input_args);
  }

  std::set<int64_t> GetValueDependArgIndices() const override { return {2}; }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(SparseSegmentMean, prim::kPrimSparseSegmentMean, AGSparseSegmentMeanInfer, false);
}  // namespace ops
}  // namespace mindspore
