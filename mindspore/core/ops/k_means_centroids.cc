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
#include <vector>

#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/primitive_infer_map.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/src/helper.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/log_adapter.h"
#include "abstract/param_validator.h"
#include "ops/other_ops.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
class MIND_API KMeansCentroidsInfer : public abstract::OpInferBase {
 public:
  // This is used for backend infer by kernel tensor.
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    const std::string op_name = primitive->name();
    std::string error_mess_prefix = std::string("For ") + op_name + ", ";
    constexpr size_t size_args_expected = 4;
    CheckArgsSize(op_name, input_args, size_args_expected);
    auto x = input_args[kIndex0];
    auto y = input_args[kIndex1];
    auto sum_square_y = input_args[kIndex2];
    auto sum_square_x = input_args[kIndex3];

    auto x_shape = x->GetShape()->GetShapeVector();
    auto y_shape = y->GetShape()->GetShapeVector();
    auto sum_square_y_shape = sum_square_y->GetShape()->GetShapeVector();
    auto sum_square_x_shape = sum_square_x->GetShape()->GetShapeVector();

    constexpr size_t expected_shape_size = 2;
    if ((x_shape.size() != expected_shape_size) || (y_shape.size() != expected_shape_size) ||
        (sum_square_y_shape.size() != expected_shape_size) || (sum_square_x_shape.size() != expected_shape_size)) {
      MS_LOG(EXCEPTION) << error_mess_prefix << "rank of  "
                        << "x, y, sum_square_y, sum_square_x should be 2, "
                        << "but rank of x get " << x_shape.size() << ", "
                        << "rank of y get " << y_shape.size() << ", "
                        << "rank of sum_square_y get " << sum_square_y_shape.size() << ", "
                        << "rank of sum_square_x get " << sum_square_x_shape.size() << ".";
    }
    if (x_shape.at(1) != y_shape.at(1)) {
      MS_LOG(EXCEPTION) << error_mess_prefix << "x's second dim should be equal to y's second dim, "
                        << "but x's second dim get " << x_shape.at(1) << ", y's second dim get " << y_shape.at(1)
                        << ".";
    }
    if (y_shape.at(0) != sum_square_y_shape.at(1)) {
      MS_LOG(EXCEPTION) << error_mess_prefix << "y's first dim should be equal to sum_square_y's second dim, "
                        << "but y's first dim get " << y_shape.at(0) << ", sum_square_y's second dim get "
                        << sum_square_y_shape.at(1) << ".";
    }
    if (x_shape.at(0) != sum_square_x_shape.at(0)) {
      MS_LOG(EXCEPTION) << error_mess_prefix << "x's first dim should be equal to sum_square_x's first dim, "
                        << "but x's first dim get " << x_shape.at(0) << ", sum_square_x's first dim get "
                        << sum_square_x_shape.at(0) << ".";
    }
    if ((sum_square_y_shape.at(0) != sum_square_x_shape.at(1)) || (sum_square_y_shape.at(0) != 1)) {
      MS_LOG(EXCEPTION) << error_mess_prefix
                        << "sum_square_x's first dim must be equal to the second dim of sum_square_x, "
                        << "and they should be 1, but sum_square_y's first dim get " << sum_square_y_shape.at(0)
                        << ", sum_square_x's second dim get " << sum_square_x_shape.at(1) << ".";
    }

    ShapeVector segment_sum_shape = y_shape;
    ShapeVector segment_count_shape = {y_shape[0], 1};
    ShapeVector kmean_total_distance_shape = {1};

    auto segment_sum_shape_ptr = std::make_shared<abstract::TensorShape>(segment_sum_shape);
    auto segment_count_shape_ptr = std::make_shared<abstract::TensorShape>(segment_count_shape);
    auto kmean_total_distance_shape_ptr = std::make_shared<abstract::TensorShape>(kmean_total_distance_shape);
    abstract::BaseShapePtrList result = {segment_sum_shape_ptr, segment_count_shape_ptr,
                                         kmean_total_distance_shape_ptr};
    return std::make_shared<abstract::TupleShape>(result);
  }

  // This is used for backend infer by kernel tensor.
  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    const std::string op_name = primitive->name();
    std::string error_mess_prefix = std::string("For ") + op_name + ", ";
    constexpr size_t size_args_expected = 4;
    CheckArgsSize(op_name, input_args, size_args_expected);
    auto x = CheckAndConvertUtils::CheckArgsType(op_name, input_args, kIndex0, kObjectTypeTensorType);
    auto y = CheckAndConvertUtils::CheckArgsType(op_name, input_args, kIndex1, kObjectTypeTensorType);
    auto sum_square_y = CheckAndConvertUtils::CheckArgsType(op_name, input_args, kIndex2, kObjectTypeTensorType);
    auto sum_square_x = CheckAndConvertUtils::CheckArgsType(op_name, input_args, kIndex3, kObjectTypeTensorType);

    (void)abstract::CheckTensorDType(x, {kFloat32}, error_mess_prefix + "input x");
    (void)abstract::CheckTensorDType(y, {kFloat32}, error_mess_prefix + "input y");
    (void)abstract::CheckTensorDType(sum_square_y, {kFloat32}, error_mess_prefix + "input sum_square_y");
    (void)abstract::CheckTensorDType(sum_square_x, {kFloat32}, error_mess_prefix + "input sum_square_x");

    auto segment_sum_type = x->GetType()->Clone();
    auto segment_count_type = y->GetType()->Clone();
    auto kmean_total_distance_type = y->GetType()->Clone();
    return std::make_shared<Tuple>(TypePtrList{segment_sum_type, segment_count_type, kmean_total_distance_type});
  }
};

class MIND_API KMeansCentroids : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(KMeansCentroids);
  /// \brief Constructor.
  KMeansCentroids() : BaseOperator("KMeansCentroids") {}
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(KMeansCentroids, prim::kPrimKMeansCentroids, KMeansCentroidsInfer, false);
}  // namespace ops
}  // namespace mindspore
