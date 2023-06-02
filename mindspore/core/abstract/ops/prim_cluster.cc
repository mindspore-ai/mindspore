/**
 * Copyright 2022-2022 Huawei Technologies Co., Ltd
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

#include <cmath>
#include "abstract/ops/infer_functions.h"
#include "abstract/param_validator.h"
#include "abstract/utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace abstract {
AbstractBasePtr InferImplKMeansCentroids(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                         const AbstractBasePtrList &args_abs_list) {
  const std::string op_name = primitive->name();
  std::string error_mess_prefix = std::string("For ") + op_name + ", ";
  constexpr size_t size_args_expected = 4;
  CheckArgsSize(op_name, args_abs_list, size_args_expected);
  AbstractTensorPtr x = CheckArg<AbstractTensor>(op_name, args_abs_list, 0);
  AbstractTensorPtr y = CheckArg<AbstractTensor>(op_name, args_abs_list, 1);
  AbstractTensorPtr sum_square_y = CheckArg<AbstractTensor>(op_name, args_abs_list, 2);
  AbstractTensorPtr sum_square_x = CheckArg<AbstractTensor>(op_name, args_abs_list, 3);
  (void)CheckTensorDType(x, {kFloat32}, error_mess_prefix + "input x");
  (void)CheckTensorDType(y, {kFloat32}, error_mess_prefix + "input y");
  (void)CheckTensorDType(sum_square_y, {kFloat32}, error_mess_prefix + "input sum_square_y");
  (void)CheckTensorDType(sum_square_x, {kFloat32}, error_mess_prefix + "input sum_square_x");

  auto x_shape = x->shape()->shape();
  auto y_shape = y->shape()->shape();
  auto sum_square_y_shape = sum_square_y->shape()->shape();
  auto sum_square_x_shape = sum_square_x->shape()->shape();

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
                      << "but x's second dim get " << x_shape.at(1) << ", y's second dim get " << y_shape.at(1) << ".";
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
  ShapeVector segment_count_shape = {y_shape.at(0), 1};
  ShapeVector kmean_total_distance_shape = {1};

  auto segment_sum_shape_tensor =
    std::make_shared<AbstractTensor>(x->element(), std::make_shared<Shape>(segment_sum_shape));
  auto segment_count_shape_tensor =
    std::make_shared<AbstractTensor>(y->element(), std::make_shared<Shape>(segment_count_shape));
  auto kmean_total_distance_shape_tensor =
    std::make_shared<AbstractTensor>(y->element(), std::make_shared<Shape>(kmean_total_distance_shape));
  AbstractBasePtrList result = {segment_sum_shape_tensor, segment_count_shape_tensor,
                                kmean_total_distance_shape_tensor};
  return std::make_shared<AbstractTuple>(result);
}
}  // namespace abstract
}  // namespace mindspore
