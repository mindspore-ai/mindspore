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
#include "abstract/infer_functions.h"
#include "abstract/param_validator.h"
#include "abstract/utils.h"
#include "utils/check_convert_utils.h"
#include "utils/shape_utils.h"

namespace mindspore {
namespace abstract {
AbstractBasePtr InferImplKMeansCentroids(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                         const AbstractBasePtrList &args_spec_list) {
  const std::string op_name = primitive->name();
  constexpr size_t size_args_expected = 4;
  CheckArgsSize(op_name, args_spec_list, size_args_expected);
  AbstractTensorPtr x = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);
  AbstractTensorPtr y = CheckArg<AbstractTensor>(op_name, args_spec_list, 1);
  AbstractTensorPtr sum_square_y = CheckArg<AbstractTensor>(op_name, args_spec_list, 2);
  AbstractTensorPtr sum_square_x = CheckArg<AbstractTensor>(op_name, args_spec_list, 3);
  (void)CheckTensorDType(x, {kFloat32}, "Input x of k_means_centroids should be %s");
  (void)CheckTensorDType(y, {kFloat32}, "Input y of k_means_centroids should be %s");
  (void)CheckTensorDType(sum_square_y, {kFloat32}, "Input sum_square_y of k_means_centroids should be %s");
  (void)CheckTensorDType(sum_square_x, {kFloat32}, "Input sum_square_x of k_means_centroids should be %s");

  auto x_shape = x->shape()->shape();
  auto y_shape = y->shape()->shape();
  auto sum_square_y_shape = sum_square_y->shape()->shape();
  auto sum_square_x_shape = sum_square_x->shape()->shape();

  constexpr size_t expected_shape_size = 2;
  if ((x_shape.size() != expected_shape_size) || (y_shape.size() != expected_shape_size) ||
      (sum_square_y_shape.size() != expected_shape_size) || (sum_square_x_shape.size() != expected_shape_size)) {
    MS_LOG(EXCEPTION) << "Rank of  " << op_name << "'s x, y, sum_square_y, sum_square_x must be 2.";
  }
  if (x_shape.at(1) != y_shape.at(1)) {
    MS_LOG(EXCEPTION) << "x of " << op_name << "'s second dim must be equal to y's second dim.";
  }
  if (y_shape.at(0) != sum_square_y_shape.at(1)) {
    MS_LOG(EXCEPTION) << "y of " << op_name << "'s first dim must be equal to sum_square_y's second dim.";
  }
  if (x_shape.at(0) != sum_square_x_shape.at(0)) {
    MS_LOG(EXCEPTION) << "x of " << op_name << "'s first dim must be equal to sum_square_x's first dim.";
  }
  if ((sum_square_y_shape.at(0) != sum_square_x_shape.at(1)) || (sum_square_y_shape.at(0) != 1)) {
    MS_LOG(EXCEPTION) << "sum_square_x of " << op_name
                      << "'s first dim must be equal to the first dim of sum_square_x, and they must be 1.";
  }

  auto segment_sum_shape = y_shape;
  auto segment_count_shape = {y_shape.at(0), 1};
  auto kmean_total_distance_shape = {1};

  auto segment_sum_shape_tensor =
    std::make_shared<AbstractTensor>(x->element(), std::make_shared<Shape>(segment_sum_shape));
  auto segment_count_shape_tensor =
    std::make_shared<AbstractTensor>(y->element(), std::make_shared<Shape>(segment_count_shape));
  auto kmean_total_distance_shape_tensor =
    std::make_shared<AbstractTensor>(y->element(), std::make_shared<Shape>(kmean_total_distance_shape));
  AbstractBasePtrList result = {segment_sum_shape_tensor, segment_count_shape_tensor,
                                kmean_total_distance_shape_tensor};
  return result;
}

}  // namespace abstract
}  // namespace mindspore
