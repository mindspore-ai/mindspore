/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include "ops/ops_func_impl/lstsq_v2.h"
#include <memory>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {

bool BroadcastLstsq(const ShapeVector &a_batch_vec, const ShapeVector &b_batch_vec, ShapeVector *broadcast_vec_ptr) {
  constexpr bool success = true;
  constexpr bool fail = false;
  for (size_t i = 0; i < a_batch_vec.size(); i++) {
    if (a_batch_vec[i] == b_batch_vec[i]) {
      broadcast_vec_ptr->emplace_back(a_batch_vec[i]);
    } else {
      auto max_dim = a_batch_vec[i] > b_batch_vec[i] ? a_batch_vec[i] : b_batch_vec[i];
      auto min_dim = a_batch_vec[i] < b_batch_vec[i] ? a_batch_vec[i] : b_batch_vec[i];
      if (min_dim == -1 || min_dim == 1) {
        broadcast_vec_ptr->emplace_back(max_dim);
      } else {
        return fail;
      }
    }
  }
  return success;
}

DriverName GetDriver(const std::vector<AbstractBasePtr> &input_args) {
  DriverName driver = DriverName::GELSY;
  auto driver_vnode = input_args[kIndex2]->GetValue();
  if (driver_vnode->ToString() != "None") {
    auto driver_opt = GetScalarValue<int64_t>(driver_vnode);
    if (driver_opt.has_value()) {
      driver = static_cast<DriverName>(driver_opt.value());
    }
  }
  return driver;
}

BaseShapePtr LstsqV2FuncImpl::InferShape(const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) const {
  auto input_a_shape = input_args[kIndex0]->GetShape()->GetShapeVector();
  auto input_b_shape = input_args[kIndex1]->GetShape()->GetShapeVector();
  auto a_rank = input_a_shape.size();
  auto b_rank = input_b_shape.size();

  constexpr size_t mat_size = 2;
  constexpr size_t vec_size = 1;
  const size_t expected_b_dim = (b_rank == a_rank - 1) ? vec_size : mat_size;

  if (IsDynamicRank(input_a_shape) || IsDynamicRank(input_b_shape)) {
    auto unknow_soluition =
      std::make_shared<abstract::TensorShape>(std::vector<int64_t>{abstract::Shape::kShapeRankAny});
    auto unknow_resition =
      std::make_shared<abstract::TensorShape>(std::vector<int64_t>{abstract::Shape::kShapeRankAny});
    auto unknow_rank = std::make_shared<abstract::TensorShape>(std::vector<int64_t>{abstract::Shape::kShapeRankAny});
    auto unknow_singular_value =
      std::make_shared<abstract::TensorShape>(std::vector<int64_t>{abstract::Shape::kShapeRankAny});
    return std::make_shared<abstract::TupleShape>(
      std::vector<abstract::BaseShapePtr>{unknow_soluition, unknow_resition, unknow_rank, unknow_singular_value});
  }

  if (a_rank < mat_size) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                             << "', dim of matrix a must greater or equal to 2, but got a at " << a_rank
                             << "-dimensional ";
  }
  if (a_rank != b_rank && a_rank != b_rank + 1) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                             << "', the dimension of `b` should be 'a.dim' or 'a.dim' - 1, which is " << a_rank
                             << " or " << (a_rank - 1) << ", but got " << b_rank << "-dimensions.";
  }
  if (input_a_shape[a_rank - mat_size] != input_b_shape[b_rank - expected_b_dim]) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                             << "', the last two dimensions of `a` and `b` should be matched, but got shape of "
                             << input_a_shape << " and " << input_b_shape
                             << ". Please make sure that the shape of `a` and `b` be like [..., M, N] X [..., M, K] or "
                                "[..., M, N] X[..., M].";
  }
  ShapeVector a_batch_vec(input_a_shape.begin(), input_a_shape.end() - mat_size);
  ShapeVector b_batch_vec(input_b_shape.begin(), input_b_shape.end() - expected_b_dim);
  ShapeVector broadcast_vec;
  bool broadcast_success = BroadcastLstsq(a_batch_vec, b_batch_vec, &broadcast_vec);
  if (!broadcast_success) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                             << "', the batch dimension of `a` and `b` should can be broadcast', but got a batch:"
                             << a_batch_vec << " and b batch:" << b_batch_vec << '.';
  }
  auto driver = GetDriver(input_args);
  auto m = input_a_shape[a_rank - 2];
  auto n = input_a_shape[a_rank - 1];
  auto k = b_rank == a_rank ? input_b_shape[b_rank - 1] : 1;
  bool calculate_res = (m == -1 || n == -1 || m > n) && driver != DriverName::GELSY;
  bool calculate_rank = driver != DriverName::GELS;
  bool calculate_singular_value = driver == DriverName::GELSS || driver == DriverName::GELSD;
  ShapeVector output_solusion_shape_vec(broadcast_vec);
  ShapeVector output_res_shape_vec;
  ShapeVector output_rank_shape_vec;
  ShapeVector output_singular_shape_vec;
  output_solusion_shape_vec.emplace_back(n);
  if (calculate_res) {
    output_res_shape_vec = ShapeVector(broadcast_vec);
    output_res_shape_vec.emplace_back(k);
  } else {
    output_res_shape_vec.emplace_back(0);
  }
  if (calculate_rank) {
    output_rank_shape_vec = ShapeVector(a_batch_vec);
  } else {
    output_rank_shape_vec.emplace_back(0);
  }
  if (calculate_singular_value) {
    output_singular_shape_vec = ShapeVector(a_batch_vec);
    output_singular_shape_vec.emplace_back(m < n ? m : n);
  } else {
    output_singular_shape_vec.emplace_back(0);
  }
  if (expected_b_dim == 2) {
    output_solusion_shape_vec.emplace_back(k);
  }

  auto output_solusion_shape = std::make_shared<abstract::TensorShape>(output_solusion_shape_vec);
  auto output_res_shape = std::make_shared<abstract::TensorShape>(output_res_shape_vec);
  auto output_rank_shape = std::make_shared<abstract::TensorShape>(output_rank_shape_vec);
  auto output_singular_shape = std::make_shared<abstract::TensorShape>(output_singular_shape_vec);

  return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{
    output_solusion_shape, output_res_shape, output_rank_shape, output_singular_shape});
}

TypePtr LstsqV2FuncImpl::InferType(const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args) const {
  auto input_a_type = input_args[kIndex0]->GetType()->cast<TensorTypePtr>();
  auto input_b_type = input_args[kIndex1]->GetType()->cast<TensorTypePtr>();
  MS_EXCEPTION_IF_NULL(input_a_type);
  auto input_a_type_id = input_a_type->element()->type_id();
  MS_EXCEPTION_IF_NULL(input_b_type);
  auto input_b_type_id = input_b_type->element()->type_id();
  if (input_a_type_id != input_b_type_id) {
    MS_EXCEPTION(TypeError) << "For '" << primitive->name()
                            << "' the type of a and b must be same, but got type of a is different from that of b! ";
  }
  auto out_solution_type = input_a_type->Clone();
  auto out_res_type = input_a_type->Clone();
  auto out_rank_type = std::make_shared<TensorType>(kInt64);
  auto out_singular_type = input_a_type->Clone();
  if (input_a_type_id == kNumberTypeComplex64) {
    out_res_type = std::make_shared<TensorType>(kFloat32);
    out_singular_type = std::make_shared<TensorType>(kFloat32);
  } else if (input_a_type_id == kNumberTypeComplex128) {
    out_res_type = std::make_shared<TensorType>(kFloat64);
    out_singular_type = std::make_shared<TensorType>(kFloat64);
  }
  return std::make_shared<Tuple>(
    std::vector<TypePtr>{out_solution_type, out_res_type, out_rank_type, out_singular_type});
}
}  // namespace ops
}  // namespace mindspore
