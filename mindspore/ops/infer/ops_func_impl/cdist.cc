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

#include "infer/ops_func_impl/cdist.h"
#include <memory>
#include <cmath>
#include <set>

#include "utils/check_convert_utils.h"
#include "primitive/op_name.h"

namespace mindspore {
namespace ops {
constexpr size_t kCdistInputDimsMin = 2;

static inline bool CdistIsValidType(TypeId t) {
  static const std::set<TypeId> valid_types = {kNumberTypeFloat16, kNumberTypeFloat32, kNumberTypeFloat64};
  return valid_types.find(t) != valid_types.end();
}

ShapeArray CdistFuncImpl::InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  int64_t batch_rank = 0;
  auto prim_name = primitive->name();
  if (primitive->HasAttr(kBatchRank)) {
    auto value_ptr = primitive->GetAttr(kBatchRank);
    batch_rank = GetValue<int64_t>(value_ptr);
  }

  auto x_shape = input_infos[kInputIndex0]->GetShape();
  auto y_shape = input_infos[kInputIndex1]->GetShape();
  auto x_size = x_shape.size();
  auto y_size = y_shape.size();
  if (IsDynamicRank(x_shape) || IsDynamicRank(y_shape)) {
    return {ShapeVector{abstract::Shape::kShapeRankAny}};
  }
  if (x_size != y_size) {
    MS_EXCEPTION(ValueError) << "For Primitive[" << prim_name
                             << "], rank of input_x and input_y must be equal, but got rank of input_x: " << x_size
                             << ", rank of input_y: " << y_size << ".";
  }
  if (batch_rank == 0) {
    CheckAndConvertUtils::CheckInRange("input_x dim", x_size, kIncludeBoth, {2, 3}, "CdistGrad");
  }
  if (x_size < kCdistInputDimsMin) {
    MS_EXCEPTION(ValueError) << "For Primitive[" << prim_name << "], rank of input must be greater than "
                             << kCdistInputDimsMin << ", but got rank of input: " << x_size << ".";
  }

  auto out_shape = x_shape;
  if (x_size > kCdistInputDimsMin) {
    for (size_t i = 0; i < x_size - kCdistInputDimsMin; i++) {
      if (x_shape[i] == abstract::TensorShape::kShapeDimAny || y_shape[i] == abstract::TensorShape::kShapeDimAny) {
        continue;
      } else if (x_shape[i] == 1) {
        out_shape[i] = y_shape[i];
      } else if (y_shape[i] == 1) {
        continue;
      } else if (x_shape[i] != y_shape[i]) {
        MS_EXCEPTION(ValueError) << "For Primitive[" << prim_name
                                 << "] The two input shape can not broadcast, x_shape: " << x_shape << ", y_shape"
                                 << y_shape;
      }
    }
  }
  if (x_shape[x_size - kInputIndex1] != y_shape[y_size - kInputIndex1]) {
    MS_EXCEPTION(ValueError) << "For Primitive[" << prim_name
                             << "], the number of columns of 'x' must be the same as the number of 'y', "
                                "but got 'x_shape["
                             << x_size - kInputIndex1 << "]': " << x_shape[x_size - kInputIndex1] << " and 'y_shape["
                             << y_size - kInputIndex1 << "]': " << y_shape[y_size - kInputIndex1];
  }
  int64_t dim_R = y_shape[y_size - kCdistInputDimsMin];
  out_shape.pop_back();
  out_shape.push_back(dim_R);
  return {out_shape};
}

std::vector<TypeId> CdistFuncImpl::InferType(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  auto prim_name = primitive->name();
  const auto p_value = input_infos[kInputIndex2]->GetScalarValue<pyfloat>();
  if (p_value < 0 || std::isnan(p_value.value())) {
    MS_EXCEPTION(ValueError) << "For Primitive[" << prim_name << "], p must be a non-negative value, but got \n"
                             << p_value.value() << ".";
  }
  const auto input_x_type = input_infos[kInputIndex0]->GetType();
  const auto input_y_type = input_infos[kInputIndex1]->GetType();
  if (input_x_type != input_y_type) {
    MS_EXCEPTION(TypeError) << "For Primitive[" << prim_name << "], the input type must be same.\n"
                            << "name:[input_x]:Tensor[" << TypeIdToString(input_x_type) << "].\n"
                            << "name:[input_y]:Tensor[" << TypeIdToString(input_y_type) << "].\n";
  }
  if (!CdistIsValidType(input_x_type)) {
    MS_EXCEPTION(TypeError) << "For Primitive[" << prim_name
                            << "], the type of the input_x tensor must be [Float16, Float32, Float64], but got "
                            << TypeIdToString(input_x_type) << "!";
  }
  return {input_x_type};
}
}  // namespace ops
}  // namespace mindspore
