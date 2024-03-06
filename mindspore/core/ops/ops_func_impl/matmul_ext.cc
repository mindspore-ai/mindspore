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

#include "ops/ops_func_impl/matmul_ext.h"
#include <algorithm>
#include <set>
#include <map>
#include <vector>
#include <memory>
#include <string>
#include "ops/op_name.h"
#include "utils/shape_utils.h"
#include "abstract/dshape.h"
#include "ir/primitive.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace ops {
ShapeVector CheckMatMulShapes(const ShapeVector &shape1, const ShapeVector &shape2) {
  ShapeVector shape_out;

  if (shape1.size() == 0 || shape2.size() == 0) {
    MS_EXCEPTION(ValueError) << "For 'MatMulExt' op, inputs must be all tensors and rank >= 1";
  }

  if (shape2.size() >= kDim2 && shape1.back() != shape2[shape2.size() - kDim2]) {
    MS_EXCEPTION(RuntimeError) << "For 'MatMulExt' op, shape1[-1] must be equal to shape2[-2], but got "
                               << shape1.back() << " and " << shape2[shape2.size() - kDim2] << ".";
  }

  int len_diff = std::abs(static_cast<int>(shape1.size()) - static_cast<int>(shape2.size()));
  ShapeVector shape1_padded, shape2_padded;

  if (shape1.size() < shape2.size()) {
    shape1_padded = ShapeVector(len_diff, 1);
    shape1_padded.insert(shape1_padded.end(), shape1.begin(), shape1.end());
    shape2_padded = shape2;
  } else {
    shape2_padded = ShapeVector(len_diff, 1);
    shape2_padded.insert(shape2_padded.end(), shape2.begin(), shape2.end());
    shape1_padded = shape1;
  }

  //  MS_LOG(ERROR) << "shape1_padded: " << shape1_padded << ", shape2_padded: " << shape2_padded;

  int max_len = std::max(static_cast<int>(shape1_padded.size()) - 2, static_cast<int>(shape2_padded.size()) - 2);
  //    MS_LOG(ERROR) << "max_len: " << max_len;
  for (int i = 0; i < max_len; ++i) {
    size_t dim1 = i < static_cast<int>(shape1_padded.size()) - 2 ? shape1_padded[i] : 1;
    size_t dim2 = i < static_cast<int>(shape2_padded.size()) - 2 ? shape2_padded[i] : 1;
    if (dim1 != 1 && dim2 != 1 && dim1 != dim2) {
      MS_EXCEPTION(RuntimeError) << "For 'MatMulExt' op,  shape1 and shape2 must be broadcastable, but got "
                                 << shape1_padded << " and " << shape2_padded;
    }
    shape_out.push_back(std::max(dim1, dim2));
  }

  //  MS_LOG(ERROR) << "shape_out: " << shape_out;

  return shape_out;
}

ShapeVector InferShapeRem(const ShapeVector &shape_backbone, const ShapeVector &shape1, const ShapeVector &shape2,
                          bool transpose_b) {
  int ndim1 = shape1.size();
  int ndim2 = shape2.size();
  ShapeVector shape_rem(shape_backbone);
  if (ndim1 >= SizeToInt(kDim2)) {
    shape_rem.push_back(shape1[ndim1 - SizeToInt(kDim2)]);
  }
  if (transpose_b) {
    if (ndim2 >= SizeToInt(kDim2)) {
      shape_rem.push_back(shape2[ndim2 - SizeToInt(kDim2)]);
    }
  } else {
    if (ndim2 >= 1) {
      shape_rem.push_back(shape2.back());
    }
  }
  return shape_rem;
}

void MatMulMakeShape(ShapeVector *output, const ShapeVector xshp, const ShapeVector yshp) {
  size_t offset = kDim2;
  if (xshp.empty() || yshp.empty()) {
    return;
  }
  auto x_rank = xshp.size();
  auto y_rank = yshp.size();
  if (x_rank == 1 && y_rank == 1) {
    return;
  }

  auto max_rank = x_rank > y_rank ? x_rank : y_rank;

  if (x_rank == 1 || y_rank == 1) {
    for (size_t i = 0; i < max_rank - 1; i++) {
      output->push_back(abstract::Shape::kShapeDimAny);
    }
    return;
  }

  ShapeVector long_input = xshp.size() > yshp.size() ? xshp : yshp;
  ShapeVector short_input = xshp.size() > yshp.size() ? yshp : xshp;
  size_t size_diff = long_input.size() - short_input.size();
  for (size_t i = 0; i < long_input.size() - offset; i++) {
    if (long_input[i] < 0) {
      output->push_back(abstract::Shape::kShapeDimAny);
    } else if (i >= size_diff) {
      output->push_back(long_input[i] > short_input[i - size_diff] ? long_input[i] : short_input[i - size_diff]);
    } else {
      output->push_back(long_input[i]);
    }
  }
  size_t x_offset = xshp.size() - offset;
  size_t y_offset = yshp.size() - offset;
  output->push_back(xshp[x_offset]);
  output->push_back(yshp[y_offset + 1]);
}
BaseShapePtr MatMulExtFuncImpl::InferShape(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  auto constexpr kMatMulExtInputNum = 2;
  (void)CheckAndConvertUtils::CheckInteger("input num", SizeToLong(input_args.size()), kEqual, kMatMulExtInputNum,
                                           primitive->name());
  auto prim_name = primitive->name();
  auto x_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->GetShape());
  auto y_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->GetShape());
  if (x_shape_map.empty()) {
    MS_LOG(EXCEPTION) << "For '" << prim_name
                      << "', input 'x' must be a Tensor type, but got:" << input_args[0]->ToString();
  }
  if (y_shape_map.empty()) {
    MS_LOG(EXCEPTION) << "For '" << prim_name
                      << "', input 'y' must be a Tensor type, but got:" << input_args[1]->ToString();
  }
  auto shape1_orig = x_shape_map[kShape];
  auto shape2_orig = y_shape_map[kShape];
  if (IsDynamicRank(shape1_orig) || IsDynamicRank(shape2_orig)) {
    return std::make_shared<abstract::Shape>(ShapeVector({abstract::Shape::kShapeRankAny}));
  }

  bool dynamic_shape = IsDynamic(shape1_orig) || IsDynamic(shape2_orig);

  if (!dynamic_shape) {
    bool transpose_b = shape2_orig.size() == 1;
    ShapeVector shape_backbone = CheckMatMulShapes(shape1_orig, shape2_orig);
    ShapeVector ret_shape = InferShapeRem(shape_backbone, shape1_orig, shape2_orig, transpose_b);
    return std::make_shared<abstract::Shape>(ret_shape);
  }

  ShapeVector ret_shape;
  MatMulMakeShape(&ret_shape, shape1_orig, shape2_orig);
  return std::make_shared<abstract::Shape>(ret_shape);
}

TypePtr MatMulExtFuncImpl::InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(prim);
  const std::set<TypePtr> valid_types = {kInt8,   kInt16,   kInt32,   kInt64,   kUInt8,     kUInt16,     kUInt32,
                                         kUInt64, kFloat16, kFloat32, kFloat64, kComplex64, kComplex128, kBFloat16};
  std::map<std::string, TypePtr> types;
  (void)types.emplace("x", input_args[0]->GetType());
  (void)types.emplace("w", input_args[1]->GetType());
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, prim->name());
  TypePtr x_type = input_args[0]->GetType();
  return x_type;
}

}  // namespace ops
}  // namespace mindspore
