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

#include <vector>
#include <memory>

#include "ops/test_ops.h"
#include "common/common_test.h"
#include "ir/dtype/type.h"
#include "abstract/dshape.h"
#include "utils/tensor_construct_utils.h"
#include "ir/primitive.h"
#include "abstract/abstract_value.h"
#include "ops/op_name.h"
#include "ops/ops_func_impl/tensor_copy_slices.h"
#include "include/backend/optimizer/helper.h"

namespace mindspore {
namespace ops {
struct TensorCopySlicesShape {
  ShapeVector x_shape;
  ShapeVector out_shape;
};

struct TensorCopySlicesDType {
  TypePtr x_dtype;
  TypePtr out_dtype;
};

class TestTensorCopySlices
    : public TestOps,
      public testing::WithParamInterface<std::tuple<TensorCopySlicesShape, TensorCopySlicesDType>> {};

TEST_P(TestTensorCopySlices, dyn_shape) {
  const auto &shape_param = std::get<0>(GetParam());
  const auto &dtype_param = std::get<1>(GetParam());
  auto tensor_copy_slices_func_impl = std::make_shared<TensorCopySlicesFuncImpl>();
  auto prim = std::make_shared<Primitive>("TensorCopySlices");

  auto x = std::make_shared<abstract::AbstractTensor>(dtype_param.x_dtype, shape_param.x_shape);
  ASSERT_NE(x, nullptr);

  auto expect_shape = std::make_shared<abstract::Shape>(shape_param.out_shape);
  auto expect_dtype = std::make_shared<TensorType>(dtype_param.out_dtype);

  auto infer_shape = tensor_copy_slices_func_impl->InferShape(prim, {x});
  ASSERT_NE(infer_shape, nullptr);
  ASSERT_TRUE(*infer_shape == *expect_shape);
  auto infer_dtype = tensor_copy_slices_func_impl->InferType(prim, {x});
  ASSERT_NE(infer_dtype, nullptr);
  ASSERT_TRUE(*infer_dtype == *expect_dtype);
}

auto TensorCopySlicesDynTestCase = testing::ValuesIn({
  TensorCopySlicesShape{{1}, {1}},
  TensorCopySlicesShape{{1, 3}, {1, 3}},
  TensorCopySlicesShape{{1, -1}, {1, -1}},
  TensorCopySlicesShape{{-1, -1}, {-1, -1}},
  TensorCopySlicesShape{{-2}, {-2}},
});

auto TensorCopySlicesDTypeTestCase = testing::ValuesIn({
  TensorCopySlicesDType{kFloat16, kFloat16},
  TensorCopySlicesDType{kFloat32, kFloat32},
  TensorCopySlicesDType{kFloat64, kFloat64},
});

INSTANTIATE_TEST_CASE_P(TestTensorCopySlicesGroup, TestTensorCopySlices,
                        testing::Combine(TensorCopySlicesDynTestCase, TensorCopySlicesDTypeTestCase));
}  // namespace ops
}  // namespace mindspore
