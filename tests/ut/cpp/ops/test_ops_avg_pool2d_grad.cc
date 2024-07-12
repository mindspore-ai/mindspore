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

#include <vector>
#include <memory>
#include "common/common_test.h"
#include "ops/ops_func_impl/avg_pool2d_grad.h"
#include "ops/auto_generate/gen_ops_name.h"
#include "ir/dtype/type.h"
#include "abstract/dshape.h"
#include "utils/tensor_construct_utils.h"
#include "ir/primitive.h"
#include "abstract/abstract_value.h"
#include "ops/test_ops.h"
#include "ops/test_value_utils.h"
#include "ops/test_ops_cmp_utils.h"

namespace mindspore::ops {
struct AvgPool2DGradParams {
  ShapeVector grad_shape;
  ShapeVector image_shape;
  ValuePtr kernel_size;  // tuple[int]
  ValuePtr stride;       // tuple[int]
  ValuePtr padding;      // tuple[int]
  ValuePtr ceil_mode;    // bool
  ShapeVector out_shape;
  bool is_static = false;
};

class TestAvgPool2DGrad : public TestOps, public testing::WithParamInterface<AvgPool2DGradParams> {};

TEST_P(TestAvgPool2DGrad, dyn_shape) {
  const auto &param = GetParam();

  auto grad_abs = std::make_shared<abstract::AbstractTensor>(kFloat32, param.grad_shape);
  ASSERT_NE(grad_abs, nullptr);
  auto image_abs = std::make_shared<abstract::AbstractTensor>(kFloat32, param.image_shape);
  ASSERT_NE(image_abs, nullptr);
  auto kernel_size_abs = param.kernel_size->ToAbstract();
  ASSERT_NE(kernel_size_abs, nullptr);
  auto stride_abs = param.stride->ToAbstract();
  ASSERT_NE(stride_abs, nullptr);
  auto padding_abs = param.padding->ToAbstract();
  ASSERT_NE(padding_abs, nullptr);
  auto ceil_mode_abs = param.ceil_mode->ToAbstract();
  ASSERT_NE(ceil_mode_abs, nullptr);
  auto count_include_pad = CreateScalar<bool>(true);
  auto count_include_pad_abs = count_include_pad->ToAbstract();
  auto divisor_override = CreateScalar<int64_t>(int64_t(1));
  auto divisor_override_abs = divisor_override->ToAbstract();

  auto prim = std::make_shared<Primitive>(kNameAvgPool2DGrad);
  auto infer_impl = std::make_shared<AvgPool2DGradFuncImpl>();
  ASSERT_NE(infer_impl, nullptr);
  // for abstract based infer
  std::vector<AbstractBasePtr> input_args{grad_abs,    image_abs,     kernel_size_abs,       stride_abs,
                                          padding_abs, ceil_mode_abs, count_include_pad_abs, divisor_override_abs};
  auto inferred_shape = infer_impl->InferShape(prim, input_args);
  auto expect_shape = std::make_shared<abstract::Shape>(param.out_shape);
  ShapeCompare(inferred_shape, expect_shape);
  // simple infer
  if (param.is_static) {
    auto grad = std::make_shared<tensor::BaseTensor>(kNumberTypeFloat32, param.grad_shape);
    auto image = std::make_shared<tensor::BaseTensor>(kNumberTypeFloat32, param.image_shape);
    std::vector<ValuePtr> input_valus{grad,          image,           param.kernel_size, param.stride,
                                      param.padding, param.ceil_mode, count_include_pad, divisor_override};
    auto expect_shape = ShapeArray{param.out_shape};
    auto infer_shape = infer_impl->InferShape(prim, input_valus);
    ShapeCompare(infer_shape, expect_shape);
  }
}

INSTANTIATE_TEST_CASE_P(
  TestAvgPool2DGrad, TestAvgPool2DGrad,
  testing::Values(AvgPool2DGradParams{ShapeVector{-2}, ShapeVector{-2}, CreatePyIntTuple({2}), CreatePyIntTuple({1}),
                                      CreatePyIntTuple({0, 0}), CreateScalar<bool>(true), ShapeVector{-2}},

                  AvgPool2DGradParams{ShapeVector{-1, -1, 2}, ShapeVector{-1, -1, 10}, CreatePyIntTuple({4}),
                                      CreatePyIntTuple({4, 4}), CreatePyIntTuple({0}), CreateScalar<bool>(false),
                                      ShapeVector{-1, -1, 10}},

                  AvgPool2DGradParams{ShapeVector{3, -1, -1}, ShapeVector{3, 10, 10}, CreatePyIntTuple({4, 4}),
                                      CreatePyIntTuple({4, 4}), CreatePyIntTuple({0, 0}), CreateScalar(kValueAny),
                                      ShapeVector{3, 10, 10}},

                  AvgPool2DGradParams{ShapeVector{4, 3, 6, 6}, ShapeVector{4, 3, 20, 20}, CreatePyIntTuple({3}),
                                      CreatePyIntTuple({3}), CreatePyIntTuple({0}), CreateScalar<bool>(false),
                                      ShapeVector{4, 3, 20, 20}, true}));
}  // namespace mindspore::ops
