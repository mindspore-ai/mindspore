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

#include "ops/test_ops.h"
#include "common/common_test.h"
#include "ir/dtype/type.h"
#include "abstract/dshape.h"
#include "utils/tensor_construct_utils.h"
#include "ir/primitive.h"
#include "abstract/abstract_value.h"
#include "ops/op_name.h"
#include "ops/ops_func_impl/max_pool_with_indices.h"
#include "include/backend/optimizer/helper.h"
#include "ops/test_value_utils.h"

namespace mindspore {
namespace ops {
#define I64(x) (static_cast<int64_t>((x)))
struct MaxPoolWithIndicesParams {
  ShapeVector x_shape;
  TypePtr x_dtype;
  ValuePtr kernel_size;
  ValuePtr strides;
  ValuePtr pads;
  ValuePtr dilation;
  ValuePtr ceil_mode;
  ValuePtr argmax_type;
  ShapeVector out1_shape;
  TypePtr out1_type;
  ShapeVector out2_shape;
  TypePtr out2_type;
};

class TestMaxPoolWithIndices : public TestOps, public testing::WithParamInterface<MaxPoolWithIndicesParams> {};

TEST_P(TestMaxPoolWithIndices, dyn_shape) {
  const auto &param = GetParam();
  auto max_pool_with_indices_func_impl = std::make_shared<MaxPoolWithIndicesFuncImpl>();
  auto prim = std::make_shared<Primitive>("MaxPoolWithIndices");

  auto x = std::make_shared<abstract::AbstractTensor>(param.x_dtype, param.x_shape);
  ASSERT_NE(x, nullptr);
  auto kernel_size = param.kernel_size->ToAbstract();
  ASSERT_NE(kernel_size, nullptr);
  auto strides = param.strides->ToAbstract();
  ASSERT_NE(strides, nullptr);
  auto pads = param.pads->ToAbstract();
  ASSERT_NE(pads, nullptr);
  auto dilation = param.dilation->ToAbstract();
  ASSERT_NE(dilation, nullptr);
  auto ceil_mode = param.ceil_mode->ToAbstract();
  ASSERT_NE(ceil_mode, nullptr);
  auto argmax_type = param.argmax_type->ToAbstract();
  ASSERT_NE(argmax_type, nullptr);
  auto expect1_shape = std::make_shared<abstract::Shape>(param.out1_shape);
  auto expect1_type = std::make_shared<TensorType>(param.out1_type);
  auto expect2_shape = std::make_shared<abstract::Shape>(param.out2_shape);
  auto expect2_type = std::make_shared<TensorType>(param.out2_type);
  std::vector<abstract::BaseShapePtr> shape_list = {expect1_shape, expect2_shape};
  auto expect_shape = std::make_shared<abstract::TupleShape>(shape_list);
  std::vector<TypePtr> type_list = {expect1_type, expect2_type};
  auto expect_type = std::make_shared<Tuple>(type_list);
  auto infer_shape = max_pool_with_indices_func_impl->InferShape(
    prim, {x, kernel_size, strides, pads, dilation, ceil_mode, argmax_type});
  ASSERT_NE(infer_shape, nullptr);
  ASSERT_TRUE(*infer_shape == *expect_shape);
  auto infer_type =
    max_pool_with_indices_func_impl->InferType(prim, {x, kernel_size, strides, pads, dilation, ceil_mode, argmax_type});
  ASSERT_NE(infer_type, nullptr);
  ASSERT_TRUE(*infer_type == *expect_type);
}

INSTANTIATE_TEST_CASE_P(TestMaxPoolWithIndicesGroup, TestMaxPoolWithIndices,
                        testing::Values(MaxPoolWithIndicesParams{{-2},
                                                                 kFloat16,
                                                                 CreateTuple({I64(4), I64(4)}),
                                                                 CreateTuple({I64(2), I64(2)}),
                                                                 CreateTuple({I64(1), I64(1)}),
                                                                 CreateTuple({I64(2), I64(2)}),
                                                                 CreateScalar(false),
                                                                 CreatePyInt(kNumberTypeInt64),
                                                                 {-1, -1, -1, -1},
                                                                 kFloat16,
                                                                 {-1, -1, -1, -1},
                                                                 kInt64},
                                        MaxPoolWithIndicesParams{{-1, -1, -1, -1},
                                                                 kFloat16,
                                                                 CreateTuple({I64(4), I64(4)}),
                                                                 CreateTuple({I64(2), I64(2)}),
                                                                 CreateTuple({I64(1), I64(1)}),
                                                                 CreateTuple({I64(2), I64(2)}),
                                                                 CreateScalar(false),
                                                                 CreatePyInt(kNumberTypeInt64),
                                                                 {-1, -1, -1, -1},
                                                                 kFloat16,
                                                                 {-1, -1, -1, -1},
                                                                 kInt64},
                                        MaxPoolWithIndicesParams{{1, 1, 8, 8},
                                                                 kFloat16,
                                                                 CreateTuple({I64(4), I64(4)}),
                                                                 CreateTuple({I64(2), I64(2)}),
                                                                 CreateTuple({I64(1), I64(1)}),
                                                                 CreateTuple({I64(2), I64(2)}),
                                                                 CreateScalar(false),
                                                                 CreatePyInt(kNumberTypeInt64),
                                                                 {1, 1, 2, 2},
                                                                 kFloat16,
                                                                 {1, 1, 2, 2},
                                                                 kInt64},
                                        MaxPoolWithIndicesParams{{1, 1, 8, 8},
                                                                 kFloat16,
                                                                 CreateTuple({I64(4), I64(4)}),
                                                                 CreateTuple({I64(2), I64(2)}),
                                                                 CreateTuple({I64(1), I64(1)}),
                                                                 CreateTuple({I64(2), I64(2)}),
                                                                 CreateScalar(true),
                                                                 CreatePyInt(kNumberTypeInt64),
                                                                 {1, 1, 3, 3},
                                                                 kFloat16,
                                                                 {1, 1, 3, 3},
                                                                 kInt64}));

}  // namespace ops
}  // namespace mindspore
