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
#include "ir/dtype/type.h"
#include "abstract/dshape.h"
#include "utils/tensor_construct_utils.h"
#include "ir/primitive.h"
#include "abstract/abstract_value.h"
#include "include/backend/optimizer/helper.h"
#include "ops/test_ops.h"
#include "ops/ops_func_impl/remainder_tensor_tensor.h"
#include "ops/test_value_utils.h"

namespace mindspore::ops {

struct RemainderTensorTensorShape {
  ShapeVector input_shape;
  ShapeVector other_shape;
  ShapeVector out_shape;
};

struct RemainderTensorTensorType {
  TypePtr input_type;
  TypePtr other_type;
  TypePtr out_type;
};

class TestRemainderTensorTensor : public TestOps, public testing::WithParamInterface<std::tuple<RemainderTensorTensorShape, RemainderTensorTensorType>> {};

TEST_P(TestRemainderTensorTensor, RemainderTensorTensor_dyn_shape) {
  const auto &shape_param = std::get<0>(GetParam());
  const auto &dtype_param = std::get<1>(GetParam());

  RemainderTensorTensorFuncImpl RemainderTensorTensor_func_impl;
  auto prim = std::make_shared<Primitive>("RemainderTensorTensor");
  auto input = std::make_shared<abstract::AbstractTensor>(dtype_param.input_type, shape_param.input_shape);
  auto other = std::make_shared<abstract::AbstractTensor>(dtype_param.other_type, shape_param.other_shape);
  auto expect_shape = std::make_shared<abstract::TensorShape>(shape_param.out_shape);
  auto expect_dtype = std::make_shared<TensorType>(dtype_param.out_type);

  auto out_shape = RemainderTensorTensor_func_impl.InferShape(prim, {input, other});
  ASSERT_TRUE(*out_shape == *expect_shape);
  auto out_dtype = RemainderTensorTensor_func_impl.InferType(prim, {input, other});
  ASSERT_TRUE(*out_dtype == *expect_dtype);
}

auto RemainderTensorTensorOpShapeTestCases = testing::ValuesIn({
    /* other is number */
    RemainderTensorTensorShape{{10}, {}, {10}},
    RemainderTensorTensorShape{{10, 1, 2}, {}, {10, 1, 2}},
    RemainderTensorTensorShape{{10, 4, 2}, {}, {10, 4, 2}},
    RemainderTensorTensorShape{{10, 1, -1}, {}, {10, 1, -1}},
    RemainderTensorTensorShape{{-2}, {}, {-2}},
    /* input is number */
    RemainderTensorTensorShape{{}, {10}, {10}},
    RemainderTensorTensorShape{{}, {10, 1, 2}, {10, 1, 2}},
    RemainderTensorTensorShape{{}, {10, 4, 2}, {10, 4, 2}},
    RemainderTensorTensorShape{{}, {10, 1, -1}, {10, 1, -1}},
    RemainderTensorTensorShape{{}, {-2}, {-2}},
    /* input and other both tensor */
    RemainderTensorTensorShape{{4, 5}, {2, 3, 4, 5}, {2, 3, 4, 5}},
    RemainderTensorTensorShape{{2, 1, 4, 5, 6, 9}, {9}, {2, 1, 4, 5, 6, 9}},
    RemainderTensorTensorShape{{2, 3, 4, -1}, {2, 3, 4, 5}, {2, 3, 4, 5}},
    RemainderTensorTensorShape{{2, 3, 4, -1}, {-1, -1, 4, 5}, {2, 3, 4, 5}},
    RemainderTensorTensorShape{{2, 1, 4, -1}, {-1, -1, 4, 5}, {2, -1, 4, 5}},
    RemainderTensorTensorShape{{2, 1, 4, 5, 6, 9}, {-2}, {-2}},
    RemainderTensorTensorShape{{2, 1, 4, 5, -1, 9}, {-2}, {-2}},
    RemainderTensorTensorShape{{-2}, {2, 1, 4, 5, 6, 9}, {-2}},
    RemainderTensorTensorShape{{-2}, {2, 1, 4, 5, -1, 9}, {-2}},
    RemainderTensorTensorShape{{-2}, {-2}, {-2}}
});

auto RemainderTensorTensorOpTypeTestCases = testing::ValuesIn({
  RemainderTensorTensorType{kFloat64, kFloat64, kFloat64},
  RemainderTensorTensorType{kInt32, kInt32, kInt32},
  RemainderTensorTensorType{kBFloat16, kBFloat16, kBFloat16},
});

INSTANTIATE_TEST_CASE_P(TestRemainderTensorTensor, TestRemainderTensorTensor,
                        testing::Combine(RemainderTensorTensorOpShapeTestCases, RemainderTensorTensorOpTypeTestCases));
}  // namespace mindspore::ops
