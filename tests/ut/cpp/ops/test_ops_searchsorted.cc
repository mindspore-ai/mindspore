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
#include "ops/ops_func_impl/searchsorted.h"
#include "ops/test_value_utils.h"

namespace mindspore {
namespace ops {

struct SearchSortedParam {
  std::vector<int64_t> sequence_shape;
  std::vector<int64_t> values_shape;
  std::vector<int64_t> sorter_shape;
  TypePtr in_dtype;
  ValuePtr out_dtype;
  bool right;
};

class TestSearchSorted : public TestOps, public testing::WithParamInterface<SearchSortedParam> {};

TEST_P(TestSearchSorted, SearchSorted_dyn_shape) {
  const auto &param = GetParam();

  SearchSortedFuncImpl func_impl;
  auto prim = std::make_shared<Primitive>("SearchSorted");
  auto sequence = std::make_shared<abstract::AbstractTensor>(param.in_dtype, param.sequence_shape);
  auto values = std::make_shared<abstract::AbstractTensor>(param.in_dtype, param.values_shape);
  auto sorter = std::make_shared<abstract::AbstractTensor>(kInt64, param.sorter_shape);
  auto right = CreateScalar(param.right)->ToAbstract();
  auto dtype = param.out_dtype->ToAbstract();

  auto expect_shape = std::make_shared<abstract::TensorShape>(param.values_shape);
  auto expect_dtype =
    std::make_shared<TensorType>(TypeIdToType(static_cast<TypeId>(param.out_dtype->cast<Int64ImmPtr>()->value())));

  auto out_shape = func_impl.InferShape(prim, {sequence, values, sorter, dtype, right});
  ASSERT_TRUE(*out_shape == *expect_shape);
  auto out_dtype = func_impl.InferType(prim, {sequence, values, sorter, dtype, right});
  ASSERT_TRUE(*out_dtype == *expect_dtype);
}

auto SearchSortedParamOpTestCases = testing::Values(
  SearchSortedParam{{10}, {10}, {}, kFloat32, CreatePyInt(kNumberTypeInt32), true},
  SearchSortedParam{{10, 1, 2}, {10, 1, 10}, {}, kFloat32, CreatePyInt(kNumberTypeInt32), true},
  SearchSortedParam{{10, 4, 2}, {10, 4, 10}, {}, kFloat32, CreatePyInt(kNumberTypeInt32), false},
  SearchSortedParam{{10, 1, -1}, {10, 1, 10}, {}, kFloat32, CreatePyInt(kNumberTypeInt32), true},

  SearchSortedParam{{10}, {10}, {}, kFloat32, CreatePyInt(kNumberTypeInt32), true},
  SearchSortedParam{{10}, {10, 1, 2}, {}, kFloat32, CreatePyInt(kNumberTypeInt32), true},
  SearchSortedParam{{10}, {10, 4, 2}, {}, kFloat32, CreatePyInt(kNumberTypeInt32), false},
  SearchSortedParam{{10}, {10, 1, -1}, {}, kFloat32, CreatePyInt(kNumberTypeInt32), true},
  SearchSortedParam{{10}, {-2}, {}, kFloat32, CreatePyInt(kNumberTypeInt32), true},

  SearchSortedParam{{4, 5}, {4, 15}, {4, 5}, kFloat32, CreatePyInt(kNumberTypeInt32), true},
  SearchSortedParam{
    {2, 1, 4, 5, 6, 9}, {2, 1, 4, 5, 6, 9}, {2, 1, 4, 5, 6, 9}, kFloat32, CreatePyInt(kNumberTypeInt32), false},
  SearchSortedParam{{2, 3, 4, -1}, {2, 3, 4, 5}, {2, 3, 4, -1}, kFloat32, CreatePyInt(kNumberTypeInt32), true},
  SearchSortedParam{{2, 3, 4, -1}, {-1, -1, 4, 5}, {2, 3, 4, -1}, kFloat32, CreatePyInt(kNumberTypeInt32), true},
  SearchSortedParam{{2, 1, 4, -1}, {-1, -1, 4, 5}, {}, kFloat32, CreatePyInt(kNumberTypeInt32), false},
  SearchSortedParam{{2, 1, 4, 5, 6, 9}, {-2}, {2, 1, 4, 5, 6, 9}, kFloat32, CreatePyInt(kNumberTypeInt32), true},
  SearchSortedParam{{2, 1, 4, 5, -1, 9}, {-2}, {}, kFloat32, CreatePyInt(kNumberTypeInt32), true},
  SearchSortedParam{{-2}, {-2}, {}, kFloat32, CreatePyInt(kNumberTypeInt32), false});

INSTANTIATE_TEST_CASE_P(TestSearchSorted, TestSearchSorted, SearchSortedParamOpTestCases);
}  // namespace ops
}  // namespace mindspore
