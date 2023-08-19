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
#include "ir/primitive.h"
#include "abstract/abstract_value.h"
#include "ops/ops_func_impl/tile.h"
#include "ops/test_ops_cmp_utils.h"

namespace mindspore {
namespace ops {
struct TileParams {
  ShapeVector x_shape;
  TypePtr x_type;
  std::vector<int64_t> multiples_value;  // -2: dynamic sequence; -1: value unknown; others: normal input.
  ShapeVector out_shape;
};

class TestTile : public TestOps, public testing::WithParamInterface<TileParams> {};

TEST_P(TestTile, dyn_shape) {
  const auto &param = GetParam();

  auto x = std::make_shared<abstract::AbstractTensor>(param.x_type, param.x_shape);
  ASSERT_NE(x, nullptr);

  AbstractBasePtrList multiple_elements;
  bool is_dynamic_len = false;
  for (auto v : param.multiples_value) {
    if (v < 0) {
      multiple_elements.push_back(std::make_shared<abstract::AbstractScalar>(kValueAny, kInt64));
    }

    if (v == -2) {
      is_dynamic_len = true;
      break;
    }

    if (v != -1) {
      multiple_elements.push_back(std::make_shared<abstract::AbstractScalar>(v));
    }
  }
  auto multiples = std::make_shared<abstract::AbstractTuple>(multiple_elements);
  ASSERT_NE(multiples, nullptr);
  if (is_dynamic_len) {
    multiples->CheckAndConvertToDynamicLenSequence();
  }

  auto expect_shape = std::make_shared<abstract::Shape>(param.out_shape);
  auto expect_type = std::make_shared<TensorType>(param.x_type);
  DoFuncImplInferAndCompare<TileFuncImpl>("Tile", abstract::AbstractBasePtrList{x, multiples}, expect_shape,
                                          expect_type);
}

INSTANTIATE_TEST_CASE_P(TestTile, TestTile,
                        testing::Values(TileParams{{3, 4}, kFloat32, {2, 2, 2}, {2, 6, 8}},
                                        TileParams{{2, 3, 4}, kFloat32, {2, 2, 2}, {4, 6, 8}},
                                        TileParams{{-1, 3, -1}, kFloat32, {2, 2, 2}, {-1, 6, -1}},
                                        TileParams{{2, 3, 4}, kFloat32, {-1, -1, -1}, {-1, -1, -1}},
                                        TileParams{{2, 3, 4}, kFloat32, {-1, 2, -1}, {-1, 6, -1}},
                                        TileParams{{2, 3, 4}, kFloat32, {-1, -1, -1, -1}, {-1, -1, -1, -1}},
                                        TileParams{{-1, 3, -1}, kFloat32, {2, -1, -1}, {-1, -1, -1}},
                                        TileParams{{2, 3, 4}, kFloat32, {-2}, {-2}},
                                        TileParams{{-1, -1, -1}, kFloat32, {-2}, {-2}},
                                        TileParams{{-2}, kFloat32, {2, 3, 4}, {-1, -1, -1}},
                                        TileParams{{-2}, kFloat32, {-1, 2, -1}, {-1, -1, -1}}));
}  // namespace ops
}  // namespace mindspore
