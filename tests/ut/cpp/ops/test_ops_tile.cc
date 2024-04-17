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
#include "abstract/ops/primitive_infer_map.h"
#include "ir/tensor.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/base/type_id.h"
#include "ops/auto_generate/gen_ops_primitive.h"
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

TEST_P(TestTile, dyn_shape_infer_shape) {
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
                                        TileParams{{-2}, kFloat32, {2, 3, 4}, {-2}},
                                        TileParams{{-2}, kFloat32, {-1, 2, -1}, {-2}},
                                        TileParams{{2, 3, 4}, kFloat32, {2, 2}, {2, 6, 8}},
                                        TileParams{{-1, 3, -1}, kFloat32, {2, -1}, {-1, 6, -1}},
                                        TileParams{{2, 3, 4}, kFloat32, {-1, -1}, {2, -1, -1}}));

tensor::TensorPtr CreateFloat32Tensor(const ShapeVector &shape, std::vector<float> value) {
  void *data_ptr = &value[0];
  auto tensor = std::make_shared<tensor::Tensor>(kNumberTypeFloat32, shape, data_ptr, kNumberTypeFloat32);
  return tensor;
}
tensor::TensorPtr CreateInt32Tensor(const ShapeVector &shape, std::vector<int32_t> value) {
  void *data_ptr = &value[0];
  auto tensor = std::make_shared<tensor::Tensor>(kNumberTypeInt32, shape, data_ptr, kNumberTypeInt32);
  return tensor;
}

struct TileInferValueParams {
  tensor::TensorPtr x;
  std::vector<int64_t> multiples_value;  // -2: dynamic sequence; -1: value unknown; others: normal input.
  tensor::TensorPtr out;
};

class TestTileInferValue : public TestOps, public testing::WithParamInterface<TileInferValueParams> {};

TEST_P(TestTileInferValue, dyn_shape_infer_value) {
  const auto &param = GetParam();
  ASSERT_NE(param.x, nullptr);
  auto x = param.x->ToAbstract();
  ASSERT_NE(x, nullptr);

  AbstractBasePtrList multiple_elements;
  for (auto v : param.multiples_value) {
    ASSERT_GT(v, 0);
    multiple_elements.push_back(std::make_shared<abstract::AbstractScalar>(v));
  }
  auto multiples = std::make_shared<abstract::AbstractTuple>(multiple_elements);
  ASSERT_NE(multiples, nullptr);

  auto input_args = abstract::AbstractBasePtrList{x, multiples};
  auto value_opt = abstract::InferValueByFuncImpl(prim::kPrimTile, input_args);
  if (!value_opt.has_value()) {
    MS_LOG(ERROR) << "Tile have no infer value implement!";
    ASSERT_TRUE(false);
  }
  auto infer_out = value_opt.value();
  if (infer_out == nullptr) {
    MS_LOG(ERROR) << "Tile can not infer value with inputs: " << input_args;
    ASSERT_TRUE(false);
  }
  auto infer_tensor = infer_out->cast<tensor::TensorPtr>();
  ASSERT_NE(infer_tensor, nullptr);
  ASSERT_TRUE(infer_tensor->ValueEqual(*param.out));
}

INSTANTIATE_TEST_CASE_P(
  TestTileInferValue, TestTileInferValue,
  testing::Values(
    TileInferValueParams{
      CreateFloat32Tensor(ShapeVector{2, 2}, std::vector<float>{2, 2, 3, 3}),
      {2, 2},
      CreateFloat32Tensor(ShapeVector{4, 4}, std::vector<float>{2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 2, 2, 3, 3, 3, 3})},
    TileInferValueParams{CreateInt32Tensor(ShapeVector{2}, std::vector<int32_t>{3, 4}),
                         {2, 2},
                         CreateInt32Tensor(ShapeVector{2, 4}, std::vector<int32_t>{3, 4, 3, 4, 3, 4, 3, 4})},
    TileInferValueParams{CreateFloat32Tensor(ShapeVector{2, 2}, std::vector<float>{2, 2, 3, 3}),
                         {2},
                         CreateFloat32Tensor(ShapeVector{2, 4}, std::vector<float>{2, 2, 2, 2, 3, 3, 3, 3})}));
}  // namespace ops
}  // namespace mindspore
