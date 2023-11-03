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

#include <memory>
#include <vector>
#include "string"
#include "abstract/dshape.h"
#include "abstract/abstract_value.h"
#include "common/common_test.h"
#include "ir/dtype/type.h"
#include "ir/primitive.h"
#include "include/backend/optimizer/helper.h"
#include "ops/ops_func_impl/nllloss.h"
#include "ops/test_ops.h"
#include "ops/test_ops_cmp_utils.h"
#include "ops/test_value_utils.h"
#include "utils/ms_context.h"
#include "utils/tensor_construct_utils.h"

namespace mindspore {
namespace ops {
struct NLLLossParams {
  ShapeVector logits_shape;
  TypePtr logits_type;
  ShapeVector labels_shape;
  TypePtr labels_type;
  ShapeVector weight_shape;
  TypePtr weight_type;
  int reduction;
  bool is_success;
  std::vector<ShapeVector> out_shape_array;
  std::vector<TypePtr> out_type_array;
};

class TestNLLLoss : public TestOps, public testing::WithParamInterface<NLLLossParams> {};

const string kNameNLLLoss_ = "NLLLoss";

TEST_P(TestNLLLoss, dyn_shape) {
  const auto &param = GetParam();
  auto prim = std::make_shared<Primitive>(kNameNLLLoss_);
  ASSERT_NE(prim, nullptr);
  auto logits = std::make_shared<abstract::AbstractTensor>(param.logits_type, param.logits_shape);
  auto labels = std::make_shared<abstract::AbstractTensor>(param.labels_type, param.labels_shape);
  auto weight = std::make_shared<abstract::AbstractTensor>(param.weight_type, param.weight_shape);
  ASSERT_NE(logits, nullptr);
  ASSERT_NE(labels, nullptr);
  ASSERT_NE(weight, nullptr);

  ValuePtr reductionPtr = nullptr;
  if (param.reduction == -1) {
    reductionPtr = CreateScalar(kValueAny);
  } else {
    reductionPtr = CreateScalar<int64_t>(param.reduction);
  }

  auto infer_impl = std::make_shared<NLLLossFuncImpl>();
  ASSERT_NE(infer_impl, nullptr);

  if (param.is_success) {
    ASSERT_TRUE(!param.out_shape_array.empty());
    ASSERT_TRUE(!param.out_type_array.empty());
    ASSERT_TRUE(param.out_shape_array.size() == param.out_type_array.size());

    std::vector<abstract::BaseShapePtr> shape_list;
    std::vector<TypePtr> type_list;
    for (size_t idx = 0; idx < param.out_shape_array.size(); ++idx) {
      auto shape = std::make_shared<abstract::TensorShape>(param.out_shape_array[idx]);
      auto type = std::make_shared<TensorType>(param.out_type_array[idx]);
      shape_list.push_back(std::move(shape));
      type_list.push_back(std::move(type));
    }
    auto expect_shape = std::make_shared<abstract::TupleShape>(shape_list);
    auto expect_type = std::make_shared<Tuple>(type_list);

    auto inferred_shape = infer_impl->InferShape(prim, {logits, labels, weight, reductionPtr->ToAbstract()});
    auto inferred_type = infer_impl->InferType(prim, {logits, labels, weight, reductionPtr->ToAbstract()});

    ShapeCompare(inferred_shape, expect_shape);
    TypeCompare(inferred_type, expect_type);
  } else {
    ASSERT_ANY_THROW(infer_impl->InferShape(prim, {logits, labels, weight, reductionPtr->ToAbstract()}));
  }
}

INSTANTIATE_TEST_CASE_P(
  TestNLLLossGroup, TestNLLLoss,
  testing::Values(
    NLLLossParams{{-1, 3}, kFloat32, {-1}, kInt32, {3}, kFloat32, 0, true, {{-1}, {}}, {kFloat32, kFloat32}},
    NLLLossParams{{-1, 3}, kFloat32, {-1}, kInt32, {3}, kFloat32, 1, true, {{}, {}}, {kFloat32, kFloat32}},
    NLLLossParams{{-1, 3}, kFloat32, {-1}, kInt32, {3}, kFloat32, 2, true, {{}, {}}, {kFloat32, kFloat32}},
    NLLLossParams{{-1, 3}, kFloat32, {-2}, kInt32, {-2}, kFloat32, 0, true, {{-1}, {}}, {kFloat32, kFloat32}},
    NLLLossParams{{-1, 3}, kFloat32, {-2}, kInt32, {-2}, kFloat32, 1, true, {{}, {}}, {kFloat32, kFloat32}},
    NLLLossParams{{-1, 3}, kFloat32, {-2}, kInt32, {-2}, kFloat32, 2, true, {{}, {}}, {kFloat32, kFloat32}},
    NLLLossParams{{-2}, kFloat32, {-2}, kInt32, {-2}, kFloat32, 0, true, {{-1}, {}}, {kFloat32, kFloat32}},
    NLLLossParams{{-2}, kFloat32, {-2}, kInt32, {-2}, kFloat32, 1, true, {{}, {}}, {kFloat32, kFloat32}},
    NLLLossParams{{-2}, kFloat32, {-2}, kInt32, {-2}, kFloat32, 2, true, {{}, {}}, {kFloat32, kFloat32}},
    NLLLossParams{{2, 3}, kFloat32, {2}, kInt32, {3}, kFloat32, 0, true, {{2}, {}}, {kFloat32, kFloat32}},
    NLLLossParams{{2, 3}, kFloat32, {2}, kInt32, {3}, kFloat32, 1, true, {{}, {}}, {kFloat32, kFloat32}},
    NLLLossParams{{2, 3}, kFloat32, {2}, kInt32, {3}, kFloat32, 2, true, {{}, {}}, {kFloat32, kFloat32}},
    NLLLossParams{{2, 3}, kFloat32, {2}, kInt32, {3}, kFloat32, -1, true, {{-1}, {}}, {kFloat32, kFloat32}}, 
    NLLLossParams{{-1, -1}, kFloat32, {-1}, kInt32, {-1}, kFloat32, 0, true, {{-1}, {}}, {kFloat32, kFloat32}},
    NLLLossParams{{-1, -1}, kFloat32, {-1}, kInt32, {-1}, kFloat32, 1, true, {{}, {}}, {kFloat32, kFloat32}},
    NLLLossParams{{-1, -1}, kFloat32, {-1}, kInt32, {-1}, kFloat32, 2, true, {{}, {}}, {kFloat32, kFloat32}},
    NLLLossParams{{-1, -1}, kFloat32, {-1}, kInt32, {-1}, kFloat32, -1, true, {{-1}, {}}, {kFloat32, kFloat32}},
    NLLLossParams{{-2}, kFloat32, {-1}, kInt32, {-1}, kFloat32, 0, true, {{-1}, {}}, {kFloat32, kFloat32}},
    NLLLossParams{{-2}, kFloat32, {-1}, kInt32, {-1}, kFloat32, 1, true, {{}, {}}, {kFloat32, kFloat32}},
    NLLLossParams{{-2}, kFloat32, {-1}, kInt32, {-1}, kFloat32, 2, true, {{}, {}}, {kFloat32, kFloat32}},
    NLLLossParams{{-2}, kFloat32, {-1}, kInt32, {-1}, kFloat32, -1, true, {{-1}, {}}, {kFloat32, kFloat32}},
    NLLLossParams{{2, 3, 4}, kFloat32, {2}, kInt32, {3}, kFloat32, 0, false, {{1}, {}}, {kFloat32, kFloat32}},
    NLLLossParams{{2, 3}, kFloat32, {2, 3}, kInt32, {3}, kFloat32, 0, false, {{1}, {}}, {kFloat32, kFloat32}},
    NLLLossParams{{2, 3}, kFloat32, {2}, kInt32, {2, 3}, kFloat32, 0, false, {{1}, {}}, {kFloat32, kFloat32}},
    NLLLossParams{{2, 3}, kFloat32, {2}, kInt32, {2}, kFloat32, 0, false, {{1}, {}}, {kFloat32, kFloat32}},
    NLLLossParams{{2, 3}, kFloat32, {3}, kInt32, {2}, kFloat32, 0, false, {{1}, {}}, {kFloat32, kFloat32}}));
}  // namespace ops
}  // namespace mindspore
