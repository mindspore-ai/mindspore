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
#include "common/common_test.h"
#include "ops/mul.h"
#include "ir/dtype/type.h"
#include "abstract/dshape.h"
#include "utils/tensor_construct_utils.h"
#include "ir/primitive.h"
#include "abstract/abstract_value.h"
#include "ops/test_ops.h"

namespace mindspore {
namespace ops {
struct MulParams {
  ShapeVector x_shape;
  TypePtr x_type;
  ShapeVector y_shape;
  TypePtr y_type;
  bool is_success;
  ShapeVector out_shape;
  TypePtr out_type;
};

struct MulShapes {
  ShapeVector x_shape;
  ShapeVector y_shape;
  bool is_success;
  ShapeVector out_shape;
};

struct MulTypes {
  TypePtr x_type;
  TypePtr y_type;
  bool is_success;
  TypePtr out_type;
};

class TestMul : public UT::Common, public testing::WithParamInterface<std::tuple<MulShapes, MulTypes>> {
 public:
  TestMul() {}
  void SetUp() {}
  void TearDown() {}
};

TEST_P(TestMul, test_mul) {
  const auto &shape_param = std::get<0>(GetParam());
  const auto &type_param = std::get<1>(GetParam());
  auto prim = std::make_shared<Primitive>(kNameMul);
  auto mul = std::make_shared<Mul>();
  mul->Init();
  auto x = abstract::MakeAbstract(std::make_shared<abstract::Shape>(shape_param.x_shape), type_param.x_type);
  auto y = abstract::MakeAbstract(std::make_shared<abstract::Shape>(shape_param.y_shape), type_param.y_type);
  MS_EXCEPTION_IF_NULL(x);
  MS_EXCEPTION_IF_NULL(y);
  if (shape_param.is_success && type_param.is_success) {
    auto expect = abstract::MakeAbstract(std::make_shared<abstract::Shape>(shape_param.out_shape), type_param.out_type);
    auto out_abstract = MulInfer(nullptr, prim, {x, y});
    ASSERT_NE(out_abstract, nullptr);
    ASSERT_TRUE(*out_abstract == *expect);
  } else {
    ASSERT_ANY_THROW(MulInfer(nullptr, prim, {x, y}));
  }
}

INSTANTIATE_TEST_CASE_P(
  TestMul, TestMul,
  testing::Combine(testing::ValuesIn({
                     MulShapes{{1, 3}, {3, 1}, true, {3, 3}},
                     MulShapes{{3, 1, 3}, {3, 3, 1}, true, {3, 3, 3}},
                     MulShapes{{3, 3, 1}, {3, 3, 1}, true, {3, 3, 1}},
                     MulShapes{{3, 1, 2, 4}, {1, 3, 2, 4}, true, {3, 3, 2, 4}},
                     MulShapes{{1, 3}, {2, 1}, true, {2, 3}},
                     MulShapes{{3, 1, 3}, {3, 2, 1}, true, {3, 2, 3}},
                     MulShapes{{1}, {1}, true, {1}},
                     MulShapes{{1, 3, 2, 4}, {3, 1, 4, 2}, false},  
                     MulShapes{{3, 2, 4}, {4, 2}, false},           
                   }),
                   testing::ValuesIn({
                     MulTypes{kFloat32, kFloat32, true, kFloat32},
                     MulTypes{kFloat16, kFloat16, true, kFloat16},
                     MulTypes{kInt32, kInt32, true, kInt32},
                     MulTypes{kInt8, kInt8, true, kInt8}, 
                     MulTypes{kInt16, kInt16, true, kInt16}, 
                   })));

}  // namespace ops
}  // namespace mindspore
