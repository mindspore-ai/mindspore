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

#include "symbol_engine/ops/symbolic_shape_test_utils.h"
#include "common/graph_optimizer_test_framework.h"

namespace mindspore::symshape::test {
/// Feature: Symbolic shape for Conv2D
/// Description: cond2d with padmode="same"
/// Expectation: success.
TEST_F(TestSymbolEngine, conv2d_samepad_1) {
  mindspore::test::ConstructGraph cg;
  auto x = cg.NewTensorInput("x", kFloat32, {4, 3, -1, -1});
  auto w = cg.NewTensorInput("w", kFloat32, {1280, 3, 14, 14});
  auto node = cg.NewCNode("Conv2D", {x, w},
                          {{"kernel_size", MakeValue<ShapeVector>({14, 14})},
                           {"mode", MakeValue<int64_t>(1)},
                           {"out_channel", MakeValue<int64_t>(1280)},
                           {"pad", MakeValue<ShapeVector>({0, 0, 0, 0})},
                           {"pad_mode", MakeValue<int64_t>(1)},
                           {"format", MakeValue<std::string>("NCHW")},
                           {"groups", MakeValue<int64_t>(1)},
                           {"group", MakeValue<int64_t>(1)},
                           {"stride", MakeValue<ShapeVector>({1, 1, 14, 14})},
                           {"dilation", MakeValue<ShapeVector>({1, 1, 1, 1})}});
  helper_->InitSymbolEngine(cg.GetGraph());
  auto out_shape = helper_->BuildSymbolicShape(node);
  UT_CHECK_NULL(out_shape);
  ASSERT_TRUE(helper_->SupportInfer());
  ASSERT_TRUE(helper_->CheckSymbolicShapeMatchesDigitalShape(node));
}

/// Feature: Symbolic shape for Conv2D
/// Description: cond2d with padmode="valid"
/// Expectation: success.
TEST_F(TestSymbolEngine, conv2d_validpad_1) {
  mindspore::test::ConstructGraph cg;
  auto x = cg.NewTensorInput("x", kFloat32, {1, 128, -1, -1});
  auto w = cg.NewTensorInput("w", kFloat32, {128, 128, 3, 3});
  auto node = cg.NewCNode("Conv2D", {x, w},
                          {{"kernel_size", MakeValue<ShapeVector>({3, 3})},
                           {"mode", MakeValue<int64_t>(1)},
                           {"out_channel", MakeValue<int64_t>(128)},
                           {"pad", MakeValue<ShapeVector>({0, 0, 0, 0})},
                           {"pad_mode", MakeValue<std::string>("valid")},
                           {"format", MakeValue<std::string>("NCHW")},
                           {"groups", MakeValue<int64_t>(1)},
                           {"group", MakeValue<int64_t>(1)},
                           {"stride", MakeValue<ShapeVector>({1, 1, 2, 2})},
                           {"dilation", MakeValue<ShapeVector>({1, 1, 1, 1})}});
  helper_->InitSymbolEngine(cg.GetGraph());
  IntSymbolInfo sym_h;
  sym_h.divisor = 64;
  sym_h.remainder = 2;
  IntSymbolInfo sym_w;
  sym_w.divisor = 64;
  sym_w.remainder = 3;
  helper_->SetSymbolicShapeInfo(x, {{}, {}, sym_h, sym_w});
  auto out_shape = helper_->BuildSymbolicShape(node);
  UT_CHECK_NULL(out_shape);
  ASSERT_TRUE(helper_->SupportInfer());
  ASSERT_TRUE(helper_->CheckSymbolicShapeMatchesDigitalShape(node));
  auto out_h = out_shape->item_as<IntSymbol>(2);
  auto out_w = out_shape->item_as<IntSymbol>(3);
  EXPECT_TRUE(out_h->is_divisible_by(32));
  EXPECT_FALSE(out_w->is_divisible_by(32));
}
}  // namespace mindspore::symshape::test
