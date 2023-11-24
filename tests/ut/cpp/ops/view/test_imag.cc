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

#include "test_view.h"
#include "mindspore/core/ops/view/imag_strides_calc.h"

namespace mindspore {
namespace ops {
class TestViewImag : public TestView {
 public:
  TestViewImag() {}
};

void TestImag(const PrimitivePtr prim, std::vector<int64_t> expect_strides, size_t expect_offset,
              const mindspore::TypePtr dtype) {
  std::vector<int64_t> tensor_data = {1, 2, 3, 4, 5, 6};
  auto input_tensor = std::make_shared<tensor::Tensor>(tensor_data, dtype);
  input_tensor->set_shape({2, 3});
  std::vector<ValuePtr> inputs;
  inputs.emplace_back(input_tensor);
  auto storage_info = ImagCalc(prim, inputs);
  ASSERT_FALSE(storage_info.empty());
  ASSERT_FALSE(storage_info[0]->is_contiguous);
  ASSERT_TRUE(storage_info[0]->strides == expect_strides);
  ASSERT_TRUE(storage_info[0]->storage_offset == expect_offset);
}

/// Feature: Imag strides calculator
/// Description: Test view Imag strides calculator is right
/// Expectation: success
TEST_F(TestViewImag, View) {
  auto prim = std::make_shared<Primitive>("Imag");
  size_t expect_offset = 1;
  std::vector<int64_t> expect_strides({6, 2});
  TestImag(prim, expect_strides, expect_offset, kComplex64);
}
}  // namespace ops
}  // namespace mindspore
