/**
 * Copyright 2026 Huawei Technologies Co., Ltd
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <exception>
#include <memory>
#include <optional>
#include <tuple>
#include <vector>
#include "gtest/gtest.h"
#include "mockcpp/mockcpp.hpp"
#include "ir/tensor_new.h"
#include "ir/tensor_storage_info.h"
#include "ops/op_def.h"
#include "mindspore/ops/kernel/cpu/pyboost/pyboost_cpu_kernel_extra_func.h"

namespace mindspore::kernel::pyboost {
namespace {
tensor::TensorPtr MakeFallbackTestTensor(bool contiguous) {
  auto tensor = tensor::from_spec(kNumberTypeFloat32, {2, 2}, device::DeviceType::kNone);
  if (!contiguous) {
    tensor->set_storage_info(std::make_shared<TensorStorageInfo>(ShapeVector{2, 2}, ShapeVector{1, 2},
                                                                 ShapeVector{2, 2}, ShapeVector{2, 1}, false));
  }
  return tensor;
}
}  // namespace

class CpuFallbackInputsTest : public testing::Test {
 protected:
  void TearDown() override { GlobalMockObject::verify(); }
};

/// Feature: Native CPU fallback eligibility.
/// Description: Use actual OpDef registrations for ordinary, missing and View operators.
/// Expectation: Missing/ordinary definitions are enabled; View and recursive operators are excluded.
TEST_F(CpuFallbackInputsTest, Policy) {
  ASSERT_EQ(ops::GetOpDef("CpuFallbackUTMissingOp"), nullptr);
  ASSERT_NE(ops::GetOpDef("MatMulExt"), nullptr);
  ASSERT_NE(ops::GetOpDef("RealView"), nullptr);
  EXPECT_TRUE(PrepareCpuFallbackEnabled("CpuFallbackUTMissingOp"));
  EXPECT_TRUE(PrepareCpuFallbackEnabled("MatMulExt"));
  EXPECT_FALSE(PrepareCpuFallbackEnabled("RealView"));
  EXPECT_FALSE(PrepareCpuFallbackEnabled("Contiguous"));
  EXPECT_FALSE(PrepareCpuFallbackEnabled("Copy"));
}

/// Feature: Native CPU Tensor preparation.
/// Description: Pass real continuous/strided tensors and mock only the CPU Contiguous execution boundary.
/// Expectation: Continuous tensor is preserved, strided tensor is replaced without changing its layout.
TEST_F(CpuFallbackInputsTest, TensorAndOptional) {
  auto contiguous = MakeFallbackTestTensor(true);
  auto strided = MakeFallbackTestTensor(false);
  ASSERT_TRUE(contiguous->is_contiguous());
  ASSERT_FALSE(strided->is_contiguous());
  MOCKER(MakeCpuFallbackContiguous).stubs().will(returnValue(contiguous));
  bool changed = false;
  EXPECT_EQ(PrepareCpuFallbackInput(contiguous, &changed), contiguous);
  EXPECT_FALSE(changed);
  EXPECT_EQ(PrepareCpuFallbackInput(strided, &changed), contiguous);
  EXPECT_TRUE(changed);
  EXPECT_FALSE(strided->is_contiguous());
  changed = false;
  EXPECT_FALSE(PrepareCpuFallbackInput(std::optional<tensor::TensorPtr>{}, &changed).has_value());
  EXPECT_FALSE(changed);
  EXPECT_EQ(PrepareCpuFallbackInput(std::optional<tensor::TensorPtr>{strided}, &changed).value(), contiguous);
  EXPECT_TRUE(changed);
  ValuePtr value = strided;
  EXPECT_EQ(PrepareCpuFallbackInput(value, &changed), contiguous);
  EXPECT_EQ(PrepareCpuFallbackInput(std::optional<ValuePtr>{value}, &changed).value(), contiguous);
}

/// Feature: Native CPU scalar sequence preparation.
/// Description: Pass real scalar values, scalar tuples and empty tuples.
/// Expectation: Preserve pointers and optional None without marking an input change.
TEST_F(CpuFallbackInputsTest, ScalarAndEmptySequences) {
  auto scalar = MakeValue(int64_t{7});
  auto empty = std::make_shared<ValueTuple>(std::vector<ValuePtr>{});
  auto scalars = std::make_shared<ValueTuple>(std::vector<ValuePtr>{scalar, scalar});
  MOCKER(MakeCpuFallbackContiguous).expects(never());
  bool changed = false;
  EXPECT_EQ(PrepareCpuFallbackInput(scalar, &changed), scalar);
  EXPECT_EQ(PrepareCpuFallbackInput(empty, &changed), empty);
  EXPECT_EQ(PrepareCpuFallbackInput(scalars, &changed), scalars);
  EXPECT_FALSE(PrepareCpuFallbackInput(std::optional<ValueTuplePtr>{}, &changed).has_value());
  EXPECT_FALSE(changed);
}

/// Feature: Native CPU recursive Tensor tuple preparation.
/// Description: Convert a strided Tensor leaf inside nested real ValueTuple objects.
/// Expectation: Rebuild only affected tuples while preserving original tensors and scalar leaves.
TEST_F(CpuFallbackInputsTest, NestedSequences) {
  auto scalar = MakeValue(int64_t{7});
  auto contiguous = MakeFallbackTestTensor(true);
  auto strided = MakeFallbackTestTensor(false);
  auto inner = std::make_shared<ValueTuple>(std::vector<ValuePtr>{contiguous, strided});
  auto outer = std::make_shared<ValueTuple>(std::vector<ValuePtr>{scalar, inner});
  MOCKER(MakeCpuFallbackContiguous).stubs().will(returnValue(contiguous));
  bool changed = false;
  auto result = PrepareCpuFallbackInput(outer, &changed);
  ASSERT_TRUE(changed);
  ASSERT_NE(result, outer);
  EXPECT_EQ(result->value()[0], scalar);
  auto result_inner = result->value()[1]->cast<ValueTuplePtr>();
  ASSERT_NE(result_inner, nullptr);
  EXPECT_EQ(result_inner->value()[0], contiguous);
  EXPECT_EQ(result_inner->value()[1], contiguous);
  EXPECT_EQ(inner->value()[1], strided);
  changed = false;
  EXPECT_EQ(PrepareCpuFallbackInput(result, &changed), result);
  EXPECT_FALSE(changed);
  EXPECT_NE(PrepareCpuFallbackInput(std::optional<ValueTuplePtr>{outer}, &changed).value(), outer);
  EXPECT_TRUE(changed);
  auto empty = std::make_shared<ValueTuple>(std::vector<ValuePtr>{});
  EXPECT_EQ(PrepareCpuFallbackInput(empty, &changed), empty);
  EXPECT_TRUE(changed);
}

/// Feature: CPU fallback argument pack forwarding.
/// Description: Prepare mixed/empty argument packs twice, allowing exactly one actual conversion.
/// Expectation: Scalars remain unchanged and already-prepared inputs do not trigger another conversion.
TEST_F(CpuFallbackInputsTest, ArgumentPack) {
  auto contiguous = MakeFallbackTestTensor(true);
  auto strided = MakeFallbackTestTensor(false);
  auto scalar = MakeValue(int64_t{7});
  MOCKER(MakeCpuFallbackContiguous).expects(once()).will(returnValue(contiguous));
  bool changed = false;
  auto result = PrepareCpuFallbackInputs(std::make_tuple(strided, scalar, int64_t{2}), &changed);
  EXPECT_TRUE(changed);
  EXPECT_EQ(std::get<0>(result), contiguous);
  EXPECT_EQ(std::get<1>(result), scalar);
  EXPECT_EQ(std::get<2>(result), 2);
  changed = false;
  (void)PrepareCpuFallbackInputs(result, &changed);
  EXPECT_FALSE(changed);
  auto empty = PrepareCpuFallbackInputs(std::tuple<>{}, &changed);
  static_assert(std::tuple_size<decltype(empty)>::value == 0);
  EXPECT_FALSE(changed);
}

/// Feature: CPU fallback result validation.
/// Description: CPU Contiguous unexpectedly returns a real strided tensor.
/// Expectation: Reject the invalid result rather than re-entering Call indefinitely.
TEST_F(CpuFallbackInputsTest, InvalidContiguousResult) {
  auto strided = MakeFallbackTestTensor(false);
  MOCKER(MakeCpuFallbackContiguous).stubs().will(returnValue(strided));
  bool changed = false;
  EXPECT_THROW(PrepareCpuFallbackInput(strided, &changed), std::exception);
  EXPECT_FALSE(changed);
}

/// Feature: CPU fallback null pointer validation.
/// Description: Pass null Tensor, Value and tuple pointers and mock a null conversion result.
/// Expectation: Reject invalid pointers before dereferencing or rebuilding arguments.
TEST_F(CpuFallbackInputsTest, NullInputsAndResult) {
  bool changed = false;
  EXPECT_THROW(PrepareCpuFallbackInput(tensor::TensorPtr{}, &changed), std::exception);
  EXPECT_THROW(PrepareCpuFallbackInput(ValuePtr{}, &changed), std::exception);
  EXPECT_THROW(PrepareCpuFallbackInput(ValueTuplePtr{}, &changed), std::exception);
  MOCKER(MakeCpuFallbackContiguous).stubs().will(returnValue(tensor::TensorPtr{}));
  EXPECT_THROW(PrepareCpuFallbackInput(MakeFallbackTestTensor(false), &changed), std::exception);
}
}  // namespace mindspore::kernel::pyboost
