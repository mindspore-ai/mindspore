/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include <climits>
#include "common/common_test.h"
#include "runtime/memory/mem_pool/max_segment_tree.h"

namespace mindspore {
class TestMaxSegmentTree : public UT::Common {
 protected:
  void SetUp() override { tree = std::make_unique<MaxSegmentTree<int>>(10, 3); }

  std::unique_ptr<MaxSegmentTree<int>> tree;
};

/// Feature: MaxSegmentTree
/// Description: test initialization.
/// Expectation: all interface work normally and can not throw exception.
TEST_F(TestMaxSegmentTree, Initialization) {
  // Test query results in initial state
  EXPECT_EQ(tree->Query(0, 4, 0), 0);
  EXPECT_EQ(tree->Query(0, 4, 1), 0);
  EXPECT_EQ(tree->Query(0, 4, 2), 0);
}

/// Feature: MaxSegmentTree
/// Description: test single point update and query.
/// Expectation: all interface work normally and can not throw exception.
TEST_F(TestMaxSegmentTree, SinglePointOperations) {
  // Update a single point
  tree->Update(2, 2, 1, 10);
  EXPECT_EQ(tree->Query(2, 2, 1), 10);

  // Update different positions
  tree->Update(3, 3, 0, 5);
  EXPECT_EQ(tree->Query(3, 3, 0), 5);

  // Update different indices
  tree->Update(1, 1, 2, 7);
  EXPECT_EQ(tree->Query(1, 1, 2), 7);
}

/// Feature: MaxSegmentTree
/// Description: test range update and query.
/// Expectation: all interface work normally and can not throw exception.
TEST_F(TestMaxSegmentTree, RangeOperations) {
  // Update a range
  tree->Update(1, 3, 0, 10);
  EXPECT_EQ(tree->Query(1, 3, 0), 10);

  // Update overlapping range
  tree->Update(2, 4, 1, 15);
  EXPECT_EQ(tree->Query(2, 4, 1), 15);

  // Update entire range
  tree->Update(0, 4, 2, 20);
  EXPECT_EQ(tree->Query(0, 4, 2), 20);
}

/// Feature: MaxSegmentTree
/// Description: test boundary conditions.
/// Expectation: all interface work normally and can not throw exception.
TEST_F(TestMaxSegmentTree, BoundaryConditions) {
  // Test single element range
  tree->Update(0, 0, 0, 1);
  EXPECT_EQ(tree->Query(0, 0, 0), 1);

  // Test entire range
  tree->Update(0, 4, 1, 2);
  EXPECT_EQ(tree->Query(0, 4, 1), 2);

  // Test adjacent elements
  tree->Update(1, 2, 2, 3);
  EXPECT_EQ(tree->Query(1, 2, 2), 3);
}

/// Feature: MaxSegmentTree
/// Description: test exception handling.
/// Expectation: throw exception when input is invalid.
TEST_F(TestMaxSegmentTree, ExceptionHandling) {
  // Test invalid range
  EXPECT_THROW(tree->Query(3, 2, 0), std::exception);
  EXPECT_THROW(tree->Update(3, 2, 0, 1), std::exception);

  // Test invalid index
  EXPECT_THROW(tree->Query(0, 4, 3), std::exception);
  EXPECT_THROW(tree->Update(0, 4, 3, 1), std::exception);
}

/// Feature: MaxSegmentTree
/// Description: test sequential update operations.
/// Expectation: all interface work normally and can not throw exception.
TEST_F(TestMaxSegmentTree, SequentialUpdates) {
  // Update different positions sequentially
  tree->Update(0, 2, 0, 10);
  tree->Update(1, 3, 1, 15);
  tree->Update(2, 4, 2, 20);

  // Verify final results
  EXPECT_EQ(tree->Query(0, 2, 0), 10);
  EXPECT_EQ(tree->Query(1, 3, 1), 15);
  EXPECT_EQ(tree->Query(2, 4, 2), 20);

  // Verify different range combinations
  EXPECT_EQ(tree->Query(0, 4, 0), 10);
  EXPECT_EQ(tree->Query(0, 4, 1), 15);
  EXPECT_EQ(tree->Query(0, 4, 2), 20);
}

/// Feature: MaxSegmentTree
/// Description: test special values.
/// Expectation: all interface work normally and can not throw exception.
TEST_F(TestMaxSegmentTree, SpecialValues) {
  // Test negative numbers
  tree->Update(2, 2, 1, -5);
  EXPECT_EQ(tree->Query(2, 2, 1), 0);  // not supported

  // Test zero
  tree->Update(3, 3, 0, 0);
  EXPECT_EQ(tree->Query(3, 3, 0), 0);

  // Test numbers near maximum value
  tree->Update(1, 1, 2, INT_MAX - 1);
  EXPECT_EQ(tree->Query(1, 1, 2), INT_MAX - 1);
}

/// Feature: MaxSegmentTree
/// Description: test different tree sizes.
/// Expectation: all interface work normally and can not throw exception.
TEST_F(TestMaxSegmentTree, DifferentSizes) {
  MaxSegmentTree<int> small_tree(2, 2);
  MaxSegmentTree<int> large_tree(100000, 5);

  // Test small tree
  small_tree.Update(0, 1, 0, 5);
  EXPECT_EQ(small_tree.Query(0, 1, 0), 5);

  // Test large tree
  large_tree.Update(5, 80000, 3, 10);
  EXPECT_EQ(large_tree.Query(5, 80000, 3), 10);
}

/// Feature: MaxSegmentTree
/// Description: test basic range behavior.
/// Expectation: all interface work normally and can not throw exception.
TEST_F(TestMaxSegmentTree, BasicRangeBehavior) {
  // First update a range
  tree->Update(1, 5, 0, 10);
  EXPECT_EQ(tree->Query(1, 5, 0), 10);

  // Update an overlapping range with larger value
  tree->Update(3, 7, 0, 20);
  EXPECT_EQ(tree->Query(1, 5, 0), 20);  // Original range's maximum is updated
  EXPECT_EQ(tree->Query(3, 7, 0), 20);  // New range's maximum
}

/// Feature: MaxSegmentTree
/// Description: test multiple overlapping ranges.
/// Expectation: all interface work normally and can not throw exception.
TEST_F(TestMaxSegmentTree, MultipleOverlappingRanges) {
  // First update
  tree->Update(0, 4, 1, 5);
  EXPECT_EQ(tree->Query(0, 4, 1), 5);

  // Second update, partial overlap
  tree->Update(2, 6, 1, 10);
  EXPECT_EQ(tree->Query(0, 4, 1), 10);  // Original range's maximum is updated
  EXPECT_EQ(tree->Query(2, 6, 1), 10);

  // Third update, complete overlap
  tree->Update(1, 3, 1, 15);
  EXPECT_EQ(tree->Query(0, 4, 1), 15);  // Original range's maximum is updated again
  EXPECT_EQ(tree->Query(1, 3, 1), 15);
}

/// Feature: MaxSegmentTree
/// Description: test different indices with overlapping ranges.
/// Expectation: all interface work normally and can not throw exception.
TEST_F(TestMaxSegmentTree, DifferentIndicesOverlap) {
  // Update range for first index
  tree->Update(1, 5, 0, 10);
  EXPECT_EQ(tree->Query(1, 5, 0), 10);

  // Update overlapping range for second index
  tree->Update(3, 7, 1, 20);
  EXPECT_EQ(tree->Query(1, 5, 0), 10);  // First index's value remains unchanged
  EXPECT_EQ(tree->Query(3, 7, 1), 20);  // Second index's new value
}

/// Feature: MaxSegmentTree
/// Description: test nested range updates.
/// Expectation: all interface work normally and can not throw exception.
TEST_F(TestMaxSegmentTree, NestedRangeUpdates) {
  // Update large range
  tree->Update(0, 9, 2, 5);
  EXPECT_EQ(tree->Query(0, 9, 2), 5);

  // Update smaller range with larger value
  tree->Update(3, 6, 2, 10);
  EXPECT_EQ(tree->Query(3, 6, 2), 10);  // Smaller range's value
  EXPECT_EQ(tree->Query(0, 9, 2), 10);  // Larger range's value is also updated
  EXPECT_EQ(tree->Query(0, 1, 2), 5);   // Smaller range's value is not updated
}

/// Feature: MaxSegmentTree
/// Description: test sequential range updates.
/// Expectation: all interface work normally and can not throw exception.
TEST_F(TestMaxSegmentTree, SequentialRangeUpdates) {
  // First update
  tree->Update(0, 4, 0, 5);
  EXPECT_EQ(tree->Query(0, 4, 0), 5);

  // Second update, partial overlap
  tree->Update(2, 6, 0, 10);
  EXPECT_EQ(tree->Query(0, 4, 0), 10);
  EXPECT_EQ(tree->Query(2, 6, 0), 10);

  // Third update, complete overlap
  tree->Update(1, 3, 0, 15);
  EXPECT_EQ(tree->Query(0, 4, 0), 15);
  EXPECT_EQ(tree->Query(1, 3, 0), 15);
}

/// Feature: MaxSegmentTree
/// Description: test range updates at boundaries.
/// Expectation: all interface work normally and can not throw exception.
TEST_F(TestMaxSegmentTree, BoundaryRangeUpdates) {
  // Update to boundaries
  tree->Update(0, 9, 0, 10);
  EXPECT_EQ(tree->Query(0, 9, 0), 10);

  // Update middle section
  tree->Update(3, 6, 0, 20);
  EXPECT_EQ(tree->Query(0, 9, 0), 20);
  EXPECT_EQ(tree->Query(3, 6, 0), 20);
}

/// Feature: MaxSegmentTree
/// Description: test complex updates with multiple indices.
/// Expectation: all interface work normally and can not throw exception.
TEST_F(TestMaxSegmentTree, ComplexMultiIndexUpdates) {
  // Update range for first index
  tree->Update(1, 5, 0, 10);
  EXPECT_EQ(tree->Query(1, 5, 0), 10);

  // Update overlapping range for second index
  tree->Update(3, 7, 1, 20);
  EXPECT_EQ(tree->Query(1, 5, 0), 10);
  EXPECT_EQ(tree->Query(3, 7, 1), 20);

  // Update containing range for third index
  tree->Update(0, 9, 2, 30);
  EXPECT_EQ(tree->Query(1, 5, 0), 10);
  EXPECT_EQ(tree->Query(3, 7, 1), 20);
  EXPECT_EQ(tree->Query(0, 9, 2), 30);
}
}  // namespace mindspore