/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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
#include <string>

#include "common/common_test.h"
#include "frontend/jit/ps/pass.h"
#include "ir/anf.h"
#include "ir/func_graph.h"
#include "ir/manager.h"
#include "primitive/math_ops.h"

namespace mindspore {
namespace pipeline {
class TestExtractAnnotationPass : public UT::Common {
 public:
  TestExtractAnnotationPass() : func_graph_(std::make_shared<FuncGraph>()) {}
  ~TestExtractAnnotationPass() = default;

  CNodePtr CreateTestCNode() {
    auto param = func_graph_->add_parameter();
    auto prim = prim::kPrimMod;
    auto value_node = NewValueNode(prim);
    return func_graph_->NewCNode({value_node, param});
  }

 private:
  FuncGraphPtr func_graph_;
};

/// Feature: ParseOffloadAnnotation
/// Description: Test parsing complete offload annotation (release + prefetch + on)
/// Expectation: All three attributes should be correctly set
TEST_F(TestExtractAnnotationPass, TestParseFullAnnotation) {
  auto cnode = CreateTestCNode();
  std::string source_line = "y = x * 2  # @jit.offload: release[2] prefetch[1] on";

  ParseOffloadAnnotation(cnode, source_line);

  // Verify release_distance
  auto release_attr = cnode->GetAttr("release_distance");
  ASSERT_EQ(GetValue<size_t>(release_attr), 2);

  // Verify prefetch_distance
  auto prefetch_attr = cnode->GetAttr("prefetch_distance");
  ASSERT_EQ(GetValue<size_t>(prefetch_attr), 1);

  // Verify enable_offload
  auto enable_attr = cnode->GetAttr("enable_offload");
  ASSERT_TRUE(GetValue<bool>(enable_attr));
}

/// Feature: ParseOffloadAnnotation
/// Description: Test annotation with only release parameter
/// Expectation: Only release_distance attribute should be set
TEST_F(TestExtractAnnotationPass, TestParseReleaseOnly) {
  auto cnode = CreateTestCNode();
  std::string source_line = "z = x + 1  # @jit.offload: release[5]";

  ParseOffloadAnnotation(cnode, source_line);

  // Verify release_distance
  ASSERT_EQ(GetValue<size_t>(cnode->GetAttr("release_distance")), 5);

  // Verify other attributes do not exist
  ASSERT_FALSE(cnode->HasAttr("prefetch_distance"));
  ASSERT_FALSE(cnode->HasAttr("enable_offload"));
}

/// Feature: ParseOffloadAnnotation
/// Description: Test annotation with only prefetch parameter
/// Expectation: Only prefetch_distance attribute should be set
TEST_F(TestExtractAnnotationPass, TestParsePrefetchOnly) {
  auto cnode = CreateTestCNode();
  std::string source_line = "a = x * 3  # @jit.offload: prefetch[10]";

  ParseOffloadAnnotation(cnode, source_line);

  // Verify prefetch_distance
  ASSERT_EQ(GetValue<size_t>(cnode->GetAttr("prefetch_distance")), 10);

  // Verify other attributes do not exist
  ASSERT_FALSE(cnode->HasAttr("release_distance"));
  ASSERT_FALSE(cnode->HasAttr("enable_offload"));
}

/// Feature: ParseOffloadAnnotation
/// Description: Test annotation with only on flag
/// Expectation: Only enable_offload attribute should be set to true
TEST_F(TestExtractAnnotationPass, TestParseOnOnly) {
  auto cnode = CreateTestCNode();
  std::string source_line = "b = x - 1  # @jit.offload: on";

  ParseOffloadAnnotation(cnode, source_line);

  // Verify enable_offload
  ASSERT_TRUE(GetValue<bool>(cnode->GetAttr("enable_offload")));

  // Verify other attributes do not exist
  ASSERT_FALSE(cnode->HasAttr("release_distance"));
  ASSERT_FALSE(cnode->HasAttr("prefetch_distance"));
}

/// Feature: ParseOffloadAnnotation
/// Description: Test annotation with off flag
/// Expectation: enable_offload attribute should be set to false
TEST_F(TestExtractAnnotationPass, TestParseOffOnly) {
  auto cnode = CreateTestCNode();
  std::string source_line = "c = x / 2  # @jit.offload: off";

  ParseOffloadAnnotation(cnode, source_line);

  // Verify enable_offload
  ASSERT_FALSE(GetValue<bool>(cnode->GetAttr("enable_offload")));

  // Verify other attributes do not exist
  ASSERT_FALSE(cnode->HasAttr("release_distance"));
  ASSERT_FALSE(cnode->HasAttr("prefetch_distance"));
}

/// Feature: ParseOffloadAnnotation
/// Description: Test release + on combination
/// Expectation: Both release_distance and enable_offload should be correctly set
TEST_F(TestExtractAnnotationPass, TestParseReleaseWithOn) {
  auto cnode = CreateTestCNode();
  std::string source_line = "d = x % 3  # @jit.offload: release[7] on";

  ParseOffloadAnnotation(cnode, source_line);

  ASSERT_EQ(GetValue<size_t>(cnode->GetAttr("release_distance")), 7);
  ASSERT_TRUE(GetValue<bool>(cnode->GetAttr("enable_offload")));

  ASSERT_FALSE(cnode->HasAttr("prefetch_distance"));
}

/// Feature: ParseOffloadAnnotation
/// Description: Test prefetch + off combination
/// Expectation: Both prefetch_distance and enable_offload should be correctly set
TEST_F(TestExtractAnnotationPass, TestParsePrefetchWithOff) {
  auto cnode = CreateTestCNode();
  std::string source_line = "e = x ** 2  # @jit.offload: prefetch[3] off";

  ParseOffloadAnnotation(cnode, source_line);

  ASSERT_EQ(GetValue<size_t>(cnode->GetAttr("prefetch_distance")), 3);
  ASSERT_FALSE(GetValue<bool>(cnode->GetAttr("enable_offload")));

  ASSERT_FALSE(cnode->HasAttr("release_distance"));
}

/// Feature: ParseOffloadAnnotation
/// Description: Test boundary value - zero values
/// Expectation: Should correctly handle release[0] and prefetch[0]
TEST_F(TestExtractAnnotationPass, TestParseZeroValues) {
  auto cnode = CreateTestCNode();
  std::string source_line = "f = x + y  # @jit.offload: release[0] prefetch[0] on";

  ParseOffloadAnnotation(cnode, source_line);

  ASSERT_EQ(GetValue<size_t>(cnode->GetAttr("release_distance")), 0);
  ASSERT_EQ(GetValue<size_t>(cnode->GetAttr("prefetch_distance")), 0);
  ASSERT_TRUE(GetValue<bool>(cnode->GetAttr("enable_offload")));
}

/// Feature: ParseOffloadAnnotation
/// Description: Test boundary value - large values
/// Expectation: Should correctly handle large numbers
TEST_F(TestExtractAnnotationPass, TestParseLargeValues) {
  auto cnode = CreateTestCNode();
  std::string source_line = "g = ops.add(x, y)  # @jit.offload: release[9999] prefetch[8888] on";

  ParseOffloadAnnotation(cnode, source_line);

  ASSERT_EQ(GetValue<size_t>(cnode->GetAttr("release_distance")), 9999);
  ASSERT_EQ(GetValue<size_t>(cnode->GetAttr("prefetch_distance")), 8888);
  ASSERT_TRUE(GetValue<bool>(cnode->GetAttr("enable_offload")));
}

/// Feature: ParseOffloadAnnotation
/// Description: Test annotation with extra spaces
/// Expectation: Should correctly handle space variations
TEST_F(TestExtractAnnotationPass, TestParseWithExtraSpaces) {
  auto cnode = CreateTestCNode();
  std::string source_line = "h = x - y  # @jit.offload:   release[4]   prefetch[6]   on";

  ParseOffloadAnnotation(cnode, source_line);

  ASSERT_EQ(GetValue<size_t>(cnode->GetAttr("release_distance")), 4);
  ASSERT_EQ(GetValue<size_t>(cnode->GetAttr("prefetch_distance")), 6);
  ASSERT_TRUE(GetValue<bool>(cnode->GetAttr("enable_offload")));
}

/// Feature: ParseOffloadAnnotation
/// Description: Test case without annotation
/// Expectation: No attributes should be set
TEST_F(TestExtractAnnotationPass, TestParseNoAnnotation) {
  auto cnode = CreateTestCNode();
  std::string source_line = "i = x * y  # normal comment";

  ParseOffloadAnnotation(cnode, source_line);

  // Verify no attributes are set
  ASSERT_FALSE(cnode->HasAttr("release_distance"));
  ASSERT_FALSE(cnode->HasAttr("prefetch_distance"));
  ASSERT_FALSE(cnode->HasAttr("enable_offload"));
}

/// Feature: ParseOffloadAnnotation
/// Description: Test empty string
/// Expectation: No attributes should be set
TEST_F(TestExtractAnnotationPass, TestParseEmptyString) {
  auto cnode = CreateTestCNode();
  std::string source_line = "";

  ParseOffloadAnnotation(cnode, source_line);

  ASSERT_FALSE(cnode->HasAttr("release_distance"));
  ASSERT_FALSE(cnode->HasAttr("prefetch_distance"));
  ASSERT_FALSE(cnode->HasAttr("enable_offload"));
}

/// Feature: ParseOffloadAnnotation
/// Description: Test invalid format should not set attributes
/// Expectation: Invalid format annotation should not set any attributes
TEST_F(TestExtractAnnotationPass, TestParseInvalidFormat) {
  // Test invalid format 1: missing colon
  auto cnode1 = CreateTestCNode();
  std::string invalid1 = "k = x + y  # @jit.offload release[2]";
  ParseOffloadAnnotation(cnode1, invalid1);
  ASSERT_FALSE(cnode1->HasAttr("release_distance"));

  // Test invalid format 2: using parentheses
  auto cnode2 = CreateTestCNode();
  std::string invalid2 = "l = x - y  # @jit.offload: release(3)";
  ParseOffloadAnnotation(cnode2, invalid2);
  ASSERT_FALSE(cnode2->HasAttr("release_distance"));

  // Test invalid format 3: non-numeric value
  auto cnode3 = CreateTestCNode();
  std::string invalid3 = "m = x * y  # @jit.offload: release[abc]";
  ParseOffloadAnnotation(cnode3, invalid3);
  ASSERT_FALSE(cnode3->HasAttr("release_distance"));

  // Test invalid format 4: invalid switch value
  auto cnode4 = CreateTestCNode();
  std::string invalid4 = "n = x / y  # @jit.offload: maybe";
  ParseOffloadAnnotation(cnode4, invalid4);
  ASSERT_FALSE(cnode4->HasAttr("enable_offload"));
}

/// Feature: ParseOffloadAnnotation
/// Description: Test extracting user code from complete call stack
/// Expectation: Should correctly extract user code from call stack containing framework code and parse annotation
TEST_F(TestExtractAnnotationPass, TestParseFromCallStack) {
  auto cnode = CreateTestCNode();

  // Simulate complete call stack (similar to user reported case)
  std::string call_stack =
    "# In file test/test.py:47~54, 4~17/    @jit(backend=\"ms_backend\", auto_offload=\"all\")/\n"
    "# In file test/test.py:53, 13~38/        m6 = mindspore.ops.sub(m1, m5) # @jit.offload: on/\n"
    "# In file test/test.py:53, 13~30/        m6 = mindspore.ops.sub(m1, m5) # @jit.offload: on/<~~This line of code "
    "can be shared by multiple nodes, and may be duplicated./\n"
    "# In file mindspore/ops/auto_generate/gen_ops_def.py:3432~3464/def sub(input, other):/\n"
    "# In file mindspore/ops/auto_generate/gen_ops_def.py:3464, 11~31/    return sub_op(input, other)/";

  ParseOffloadAnnotation(cnode, call_stack);

  // Verify correct extraction and parsing of annotation in user code
  ASSERT_TRUE(GetValue<bool>(cnode->GetAttr("enable_offload")));

  ASSERT_FALSE(cnode->HasAttr("release_distance"));
  ASSERT_FALSE(cnode->HasAttr("prefetch_distance"));
}

/// Feature: ParseOffloadAnnotation
/// Description: Test call stack with only framework code without user annotation
/// Expectation: No attributes should be set
TEST_F(TestExtractAnnotationPass, TestParseCallStackNoAnnotation) {
  auto cnode = CreateTestCNode();

  // Simulate call stack without annotation
  std::string call_stack =
    "In file mindspore/ops/auto_generate/gen_ops_def.py:1126, 11~31/    return add_op(input, other)/\n"
    "In file test/test.py:68, 13~18/        y2 = x + 2/";

  ParseOffloadAnnotation(cnode, call_stack);

  // Verify no attributes are set
  ASSERT_FALSE(cnode->HasAttr("release_distance"));
  ASSERT_FALSE(cnode->HasAttr("prefetch_distance"));
  ASSERT_FALSE(cnode->HasAttr("enable_offload"));
}

}  // namespace pipeline
}  // namespace mindspore
