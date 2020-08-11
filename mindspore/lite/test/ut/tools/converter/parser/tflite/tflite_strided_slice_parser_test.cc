/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "ut/tools/converter/parser/tflite/tflite_parsers_test_utils.h"
#include <iostream>
#include "common/common_test.h"

namespace mindspore {
class TestTfliteParserStridedSlice : public TestTfliteParser {
 public:
  TestTfliteParserStridedSlice() = default;
  void SetUp() override {
    meta_graph = LoadAndConvert("./strided_slice.tflite");
  }
};

TEST_F(TestTfliteParserStridedSlice, OpType) {
  ASSERT_NE(meta_graph, nullptr);
  ASSERT_GT(meta_graph->nodes.size(), 0);
  ASSERT_NE(meta_graph->nodes.front()->primitive.get(), nullptr);
  ASSERT_EQ(meta_graph->nodes.front()->primitive->value.type, schema::PrimitiveType_StridedSlice) << "wrong Op Type";
}

TEST_F(TestTfliteParserStridedSlice, AttrValue) {
  std::vector<int> begin{1, -1, 0};
  std::vector<int> end{2, -3, 3};
  std::vector<int> stride{1, -1, 1};
  std::vector<int> isscale{3, 2, 3};
  ASSERT_NE(meta_graph, nullptr);
  ASSERT_GT(meta_graph->nodes.size(), 0);
  ASSERT_NE(meta_graph->nodes.front()->primitive.get(), nullptr);
  ASSERT_NE(meta_graph->nodes.front()->primitive->value.AsStridedSlice(), nullptr);
  ASSERT_EQ(meta_graph->nodes.front()->primitive->value.AsStridedSlice()->beginMask, 0);
  ASSERT_EQ(meta_graph->nodes.front()->primitive->value.AsStridedSlice()->endMask, 0);
  ASSERT_EQ(meta_graph->nodes.front()->primitive->value.AsStridedSlice()->beginMask, 0);
  ASSERT_EQ(meta_graph->nodes.front()->primitive->value.AsStridedSlice()->beginMask, 0);
  ASSERT_EQ(meta_graph->nodes.front()->primitive->value.AsStridedSlice()->begin, begin);
  ASSERT_EQ(meta_graph->nodes.front()->primitive->value.AsStridedSlice()->end, end);
  ASSERT_EQ(meta_graph->nodes.front()->primitive->value.AsStridedSlice()->stride, stride);
  ASSERT_EQ(meta_graph->nodes.front()->primitive->value.AsStridedSlice()->isScale, isscale);
}
}  // namespace mindspore
