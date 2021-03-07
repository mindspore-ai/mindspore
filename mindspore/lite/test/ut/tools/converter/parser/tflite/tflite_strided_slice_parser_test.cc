/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
  void SetUp() override { meta_graph = LoadAndConvert("./strided_slice.tflite"); }
};

TEST_F(TestTfliteParserStridedSlice, OpType) {
  ASSERT_NE(meta_graph, nullptr);
  ASSERT_GT(meta_graph->nodes.size(), 0);
  ASSERT_NE(meta_graph->nodes.front()->primitive.get(), nullptr);
  ASSERT_EQ(meta_graph->nodes.front()->primitive->value.type, schema::PrimitiveType_StridedSlice) << "wrong Op Type";
}

TEST_F(TestTfliteParserStridedSlice, AttrValue) {
  ASSERT_NE(meta_graph->nodes.front()->primitive->value.AsStridedSlice(), nullptr);
  auto val = meta_graph->nodes.front()->primitive->value.AsStridedSlice();
  ASSERT_EQ(val->end_mask, 0);
  ASSERT_EQ(val->begin_mask, 0);
}
}  // namespace mindspore
