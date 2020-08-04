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
class TestTfliteParserTopKV2 : public TestTfliteParser {
 public:
  TestTfliteParserTopKV2() = default;
  void SetUp() override {
    meta_graph = LoadAndConvert("./topk_v2.tflite");
  }
};

TEST_F(TestTfliteParserTopKV2, OpType) {
  ASSERT_NE(meta_graph, nullptr);
  ASSERT_GT(meta_graph->nodes.size(), 0);
  ASSERT_NE(meta_graph->nodes.front()->primitive.get(), nullptr);
  ASSERT_EQ(meta_graph->nodes.front()->primitive->value.type, schema::PrimitiveType_TopKV2) << "wrong Op Type";
}

TEST_F(TestTfliteParserTopKV2, AttrValue) {
  // attr->sorted default is true
  std::vector<int> k{3};
  ASSERT_NE(meta_graph, nullptr);
  ASSERT_GT(meta_graph->nodes.size(), 0);
  ASSERT_NE(meta_graph->nodes.front()->primitive.get(), nullptr);
  ASSERT_NE(meta_graph->nodes.front()->primitive->value.AsTopKV2(), nullptr);
  ASSERT_EQ(meta_graph->nodes.front()->primitive->value.AsTopKV2()->k, k);
  ASSERT_EQ(meta_graph->nodes.front()->primitive->value.AsTopKV2()->sorted, true);
}
}  // namespace mindspore
