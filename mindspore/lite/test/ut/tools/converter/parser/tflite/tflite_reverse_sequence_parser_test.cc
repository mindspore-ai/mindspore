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
class TestTfliteParserReverseSequence : public TestTfliteParser {
 public:
  TestTfliteParserReverseSequence() = default;
  void SetUp() override {
    meta_graph = LoadAndConvert("./reverse_sequence.tflite");
  }
};

TEST_F(TestTfliteParserReverseSequence, OpType) {
  ASSERT_NE(meta_graph, nullptr);
  ASSERT_GT(meta_graph->nodes.size(), 0);
  ASSERT_NE(meta_graph->nodes.front()->primitive.get(), nullptr);
  ASSERT_EQ(meta_graph->nodes.front()->primitive->value.type, schema::PrimitiveType_ReverseSequence) << "wrong Op Type";
}

TEST_F(TestTfliteParserReverseSequence, AttrValue) {
  std::vector<int> seq_length{7, 2, 3, 5};
  ASSERT_NE(meta_graph, nullptr);
  ASSERT_GT(meta_graph->nodes.size(), 0);
  ASSERT_NE(meta_graph->nodes.front()->primitive.get(), nullptr);
  ASSERT_NE(meta_graph->nodes.front()->primitive->value.AsReverseSequence(), nullptr);
  ASSERT_EQ(meta_graph->nodes.front()->primitive->value.AsReverseSequence()->seqAxis, 1);
  ASSERT_EQ(meta_graph->nodes.front()->primitive->value.AsReverseSequence()->seqAxis, 1);
  ASSERT_EQ(meta_graph->nodes.front()->primitive->value.AsReverseSequence()->seqLengths, seq_length);
}
}  // namespace mindspore
