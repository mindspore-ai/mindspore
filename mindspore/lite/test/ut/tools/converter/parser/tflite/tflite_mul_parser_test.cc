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
class TestTfliteParserMul1 : public TestTfliteParser {
 public:
  TestTfliteParserMul1() = default;
  void SetUp() override { meta_graph = LoadAndConvert("./mul1.tflite", ""); }
};

TEST_F(TestTfliteParserMul1, OpType) {
  ASSERT_GT(meta_graph->nodes.size(), 0);
  ASSERT_NE(meta_graph->nodes.front()->primitive.get(), nullptr);
  ASSERT_EQ(meta_graph->nodes.front()->primitive->value.type, schema::PrimitiveType_Mul) << "wrong Op Type";
}

TEST_F(TestTfliteParserMul1, Tensor) {
  ASSERT_GT(meta_graph->allTensors.size(), 0);
  ASSERT_EQ(meta_graph->allTensors.at(0)->data.size(), 0);
  ASSERT_GT(meta_graph->allTensors.at(1)->data.size(), 0);
  ASSERT_EQ(meta_graph->allTensors.at(2)->data.size(), 0);
}

class TestTfliteParserMul2 : public TestTfliteParser {
 public:
  TestTfliteParserMul2() = default;
  void SetUp() override { meta_graph = LoadAndConvert("./mul2.tflite", ""); }
};

TEST_F(TestTfliteParserMul2, OpType) {
  ASSERT_GT(meta_graph->nodes.size(), 0);
  ASSERT_NE(meta_graph->nodes.front()->primitive.get(), nullptr);
  ASSERT_EQ(meta_graph->nodes.front()->primitive->value.type, schema::PrimitiveType_Mul) << "wrong Op Type";
}

TEST_F(TestTfliteParserMul2, Tensor) {
  ASSERT_GT(meta_graph->allTensors.size(), 0);
  ASSERT_EQ(meta_graph->allTensors.at(0)->data.size(), 0);
  ASSERT_GT(meta_graph->allTensors.at(1)->data.size(), 0);
  ASSERT_EQ(meta_graph->allTensors.at(2)->data.size(), 0);
}

class TestTfliteParserMul3 : public TestTfliteParser {
 public:
  TestTfliteParserMul3() = default;
  void SetUp() override { meta_graph = LoadAndConvert("./mul3.tflite", ""); }
};

TEST_F(TestTfliteParserMul3, OpType) {
  ASSERT_GT(meta_graph->nodes.size(), 0);
  ASSERT_NE(meta_graph->nodes.front()->primitive.get(), nullptr);
  ASSERT_EQ(meta_graph->nodes.front()->primitive->value.type, schema::PrimitiveType_Mul) << "wrong Op Type";
}

TEST_F(TestTfliteParserMul3, Tensor) {
  ASSERT_GT(meta_graph->allTensors.size(), 0);
  ASSERT_EQ(meta_graph->allTensors.at(0)->data.size(), 0);
  ASSERT_EQ(meta_graph->allTensors.at(1)->data.size(), 0);
  ASSERT_EQ(meta_graph->allTensors.at(2)->data.size(), 0);
}

}  // namespace mindspore
