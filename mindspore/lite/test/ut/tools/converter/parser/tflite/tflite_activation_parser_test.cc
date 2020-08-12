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

class TestTfliteParserRelu : public TestTfliteParser {
 public:
  TestTfliteParserRelu() = default;
  void SetUp() override { meta_graph = LoadAndConvert("./relu.tflite", ""); }
};

TEST_F(TestTfliteParserRelu, OpType) {
  ASSERT_GT(meta_graph->nodes.size(), 0);
  ASSERT_NE(meta_graph->nodes.front()->primitive.get(), nullptr);
  ASSERT_EQ(meta_graph->nodes.front()->primitive->value.type, schema::PrimitiveType_Activation) << "wrong Op Type";
}

class TestTfliteParserRelu6 : public TestTfliteParser {
 public:
  TestTfliteParserRelu6() = default;
  void SetUp() override { meta_graph = LoadAndConvert("./relu6.tflite", ""); }
};

TEST_F(TestTfliteParserRelu6, OpType) {
  ASSERT_GT(meta_graph->nodes.size(), 0);
  ASSERT_NE(meta_graph->nodes.front()->primitive.get(), nullptr);
  ASSERT_EQ(meta_graph->nodes.front()->primitive->value.type, schema::PrimitiveType_Activation) << "wrong Op Type";
}

class TestTfliteParserTanh : public TestTfliteParser {
 public:
  TestTfliteParserTanh() = default;
  void SetUp() override { meta_graph = LoadAndConvert("./tanh.tflite", ""); }
};

TEST_F(TestTfliteParserTanh, OpType) {
  ASSERT_GT(meta_graph->nodes.size(), 0);
  ASSERT_NE(meta_graph->nodes.front()->primitive.get(), nullptr);
  ASSERT_EQ(meta_graph->nodes.front()->primitive->value.type, schema::PrimitiveType_Activation) << "wrong Op Type";
}

// logistic

class TestTfliteParserPrelu : public TestTfliteParser {
 public:
  TestTfliteParserPrelu() = default;
  void SetUp() override {
    meta_graph = LoadAndConvert("./prelu.tflite");
  }
};

TEST_F(TestTfliteParserPrelu, OpType) {
  ASSERT_NE(meta_graph, nullptr);
  ASSERT_GT(meta_graph->nodes.size(), 0);
  ASSERT_NE(meta_graph->nodes.front()->primitive.get(), nullptr);
  ASSERT_EQ(meta_graph->nodes.front()->primitive->value.type, schema::PrimitiveType_Prelu) << "wrong Op Type";
}

TEST_F(TestTfliteParserPrelu, AttrValue) {
  std::vector<float> slope(20, 0);
  ASSERT_NE(meta_graph, nullptr);
  ASSERT_GT(meta_graph->nodes.size(), 0);
  ASSERT_NE(meta_graph->nodes.front()->primitive.get(), nullptr);
  ASSERT_NE(meta_graph->nodes.front()->primitive->value.AsPrelu(), nullptr);
  ASSERT_EQ(meta_graph->nodes.front()->primitive->value.AsPrelu()->slope, slope);
}

class TestTfliteParserLeakyRelu : public TestTfliteParser {
 public:
  TestTfliteParserLeakyRelu() = default;
  void SetUp() override { meta_graph = LoadAndConvert("./leaky_relu.tflite", ""); }
};

TEST_F(TestTfliteParserLeakyRelu, OpType) {
  ASSERT_GT(meta_graph->nodes.size(), 0);
  ASSERT_NE(meta_graph->nodes.front()->primitive.get(), nullptr);
  ASSERT_EQ(meta_graph->nodes.front()->primitive->value.type, schema::PrimitiveType_LeakyReLU) << "wrong Op Type";
}

TEST_F(TestTfliteParserLeakyRelu, AttrValue) {
  ASSERT_GT(meta_graph->nodes.size(), 0);
  ASSERT_NE(meta_graph->nodes.front()->primitive.get(), nullptr);

  auto val = meta_graph->nodes.front()->primitive->value.AsLeakyReLU();
  ASSERT_NE(val, nullptr);
  ASSERT_EQ(val->negativeSlope, 0.20000000298023224);
}

}  // namespace mindspore
