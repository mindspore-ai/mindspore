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

TEST_F(TestTfliteParserRelu, AttrValue) {
  ASSERT_NE(meta_graph->nodes.front()->primitive->value.AsActivation(), nullptr);
  auto val = meta_graph->nodes.front()->primitive->value.AsActivation();
  ASSERT_EQ(val->activation_type, schema::ActivationType_RELU);
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

TEST_F(TestTfliteParserRelu6, AttrValue) {
  ASSERT_NE(meta_graph->nodes.front()->primitive->value.AsActivation(), nullptr);
  auto val = meta_graph->nodes.front()->primitive->value.AsActivation();
  ASSERT_EQ(val->activation_type, schema::ActivationType_RELU6);
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

TEST_F(TestTfliteParserTanh, AttrValue) {
  ASSERT_NE(meta_graph->nodes.front()->primitive->value.AsActivation(), nullptr);
  auto val = meta_graph->nodes.front()->primitive->value.AsActivation();
  ASSERT_EQ(val->activation_type, schema::ActivationType_TANH);
}

class TestTfliteParserLogistic : public TestTfliteParser {
 public:
  TestTfliteParserLogistic() = default;
  void SetUp() override { meta_graph = LoadAndConvert("./logistic.tflite", ""); }
};

TEST_F(TestTfliteParserLogistic, OpType) {
  ASSERT_GT(meta_graph->nodes.size(), 0);
  ASSERT_NE(meta_graph->nodes.front()->primitive.get(), nullptr);
  ASSERT_EQ(meta_graph->nodes.front()->primitive->value.type, schema::PrimitiveType_Activation) << "wrong Op Type";
}
TEST_F(TestTfliteParserLogistic, AttrValue) {
  ASSERT_NE(meta_graph->nodes.front()->primitive->value.AsActivation(), nullptr);
  auto val = meta_graph->nodes.front()->primitive->value.AsActivation();
  ASSERT_EQ(val->activation_type, schema::ActivationType_SIGMOID);
}

class TestTfliteParserHardSwish : public TestTfliteParser {
 public:
  TestTfliteParserHardSwish() = default;
  void SetUp() override { meta_graph = LoadAndConvert("./hardswish.tflite", ""); }
};

TEST_F(TestTfliteParserHardSwish, OpType) {
  ASSERT_GT(meta_graph->nodes.size(), 0);
  ASSERT_NE(meta_graph->nodes.front()->primitive.get(), nullptr);
  ASSERT_EQ(meta_graph->nodes.front()->primitive->value.type, schema::PrimitiveType_Activation) << "wrong Op Type";
}
TEST_F(TestTfliteParserHardSwish, AttrValue) {
  ASSERT_NE(meta_graph->nodes.front()->primitive->value.AsActivation(), nullptr);
  auto val = meta_graph->nodes.front()->primitive->value.AsActivation();
  ASSERT_EQ(val->activation_type, schema::ActivationType_SIGMOID);
}

class TestTfliteParserPrelu : public TestTfliteParser {
 public:
  TestTfliteParserPrelu() = default;
  void SetUp() override { meta_graph = LoadAndConvert("./prelu.tflite"); }
};

TEST_F(TestTfliteParserPrelu, OpType) {
  ASSERT_NE(meta_graph, nullptr);
  ASSERT_GT(meta_graph->nodes.size(), 0);
  ASSERT_NE(meta_graph->nodes.front()->primitive.get(), nullptr);
}

class TestTfliteParserLeakyRelu : public TestTfliteParser {
 public:
  TestTfliteParserLeakyRelu() = default;
  void SetUp() override { meta_graph = LoadAndConvert("./leaky_relu.tflite", ""); }
};

TEST_F(TestTfliteParserLeakyRelu, OpType) {
  ASSERT_GT(meta_graph->nodes.size(), 0);
  ASSERT_NE(meta_graph->nodes.front()->primitive.get(), nullptr);
  ASSERT_EQ(meta_graph->nodes.front()->primitive->value.type, schema::PrimitiveType_LeakyRelu) << "wrong Op Type";
}

TEST_F(TestTfliteParserLeakyRelu, AttrValue) {
  ASSERT_NE(meta_graph->nodes.front()->primitive->value.AsLeakyRelu(), nullptr);
  auto val = meta_graph->nodes.front()->primitive->value;
  ASSERT_EQ(val.AsLeakyRelu()->negative_slope, 0.20000000298023224);
  ASSERT_EQ(val.type, schema::PrimitiveType_LeakyRelu);
}

}  // namespace mindspore
