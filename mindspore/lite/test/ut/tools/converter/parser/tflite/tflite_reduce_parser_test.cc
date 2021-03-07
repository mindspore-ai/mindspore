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
class TestTfliteParserReduceMax : public TestTfliteParser {
 public:
  TestTfliteParserReduceMax() = default;
  void SetUp() override { meta_graph = LoadAndConvert("./reduce_max.tflite"); }
};

TEST_F(TestTfliteParserReduceMax, OpType) {
  ASSERT_NE(meta_graph, nullptr);
  ASSERT_GT(meta_graph->nodes.size(), 0);
  ASSERT_NE(meta_graph->nodes.front()->primitive.get(), nullptr);
  ASSERT_EQ(meta_graph->nodes.front()->primitive->value.type, schema::PrimitiveType_ReduceFusion) << "wrong Op Type";
}

TEST_F(TestTfliteParserReduceMax, AttrValue) {
  ASSERT_NE(meta_graph->nodes.front()->primitive->value.AsReduceFusion(), nullptr);
  auto val = meta_graph->nodes.front()->primitive->value.AsReduceFusion();
  ASSERT_EQ(val->mode, schema::ReduceMode_ReduceMax);
  ASSERT_EQ(val->keep_dims, false);
}

class TestTfliteParserReduceMin : public TestTfliteParser {
 public:
  TestTfliteParserReduceMin() = default;
  void SetUp() override { meta_graph = LoadAndConvert("./reduce_min.tflite"); }
};

TEST_F(TestTfliteParserReduceMin, OpType) {
  ASSERT_NE(meta_graph, nullptr);
  ASSERT_GT(meta_graph->nodes.size(), 0);
  ASSERT_NE(meta_graph->nodes.front()->primitive.get(), nullptr);
  ASSERT_EQ(meta_graph->nodes.front()->primitive->value.type, schema::PrimitiveType_ReduceFusion) << "wrong Op Type";
}

TEST_F(TestTfliteParserReduceMin, AttrValue) {
  ASSERT_NE(meta_graph->nodes.front()->primitive->value.AsReduceFusion(), nullptr);
  auto val = meta_graph->nodes.front()->primitive->value.AsReduceFusion();
  ASSERT_EQ(val->mode, schema::ReduceMode_ReduceMin);
  ASSERT_EQ(val->keep_dims, false);
}

class TestTfliteParserReduceProd : public TestTfliteParser {
 public:
  TestTfliteParserReduceProd() = default;
  void SetUp() override { meta_graph = LoadAndConvert("./reduce_prod.tflite"); }
};

TEST_F(TestTfliteParserReduceProd, OpType) {
  ASSERT_NE(meta_graph, nullptr);
  ASSERT_GT(meta_graph->nodes.size(), 0);
  ASSERT_NE(meta_graph->nodes.front()->primitive.get(), nullptr);
  ASSERT_EQ(meta_graph->nodes.front()->primitive->value.type, schema::PrimitiveType_ReduceFusion) << "wrong Op Type";
}

TEST_F(TestTfliteParserReduceProd, AttrValue) {
  ASSERT_NE(meta_graph->nodes.front()->primitive->value.AsReduceFusion(), nullptr);
  auto val = meta_graph->nodes.front()->primitive->value.AsReduceFusion();
  ASSERT_EQ(val->mode, schema::ReduceMode_ReduceProd);
  ASSERT_EQ(val->keep_dims, false);
}

class TestTfliteParserSum : public TestTfliteParser {
 public:
  TestTfliteParserSum() = default;

  void SetUp() override { meta_graph = LoadAndConvert("./sum.tflite"); }
};

TEST_F(TestTfliteParserSum, OpType) {
  ASSERT_NE(meta_graph, nullptr);
  ASSERT_GT(meta_graph->nodes.size(), 0);
  ASSERT_NE(meta_graph->nodes.front()->primitive.get(), nullptr);
  ASSERT_EQ(meta_graph->nodes.front()->primitive->value.type, schema::PrimitiveType_ReduceFusion) << "wrong Op Type";
}

TEST_F(TestTfliteParserSum, AttrValue) {
  ASSERT_NE(meta_graph->nodes.front()->primitive->value.AsReduceFusion(), nullptr);
  auto val = meta_graph->nodes.front()->primitive->value.AsReduceFusion();
  ASSERT_EQ(val->mode, schema::ReduceMode_ReduceSum);
  ASSERT_EQ(val->keep_dims, false);
}

class TestTfliteParserMean : public TestTfliteParser {
 public:
  TestTfliteParserMean() = default;

  void SetUp() override { meta_graph = LoadAndConvert("./mean.tflite"); }
};

TEST_F(TestTfliteParserMean, OpType) {
  ASSERT_NE(meta_graph, nullptr);
  ASSERT_GT(meta_graph->nodes.size(), 0);
  ASSERT_NE(meta_graph->nodes.front()->primitive.get(), nullptr);
  ASSERT_EQ(meta_graph->nodes.front()->primitive->value.type, schema::PrimitiveType_ReduceFusion) << "wrong Op Type";
}

TEST_F(TestTfliteParserMean, AttrValue) {
  ASSERT_NE(meta_graph->nodes.front()->primitive->value.AsReduceFusion(), nullptr);
  auto val = meta_graph->nodes.front()->primitive->value.AsReduceFusion();
  ASSERT_EQ(val->mode, schema::ReduceMode_ReduceMean);
  ASSERT_EQ(val->keep_dims, true);
}

// reduceAny

}  // namespace mindspore
