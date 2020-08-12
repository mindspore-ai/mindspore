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
class TestTfliteParserMaxPooling : public TestTfliteParser {
 public:
  TestTfliteParserMaxPooling() = default;
  void SetUp() override {
    meta_graph = LoadAndConvert("./max_pooling.tflite");
  }
};

TEST_F(TestTfliteParserMaxPooling, OpType) {
  ASSERT_NE(meta_graph, nullptr);
  ASSERT_GT(meta_graph->nodes.size(), 0);
  ASSERT_NE(meta_graph->nodes.front()->primitive.get(), nullptr);
  ASSERT_EQ(meta_graph->nodes.front()->primitive->value.type, schema::PrimitiveType_Pooling) << "wrong Op Type";
}

TEST_F(TestTfliteParserMaxPooling, AttrValue) {
  ASSERT_NE(meta_graph, nullptr);
  ASSERT_GT(meta_graph->nodes.size(), 0);
  ASSERT_NE(meta_graph->nodes.front()->primitive.get(), nullptr);

  auto val = meta_graph->nodes.front()->primitive->value.AsPooling();
  ASSERT_NE(val, nullptr);
  ASSERT_EQ(val->format, schema::Format_NHWC);
  ASSERT_EQ(val->poolingMode, schema::PoolMode_MAX_POOLING);
  ASSERT_EQ(val->global, false);
  ASSERT_EQ(val->windowW, 2);
  ASSERT_EQ(val->windowH, 2);
  ASSERT_EQ(val->strideW, 1);
  ASSERT_EQ(val->strideH, 1);
  ASSERT_EQ(val->padMode, schema::PadMode_VALID);
  ASSERT_EQ(val->padUp, 0);
  ASSERT_EQ(val->padDown, 0);
  ASSERT_EQ(val->padLeft, 0);
  ASSERT_EQ(val->padRight, 0);
  ASSERT_EQ(val->roundMode, schema::RoundMode_FLOOR);
}

class TestTfliteParserAvgPooling : public TestTfliteParser {
 public:
  TestTfliteParserAvgPooling() = default;
  void SetUp() override {
    meta_graph = LoadAndConvert("./avg_pooling.tflite");
  }
};

TEST_F(TestTfliteParserAvgPooling, OpType) {
  ASSERT_NE(meta_graph, nullptr);
  ASSERT_GT(meta_graph->nodes.size(), 0);
  ASSERT_NE(meta_graph->nodes.front()->primitive.get(), nullptr);
  ASSERT_EQ(meta_graph->nodes.front()->primitive->value.type, schema::PrimitiveType_Pooling) << "wrong Op Type";
}

TEST_F(TestTfliteParserAvgPooling, AttrValue) {
  ASSERT_NE(meta_graph, nullptr);
  ASSERT_GT(meta_graph->nodes.size(), 0);
  ASSERT_NE(meta_graph->nodes.front()->primitive.get(), nullptr);

  auto val = meta_graph->nodes.front()->primitive->value.AsPooling();
  ASSERT_NE(val, nullptr);
  ASSERT_EQ(val->format, schema::Format_NHWC);
  ASSERT_EQ(val->poolingMode, schema::PoolMode_MEAN_POOLING);
  ASSERT_EQ(val->global, false);
  ASSERT_EQ(val->windowW, 2);
  ASSERT_EQ(val->windowH, 2);
  ASSERT_EQ(val->strideW, 1);
  ASSERT_EQ(val->strideH, 1);
  ASSERT_EQ(val->padMode, schema::PadMode_SAME);
  ASSERT_EQ(val->padUp, 0);
  ASSERT_EQ(val->padDown, 1);
  ASSERT_EQ(val->padLeft, 0);
  ASSERT_EQ(val->padRight, 1);
  ASSERT_EQ(val->roundMode, schema::RoundMode_FLOOR);
}
}  // namespace mindspore
