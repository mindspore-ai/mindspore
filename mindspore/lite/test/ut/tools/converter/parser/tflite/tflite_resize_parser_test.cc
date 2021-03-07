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

class TestTfliteParserResizeNN : public TestTfliteParser {
 public:
  TestTfliteParserResizeNN() = default;
  void SetUp() override { meta_graph = LoadAndConvert("./resize_nearest_neighbor.tflite", ""); }
};

TEST_F(TestTfliteParserResizeNN, OpType) {
  ASSERT_NE(meta_graph, nullptr);
  ASSERT_GT(meta_graph->nodes.size(), 0);
  ASSERT_NE(meta_graph->nodes.front()->primitive.get(), nullptr);
  ASSERT_EQ(meta_graph->nodes.front()->primitive->value.type, schema::PrimitiveType_Resize) << "wrong Op Type";
}

TEST_F(TestTfliteParserResizeNN, AttrValue) {
  ASSERT_NE(meta_graph->nodes.front()->primitive->value.AsResize(), nullptr);
  auto val = meta_graph->nodes.front()->primitive->value.AsResize();
  ASSERT_EQ(val->new_height, 3);
  ASSERT_EQ(val->new_width, 100);
  ASSERT_EQ(val->format, schema::Format_NHWC);
  ASSERT_EQ(val->preserve_aspect_ratio, false);
  ASSERT_EQ(val->method, schema::ResizeMethod_NEAREST);
}

class TestTfliteParserResizeBilinear : public TestTfliteParser {
 public:
  TestTfliteParserResizeBilinear() = default;
  void SetUp() override { meta_graph = LoadAndConvert("./resize_bilinear.tflite", ""); }
};

TEST_F(TestTfliteParserResizeBilinear, OpType) {
  ASSERT_NE(meta_graph, nullptr);
  ASSERT_GT(meta_graph->nodes.size(), 0);
  ASSERT_NE(meta_graph->nodes.front()->primitive.get(), nullptr);
  ASSERT_EQ(meta_graph->nodes.front()->primitive->value.type, schema::PrimitiveType_Resize) << "wrong Op Type";
}

TEST_F(TestTfliteParserResizeBilinear, AttrValue) {
  ASSERT_NE(meta_graph->nodes.front()->primitive->value.AsResize(), nullptr);
  auto val = meta_graph->nodes.front()->primitive->value.AsResize();
  ASSERT_EQ(val->new_height, 75);
  ASSERT_EQ(val->new_width, 4);
  ASSERT_EQ(val->format, schema::Format_NHWC);
  ASSERT_EQ(val->preserve_aspect_ratio, false);
  ASSERT_EQ(val->method, schema::ResizeMethod_LINEAR);
}

}  // namespace mindspore
