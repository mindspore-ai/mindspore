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
class TestTfliteParserDeConv : public TestTfliteParser {
 public:
  TestTfliteParserDeConv() = default;
  void SetUp() override { meta_graph = LoadAndConvert("./deconv.tflite", ""); }
};

TEST_F(TestTfliteParserDeConv, OpType) {
  ASSERT_NE(meta_graph, nullptr);
  ASSERT_GT(meta_graph->nodes.size(), 0);
  ASSERT_NE(meta_graph->nodes.front()->primitive.get(), nullptr);
  ASSERT_EQ(meta_graph->nodes.front()->primitive->value.type, schema::PrimitiveType_DeConv2D) << "wrong Op Type";
}

TEST_F(TestTfliteParserDeConv, AttrValue) {
  ASSERT_NE(meta_graph->nodes.front()->primitive->value.AsDeConv2D(), nullptr);
  auto val = meta_graph->nodes.front()->primitive->value.AsDeConv2D();
  ASSERT_EQ(val->format, schema::Format_NHWC);
  ASSERT_EQ(val->group, 1);
  ASSERT_EQ(val->activationType, schema::ActivationType_NO_ACTIVATION);

  ASSERT_EQ(val->channelIn, 1);
  ASSERT_EQ(val->channelOut, 4);
  ASSERT_EQ(val->kernelH, 3);
  ASSERT_EQ(val->kernelW, 3);
  ASSERT_EQ(val->strideH, 1);
  ASSERT_EQ(val->strideW, 1);
  ASSERT_EQ(val->dilateH, 1);
  ASSERT_EQ(val->dilateW, 1);
  ASSERT_EQ(val->padMode, schema::PadMode_SAME_UPPER);
  ASSERT_EQ(val->padUp, 1);
  ASSERT_EQ(val->padDown, 1);
  ASSERT_EQ(val->padLeft, 1);
  ASSERT_EQ(val->padRight, 1);
}

}  // namespace mindspore
