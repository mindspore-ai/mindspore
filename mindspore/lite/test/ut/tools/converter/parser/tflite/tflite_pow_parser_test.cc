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
class TestTfliteParserPow : public TestTfliteParser {
 public:
  TestTfliteParserPow() = default;
  void SetUp() override { meta_graph = LoadAndConvert("./pow.tflite", ""); }
};

TEST_F(TestTfliteParserPow, OpType) {
  ASSERT_GT(meta_graph->nodes.size(), 0);
  ASSERT_NE(meta_graph->nodes.front()->primitive.get(), nullptr);
  ASSERT_EQ(meta_graph->nodes.front()->primitive->value.type, schema::PrimitiveType_Power) << "wrong Op Type";
}

TEST_F(TestTfliteParserPow, AttrValue) {
  ASSERT_GT(meta_graph->nodes.size(), 0);
  ASSERT_NE(meta_graph->nodes.front()->primitive.get(), nullptr);

  auto val = meta_graph->nodes.front()->primitive->value.AsPower();

  ASSERT_EQ(val->scale, 1.0);
  ASSERT_EQ(val->shift, 0.0);
  ASSERT_EQ(val->power, 0.0);
}

}  // namespace mindspore
