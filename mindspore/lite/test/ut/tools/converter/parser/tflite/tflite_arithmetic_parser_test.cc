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
// doubleInputOp
class TestTfliteParserAdd1 : public TestTfliteParser {
 public:
  TestTfliteParserAdd1() = default;
  void SetUp() override { meta_graph = LoadAndConvert("./add1.tflite", ""); }
};

TEST_F(TestTfliteParserAdd1, OpType) {
  ASSERT_GT(meta_graph->nodes.size(), 0);
  ASSERT_NE(meta_graph->nodes.front()->primitive.get(), nullptr);
  ASSERT_EQ(meta_graph->nodes.front()->primitive->value.type, schema::PrimitiveType_Add) << "wrong Op Type";
}

TEST_F(TestTfliteParserAdd1, Tensor) {
  ASSERT_GT(meta_graph->allTensors.size(), 0);
  ASSERT_EQ(meta_graph->allTensors.at(0)->data.size(), 0);
  ASSERT_GT(meta_graph->allTensors.at(1)->data.size(), 0);
  ASSERT_EQ(meta_graph->allTensors.at(2)->data.size(), 0);
}

class TestTfliteParserAdd2 : public TestTfliteParser {
 public:
  TestTfliteParserAdd2() = default;
  void SetUp() override { meta_graph = LoadAndConvert("./add2.tflite", ""); }
};

TEST_F(TestTfliteParserAdd2, OpType) {
  ASSERT_GT(meta_graph->nodes.size(), 0);
  ASSERT_NE(meta_graph->nodes.front()->primitive.get(), nullptr);
  ASSERT_EQ(meta_graph->nodes.front()->primitive->value.type, schema::PrimitiveType_Add) << "wrong Op Type";
}

TEST_F(TestTfliteParserAdd2, Tensor) {
  ASSERT_GT(meta_graph->allTensors.size(), 0);
  ASSERT_EQ(meta_graph->allTensors.at(0)->data.size(), 0);
  ASSERT_GT(meta_graph->allTensors.at(1)->data.size(), 0);
  ASSERT_EQ(meta_graph->allTensors.at(2)->data.size(), 0);
}

class TestTfliteParserAdd3 : public TestTfliteParser {
 public:
  TestTfliteParserAdd3() = default;
  void SetUp() override { meta_graph = LoadAndConvert("./add3.tflite", ""); }
};

TEST_F(TestTfliteParserAdd3, OpType) {
  ASSERT_GT(meta_graph->nodes.size(), 0);
  ASSERT_NE(meta_graph->nodes.front()->primitive.get(), nullptr);
  ASSERT_EQ(meta_graph->nodes.front()->primitive->value.type, schema::PrimitiveType_Add) << "wrong Op Type";
}

TEST_F(TestTfliteParserAdd3, Tensor) {
  ASSERT_GT(meta_graph->allTensors.size(), 0);
  ASSERT_EQ(meta_graph->allTensors.at(0)->data.size(), 0);
  ASSERT_EQ(meta_graph->allTensors.at(1)->data.size(), 0);
  ASSERT_EQ(meta_graph->allTensors.at(2)->data.size(), 0);
}

class TestTfliteParserSub1 : public TestTfliteParser {
 public:
  TestTfliteParserSub1() = default;
  void SetUp() override { meta_graph = LoadAndConvert("./sub1.tflite", ""); }
};

TEST_F(TestTfliteParserSub1, OpType) {
  ASSERT_GT(meta_graph->nodes.size(), 0);
  ASSERT_NE(meta_graph->nodes.front()->primitive.get(), nullptr);
  ASSERT_EQ(meta_graph->nodes.front()->primitive->value.type, schema::PrimitiveType_Sub) << "wrong Op Type";
}

TEST_F(TestTfliteParserSub1, Tensor) {
  ASSERT_GT(meta_graph->allTensors.size(), 0);
  ASSERT_EQ(meta_graph->allTensors.at(0)->data.size(), 0);
  ASSERT_GT(meta_graph->allTensors.at(1)->data.size(), 0);
  ASSERT_EQ(meta_graph->allTensors.at(2)->data.size(), 0);
}

class TestTfliteParserSub2 : public TestTfliteParser {
 public:
  TestTfliteParserSub2() = default;
  void SetUp() override { meta_graph = LoadAndConvert("./sub2.tflite", ""); }
};

TEST_F(TestTfliteParserSub2, OpType) {
  ASSERT_GT(meta_graph->nodes.size(), 0);
  ASSERT_NE(meta_graph->nodes.front()->primitive.get(), nullptr);
  ASSERT_EQ(meta_graph->nodes.front()->primitive->value.type, schema::PrimitiveType_Sub) << "wrong Op Type";
}

TEST_F(TestTfliteParserSub2, Tensor) {
  ASSERT_GT(meta_graph->allTensors.size(), 0);
  ASSERT_EQ(meta_graph->allTensors.at(0)->data.size(), 0);
  ASSERT_GT(meta_graph->allTensors.at(1)->data.size(), 0);
  ASSERT_EQ(meta_graph->allTensors.at(2)->data.size(), 0);
}

class TestTfliteParserSub3 : public TestTfliteParser {
 public:
  TestTfliteParserSub3() = default;
  void SetUp() override { meta_graph = LoadAndConvert("./sub3.tflite", ""); }
};

TEST_F(TestTfliteParserSub3, OpType) {
  ASSERT_GT(meta_graph->nodes.size(), 0);
  ASSERT_NE(meta_graph->nodes.front()->primitive.get(), nullptr);
  ASSERT_EQ(meta_graph->nodes.front()->primitive->value.type, schema::PrimitiveType_Sub) << "wrong Op Type";
}

TEST_F(TestTfliteParserSub3, Tensor) {
  ASSERT_GT(meta_graph->allTensors.size(), 0);
  ASSERT_EQ(meta_graph->allTensors.at(0)->data.size(), 0);
  ASSERT_EQ(meta_graph->allTensors.at(1)->data.size(), 0);
  ASSERT_EQ(meta_graph->allTensors.at(2)->data.size(), 0);
}

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

class TestTfliteParserDiv1 : public TestTfliteParser {
 public:
  TestTfliteParserDiv1() = default;
  void SetUp() override { meta_graph = LoadAndConvert("./div1.tflite", ""); }
};

TEST_F(TestTfliteParserDiv1, OpType) {
  ASSERT_GT(meta_graph->nodes.size(), 0);
  ASSERT_NE(meta_graph->nodes.front()->primitive.get(), nullptr);
  ASSERT_EQ(meta_graph->nodes.front()->primitive->value.type, schema::PrimitiveType_Div) << "wrong Op Type";
}

TEST_F(TestTfliteParserDiv1, Tensor) {
  ASSERT_GT(meta_graph->allTensors.size(), 0);
  ASSERT_EQ(meta_graph->allTensors.at(0)->data.size(), 0);
  ASSERT_GT(meta_graph->allTensors.at(1)->data.size(), 0);
  ASSERT_EQ(meta_graph->allTensors.at(2)->data.size(), 0);
}

class TestTfliteParserDiv2 : public TestTfliteParser {
 public:
  TestTfliteParserDiv2() = default;
  void SetUp() override { meta_graph = LoadAndConvert("./div2.tflite", ""); }
};

TEST_F(TestTfliteParserDiv2, OpType) {
  ASSERT_GT(meta_graph->nodes.size(), 0);
  ASSERT_NE(meta_graph->nodes.front()->primitive.get(), nullptr);
  ASSERT_EQ(meta_graph->nodes.front()->primitive->value.type, schema::PrimitiveType_Div) << "wrong Op Type";
}

TEST_F(TestTfliteParserDiv2, Tensor) {
  ASSERT_GT(meta_graph->allTensors.size(), 0);
  ASSERT_EQ(meta_graph->allTensors.at(0)->data.size(), 0);
  ASSERT_GT(meta_graph->allTensors.at(1)->data.size(), 0);
  ASSERT_EQ(meta_graph->allTensors.at(2)->data.size(), 0);
}

class TestTfliteParserDiv3 : public TestTfliteParser {
 public:
  TestTfliteParserDiv3() = default;
  void SetUp() override { meta_graph = LoadAndConvert("./div3.tflite", ""); }
};

TEST_F(TestTfliteParserDiv3, OpType) {
  ASSERT_GT(meta_graph->nodes.size(), 0);
  ASSERT_NE(meta_graph->nodes.front()->primitive.get(), nullptr);
  ASSERT_EQ(meta_graph->nodes.front()->primitive->value.type, schema::PrimitiveType_Div) << "wrong Op Type";
}

TEST_F(TestTfliteParserDiv3, Tensor) {
  ASSERT_GT(meta_graph->allTensors.size(), 0);
  ASSERT_EQ(meta_graph->allTensors.at(0)->data.size(), 0);
  ASSERT_EQ(meta_graph->allTensors.at(1)->data.size(), 0);
  ASSERT_EQ(meta_graph->allTensors.at(2)->data.size(), 0);
}

class TestTfliteParserFloorDiv : public TestTfliteParser {
 public:
  TestTfliteParserFloorDiv() = default;
  void SetUp() override { meta_graph = LoadAndConvert("./floor_div.tflite", ""); }
};

TEST_F(TestTfliteParserFloorDiv, OpType) {
  ASSERT_GT(meta_graph->nodes.size(), 0);
  ASSERT_NE(meta_graph->nodes.front()->primitive.get(), nullptr);
  ASSERT_EQ(meta_graph->nodes.front()->primitive->value.type, schema::PrimitiveType_FloorDiv) << "wrong Op Type";
}

class TestTfliteParserFloorMod : public TestTfliteParser {
 public:
  TestTfliteParserFloorMod() = default;
  void SetUp() override { meta_graph = LoadAndConvert("./floor_mod.tflite", ""); }
};

TEST_F(TestTfliteParserFloorMod, OpType) {
  ASSERT_GT(meta_graph->nodes.size(), 0);
  ASSERT_NE(meta_graph->nodes.front()->primitive.get(), nullptr);
  ASSERT_EQ(meta_graph->nodes.front()->primitive->value.type, schema::PrimitiveType_FloorMod) << "wrong Op Type";
}

// realDiv

class TestTfliteParserSquaredDifference : public TestTfliteParser {
 public:
  TestTfliteParserSquaredDifference() = default;
  void SetUp() override {
    meta_graph = LoadAndConvert("./squared_difference.tflite");
  }
};

TEST_F(TestTfliteParserSquaredDifference, OpType) {
  ASSERT_NE(meta_graph, nullptr);
  ASSERT_GT(meta_graph->nodes.size(), 0);
  ASSERT_NE(meta_graph->nodes.front()->primitive.get(), nullptr);
  ASSERT_EQ(meta_graph->nodes.front()->primitive->value.type, schema::PrimitiveType_SquaredDifference)
            << "wrong Op Type";
}

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

class TestTfliteParserMaximum : public TestTfliteParser {
 public:
  TestTfliteParserMaximum() = default;
  void SetUp() override { meta_graph = LoadAndConvert("./maximum.tflite"); }
};

TEST_F(TestTfliteParserMaximum, OpType) {
  ASSERT_NE(meta_graph, nullptr);
  ASSERT_GT(meta_graph->nodes.size(), 0);
  ASSERT_NE(meta_graph->nodes.front()->primitive.get(), nullptr);
  ASSERT_EQ(meta_graph->nodes.front()->primitive->value.type, schema::PrimitiveType_Maximum) << "wrong Op Type";
}

class TestTfliteParserMinimum : public TestTfliteParser {
 public:
  TestTfliteParserMinimum() = default;
  void SetUp() override { meta_graph = LoadAndConvert("./minimum.tflite"); }
};

TEST_F(TestTfliteParserMinimum, OpType) {
  ASSERT_NE(meta_graph, nullptr);
  ASSERT_GT(meta_graph->nodes.size(), 0);
  ASSERT_NE(meta_graph->nodes.front()->primitive.get(), nullptr);
  ASSERT_EQ(meta_graph->nodes.front()->primitive->value.type, schema::PrimitiveType_Minimum) << "wrong Op Type";
}

// singleInputOp
class TestTfliteParserAbs : public TestTfliteParser {
 public:
  TestTfliteParserAbs() = default;
  void SetUp() override { meta_graph = LoadAndConvert("./abs.tflite", ""); }
};

TEST_F(TestTfliteParserAbs, OpType) {
  ASSERT_NE(meta_graph, nullptr);
  ASSERT_GT(meta_graph->nodes.size(), 0);
  ASSERT_NE(meta_graph->nodes.front()->primitive.get(), nullptr);
  ASSERT_EQ(meta_graph->nodes.front()->primitive->value.type, schema::PrimitiveType_Abs) << "wrong Op Type";
}

class TestTfliteParserExp : public TestTfliteParser {
 public:
  TestTfliteParserExp() = default;
  void SetUp() override { meta_graph = LoadAndConvert("./exp.tflite", ""); }
};

TEST_F(TestTfliteParserExp, OpType) {
  ASSERT_NE(meta_graph, nullptr);
  ASSERT_GT(meta_graph->nodes.size(), 0);
  ASSERT_NE(meta_graph->nodes.front()->primitive.get(), nullptr);
  ASSERT_EQ(meta_graph->nodes.front()->primitive->value.type, schema::PrimitiveType_Exp) << "wrong Op Type";
}

class TestTfliteParserSqrt : public TestTfliteParser {
 public:
  TestTfliteParserSqrt() = default;
  void SetUp() override { meta_graph = LoadAndConvert("./sqrt.tflite", ""); }
};

TEST_F(TestTfliteParserSqrt, OpType) {
  ASSERT_NE(meta_graph, nullptr);
  ASSERT_GT(meta_graph->nodes.size(), 0);
  ASSERT_NE(meta_graph->nodes.front()->primitive.get(), nullptr);
  ASSERT_EQ(meta_graph->nodes.front()->primitive->value.type, schema::PrimitiveType_Sqrt) << "wrong Op Type";
}

class TestTfliteParserRsqrt : public TestTfliteParser {
 public:
  TestTfliteParserRsqrt() = default;
  void SetUp() override { meta_graph = LoadAndConvert("./rsqrt.tflite", ""); }
};

TEST_F(TestTfliteParserRsqrt, OpType) {
  ASSERT_NE(meta_graph, nullptr);
  ASSERT_GT(meta_graph->nodes.size(), 0);
  ASSERT_NE(meta_graph->nodes.front()->primitive.get(), nullptr);
  ASSERT_EQ(meta_graph->nodes.front()->primitive->value.type, schema::PrimitiveType_Rsqrt) << "wrong Op Type";
}

class TestTfliteParserSquare : public TestTfliteParser {
 public:
  TestTfliteParserSquare() = default;
  void SetUp() override { meta_graph = LoadAndConvert("./square.tflite", ""); }
};

TEST_F(TestTfliteParserSquare, OpType) {
  ASSERT_NE(meta_graph, nullptr);
  ASSERT_GT(meta_graph->nodes.size(), 0);
  ASSERT_NE(meta_graph->nodes.front()->primitive.get(), nullptr);
  ASSERT_EQ(meta_graph->nodes.front()->primitive->value.type, schema::PrimitiveType_Square) << "wrong Op Type";
}

class TestTfliteParserSin : public TestTfliteParser {
 public:
  TestTfliteParserSin() = default;
  void SetUp() override { meta_graph = LoadAndConvert("./sin.tflite", ""); }
};

TEST_F(TestTfliteParserSin, OpType) {
  ASSERT_NE(meta_graph, nullptr);
  ASSERT_GT(meta_graph->nodes.size(), 0);
  ASSERT_NE(meta_graph->nodes.front()->primitive.get(), nullptr);
  ASSERT_EQ(meta_graph->nodes.front()->primitive->value.type, schema::PrimitiveType_Sin) << "wrong Op Type";
}

class TestTfliteParserCos : public TestTfliteParser {
 public:
  TestTfliteParserCos() = default;
  void SetUp() override { meta_graph = LoadAndConvert("./cos.tflite", ""); }
};

TEST_F(TestTfliteParserCos, OpType) {
  ASSERT_NE(meta_graph, nullptr);
  ASSERT_GT(meta_graph->nodes.size(), 0);
  ASSERT_NE(meta_graph->nodes.front()->primitive.get(), nullptr);
  ASSERT_EQ(meta_graph->nodes.front()->primitive->value.type, schema::PrimitiveType_Cos) << "wrong Op Type";
}

class TestTfliteParserLog : public TestTfliteParser {
 public:
  TestTfliteParserLog() = default;
  void SetUp() override { meta_graph = LoadAndConvert("./log.tflite", ""); }
};

TEST_F(TestTfliteParserLog, OpType) {
  ASSERT_NE(meta_graph, nullptr);
  ASSERT_GT(meta_graph->nodes.size(), 0);
  ASSERT_NE(meta_graph->nodes.front()->primitive.get(), nullptr);
  ASSERT_EQ(meta_graph->nodes.front()->primitive->value.type, schema::PrimitiveType_Log) << "wrong Op Type";
}

class TestTfliteParserRound : public TestTfliteParser {
 public:
  TestTfliteParserRound() = default;
  void SetUp() override {
    meta_graph = LoadAndConvert("./round.tflite");
  }
};

TEST_F(TestTfliteParserRound, OpType) {
  ASSERT_NE(meta_graph, nullptr);
  ASSERT_GT(meta_graph->nodes.size(), 0);
  ASSERT_NE(meta_graph->nodes.front()->primitive.get(), nullptr);
  ASSERT_EQ(meta_graph->nodes.front()->primitive->value.type, schema::PrimitiveType_Round) << "wrong Op Type";
}

class TestTfliteParserCeil : public TestTfliteParser {
 public:
  TestTfliteParserCeil() = default;
  void SetUp() override { meta_graph = LoadAndConvert("./ceil.tflite", ""); }
};

TEST_F(TestTfliteParserCeil, OpType) {
  ASSERT_GT(meta_graph->nodes.size(), 0);
  ASSERT_NE(meta_graph->nodes.front()->primitive.get(), nullptr);
  ASSERT_EQ(meta_graph->nodes.front()->primitive->value.type, schema::PrimitiveType_Ceil) << "wrong Op Type";
}

class TestTfliteParserFloor : public TestTfliteParser {
 public:
  TestTfliteParserFloor() = default;
  void SetUp() override { meta_graph = LoadAndConvert("./floor.tflite", ""); }
};

TEST_F(TestTfliteParserFloor, OpType) {
  ASSERT_GT(meta_graph->nodes.size(), 0);
  ASSERT_NE(meta_graph->nodes.front()->primitive.get(), nullptr);
  ASSERT_EQ(meta_graph->nodes.front()->primitive->value.type, schema::PrimitiveType_Floor) << "wrong Op Type";
}

// comareOp
class TestTfliteParserEqual : public TestTfliteParser {
 public:
  TestTfliteParserEqual() = default;
  void SetUp() override {
    meta_graph = LoadAndConvert("./equal.tflite");
  }
};

TEST_F(TestTfliteParserEqual, OpType) {
  ASSERT_NE(meta_graph, nullptr);
  ASSERT_GT(meta_graph->nodes.size(), 0);
  ASSERT_NE(meta_graph->nodes.front()->primitive.get(), nullptr);
  ASSERT_EQ(meta_graph->nodes.front()->primitive->value.type, schema::PrimitiveType_Equal) << "wrong Op Type";
}

class TestTfliteParserNotEqual : public TestTfliteParser {
 public:
  TestTfliteParserNotEqual() = default;
  void SetUp() override {
    meta_graph = LoadAndConvert("./not_equal.tflite");
  }
};

TEST_F(TestTfliteParserNotEqual, OpType) {
  ASSERT_NE(meta_graph, nullptr);
  ASSERT_GT(meta_graph->nodes.size(), 0);
  ASSERT_NE(meta_graph->nodes.front()->primitive.get(), nullptr);
  ASSERT_EQ(meta_graph->nodes.front()->primitive->value.type, schema::PrimitiveType_NotEqual) << "wrong Op Type";
}

class TestTfliteParserGreater : public TestTfliteParser {
 public:
  TestTfliteParserGreater() = default;
  void SetUp() override {
    meta_graph = LoadAndConvert("./greater.tflite");
  }
};

TEST_F(TestTfliteParserGreater, OpType) {
  ASSERT_NE(meta_graph, nullptr);
  ASSERT_GT(meta_graph->nodes.size(), 0);
  ASSERT_NE(meta_graph->nodes.front()->primitive.get(), nullptr);
  ASSERT_EQ(meta_graph->nodes.front()->primitive->value.type, schema::PrimitiveType_Greater) << "wrong Op Type";
}

class TestTfliteParserGreaterEqual : public TestTfliteParser {
 public:
  TestTfliteParserGreaterEqual() = default;
  void SetUp() override {
    meta_graph = LoadAndConvert("./greater_equal.tflite");
  }
};

TEST_F(TestTfliteParserGreaterEqual, OpType) {
  ASSERT_NE(meta_graph, nullptr);
  ASSERT_GT(meta_graph->nodes.size(), 0);
  ASSERT_NE(meta_graph->nodes.front()->primitive.get(), nullptr);
  ASSERT_EQ(meta_graph->nodes.front()->primitive->value.type, schema::PrimitiveType_GreaterEqual) << "wrong Op Type";
}

class TestTfliteParserLess : public TestTfliteParser {
 public:
  TestTfliteParserLess() = default;
  void SetUp() override {
    meta_graph = LoadAndConvert("./less.tflite");
  }
};

TEST_F(TestTfliteParserLess, OpType) {
  ASSERT_NE(meta_graph, nullptr);
  ASSERT_GT(meta_graph->nodes.size(), 0);
  ASSERT_NE(meta_graph->nodes.front()->primitive.get(), nullptr);
  ASSERT_EQ(meta_graph->nodes.front()->primitive->value.type, schema::PrimitiveType_Less) << "wrong Op Type";
}

class TestTfliteParserLessEqual : public TestTfliteParser {
 public:
  TestTfliteParserLessEqual() = default;
  void SetUp() override {
    meta_graph = LoadAndConvert("./less_equal.tflite");
  }
};

TEST_F(TestTfliteParserLessEqual, OpType) {
  ASSERT_NE(meta_graph, nullptr);
  ASSERT_GT(meta_graph->nodes.size(), 0);
  ASSERT_NE(meta_graph->nodes.front()->primitive.get(), nullptr);
  ASSERT_EQ(meta_graph->nodes.front()->primitive->value.type, schema::PrimitiveType_LessEqual) << "wrong Op Type";
}

}  // namespace mindspore
