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
#include <iostream>
#include <memory>

#include "common/common_test.h"

#include "abstract/dshape.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace abstract {
class TestDShape : public UT::Common {
 public:
  Shape shp_1;
  Shape shp_2;
  Shape shp_3;
  Shape shp_4;

  NoShape shp_noshp_1;
  NoShape shp_noshp_2;

  TupleShape shp_tuple_1;
  TupleShape shp_tuple_2;
  TupleShape shp_tuple_3;
  TupleShape shp_tuple_4;
  TestDShape()
      : shp_1({1, 1}),
        shp_2({1, 1}),
        shp_3({1, 2}),
        shp_4({1}),

        shp_noshp_1(),
        shp_noshp_2(),

        shp_tuple_1({NoShape().Clone(), Shape({1, 1}).Clone()}),
        shp_tuple_2({NoShape().Clone(), Shape({1, 1, 1}).Clone()}),
        shp_tuple_3({NoShape().Clone(), Shape({1, 2, 1}).Clone()}),
        shp_tuple_4({NoShape().Clone()}) {}
};

TEST_F(TestDShape, EqualTest) {
  ASSERT_TRUE(shp_1 == shp_2);
  ASSERT_FALSE(shp_1 == shp_3);
  ASSERT_FALSE(shp_1 == shp_noshp_1);

  ASSERT_TRUE(shp_noshp_1 == shp_noshp_2);

  ASSERT_FALSE(shp_tuple_1 == shp_1);
  ASSERT_FALSE(shp_tuple_1 == shp_tuple_2);
  ASSERT_FALSE(shp_tuple_1 == shp_tuple_4);
}
TEST_F(TestDShape, ToString) {
  ASSERT_EQ(shp_3.ToString(), "(1, 2)");
  ASSERT_EQ(shp_noshp_1.ToString(), "NoShape");
  ASSERT_EQ(shp_tuple_2.ToString(), "TupleShape(NoShape, (1, 1, 1))");
}

TEST_F(TestDShape, Clone) {
  ASSERT_EQ(*shp_3.Clone(), shp_3);
  ASSERT_EQ(*shp_noshp_1.Clone(), shp_noshp_1);
  ASSERT_EQ(*shp_tuple_2.Clone(), shp_tuple_2);
}
}  // namespace abstract
}  // namespace mindspore
