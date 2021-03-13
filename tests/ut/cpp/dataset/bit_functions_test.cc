/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/include/constants.h"
#include "common/common.h"

using namespace mindspore::dataset;

class MindDataTestBits : public UT::Common {
 public:
    MindDataTestBits() {}
};


TEST_F(MindDataTestBits, Basics) {
  uint32_t x = 0;  // 00000
  BitSet(&x, 16);  // 10000

  ASSERT_TRUE(BitTest(x, 16));  // 10000
  ASSERT_FALSE(BitTest(x, 1));  // 00001
  ASSERT_FALSE(BitTest(x, 17));  // 10001 is failing


  BitSet(&x, 1);  // 10001
  ASSERT_TRUE(BitTest(x, 16));  // 10000
  ASSERT_TRUE(BitTest(x, 1));  // 00001
  ASSERT_TRUE(BitTest(x, 17));  // 10001 is failing

  BitClear(&x, 16);  // 00001
  ASSERT_FALSE(BitTest(x, 16));  // 10000
  ASSERT_TRUE(BitTest(x, 1));  // 00001
//  ASSERT_FALSE(BitTest(x, 17));  // 10001 is failing

  BitSet(&x, 31);  //  11111
  for (uint32_t i = 1; i < 32; i++) {
    ASSERT_TRUE(BitTest(x, i));
  }
  BitClear(&x, 31);  //  00000
  for (uint32_t i = 1; i < 32; i++) {
    ASSERT_FALSE(BitTest(x, i));
  }
}
