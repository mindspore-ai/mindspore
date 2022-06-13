/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/util/treap.h"
#include "common/common.h"
#include "gtest/gtest.h"

using namespace mindspore::dataset;

class MindDataTestTreap : public UT::Common {
 public:
  MindDataTestTreap() {}
};

/// Feature: Treap
/// Description: Test all functions of Treap
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestTreap, TestALLFunction) {
  Treap<uint64_t, uint64_t> tree;
  srand(time(NULL));
  for (uint64_t i = 0; i < 1000; i++) {
    uint64_t sz = rand() % 500;
    tree.Insert(i, sz);
  }

  EXPECT_EQ(tree.size(), 1000);

  int n = 0;
  uint64_t key = 0;
  for (auto it : tree) {
    if (n > 0) {
      EXPECT_GT(it.key, key);
    }
    key = it.key;
    n++;
  }

  EXPECT_EQ(n, 1000);

  uint64_t prev = 0;
  n = 0;
  while (!tree.empty()) {
    auto p = tree.Top();
    EXPECT_TRUE(p.second);
    uint64_t v = p.first.priority;
    if (n > 0) {
      EXPECT_GE(prev, v);
    }
    prev = v;
    n++;
    tree.Pop();
  }
}
