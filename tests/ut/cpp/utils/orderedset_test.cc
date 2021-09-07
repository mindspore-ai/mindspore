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
#include <sstream>
#include <memory>
#include <algorithm>

#include "utils/ordered_set.h"
#include "utils/ordered_map.h"
#include "common/common_test.h"

using std::cout;
using std::endl;
using std::string;

namespace mindspore {

class TestOrderedSet : public UT::Common {
 public:
  TestOrderedSet() {
    std::shared_ptr<int> e;
    for (int i = 1; i <= 10; i++) {
      e = std::make_shared<int>(i);
      osa.add(e);
    }
    for (int i = 11; i <= 20; i++) {
      e = std::make_shared<int>(i);
      osa.add(e);
      osb.add(e);
    }
    for (int i = 21; i <= 30; i++) {
      e = std::make_shared<int>(i);
      osb.add(e);
      osc.add(e);
    }
  }

 public:
  OrderedSet<std::shared_ptr<int>> osa;
  OrderedSet<std::shared_ptr<int>> osb;
  OrderedSet<std::shared_ptr<int>> osc;
};

TEST_F(TestOrderedSet, test_constructor) {
  OrderedSet<std::shared_ptr<int>> osa_copy = osa;
  ASSERT_EQ(osa_copy.size(), osa.size());

  std::shared_ptr<int> e = std::make_shared<int>(1);
  OrderedSet<std::shared_ptr<int>> se;
  se.add(std::make_shared<int>(10));
  se.add(std::make_shared<int>(20));
  OrderedSet<std::shared_ptr<int>> order_se(se);
  ASSERT_EQ(order_se.size(), 2);
}

TEST_F(TestOrderedSet, test_add_remove_clear) {
  OrderedSet<std::shared_ptr<int>> res;
  res.add(std::make_shared<int>(1));
  std::shared_ptr<int> e = std::make_shared<int>(2);
  std::shared_ptr<int> e2 = std::make_shared<int>(10);
  res.add(e);
  ASSERT_EQ(res.size(), 2);
  ASSERT_EQ(res.count(e), 1);
  auto elem = res.back();
  ASSERT_EQ(elem, e);
  res.erase(e);
  ASSERT_EQ(res.size(), 1);
  res.clear();
  ASSERT_EQ(res.size(), 0);
}

TEST_F(TestOrderedSet, test_add_remove_first) {
  OrderedSet<int> a;
  a.add(1);
  a.add(2);
  a.add(3);
  a.erase(1);
  auto first = a.pop();
  // 1 removed, 2 3 followed, 2 should be the popped one, remaining size = 1
  ASSERT_EQ(first, 2);
  ASSERT_EQ(a.size(), 1);
}

TEST_F(TestOrderedSet, test_compare) {
  OrderedSet<std::shared_ptr<int>> c1;
  OrderedSet<std::shared_ptr<int>> c2;
  std::shared_ptr<int> e1 = std::make_shared<int>(10);
  std::shared_ptr<int> e2 = std::make_shared<int>(20);
  c1.add(e1);
  c1.add(e2);
  c2.add(e1);
  c2.add(e2);
  ASSERT_EQ(c1, c2);
}

TEST_F(TestOrderedSet, test_pop) {
  OrderedSet<std::shared_ptr<int>> oset;
  oset.add(std::make_shared<int>(10));
  oset.add(std::make_shared<int>(20));
  oset.add(std::make_shared<int>(30));
  std::shared_ptr<int> ele = oset.pop();
  int pop_size = 0;
  pop_size++;
  while (oset.size() != 0) {
    ele = oset.pop();
    pop_size++;
  }
  ASSERT_EQ(pop_size, 3);
  ASSERT_EQ(oset.size(), 0);
}

TEST_F(TestOrderedSet, test_operation) {
  ASSERT_TRUE(osc.is_disjoint(osa));
  ASSERT_TRUE(!osb.is_disjoint(osa));

  ASSERT_TRUE(osc.is_subset(osb));
  ASSERT_TRUE(!osc.is_subset(osa));

  OrderedSet<std::shared_ptr<int>> res_inter = osa | osb;
  ASSERT_EQ(res_inter.size(), 30);
  OrderedSet<std::shared_ptr<int>> res_union = osa & osb;
  ASSERT_EQ(res_union.size(), 10);
  OrderedSet<std::shared_ptr<int>> res_diff = osa - osb;
  ASSERT_EQ(res_diff.size(), 10);
  OrderedSet<std::shared_ptr<int>> res_symdiff = osa ^ osb;
  ASSERT_EQ(res_symdiff.size(), 20);
}

TEST_F(TestOrderedSet, test_contains) {
  OrderedSet<std::shared_ptr<int>> res;
  std::shared_ptr<int> e1 = std::make_shared<int>(10);
  std::shared_ptr<int> e2 = std::make_shared<int>(20);
  res.add(e1);
  ASSERT_TRUE(res.contains(e1));
  ASSERT_TRUE(!res.contains(e2));
}

TEST_F(TestOrderedSet, test_assign) {
  OrderedSet<int> s;
  s.add(10);
  ASSERT_EQ(s.size(), 1);
  OrderedSet<int> s1;
  s1.add(20);
  s1.add(30);
  ASSERT_EQ(s1.size(), 2);
  s = s1;
  ASSERT_EQ(s.size(), 2);
  ASSERT_EQ(s, s1);
}

TEST_F(TestOrderedSet, test_map_assign) {
  OrderedMap<int, int> m;
  m[10] = 10;
  ASSERT_EQ(m.size(), 1);
  OrderedMap<int, int> m1;
  m1[20] = 20;
  m1[30] = 30;
  ASSERT_EQ(m1.size(), 2);
  m = m1;
  ASSERT_EQ(m.size(), 2);
  ASSERT_EQ(m[20], m1[20]);
  ASSERT_EQ(m[30], m1[30]);
}

}  // namespace mindspore
