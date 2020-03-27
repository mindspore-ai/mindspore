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
#include "utils/counter.h"
#include "common/common_test.h"

namespace mindspore {
class TestCounter : public UT::Common {
 public:
  TestCounter() {
    std::string s1 = "abcdeedfrgbhrtfsfd";
    std::string s2 = "shceufhvogawrycawr";

    for (auto c : s1) {
      std::string key(1, c);
      counter_a[key] += 1;
    }

    for (auto c : s2) {
      std::string key(1, c);
      counter_b[key] += 1;
    }
  }

 public:
  Counter<std::string> counter_a;
  Counter<std::string> counter_b;
};

TEST_F(TestCounter, test_constructor) {
  assert(counter_a.size() == 11);
  assert(counter_b.size() == 13);
}

TEST_F(TestCounter, test_subtitle) {
  std::string s = "d";
  assert(counter_a[s] == 3);
  s = "f";
  assert(counter_a[s] == 3);
  s = "h";
  assert(counter_b[s] = 2);
  s = "c";
  assert(counter_b[s] = 2);
}

TEST_F(TestCounter, test_contains) {
  std::string s = "d";
  assert(counter_a.contains(s) == true);
  s = "z";
  assert(counter_a.contains(s) == false);
  s = "q";
  assert(counter_b.contains(s) == false);
}

TEST_F(TestCounter, test_add) {
  auto counter_add = counter_a + counter_b;
  assert(counter_add.size() == 16);
  std::string s = "f";
  assert(counter_add[s] == 4);
  s = "r";
  assert(counter_add[s] == 4);
  s = "y";
  assert(counter_add[s] == 1);
}

TEST_F(TestCounter, test_minus) {
  auto counter_minus = counter_a - counter_b;
  assert(counter_minus.size() == 5);
  std::string s = "d";
  assert(counter_minus[s] == 3);
  s = "t";
  assert(counter_minus[s] == 1);
  s = "a";
  assert(counter_minus.contains(s) == false);
}
}  // namespace mindspore
