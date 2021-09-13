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

struct MyStruct {
  int a = 0;
  int b = 0;
};

struct MyHash {
  std::size_t operator()(const MyStruct &e) const noexcept {  //
    return (static_cast<std::size_t>(e.a) << 16) + e.b;
  }
};

struct MyEqual {
  bool operator()(const MyStruct &lhs, const MyStruct &rhs) const noexcept {  //
    return lhs.a == rhs.a && lhs.b == rhs.b;
  }
};

TEST_F(TestCounter, test_struct) {
  using MyCounter = Counter<MyStruct, MyHash, MyEqual>;
  MyCounter counter;
  counter.add(MyStruct{100, 1});
  counter.add(MyStruct{100, 2});
  counter.add(MyStruct{100, 2});
  counter.add(MyStruct{100, 3});
  counter.add(MyStruct{100, 3});
  counter.add(MyStruct{100, 3});
  ASSERT_EQ(1, (counter[MyStruct{100, 1}]));
  ASSERT_EQ(2, (counter[MyStruct{100, 2}]));
  ASSERT_EQ(3, (counter[MyStruct{100, 3}]));

  MyCounter counter2;
  counter2.add(MyStruct{100, 2});
  counter2.add(MyStruct{100, 3});
  counter2.add(MyStruct{100, 3});
  counter2.add(MyStruct{100, 3});
  counter2.add(MyStruct{100, 4});

  auto result = counter.subtract(counter2);
  ASSERT_EQ(2, result.size());
  ASSERT_TRUE((MyEqual{}(MyStruct{100, 1}, result[0])));
  ASSERT_TRUE((MyEqual{}(MyStruct{100, 2}, result[1])));

  counter2 = counter;
  ASSERT_EQ(3, counter2.size());
  ASSERT_EQ(1, (counter2[MyStruct{100, 1}]));
  ASSERT_EQ(2, (counter2[MyStruct{100, 2}]));
  ASSERT_EQ(3, (counter2[MyStruct{100, 3}]));
}

}  // namespace mindspore
