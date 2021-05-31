/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "actor/actor.h"
#include "actor/op_actor.h"
#include "async/uuid_base.h"
#include "async/future.h"
#include "src/lite_mindrt.h"
#include "thread/hqueue.h"
#include "common/common_test.h"

namespace mindspore {
class LiteMindRtTest : public mindspore::CommonTest {
 public:
  LiteMindRtTest() {}
};

TEST_F(LiteMindRtTest, HQueueTest) {
  HQueue<int> hq(10000);
  std::vector<int *> v1(2000);
  int d1 = 1;
  for (size_t s = 0; s < v1.size(); s++) {
    v1[s] = new int(d1);
  }
  std::vector<int *> v2(2000);
  int d2 = 2;
  for (size_t s = 0; s < v2.size(); s++) {
    v2[s] = new int(d2);
  }

  std::thread t1([&]() {
    for (size_t s = 0; s < v1.size(); s++) {
      ASSERT_EQ(hq.Enqueue(v1[s]), true);
    }
  });
  std::thread t2([&]() {
    for (size_t s = 0; s < v2.size(); s++) {
      ASSERT_EQ(hq.Enqueue(v2[s]), true);
    }
  });

  int c1 = 0;
  int c2 = 0;

  std::thread t3([&]() {
    size_t loop = v1.size() + v2.size();
    while (loop) {
      int *val = hq.Dequeue();
      if (val == nullptr) {
        continue;
      }
      loop--;
      if (*val == d1) {
        c1++;
      } else if (*val == d2) {
        c2++;
      } else {
        // should never come here
        ASSERT_EQ(0, 1);
      }
    }
  });

  t1.join();
  t2.join();
  t3.join();

  ASSERT_EQ(c1, v1.size());
  ASSERT_EQ(c2, v2.size());
  ASSERT_EQ(hq.Dequeue(), nullptr);

  for (size_t s = 0; s < v1.size(); s++) {
    delete v1[s];
  }

  for (size_t s = 0; s < v2.size(); s++) {
    delete v2[s];
  }
}
}  // namespace mindspore
