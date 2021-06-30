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
  HQueue<int *> hq;
  hq.Init();
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
      hq.Enqueue(v1[s]);
    }
  });
  std::thread t2([&]() {
    for (size_t s = 0; s < v2.size(); s++) {
      hq.Enqueue(v2[s]);
    }
  });

  int c1 = 0;
  int c2 = 0;

  std::thread t3([&]() {
    size_t loop = v1.size() + v2.size();
    while (loop) {
      int *val = nullptr;
      hq.Dequeue(&val);
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
  int *tmp = nullptr;
  ASSERT_EQ(hq.Dequeue(&tmp), false);

  for (size_t s = 0; s < v1.size(); s++) {
    delete v1[s];
  }

  for (size_t s = 0; s < v2.size(); s++) {
    delete v2[s];
  }
}

class TestActor : public ActorBase {
 public:
  explicit TestActor(const std::string &nm, ActorThreadPool *pool, const int i) : ActorBase(nm, pool), data(i) {}
  int Fn1(int *val) {
    if (val) {
      (*val)++;
    }
    return 0;
  }
  int Fn2(int *val) { return data + (*val); }

 public:
  int data = 0;
};

TEST_F(LiteMindRtTest, ActorThreadPoolTest) {
  Initialize("", "", "", "", 4);
  auto pool = ActorThreadPool::CreateThreadPool(4, kThreadSpin);
  AID t1 = Spawn(ActorReference(new TestActor("t1", pool, 1)));
  AID t2 = Spawn(ActorReference(new TestActor("t2", pool, 2)));
  AID t3 = Spawn(ActorReference(new TestActor("t3", pool, 3)));
  AID t4 = Spawn(ActorReference(new TestActor("t4", pool, 4)));
  AID t5 = Spawn(ActorReference(new TestActor("t5", pool, 5)));
  AID t6 = Spawn(ActorReference(new TestActor("t6", pool, 6)));

  std::vector<int *> vv;
  std::vector<Future<int>> fv;
  size_t sz = 2000;

  for (size_t i = 0; i < sz; i++) {
    vv.emplace_back(new int(i));
  }

  for (size_t i = 0; i < sz; i++) {
    int *val = vv[i];
    Future<int> ret;
    ret = Async(t1, &TestActor::Fn1, val)                 // (*vv[i])++;
            .Then(Defer(t2, &TestActor::Fn2, val), ret)   // t2.data += (*vv[i]);
            .Then(Defer(t3, &TestActor::Fn1, val), ret)   // (*vv[i])++;
            .Then(Defer(t4, &TestActor::Fn2, val), ret)   // t4.data += (*vv[i]);
            .Then(Defer(t5, &TestActor::Fn1, val), ret)   // (*vv[i])++;
            .Then(Defer(t6, &TestActor::Fn2, val), ret);  // t6.data += (*vv[i]);
    fv.emplace_back(ret);
  }

  for (size_t i = 0; i < vv.size(); i++) {
    int val = static_cast<int>(i);
    int expected = 0;

    val += 3;      // t1.Fn1
    expected = 6;  // t6.data
    expected += val;

    ASSERT_EQ(fv[i].Get(), expected);
    ASSERT_EQ(*vv[i], val);
  }

  Finalize();

  for (size_t i = 0; i < vv.size(); i++) {
    delete vv[i];
  }
}

}  // namespace mindspore
