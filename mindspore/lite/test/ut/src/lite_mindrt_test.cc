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
// #include <sys/time.h>
#include "actor/actor.h"
#include "actor/op_actor.h"
#include "async/uuid_base.h"
#include "async/future.h"
#include "src/lite_mindrt.h"
#include "thread/hqueue.h"
#include "thread/actor_threadpool.h"
#include "common/common_test.h"
#include "schema/model_generated.h"
#include "include/model.h"

namespace mindspore {
class LiteMindRtTest : public mindspore::CommonTest {
 public:
  LiteMindRtTest() {}
};

// TEST_F(LiteMindRtTest, HQueueTest) {
//  HQueue<int *> hq;
//  hq.Init();
//  std::vector<int *> v1(2000);
//  int d1 = 1;
//  for (size_t s = 0; s < v1.size(); s++) {
//    v1[s] = new int(d1);
//  }
//  std::vector<int *> v2(2000);
//  int d2 = 2;
//  for (size_t s = 0; s < v2.size(); s++) {
//    v2[s] = new int(d2);
//  }
//
//  std::thread t1([&]() {
//    for (size_t s = 0; s < v1.size(); s++) {
//      hq.Enqueue(v1[s]);
//    }
//  });
//  std::thread t2([&]() {
//    for (size_t s = 0; s < v2.size(); s++) {
//      hq.Enqueue(v2[s]);
//    }
//  });
//
//  int c1 = 0;
//  int c2 = 0;
//
//  std::thread t3([&]() {
//    size_t loop = v1.size() + v2.size();
//    while (loop) {
//      int *val = nullptr;
//      hq.Dequeue(&val);
//      if (val == nullptr) {
//        continue;
//      }
//      loop--;
//      if (*val == d1) {
//        c1++;
//      } else if (*val == d2) {
//        c2++;
//      } else {
//        // should never come here
//        ASSERT_EQ(0, 1);
//      }
//    }
//  });
//
//  t1.join();
//  t2.join();
//  t3.join();
//
//  ASSERT_EQ(c1, v1.size());
//  ASSERT_EQ(c2, v2.size());
//  int *tmp = nullptr;
//  ASSERT_EQ(hq.Dequeue(&tmp), false);
//
//  for (size_t s = 0; s < v1.size(); s++) {
//    delete v1[s];
//  }
//
//  for (size_t s = 0; s < v2.size(); s++) {
//    delete v2[s];
//  }
//}

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
  Initialize("", "", "", "", 40);
  auto pool = ActorThreadPool::CreateThreadPool(40);
  std::vector<AID> actors;
  for (size_t i = 0; i < 200; i++) {
    AID t1 = Spawn(ActorReference(new TestActor(to_string(i), pool, i)));
    actors.emplace_back(t1);
  }

  std::vector<int *> vv;
  std::vector<Future<int>> fv;
  std::vector<int> expected;
  size_t sz = 300;

  //  struct timeval start, end;
  //  gettimeofday(&start, NULL);
  for (size_t i = 0; i < sz; i++) {
    int data = 0;
    for (auto a : actors) {
      int *val = new int(i);
      vv.emplace_back(val);
      Future<int> ret;
      ret = Async(a, &TestActor::Fn1, val)                 // (*vv[i])++;
              .Then(Defer(a, &TestActor::Fn2, val), ret);  // t2.data += (*vv[i]);
      fv.emplace_back(ret);
      expected.emplace_back(data + i + 1);
      data++;
    }
  }
  for (size_t i = 0; i < vv.size(); i++) {
    int ret = fv[i].Get();
    ASSERT_EQ(ret, expected[i]);
  }
  //  gettimeofday(&end, NULL);
  //
  //  std::cout << "start: " << start.tv_sec << "." << start.tv_usec << std::endl;
  //  std::cout << "end: " << end.tv_sec << "." << end.tv_usec << std::endl;
  //  std::cout << "consumed: " << (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec) << " us"
  //            << std::endl;

  Finalize();

  for (size_t i = 0; i < vv.size(); i++) {
    delete vv[i];
  }
}

}  // namespace mindspore
