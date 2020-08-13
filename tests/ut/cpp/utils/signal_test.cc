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

#include "ir/named.h"
#include "utils/signal.h"
#include "common/common_test.h"

using std::cout;
using std::endl;
using std::string;

namespace mindspore {
class TestSignal : public UT::Common {
 public:
  TestSignal() {}
};

struct signals {
  Signal<void(int, float, std::string)> signal;
  Signal<void(std::shared_ptr<Named>)> signal1;
};

class A {
 public:
  A() {}
  explicit A(signals *sigs) : sigs_(sigs) {
    sigs_->signal.connect(this, &A::FuncA);
    sigs_->signal1.connect(this, &A::Funct);
    printf("conn:%p\n", this);
    i = std::make_shared<int>(1);
  }
  void Funct(std::shared_ptr<Named> a) {}
  virtual void FuncA(int v1, float v2, std::string str) { printf("A: --%d--%f--%s--\n", v1, v2, str.c_str()); }

 private:
  signals *sigs_;
  std::shared_ptr<int> i;
};

class Ca : public A {
 public:
  Ca() {}
  explicit Ca(signals *sigs) : A(sigs) { printf("conn C:%p\n", this); }
  void FuncA(int v1, float v2, std::string str) { printf("C: --%d--%f--%s--\n", v1, v2, str.c_str()); }
};

class B : public A {
 public:
  B() {}
  explicit B(signals *sigs) : A(sigs) { printf("conn B:%p\n", this); }
  void FuncA(int v1, float v2, std::string str) { printf("B: --%d--%f--%s--\n", v1, v2, str.c_str()); }
};

TEST_F(TestSignal, test_common) {
  A objA;
  B objB;
  Ca objC;

  Signal<void(int, float, std::string)> signal;

  signal.connect(&objA, &A::FuncA);
  signal.connect(&objB, &B::FuncA);
  signal.connect(&objC, &Ca::FuncA);
  signal(20, 20, "Signal-Slot test");
}

TEST_F(TestSignal, test_sigs) {
  signals sigs;
  A objA(&sigs);
  B objB(&sigs);
  Ca objC(&sigs);

  sigs.signal.connect(&objA, &A::FuncA);
  sigs.signal.connect(&objB, &B::FuncA);
  sigs.signal.connect(&objC, &Ca::FuncA);
  sigs.signal(20, 20, "sigs Signal-Slot test");
}

TEST_F(TestSignal, test_sigs_Named) {
  signals sigs;
  A objA(&sigs);
  B objB(&sigs);
  Ca objC(&sigs);

  sigs.signal(10, 20, "Signal-Slot test");
  std::shared_ptr<Named> a;
  sigs.signal1(a);
}

}  // namespace mindspore
