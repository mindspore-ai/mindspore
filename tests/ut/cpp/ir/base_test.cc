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
#include <memory>

#include "common/common_test.h"
#include "utils/any.h"
#include "ir/base.h"
#include "ir/anf.h"
#include "utils/log_adapter.h"

namespace mindspore {

class TestNode : public UT::Common {
 public:
  TestNode() {}
};

class ChildA : public Base {
 public:
  ChildA() {}
  ~ChildA() {}
  MS_DECLARE_PARENT(ChildA, Base);
  std::string name() { return "ChildA"; }
  std::size_t hash() const override { return 1; }
};
class ChildAA : public ChildA {
 public:
  ChildAA() {}
  ~ChildAA() {}
  MS_DECLARE_PARENT(ChildAA, ChildA);
  std::size_t hash() const override { return 1; }
  std::string name() { return "ChildAA"; }
};

class ChildB : public Base {
 public:
  ChildB() {}
  ~ChildB() {}
  MS_DECLARE_PARENT(ChildB, Base);
  std::size_t hash() const override { return 1; }
  std::string name() { return "ChildB"; }
};

TEST_F(TestNode, test_dyn_cast) {
  auto aa = std::make_shared<ChildAA>();
  std::shared_ptr<Base> n = aa;
  MS_LOG(INFO) << "aa ptr_name: " << aa->name();
  MS_LOG(INFO) << "aa type_name: " << aa->type_name();
  MS_LOG(INFO) << "n ptr_name: " << demangle(typeid(n).name());
  MS_LOG(INFO) << "n type_name: " << n->type_name();
  ASSERT_TRUE(n != nullptr);
  ASSERT_EQ(std::string(n->type_name().c_str()), "ChildAA");
  auto a = dyn_cast<ChildA>(n);
  MS_LOG(INFO) << "a ptr_name: " << a->name();
  MS_LOG(INFO) << "a type_name: " << a->type_name();
  ASSERT_TRUE(a != nullptr);
  ASSERT_EQ(std::string(a->name()), "ChildA");
  ASSERT_EQ(std::string(a->type_name().c_str()), "ChildAA");
  auto b_null = dyn_cast<ChildB>(n);
  ASSERT_TRUE(b_null == nullptr);

  ChildA* pa = cast<ChildA>(n.get());
  ASSERT_TRUE(pa != nullptr);
  MS_LOG(INFO) << "a ptr_name: " << pa->name();
  MS_LOG(INFO) << "a type_name: " << pa->type_name();
}

TEST_F(TestNode, test_isa) {
  auto a = std::make_shared<ChildA>();
  BasePtr n = a;
  ASSERT_TRUE(n->isa<ChildA>() == true);
  ASSERT_TRUE(n->isa<ChildAA>() == false);

  auto aa = std::make_shared<ChildAA>();
  n = aa;
  ASSERT_TRUE(n->isa<ChildA>() == true);
  ASSERT_TRUE(n->isa<ChildAA>() == true);

  auto b = std::make_shared<ChildB>();
  n = b;
  ASSERT_TRUE(n->isa<ChildB>() == true);
  ASSERT_TRUE(n->isa<ChildA>() == false);
  ASSERT_TRUE(n->isa<ChildAA>() == false);
}

}  // namespace mindspore
