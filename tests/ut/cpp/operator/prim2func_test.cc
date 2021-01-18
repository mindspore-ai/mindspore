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
#include <memory>
#include <vector>

#include "common/common_test.h"

#include "ir/anf.h"
#include "ir/dtype.h"
#include "frontend/operator/prim_to_function.h"
#include "base/core_ops.h"

namespace mindspore {
namespace prim {

class TestPrimFunc : public UT::Common {
 public:
  TestPrimFunc() {}
  virtual void SetUp() {}
};

TEST_F(TestPrimFunc, ScalarAddTest) {
  auto prim = std::make_shared<Primitive>(prim::kScalarAdd);
  FunctionPtr func = nullptr;
  PrimToFunction::GetInstance().GetFunction(prim, &func);

  std::vector<std::shared_ptr<Type>> two_args{std::make_shared<Number>(), std::make_shared<Number>()};
  std::shared_ptr<Type> retval = std::make_shared<Number>();
  Function func_add = Function(two_args, retval);

  std::cout << "func_add: " + func_add.ToString() << std::endl;
  std::cout << "prim_func: " + func->ToString() << std::endl;

  ASSERT_EQ(func_add.ToString(), func->ToString());
}

TEST_F(TestPrimFunc, ScalarExpTest) {
  auto prim = std::make_shared<Primitive>("scalar_exp");
  FunctionPtr func = nullptr;
  PrimToFunction::GetInstance().GetFunction(prim, &func);

  std::vector<std::shared_ptr<Type>> one_arg{std::make_shared<Number>()};
  std::shared_ptr<Type> retval = std::make_shared<Number>();
  Function func_add = Function(one_arg, retval);

  std::cout << "func_exp: " + func_add.ToString() << std::endl;
  std::cout << "prim_func: " + func->ToString() << std::endl;

  ASSERT_EQ(func_add.ToString(), func->ToString());
}

}  // namespace prim
}  // namespace mindspore
