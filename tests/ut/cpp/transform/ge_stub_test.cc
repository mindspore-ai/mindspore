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
#include "common/common_test.h"
#include "graph/tensor.h"

#ifdef OPEN_SOURCE
#include "ge/client/ge_api.h"
#else
#include "external/ge/ge_api.h"
#endif
#include "graph/operator.h"
#include "graph/operator_reg.h"

namespace mindspore {
namespace transform {

class TestGEStub : public UT::Common {
 public:
  TestGEStub() {}
};

TEST_F(TestGEStub, TestAPI) {
  // only test for ge header compiling
  ASSERT_TRUE(true);
}

}  // namespace transform
}  // namespace mindspore
