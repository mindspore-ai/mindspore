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
#include <vector>
#include <memory>
#include "common/common_test.h"
#include "c_ops/conv2d.h"
#include "ir/dtype/type.h"
#include "abstract/dshape.h"
#include "utils/tensor_construct_utils.h"
namespace mindspore {
class TestConv2d : public UT::Common {
 public:
  TestConv2d() {}
  void SetUp() {}
  void TearDown() {}
};

TEST_F(TestConv2d, test_cops_conv2d) {
  auto conv_2d = std::make_shared<Conv2D>();
  conv_2d->Init(64, {7, 7});
  auto tensor_x = TensorConstructUtils::CreateOnesTensor(kNumberTypeFloat32, std::vector<int64_t>{32, 3, 224, 224});
  auto tensor_w = TensorConstructUtils::CreateOnesTensor(kNumberTypeFloat32, std::vector<int64_t>{64, 3, 7, 7});
  conv_2d->Infer({tensor_w->ToAbstract(), tensor_w->ToAbstract()});
}

}  // namespace mindspore