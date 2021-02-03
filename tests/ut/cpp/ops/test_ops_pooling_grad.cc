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
#include "ops/grad/pooling_grad.h"
#include "ir/dtype/type.h"
#include "ir/value.h"
#include "abstract/dshape.h"
#include "utils/tensor_construct_utils.h"

namespace mindspore {
namespace ops {
class TestPoolingGrad : public UT::Common {
 public:
  TestPoolingGrad() {}
  void SetUp() {}
  void TearDown() {}
};

TEST_F(TestPoolingGrad, test_ops_pooling_grad1) {
  auto pooling_grad = std::make_shared<PoolingGrad>();
  pooling_grad->Init(MAX_POOLING, std::vector<int64_t>{1, 1}, std::vector<int64_t>{1, 1}, VALID,
                     std::vector<int64_t>{1, 1, 1, 1}, FLOOR, NCHW, false);
  EXPECT_EQ(pooling_grad->get_pool_mode(), MAX_POOLING);
  //  EXPECT_EQ(pooling_grad->get_window(), std::vector<int64_t>{1, 1});
  EXPECT_EQ(pooling_grad->get_pad_mode(), VALID);
  //  EXPECT_EQ(pooling_grad->get_stride(), std::vector<int64_t>{1, 1});
  //  EXPECT_EQ(pooling_grad->get_pad_list(), std::vector<int64_t>{1, 1, 1, 1});
  EXPECT_EQ(pooling_grad->get_round_mode(), FLOOR);
  EXPECT_EQ(pooling_grad->get_format(), NCHW);
  EXPECT_EQ(pooling_grad->get_global(), false);
  auto input0 = TensorConstructUtils::CreateOnesTensor(kNumberTypeFloat32, std::vector<int64_t>{1});
  auto input1 = TensorConstructUtils::CreateOnesTensor(kNumberTypeFloat32, std::vector<int64_t>{1});
  auto input2 = TensorConstructUtils::CreateOnesTensor(kNumberTypeFloat32, std::vector<int64_t>{3, 3});
  MS_EXCEPTION_IF_NULL(input0);
  MS_EXCEPTION_IF_NULL(input1);
  MS_EXCEPTION_IF_NULL(input2);
  auto abstract = pooling_grad->Infer({input0->ToAbstract(), input1->ToAbstract(), input2->ToAbstract()});
  MS_EXCEPTION_IF_NULL(abstract);
  EXPECT_EQ(abstract->isa<abstract::AbstractTensor>(), true);
  auto shape_ptr = abstract->BuildShape();
  MS_EXCEPTION_IF_NULL(shape_ptr);
  EXPECT_EQ(shape_ptr->isa<abstract::Shape>(), true);
  auto shape = shape_ptr->cast<abstract::ShapePtr>();
  MS_EXCEPTION_IF_NULL(shape);
  auto shape_vec = shape->shape();
  EXPECT_EQ(shape_vec.size(), 2);
  EXPECT_EQ(shape_vec[0], 3);
  EXPECT_EQ(shape_vec[1], 3);
  auto type = abstract->BuildType();
  MS_EXCEPTION_IF_NULL(type);
  EXPECT_EQ(type->isa<TensorType>(), true);
  auto tensor_type = type->cast<TensorTypePtr>();
  MS_EXCEPTION_IF_NULL(tensor_type);
  auto data_type = tensor_type->element();
  MS_EXCEPTION_IF_NULL(data_type);
  EXPECT_EQ(data_type->type_id(), kNumberTypeFloat32);
}
}  // namespace ops
}  // namespace mindspore
