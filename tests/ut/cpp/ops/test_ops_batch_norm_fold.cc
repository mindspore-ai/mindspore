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
#include "ops/batch_norm_fold.h"
#include "ir/dtype/type.h"
#include "ir/value.h"
#include "abstract/dshape.h"
#include "abstract/abstract_value.h"
#include "utils/tensor_construct_utils.h"

namespace mindspore {
namespace ops {

class TestBatchNormFold : public UT::Common {
 public:
  TestBatchNormFold() {}
  void SetUp() {}
  void TearDown() {}
};

TEST_F(TestBatchNormFold, test_ops_batch_norm_fold1) {
  auto batch_norm_fold = std::make_shared<BatchNormFold>();
  batch_norm_fold->Init(0.9, 1e-5, true, 0);
  EXPECT_EQ((int64_t)(batch_norm_fold->get_momentum() - 0.9), 0);
  EXPECT_EQ((int64_t)(batch_norm_fold->get_epsilon() - 1e-05), 0);
  EXPECT_EQ(batch_norm_fold->get_is_training(), true);
  EXPECT_EQ(batch_norm_fold->get_freeze_bn(), 0);
  auto input_x = TensorConstructUtils::CreateOnesTensor(kNumberTypeFloat32, std::vector<int64_t>{2, 3});
  auto mean = TensorConstructUtils::CreateOnesTensor(kNumberTypeFloat32, std::vector<int64_t>{3});
  auto variance = TensorConstructUtils::CreateOnesTensor(kNumberTypeFloat32, std::vector<int64_t>{3});
  auto global_step = TensorConstructUtils::CreateOnesTensor(kNumberTypeInt32, std::vector<int64_t>{1});
  auto abstract = batch_norm_fold->Infer(
    {input_x->ToAbstract(), mean->ToAbstract(), variance->ToAbstract(), global_step->ToAbstract()});
  MS_EXCEPTION_IF_NULL(abstract);
  EXPECT_EQ(abstract->isa<abstract::AbstractTuple>(), true);
  auto shape_ptr = abstract->BuildShape();
  MS_EXCEPTION_IF_NULL(shape_ptr);
  EXPECT_EQ(shape_ptr->isa<abstract::TupleShape>(), true);
  auto shape = shape_ptr->cast<abstract::TupleShapePtr>();
  MS_EXCEPTION_IF_NULL(shape);
  auto shape_vec = shape->shape();
  EXPECT_EQ(shape_vec.size(), 4);
  auto shape1 = shape_vec[0]->cast<abstract::ShapePtr>()->shape();
  EXPECT_EQ(shape1.size(), 1);
  EXPECT_EQ(shape1[0], 3);
  auto type_ptr = abstract->BuildType();
  MS_EXCEPTION_IF_NULL(type_ptr);
  auto type = type_ptr->cast<TuplePtr>();
  MS_EXCEPTION_IF_NULL(type);
  auto type_vec = type->elements();
  MS_EXCEPTION_IF_NULL(type_vec[0]);
  auto data_type = type_vec[0]->cast<TensorTypePtr>()->element();
  MS_EXCEPTION_IF_NULL(data_type);
  EXPECT_EQ(data_type->type_id(), kNumberTypeFloat32);
}

}  // namespace ops
}  // namespace mindspore
