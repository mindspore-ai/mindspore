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
#include "ops/prior_box.h"
#include "ir/dtype/type.h"
#include "ir/value.h"
#include "abstract/dshape.h"
#include "utils/tensor_construct_utils.h"

namespace mindspore {
namespace ops {

class TestPriorBox : public UT::Common {
 public:
  TestPriorBox() {}
  void SetUp() {}
  void TearDown() {}
};

TEST_F(TestPriorBox, test_ops_prior_box1) {
  auto prior_box = std::make_shared<PriorBox>();
  prior_box->Init({1}, {2}, {0.1}, {0.1}, 1, 1, 0.1, 0.1, false, false, 0.1);
  EXPECT_EQ(prior_box->get_min_sizes(), std::vector<int64_t>{1});
  EXPECT_EQ(prior_box->get_max_sizes(), std::vector<int64_t>{2});
  EXPECT_EQ(prior_box->get_aspect_ratios(), std::vector<float>{0.1});
  EXPECT_EQ(prior_box->get_variances(), std::vector<float>{0.1});
  EXPECT_EQ(prior_box->get_image_size_w(), 1);
  EXPECT_EQ(prior_box->get_image_size_h(), 1);
  EXPECT_EQ((int64_t)(prior_box->get_step_w() - 0.1), 0);
  EXPECT_EQ((int64_t)(prior_box->get_step_h() - 0.1), 0);
  EXPECT_EQ(prior_box->get_clip(), false);
  EXPECT_EQ(prior_box->get_flip(), false);
  EXPECT_EQ((int64_t)(prior_box->get_offset() - 0.1), 0);
  auto input0 = TensorConstructUtils::CreateOnesTensor(kNumberTypeFloat32, std::vector<int64_t>{1, 1});
  MS_EXCEPTION_IF_NULL(input0);
  auto abstract = prior_box->Infer({input0->ToAbstract()});
  MS_EXCEPTION_IF_NULL(abstract);
  EXPECT_EQ(abstract->isa<abstract::AbstractTensor>(), true);
  auto shape_ptr = abstract->BuildShape();
  MS_EXCEPTION_IF_NULL(shape_ptr);
  EXPECT_EQ(shape_ptr->isa<abstract::Shape>(), true);
  auto shape = shape_ptr->cast<abstract::ShapePtr>();
  MS_EXCEPTION_IF_NULL(shape);
  auto shape_vec = shape->shape();
  EXPECT_EQ(shape_vec.size(), 4);
  EXPECT_EQ(shape_vec[0], 1);
  EXPECT_EQ(shape_vec[1], 12);
  EXPECT_EQ(shape_vec[2], 1);
  EXPECT_EQ(shape_vec[3], 2);
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
