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

#include <memory>
#include <string>
#include "common/common.h"
#include "minddata/dataset/include/datasets.h"
#include "minddata/dataset/include/transforms.h"
#include "minddata/dataset/include/vision.h"
#include "minddata/dataset/kernels/ir/vision/vision_ir.h"

using namespace mindspore::dataset;

class MindDataTestIRVision : public UT::DatasetOpTesting {
 public:
  MindDataTestIRVision() = default;
};


TEST_F(MindDataTestIRVision, TestAutoContrastIRFail1) {
  MS_LOG(INFO) << "Doing MindDataTestIRVision-TestAutoContrastIRFail1.";

  // Testing invalid cutoff < 0
  std::shared_ptr<TensorOperation> auto_contrast1(new vision::AutoContrastOperation(-1.0,{}));
  ASSERT_NE(auto_contrast1, nullptr);

  Status rc1 = auto_contrast1->ValidateParams();
  EXPECT_ERROR(rc1);

  // Testing invalid cutoff > 100
  std::shared_ptr<TensorOperation> auto_contrast2(new vision::AutoContrastOperation(110.0, {10, 20}));
  ASSERT_NE(auto_contrast2, nullptr);

  Status rc2 = auto_contrast2->ValidateParams();
  EXPECT_ERROR(rc2);
}

TEST_F(MindDataTestIRVision, TestNormalizeFail) {
  MS_LOG(INFO) << "Doing MindDataTestIRVision-TestNormalizeFail with invalid parameters.";

  // std value at 0.0
  std::shared_ptr<TensorOperation> normalize1(new vision::NormalizeOperation({121.0, 115.0, 100.0}, {0.0, 68.0, 71.0}));
  ASSERT_NE(normalize1, nullptr);

  Status rc1 = normalize1->ValidateParams();
  EXPECT_ERROR(rc1);

  // mean out of range
  std::shared_ptr<TensorOperation> normalize2(new vision::NormalizeOperation({121.0, 0.0, 100.0}, {256.0, 68.0, 71.0}));
  ASSERT_NE(normalize2, nullptr);

  Status rc2 = normalize2->ValidateParams();
  EXPECT_ERROR(rc2);

  // mean out of range
  std::shared_ptr<TensorOperation> normalize3(new vision::NormalizeOperation({256.0, 0.0, 100.0}, {70.0, 68.0, 71.0}));
  ASSERT_NE(normalize3, nullptr);

  Status rc3 = normalize3->ValidateParams();
  EXPECT_ERROR(rc3);

  // mean out of range
  std::shared_ptr<TensorOperation> normalize4(new vision::NormalizeOperation({-1.0, 0.0, 100.0}, {70.0, 68.0, 71.0}));
  ASSERT_NE(normalize4, nullptr);

  Status rc4 = normalize4->ValidateParams();
  EXPECT_ERROR(rc4);

  // normalize with 2 values (not 3 values) for mean
  std::shared_ptr<TensorOperation> normalize5(new vision::NormalizeOperation({121.0, 115.0}, {70.0, 68.0, 71.0}));
  ASSERT_NE(normalize5, nullptr);

  Status rc5 = normalize5->ValidateParams();
  EXPECT_ERROR(rc5);

  // normalize with 2 values (not 3 values) for standard deviation
  std::shared_ptr<TensorOperation> normalize6(new vision::NormalizeOperation({121.0, 115.0, 100.0}, {68.0, 71.0}));
  ASSERT_NE(normalize6, nullptr);

  Status rc6 = normalize6->ValidateParams();
  EXPECT_ERROR(rc6);
}
