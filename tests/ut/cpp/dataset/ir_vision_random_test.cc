/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#include "common/common.h"
#include "minddata/dataset/vision/transform/random_color_adjust_ir.h"
#include "minddata/dataset/vision/transform/random_crop_decode_resize_ir.h"
#include "minddata/dataset/vision/transform/random_crop_ir.h"
#include "minddata/dataset/vision/transform/random_horizontal_flip_ir.h"
#include "minddata/dataset/vision/transform/random_resized_crop_ir.h"
#include "minddata/dataset/vision/transform/random_vertical_flip_ir.h"

using namespace mindspore::dataset;

class MindDataTestIRVision : public UT::DatasetOpTesting {
 public:
  MindDataTestIRVision() = default;
};

/// Feature: RandomColorAdjust IR
/// Description: Test RandomColorAdjustOperation with invalid parameters
/// Expectation: Throw correct error and message
TEST_F(MindDataTestIRVision, TestRandomColorAdjustIRFail) {
  MS_LOG(INFO) << "Doing MindDataTestIRVision-TestRandomColorAdjustIRFail.";

  // Testing invalid brightness out of range
  auto random_color_adjust1 = std::make_shared<vision::RandomColorAdjustOperation>(
    std::vector<float>{-1.0}, std::vector<float>{0.0}, std::vector<float>{0.0}, std::vector<float>{0.0});
  Status rc1 = random_color_adjust1->ValidateParams();
  EXPECT_ERROR(rc1);

  // Testing invalid contrast out of range
  auto random_color_adjust2 = std::make_shared<vision::RandomColorAdjustOperation>(
    std::vector<float>{1.0}, std::vector<float>{-0.1}, std::vector<float>{0.0}, std::vector<float>{0.0});
  Status rc2 = random_color_adjust2->ValidateParams();
  EXPECT_ERROR(rc2);

  // Testing invalid saturation out of range
  auto random_color_adjust3 = std::make_shared<vision::RandomColorAdjustOperation>(
    std::vector<float>{0.0}, std::vector<float>{0.0}, std::vector<float>{-0.2}, std::vector<float>{0.0});
  Status rc3 = random_color_adjust3->ValidateParams();
  EXPECT_ERROR(rc3);

  // Testing invalid hue out of range
  auto random_color_adjust4 = std::make_shared<vision::RandomColorAdjustOperation>(
    std::vector<float>{0.0}, std::vector<float>{0.0}, std::vector<float>{0.0}, std::vector<float>{-0.6});
  Status rc4 = random_color_adjust4->ValidateParams();
  EXPECT_ERROR(rc4);

  // Testing invalid hue out of range
  auto random_color_adjust5 = std::make_shared<vision::RandomColorAdjustOperation>(
    std::vector<float>{0.0}, std::vector<float>{0.0}, std::vector<float>{0.0}, std::vector<float>{-0.5, 0.6});
  Status rc5 = random_color_adjust5->ValidateParams();
  EXPECT_ERROR(rc5);

  // Testing invalid hue
  auto random_color_adjust6 = std::make_shared<vision::RandomColorAdjustOperation>(
    std::vector<float>{0.0}, std::vector<float>{0.0}, std::vector<float>{0.0}, std::vector<float>{0.51});
  Status rc6 = random_color_adjust4->ValidateParams();
  EXPECT_ERROR(rc6);
}

/// Feature: RandomHorizontalFlip IR
/// Description: Test RandomHorizontalFlipOperation with invalid parameters
/// Expectation: Throw correct error and message
TEST_F(MindDataTestIRVision, TestRandomHorizontalFlipIRFail) {
  MS_LOG(INFO) << "Doing MindDataTestIRVision-TestRandomHorizontalFlipIRFail.";

  // Testing invalid negative input
  auto random_horizontal_flip1 = std::make_shared<vision::RandomHorizontalFlipOperation>(-0.5);
  Status rc1 = random_horizontal_flip1->ValidateParams();
  EXPECT_ERROR(rc1);

  // Testing invalid >1 input
  auto random_horizontal_flip2 = std::make_shared<vision::RandomHorizontalFlipOperation>(2);
  Status rc2 = random_horizontal_flip2->ValidateParams();
  EXPECT_ERROR(rc2);
}

/// Feature: RandomVerticalFlip IR
/// Description: Test RandomVerticalFlipOperation with invalid parameters
/// Expectation: Throw correct error and message
TEST_F(MindDataTestIRVision, TestRandomVerticalFlipIRFail) {
  MS_LOG(INFO) << "Doing MindDataTestIRVision-TestRandomVerticalFlipIRFail.";

  // Testing invalid negative input
  auto random_vertical_flip1 = std::make_shared<vision::RandomVerticalFlipOperation>(-0.5);
  Status rc1 = random_vertical_flip1->ValidateParams();
  EXPECT_ERROR(rc1);

  // Testing invalid >1 input
  auto random_vertical_flip2 = std::make_shared<vision::RandomVerticalFlipOperation>(1.1);
  Status rc2 = random_vertical_flip2->ValidateParams();
  EXPECT_ERROR(rc2);
}
