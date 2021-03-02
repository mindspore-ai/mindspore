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
#include "common/common.h"
#include "minddata/dataset/kernels/ir/vision/vision_ir.h"

using namespace mindspore::dataset;

class MindDataTestIRVision : public UT::DatasetOpTesting {
 public:
  MindDataTestIRVision() = default;
};

TEST_F(MindDataTestIRVision, TestRandomColorIRFail) {
  MS_LOG(INFO) << "Doing MindDataTestIRVision-TestRandomColorIRFail.";

  // Testing invalid lower bound > upper bound
  std::shared_ptr<TensorOperation> random_color1(new vision::RandomColorOperation(1.0, 0.1));
  Status rc1 = random_color1->ValidateParams();
  EXPECT_ERROR(rc1);

  // Testing invalid negative lower bound
  std::shared_ptr<TensorOperation> random_color2(new vision::RandomColorOperation(-0.5, 0.5));
  Status rc2 = random_color2->ValidateParams();
  EXPECT_ERROR(rc2);
}

TEST_F(MindDataTestIRVision, TestRandomColorAdjustIRFail) {
  MS_LOG(INFO) << "Doing MindDataTestIRVision-TestRandomColorAdjustIRFail.";

  // Testing invalid brightness out of range
  std::shared_ptr<TensorOperation> random_color_adjust1(
    new vision::RandomColorAdjustOperation({-1.0}, {0.0}, {0.0}, {0.0}));
  Status rc1 = random_color_adjust1->ValidateParams();
  EXPECT_ERROR(rc1);

  // Testing invalid contrast out of range
  std::shared_ptr<TensorOperation> random_color_adjust2(
    new vision::RandomColorAdjustOperation({1.0}, {-0.1}, {0.0}, {0.0}));
  Status rc2 = random_color_adjust2->ValidateParams();
  EXPECT_ERROR(rc2);

  // Testing invalid saturation out of range
  std::shared_ptr<TensorOperation> random_color_adjust3(
    new vision::RandomColorAdjustOperation({0.0}, {0.0}, {-0.2}, {0.0}));
  Status rc3 = random_color_adjust3->ValidateParams();
  EXPECT_ERROR(rc3);

  // Testing invalid hue out of range
  std::shared_ptr<TensorOperation> random_color_adjust4(
    new vision::RandomColorAdjustOperation({0.0}, {0.0}, {0.0}, {-0.6}));
  Status rc4 = random_color_adjust4->ValidateParams();
  EXPECT_ERROR(rc4);

  // Testing invalid hue out of range
  std::shared_ptr<TensorOperation> random_color_adjust5(
    new vision::RandomColorAdjustOperation({0.0}, {0.0}, {0.0}, {-0.5, 0.6}));
  Status rc5 = random_color_adjust5->ValidateParams();
  EXPECT_ERROR(rc5);

  // Testing invalid hue
  std::shared_ptr<TensorOperation> random_color_adjust6(
    new vision::RandomColorAdjustOperation({0.0}, {0.0}, {0.0}, {0.51}));
  Status rc6 = random_color_adjust4->ValidateParams();
  EXPECT_ERROR(rc6);
}

TEST_F(MindDataTestIRVision, TestRandomHorizontalFlipIRFail) {
  MS_LOG(INFO) << "Doing MindDataTestIRVision-TestRandomHorizontalFlipIRFail.";

  // Testing invalid negative input
  std::shared_ptr<TensorOperation> random_horizontal_flip1(new vision::RandomHorizontalFlipOperation(-0.5));
  Status rc1 = random_horizontal_flip1->ValidateParams();
  EXPECT_ERROR(rc1);

  // Testing invalid >1 input
  std::shared_ptr<TensorOperation> random_horizontal_flip2(new vision::RandomHorizontalFlipOperation(2));
  Status rc2 = random_horizontal_flip2->ValidateParams();
  EXPECT_ERROR(rc2);
}

TEST_F(MindDataTestIRVision, TestRandomHorizontalFlipWithBBoxIRFail) {
  MS_LOG(INFO) << "Doing MindDataTestIRVision-TestRandomHorizontalFlipWithBBoxIRFail.";

  // Testing invalid negative input
  std::shared_ptr<TensorOperation> random_horizontal_flip_bbox1(
    new vision::RandomHorizontalFlipWithBBoxOperation(-1.0));
  Status rc1 = random_horizontal_flip_bbox1->ValidateParams();
  EXPECT_ERROR(rc1);

  // Testing invalid >1 input
  std::shared_ptr<TensorOperation> random_horizontal_flip_bbox2(new vision::RandomHorizontalFlipWithBBoxOperation(2.0));
  Status rc2 = random_horizontal_flip_bbox2->ValidateParams();
  EXPECT_ERROR(rc2);
}

TEST_F(MindDataTestIRVision, TestRandomPosterizeIRFail) {
  MS_LOG(INFO) << "Doing MindDataTestIRVision-TestRandomPosterizeIRFail.";

  // Testing invalid max > 8
  std::shared_ptr<TensorOperation> random_posterize1(new vision::RandomPosterizeOperation({1, 9}));
  Status rc1 = random_posterize1->ValidateParams();
  EXPECT_ERROR(rc1);

  // Testing invalid min < 1
  std::shared_ptr<TensorOperation> random_posterize2(new vision::RandomPosterizeOperation({0, 8}));
  Status rc2 = random_posterize2->ValidateParams();
  EXPECT_ERROR(rc2);

  // Testing invalid min > max
  std::shared_ptr<TensorOperation> random_posterize3(new vision::RandomPosterizeOperation({8, 1}));
  Status rc3 = random_posterize3->ValidateParams();
  EXPECT_ERROR(rc3);

  // Testing invalid empty input
  std::shared_ptr<TensorOperation> random_posterize4(new vision::RandomPosterizeOperation({}));
  Status rc4 = random_posterize4->ValidateParams();
  EXPECT_ERROR(rc4);
}

TEST_F(MindDataTestIRVision, TestRandomResizeIRFail) {
  MS_LOG(INFO) << "Doing MindDataTestIRVision-TestRandomResizeIRFail.";

  // Testing invalid: size must only contain positive integers
  std::shared_ptr<TensorOperation> random_resize1(new vision::RandomResizeOperation({-66, 77}));
  Status rc1 = random_resize1->ValidateParams();
  EXPECT_ERROR(rc1);

  // Testing invalid: size must only contain positive integers
  std::shared_ptr<TensorOperation> random_resize2(new vision::RandomResizeOperation({0, 77}));
  Status rc2 = random_resize2->ValidateParams();
  EXPECT_ERROR(rc2);

  // Testing invalid: size must be a vector of one or two values
  std::shared_ptr<TensorOperation> random_resize3(new vision::RandomResizeOperation({1, 2, 3}));
  Status rc3 = random_resize3->ValidateParams();
  EXPECT_ERROR(rc3);

  // Testing invalid: size must be a vector of one or two values
  std::shared_ptr<TensorOperation> random_resize4(new vision::RandomResizeOperation({}));
  Status rc4 = random_resize4->ValidateParams();
  EXPECT_ERROR(rc4);
}

TEST_F(MindDataTestIRVision, TestRandomResizeWithBBoxIRFail) {
  MS_LOG(INFO) << "Doing MindDataTestIRVision-TestRandomResizeWithBBoxIRFail.";

  // Testing invalid: size must only contain positive integers
  std::shared_ptr<TensorOperation> random_resize_with_bbox1(new vision::RandomResizeWithBBoxOperation({-66, 77}));
  Status rc1 = random_resize_with_bbox1->ValidateParams();
  EXPECT_ERROR(rc1);

  // Testing invalid: size must be a vector of one or two values
  std::shared_ptr<TensorOperation> random_resize_with_bbox2(new vision::RandomResizeWithBBoxOperation({1, 2, 3}));
  Status rc2 = random_resize_with_bbox2->ValidateParams();
  EXPECT_ERROR(rc2);

  // Testing invalid: size must be a vector of one or two values
  std::shared_ptr<TensorOperation> random_resize_with_bbox3(new vision::RandomResizeWithBBoxOperation({}));
  Status rc3 = random_resize_with_bbox3->ValidateParams();
  EXPECT_ERROR(rc3);
}

TEST_F(MindDataTestIRVision, TestRandomSharpnessIRFail) {
  MS_LOG(INFO) << "Doing MindDataTestIRVision-TestRandomSharpnessIRFail.";

  // Testing invalid empty degrees vector
  std::shared_ptr<TensorOperation> random_sharpness1(new vision::RandomSharpnessOperation({}));
  Status rc1 = random_sharpness1->ValidateParams();
  EXPECT_ERROR(rc1);

  // Testing invalid single degree value
  std::shared_ptr<TensorOperation> random_sharpness2(new vision::RandomSharpnessOperation({0.1}));
  Status rc2 = random_sharpness2->ValidateParams();
  EXPECT_ERROR(rc2);
}

TEST_F(MindDataTestIRVision, TestRandomSolarizeIRFail) {
  MS_LOG(INFO) << "Doing MindDataTestIRVision-TestRandomSolarizeIRFail.";

  // Testing invalid lower bound > upper bound
  std::shared_ptr<TensorOperation> random_solarize1(new vision::RandomSolarizeOperation({13, 1}));
  Status rc1 = random_solarize1->ValidateParams();
  EXPECT_ERROR(rc1);

  // Testing invalid threshold must be a vector of two values
  std::shared_ptr<TensorOperation> random_solarize2(new vision::RandomSolarizeOperation({1, 2, 3}));
  Status rc2 = random_solarize2->ValidateParams();
  EXPECT_ERROR(rc2);

  // Testing invalid threshold must be a vector of two values
  std::shared_ptr<TensorOperation> random_solarize3(new vision::RandomSolarizeOperation({1}));
  Status rc3 = random_solarize3->ValidateParams();
  EXPECT_ERROR(rc3);

  // Testing invalid empty threshold
  std::shared_ptr<TensorOperation> random_solarize4(new vision::RandomSolarizeOperation({}));
  Status rc4 = random_solarize4->ValidateParams();
  EXPECT_ERROR(rc4);
}

TEST_F(MindDataTestIRVision, TestRandomVerticalFlipIRFail) {
  MS_LOG(INFO) << "Doing MindDataTestIRVision-TestRandomVerticalFlipIRFail.";

  // Testing invalid negative input
  std::shared_ptr<TensorOperation> random_vertical_flip1(new vision::RandomVerticalFlipOperation(-0.5));
  Status rc1 = random_vertical_flip1->ValidateParams();
  EXPECT_ERROR(rc1);

  // Testing invalid >1 input
  std::shared_ptr<TensorOperation> random_vertical_flip2(new vision::RandomVerticalFlipOperation(1.1));
  Status rc2 = random_vertical_flip2->ValidateParams();
  EXPECT_ERROR(rc2);
}

TEST_F(MindDataTestIRVision, TestRandomVerticalFlipWithBBoxIRFail) {
  MS_LOG(INFO) << "Doing MindDataTestIRVision-TestRandomVerticalFlipWithBBoxIRFail.";

  // Testing invalid negative input
  std::shared_ptr<TensorOperation> random_vertical_flip1(new vision::RandomVerticalFlipWithBBoxOperation(-0.5));
  Status rc1 = random_vertical_flip1->ValidateParams();
  EXPECT_ERROR(rc1);

  // Testing invalid >1 input
  std::shared_ptr<TensorOperation> random_vertical_flip2(new vision::RandomVerticalFlipWithBBoxOperation(3.0));
  Status rc2 = random_vertical_flip2->ValidateParams();
  EXPECT_ERROR(rc2);
}
