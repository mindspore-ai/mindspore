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
#include "minddata/dataset/kernels/ir/vision/random_affine_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_color_adjust_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_color_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_crop_decode_resize_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_crop_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_crop_with_bbox_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_horizontal_flip_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_horizontal_flip_with_bbox_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_posterize_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_resized_crop_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_resized_crop_with_bbox_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_resize_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_resize_with_bbox_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_rotation_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_select_subpolicy_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_sharpness_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_solarize_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_vertical_flip_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_vertical_flip_with_bbox_ir.h"

using namespace mindspore::dataset;

class MindDataTestIRVision : public UT::DatasetOpTesting {
 public:
  MindDataTestIRVision() = default;
};

// Feature: RandomColor IR
// Description: Test RandomColorOperation with invalid parameters
// Expectation: Throw correct error and message
TEST_F(MindDataTestIRVision, TestRandomColorIRFail) {
  MS_LOG(INFO) << "Doing MindDataTestIRVision-TestRandomColorIRFail.";

  // Testing invalid lower bound > upper bound
  auto random_color1 = std::make_shared<vision::RandomColorOperation>(1.0, 0.1);
  Status rc1 = random_color1->ValidateParams();
  EXPECT_ERROR(rc1);

  // Testing invalid negative lower bound
  auto random_color2 = std::make_shared<vision::RandomColorOperation>(-0.5, 0.5);
  Status rc2 = random_color2->ValidateParams();
  EXPECT_ERROR(rc2);
}

// Feature: RandomColorAdjust IR
// Description: Test RandomColorAdjustOperation with invalid parameters
// Expectation: Throw correct error and message
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

// Feature: RandomHorizontalFlip IR
// Description: Test RandomHorizontalFlipOperation with invalid parameters
// Expectation: Throw correct error and message
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

// Feature: RandomHorizontalFlipWithBBox IR
// Description: Test RandomHorizontalFlipWithBBoxOperation with invalid parameters
// Expectation: Throw correct error and message
TEST_F(MindDataTestIRVision, TestRandomHorizontalFlipWithBBoxIRFail) {
  MS_LOG(INFO) << "Doing MindDataTestIRVision-TestRandomHorizontalFlipWithBBoxIRFail.";

  // Testing invalid negative input
  auto random_horizontal_flip_bbox1 = std::make_shared<vision::RandomHorizontalFlipWithBBoxOperation>(
    -1.0);
  Status rc1 = random_horizontal_flip_bbox1->ValidateParams();
  EXPECT_ERROR(rc1);

  // Testing invalid >1 input
  auto random_horizontal_flip_bbox2 = std::make_shared<vision::RandomHorizontalFlipWithBBoxOperation>(2.0);
  Status rc2 = random_horizontal_flip_bbox2->ValidateParams();
  EXPECT_ERROR(rc2);
}

// Feature: RandomPosterize IR
// Description: Test RandomPosterizeOperation with invalid parameters
// Expectation: Throw correct error and message
TEST_F(MindDataTestIRVision, TestRandomPosterizeIRFail) {
  MS_LOG(INFO) << "Doing MindDataTestIRVision-TestRandomPosterizeIRFail.";

  // Testing invalid max > 8
  auto random_posterize1 = std::make_shared<vision::RandomPosterizeOperation>(std::vector<uint8_t>{1, 9});
  Status rc1 = random_posterize1->ValidateParams();
  EXPECT_ERROR(rc1);

  // Testing invalid min < 1
  auto random_posterize2 = std::make_shared<vision::RandomPosterizeOperation>(std::vector<uint8_t>{0, 8});
  Status rc2 = random_posterize2->ValidateParams();
  EXPECT_ERROR(rc2);

  // Testing invalid min > max
  auto random_posterize3 = std::make_shared<vision::RandomPosterizeOperation>(std::vector<uint8_t>{8, 1});
  Status rc3 = random_posterize3->ValidateParams();
  EXPECT_ERROR(rc3);

  // Testing invalid empty input
  auto random_posterize4 = std::make_shared<vision::RandomPosterizeOperation>(std::vector<uint8_t>{});
  Status rc4 = random_posterize4->ValidateParams();
  EXPECT_ERROR(rc4);
}

// Feature: RandomResize IR
// Description: Test RandomResizeOperation with invalid parameters
// Expectation: Throw correct error and message
TEST_F(MindDataTestIRVision, TestRandomResizeIRFail) {
  MS_LOG(INFO) << "Doing MindDataTestIRVision-TestRandomResizeIRFail.";

  // Testing invalid: size must only contain positive integers
  auto random_resize1 = std::make_shared<vision::RandomResizeOperation>(std::vector<int32_t>{-66, 77});
  Status rc1 = random_resize1->ValidateParams();
  EXPECT_ERROR(rc1);

  // Testing invalid: size must only contain positive integers
  auto random_resize2 = std::make_shared<vision::RandomResizeOperation>(std::vector<int32_t>{0, 77});
  Status rc2 = random_resize2->ValidateParams();
  EXPECT_ERROR(rc2);

  // Testing invalid: size must be a vector of one or two values
  auto random_resize3 = std::make_shared<vision::RandomResizeOperation>(std::vector<int32_t>{1, 2, 3});
  Status rc3 = random_resize3->ValidateParams();
  EXPECT_ERROR(rc3);

  // Testing invalid: size must be a vector of one or two values
  auto random_resize4 = std::make_shared<vision::RandomResizeOperation>(std::vector<int32_t>{});
  Status rc4 = random_resize4->ValidateParams();
  EXPECT_ERROR(rc4);
}

// Feature: RandomResizeWithBBox IR
// Description: Test RandomResizeWithBBoxOperation with invalid parameters
// Expectation: Throw correct error and message
TEST_F(MindDataTestIRVision, TestRandomResizeWithBBoxIRFail) {
  MS_LOG(INFO) << "Doing MindDataTestIRVision-TestRandomResizeWithBBoxIRFail.";

  // Testing invalid: size must only contain positive integers
  auto random_resize_with_bbox1 = std::make_shared<vision::RandomResizeWithBBoxOperation>(
    std::vector<int32_t>{-66, 77});
  Status rc1 = random_resize_with_bbox1->ValidateParams();
  EXPECT_ERROR(rc1);

  // Testing invalid: size must be a vector of one or two values
  auto random_resize_with_bbox2 = std::make_shared<vision::RandomResizeWithBBoxOperation>(
    std::vector<int32_t>{1, 2, 3});
  Status rc2 = random_resize_with_bbox2->ValidateParams();
  EXPECT_ERROR(rc2);

  // Testing invalid: size must be a vector of one or two values
  auto random_resize_with_bbox3 = std::make_shared<vision::RandomResizeWithBBoxOperation>(
    std::vector<int32_t>{});
  Status rc3 = random_resize_with_bbox3->ValidateParams();
  EXPECT_ERROR(rc3);
}

// Feature: RandomSharpness IR
// Description: Test RandomSharpnessOperation with invalid parameters
// Expectation: Throw correct error and message
TEST_F(MindDataTestIRVision, TestRandomSharpnessIRFail) {
  MS_LOG(INFO) << "Doing MindDataTestIRVision-TestRandomSharpnessIRFail.";

  // Testing invalid empty degrees vector
  auto random_sharpness1 = std::make_shared<vision::RandomSharpnessOperation>(std::vector<float>{});
  Status rc1 = random_sharpness1->ValidateParams();
  EXPECT_ERROR(rc1);

  // Testing invalid single degree value
  auto random_sharpness2 = std::make_shared<vision::RandomSharpnessOperation>(std::vector<float>{0.1});
  Status rc2 = random_sharpness2->ValidateParams();
  EXPECT_ERROR(rc2);
}

// Feature: RandomSolarize IR
// Description: Test RandomSolarizeOperation with invalid parameters
// Expectation: Throw correct error and message
TEST_F(MindDataTestIRVision, TestRandomSolarizeIRFail) {
  MS_LOG(INFO) << "Doing MindDataTestIRVision-TestRandomSolarizeIRFail.";

  // Testing invalid lower bound > upper bound
  auto random_solarize1 = std::make_shared<vision::RandomSolarizeOperation>(std::vector<uint8_t>{13, 1});
  Status rc1 = random_solarize1->ValidateParams();
  EXPECT_ERROR(rc1);

  // Testing invalid threshold must be a vector of two values
  auto random_solarize2 = std::make_shared<vision::RandomSolarizeOperation>(std::vector<uint8_t>{1, 2, 3});
  Status rc2 = random_solarize2->ValidateParams();
  EXPECT_ERROR(rc2);

  // Testing invalid threshold must be a vector of two values
  auto random_solarize3 = std::make_shared<vision::RandomSolarizeOperation>(std::vector<uint8_t>{1});
  Status rc3 = random_solarize3->ValidateParams();
  EXPECT_ERROR(rc3);

  // Testing invalid empty threshold
  auto random_solarize4 = std::make_shared<vision::RandomSolarizeOperation>(std::vector<uint8_t>{});
  Status rc4 = random_solarize4->ValidateParams();
  EXPECT_ERROR(rc4);
}

// Feature: RandomVerticalFlip IR
// Description: Test RandomVerticalFlipOperation with invalid parameters
// Expectation: Throw correct error and message
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

// Feature: RandomVerticalFlipWithBBox IR
// Description: Test RandomVerticalFlipWithBBoxOperation with invalid parameters
// Expectation: Throw correct error and message
TEST_F(MindDataTestIRVision, TestRandomVerticalFlipWithBBoxIRFail) {
  MS_LOG(INFO) << "Doing MindDataTestIRVision-TestRandomVerticalFlipWithBBoxIRFail.";

  // Testing invalid negative input
  auto random_vertical_flip1 = std::make_shared<vision::RandomVerticalFlipWithBBoxOperation>(-0.5);
  Status rc1 = random_vertical_flip1->ValidateParams();
  EXPECT_ERROR(rc1);

  // Testing invalid >1 input
  auto random_vertical_flip2 = std::make_shared<vision::RandomVerticalFlipWithBBoxOperation>(3.0);
  Status rc2 = random_vertical_flip2->ValidateParams();
  EXPECT_ERROR(rc2);
}
