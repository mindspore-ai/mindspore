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
#include <string>
#include "common/common.h"
#include "minddata/dataset/kernels/ir/vision/affine_ir.h"
#include "minddata/dataset/kernels/ir/vision/auto_contrast_ir.h"
#include "minddata/dataset/kernels/ir/vision/bounding_box_augment_ir.h"
#include "minddata/dataset/kernels/ir/vision/center_crop_ir.h"
#include "minddata/dataset/kernels/ir/vision/crop_ir.h"
#include "minddata/dataset/kernels/ir/vision/cutmix_batch_ir.h"
#include "minddata/dataset/kernels/ir/vision/cutout_ir.h"
#include "minddata/dataset/kernels/ir/vision/decode_ir.h"
#include "minddata/dataset/kernels/ir/vision/equalize_ir.h"
#include "minddata/dataset/kernels/ir/vision/hwc_to_chw_ir.h"
#include "minddata/dataset/kernels/ir/vision/invert_ir.h"
#include "minddata/dataset/kernels/ir/vision/mixup_batch_ir.h"
#include "minddata/dataset/kernels/ir/vision/normalize_ir.h"
#include "minddata/dataset/kernels/ir/vision/normalize_pad_ir.h"
#include "minddata/dataset/kernels/ir/vision/pad_ir.h"
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
#include "minddata/dataset/kernels/ir/vision/rescale_ir.h"
#include "minddata/dataset/kernels/ir/vision/resize_ir.h"
#include "minddata/dataset/kernels/ir/vision/resize_preserve_ar_ir.h"
#include "minddata/dataset/kernels/ir/vision/resize_with_bbox_ir.h"
#include "minddata/dataset/kernels/ir/vision/rgba_to_bgr_ir.h"
#include "minddata/dataset/kernels/ir/vision/rgba_to_rgb_ir.h"
#include "minddata/dataset/kernels/ir/vision/rgb_to_gray_ir.h"
#include "minddata/dataset/kernels/ir/vision/rotate_ir.h"
#include "minddata/dataset/kernels/ir/vision/swap_red_blue_ir.h"
#include "minddata/dataset/kernels/ir/vision/uniform_aug_ir.h"

using namespace mindspore::dataset;

class MindDataTestIRVision : public UT::DatasetOpTesting {
 public:
  MindDataTestIRVision() = default;
};

/// Feature: AutoContrast op
/// Description: Test AutoContrast op with invalid cutoff
/// Expectation: Throw correct error and message
TEST_F(MindDataTestIRVision, TestAutoContrastFail1) {
  MS_LOG(INFO) << "Doing MindDataTestIRVision-TestAutoContrastFail1.";

  // Testing invalid cutoff < 0
  auto auto_contrast1 = std::make_shared<vision::AutoContrastOperation>(-1.0, std::vector<uint32_t>{});
  Status rc1 = auto_contrast1->ValidateParams();
  EXPECT_ERROR(rc1);

  // Testing invalid cutoff > 100
  auto auto_contrast2 = std::make_shared<vision::AutoContrastOperation>(110.0, std::vector<uint32_t>{10, 20});
  Status rc2 = auto_contrast2->ValidateParams();
  EXPECT_ERROR(rc2);
}

/// Feature: CenterCrop op
/// Description: Test CenterCrop op with invalid parameters
/// Expectation: Throw correct error and message
TEST_F(MindDataTestIRVision, TestCenterCropFail) {
  MS_LOG(INFO) << "Doing MindDataTestIRVision-TestCenterCrop with invalid parameters.";

  Status rc;

  // center crop height value negative
  auto center_crop1 = std::make_shared<vision::CenterCropOperation>(std::vector<int32_t>{-32, 32});
  rc = center_crop1->ValidateParams();
  EXPECT_ERROR(rc);

  // center crop width value negative
  auto center_crop2 = std::make_shared<vision::CenterCropOperation>(std::vector<int32_t>{32, -32});
  rc = center_crop2->ValidateParams();
  EXPECT_ERROR(rc);

  // 0 value would result in nullptr
  auto center_crop3 = std::make_shared<vision::CenterCropOperation>(std::vector<int32_t>{0, 32});
  rc = center_crop3->ValidateParams();
  EXPECT_ERROR(rc);

  // center crop with 3 values
  auto center_crop4 = std::make_shared<vision::CenterCropOperation>(std::vector<int32_t>{10, 20, 30});
  rc = center_crop4->ValidateParams();
  EXPECT_ERROR(rc);
}

/// Feature: Crop op
/// Description: Test Crop op with invalid parameters
/// Expectation: Throw correct error and message
TEST_F(MindDataTestIRVision, TestCropFail) {
  MS_LOG(INFO) << "Doing MindDataTestIRVision-TestCrop with invalid parameters.";

  Status rc;

  // wrong width
  auto crop1 = std::make_shared<vision::CropOperation>(
    std::vector<int32_t>{0, 0}, std::vector<int32_t>{32, -32});
  rc = crop1->ValidateParams();
  EXPECT_ERROR(rc);

  // wrong height
  auto crop2 = std::make_shared<vision::CropOperation>(
    std::vector<int32_t>{0, 0}, std::vector<int32_t>{-32, -32});
  rc = crop2->ValidateParams();
  EXPECT_ERROR(rc);

  // zero height
  auto crop3 = std::make_shared<vision::CropOperation>(
    std::vector<int32_t>{0, 0}, std::vector<int32_t>{0, 32});
  rc = crop3->ValidateParams();
  EXPECT_ERROR(rc);

  // negative coordinates
  auto crop4 = std::make_shared<vision::CropOperation>(
    std::vector<int32_t>{-1, 0}, std::vector<int32_t>{32, 32});
  rc = crop4->ValidateParams();
  EXPECT_ERROR(rc);
}

/// Feature: CutOut op
/// Description: Test CutOut op with invalid parameters (negative length and number of patches)
/// Expectation: Throw correct error and message
TEST_F(MindDataTestIRVision, TestCutOutFail1) {
  MS_LOG(INFO) << "Doing MindDataTestIRVision-TestCutOutFail1 with invalid parameters.";

  Status rc;

  // Create object for the tensor op
  // Invalid negative length
  std::shared_ptr<TensorOperation> cutout_op = std::make_shared<vision::CutOutOperation>(-10, 1, true);
  rc = cutout_op->ValidateParams();
  EXPECT_ERROR(rc);

  // Invalid negative number of patches
  cutout_op = std::make_shared<vision::CutOutOperation>(10, -1, true);
  rc = cutout_op->ValidateParams();
  EXPECT_ERROR(rc);
}

/// Feature: CutOut op
/// Description: Test CutOut op with invalid parameters (zero length and number of patches)
/// Expectation: Throw correct error and message
TEST_F(MindDataTestIRVision, TestCutOutFail2) {
  MS_LOG(INFO) << "Doing MindDataTestIRVision-TestCutOutFail2 with invalid params, boundary cases.";

  Status rc;

  // Create object for the tensor op
  // Invalid zero length
  std::shared_ptr<TensorOperation> cutout_op = std::make_shared<vision::CutOutOperation>(0, 1, true);
  rc = cutout_op->ValidateParams();
  EXPECT_ERROR(rc);

  // Invalid zero number of patches
  cutout_op = std::make_shared<vision::CutOutOperation>(10, 0, true);
  rc = cutout_op->ValidateParams();
  EXPECT_ERROR(rc);
}

/// Feature: Normalize op
/// Description: Test invalid input parameters at IR level
/// Expectation: Throw correct error and message
TEST_F(MindDataTestIRVision, TestNormalizeFail) {
  MS_LOG(INFO) << "Doing MindDataTestIRVision-TestNormalizeFail with invalid parameters.";

  Status rc;
  std::vector<float> mean;
  std::vector<float> std;

  // std value 0.0 out of range
  mean = {121.0, 115.0, 100.0};
  std = {0.0, 68.0, 71.0};
  std::shared_ptr<TensorOperation> normalize1 = std::make_shared<vision::NormalizeOperation>(mean, std, true);
  rc = normalize1->ValidateParams();
  EXPECT_ERROR(rc);

  // std value 256.0 out of range
  mean = {121.0, 115.0, 100.0};
  std = {256.0, 68.0, 71.0};
  std::shared_ptr<TensorOperation> normalize2 = std::make_shared<vision::NormalizeOperation>(mean, std, true);
  rc = normalize2->ValidateParams();
  EXPECT_ERROR(rc);

  // mean value 256.0 out of range
  mean = {256.0, 0.0, 100.0};
  std = {70.0, 68.0, 71.0};
  std::shared_ptr<TensorOperation> normalize3 = std::make_shared<vision::NormalizeOperation>(mean, std, true);
  rc = normalize3->ValidateParams();
  EXPECT_ERROR(rc);

  // mean value 0.0 out of range
  mean = {-1.0, 0.0, 100.0};
  std = {70.0, 68.0, 71.0};
  std::shared_ptr<TensorOperation> normalize4 = std::make_shared<vision::NormalizeOperation>(mean, std, true);
  rc = normalize4->ValidateParams();
  EXPECT_ERROR(rc);

  // normalize with 2 values (not 3 values) for mean
  mean = {121.0, 115.0};
  std = {70.0, 68.0, 71.0};
  std::shared_ptr<TensorOperation> normalize5 = std::make_shared<vision::NormalizeOperation>(mean, std, true);
  rc = normalize5->ValidateParams();
  EXPECT_ERROR(rc);

  // normalize with 2 values (not 3 values) for standard deviation
  mean = {121.0, 115.0, 100.0};
  std = {68.0, 71.0};
  std::shared_ptr<TensorOperation> normalize6 = std::make_shared<vision::NormalizeOperation>(mean, std, true);
  rc = normalize6->ValidateParams();
  EXPECT_ERROR(rc);
}

/// Feature: NormalizePad op
/// Description: Test invalid input parameters at IR level
/// Expectation: Throw correct error and message
TEST_F(MindDataTestIRVision, TestNormalizePadFail) {
  MS_LOG(INFO) << "Doing MindDataTestIRVision-TestNormalizePadFail with invalid parameters.";

  Status rc;
  std::vector<float> mean;
  std::vector<float> std;

  // std value at 0.0
  mean = {121.0, 115.0, 100.0};
  std = {0.0, 68.0, 71.0};
  std::shared_ptr<TensorOperation> normalizepad1 =
    std::make_shared<vision::NormalizePadOperation>(mean, std, "float32", true);
  rc = normalizepad1->ValidateParams();
  EXPECT_ERROR(rc);

  // normalizepad with 2 values (not 3 values) for mean
  mean = {121.0, 115.0};
  std = {70.0, 68.0, 71.0};
  std::shared_ptr<TensorOperation> normalizepad2 =
    std::make_shared<vision::NormalizePadOperation>(mean, std, "float32", true);
  rc = normalizepad2->ValidateParams();
  EXPECT_ERROR(rc);

  // normalizepad with 2 values (not 3 values) for standard deviation
  mean = {121.0, 115.0, 100.0};
  std = {68.0, 71.0};
  std::shared_ptr<TensorOperation> normalizepad3 =
    std::make_shared<vision::NormalizePadOperation>(mean, std, "float32", true);
  rc = normalizepad3->ValidateParams();
  EXPECT_ERROR(rc);

  // normalizepad with invalid dtype
  mean = {121.0, 115.0, 100.0};
  std = {68.0, 71.0, 71.0};
  std::shared_ptr<TensorOperation> normalizepad4 =
    std::make_shared<vision::NormalizePadOperation>(mean, std, "123", true);
  rc = normalizepad4->ValidateParams();
  EXPECT_ERROR(rc);
}

/// Feature: Rescale op
/// Description: Test Rescale op with negative rescale parameter
/// Expectation: Throw correct error and message
TEST_F(MindDataTestIRVision, TestRescaleFail) {
  MS_LOG(INFO) << "Doing MindDataTestIRVision-TestRescaleFail with invalid params.";

  Status rc;

  // incorrect negative rescale parameter
  auto rescale = std::make_shared<vision::RescaleOperation>(-1.0, 0.0);
  rc = rescale->ValidateParams();
  EXPECT_ERROR(rc);
}

/// Feature: Resize op
/// Description: Test Resize op with invalid resize values
/// Expectation: Throw correct error and message
TEST_F(MindDataTestIRVision, TestResizeFail) {
  MS_LOG(INFO) << "Doing MindDataTestIRVision-TestResize with invalid parameters.";

  Status rc;

  // negative resize value
  auto resize_op1 = std::make_shared<vision::ResizeOperation>(
    std::vector<int32_t>{30, -30}, InterpolationMode::kLinear);
  rc = resize_op1->ValidateParams();
  EXPECT_ERROR(rc);

  // zero resize value
  auto resize_op2 = std::make_shared<vision::ResizeOperation>(
    std::vector<int32_t>{0, 30}, InterpolationMode::kLinear);
  rc = resize_op2->ValidateParams();
  EXPECT_ERROR(rc);

  // resize with 3 values
  auto resize_op3 = std::make_shared<vision::ResizeOperation>(
    std::vector<int32_t>{30, 20, 10}, InterpolationMode::kLinear);
  rc = resize_op3->ValidateParams();
  EXPECT_ERROR(rc);
}

/// Feature: ResizeWithBBox op
/// Description: Test ResizeWithBBox op with invalid resize values
/// Expectation: Throw correct error and message
TEST_F(MindDataTestIRVision, TestResizeWithBBoxFail) {
  MS_LOG(INFO) << "Doing MindDataTestIRVision-TestResizeWithBBoxFail with invalid parameters.";

  Status rc;

  // Testing negative resize value
  auto resize_with_bbox_op = std::make_shared<vision::ResizeWithBBoxOperation>(
    std::vector<int32_t>{10, -10}, InterpolationMode::kLinear);
  EXPECT_NE(resize_with_bbox_op, nullptr);
  rc = resize_with_bbox_op->ValidateParams();
  EXPECT_ERROR(rc);

  // Testing negative resize value
  auto resize_with_bbox_op1 = std::make_shared<vision::ResizeWithBBoxOperation>(
    std::vector<int32_t>{-10}, InterpolationMode::kLinear);
  EXPECT_NE(resize_with_bbox_op1, nullptr);
  rc = resize_with_bbox_op1->ValidateParams();
  EXPECT_ERROR(rc);

  // Testing zero resize value
  auto resize_with_bbox_op2 = std::make_shared<vision::ResizeWithBBoxOperation>(
    std::vector<int32_t>{0, 10}, InterpolationMode::kLinear);
  EXPECT_NE(resize_with_bbox_op2, nullptr);
  rc = resize_with_bbox_op2->ValidateParams();
  EXPECT_ERROR(rc);

  // Testing resize with 3 values
  auto resize_with_bbox_op3 = std::make_shared<vision::ResizeWithBBoxOperation>(
    std::vector<int32_t>{10, 10, 10}, InterpolationMode::kLinear);
  EXPECT_NE(resize_with_bbox_op3, nullptr);
  rc = resize_with_bbox_op3->ValidateParams();
  EXPECT_ERROR(rc);
}

/// Feature: Vision operation name
/// Description: Create a vision tensor operation and check the name
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestIRVision, TestVisionOperationName) {
  MS_LOG(INFO) << "Doing MindDataTestIRVision-TestVisionOperationName.";

  std::string correct_name;

  // Create object for the tensor op, and check the name
  std::shared_ptr<TensorOperation> random_vertical_flip_op = std::make_shared<vision::RandomVerticalFlipOperation>(0.5);
  correct_name = "RandomVerticalFlip";
  EXPECT_EQ(correct_name, random_vertical_flip_op->Name());
}
