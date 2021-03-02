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
#include "minddata/dataset/kernels/ir/vision/vision_ir.h"

using namespace mindspore::dataset;

class MindDataTestIRVision : public UT::DatasetOpTesting {
 public:
  MindDataTestIRVision() = default;
};

TEST_F(MindDataTestIRVision, TestAutoContrastFail1) {
  MS_LOG(INFO) << "Doing MindDataTestIRVision-TestAutoContrastFail1.";

  // Testing invalid cutoff < 0
  std::shared_ptr<TensorOperation> auto_contrast1(new vision::AutoContrastOperation(-1.0, {}));
  Status rc1 = auto_contrast1->ValidateParams();
  EXPECT_ERROR(rc1);

  // Testing invalid cutoff > 100
  std::shared_ptr<TensorOperation> auto_contrast2(new vision::AutoContrastOperation(110.0, {10, 20}));
  Status rc2 = auto_contrast2->ValidateParams();
  EXPECT_ERROR(rc2);
}

TEST_F(MindDataTestIRVision, TestCenterCropFail) {
  MS_LOG(INFO) << "Doing MindDataTestIRVision-TestCenterCrop with invalid parameters.";

  Status rc;

  // center crop height value negative
  std::shared_ptr<TensorOperation> center_crop1(new vision::CenterCropOperation({-32, 32}));
  rc = center_crop1->ValidateParams();
  EXPECT_ERROR(rc);

  // center crop width value negative
  std::shared_ptr<TensorOperation> center_crop2(new vision::CenterCropOperation({32, -32}));
  rc = center_crop2->ValidateParams();
  EXPECT_ERROR(rc);

  // 0 value would result in nullptr
  std::shared_ptr<TensorOperation> center_crop3(new vision::CenterCropOperation({0, 32}));
  rc = center_crop3->ValidateParams();
  EXPECT_ERROR(rc);

  // center crop with 3 values
  std::shared_ptr<TensorOperation> center_crop4(new vision::CenterCropOperation({10, 20, 30}));
  rc = center_crop4->ValidateParams();
  EXPECT_ERROR(rc);
}

TEST_F(MindDataTestIRVision, TestCropFail) {
  MS_LOG(INFO) << "Doing MindDataTestIRVision-TestCrop with invalid parameters.";

  Status rc;

  // wrong width
  std::shared_ptr<TensorOperation> crop1(new vision::CropOperation({0, 0}, {32, -32}));
  rc = crop1->ValidateParams();
  EXPECT_ERROR(rc);

  // wrong height
  std::shared_ptr<TensorOperation> crop2(new vision::CropOperation({0, 0}, {-32, -32}));
  rc = crop2->ValidateParams();
  EXPECT_ERROR(rc);

  // zero height
  std::shared_ptr<TensorOperation> crop3(new vision::CropOperation({0, 0}, {0, 32}));
  rc = crop3->ValidateParams();
  EXPECT_ERROR(rc);

  // negative coordinates
  std::shared_ptr<TensorOperation> crop4(new vision::CropOperation({-1, 0}, {32, 32}));
  rc = crop4->ValidateParams();
  EXPECT_ERROR(rc);
}

TEST_F(MindDataTestIRVision, TestCutOutFail1) {
  MS_LOG(INFO) << "Doing MindDataTestIRVision-TestCutOutFail1 with invalid parameters.";

  Status rc;

  // Create object for the tensor op
  // Invalid negative length
  std::shared_ptr<TensorOperation> cutout_op = std::make_shared<vision::CutOutOperation>(-10, 1);
  rc = cutout_op->ValidateParams();
  EXPECT_ERROR(rc);

  // Invalid negative number of patches
  cutout_op = std::make_shared<vision::CutOutOperation>(10, -1);
  rc = cutout_op->ValidateParams();
  EXPECT_ERROR(rc);
}

TEST_F(MindDataTestIRVision, TestCutOutFail2) {
  MS_LOG(INFO) << "Doing MindDataTestIRVision-TestCutOutFail2 with invalid params, boundary cases.";

  Status rc;

  // Create object for the tensor op
  // Invalid zero length
  std::shared_ptr<TensorOperation> cutout_op = std::make_shared<vision::CutOutOperation>(0, 1);
  rc = cutout_op->ValidateParams();
  EXPECT_ERROR(rc);

  // Invalid zero number of patches
  cutout_op = std::make_shared<vision::CutOutOperation>(10, 0);
  rc = cutout_op->ValidateParams();
  EXPECT_ERROR(rc);
}

TEST_F(MindDataTestIRVision, TestNormalizeFail) {
  MS_LOG(INFO) << "Doing MindDataTestIRVision-TestNormalizeFail with invalid parameters.";

  Status rc;

  // std value 0.0 out of range
  std::shared_ptr<TensorOperation> normalize1(new vision::NormalizeOperation({121.0, 115.0, 100.0}, {0.0, 68.0, 71.0}));
  rc = normalize1->ValidateParams();
  EXPECT_ERROR(rc);

  // std value 256.0 out of range
  std::shared_ptr<TensorOperation> normalize2(
    new vision::NormalizeOperation({121.0, 10.0, 100.0}, {256.0, 68.0, 71.0}));
  rc = normalize2->ValidateParams();
  EXPECT_ERROR(rc);

  // mean value 256.0 out of range
  std::shared_ptr<TensorOperation> normalize3(new vision::NormalizeOperation({256.0, 0.0, 100.0}, {70.0, 68.0, 71.0}));
  rc = normalize3->ValidateParams();
  EXPECT_ERROR(rc);

  // mean value 0.0 out of range
  std::shared_ptr<TensorOperation> normalize4(new vision::NormalizeOperation({-1.0, 0.0, 100.0}, {70.0, 68.0, 71.0}));
  rc = normalize4->ValidateParams();
  EXPECT_ERROR(rc);

  // normalize with 2 values (not 3 values) for mean
  std::shared_ptr<TensorOperation> normalize5(new vision::NormalizeOperation({121.0, 115.0}, {70.0, 68.0, 71.0}));
  rc = normalize5->ValidateParams();
  EXPECT_ERROR(rc);

  // normalize with 2 values (not 3 values) for standard deviation
  std::shared_ptr<TensorOperation> normalize6(new vision::NormalizeOperation({121.0, 115.0, 100.0}, {68.0, 71.0}));
  rc = normalize6->ValidateParams();
  EXPECT_ERROR(rc);
}

TEST_F(MindDataTestIRVision, TestNormalizePadFail) {
  MS_LOG(INFO) << "Doing MindDataTestIRVision-TestNormalizePadFail with invalid parameters.";

  Status rc;

  // std value at 0.0
  std::shared_ptr<TensorOperation> normalizepad1(
    new vision::NormalizePadOperation({121.0, 115.0, 100.0}, {0.0, 68.0, 71.0}, "float32"));
  rc = normalizepad1->ValidateParams();
  EXPECT_ERROR(rc);

  // normalizepad with 2 values (not 3 values) for mean
  std::shared_ptr<TensorOperation> normalizepad2(
    new vision::NormalizePadOperation({121.0, 115.0}, {70.0, 68.0, 71.0}, "float32"));
  rc = normalizepad2->ValidateParams();
  EXPECT_ERROR(rc);

  // normalizepad with 2 values (not 3 values) for standard deviation
  std::shared_ptr<TensorOperation> normalizepad3(
    new vision::NormalizePadOperation({121.0, 115.0, 100.0}, {68.0, 71.0}, "float32"));
  rc = normalizepad3->ValidateParams();
  EXPECT_ERROR(rc);

  // normalizepad with invalid dtype
  std::shared_ptr<TensorOperation> normalizepad4(
    new vision::NormalizePadOperation({121.0, 115.0, 100.0}, {68.0, 71.0, 71.0}, "123"));
  rc = normalizepad4->ValidateParams();
  EXPECT_ERROR(rc);
}

TEST_F(MindDataTestIRVision, TestRescaleFail) {
  MS_LOG(INFO) << "Doing MindDataTestIRVision-TestRescaleFail with invalid params.";

  Status rc;

  // incorrect negative rescale parameter
  std::shared_ptr<TensorOperation> rescale(new vision::RescaleOperation(-1.0, 0.0));
  rc = rescale->ValidateParams();
  EXPECT_ERROR(rc);
}

TEST_F(MindDataTestIRVision, TestResizeFail) {
  MS_LOG(INFO) << "Doing MindDataTestIRVision-TestResize with invalid parameters.";

  Status rc;

  // negative resize value
  std::shared_ptr<TensorOperation> resize_op1(new vision::ResizeOperation({30, -30}, InterpolationMode::kLinear));
  rc = resize_op1->ValidateParams();
  EXPECT_ERROR(rc);

  // zero resize value
  std::shared_ptr<TensorOperation> resize_op2(new vision::ResizeOperation({0, 30}, InterpolationMode::kLinear));
  rc = resize_op2->ValidateParams();
  EXPECT_ERROR(rc);

  // resize with 3 values
  std::shared_ptr<TensorOperation> resize_op3(new vision::ResizeOperation({30, 20, 10}, InterpolationMode::kLinear));
  rc = resize_op3->ValidateParams();
  EXPECT_ERROR(rc);
}

TEST_F(MindDataTestIRVision, TestResizeWithBBoxFail) {
  MS_LOG(INFO) << "Doing MindDataTestIRVision-TestResizeWithBBoxFail with invalid parameters.";

  Status rc;

  // Testing negative resize value
  std::shared_ptr<TensorOperation> resize_with_bbox_op(
    new vision::ResizeWithBBoxOperation({10, -10}, InterpolationMode::kLinear));
  EXPECT_NE(resize_with_bbox_op, nullptr);
  rc = resize_with_bbox_op->ValidateParams();
  EXPECT_ERROR(rc);

  // Testing negative resize value
  std::shared_ptr<TensorOperation> resize_with_bbox_op1(
    new vision::ResizeWithBBoxOperation({-10}, InterpolationMode::kLinear));
  EXPECT_NE(resize_with_bbox_op1, nullptr);
  rc = resize_with_bbox_op1->ValidateParams();
  EXPECT_ERROR(rc);

  // Testing zero resize value
  std::shared_ptr<TensorOperation> resize_with_bbox_op2(
    new vision::ResizeWithBBoxOperation({0, 10}, InterpolationMode::kLinear));
  EXPECT_NE(resize_with_bbox_op2, nullptr);
  rc = resize_with_bbox_op2->ValidateParams();
  EXPECT_ERROR(rc);

  // Testing resize with 3 values
  std::shared_ptr<TensorOperation> resize_with_bbox_op3(
    new vision::ResizeWithBBoxOperation({10, 10, 10}, InterpolationMode::kLinear));
  EXPECT_NE(resize_with_bbox_op3, nullptr);
  rc = resize_with_bbox_op3->ValidateParams();
  EXPECT_ERROR(rc);
}

TEST_F(MindDataTestIRVision, TestSoftDvppDecodeRandomCropResizeJpegFail) {
  MS_LOG(INFO) << "Doing MindDataTestIRVision-TestSoftDvppDecodeRandomCropResizeJpegFail with incorrect parameters.";

  Status rc;

  // SoftDvppDecodeRandomCropResizeJpeg: size must only contain positive integers
  std::shared_ptr<TensorOperation> soft_dvpp_decode_random_crop_resize_jpeg1(
    new vision::SoftDvppDecodeRandomCropResizeJpegOperation({-500, 600}, {0.08, 1.0}, {3. / 4., 4. / 3.}, 10));
  rc = soft_dvpp_decode_random_crop_resize_jpeg1->ValidateParams();
  EXPECT_ERROR(rc);

  // SoftDvppDecodeRandomCropResizeJpeg: size must only contain positive integers
  std::shared_ptr<TensorOperation> soft_dvpp_decode_random_crop_resize_jpeg2(
    new vision::SoftDvppDecodeRandomCropResizeJpegOperation({-500}, {0.08, 1.0}, {3. / 4., 4. / 3.}, 10));
  rc = soft_dvpp_decode_random_crop_resize_jpeg2->ValidateParams();
  EXPECT_ERROR(rc);

  // SoftDvppDecodeRandomCropResizeJpeg: size must be a vector of one or two values
  std::shared_ptr<TensorOperation> soft_dvpp_decode_random_crop_resize_jpeg3(
    new vision::SoftDvppDecodeRandomCropResizeJpegOperation({500, 600, 700}, {0.08, 1.0}, {3. / 4., 4. / 3.}, 10));
  rc = soft_dvpp_decode_random_crop_resize_jpeg3->ValidateParams();
  EXPECT_ERROR(rc);

  // SoftDvppDecodeRandomCropResizeJpeg: scale must be greater than or equal to 0
  std::shared_ptr<TensorOperation> soft_dvpp_decode_random_crop_resize_jpeg4(
    new vision::SoftDvppDecodeRandomCropResizeJpegOperation({500}, {-0.1, 0.9}, {3. / 4., 4. / 3.}, 1));
  rc = soft_dvpp_decode_random_crop_resize_jpeg4->ValidateParams();
  EXPECT_ERROR(rc);

  // SoftDvppDecodeRandomCropResizeJpeg: scale must be in the format of (min, max)
  std::shared_ptr<TensorOperation> soft_dvpp_decode_random_crop_resize_jpeg5(
    new vision::SoftDvppDecodeRandomCropResizeJpegOperation({500}, {0.6, 0.2}, {3. / 4., 4. / 3.}, 1));
  rc = soft_dvpp_decode_random_crop_resize_jpeg5->ValidateParams();
  EXPECT_ERROR(rc);

  // SoftDvppDecodeRandomCropResizeJpeg: scale must be a vector of two values
  std::shared_ptr<TensorOperation> soft_dvpp_decode_random_crop_resize_jpeg6(
    new vision::SoftDvppDecodeRandomCropResizeJpegOperation({500}, {0.5, 0.6, 0.7}, {3. / 4., 4. / 3.}, 1));
  rc = soft_dvpp_decode_random_crop_resize_jpeg6->ValidateParams();
  EXPECT_ERROR(rc);

  // SoftDvppDecodeRandomCropResizeJpeg: ratio must be greater than or equal to 0
  std::shared_ptr<TensorOperation> soft_dvpp_decode_random_crop_resize_jpeg7(
    new vision::SoftDvppDecodeRandomCropResizeJpegOperation({500}, {0.5, 0.9}, {-0.2, 0.4}, 5));
  rc = soft_dvpp_decode_random_crop_resize_jpeg7->ValidateParams();
  EXPECT_ERROR(rc);

  // SoftDvppDecodeRandomCropResizeJpeg: ratio must be in the format of (min, max)
  std::shared_ptr<TensorOperation> soft_dvpp_decode_random_crop_resize_jpeg8(
    new vision::SoftDvppDecodeRandomCropResizeJpegOperation({500}, {0.5, 0.9}, {0.4, 0.2}, 5));
  rc = soft_dvpp_decode_random_crop_resize_jpeg8->ValidateParams();
  EXPECT_ERROR(rc);
  // SoftDvppDecodeRandomCropResizeJpeg: ratio must be a vector of two values
  std::shared_ptr<TensorOperation> soft_dvpp_decode_random_crop_resize_jpeg9(
    new vision::SoftDvppDecodeRandomCropResizeJpegOperation({500}, {0.5, 0.9}, {0.1, 0.2, 0.3}, 5));
  rc = soft_dvpp_decode_random_crop_resize_jpeg9->ValidateParams();
  EXPECT_ERROR(rc);

  // SoftDvppDecodeRandomCropResizeJpeg: max_attempts must be greater than or equal to 1
  std::shared_ptr<TensorOperation> soft_dvpp_decode_random_crop_resize_jpeg10(
    new vision::SoftDvppDecodeRandomCropResizeJpegOperation({500}, {0.5, 0.9}, {0.1, 0.2}, 0));
  rc = soft_dvpp_decode_random_crop_resize_jpeg10->ValidateParams();
  EXPECT_ERROR(rc);
}

TEST_F(MindDataTestIRVision, TestSoftDvppDecodeResizeJpegFail) {
  MS_LOG(INFO) << "Doing MindDataTestIRVision-TestSoftDvppDecodeResizeJpegFail with incorrect size.";

  Status rc;

  // SoftDvppDecodeResizeJpeg: size must be a vector of one or two values
  std::shared_ptr<TensorOperation> soft_dvpp_decode_resize_jpeg_op1(new vision::SoftDvppDecodeResizeJpegOperation({}));
  rc = soft_dvpp_decode_resize_jpeg_op1->ValidateParams();
  EXPECT_ERROR(rc);

  // SoftDvppDecodeResizeJpeg: size must be a vector of one or two values
  std::shared_ptr<TensorOperation> soft_dvpp_decode_resize_jpeg_op2(
    new vision::SoftDvppDecodeResizeJpegOperation({1, 2, 3}));
  rc = soft_dvpp_decode_resize_jpeg_op2->ValidateParams();
  EXPECT_ERROR(rc);

  // SoftDvppDecodeResizeJpeg: size must only contain positive integers
  std::shared_ptr<TensorOperation> soft_dvpp_decode_resize_jpeg_op3(
    new vision::SoftDvppDecodeResizeJpegOperation({20, -20}));
  rc = soft_dvpp_decode_resize_jpeg_op3->ValidateParams();
  EXPECT_ERROR(rc);

  // SoftDvppDecodeResizeJpeg: size must only contain positive integers
  std::shared_ptr<TensorOperation> soft_dvpp_decode_resize_jpeg_op4(new vision::SoftDvppDecodeResizeJpegOperation({0}));
  rc = soft_dvpp_decode_resize_jpeg_op4->ValidateParams();
  EXPECT_ERROR(rc);
}

TEST_F(MindDataTestIRVision, TestVisionOperationName) {
  MS_LOG(INFO) << "Doing MindDataTestIRVision-TestVisionOperationName.";

  std::string correct_name;

  // Create object for the tensor op, and check the name
  std::shared_ptr<TensorOperation> random_vertical_flip_op = std::make_shared<vision::RandomVerticalFlipOperation>(0.5);
  correct_name = "RandomVerticalFlip";
  EXPECT_EQ(correct_name, random_vertical_flip_op->Name());

  // Create object for the tensor op, and check the name
  std::shared_ptr<TensorOperation> softDvpp_decode_resize_jpeg_op(
    new vision::SoftDvppDecodeResizeJpegOperation({1, 1}));
  correct_name = "SoftDvppDecodeResizeJpeg";
  EXPECT_EQ(correct_name, softDvpp_decode_resize_jpeg_op->Name());
}
