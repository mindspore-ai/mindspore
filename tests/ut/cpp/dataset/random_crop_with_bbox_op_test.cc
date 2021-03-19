/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "common/bboxop_common.h"
#include "minddata/dataset/kernels/image/random_crop_with_bbox_op.h"
#include "utils/log_adapter.h"

#include "minddata/dataset/core/config_manager.h"
#include "minddata/dataset/core/global_context.h"

using namespace mindspore::dataset;
using mindspore::LogStream;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::MsLogLevel::INFO;

const bool kSaveExpected = false;
const char kOpName[] = "random_crop_with_bbox_c";

class MindDataTestRandomCropWithBBoxOp : public UT::CVOP::BBOXOP::BBoxOpCommon {
 protected:
  MindDataTestRandomCropWithBBoxOp() : BBoxOpCommon() {}
  TensorRow output_tensor_row_;
};

TEST_F(MindDataTestRandomCropWithBBoxOp, TestOp1) {
  MS_LOG(INFO) << "Doing testRandomCropWithBBoxOp1.";
  TensorTable results;
  unsigned int crop_height = 128;
  unsigned int crop_width = 128;
  // setting seed here
  uint32_t current_seed = GlobalContext::config_manager()->seed();
  GlobalContext::config_manager()->set_seed(327362);
  std::unique_ptr<RandomCropWithBBoxOp> op(
    new RandomCropWithBBoxOp(crop_height, crop_width, 0, 0, 0, 0, BorderType::kConstant, false));
  for (auto tensor_row_ : images_and_annotations_) {
    Status s = op->Compute(tensor_row_, &output_tensor_row_);
    size_t actual = 0;
    if (s == Status::OK()) {
      TensorShape get_shape = output_tensor_row_[0]->shape();
      actual = get_shape[0] * get_shape[1] * get_shape[2];
      results.push_back(output_tensor_row_);
    }
    EXPECT_EQ(actual, crop_height * crop_width * 3);
    EXPECT_EQ(s, Status::OK());
    EXPECT_EQ(4, output_tensor_row_[1]->shape()[1]);  // check for existence of 4 columns
    // Compare Code
    if (kSaveExpected) {
      SaveImagesWithAnnotations(FileType::kExpected, std::string(kOpName), results);
    }
    SaveImagesWithAnnotations(FileType::kActual, std::string(kOpName), results);
    if (!kSaveExpected) {
      CompareActualAndExpected(std::string(kOpName));
    }
    GlobalContext::config_manager()->set_seed(current_seed);
  }
  MS_LOG(INFO) << "testRandomCropWithBBoxOp1 end.";
}

TEST_F(MindDataTestRandomCropWithBBoxOp, TestOp2) {
  MS_LOG(INFO) << "Doing testRandomCropWithBBoxOp2.";
  // Crop params
  unsigned int crop_height = 1280;
  unsigned int crop_width = 1280;
  // setting seed here to prevent random core dump
  uint32_t current_seed = GlobalContext::config_manager()->seed();
  GlobalContext::config_manager()->set_seed(327362);

  std::unique_ptr<RandomCropWithBBoxOp> op(
    new RandomCropWithBBoxOp(crop_height, crop_width, 513, 513, 513, 513, BorderType::kConstant, false));

  for (auto tensor_row_ : images_and_annotations_) {
    Status s = op->Compute(tensor_row_, &output_tensor_row_);
    size_t actual = 0;
    if (s == Status::OK()) {
      TensorShape get_shape = output_tensor_row_[0]->shape();
      actual = get_shape[0] * get_shape[1] * get_shape[2];
    }
    EXPECT_EQ(actual, crop_height * crop_width * 3);
    EXPECT_EQ(s, Status::OK());
    EXPECT_EQ(4, output_tensor_row_[1]->shape()[1]);  // check for existence of 4 columns
  }
  MS_LOG(INFO) << "testRandomCropWithBBoxOp2 end.";
  GlobalContext::config_manager()->set_seed(current_seed);
}

TEST_F(MindDataTestRandomCropWithBBoxOp, TestOp3) {
  MS_LOG(INFO) << "Doing testRandomCropWithBBoxOp3.";
  // Crop params
  unsigned int crop_height = 1280;
  unsigned int crop_width = 1280;
  // setting seed here to prevent random core dump
  uint32_t current_seed = GlobalContext::config_manager()->seed();
  GlobalContext::config_manager()->set_seed(327362);

  std::unique_ptr<RandomCropWithBBoxOp> op(new RandomCropWithBBoxOp(crop_height, crop_width, crop_height * 3 + 1,
                                                                    crop_height * 3 + 1, crop_width * 3 + 1,
                                                                    crop_width * 3 + 1, BorderType::kConstant, false));

  for (auto tensor_row_ : images_and_annotations_) {
    Status s = op->Compute(tensor_row_, &output_tensor_row_);
    EXPECT_TRUE(s.IsError());
    ASSERT_TRUE(s.StatusCode() == StatusCode::kMDUnexpectedError);
  }
  MS_LOG(INFO) << "testRandomCropWithBBoxOp3 end.";
  GlobalContext::config_manager()->set_seed(current_seed);
}
