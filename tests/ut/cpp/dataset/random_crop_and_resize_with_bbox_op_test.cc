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
#include "minddata/dataset/kernels/image/random_crop_and_resize_with_bbox_op.h"
#include "utils/log_adapter.h"

#include "minddata/dataset/core/config_manager.h"
#include "minddata/dataset/core/global_context.h"

using namespace mindspore::dataset;
using mindspore::LogStream;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::MsLogLevel::INFO;

const bool kSaveExpected = false;
const char kOpName[] = "random_resized_crop_with_bbox_c";

class MindDataTestRandomCropAndResizeWithBBoxOp : public UT::CVOP::BBOXOP::BBoxOpCommon {
 protected:
  MindDataTestRandomCropAndResizeWithBBoxOp() : BBoxOpCommon() {}
};

TEST_F(MindDataTestRandomCropAndResizeWithBBoxOp, TestOp1) {
  MS_LOG(INFO) << "Doing testRandomCropAndResizeWithBBoxOp1.";
  // setting seed here
  uint32_t current_seed = GlobalContext::config_manager()->seed();
  GlobalContext::config_manager()->set_seed(327362);
  TensorRow output_tensor_row_;
  TensorTable results;
  int h_out = 1024;
  int w_out = 2048;
  float aspect_lb = 2;
  float aspect_ub = 2.5;
  float scale_lb = 0.2;
  float scale_ub = 2.0;
  auto op = std::make_unique<RandomCropAndResizeWithBBoxOp>(h_out, w_out, scale_lb, scale_ub, aspect_lb, aspect_ub);
  Status s;
  for (auto tensor_row_ : images_and_annotations_) {
    s = op->Compute(tensor_row_, &output_tensor_row_);
    EXPECT_TRUE(s.IsOk());
    results.push_back(output_tensor_row_);
  }
  if (kSaveExpected) {
    SaveImagesWithAnnotations(FileType::kExpected, std::string(kOpName), results);
  }
  SaveImagesWithAnnotations(FileType::kActual, std::string(kOpName), results);
  if (!kSaveExpected) {
    CompareActualAndExpected(std::string(kOpName));
  }
  GlobalContext::config_manager()->set_seed(current_seed);
}

TEST_F(MindDataTestRandomCropAndResizeWithBBoxOp, TestOp2) {
  MS_LOG(INFO) << "Doing testRandomCropAndResizeWithBBoxOp2.";
  // setting seed here to prevent random core dump
  uint32_t current_seed = GlobalContext::config_manager()->seed();
  GlobalContext::config_manager()->set_seed(327362);

  TensorRow output_tensor_row_;
  int h_out = 1024;
  int w_out = 2048;
  float aspect_lb = 1;
  float aspect_ub = 1.5;
  float scale_lb = 0.2;
  float scale_ub = 2.0;
  auto op = std::make_unique<RandomCropAndResizeWithBBoxOp>(h_out, w_out, scale_lb, scale_ub, aspect_lb, aspect_ub);
  Status s;
  for (auto tensor_row_ : images_and_annotations_) {
    s = op->Compute(tensor_row_, &output_tensor_row_);
    EXPECT_TRUE(s.IsOk());
  }
  GlobalContext::config_manager()->set_seed(current_seed);
}

TEST_F(MindDataTestRandomCropAndResizeWithBBoxOp, TestOp3) {
  MS_LOG(INFO) << "Doing testRandomCropAndResizeWithBBoxOp3.";
  TensorRow output_tensor_row_;
  int h_out = 1024;
  int w_out = 2048;
  float aspect_lb = 0.2;
  float aspect_ub = 3;
  float scale_lb = 0.2;
  float scale_ub = 2.0;
  auto op = std::make_unique<RandomCropAndResizeWithBBoxOp>(h_out, w_out, scale_lb, scale_ub, aspect_lb, aspect_ub);
  Status s;
  for (auto tensor_row_ : images_and_annotations_) {
    s = op->Compute(tensor_row_, &output_tensor_row_);
    EXPECT_TRUE(s.IsOk());
  }
  MS_LOG(INFO) << "testRandomCropAndResizeWithBBoxOp end.";
}
