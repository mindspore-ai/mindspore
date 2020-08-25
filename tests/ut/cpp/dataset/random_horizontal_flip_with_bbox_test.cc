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
#include "common/bboxop_common.h"
#include "minddata/dataset/kernels/image/random_horizontal_flip_with_bbox_op.h"
#include "utils/log_adapter.h"

using namespace mindspore::dataset;
using mindspore::MsLogLevel::INFO;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::LogStream;

const bool kSaveExpected = false;
const char kOpName[] = "random_horizontal_flip_with_bbox_c";

class MindDataTestRandomHorizontalFlipWithBBoxOp : public UT::CVOP::BBOXOP::BBoxOpCommon {
 protected:
  MindDataTestRandomHorizontalFlipWithBBoxOp() : UT::CVOP::BBOXOP::BBoxOpCommon() {}
};

TEST_F(MindDataTestRandomHorizontalFlipWithBBoxOp, TestOp) {
  MS_LOG(INFO) << "Doing testRandomHorizontalFlipWithBBox.";
  TensorTable results;
  std::unique_ptr<RandomHorizontalFlipWithBBoxOp> op(new RandomHorizontalFlipWithBBoxOp(1));
  for (const auto &row: images_and_annotations_) {
    TensorRow output_row;
    Status s = op->Compute(row, &output_row);
    EXPECT_TRUE(s.IsOk());
    results.push_back(output_row);
  }
  if (kSaveExpected) {
    SaveImagesWithAnnotations(FileType::kExpected, std::string(kOpName), results);
  }
  SaveImagesWithAnnotations(FileType::kActual , std::string(kOpName), results);
  if (!kSaveExpected) {
    CompareActualAndExpected(std::string(kOpName));
  }
}
