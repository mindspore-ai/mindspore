/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include "common/common.h"
#include "common/cvop_common.h"
#include "dataset/kernels/image/normalize_op.h"
#include "dataset/core/cv_tensor.h"
#include "utils/log_adapter.h"
#include <opencv2/opencv.hpp>

using namespace mindspore::dataset;
using mindspore::MsLogLevel::INFO;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::LogStream;

class MindDataTestNormalizeOP : public UT::CVOP::CVOpCommon {
 public:
  MindDataTestNormalizeOP() : CVOpCommon() {}
};

TEST_F(MindDataTestNormalizeOP, TestOp) {
  MS_LOG(INFO) << "Doing TestNormalizeOp::TestOp2.";
  std::shared_ptr<Tensor> output_tensor;

  // Numbers are from the resnet50 model implementation
  float mean[3] = {121.0, 115.0, 100.0};
  float std[3] = {70.0, 68.0, 71.0};

  // Normalize Op
  std::unique_ptr<NormalizeOp> op(new NormalizeOp(mean[0], mean[1], mean[2], std[0], std[1], std[2]));
  EXPECT_TRUE(op->OneToOne());
  Status s = op->Compute(input_tensor_, &output_tensor);
  EXPECT_TRUE(s.IsOk());

  std::string output_filename = GetFilename();
  output_filename.replace(output_filename.end() - 8, output_filename.end(), "imagefolder/normalizeOpOut.yml");

  std::shared_ptr<CVTensor> p = CVTensor::AsCVTensor(output_tensor);
  cv::Mat cv_output_image;
  cv_output_image = p->mat();

  MS_LOG(DEBUG) << "Storing output file to : " << output_filename << std::endl;
  cv::FileStorage file(output_filename, cv::FileStorage::WRITE);
  file << "imageData" << cv_output_image;
}
