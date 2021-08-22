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
#ifndef TESTS_DATASET_UT_CORE_COMMON_DE_UT_CVOP_COMMON_H_
#define TESTS_DATASET_UT_CORE_COMMON_DE_UT_CVOP_COMMON_H_

#include <memory>
#include <string>
#include "common.h"
#include "minddata/dataset/kernels/image/image_utils.h"

namespace UT {
namespace CVOP {
using namespace mindspore::dataset;

class CVOpCommon : public Common {
 public:
  enum OperatorType {
    kResizeArea,
    kResizeBilinear,
    kRescale,
    kFlipVertical,
    kFlipHorizontal,
    kDecode,
    kChannelSwap,
    kChangeMode,
    kRandomSolarize,
    kTemplate,
    kCrop,
    kRandomSharpness,
    kInvert,
    kRandomAffine,
    kRandomPosterize,
    kAutoContrast,
    kAdjustGamma,
    kEqualize
  };

  CVOpCommon();

  ~CVOpCommon();

  void SetUp();

  std::string GetFilename();

  void GetInputImage(std::string filename);

  void Save(const std::shared_ptr<Tensor> &tensor, std::string filename);

  std::string GetJPGStr(const cv::Mat &image);

  bool CompareCVMat(const cv::Mat &actual, const cv::Mat &expect, OperatorType type);

  void CheckImageShapeAndData(const std::shared_ptr<Tensor> &output_tensor, OperatorType type);

  std::string filename_;
  cv::Mat raw_cv_image_;

  std::shared_ptr<Tensor> raw_input_tensor_;
  std::shared_ptr<Tensor> input_tensor_;
};
}  // namespace CVOP
}  // namespace UT
#endif  // TESTS_DATASET_UT_CORE_COMMON_DE_UT_CVOP_COMMON_H_
