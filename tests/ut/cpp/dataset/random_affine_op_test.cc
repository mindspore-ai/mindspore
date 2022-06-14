/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/kernels/image/random_affine_op.h"
#include "minddata/dataset/core/cv_tensor.h"
#include "utils/log_adapter.h"

using namespace mindspore::dataset;

class MindDataTestRandomAffineOp : public UT::CVOP::CVOpCommon {
 public:
  MindDataTestRandomAffineOp() : CVOpCommon() {}
};

/// Feature: RandomAffine op
/// Description: Test RandomAffineOp basic usage and check OneToOne
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestRandomAffineOp, TestOp1) {
  MS_LOG(INFO) << "Doing testRandomAffineOp.";

  std::shared_ptr<Tensor> output_tensor;
  auto op = std::make_unique<RandomAffineOp>(std::vector<float>{30.0, 30.0}, std::vector<float>{0.0, 0.0, 0.0, 0.0}, 
                                                        std::vector<float>{2.0, 2.0},
                                                        std::vector<float>{10.0, 10.0, 20.0, 20.0}, 
                                                        InterpolationMode::kNearestNeighbour,
                                                        std::vector<uint8_t>{255, 0, 0});
  EXPECT_TRUE(op->OneToOne());
  Status s = op->Compute(input_tensor_, &output_tensor);
  EXPECT_TRUE(s.IsOk());
  CheckImageShapeAndData(output_tensor, kRandomAffine);
}
