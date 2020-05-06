/**
 * Copyright 2019 Huawei Technologies Co., Ltd

 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at

 * http://www.apache.org/licenses/LICENSE-2.0

 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/
#include "dataset/kernels/image/cut_out_op.h"

#include <random>

#include "dataset/core/config_manager.h"
#include "dataset/core/cv_tensor.h"
#include "dataset/kernels/image/image_utils.h"
#include "dataset/util/random.h"
#include "dataset/util/status.h"

namespace mindspore {
namespace dataset {
const bool CutOutOp::kDefRandomColor = false;
const uint8_t CutOutOp::kDefFillR = 0;
const uint8_t CutOutOp::kDefFillG = 0;
const uint8_t CutOutOp::kDefFillB = 0;

// constructor
CutOutOp::CutOutOp(int32_t box_height, int32_t box_width, int32_t num_patches, bool random_color, uint8_t fill_r,
                   uint8_t fill_g, uint8_t fill_b)
    : rnd_(GetSeed()),
      box_height_(box_height),
      box_width_(box_width),
      num_patches_(num_patches),
      random_color_(random_color),
      fill_r_(fill_r),
      fill_g_(fill_g),
      fill_b_(fill_b) {}

// main function call for cut out
Status CutOutOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  std::shared_ptr<CVTensor> inputCV = CVTensor::AsCVTensor(input);
  // cut out will clip the erasing area if the box is near the edge of the image and the boxes are black
  RETURN_IF_NOT_OK(Erase(inputCV, output, box_height_, box_width_, num_patches_, false, random_color_, &rnd_, fill_r_,
                         fill_g_, fill_b_));
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
