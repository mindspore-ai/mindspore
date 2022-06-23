/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd

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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_CUT_OUT_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_CUT_OUT_OP_H_

#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/kernels/image/image_utils.h"
#include "minddata/dataset/kernels/tensor_op.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
class CutOutOp : public TensorOp {
 public:
  // Default values, also used by python_bindings.cc
  static const bool kDefRandomColor;

  // Constructor for CutOutOp
  // @param box_height box height
  // @param box_width box_width
  // @param num_patches how many patches to erase from image
  // @param random_color boolean value to indicate fill patch with random color
  // @param fill_colors value for the color to fill patch with
  // @param is_hwc Check if input is HWC/CHW format
  // @note maybe using unsigned long int isn't the best here according to our coding rules
  CutOutOp(int32_t box_height, int32_t box_width, int32_t num_patches, bool random_color = kDefRandomColor,
           std::vector<uint8_t> fill_colors = {}, bool is_hwc = true);

  ~CutOutOp() override = default;

  void Print(std::ostream &out) const override {
    out << "CutOut:: box_height: " << box_height_ << " box_width: " << box_width_ << " num_patches: " << num_patches_;
  }

  // Overrides the base class compute function
  // Calls the erase function in ImageUtils, this function takes an input tensor
  // and overwrites some of its data using openCV, the output memory is manipulated to contain the result
  // @return Status The status code returned
  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  std::string Name() const override { return kCutOutOp; }

 private:
  std::mt19937 rnd_;
  int32_t box_height_;
  int32_t box_width_;
  int32_t num_patches_;
  bool random_color_;
  std::vector<uint8_t> fill_colors_;
  bool is_hwc_;
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_CUT_OUT_OP_H_
