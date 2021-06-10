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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_SLICE_PATCHES_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_SLICE_PATCHES_OP_H_

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/kernels/tensor_op.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
class SlicePatchesOp : public TensorOp {
 public:
  // Default values, also used by python_bindings.cc
  static const int32_t kDefNumH;
  static const int32_t kDefNumW;
  static const uint8_t kDefFillV;
  static const SliceMode kDefSliceMode;

  SlicePatchesOp(int32_t num_height = kDefNumH, int32_t num_width = kDefNumW, SliceMode slice_mode = kDefSliceMode,
                 uint8_t fill_value = kDefFillV);

  ~SlicePatchesOp() override = default;

  void Print(std::ostream &out) const override {
    out << Name() << " patches number on height: " << num_height_ << ", patches number on width: " << num_width_;
  }

  Status Compute(const TensorRow &input, TensorRow *output) override;

  std::string Name() const override { return kSlicePatchesOp; }

 protected:
  int32_t num_height_;    // number of patches on height axis
  int32_t num_width_;     // number of patches on width axis
  SliceMode slice_mode_;  // PadModel, DropModel
  uint8_t fill_value_;    // border width in number of pixels in right and bottom direction
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_SLICE_PATCHES_OP_H_
