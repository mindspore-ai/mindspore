/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_AUDIO_IR_KERNELS_MASK_ALONG_AXIS_IR_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_AUDIO_IR_KERNELS_MASK_ALONG_AXIS_IR_H_

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "include/api/status.h"
#include "minddata/dataset/include/dataset/constants.h"
#include "minddata/dataset/kernels/ir/tensor_operation.h"

namespace mindspore {
namespace dataset {
namespace audio {
constexpr char kMaskAlongAxisOperation[] = "MaskAlongAxis";

class MaskAlongAxisOperation : public TensorOperation {
 public:
  MaskAlongAxisOperation(int32_t mask_start, int32_t mask_width, float mask_value, int32_t axis);

  ~MaskAlongAxisOperation() override;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override;

  Status to_json(nlohmann::json *out_json) override;

 private:
  int32_t mask_start_;
  int32_t mask_width_;
  float mask_value_;
  int32_t axis_;
};  // class MaskAlongAxisOperation
}  // namespace audio
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_AUDIO_IR_KERNELS_MASK_ALONG_AXIS_IR_H_
