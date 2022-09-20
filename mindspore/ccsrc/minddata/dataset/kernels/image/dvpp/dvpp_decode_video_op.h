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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_DVPP_DVPP_DECODE_VIDEO_OP_H
#define MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_DVPP_DVPP_DECODE_VIDEO_OP_H

#include <memory>
#include <string>
#include <vector>

#include "mindspore/core/utils/log_adapter.h"
#include "minddata/dataset/core/data_type.h"
#include "minddata/dataset/core/device_resource.h"
#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/kernels/image/dvpp/acl_adapter.h"
#include "minddata/dataset/kernels/tensor_op.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
class DvppDecodeVideoOp : public TensorOp {
 public:
  // Default values
  static const VdecOutputFormat kDefVdecOutputFormat;
  static const char kDefOutput[];

  DvppDecodeVideoOp(uint32_t width, uint32_t height, VdecStreamFormat type,
                    VdecOutputFormat out_format = kDefVdecOutputFormat, const std::string output = kDefOutput)
      : width_(width), height_(height), format_(out_format), en_type_(type), output_(output) {}

  /// \brief Destructor
  ~DvppDecodeVideoOp() = default;

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  Status OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) override;

  std::string Name() const override { return kDvppDecodeVideoOp; }

 private:
  uint32_t width_;

  uint32_t height_;

  /* 1：YUV420 semi-planner（nv12）
     2：YVU420 semi-planner（nv21）
  */
  VdecOutputFormat format_;

  /* 0：H265 main level
   * 1：H264 baseline level
   * 2：H264 main level
   * 3：H264 high level
   */
  VdecStreamFormat en_type_;

  std::string output_;
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_DVPP_DVPP_DECODE_VIDEO_OP_H
