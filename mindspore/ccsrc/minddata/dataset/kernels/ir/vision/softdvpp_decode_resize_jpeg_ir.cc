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
#include <algorithm>

#include "minddata/dataset/kernels/ir/vision/softdvpp_decode_resize_jpeg_ir.h"

#ifndef ENABLE_ANDROID
#include "minddata/dataset/kernels/image/soft_dvpp/soft_dvpp_decode_resize_jpeg_op.h"
#endif

#include "minddata/dataset/kernels/ir/validators.h"

namespace mindspore {
namespace dataset {

namespace vision {
#ifndef ENABLE_ANDROID

// SoftDvppDecodeResizeJpegOperation
SoftDvppDecodeResizeJpegOperation::SoftDvppDecodeResizeJpegOperation(std::vector<int32_t> size) : size_(size) {}

SoftDvppDecodeResizeJpegOperation::~SoftDvppDecodeResizeJpegOperation() = default;

std::string SoftDvppDecodeResizeJpegOperation::Name() const { return kSoftDvppDecodeResizeJpegOperation; }

Status SoftDvppDecodeResizeJpegOperation::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateVectorSize("SoftDvppDecodeResizeJpeg", size_));
  for (int32_t i = 0; i < size_.size(); i++) {
    if (size_[i] % 2 == 1) {
      std::string err_msg = "SoftDvppDecodeResizeJpeg: size[" + std::to_string(i) +
                            "] must be even values, got: " + std::to_string(size_[i]);
      MS_LOG(ERROR) << err_msg;
      RETURN_STATUS_SYNTAX_ERROR(err_msg);
    }
  }
  return Status::OK();
}

std::shared_ptr<TensorOp> SoftDvppDecodeResizeJpegOperation::Build() {
  // If size is a single value, the smaller edge of the image will be
  // resized to this value with the same image aspect ratio.
  int32_t height = size_[0];
  int32_t width = 0;

  // User specified the width value.
  if (size_.size() == 2) {
    width = size_[1];
  }
  std::shared_ptr<SoftDvppDecodeResizeJpegOp> tensor_op = std::make_shared<SoftDvppDecodeResizeJpegOp>(height, width);
  return tensor_op;
}

Status SoftDvppDecodeResizeJpegOperation::to_json(nlohmann::json *out_json) {
  (*out_json)["size"] = size_;
  return Status::OK();
}

#endif

}  // namespace vision
}  // namespace dataset
}  // namespace mindspore
