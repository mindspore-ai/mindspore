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
SoftDvppDecodeResizeJpegOperation::SoftDvppDecodeResizeJpegOperation(const std::vector<int32_t> &size) : size_(size) {}

SoftDvppDecodeResizeJpegOperation::~SoftDvppDecodeResizeJpegOperation() = default;

std::string SoftDvppDecodeResizeJpegOperation::Name() const { return kSoftDvppDecodeResizeJpegOperation; }

Status SoftDvppDecodeResizeJpegOperation::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateVectorSize("SoftDvppDecodeResizeJpeg", size_));
  constexpr int32_t value_one = 1;
  constexpr int32_t value_two = 2;
  for (size_t i = 0; i < size_.size(); i++) {
    if (size_[i] % value_two == value_one) {
      std::string err_msg = "SoftDvppDecodeResizeJpeg: size[" + std::to_string(i) +
                            "] must be even values, got: " + std::to_string(size_[i]);
      MS_LOG(ERROR) << err_msg;
      RETURN_STATUS_SYNTAX_ERROR(err_msg);
    }
  }
  return Status::OK();
}

std::shared_ptr<TensorOp> SoftDvppDecodeResizeJpegOperation::Build() {
  constexpr size_t dimension_zero = 0;
  constexpr size_t dimension_one = 1;
  constexpr size_t size_two = 2;

  // If size is a single value, the smaller edge of the image will be
  // resized to this value with the same image aspect ratio.
  int32_t height = size_[dimension_zero];
  int32_t width = 0;

  // User specified the width value.
  if (size_.size() == size_two) {
    width = size_[dimension_one];
  }
  std::shared_ptr<SoftDvppDecodeResizeJpegOp> tensor_op = std::make_shared<SoftDvppDecodeResizeJpegOp>(height, width);
  return tensor_op;
}

Status SoftDvppDecodeResizeJpegOperation::to_json(nlohmann::json *out_json) {
  (*out_json)["size"] = size_;
  return Status::OK();
}

Status SoftDvppDecodeResizeJpegOperation::from_json(nlohmann::json op_params,
                                                    std::shared_ptr<TensorOperation> *operation) {
  CHECK_FAIL_RETURN_UNEXPECTED(op_params.find("size") != op_params.end(), "Failed to find size");
  std::vector<int32_t> size = op_params["size"];
  *operation = std::make_shared<vision::SoftDvppDecodeResizeJpegOperation>(size);
  return Status::OK();
}

#endif
}  // namespace vision
}  // namespace dataset
}  // namespace mindspore
