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
#include "minddata/dataset/kernels/ir/vision/ascend_vision_ir.h"

#include <algorithm>

#include "minddata/dataset/kernels/image/dvpp/dvpp_crop_jpeg_op.h"
#include "minddata/dataset/kernels/image/dvpp/dvpp_decode_jpeg_op.h"
#include "minddata/dataset/kernels/image/dvpp/dvpp_decode_png_op.h"
#include "minddata/dataset/kernels/image/dvpp/dvpp_decode_resize_crop_jpeg_op.h"
#include "minddata/dataset/kernels/image/dvpp/dvpp_decode_resize_jpeg_op.h"
#include "minddata/dataset/kernels/image/dvpp/dvpp_decode_video_op.h"
#include "minddata/dataset/kernels/image/dvpp/dvpp_normalize_op.h"
#include "minddata/dataset/kernels/image/dvpp/dvpp_resize_jpeg_op.h"
#include "minddata/dataset/util/path.h"
#include "minddata/dataset/util/validators.h"

namespace mindspore {
namespace dataset {

// Transform operations for computer vision
namespace vision {
/* ####################################### Derived TensorOperation classes ################################# */

// DvppCropOperation
DvppCropJpegOperation::DvppCropJpegOperation(const std::vector<uint32_t> &crop) : crop_(crop) {}

Status DvppCropJpegOperation::ValidateParams() {
  // size
  if (crop_.empty() || crop_.size() > 2) {
    std::string err_msg =
      "DvppCropJpeg: Crop resolution must be a vector of one or two elements, got: " + std::to_string(crop_.size());
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  if (*min_element(crop_.begin(), crop_.end()) < 32 || *max_element(crop_.begin(), crop_.end()) > 2048) {
    std::string err_msg = "Dvpp module supports crop image with resolution in range [32, 2048], got crop Parameters: ";
    if (crop_.size() == 2) {
      MS_LOG(ERROR) << err_msg << "[" << crop_[0] << ", " << crop_[1] << "]";
    } else {
      MS_LOG(ERROR) << err_msg << "[" << crop_[0] << ", " << crop_[0] << "]";
    }
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}

std::shared_ptr<TensorOp> DvppCropJpegOperation::Build() {
  // If size is a single value, the smaller edge of the image will be
  // resized to this value with the same image aspect ratio.
  uint32_t cropHeight, cropWidth;
  // User specified the width value.
  if (crop_.size() == 1) {
    cropHeight = crop_[0];
    cropWidth = crop_[0];
  } else {
    cropHeight = crop_[0];
    cropWidth = crop_[1];
  }
  std::shared_ptr<DvppCropJpegOp> tensor_op = std::make_shared<DvppCropJpegOp>(cropHeight, cropWidth);
  return tensor_op;
}

Status DvppCropJpegOperation::to_json(nlohmann::json *out_json) {
  nlohmann::json args;
  args["size"] = crop_;
  *out_json = args;
  return Status::OK();
}

Status DvppCropJpegOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "size", kDvppCropJpegOperation));
  std::vector<uint32_t> resize = op_params["size"];
  *operation = std::make_shared<vision::DvppCropJpegOperation>(resize);
  return Status::OK();
}

// DvppDecodeResizeOperation
DvppDecodeResizeOperation::DvppDecodeResizeOperation(const std::vector<uint32_t> &resize) : resize_(resize) {}

Status DvppDecodeResizeOperation::ValidateParams() {
  // size
  if (resize_.empty() || resize_.size() > 2) {
    std::string err_msg = "DvppDecodeResizeJpeg: resize resolution must be a vector of one or two elements, got: " +
                          std::to_string(resize_.size());
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  if (*min_element(resize_.begin(), resize_.end()) < 32 || *max_element(resize_.begin(), resize_.end()) > 2048) {
    std::string err_msg =
      "Dvpp module supports resize image with resolution in range [32, 2048], got resize Parameters: ";
    if (resize_.size() == 2) {
      MS_LOG(ERROR) << err_msg << "[" << resize_[0] << ", " << resize_[1] << "]";
    } else {
      MS_LOG(ERROR) << err_msg << "[" << resize_[0] << ", " << resize_[0] << "]";
    }
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}

std::shared_ptr<TensorOp> DvppDecodeResizeOperation::Build() {
  // If size is a single value, the smaller edge of the image will be
  // resized to this value with the same image aspect ratio.
  uint32_t resizeHeight, resizeWidth;
  // User specified the width value.
  if (resize_.size() == 1) {
    resizeHeight = resize_[0];
    resizeWidth = 0;
  } else {
    resizeHeight = resize_[0];
    resizeWidth = resize_[1];
  }
  std::shared_ptr<DvppDecodeResizeJpegOp> tensor_op =
    std::make_shared<DvppDecodeResizeJpegOp>(resizeHeight, resizeWidth);
  return tensor_op;
}

Status DvppDecodeResizeOperation::to_json(nlohmann::json *out_json) {
  nlohmann::json args;
  args["size"] = resize_;
  *out_json = args;
  return Status::OK();
}

Status DvppDecodeResizeOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "size", kDvppDecodeResizeOperation));
  std::vector<uint32_t> resize = op_params["size"];
  *operation = std::make_shared<vision::DvppDecodeResizeOperation>(resize);
  return Status::OK();
}

// DvppDecodeVideoOperation
DvppDecodeVideoOperation::DvppDecodeVideoOperation(const std::vector<uint32_t> &size, VdecStreamFormat type,
                                                   VdecOutputFormat out_format, const std::string &output)
    : size_(size), format_(out_format), en_type_(type), output_(output) {}

Status DvppDecodeVideoOperation::ValidateParams() {
  // check size_
  if (size_.empty() || size_.size() > 2) {
    std::string err_msg =
      "DvppDecodeVideo: Video frame size must be a vector of one or two elements, got: " + std::to_string(size_.size());
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  // check height and width
  uint32_t height, width;
  height = size_[0];
  width = size_[1];

  if ((width < kFrameWidthMin) || (width > kFrameWidthMax)) {
    std::string err_msg = "DvppDecodeVideo: video frame width " + std::to_string(width) +
                          " is invalid, the legal range is [" + std::to_string(kFrameWidthMin) + ", " +
                          std::to_string(kFrameWidthMax) + "]";
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  if ((height < kFrameHeightMin) || (height > kFrameHeightMax)) {
    std::string err_msg = "DvppDecodeVideo: video frame height " + std::to_string(height) +
                          " is invalid, the legal range is [" + std::to_string(kFrameHeightMin) + ", " +
                          std::to_string(kFrameHeightMax) + "]";
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  if (en_type_ < VdecStreamFormat::kH265MainLevel || en_type_ > VdecStreamFormat::kH264HighLevel) {
    std::string err_msg = "DvppDecodeVideo: Invalid VdecStreamFormat, check input value of enum.";
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  if (format_ < VdecOutputFormat::kYuvSemiplanar420 || format_ > VdecOutputFormat::kYvuSemiplanar420) {
    std::string err_msg = "DvppDecodeVideo: Invalid VdecOutputFormat, check input value of enum.";
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  // check and normalize output path
  Path output(output_);
  if (!output.Exists()) {
    RETURN_IF_NOT_OK(output.CreateDirectories());
  }
  if (!output.IsDirectory()) {
    std::string err_msg = "DvppDecodeVideo: Invalid out path, check path: " + output.ToString();
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  output_ = output.ToString();
  return Status::OK();
}

std::shared_ptr<TensorOp> DvppDecodeVideoOperation::Build() {
  uint32_t height, width;
  height = size_[0];
  width = size_[1];
  auto tensor_op = std::make_shared<DvppDecodeVideoOp>(width, height, en_type_, format_, output_);
  return tensor_op;
}

Status DvppDecodeVideoOperation::to_json(nlohmann::json *out_json) {
  nlohmann::json args;
  args["size"] = size_;
  args["en_type"] = en_type_;
  args["out_format"] = format_;
  args["output"] = output_;
  *out_json = args;
  return Status::OK();
}

Status DvppDecodeVideoOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "size", kDvppDecodeVideoOperation));
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "en_type", kDvppDecodeVideoOperation));
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "out_format", kDvppDecodeVideoOperation));
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "output", kDvppDecodeVideoOperation));

  std::vector<uint32_t> size = op_params["size"];
  VdecStreamFormat type = static_cast<VdecStreamFormat>(op_params["en_type"]);
  VdecOutputFormat out_format = static_cast<VdecOutputFormat>(op_params["out_format"]);
  std::string output = op_params["output"];

  *operation = std::make_shared<vision::DvppDecodeVideoOperation>(size, type, out_format, output);
  return Status::OK();
}

// DvppDecodeResizeCropOperation
DvppDecodeResizeCropOperation::DvppDecodeResizeCropOperation(const std::vector<uint32_t> &crop,
                                                             const std::vector<uint32_t> &resize)
    : crop_(crop), resize_(resize) {}

Status DvppDecodeResizeCropOperation::ValidateParams() {
  // size
  if (crop_.empty() || crop_.size() > 2) {
    std::string err_msg = "DvppDecodeResizeCropJpeg: crop resolution must be a vector of one or two elements, got: " +
                          std::to_string(crop_.size());
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  if (resize_.empty() || resize_.size() > 2) {
    std::string err_msg = "DvppDecodeResizeCropJpeg: resize resolution must be a vector of one or two elements, got: " +
                          std::to_string(resize_.size());
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  if (*min_element(crop_.begin(), crop_.end()) < 32 || *max_element(crop_.begin(), crop_.end()) > 2048) {
    std::string err_msg = "Dvpp module supports crop image with resolution in range [32, 2048], got Crop Parameters: ";
    if (crop_.size() == 2) {
      MS_LOG(ERROR) << err_msg << "[" << crop_[0] << ", " << crop_[1] << "]";
    } else {
      MS_LOG(ERROR) << err_msg << "[" << crop_[0] << ", " << crop_[0] << "]";
    }
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  if (*min_element(resize_.begin(), resize_.end()) < 32 || *max_element(resize_.begin(), resize_.end()) > 2048) {
    std::string err_msg =
      "Dvpp module supports resize image with resolution in range [32, 2048], got Crop Parameters: ";
    if (resize_.size() == 2) {
      MS_LOG(ERROR) << err_msg << "[" << resize_[0] << ", " << resize_[1] << "]";
    } else {
      MS_LOG(ERROR) << err_msg << "[" << resize_[0] << "]";
    }
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  if (crop_.size() < resize_.size()) {
    if (crop_[0] > std::min(resize_[0], resize_[1])) {
      std::string err_msg =
        "Each value of crop parameter must be smaller than corresponding resize parameter, for example: x[0] <= "
        "y[0],  and x[1] <= y[1], please verify your input parameters.";
      LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
    }
  }
  if (crop_.size() > resize_.size()) {
    if (std::max(crop_[0], crop_[1]) > resize_[0]) {
      std::string err_msg =
        "Each value of crop parameter must be smaller than corresponding resize parameter, for example: x[0] <= "
        "y[0],  and x[1] <= y[1], please verify your input parameters.";
      LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
    }
  }
  if (crop_.size() == resize_.size()) {
    for (int32_t i = 0; i < crop_.size(); ++i) {
      if (crop_[i] > resize_[i]) {
        std::string err_msg =
          "Each value of crop parameter must be smaller than corresponding resize parameter, for example: x[0] <= "
          "y[0],  and x[1] <= y[1], please verify your input parameters.";
        LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
      }
    }
  }
  return Status::OK();
}

std::shared_ptr<TensorOp> DvppDecodeResizeCropOperation::Build() {
  // If size is a single value, the smaller edge of the image will be
  // resized to this value with the same image aspect ratio.
  uint32_t cropHeight, cropWidth, resizeHeight, resizeWidth;
  if (crop_.size() == 1) {
    cropHeight = crop_[0];
    cropWidth = crop_[0];
  } else {
    cropHeight = crop_[0];
    cropWidth = crop_[1];
  }
  // User specified the width value.
  if (resize_.size() == 1) {
    resizeHeight = resize_[0];
    resizeWidth = 0;
  } else {
    resizeHeight = resize_[0];
    resizeWidth = resize_[1];
  }
  std::shared_ptr<DvppDecodeResizeCropJpegOp> tensor_op =
    std::make_shared<DvppDecodeResizeCropJpegOp>(cropHeight, cropWidth, resizeHeight, resizeWidth);
  return tensor_op;
}

Status DvppDecodeResizeCropOperation::to_json(nlohmann::json *out_json) {
  nlohmann::json args;
  args["crop_size"] = crop_;
  args["resize_size"] = resize_;
  *out_json = args;
  return Status::OK();
}

Status DvppDecodeResizeCropOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "crop_size", kDvppDecodeResizeCropOperation));
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "resize_size", kDvppDecodeResizeCropOperation));
  std::vector<uint32_t> crop = op_params["crop_size"];
  std::vector<uint32_t> resize = op_params["resize_size"];
  *operation = std::make_shared<vision::DvppDecodeResizeCropOperation>(crop, resize);
  return Status::OK();
}

// DvppDecodeJPEG
Status DvppDecodeJpegOperation::ValidateParams() { return Status::OK(); }

std::shared_ptr<TensorOp> DvppDecodeJpegOperation::Build() { return std::make_shared<DvppDecodeJpegOp>(); }

// DvppDecodePNG
Status DvppDecodePngOperation::ValidateParams() { return Status::OK(); }

std::shared_ptr<TensorOp> DvppDecodePngOperation::Build() { return std::make_shared<DvppDecodePngOp>(); }

// DvppNormalize
DvppNormalizeOperation::DvppNormalizeOperation(const std::vector<float> &mean, const std::vector<float> &std)
    : mean_(mean), std_(std) {}

Status DvppNormalizeOperation::ValidateParams() {
  if (mean_.size() != 3) {
    std::string err_msg = "DvppNormalization:: mean expecting size 3, got size: " + std::to_string(mean_.size());
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  if (std_.size() != 3) {
    std::string err_msg = "DvppNormalization: std expecting size 3, got size: " + std::to_string(std_.size());
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  if (*min_element(mean_.begin(), mean_.end()) < 0 || *max_element(mean_.begin(), mean_.end()) > 256) {
    std::string err_msg =
      "Normalization can take parameters in range [0, 256] according to math theory of mean and sigma, got mean "
      "vector" +
      std::to_string(std_.size());
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  if (*min_element(std_.begin(), std_.end()) < 0 || *max_element(std_.begin(), std_.end()) > 256) {
    std::string err_msg =
      "Normalization can take parameters in range [0, 256] according to math theory of mean and sigma, got mean "
      "vector" +
      std::to_string(std_.size());
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}

std::shared_ptr<TensorOp> DvppNormalizeOperation::Build() {
  std::shared_ptr<DvppNormalizeOp> tensor_op = std::make_shared<DvppNormalizeOp>(mean_, std_);
  return tensor_op;
}

Status DvppNormalizeOperation::to_json(nlohmann::json *out_json) {
  nlohmann::json args;
  std::vector<uint32_t> enlarge_mean_;
  std::vector<uint32_t> enlarge_std_;
  std::transform(mean_.begin(), mean_.end(), std::back_inserter(enlarge_mean_),
                 [](float i) -> uint32_t { return static_cast<uint32_t>(10000 * i); });
  std::transform(std_.begin(), std_.end(), std::back_inserter(enlarge_std_),
                 [](float j) -> uint32_t { return static_cast<uint32_t>(10000 * j); });
  args["mean"] = enlarge_mean_;
  args["std"] = enlarge_std_;
  *out_json = args;
  return Status::OK();
}

Status DvppNormalizeOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "mean", kDvppNormalizeOperation));
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "std", kDvppNormalizeOperation));
  std::vector<float> mean = op_params["mean"];
  std::vector<float> std = op_params["std"];
  *operation = std::make_shared<vision::DvppNormalizeOperation>(mean, std);
  return Status::OK();
}

// DvppResizeOperation
DvppResizeJpegOperation::DvppResizeJpegOperation(const std::vector<uint32_t> &resize) : resize_(resize) {}

Status DvppResizeJpegOperation::ValidateParams() {
  // size
  if (resize_.empty() || resize_.size() > 2) {
    std::string err_msg = "DvppResizeJpeg: resize resolution must be a vector of one or two elements, got: " +
                          std::to_string(resize_.size());
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  if (*min_element(resize_.begin(), resize_.end()) < 32 || *max_element(resize_.begin(), resize_.end()) > 2048) {
    std::string err_msg =
      "Dvpp module supports resize image with resolution in range [32, 2048], got resize Parameters: ";
    if (resize_.size() == 2) {
      MS_LOG(ERROR) << err_msg << "[" << resize_[0] << ", " << resize_[1] << "]";
    } else {
      MS_LOG(ERROR) << err_msg << "[" << resize_[0] << ", " << resize_[0] << "]";
    }
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}

std::shared_ptr<TensorOp> DvppResizeJpegOperation::Build() {
  // If size is a single value, the smaller edge of the image will be
  // resized to this value with the same image aspect ratio.
  uint32_t resizeHeight, resizeWidth;
  // User specified the width value.
  if (resize_.size() == 1) {
    resizeHeight = resize_[0];
    resizeWidth = 0;
  } else {
    resizeHeight = resize_[0];
    resizeWidth = resize_[1];
  }
  std::shared_ptr<DvppResizeJpegOp> tensor_op = std::make_shared<DvppResizeJpegOp>(resizeHeight, resizeWidth);
  return tensor_op;
}

Status DvppResizeJpegOperation::to_json(nlohmann::json *out_json) {
  nlohmann::json args;
  args["size"] = resize_;
  *out_json = args;
  return Status::OK();
}

Status DvppResizeJpegOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "size", kDvppResizeJpegOperation));
  std::vector<uint32_t> resize = op_params["size"];
  *operation = std::make_shared<vision::DvppResizeJpegOperation>(resize);
  return Status::OK();
}
}  // namespace vision
}  // namespace dataset
}  // namespace mindspore
