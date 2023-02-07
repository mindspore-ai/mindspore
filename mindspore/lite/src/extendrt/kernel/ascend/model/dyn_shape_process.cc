/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "extendrt/kernel/ascend/model/dyn_shape_process.h"
#include <utility>
#include "mindspore/ccsrc/plugin/device/cpu/kernel/nnacl/op_base.h"
#include "include/errorcode.h"

namespace mindspore::kernel {
namespace acl {
namespace {
constexpr auto kInputDimNum = 4;
constexpr auto kNHWCNIdx = 0;
constexpr auto kNHWCHeightIdx = 1;
constexpr auto kNHWCWidthIdx = 2;
constexpr auto kNHWCCIdx = 3;
constexpr auto kNCHWNIdx = 0;
constexpr auto kNCHWCIdx = 1;
constexpr auto kNCHWHeightIdx = 2;
constexpr auto kNCHWWidthIdx = 3;
constexpr auto kImageSizeHwNum = 2;
constexpr auto kUnknownDim = -1;
}  // namespace
std::string GenResultStr(const std::vector<int64_t> &input_vec) {
  std::string res;
  for (size_t i = 0; i < input_vec.size(); ++i) {
    res += std::to_string(input_vec[i]);
    if (i != input_vec.size() - 1) {
      res += ",";
    }
  }
  return res;
}

bool DynShapeProcess::Init(const AclDynamicShapeOptions &options) {
  acl_options_ = options;
  for (size_t i = 0; i < options.input_shapes.size(); i++) {
    auto &shape = options.input_shapes[i];
    if (std::any_of(shape.begin(), shape.end(), [](auto dim) { return dim < 0; })) {
      input_data_idx_ = i;
      break;
    }
  }
  if (input_data_idx_ >= acl_options_.input_shapes.size()) {
    MS_LOG(ERROR) << "Input data index " << input_data_idx_
                  << " is invalid, inputs count: " << acl_options_.input_shapes.size();
    return false;
  }
  return true;
}

bool DynShapeProcess::CheckAndGetBatchSize(const std::vector<ShapeVector> &new_shapes, int32_t *batch_size) {
  if (acl_options_.batch_size.empty()) {
    MS_LOG(ERROR) << "Not support dynamic batch size";
    return false;
  }
  if (batch_size == nullptr) {
    MS_LOG(ERROR) << "Input parameter batch size cannot be nullptr";
    return false;
  }
  if (!CheckBatchSize(new_shapes)) {
    return false;
  }
  return GetRealBatchSize(new_shapes, batch_size);
}

bool DynShapeProcess::CheckAndGetImageSize(const std::vector<ShapeVector> &new_shapes, int32_t *height,
                                           int32_t *width) {
  if (acl_options_.image_size.empty()) {
    MS_LOG(ERROR) << "Not support image batch size";
    return false;
  }
  if (height == nullptr || width == nullptr) {
    MS_LOG(ERROR) << "Input parameter image size cannot be nullptr";
    return false;
  }
  if (!CheckImageSize(new_shapes)) {
    return false;
  }
  return GetRealImageSize(new_shapes, height, width);
}

bool DynShapeProcess::CheckBatchSize(const std::vector<ShapeVector> &new_shapes) {
  if (input_data_idx_ >= new_shapes.size()) {
    MS_LOG(ERROR) << " Input data index " << input_data_idx_ << " is larger than input size " << new_shapes.size();
    return false;
  }
  std::vector<int64_t> original_shape = acl_options_.input_shapes[input_data_idx_];
  std::vector<int64_t> cur_shape = new_shapes[input_data_idx_];
  if (cur_shape.empty() || original_shape.empty()) {
    MS_LOG(ERROR) << "Shape is empty, input index = " << input_data_idx_;
    return false;
  }
  if (cur_shape.size() != original_shape.size()) {
    MS_LOG(ERROR) << "Cur shape size " << cur_shape.size() << " is not equal with original shape size "
                  << original_shape.size();
    return false;
  }
  for (size_t i = 1; i < cur_shape.size(); ++i) {
    if (cur_shape[i] <= 0) {
      MS_LOG(ERROR) << "Invalid new shape " << cur_shape << " for input " << i;
      return false;
    }
    if (original_shape[i] != kUnknownDim && (original_shape[i] != cur_shape[i])) {
      MS_LOG(ERROR) << "Shape Conflict: Original Shape:[" << GenResultStr(original_shape) << "], Current Shape:["
                    << GenResultStr(cur_shape) << "]";
      return false;
    }
  }
  return true;
}

bool DynShapeProcess::CheckImageSize(const std::vector<ShapeVector> &new_shapes) {
  if (input_data_idx_ >= new_shapes.size() || input_data_idx_ >= acl_options_.input_format.size()) {
    MS_LOG(ERROR) << "Input data index " << input_data_idx_ << " is invalid, inputs size " << new_shapes.size()
                  << " input formats size " << acl_options_.input_format.size();
    return false;
  }
  std::vector<int64_t> original_shape = acl_options_.input_shapes[input_data_idx_];
  std::vector<int64_t> cur_shape = new_shapes[input_data_idx_];
  if (original_shape.size() != kInputDimNum) {
    MS_LOG(ERROR) << "Shape size " << original_shape.size() << " is invalid, input index = " << input_data_idx_;
    return false;
  }
  if (cur_shape.size() != original_shape.size()) {
    MS_LOG(ERROR) << "Cur shape size " << cur_shape.size() << " is not equal with original shape size "
                  << original_shape.size();
    return false;
  }
  for (size_t i = 1; i < cur_shape.size(); ++i) {
    if (cur_shape[i] <= 0) {
      MS_LOG(ERROR) << "Invalid new shape " << cur_shape << " for input " << i;
      return false;
    }
    if (original_shape[i] != kUnknownDim && (original_shape[i] != cur_shape[i])) {
      MS_LOG(ERROR) << "Shape Conflict: Original Shape:[" << GenResultStr(original_shape) << "], Current Shape:["
                    << GenResultStr(cur_shape) << "]";
      return false;
    }
  }
  auto format = acl_options_.input_format[input_data_idx_];
  if (format == mindspore::Format::NHWC) {
    if ((original_shape[kNHWCCIdx] != kUnknownDim && (original_shape[kNHWCCIdx] != cur_shape[kNHWCCIdx])) ||
        (original_shape[kNHWCNIdx] != kUnknownDim && (original_shape[kNHWCNIdx] != cur_shape[kNHWCNIdx]))) {
      MS_LOG(ERROR) << "Shape Conflict: Original Shape:[" << GenResultStr(original_shape) << "], Current Shape:["
                    << GenResultStr(cur_shape) << "]";
      return false;
    }
  } else {
    if ((original_shape[kNCHWCIdx] != kUnknownDim && (original_shape[kNCHWCIdx] != cur_shape[kNCHWCIdx])) ||
        (original_shape[kNCHWNIdx] != kUnknownDim && (original_shape[kNCHWNIdx] != cur_shape[kNCHWNIdx]))) {
      MS_LOG(ERROR) << "Shape Conflict: Original Shape:[" << GenResultStr(original_shape) << "], Current Shape:["
                    << GenResultStr(cur_shape) << "]";
      return false;
    }
  }
  return true;
}

bool DynShapeProcess::GetRealBatchSize(const std::vector<ShapeVector> &new_shapes, int32_t *batch_size) {
  if (input_data_idx_ >= new_shapes.size()) {
    MS_LOG(ERROR) << " Input data index " << input_data_idx_ << " is larger than input size " << new_shapes.size();
    return false;
  }
  std::vector<int64_t> shape = new_shapes[input_data_idx_];
  if (shape.empty()) {
    MS_LOG(ERROR) << "Shape is empty, input index = " << input_data_idx_;
    return false;
  }
  int32_t cur_batch_size = static_cast<uint64_t>(shape[0]);
  auto iter = acl_options_.batch_size.find(cur_batch_size);
  if (iter == acl_options_.batch_size.end()) {
    MS_LOG(ERROR) << "Current batch size " << cur_batch_size << " is invalid, please check device info of context";
    return false;
  }
  *batch_size = cur_batch_size;
  MS_LOG(DEBUG) << "Current batch size " << cur_batch_size;
  return true;
}

bool DynShapeProcess::GetRealImageSize(const std::vector<ShapeVector> &new_shapes, int32_t *height_p,
                                       int32_t *width_p) {
  if (input_data_idx_ >= new_shapes.size() || input_data_idx_ >= acl_options_.input_format.size()) {
    MS_LOG(ERROR) << "Input data index " << input_data_idx_ << " is invalid, inputs size " << new_shapes.size()
                  << " input formats size " << acl_options_.input_format.size();
    return false;
  }
  std::vector<int64_t> shape = new_shapes[input_data_idx_];
  if (shape.size() != kInputDimNum) {
    MS_LOG(ERROR) << "Shape size " << shape.size() << " is invalid, input index = " << input_data_idx_;
    return false;
  }
  auto format = acl_options_.input_format[input_data_idx_];
  int64_t height;
  int64_t width;
  if (format == mindspore::Format::NHWC) {
    height = shape[kNHWCHeightIdx];
    width = shape[kNHWCWidthIdx];
  } else {
    height = shape[kNCHWHeightIdx];
    width = shape[kNCHWWidthIdx];
  }
  auto cur_image_size = std::pair<int32_t, int32_t>(static_cast<int32_t>(height), static_cast<int32_t>(width));
  auto iter = acl_options_.image_size.find(cur_image_size);
  if (iter == acl_options_.image_size.end()) {
    MS_LOG(ERROR) << "Image size height " << height << ",weight " << width
                  << " is invalid, please check device info of context.";
    return false;
  }
  *height_p = LongToInt(height);
  *width_p = LongToInt(width);
  MS_LOG(DEBUG) << "Current height " << height << " width " << width;
  return true;
}
}  // namespace acl
}  // namespace mindspore::kernel
