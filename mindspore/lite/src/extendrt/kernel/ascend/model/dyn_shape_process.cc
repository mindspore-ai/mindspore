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

int DynShapeProcess::ProcDynamicInput(std::vector<KernelTensorPtr> *const original_datas,
                                      std::vector<KernelTensorPtr> *const inputs) {
  MS_CHECK_TRUE_MSG(acl_options_ != nullptr, lite::RET_ERROR, "Acl options ptr is nullptr.");
  if (!acl_options_->batch_size.empty() && !acl_options_->image_size.empty()) {
    MS_LOG(ERROR) << "Batch size and image size can't be set at the same time.";
    return lite::RET_ERROR;
  }
  MS_CHECK_TRUE_MSG(original_datas != nullptr, lite::RET_ERROR, "Original Data is nullptr.");
  MS_CHECK_TRUE_MSG(inputs != nullptr, lite::RET_ERROR, "Inputs is nullptr.");
  MS_CHECK_TRUE_MSG((*original_datas).size() == (*inputs).size(), lite::RET_ERROR,
                    "The size of Original Data and Input is not equal.");
  if (!acl_options_->batch_size.empty()) {
    if (AddBatchSizeInput(original_datas, inputs) != lite::RET_OK) {
      MS_LOG(ERROR) << "Add batch size input failed.";
      return lite::RET_ERROR;
    }
  }
  if (!acl_options_->image_size.empty()) {
    if (AddImageSizeInput(original_datas, inputs) != lite::RET_OK) {
      MS_LOG(ERROR) << "Add Image size input failed.";
      return lite::RET_ERROR;
    }
  }
  return lite::RET_OK;
}

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

int DynShapeProcess::CheckBatchSize(std::vector<KernelTensorPtr> *const original_datas,
                                    std::vector<KernelTensorPtr> *const inputs) {
  if (input_data_idx_ >= inputs->size()) {
    MS_LOG(ERROR) << " Input data index " << input_data_idx_ << " is larger than input size " << inputs->size();
    return lite::RET_ERROR;
  }
  auto original_tensor = (*original_datas)[input_data_idx_];
  auto cur_tensor = (*inputs)[input_data_idx_];
  std::vector<int64_t> original_shape = original_tensor->GetShapeVector();
  std::vector<int64_t> cur_shape = cur_tensor->GetShapeVector();
  if (cur_shape.empty() || original_shape.empty()) {
    MS_LOG(ERROR) << "Shape is empty, input index = " << input_data_idx_;
    return lite::RET_ERROR;
  }
  for (uint32_t i = 1; i < cur_shape.size(); ++i) {
    if (original_shape[i] != kUnknownDim && (original_shape[i] != cur_shape[i])) {
      MS_LOG(ERROR) << "Shape Conflict: Original Shape:[" << GenResultStr(original_shape) << "], Current Shape:["
                    << GenResultStr(cur_shape) << "]";
      return lite::RET_ERROR;
    }
  }
  return lite::RET_OK;
}

int DynShapeProcess::CheckImageSize(std::vector<KernelTensorPtr> *const original_datas,
                                    std::vector<KernelTensorPtr> *const inputs) {
  if (input_data_idx_ >= inputs->size() || input_data_idx_ >= acl_options_->input_format.size()) {
    MS_LOG(ERROR) << "Input data index " << input_data_idx_ << " is invalid, inputs size " << inputs->size()
                  << " input formats size " << acl_options_->input_format.size();
    return lite::RET_ERROR;
  }
  auto original_tensor = (*original_datas)[input_data_idx_];
  auto cur_tensor = (*inputs)[input_data_idx_];
  std::vector<int64_t> original_shape = original_tensor->GetShapeVector();
  std::vector<int64_t> cur_shape = cur_tensor->GetShapeVector();
  if (original_shape.size() != kInputDimNum) {
    MS_LOG(ERROR) << "Shape size " << original_shape.size() << " is invalid, input index = " << input_data_idx_;
    return lite::RET_ERROR;
  }
  if (cur_shape.size() != kInputDimNum) {
    MS_LOG(ERROR) << "Shape size " << cur_shape.size() << " is invalid, input index = " << input_data_idx_;
    return lite::RET_ERROR;
  }
  auto format = acl_options_->input_format[input_data_idx_];
  if (format == mindspore::Format::NHWC) {
    if ((original_shape[kNHWCCIdx] != kUnknownDim && (original_shape[kNHWCCIdx] != cur_shape[kNHWCCIdx])) ||
        (original_shape[kNHWCNIdx] != kUnknownDim && (original_shape[kNHWCNIdx] != cur_shape[kNHWCNIdx]))) {
      MS_LOG(ERROR) << "Shape Conflict: Original Shape:[" << GenResultStr(original_shape) << "], Current Shape:["
                    << GenResultStr(cur_shape) << "]";
      return lite::RET_ERROR;
    }
  } else {
    if ((original_shape[kNCHWCIdx] != kUnknownDim && (original_shape[kNCHWCIdx] != cur_shape[kNCHWCIdx])) ||
        (original_shape[kNCHWNIdx] != kUnknownDim && (original_shape[kNCHWNIdx] != cur_shape[kNCHWNIdx]))) {
      MS_LOG(ERROR) << "Shape Conflict: Original Shape:[" << GenResultStr(original_shape) << "], Current Shape:["
                    << GenResultStr(cur_shape) << "]";
      return lite::RET_ERROR;
    }
  }
  return lite::RET_OK;
}

int DynShapeProcess::AddBatchSizeInput(std::vector<KernelTensorPtr> *const original_datas,
                                       std::vector<KernelTensorPtr> *const inputs) {
  int32_t *batch_size_addr = reinterpret_cast<int32_t *>(malloc(sizeof(int32_t)));
  if (batch_size_addr == nullptr) {
    MS_LOG(ERROR) << "Malloc batch size failed.";
    return lite::RET_ERROR;
  }
  if (CheckBatchSize(original_datas, inputs) != lite::RET_OK) {
    MS_LOG(ERROR) << "Check dynamic batch size failed.";
    free(batch_size_addr);
    return lite::RET_ERROR;
  }
  if (GetRealBatchSize(inputs, batch_size_addr) != lite::RET_OK) {
    MS_LOG(ERROR) << "Get real batch size failed.";
    free(batch_size_addr);
    return lite::RET_ERROR;
  }
  batch_size_ptr_ = std::make_shared<Address>(batch_size_addr, sizeof(int32_t));
  if (batch_size_ptr_ == nullptr) {
    MS_LOG(ERROR) << "Create Address failed.";
    free(batch_size_addr);
    return lite::RET_ERROR;
  }
  auto tensor_ptr = std::make_shared<KernelTensor>();
  if (tensor_ptr == nullptr) {
    MS_LOG(ERROR) << "Create KernelTensor failed.";
    free(batch_size_addr);
    return lite::RET_ERROR;
  }

  tensor_ptr->SetData(batch_size_ptr_);
  auto abstract = std::make_shared<abstract::AbstractTensor>(kInt32, std::vector<int64_t>());
  tensor_ptr->SetAbstract(abstract);
  inputs->emplace_back(tensor_ptr);
  return lite::RET_OK;
}

int DynShapeProcess::AddImageSizeInput(std::vector<KernelTensorPtr> *const original_datas,
                                       std::vector<KernelTensorPtr> *const inputs) {
  int32_t *image_size_addr = reinterpret_cast<int32_t *>(malloc(kImageSizeHwNum * sizeof(int32_t)));
  if (image_size_addr == nullptr) {
    MS_LOG(ERROR) << "Malloc image size failed.";
    return lite::RET_ERROR;
  }
  if (CheckImageSize(original_datas, inputs) != lite::RET_OK) {
    MS_LOG(ERROR) << "Check dynamic image size failed.";
    free(image_size_addr);
    return lite::RET_ERROR;
  }
  if (GetRealImageSize(inputs, image_size_addr, kImageSizeHwNum) != lite::RET_OK) {
    MS_LOG(ERROR) << "Get real image size failed.";
    free(image_size_addr);
    return lite::RET_ERROR;
  }
  image_size_ptr_ = std::make_shared<Address>(image_size_addr, kImageSizeHwNum * sizeof(int32_t));
  if (image_size_ptr_ == nullptr) {
    MS_LOG(ERROR) << "Create Address failed.";
    free(image_size_addr);
    return lite::RET_ERROR;
  }
  auto tensor_ptr = std::make_shared<KernelTensor>();
  if (tensor_ptr == nullptr) {
    MS_LOG(ERROR) << "Create KernelTensor failed.";
    free(image_size_addr);
    return lite::RET_ERROR;
  }

  tensor_ptr->SetData(image_size_ptr_);
  auto abstract = std::make_shared<abstract::AbstractTensor>(kInt32, std::vector<int64_t>());
  tensor_ptr->SetAbstract(abstract);
  inputs->emplace_back(tensor_ptr);
  return lite::RET_OK;
}

int DynShapeProcess::GetRealBatchSize(std::vector<KernelTensorPtr> *const inputs, int32_t *batch_size) {
  MS_CHECK_TRUE_MSG(batch_size != nullptr, lite::RET_ERROR, "Batch size ptr is nullptr.");
  if (input_data_idx_ >= inputs->size()) {
    MS_LOG(ERROR) << " Input data index " << input_data_idx_ << " is larger than input size " << inputs->size();
    return lite::RET_ERROR;
  }
  auto tensor = (*inputs)[input_data_idx_];
  std::vector<int64_t> shape = tensor->GetShapeVector();
  if (shape.empty()) {
    MS_LOG(ERROR) << "Shape is empty, input index = " << input_data_idx_;
    return lite::RET_ERROR;
  }
  int32_t cur_batch_size = static_cast<uint64_t>(shape[0]);
  auto iter = acl_options_->batch_size.find(cur_batch_size);
  if (iter == acl_options_->batch_size.end()) {
    MS_LOG(ERROR) << "Current batch size " << cur_batch_size << " is invalid, please check device info of context";
    return lite::RET_ERROR;
  }
  *batch_size = cur_batch_size;
  MS_LOG(DEBUG) << "Current batch size " << cur_batch_size;
  return lite::RET_OK;
}

int DynShapeProcess::GetRealImageSize(std::vector<KernelTensorPtr> *const inputs, int32_t *image_size, int32_t num) {
  MS_CHECK_TRUE_MSG(image_size != nullptr, lite::RET_ERROR, "Image size ptr is nullptr.");
  if (input_data_idx_ >= inputs->size() || input_data_idx_ >= acl_options_->input_format.size()) {
    MS_LOG(ERROR) << "Input data index " << input_data_idx_ << " is invalid, inputs size " << inputs->size()
                  << " input formats size " << acl_options_->input_format.size();
    return lite::RET_ERROR;
  }
  auto tensor = (*inputs)[input_data_idx_];
  std::vector<int64_t> shape = tensor->GetShapeVector();
  if (shape.size() != kInputDimNum) {
    MS_LOG(ERROR) << "Shape size " << shape.size() << " is invalid, input index = " << input_data_idx_;
    return lite::RET_ERROR;
  }
  auto format = acl_options_->input_format[input_data_idx_];
  uint64_t height;
  uint64_t width;
  if (format == mindspore::Format::NHWC) {
    height = shape[kNHWCHeightIdx];
    width = shape[kNHWCWidthIdx];
  } else {
    height = shape[kNCHWHeightIdx];
    width = shape[kNCHWWidthIdx];
  }
  auto cur_image_size = std::pair<int32_t, int32_t>(static_cast<uint64_t>(height), static_cast<uint64_t>(width));
  auto iter = acl_options_->image_size.find(cur_image_size);
  if (iter == acl_options_->image_size.end()) {
    MS_LOG(ERROR) << "Image size height " << height << ",weight " << width
                  << " is invalid, please check device info of context.";
    return lite::RET_ERROR;
  }
  if (num != kImageSizeHwNum) {
    MS_LOG(ERROR) << "The hw num should be " << kImageSizeHwNum << ",real num " << num;
    return lite::RET_ERROR;
  }
  image_size[0] = height;
  image_size[1] = width;
  MS_LOG(DEBUG) << "Current height " << height << " width " << width;
  return lite::RET_OK;
}

void DynShapeProcess::DestroyDynamicInput(std::vector<KernelTensorPtr> *const inputs) {
  if (inputs == nullptr) {
    MS_LOG(ERROR) << "Inputs ptr is nullptr.";
    return;
  }
  if (batch_size_ptr_ != nullptr && batch_size_ptr_->addr != nullptr) {
    free(batch_size_ptr_->addr);
    batch_size_ptr_->addr = nullptr;
    batch_size_ptr_->size = 0;
  }
  if (image_size_ptr_ != nullptr && image_size_ptr_->addr != nullptr) {
    free(image_size_ptr_->addr);
    image_size_ptr_->addr = nullptr;
    image_size_ptr_->size = 0;
  }
  if (!inputs->empty()) {
    (*inputs).pop_back();
  }
}
}  // namespace acl
}  // namespace mindspore::kernel
