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

#include "src/litert/kernel/ascend/src/custom_kernel.h"
#include <utility>
#include <map>
#include "include/registry/register_kernel.h"
#include "include/api/types.h"
#include "include/api/data_type.h"
#include "src/litert/kernel/ascend/src/model_infer.h"
#include "src/litert/kernel/ascend/src/acl_options_parser.h"
#include "src/common/log_util.h"
#include "common/log_adapter.h"

namespace mindspore::kernel {
namespace acl {
namespace {
constexpr auto kInputDimNum = 4;
constexpr auto kNHWCHeightIdx = 1;
constexpr auto kNHWCWidthIdx = 2;
constexpr auto kNCHWHeightIdx = 2;
constexpr auto kNCHWWidthIdx = 3;
constexpr auto kImageSizeHwNum = 2;
constexpr auto kSharingWorkspaceSection = "inner_common";
}  // namespace
CustomAscendKernel::CustomAscendKernel(const std::vector<mindspore::MSTensor> &inputs,
                                       const std::vector<mindspore::MSTensor> &outputs,
                                       const schema::Primitive *primitive, const mindspore::Context *ctx)
    : Kernel(inputs, outputs, primitive, ctx),
      load_model_(false),
      prepare_flag_(false),
      acl_options_({}),
      model_infer_(nullptr),
      InputDataIndex_(0) {}

CustomAscendKernel::~CustomAscendKernel() {
  if (load_model_) {
    int ret = model_infer_->Finalize();
    if (ret != lite::RET_OK) {
      MS_LOG(ERROR) << "Model finalize failed.";
    }
  }
}

STATUS CustomAscendKernel::PrepareModelInfer() {
  if (inputs_.size() < 1) {
    MS_LOG(ERROR) << "Inputs size should not be less than 1.";
    return lite::RET_ERROR;
  }
  // last input is om data tensor
  int idx = inputs_.size() - 1;
  if (model_infer_ == nullptr) {
    Buffer om_data(inputs_[idx].Data().get(), inputs_[idx].DataSize());
    AclOptionsParser parser;
    if (parser.ParseAclOptions(context_, &acl_options_) != lite::RET_OK) {
      MS_LOG(ERROR) << "Parse acl options failed.";
      return lite::RET_ERROR;
    }
    model_infer_ = std::make_shared<ModelInfer>(om_data, acl_options_, this->GetConfig(kSharingWorkspaceSection));
    CHECK_NULL_RETURN(model_infer_);
  }
  int ret = model_infer_->Init();
  if (ret != lite::RET_OK) {
    MS_LOG(ERROR) << "Model infer init failed.";
    return lite::RET_ERROR;
  }
  ret = model_infer_->Load();
  if (ret != lite::RET_OK) {
    MS_LOG(ERROR) << "Load om data failed.";
    return lite::RET_ERROR;
  }
  if (prepare_flag_) {
    MS_LOG(INFO) << "Update workspace success.";
    return lite::RET_OK;
  }
  acl_options_.batch_size = model_infer_->GetDynamicBatch();
  acl_options_.image_size = model_infer_->GetDynamicImage();
  MS_LOG(INFO) << "Load om data success.";
  return lite::RET_OK;
}

STATUS CustomAscendKernel::Prepare() {
  if (load_model_) {
    MS_LOG(INFO) << "Custom kernel has been prepared.";
    return lite::RET_OK;
  }
  const std::string calc_workspace_size = "inner_calc_workspace_size";
  const std::map<std::string, std::string> &config_comm = this->GetConfig(kSharingWorkspaceSection);
  if (config_comm.find(calc_workspace_size) != config_comm.end()) {
    prepare_flag_ = true;
  }
  if (PrepareModelInfer() != lite::RET_OK) {
    MS_LOG(ERROR) << "Model infer prepare is not ok.";
    return lite::RET_ERROR;
  }
  if (prepare_flag_) {
    return lite::RET_OK;
  }
  RecordInputDataIndex();

  load_model_ = true;
  return lite::RET_OK;
}

void CustomAscendKernel::RecordInputDataIndex() {
  for (size_t idx = 0; idx < inputs_.size(); ++idx) {
    if (inputs_[idx].Data() == nullptr) {
      InputDataIndex_ = idx;
      break;
    }
  }
}

STATUS CustomAscendKernel::ReSize() {
  if (!load_model_) {
    return Prepare();
  }
  return lite::RET_OK;
}

STATUS CustomAscendKernel::ProcDynamicInput(std::vector<mindspore::MSTensor> *inputs) {
  if (acl_options_.batch_size.empty() && acl_options_.image_size.empty()) {
    MS_LOG(INFO) << "Input is not dynamic mode.";
    return lite::RET_OK;
  }
  if (!acl_options_.batch_size.empty() && !acl_options_.image_size.empty()) {
    MS_LOG(ERROR) << "Batch size and image size can't be set at the same time.";
    return lite::RET_ERROR;
  }
  CHECK_NULL_RETURN(inputs);
  if (!acl_options_.batch_size.empty()) {
    int32_t *batch_size = reinterpret_cast<int32_t *>(malloc(sizeof(int32_t)));
    if (batch_size == nullptr) {
      MS_LOG(ERROR) << "Malloc batch size failed.";
      return lite::RET_ERROR;
    }
    if (GetRealBatchSize(inputs, batch_size) != lite::RET_OK) {
      MS_LOG(ERROR) << "Get real batch size failed.";
      free(batch_size);
      return lite::RET_ERROR;
    }
    mindspore::MSTensor batch_size_input("batch", DataType::kNumberTypeInt32, {1}, batch_size, sizeof(int32_t));
    inputs->emplace_back(batch_size_input);
    free(batch_size);
  }
  if (!acl_options_.image_size.empty()) {
    int32_t *image_size = reinterpret_cast<int32_t *>(malloc(kImageSizeHwNum * sizeof(int32_t)));
    if (image_size == nullptr) {
      MS_LOG(ERROR) << "Malloc image size failed.";
      return lite::RET_ERROR;
    }
    if (GetRealImageSize(inputs, image_size, kImageSizeHwNum) != lite::RET_OK) {
      MS_LOG(ERROR) << "Get real image size failed.";
      free(image_size);
      return lite::RET_ERROR;
    }
    mindspore::MSTensor image_size_input("batch", DataType::kNumberTypeInt32, {2}, image_size,
                                         kImageSizeHwNum * sizeof(int32_t));
    inputs->emplace_back(image_size_input);
    free(image_size);
  }
  return lite::RET_OK;
}

STATUS CustomAscendKernel::GetRealBatchSize(std::vector<mindspore::MSTensor> *inputs, int32_t *batch_size) {
  CHECK_NULL_RETURN(batch_size);
  if (InputDataIndex_ >= inputs->size()) {
    MS_LOG(ERROR) << " Input data index " << InputDataIndex_ << " is larger than input size " << inputs->size();
    return lite::RET_ERROR;
  }
  auto tensor = (*inputs)[InputDataIndex_];
  std::vector<int64_t> shape = tensor.Shape();
  if (shape.empty()) {
    MS_LOG(ERROR) << "Shape is empty, input index = " << InputDataIndex_;
    return lite::RET_ERROR;
  }
  int32_t cur_batch_size = static_cast<uint64_t>(shape[0]);
  auto iter = acl_options_.batch_size.find(cur_batch_size);
  if (iter == acl_options_.batch_size.end()) {
    MS_LOG(ERROR) << "Current batch size " << cur_batch_size << " is invalid, please check device info of context";
    return lite::RET_ERROR;
  }
  *batch_size = cur_batch_size;
  MS_LOG(DEBUG) << "Current batch size " << cur_batch_size;
  return lite::RET_OK;
}

STATUS CustomAscendKernel::GetRealImageSize(std::vector<mindspore::MSTensor> *inputs, int32_t *image_size,
                                            int32_t num) {
  CHECK_NULL_RETURN(image_size);
  if (InputDataIndex_ >= inputs->size()) {
    MS_LOG(ERROR) << "Input data index " << InputDataIndex_ << " is larger than input size " << inputs->size();
    return lite::RET_ERROR;
  }
  auto tensor = (*inputs)[InputDataIndex_];
  std::vector<int64_t> shape = tensor.Shape();
  if (shape.size() != kInputDimNum) {
    MS_LOG(ERROR) << "Shape size " << shape.size() << " is invalid, input index = " << InputDataIndex_;
    return lite::RET_ERROR;
  }
  auto format = tensor.format();
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
  auto iter = acl_options_.image_size.find(cur_image_size);
  if (iter == acl_options_.image_size.end()) {
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

STATUS CustomAscendKernel::Execute() {
  if (!load_model_) {
    MS_LOG(ERROR) << "Custom kernel has not been prepared.";
    return lite::RET_ERROR;
  }
  std::vector<mindspore::MSTensor> inputs(inputs_.begin(), inputs_.end() - 1);
  if (ProcDynamicInput(&inputs) != lite::RET_OK) {
    MS_LOG(ERROR) << "Proc dynamic batch size input failed.";
    return lite::RET_ERROR;
  }
  if (model_infer_->Inference(inputs, &outputs_) != lite::RET_OK) {
    MS_LOG(ERROR) << "Custom kernel execute failed.";
    return lite::RET_ERROR;
  }
  return lite::RET_OK;
}

std::shared_ptr<kernel::Kernel> CustomCreateKernel(const std::vector<mindspore::MSTensor> &inputs,
                                                   const std::vector<mindspore::MSTensor> &outputs,
                                                   const schema::Primitive *primitive, const mindspore::Context *ctx) {
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "Primitive is nullptr.";
    return nullptr;
  }
  if (primitive->value_type() != schema::PrimitiveType_Custom) {
    MS_LOG(ERROR) << "Primitive type is not PrimitiveType_Custom";
    return nullptr;
  }

  auto kernel = std::make_shared<CustomAscendKernel>(inputs, outputs, primitive, ctx);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "New custom kernel is nullptr";
    return nullptr;
  }
  return kernel;
}
}  // namespace acl
}  // namespace mindspore::kernel
namespace mindspore {
namespace registry {
namespace {
const auto kFloat32 = DataType::kNumberTypeFloat32;
const auto kFloat16 = DataType::kNumberTypeFloat16;
const auto kInt32 = DataType::kNumberTypeInt32;
const auto kInt8 = DataType::kNumberTypeInt8;
const auto kUInt8 = DataType::kNumberTypeUInt8;
const auto kBool = DataType::kNumberTypeBool;
}  // namespace
REGISTER_CUSTOM_KERNEL(ASCEND, ACL, kFloat32, ACL, kernel::acl::CustomCreateKernel)
REGISTER_CUSTOM_KERNEL(ASCEND, ACL, kFloat16, ACL, kernel::acl::CustomCreateKernel)
REGISTER_CUSTOM_KERNEL(ASCEND, ACL, kInt32, ACL, kernel::acl::CustomCreateKernel)
REGISTER_CUSTOM_KERNEL(ASCEND, ACL, kInt8, ACL, kernel::acl::CustomCreateKernel)
REGISTER_CUSTOM_KERNEL(ASCEND, ACL, kUInt8, ACL, kernel::acl::CustomCreateKernel)
REGISTER_CUSTOM_KERNEL(ASCEND, ACL, kBool, ACL, kernel::acl::CustomCreateKernel)
}  // namespace registry
}  // namespace mindspore
