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
#include <fstream>
#include "minddata/dataset/include/execute.h"
#include "minddata/dataset/core/de_tensor.h"
#include "minddata/dataset/core/device_resource.h"
#include "minddata/dataset/core/device_tensor.h"
#include "minddata/dataset/core/tensor_row.h"
#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/core/type_id.h"
#include "minddata/dataset/kernels/ir/tensor_operation.h"
#include "minddata/dataset/kernels/tensor_op.h"
#ifndef ENABLE_ANDROID
#include "utils/log_adapter.h"
#else
#include "mindspore/lite/src/common/log_adapter.h"
#endif
#ifdef ENABLE_ACL
#include "minddata/dataset/core/ascend_resource.h"
#include "minddata/dataset/kernels/image/dvpp/utils/CommonDataType.h"
#include "minddata/dataset/kernels/ir/vision/ascend_vision_ir.h"
#endif

namespace mindspore {
namespace dataset {

using json = nlohmann::json;
struct Execute::ExtraInfo {
  std::multimap<std::string, std::vector<uint32_t>> aipp_cfg_;
  bool init_with_shared_ptr_ = true;  // Initial execute object with shared_ptr as default
#ifdef ENABLE_ACL
  std::multimap<std::string, std::string> op2para_map_ = {{vision::kDvppCropJpegOperation, "size"},
                                                          {vision::kDvppDecodeResizeOperation, "size"},
                                                          {vision::kDvppDecodeResizeCropOperation, "crop_size"},
                                                          {vision::kDvppDecodeResizeCropOperation, "resize_size"},
                                                          {vision::kDvppNormalizeOperation, "mean"},
                                                          {vision::kDvppNormalizeOperation, "std"},
                                                          {vision::kDvppResizeJpegOperation, "size"}};
#endif
};

// FIXME - Temporarily overload Execute to support both TensorOperation and TensorTransform
Execute::Execute(std::shared_ptr<TensorOperation> op, MapTargetDevice deviceType, uint32_t device_id) {
  ops_.emplace_back(std::move(op));
  device_type_ = deviceType;
  info_ = std::make_shared<ExtraInfo>();
#ifdef ENABLE_ACL
  if (device_type_ == MapTargetDevice::kAscend310) {
    device_resource_ = std::make_shared<AscendResource>();
    Status rc = device_resource_->InitResource(device_id);
    if (!rc.IsOk()) {
      device_resource_ = nullptr;
      MS_LOG(ERROR) << "Initialize Ascend310 resource fail";
    }
  }
#endif
}

Execute::Execute(std::shared_ptr<TensorTransform> op, MapTargetDevice deviceType, uint32_t device_id) {
  // Initialize the op and other context
  transforms_.emplace_back(op);

  info_ = std::make_shared<ExtraInfo>();
  device_type_ = deviceType;
#ifdef ENABLE_ACL
  if (device_type_ == MapTargetDevice::kAscend310) {
    device_resource_ = std::make_shared<AscendResource>();
    Status rc = device_resource_->InitResource(device_id);
    if (!rc.IsOk()) {
      device_resource_ = nullptr;
      MS_LOG(ERROR) << "Initialize Ascend310 resource fail";
    }
  }
#endif
}

Execute::Execute(std::reference_wrapper<TensorTransform> op, MapTargetDevice deviceType, uint32_t device_id) {
  // Initialize the transforms_ and other context
  std::shared_ptr<TensorOperation> operation = op.get().Parse();
  ops_.emplace_back(std::move(operation));

  info_ = std::make_shared<ExtraInfo>();
  info_->init_with_shared_ptr_ = false;
  device_type_ = deviceType;
#ifdef ENABLE_ACL
  if (device_type_ == MapTargetDevice::kAscend310) {
    device_resource_ = std::make_shared<AscendResource>();
    Status rc = device_resource_->InitResource(device_id);
    if (!rc.IsOk()) {
      device_resource_ = nullptr;
      MS_LOG(ERROR) << "Initialize Ascend310 resource fail";
    }
  }
#endif
}

// Execute function for the example case: auto decode(new vision::Decode());
Execute::Execute(TensorTransform *op, MapTargetDevice deviceType, uint32_t device_id) {
  // Initialize the transforms_ and other context
  std::shared_ptr<TensorTransform> smart_ptr_op(op);
  transforms_.emplace_back(smart_ptr_op);

  info_ = std::make_shared<ExtraInfo>();
  device_type_ = deviceType;
#ifdef ENABLE_ACL
  if (device_type_ == MapTargetDevice::kAscend310) {
    device_resource_ = std::make_shared<AscendResource>();
    Status rc = device_resource_->InitResource(device_id);
    if (!rc.IsOk()) {
      device_resource_ = nullptr;
      MS_LOG(ERROR) << "Initialize Ascend310 resource fail";
    }
  }
#endif
}

Execute::Execute(std::vector<std::shared_ptr<TensorOperation>> ops, MapTargetDevice deviceType, uint32_t device_id)
    : ops_(std::move(ops)), device_type_(deviceType) {
  info_ = std::make_shared<ExtraInfo>();
#ifdef ENABLE_ACL
  if (device_type_ == MapTargetDevice::kAscend310) {
    device_resource_ = std::make_shared<AscendResource>();
    Status rc = device_resource_->InitResource(device_id);
    if (!rc.IsOk()) {
      device_resource_ = nullptr;
      MS_LOG(ERROR) << "Initialize Ascend310 resource fail";
    }
  }
#endif
}

Execute::Execute(std::vector<std::shared_ptr<TensorTransform>> ops, MapTargetDevice deviceType, uint32_t device_id) {
  // Initialize the transforms_ and other context
  transforms_ = ops;

  info_ = std::make_shared<ExtraInfo>();
  device_type_ = deviceType;
#ifdef ENABLE_ACL
  if (device_type_ == MapTargetDevice::kAscend310) {
    device_resource_ = std::make_shared<AscendResource>();
    Status rc = device_resource_->InitResource(device_id);
    if (!rc.IsOk()) {
      device_resource_ = nullptr;
      MS_LOG(ERROR) << "Initialize Ascend310 resource fail";
    }
  }
#endif
}

Execute::Execute(const std::vector<std::reference_wrapper<TensorTransform>> ops, MapTargetDevice deviceType,
                 uint32_t device_id) {
  // Initialize the transforms_ and other context
  if (deviceType == MapTargetDevice::kCpu) {
    (void)std::transform(
      ops.begin(), ops.end(), std::back_inserter(ops_),
      [](TensorTransform &operation) -> std::shared_ptr<TensorOperation> { return operation.Parse(); });
  } else {
    for (auto &op : ops) {
      ops_.emplace_back(op.get().Parse(deviceType));
    }
  }

  info_ = std::make_shared<ExtraInfo>();
  info_->init_with_shared_ptr_ = false;
  device_type_ = deviceType;
#ifdef ENABLE_ACL
  if (device_type_ == MapTargetDevice::kAscend310) {
    device_resource_ = std::make_shared<AscendResource>();
    Status rc = device_resource_->InitResource(device_id);
    if (!rc.IsOk()) {
      device_resource_ = nullptr;
      MS_LOG(ERROR) << "Initialize Ascend310 resource fail";
    }
  }
#endif
}

// Execute function for the example vector case: auto decode(new vision::Decode());
Execute::Execute(const std::vector<TensorTransform *> &ops, MapTargetDevice deviceType, uint32_t device_id) {
  // Initialize the transforms_ and other context
  for (auto &op : ops) {
    std::shared_ptr<TensorTransform> smart_ptr_op(op);
    transforms_.emplace_back(smart_ptr_op);
  }

  info_ = std::make_shared<ExtraInfo>();
  device_type_ = deviceType;
#ifdef ENABLE_ACL
  if (device_type_ == MapTargetDevice::kAscend310) {
    device_resource_ = std::make_shared<AscendResource>();
    Status rc = device_resource_->InitResource(device_id);
    if (!rc.IsOk()) {
      device_resource_ = nullptr;
      MS_LOG(ERROR) << "Initialize Ascend310 resource fail";
    }
  }
#endif
}

Execute::~Execute() {
#ifdef ENABLE_ACL
  if (device_type_ == MapTargetDevice::kAscend310) {
    if (device_resource_) {
      device_resource_->FinalizeResource();
    } else {
      MS_LOG(ERROR) << "Device resource is nullptr which is illegal under case Ascend310";
    }
  }
#endif
}

Status Execute::operator()(const mindspore::MSTensor &input, mindspore::MSTensor *output) {
  // Validate input tensor
  CHECK_FAIL_RETURN_UNEXPECTED(input.DataSize() > 0, "Input Tensor has no data");
  CHECK_FAIL_RETURN_UNEXPECTED(validate_device_(), "Device Type should be 'Ascend310' or 'CPU'");

  // Parse TensorTransform transforms_ into TensorOperation ops_
  if (info_->init_with_shared_ptr_) {
    RETURN_IF_NOT_OK(ParseTransforms_());
    info_->init_with_shared_ptr_ = false;
  }
  CHECK_FAIL_RETURN_UNEXPECTED(!ops_.empty(), "Input TensorOperation should be provided");

  // Validate and build runtime ops
  std::vector<std::shared_ptr<TensorOp>> transforms;  // record the transformations

  std::map<MapTargetDevice, std::string> env_list = {
    {MapTargetDevice::kCpu, "kCpu"}, {MapTargetDevice::kGpu, "kGpu"}, {MapTargetDevice::kAscend310, "kAscend310"}};

  for (int32_t i = 0; i < ops_.size(); i++) {
    if (ops_[i] == nullptr) {
      MS_LOG(ERROR) << "Input TensorOperation["
                    << std::to_string(i) + "] is unsupported on your input device:" << env_list.at(device_type_);
    }
    CHECK_FAIL_RETURN_UNEXPECTED(ops_[i] != nullptr, "Input TensorOperation[" + std::to_string(i) + "] is null");
    RETURN_IF_NOT_OK(ops_[i]->ValidateParams());
    transforms.emplace_back(ops_[i]->Build());
  }

  if (device_type_ == MapTargetDevice::kCpu) {
    // Convert mindspore::Tensor to dataset::Tensor
    std::shared_ptr<dataset::Tensor> de_tensor;
    Status rc = dataset::Tensor::CreateFromMemory(dataset::TensorShape(input.Shape()),
                                                  MSTypeToDEType(static_cast<TypeId>(input.DataType())),
                                                  (const uchar *)(input.Data().get()), input.DataSize(), &de_tensor);
    if (rc.IsError()) {
      MS_LOG(ERROR) << rc;
      return rc;
    }

    // Apply transforms on tensor
    for (auto &t : transforms) {
      std::shared_ptr<dataset::Tensor> de_output;
      Status rc_ = t->Compute(de_tensor, &de_output);
      if (rc_.IsError()) {
        MS_LOG(ERROR) << rc_;
        return rc_;
      }

      // For next transform
      de_tensor = std::move(de_output);
    }

    // Convert dataset::Tensor to mindspore::Tensor
    CHECK_FAIL_RETURN_UNEXPECTED(de_tensor->HasData(), "Apply transform failed, output tensor has no data");
    *output = mindspore::MSTensor(std::make_shared<DETensor>(de_tensor));
  } else {  // Ascend310 case, where we must set Ascend resource on each operators
#ifdef ENABLE_ACL
    CHECK_FAIL_RETURN_UNEXPECTED(device_resource_, "Device resource is nullptr which is illegal under case Ascend310");
    // Sink data from host into device
    std::shared_ptr<mindspore::dataset::DeviceTensor> device_input;
    RETURN_IF_NOT_OK(device_resource_->Sink(input, &device_input));

    for (auto &t : transforms) {
      // Initialize AscendResource for each operators
      std::shared_ptr<DeviceTensor> device_output;
      RETURN_IF_NOT_OK(t->SetAscendResource(device_resource_));

      RETURN_IF_NOT_OK(t->Compute(device_input, &device_output));

      // For next transform
      device_input = std::move(device_output);
    }
    CHECK_FAIL_RETURN_UNEXPECTED(device_input->HasDeviceData(), "Apply transform failed, output tensor has no data");

    *output = mindspore::MSTensor(std::make_shared<DETensor>(device_input, true));
#endif
  }
  return Status::OK();
}

Status Execute::operator()(const std::vector<MSTensor> &input_tensor_list, std::vector<MSTensor> *output_tensor_list) {
  // Validate input tensor
  CHECK_FAIL_RETURN_UNEXPECTED(!input_tensor_list.empty(), "Input Tensor is not valid");
  for (auto &tensor : input_tensor_list) {
    CHECK_FAIL_RETURN_UNEXPECTED(tensor.DataSize() > 0, "Input Tensor has no data");
  }
  CHECK_FAIL_RETURN_UNEXPECTED(validate_device_(), "Device Type should be 'Ascend310' or 'CPU'");

  // Parse TensorTransform transforms_ into TensorOperation ops_
  if (info_->init_with_shared_ptr_) {
    RETURN_IF_NOT_OK(ParseTransforms_());
    info_->init_with_shared_ptr_ = false;
  }
  CHECK_FAIL_RETURN_UNEXPECTED(!ops_.empty(), "Input TensorOperation should be provided");

  std::map<MapTargetDevice, std::string> env_list = {
    {MapTargetDevice::kCpu, "kCpu"}, {MapTargetDevice::kGpu, "kGpu"}, {MapTargetDevice::kAscend310, "kAscend310"}};

  // Validate and build runtime ops
  std::vector<std::shared_ptr<TensorOp>> transforms;
  for (int32_t i = 0; i < ops_.size(); i++) {
    if (ops_[i] == nullptr) {
      MS_LOG(ERROR) << "Input TensorOperation["
                    << std::to_string(i) + "] is unsupported on your input device:" << env_list.at(device_type_);
    }
    CHECK_FAIL_RETURN_UNEXPECTED(ops_[i] != nullptr, "Input TensorOperation[" + std::to_string(i) + "] is null");
    RETURN_IF_NOT_OK(ops_[i]->ValidateParams());
    transforms.emplace_back(ops_[i]->Build());
  }
  if (device_type_ == MapTargetDevice::kCpu) {  // Case CPU
    TensorRow de_tensor_list;
    for (auto &tensor : input_tensor_list) {
      std::shared_ptr<dataset::Tensor> de_tensor;
      Status rc = dataset::Tensor::CreateFromMemory(
        dataset::TensorShape(tensor.Shape()), MSTypeToDEType(static_cast<TypeId>(tensor.DataType())),
        (const uchar *)(tensor.Data().get()), tensor.DataSize(), &de_tensor);
      if (rc.IsError()) {
        MS_LOG(ERROR) << rc;
        RETURN_IF_NOT_OK(rc);
      }
      de_tensor_list.emplace_back(std::move(de_tensor));
    }
    // Apply transforms on tensor
    for (auto &t : transforms) {
      TensorRow de_output_list;
      RETURN_IF_NOT_OK(t->Compute(de_tensor_list, &de_output_list));
      // For next transform
      de_tensor_list = std::move(de_output_list);
    }

    for (auto &tensor : de_tensor_list) {
      CHECK_FAIL_RETURN_UNEXPECTED(tensor->HasData(), "Apply transform failed, output tensor has no data");
      auto ms_tensor = mindspore::MSTensor(std::make_shared<DETensor>(tensor));
      output_tensor_list->emplace_back(ms_tensor);
    }
    CHECK_FAIL_RETURN_UNEXPECTED(!output_tensor_list->empty(), "Output Tensor is not valid");
  } else {  // Case Ascend310
#ifdef ENABLE_ACL
    CHECK_FAIL_RETURN_UNEXPECTED(device_resource_, "Device resource is nullptr which is illegal under case Ascend310");
    for (auto &input_tensor : input_tensor_list) {
      // Sink each data from host into device
      std::shared_ptr<dataset::DeviceTensor> device_input;
      RETURN_IF_NOT_OK(device_resource_->Sink(input_tensor, &device_input));

      for (auto &t : transforms) {
        std::shared_ptr<DeviceTensor> device_output;
        RETURN_IF_NOT_OK(t->SetAscendResource(device_resource_));

        RETURN_IF_NOT_OK(t->Compute(device_input, &device_output));

        // For next transform
        device_input = std::move(device_output);
      }
      CHECK_FAIL_RETURN_UNEXPECTED(device_input->HasDeviceData(), "Apply transform failed, output tensor has no data");
      // Due to the limitation of Ascend310 memory, we have to pop every data onto host memory
      // So the speed of this batch method is slower than solo mode
      std::shared_ptr<mindspore::dataset::Tensor> host_output;
      RETURN_IF_NOT_OK(device_resource_->Pop(device_input, &host_output));

      auto ms_tensor = mindspore::MSTensor(std::make_shared<DETensor>(host_output));
      output_tensor_list->emplace_back(ms_tensor);
      // Release the data on the device because we have copied one piece onto host
      RETURN_IF_NOT_OK(device_resource_->DeviceDataRelease());
    }
    CHECK_FAIL_RETURN_UNEXPECTED(!output_tensor_list->empty(), "Output Tensor vector is empty");
#endif
  }
  return Status::OK();
}

std::vector<uint32_t> AippSizeFilter(const std::vector<uint32_t> &resize_para, const std::vector<uint32_t> &crop_para) {
  std::vector<uint32_t> aipp_size;

  // Special condition where (no Crop and no Resize) or (no Crop and resize with fixed ratio) will lead to dynamic input
  if ((resize_para.size() == 0 || resize_para.size() == 1) && crop_para.size() == 0) {
    aipp_size = {0, 0};
    MS_LOG(WARNING) << "Dynamic input shape is not supported, incomplete aipp config file will be generated. Please "
                       "checkout your TensorTransform input, both src_image_size_h and src_image_size will be 0";
    return aipp_size;
  }

  if (resize_para.size() == 0) {  // If only Crop operator exists
    aipp_size = crop_para;
  } else if (crop_para.size() == 0) {  // If only Resize operator with 2 parameters exists
    aipp_size = resize_para;
  } else {  // If both of them exist
    if (resize_para.size() == 1) {
      aipp_size = crop_para;
    } else {
      aipp_size =
        *min_element(resize_para.begin(), resize_para.end()) < *min_element(crop_para.begin(), crop_para.end())
          ? resize_para
          : crop_para;
    }
  }

#ifdef ENABLE_ACL
  aipp_size[0] = DVPP_ALIGN_UP(aipp_size[0], VPC_HEIGHT_ALIGN);  // H
  aipp_size[1] = DVPP_ALIGN_UP(aipp_size[1], VPC_WIDTH_ALIGN);   // W
#endif
  return aipp_size;
}

std::vector<uint32_t> AippMeanFilter(const std::vector<uint32_t> &normalize_para) {
  std::vector<uint32_t> aipp_mean;
  if (normalize_para.size() == 6) {  // If Normalize operator exist
    std::transform(normalize_para.begin(), normalize_para.begin() + 3, std::back_inserter(aipp_mean),
                   [](uint32_t i) { return static_cast<uint32_t>(i / 10000); });
  } else {
    aipp_mean = {0, 0, 0};
  }
  return aipp_mean;
}

std::vector<float> AippStdFilter(const std::vector<uint32_t> &normalize_para) {
  std::vector<float> aipp_std;
  if (normalize_para.size() == 6) {  // If Normalize operator exist
    auto zeros = std::find(std::begin(normalize_para), std::end(normalize_para), 0);
    if (zeros == std::end(normalize_para)) {
      std::transform(normalize_para.begin() + 3, normalize_para.end(), std::back_inserter(aipp_std),
                     [](uint32_t i) { return 10000 / static_cast<float>(i); });
    } else {  // If 0 occurs in std vector
      MS_LOG(WARNING) << "Detect 0 in std vector, please verify your input";
      aipp_std = {1.0, 1.0, 1.0};
    }
  } else {
    aipp_std = {1.0, 1.0, 1.0};
  }
  return aipp_std;
}

Status AippInfoCollection(std::map<std::string, std::string> *aipp_options, const std::vector<uint32_t> &aipp_size,
                          const std::vector<uint32_t> &aipp_mean, const std::vector<float> &aipp_std) {
  // Several aipp config parameters
  aipp_options->insert(std::make_pair("related_input_rank", "0"));
  aipp_options->insert(std::make_pair("src_image_size_w", std::to_string(aipp_size[1])));
  aipp_options->insert(std::make_pair("src_image_size_h", std::to_string(aipp_size[0])));
  aipp_options->insert(std::make_pair("crop", "false"));
  aipp_options->insert(std::make_pair("input_format", "YUV420SP_U8"));
  aipp_options->insert(std::make_pair("aipp_mode", "static"));
  aipp_options->insert(std::make_pair("csc_switch", "true"));
  aipp_options->insert(std::make_pair("rbuv_swap_switch", "false"));
  // Y = AX + b,  this part is A
  std::vector<int32_t> color_space_matrix = {256, 0, 359, 256, -88, -183, 256, 454, 0};
  int count = 0;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      std::string key_word = "matrix_r" + std::to_string(i) + "c" + std::to_string(j);
      aipp_options->insert(std::make_pair(key_word, std::to_string(color_space_matrix[count])));
      ++count;
    }
  }
  // This part is b
  std::vector<uint32_t> color_space_bias = {0, 128, 128};
  for (int i = 0; i < 3; i++) {
    std::string key_word = "input_bias_" + std::to_string(i);
    aipp_options->insert(std::make_pair(key_word, std::to_string(color_space_bias[i])));
  }
  // Y = (X - mean - min) * [std^(-1)], this part is mean
  for (int i = 0; i < aipp_mean.size(); i++) {
    std::string key_word = "mean_chn_" + std::to_string(i);
    aipp_options->insert(std::make_pair(key_word, std::to_string(aipp_mean[i])));
  }
  // This part is min
  for (int i = 0; i < aipp_mean.size(); i++) {
    std::string key_word = "min_chn_" + std::to_string(i);
    aipp_options->insert(std::make_pair(key_word, "0.0"));
  }
  // This part is std^(-1)
  for (int i = 0; i < aipp_std.size(); i++) {
    std::string key_word = "var_reci_chn_" + std::to_string(i);
    aipp_options->insert(std::make_pair(key_word, std::to_string(aipp_std[i])));
  }
  return Status::OK();
}

std::string Execute::AippCfgGenerator() {
  std::string config_location = "./aipp.cfg";
#ifdef ENABLE_ACL
  if (info_->init_with_shared_ptr_) {
    ParseTransforms_();
    info_->init_with_shared_ptr_ = false;
  }
  std::vector<uint32_t> paras;  // Record the parameters value of each Ascend operators
  for (int32_t i = 0; i < ops_.size(); i++) {
    // Validate operator ir
    json ir_info;
    if (ops_[i] == nullptr) {
      MS_LOG(ERROR) << "Input TensorOperation[" + std::to_string(i) + "] is null";
      return "";
    }

    // Define map between operator name and parameter name
    ops_[i]->to_json(&ir_info);

    // Collect the information of operators
    for (auto pos = info_->op2para_map_.equal_range(ops_[i]->Name()); pos.first != pos.second; ++pos.first) {
      auto paras_key_word = pos.first->second;
      paras = ir_info[paras_key_word].get<std::vector<uint32_t>>();
      info_->aipp_cfg_.insert(std::make_pair(ops_[i]->Name(), paras));
    }
  }

  std::ofstream outfile;
  outfile.open(config_location, std::ofstream::out);

  if (!outfile.is_open()) {
    MS_LOG(ERROR) << "Fail to open Aipp config file, please verify your system config(including authority)"
                  << "We will return empty string which represent the location of Aipp config file in this case";
    std::string except = "";
    return except;
  }

  if (device_type_ == MapTargetDevice::kAscend310) {
    // Process resize parameters and crop parameters to find out the final size of input data
    std::vector<uint32_t> resize_paras;
    std::vector<uint32_t> crop_paras;

    // Find resize parameters
    std::map<std::string, std::vector<uint32_t>>::iterator iter;
    if (info_->aipp_cfg_.find(vision::kDvppResizeJpegOperation) != info_->aipp_cfg_.end()) {
      iter = info_->aipp_cfg_.find(vision::kDvppResizeJpegOperation);
      resize_paras = iter->second;
    } else if (info_->aipp_cfg_.find(vision::kDvppDecodeResizeOperation) != info_->aipp_cfg_.end()) {
      iter = info_->aipp_cfg_.find(vision::kDvppDecodeResizeOperation);
      resize_paras = iter->second;
    }

    // Find crop parameters
    if (info_->aipp_cfg_.find(vision::kDvppCropJpegOperation) != info_->aipp_cfg_.end()) {
      iter = info_->aipp_cfg_.find(vision::kDvppCropJpegOperation);
      crop_paras = iter->second;
    } else if (info_->aipp_cfg_.find(vision::kDvppDecodeResizeCropOperation) != info_->aipp_cfg_.end()) {
      iter = info_->aipp_cfg_.find(vision::kDvppDecodeResizeCropOperation);
      crop_paras = iter->second;
    }
    if (crop_paras.size() == 1) {
      crop_paras.emplace_back(crop_paras[0]);
    }

    std::vector<uint32_t> aipp_size = AippSizeFilter(resize_paras, crop_paras);

    // Process normalization parameters to find out the final normalization parameters for Aipp module
    std::vector<uint32_t> normalize_paras;
    if (info_->aipp_cfg_.find(vision::kDvppNormalizeOperation) != info_->aipp_cfg_.end()) {
      for (auto pos = info_->aipp_cfg_.equal_range(vision::kDvppNormalizeOperation); pos.first != pos.second;
           ++pos.first) {
        auto mean_or_std = pos.first->second;
        normalize_paras.insert(normalize_paras.end(), mean_or_std.begin(), mean_or_std.end());
      }
    }

    std::vector<uint32_t> aipp_mean = AippMeanFilter(normalize_paras);
    std::vector<float> aipp_std = AippStdFilter(normalize_paras);

    std::map<std::string, std::string> aipp_options;
    AippInfoCollection(&aipp_options, aipp_size, aipp_mean, aipp_std);

    std::string tab_char(4, ' ');
    outfile << "aipp_op {" << std::endl;
    for (auto &option : aipp_options) {
      outfile << tab_char << option.first << " : " << option.second << std::endl;
    }
    outfile << "}";
    outfile.close();
  } else {  // For case GPU or CPU
    outfile << "aipp_op {" << std::endl << "}";
    outfile.close();
    MS_LOG(WARNING) << "Your runtime environment is not Ascend310, this config file will lead to undefined behavior on "
                       "computing result. Please check that.";
  }
#endif
  return config_location;
}

bool IsEmptyPtr(std::shared_ptr<TensorTransform> api_ptr) { return api_ptr == nullptr; }

Status Execute::ParseTransforms_() {
  auto iter = std::find_if(transforms_.begin(), transforms_.end(), IsEmptyPtr);
  if (iter != transforms_.end()) {
    std::string err_msg = "Your input TensorTransforms contain at least one nullptr, please check your input";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_UNEXPECTED(err_msg);
  }

  if (device_type_ == MapTargetDevice::kCpu) {
    (void)std::transform(transforms_.begin(), transforms_.end(), std::back_inserter(ops_),
                         [](std::shared_ptr<TensorTransform> operation) -> std::shared_ptr<TensorOperation> {
                           return operation->Parse();
                         });
  } else {
    for (auto &transform_ : transforms_) {
      ops_.emplace_back(transform_->Parse(device_type_));
    }
  }

  return Status::OK();
}

Status Execute::validate_device_() {
  if (device_type_ != MapTargetDevice::kCpu && device_type_ != MapTargetDevice::kAscend310 &&
      device_type_ != MapTargetDevice::kGpu) {
    std::string err_msg = "Your input device is not supported. (Option: CPU or GPU or Ascend310)";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  return Status::OK();
}

Status Execute::DeviceMemoryRelease() {
  CHECK_FAIL_RETURN_UNEXPECTED(device_resource_, "Device resource is nullptr which is illegal under case Ascend310");
  Status rc = device_resource_->DeviceDataRelease();
  if (rc.IsError()) {
    std::string err_msg = "Error in device data release";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  return Status::OK();
}

}  // namespace dataset
}  // namespace mindspore
