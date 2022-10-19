/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "minddata/dataset/include/dataset/execute.h"

#include <algorithm>
#include <fstream>

#include "minddata/dataset/core/de_tensor.h"
#include "minddata/dataset/core/tensor_row.h"
#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/core/type_id.h"
#include "minddata/dataset/kernels/ir/tensor_operation.h"
#include "minddata/dataset/kernels/tensor_op.h"
#include "minddata/dataset/util/log_adapter.h"
#include "minddata/dataset/core/ascend_resource.h"
#include "minddata/dataset/kernels/image/dvpp/utils/CommonDataType.h"
#include "minddata/dataset/kernels/ir/vision/ascend_vision_ir.h"

namespace mindspore {
namespace dataset {
using json = nlohmann::json;

struct Execute::ExtraInfo {
  std::multimap<std::string, std::vector<uint32_t>> aipp_cfg_;
  bool init_with_shared_ptr_ = true;  // Initial execute object with shared_ptr as default
#if defined(WITH_BACKEND) || defined(ENABLE_ACL)
  std::multimap<std::string, std::string> op2para_map_ = {{vision::kDvppCropJpegOperation, "size"},
                                                          {vision::kDvppDecodeResizeOperation, "size"},
                                                          {vision::kDvppDecodeResizeCropOperation, "crop_size"},
                                                          {vision::kDvppDecodeResizeCropOperation, "resize_size"},
                                                          {vision::kDvppNormalizeOperation, "mean"},
                                                          {vision::kDvppNormalizeOperation, "std"},
                                                          {vision::kDvppResizeJpegOperation, "size"}};
#endif
};

Status Execute::InitResource(MapTargetDevice device_type, uint32_t device_id) {
  if (device_type_ == MapTargetDevice::kAscend310) {
#if defined(WITH_BACKEND) || defined(ENABLE_ACL)
    device_resource_ = std::make_shared<AscendResource>();
    Status rc = device_resource_->InitResource(device_id);
    if (!rc.IsOk()) {
      device_resource_ = nullptr;
      std::string err_msg = "Initialize Ascend310 resource fail";
      MS_LOG(ERROR) << err_msg;
      RETURN_STATUS_UNEXPECTED(err_msg);
    }
#endif
  }
  return Status::OK();
}

Execute::Execute(const std::shared_ptr<TensorOperation> &op, MapTargetDevice device_type, uint32_t device_id) {
  (void)ops_.emplace_back(op);
  device_type_ = device_type;
  info_ = std::make_shared<ExtraInfo>();
  (void)InitResource(device_type, device_id);
}

Execute::Execute(const std::shared_ptr<TensorTransform> &op, MapTargetDevice device_type, uint32_t device_id) {
  // Initialize the op and other context
  (void)transforms_.emplace_back(op);

  info_ = std::make_shared<ExtraInfo>();
  device_type_ = device_type;
  (void)InitResource(device_type, device_id);
}

Execute::Execute(const std::reference_wrapper<TensorTransform> &op, MapTargetDevice device_type, uint32_t device_id) {
  // Initialize the transforms_ and other context
  std::shared_ptr<TensorOperation> operation = op.get().Parse();
  (void)ops_.emplace_back(std::move(operation));

  info_ = std::make_shared<ExtraInfo>();
  info_->init_with_shared_ptr_ = false;
  device_type_ = device_type;
  (void)InitResource(device_type, device_id);
}

// Execute function for the example case: auto decode(new vision::Decode());
Execute::Execute(TensorTransform *op, MapTargetDevice device_type, uint32_t device_id) {
  // Initialize the transforms_ and other context
  (void)ops_.emplace_back(op->Parse());

  info_ = std::make_shared<ExtraInfo>();
  info_->init_with_shared_ptr_ = false;
  device_type_ = device_type;
  (void)InitResource(device_type, device_id);
}

Execute::Execute(const std::vector<std::shared_ptr<TensorOperation>> &ops, MapTargetDevice device_type,
                 uint32_t device_id)
    : ops_(ops), device_type_(device_type) {
  info_ = std::make_shared<ExtraInfo>();
  (void)InitResource(device_type, device_id);
}

Execute::Execute(const std::vector<std::shared_ptr<TensorTransform>> &ops, MapTargetDevice device_type,
                 uint32_t device_id) {
  // Initialize the transforms_ and other context
  transforms_ = ops;

  info_ = std::make_shared<ExtraInfo>();
  device_type_ = device_type;
  (void)InitResource(device_type, device_id);
}

Execute::Execute(const std::vector<std::reference_wrapper<TensorTransform>> &ops, MapTargetDevice device_type,
                 uint32_t device_id) {
  // Initialize the transforms_ and other context
  if (device_type == MapTargetDevice::kCpu) {
    (void)std::transform(
      ops.begin(), ops.end(), std::back_inserter(ops_),
      [](TensorTransform &operation) -> std::shared_ptr<TensorOperation> { return operation.Parse(); });
  } else {
    for (auto &op : ops) {
      (void)ops_.emplace_back(op.get().Parse(device_type));
    }
  }

  info_ = std::make_shared<ExtraInfo>();
  info_->init_with_shared_ptr_ = false;
  device_type_ = device_type;
  (void)InitResource(device_type, device_id);
}

// Execute function for the example vector case: auto decode(new vision::Decode());
Execute::Execute(const std::vector<TensorTransform *> &ops, MapTargetDevice device_type, uint32_t device_id) {
  // Initialize the transforms_ and other context
  (void)std::transform(
    ops.begin(), ops.end(), std::back_inserter(ops_),
    [](TensorTransform *operation) -> std::shared_ptr<TensorOperation> { return operation->Parse(); });

  info_ = std::make_shared<ExtraInfo>();
  info_->init_with_shared_ptr_ = false;
  device_type_ = device_type;
  (void)InitResource(device_type, device_id);
}

Execute::~Execute() {
  if (device_type_ == MapTargetDevice::kAscend310) {
    if (device_resource_) {
      auto rc = device_resource_->FinalizeResource();
      if (rc.IsError()) {
        MS_LOG(ERROR) << "Device resource release failed, error msg is " << rc;
      }
    } else {
      MS_LOG(ERROR) << "Device resource is nullptr which is illegal under case Ascend310";
    }
  }
}

Status Execute::BuildTransforms() {
  // Parse TensorTransform transforms_ into TensorOperation ops_
  if (info_->init_with_shared_ptr_) {
    RETURN_IF_NOT_OK(ParseTransforms());
    info_->init_with_shared_ptr_ = false;
  }
  CHECK_FAIL_RETURN_UNEXPECTED(!ops_.empty(), "Input TensorOperation should be provided.");

  std::map<MapTargetDevice, std::string> env_list = {
    {MapTargetDevice::kCpu, "kCpu"}, {MapTargetDevice::kGpu, "kGpu"}, {MapTargetDevice::kAscend310, "kAscend310"}};

  // Validate and build runtime ops
  for (size_t i = 0; i < ops_.size(); i++) {
    if (ops_[i] == nullptr) {
      std::string err_msg = "Input TensorOperation[" + std::to_string(i) +
                            "] is unsupported on your input device:" + env_list.at(device_type_);
      MS_LOG(ERROR) << err_msg;
      RETURN_STATUS_UNEXPECTED(err_msg);
    }
    RETURN_IF_NOT_OK(ops_[i]->ValidateParams());
    (void)transforms_rt_.emplace_back(ops_[i]->Build());
  }
  return Status::OK();
}

Status Execute::operator()(const mindspore::MSTensor &input, mindspore::MSTensor *output) {
  // Validate input tensor
  RETURN_UNEXPECTED_IF_NULL(output);
  CHECK_FAIL_RETURN_UNEXPECTED(input.DataSize() > 0, "Input Tensor has no data.");
  CHECK_FAIL_RETURN_UNEXPECTED(ValidateDevice(), "Device Type should be 'Ascend310' or 'CPU'.");

  if (!ops_created) {
    CHECK_FAIL_RETURN_UNEXPECTED(BuildTransforms(), "Building Transform ops failed!");
    ops_created = true;
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
    for (auto &t : transforms_rt_) {
      TensorRow de_tensor_row;
      TensorRow de_output_row;
      de_tensor_row.push_back(de_tensor);
      de_output_row.resize(1);
      Status rc_ = t->Compute(de_tensor_row, &de_output_row);
      if (rc_.IsError()) {
        MS_LOG(ERROR) << rc_;
        return rc_;
      }

      // For next transform
      de_tensor = std::move(de_output_row[0]);
    }

    // Convert dataset::Tensor to mindspore::Tensor
    if (!de_tensor->HasData()) {
      std::stringstream ss;
      ss << "Transformation returned an empty tensor with shape " << de_tensor->shape();
      RETURN_STATUS_UNEXPECTED(ss.str());
    }
    *output = mindspore::MSTensor(std::make_shared<DETensor>(de_tensor));
  } else if (device_type_ ==
             MapTargetDevice::kAscend310) {  // Ascend310 case, where we must set Ascend resource on each operations
#if defined(WITH_BACKEND) || defined(ENABLE_ACL)
    CHECK_FAIL_RETURN_UNEXPECTED(device_resource_, "Device resource is nullptr which is illegal under case Ascend310.");
    // Sink data from host into device
    std::shared_ptr<mindspore::dataset::DeviceTensor> device_input;
    RETURN_IF_NOT_OK(device_resource_->Sink(input, &device_input));

    for (auto &t : transforms_rt_) {
      // Initialize AscendResource for each operations
      std::shared_ptr<DeviceTensor> device_output;
      RETURN_IF_NOT_OK(t->SetAscendResource(device_resource_));

      RETURN_IF_NOT_OK(t->Compute(device_input, &device_output));

      // For next transform
      device_input = std::move(device_output);
    }
    CHECK_FAIL_RETURN_UNEXPECTED(device_input->HasDeviceData(), "Apply transform failed, output tensor has no data.");

    *output = mindspore::MSTensor(std::make_shared<DETensor>(device_input, true));
#endif
  } else {
    std::string err_msg = "Your input device is not supported. (Option: CPU or Ascend310)";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  return Status::OK();
}

Status Execute::operator()(const std::vector<MSTensor> &input_tensor_list, std::vector<MSTensor> *output_tensor_list) {
  // Validate input tensor
  RETURN_UNEXPECTED_IF_NULL(output_tensor_list);
  CHECK_FAIL_RETURN_UNEXPECTED(!input_tensor_list.empty(), "Input Tensor is not valid.");
  output_tensor_list->clear();
  for (auto &tensor : input_tensor_list) {
    CHECK_FAIL_RETURN_UNEXPECTED(tensor.DataSize() > 0, "Input Tensor has no data.");
  }
  CHECK_FAIL_RETURN_UNEXPECTED(ValidateDevice(), "Device Type should be 'Ascend310' or 'CPU'.");

  if (!ops_created) {
    CHECK_FAIL_RETURN_UNEXPECTED(BuildTransforms(), "Building Transform ops failed!");
    ops_created = true;
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
      (void)de_tensor_list.emplace_back(std::move(de_tensor));
    }
    // Apply transforms on tensor
    for (auto &t : transforms_rt_) {
      TensorRow de_output_list;
      RETURN_IF_NOT_OK(t->Compute(de_tensor_list, &de_output_list));
      // For next transform
      de_tensor_list = std::move(de_output_list);
    }
    int32_t idx = 0;
    for (auto &tensor : de_tensor_list) {
      if (!tensor->HasData()) {
        std::stringstream ss;
        ss << "Transformation returned an empty tensor at location " << idx << ". ";
        ss << "The shape of the tensor is " << tensor->shape();
        MS_LOG(WARNING) << ss.str();
      }
      auto ms_tensor = mindspore::MSTensor(std::make_shared<DETensor>(tensor));
      (void)output_tensor_list->emplace_back(ms_tensor);
      ++idx;
    }
    CHECK_FAIL_RETURN_UNEXPECTED(!output_tensor_list->empty(), "Output Tensor is not valid.");
  } else if (device_type_ ==
             MapTargetDevice::kAscend310) {  // Ascend310 case, where we must set Ascend resource on each operations
    CHECK_FAIL_RETURN_UNEXPECTED(device_resource_, "Device resource is nullptr which is illegal under case Ascend310.");
    for (auto &input_tensor : input_tensor_list) {
      // Sink each data from host into device
      std::shared_ptr<dataset::DeviceTensor> device_input;
      RETURN_IF_NOT_OK(device_resource_->Sink(input_tensor, &device_input));

      for (auto &t : transforms_rt_) {
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
      (void)output_tensor_list->emplace_back(ms_tensor);
      // Release the data on the device because we have copied one piece onto host
      RETURN_IF_NOT_OK(device_resource_->DeviceDataRelease());
    }
    CHECK_FAIL_RETURN_UNEXPECTED(!output_tensor_list->empty(), "Output Tensor vector is empty.");
  } else {
    std::string err_msg = "Your input device is not supported. (Option: CPU or Ascend310)";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  return Status::OK();
}

Status PyExecute::operator()(const std::shared_ptr<Tensor> &input_tensor, std::shared_ptr<Tensor> *out) {
  // Validate input tensors
  CHECK_FAIL_RETURN_UNEXPECTED(input_tensor->Size() > 0, "Input Tensor has no data.");
  RETURN_UNEXPECTED_IF_NULL(out);
  CHECK_FAIL_RETURN_UNEXPECTED(ValidateDevice(), "Device Type should be 'CPU'.");

  if (!ops_created) {
    CHECK_FAIL_RETURN_UNEXPECTED(BuildTransforms(), "Building Transform ops failed!");
    ops_created = true;
  }

  if (device_type_ == MapTargetDevice::kCpu) {
    TensorRow de_tensor_list({input_tensor});

    // Apply transforms on tensor
    for (auto &t : transforms_rt_) {
      TensorRow de_output_list;
      RETURN_IF_NOT_OK(t->Compute(de_tensor_list, &de_output_list));
      // For next transform
      de_tensor_list = std::move(de_output_list);
    }
    CHECK_FAIL_RETURN_UNEXPECTED(de_tensor_list.size() > 0,
                                 "[Internal] Transformation resulted in a tensor with size=0!");
    *out = std::move(de_tensor_list.getRow())[0];
  } else {
    std::string err_msg = "Your input device is not supported. (Option: CPU)";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  return Status::OK();
}

Status PyExecute::operator()(const std::vector<std::shared_ptr<Tensor>> &input_tensor_list,
                             std::vector<std::shared_ptr<Tensor>> *out) {
  // Validate input tensor
  CHECK_FAIL_RETURN_UNEXPECTED(!input_tensor_list.empty(), "Input Tensor is not valid.");
  RETURN_UNEXPECTED_IF_NULL(out);
  out->clear();
  for (auto &tensor : input_tensor_list) {
    CHECK_FAIL_RETURN_UNEXPECTED(tensor->Size() > 0, "Input Tensor has no data.");
  }
  CHECK_FAIL_RETURN_UNEXPECTED(ValidateDevice(), "Device Type should be 'CPU'.");

  if (!ops_created) {
    CHECK_FAIL_RETURN_UNEXPECTED(BuildTransforms(), "Building Transform ops failed!");
    ops_created = true;
  }

  if (device_type_ == MapTargetDevice::kCpu) {
    TensorRow de_tensor_list(input_tensor_list);

    // Apply transforms on tensor
    for (auto &t : transforms_rt_) {
      TensorRow de_output_list;
      RETURN_IF_NOT_OK(t->Compute(de_tensor_list, &de_output_list));
      // For next transform
      de_tensor_list = std::move(de_output_list);
    }
    *out = std::move(de_tensor_list.getRow());
    CHECK_FAIL_RETURN_UNEXPECTED(!out->empty(), "Output Tensor is not valid.");
  } else {
    std::string err_msg = "Your input device is not supported. (Option: CPU)";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  return Status::OK();
}

std::vector<uint32_t> AippSizeFilter(const std::vector<uint32_t> &resize_para, const std::vector<uint32_t> &crop_para) {
  std::vector<uint32_t> aipp_size;

  // Special condition where (no Crop and no Resize) or (no Crop and resize with fixed ratio) will lead to dynamic input
  if ((resize_para.empty() || resize_para.size() == 1) && crop_para.empty()) {
    aipp_size = {0, 0};
    MS_LOG(WARNING) << "Dynamic input shape is not supported, incomplete aipp config file will be generated. Please "
                       "checkout your TensorTransform input, both src_image_size_h and src_image_size will be 0.";
    return aipp_size;
  }

  if (resize_para.empty()) {  // If only Crop operation exists
    aipp_size = crop_para;
  } else if (crop_para.empty()) {  // If only Resize operation with 2 parameters exists
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

  aipp_size[0] = DVPP_ALIGN_UP(aipp_size[0], VPC_HEIGHT_ALIGN);  // H
  aipp_size[1] = DVPP_ALIGN_UP(aipp_size[1], VPC_WIDTH_ALIGN);   // W
  return aipp_size;
}

std::vector<uint32_t> AippMeanFilter(const std::vector<uint32_t> &normalize_para) {
  std::vector<uint32_t> aipp_mean;
  if (normalize_para.size() == 6) {  // If Normalize operation exist
    std::transform(normalize_para.begin(), normalize_para.begin() + 3, std::back_inserter(aipp_mean),
                   [](uint32_t i) { return static_cast<uint32_t>(i / 10000); });
  } else {
    aipp_mean = {0, 0, 0};
  }
  return aipp_mean;
}

std::vector<float> AippStdFilter(const std::vector<uint32_t> &normalize_para) {
  std::vector<float> aipp_std;
  if (normalize_para.size() == 6) {  // If Normalize operation exist
    auto zeros = std::find(std::begin(normalize_para), std::end(normalize_para), 0);
    if (zeros == std::end(normalize_para)) {
      if (std::any_of(normalize_para.begin() + 3, normalize_para.end(), [](uint32_t i) { return i == 0; })) {
        MS_LOG(ERROR) << "value in normalize para got 0.";
        return {};
      }
      std::transform(normalize_para.begin() + 3, normalize_para.end(), std::back_inserter(aipp_std),
                     [](uint32_t i) { return 10000 / static_cast<float>(i); });
    } else {  // If 0 occurs in std vector
      MS_LOG(WARNING) << "Detect 0 in std vector, please verify your input.";
      aipp_std = {1.0, 1.0, 1.0};
    }
  } else {
    aipp_std = {1.0, 1.0, 1.0};
  }
  return aipp_std;
}

Status AippInfoCollection(std::map<std::string, std::string> *aipp_options, const std::vector<uint32_t> &aipp_size,
                          const std::vector<uint32_t> &aipp_mean, const std::vector<float> &aipp_std) {
  RETURN_UNEXPECTED_IF_NULL(aipp_options);
  // Several aipp config parameters
  aipp_options->insert(std::make_pair("related_input_rank", "0"));
  aipp_options->insert(std::make_pair("src_image_size_w", std::to_string(aipp_size[1])));
  aipp_options->insert(std::make_pair("src_image_size_h", std::to_string(aipp_size[0])));
  aipp_options->insert(std::make_pair("crop", "false"));
  aipp_options->insert(std::make_pair("input_format", "YUV420SP_U8"));
  aipp_options->insert(std::make_pair("aipp_mode", "static"));
  aipp_options->insert(std::make_pair("csc_switch", "true"));
  aipp_options->insert(std::make_pair("rbuv_swap_switch", "false"));
  // Y = AX + b, this part is A
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
  if (info_ == nullptr) {
    MS_LOG(ERROR) << "info_ is null";
    return "";
  }
#if defined(WITH_BACKEND) || defined(ENABLE_ACL)
  if (info_->init_with_shared_ptr_) {
    auto rc = ParseTransforms();
    RETURN_SECOND_IF_ERROR(rc, "");
    info_->init_with_shared_ptr_ = false;
  }
  std::vector<uint32_t> paras;  // Record the parameters value of each Ascend operations
  for (int32_t i = 0; i < ops_.size(); i++) {
    // Validate operation ir
    json ir_info;
    if (ops_[i] == nullptr) {
      MS_LOG(ERROR) << "Input TensorOperation[" + std::to_string(i) + "] is null.";
      return "";
    }

    // Define map between operation name and parameter name
    auto rc = ops_[i]->to_json(&ir_info);
    if (rc.IsError()) {
      MS_LOG(ERROR) << "IR information serialize to json failed, error msg is " << rc;
      return "";
    }

    // Collect the information of operations
    for (auto pos = info_->op2para_map_.equal_range(ops_[i]->Name()); pos.first != pos.second; ++pos.first) {
      auto paras_key_word = pos.first->second;
      paras = ir_info[paras_key_word].get<std::vector<uint32_t>>();
      info_->aipp_cfg_.insert(std::make_pair(ops_[i]->Name(), paras));
    }
  }

  std::ofstream outfile;
  outfile.open(config_location, std::ofstream::out);

  if (!outfile.is_open()) {
    MS_LOG(ERROR) << "Fail to open Aipp config file, please verify your system config(including authority)."
                  << "We will return empty string which represent the location of Aipp config file in this case.";
    return "";
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
      (void)crop_paras.emplace_back(crop_paras[0]);
    }

    std::vector<uint32_t> aipp_size = AippSizeFilter(resize_paras, crop_paras);

    // Process Normalization parameters to find out the final Normalization parameters for Aipp module
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
    auto rc = AippInfoCollection(&aipp_options, aipp_size, aipp_mean, aipp_std);
    if (rc.IsError()) {
      MS_LOG(ERROR) << "Aipp information initialization failed, error msg is " << rc;
      outfile.close();
      return "";
    }

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

bool IsEmptyPtr(const std::shared_ptr<TensorTransform> &api_ptr) { return api_ptr == nullptr; }

Status Execute::ParseTransforms() {
  auto iter = std::find_if(transforms_.begin(), transforms_.end(), IsEmptyPtr);
  if (iter != transforms_.end()) {
    std::string err_msg = "Your input TensorTransforms contain at least one nullptr, please check your input.";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_UNEXPECTED(err_msg);
  }

  if (device_type_ == MapTargetDevice::kCpu) {
    (void)std::transform(transforms_.begin(), transforms_.end(), std::back_inserter(ops_),
                         [](const std::shared_ptr<TensorTransform> &operation) -> std::shared_ptr<TensorOperation> {
                           return operation->Parse();
                         });
  } else if (device_type_ == MapTargetDevice::kAscend310) {
    for (auto &transform_ : transforms_) {
      (void)ops_.emplace_back(transform_->Parse(device_type_));
    }
  } else {
    std::string err_msg = "Your input device is not supported. (Option: CPU or Ascend310)";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_UNEXPECTED(err_msg);
  }

  return Status::OK();
}

Status Execute::ValidateDevice() {
  if (device_type_ != MapTargetDevice::kCpu && device_type_ != MapTargetDevice::kAscend310 &&
      device_type_ != MapTargetDevice::kGpu) {
    std::string err_msg = "Your input device is not supported. (Option: CPU or GPU or Ascend310).";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  return Status::OK();
}

Status Execute::DeviceMemoryRelease() {
  CHECK_FAIL_RETURN_UNEXPECTED(device_resource_, "Device resource is nullptr which is illegal under case Ascend310.");
  Status rc = device_resource_->DeviceDataRelease();
  if (rc.IsError()) {
    std::string err_msg = "Error in device data release";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  return Status::OK();
}

Status Execute::Run(const std::vector<std::shared_ptr<dataset::Execute>> &data_graph,
                    const std::vector<mindspore::MSTensor> &inputs, std::vector<mindspore::MSTensor> *outputs) {
  RETURN_UNEXPECTED_IF_NULL(outputs);
  std::vector<MSTensor> transform_inputs = inputs;
  std::vector<MSTensor> transform_outputs;
  if (!data_graph.empty()) {
    for (const auto &exes : data_graph) {
      CHECK_FAIL_RETURN_UNEXPECTED(exes != nullptr, "Given execute object is null.");
      Status ret = exes->operator()(transform_inputs, &transform_outputs);
      if (ret != kSuccess) {
        MS_LOG(ERROR) << "Run preprocess failed:" << ret.GetErrDescription();
        return ret;
      }
      MS_LOG(DEBUG) << "transform_outputs[0].Shape: " << transform_outputs[0].Shape();
      transform_inputs = transform_outputs;
    }
    *outputs = std::move(transform_outputs);
  } else {
    std::string msg = "The set of Executors can not be empty.";
    MS_LOG(ERROR) << msg;
    RETURN_STATUS_UNEXPECTED(msg);
  }
  return Status::OK();
}

// In the current stage, there is a cyclic dependency between libmindspore.so and c_dataengine.so,
// we make a C function here and dlopen by libminspore.so to avoid linking explicitly,
// will be fix after decouling libminspore.so into multi submodules
extern "C" {
// ExecuteRun_C has C-linkage specified, but returns user-defined type 'mindspore::Status' which is incompatible with C
void ExecuteRun_C(const std::vector<std::shared_ptr<dataset::Execute>> &data_graph,
                  const std::vector<mindspore::MSTensor> &inputs, std::vector<mindspore::MSTensor> *outputs,
                  Status *s) {
  Status ret = Execute::Run(data_graph, inputs, outputs);
  if (s == nullptr) {
    return;
  }
  *s = Status(ret);
}
}
}  // namespace dataset
}  // namespace mindspore
