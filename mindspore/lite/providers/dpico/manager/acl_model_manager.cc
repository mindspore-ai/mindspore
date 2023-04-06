/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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

#include "manager/acl_model_manager.h"
#include <cstring>
#include <map>
#include <string>
#include <mutex>
#include "include/errorcode.h"
#include "common/check_base.h"
#include "common/infer_util.h"
#include "common/op_attr.h"
#include "src/custom_allocator.h"
#include "manager/acl_model_helper.h"
#include "manager/acl_buf_manager.h"
namespace mindspore {
namespace lite {
AllocatorPtr AclModelManager::custom_allocator_ = std::make_shared<CustomAllocator>();
CustomConfigManagerPtr AclModelManager::custom_config_manager_ptr_ = std::make_shared<CustomConfigManager>();
AclContextManagerPtr AclModelManager::acl_context_manager_ = std::make_shared<AclContextManager>();
static std::mutex acl_run_mutex;
namespace {
constexpr size_t kNumOfInputOm = 1;     // om parameter is the last input of MS Input tensor
constexpr size_t kMinAclInputSize = 2;  // {work_buf, task_buf}
constexpr size_t kDetectParamNum = 4;   // {nms_threshold, score_threshold,min_height,min_width}
}  // namespace
AclModelManager::~AclModelManager() {
  int ret = UnloadModel();
  MS_CHECK_TRUE_MSG_VOID(ret == RET_OK, "unload acl model failed.");
  ret = DestroyAclDataset(&acl_inputs_, inputs_mem_managed_by_tensor, custom_allocator_);
  MS_CHECK_TRUE_MSG_VOID(ret == RET_OK, "destroy acl inputs failed.");
  ret = DestroyAclDataset(&acl_outputs_, outputs_mem_managed_by_tensor, custom_allocator_);
  MS_CHECK_TRUE_MSG_VOID(ret == RET_OK, "destroy acl outputs failed.");
  inputs_mem_managed_by_tensor.clear();
  outputs_mem_managed_by_tensor.clear();
}

int AclModelManager::LoadModel(const std::vector<mindspore::MSTensor> &input_tensors) {
  MS_CHECK_TRUE_MSG(!input_tensors.empty(), RET_ERROR, "input tensors is empty.");
  MS_CHECK_TRUE_MSG(acl_model_ptr_ == nullptr, RET_ERROR, "acl model ptr has been allocated.");
  auto acl_model_tensor = input_tensors[input_tensors.size() - kNumOfInputOm];
  auto model_mem_ptr = acl_model_tensor.MutableData();
  MS_CHECK_TRUE_MSG(model_mem_ptr != nullptr, RET_ERROR, "model_mem_ptr is nullptr.");
  int ret = AclMalloc(&acl_model_ptr_, acl_model_tensor.DataSize());
  MS_CHECK_TRUE_MSG(ret == RET_OK, RET_ERROR, "svp acl rt malloc model buffer failed.");
  memcpy(acl_model_ptr_, model_mem_ptr, acl_model_tensor.DataSize());
  ret = svp_acl_mdl_load_from_mem(acl_model_ptr_, acl_model_tensor.DataSize(), &acl_model_id_);
  if (ret != SVP_ACL_SUCCESS) {
    MS_LOG(ERROR) << "svp acl mdl load from mem failed.";
    ret = AclFree(&acl_model_ptr_);
    MS_CHECK_TRUE_MSG(ret == SVP_ACL_SUCCESS, RET_ERROR, "AclFree acl_model_ptr_ failed");
    return RET_ERROR;
  }
  return RET_OK;
}

int AclModelManager::CreateModelDesc() {
  MS_CHECK_TRUE_MSG(acl_model_desc_ == nullptr, RET_ERROR, "model_desc has been created.");
  acl_model_desc_ = svp_acl_mdl_create_desc();
  MS_CHECK_TRUE_MSG(acl_model_desc_ != nullptr, RET_ERROR, "create model desc failed.");
  int ret = svp_acl_mdl_get_desc(acl_model_desc_, acl_model_id_);
  MS_CHECK_TRUE_MSG(ret == SVP_ACL_SUCCESS, RET_ERROR, "get model desc failed.");
  return RET_OK;
}

int AclModelManager::GetMaxTaskAndWorkBufSize() {
  size_t input_size = svp_acl_mdl_get_num_inputs(acl_model_desc_);
  MS_CHECK_TRUE_MSG(input_size > kMinAclInputSize, RET_ERROR,
                    "acl model input size should be greater than " << kMinAclInputSize);
  AclDataInfo acl_data_info(AclDataInfo::Input);
  int ret = GetAclDataInfo(&acl_data_info, acl_model_desc_, input_size - 2);
  MS_CHECK_TRUE_MSG(ret == RET_OK, RET_ERROR, "get acl data info failed.");
  ret = AclBufManager::GetInstance()->UpdateTaskBufSize(acl_data_info.data_size);
  MS_CHECK_TRUE_MSG(ret == RET_OK, RET_ERROR, "update task buf max size failed.");
  ret = GetAclDataInfo(&acl_data_info, acl_model_desc_, input_size - 1);
  MS_CHECK_TRUE_MSG(ret == RET_OK, RET_ERROR, "get acl data info failed.");
  ret = AclBufManager::GetInstance()->UpdateWorkBufSize(acl_data_info.data_size);
  MS_CHECK_TRUE_MSG(ret == RET_OK, RET_ERROR, "update work buf max size failed.");
  return RET_OK;
}

int AclModelManager::SetDetectParams(void *data) {
  MS_CHECK_TRUE_MSG(data != nullptr, RET_ERROR, "detect param data is nullptr");
  auto param = reinterpret_cast<float *>(data);
  param[kNmsThreshold] = custom_config_manager_ptr_->NmsThreshold();
  param[kScoreThreshold] = custom_config_manager_ptr_->ScoreThreshold();
  param[kMinHeight] = custom_config_manager_ptr_->MinHeight();
  param[kMinWidth] = custom_config_manager_ptr_->MinWidth();
  return RET_OK;
}

int AclModelManager::AddDetectParamInput() {
  void *data = nullptr;
  size_t detect_param_stride = sizeof(float) * kDetectParamNum;
  int ret = AclMalloc(&data, actual_batch_size_ * detect_param_stride);
  MS_CHECK_TRUE_MSG(ret == RET_OK, RET_ERROR, "svp acl rt malloc detect input buffer failed.");
  for (size_t loop = 0; loop < actual_batch_size_; loop++) {
    ret = SetDetectParams(reinterpret_cast<uint8_t *>(data) + loop * detect_param_stride);
    MS_CHECK_TRUE_MSG(ret == RET_OK, RET_ERROR, "set detect params failed.");
  }
  AclDataInfo acl_data_info(AclDataInfo::Input);
  acl_data_info.data_size = sizeof(float) * kDetectParamNum;
  acl_data_info.stride = sizeof(float) * kDetectParamNum;
  ret = AddDatasetBuffer(acl_inputs_, acl_data_info.data_size, acl_data_info.stride, data);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "add dataset buffer failed.";
    ret = AclFree(&data);
    MS_CHECK_TRUE_MSG(ret == SVP_ACL_SUCCESS, RET_ERROR, "AclFree data failed");
    return RET_ERROR;
  }
  size_t cur_num_buffers = svp_acl_mdl_get_dataset_num_buffers(acl_inputs_);
  MS_CHECK_TRUE_MSG(cur_num_buffers - 1 == inputs_mem_managed_by_tensor.size(), RET_ERROR,
                    "detect param index is invalid");
  inputs_mem_managed_by_tensor[cur_num_buffers - 1] = false;
  return RET_OK;
}

int AclModelManager::DetectPostProcess(mindspore::MSTensor *detect_output_tensor) {
  MS_CHECK_TRUE_MSG(detect_output_tensor != nullptr, RET_ERROR, "detect_output_tensor is nullptr.");
  std::vector<std::vector<float>> valid_det_boxes;
  int ret = ComputeValidDetectBoxes(acl_model_desc_, acl_outputs_, &valid_det_boxes);
  MS_CHECK_TRUE_MSG(ret == RET_OK, RET_ERROR, "compute valid detect boxes failed.");
  ret = WriteDetBoxesToTensorData(valid_det_boxes, detect_output_tensor);
  MS_CHECK_TRUE_MSG(ret == RET_OK, RET_ERROR, "write det boxes to detect output tensor data failed.");
  return RET_OK;
}

int AclModelManager::CreateTaskBufAndWorkBuf() {
  size_t input_size = svp_acl_mdl_get_num_inputs(acl_model_desc_);
  MS_CHECK_TRUE_MSG(input_size > kMinAclInputSize, RET_ERROR,
                    "acl model input size should be greater than " << kMinAclInputSize);
  size_t cur_num_buffers = svp_acl_mdl_get_dataset_num_buffers(acl_inputs_);
  MS_CHECK_TRUE_MSG(cur_num_buffers > 0, RET_ERROR, "acl model input size should be greater than 0");
  for (size_t i = cur_num_buffers; i < input_size; i++) {
    AclDataInfo acl_data_info(AclDataInfo::Input);
    int ret = GetAclDataInfo(&acl_data_info, acl_model_desc_, i);
    MS_CHECK_TRUE_MSG(ret == RET_OK, RET_ERROR, "get acl data info failed.");
    if (i == cur_num_buffers) {
      ret = AddDatasetBuffer(acl_inputs_, acl_data_info.data_size, acl_data_info.stride,
                             AclBufManager::GetInstance()->GetTaskBufPtr());
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "add dataset buffer failed.";
        return RET_ERROR;
      }
    } else {
      ret = AddDatasetBuffer(acl_inputs_, acl_data_info.data_size, acl_data_info.stride,
                             AclBufManager::GetInstance()->GetWorkBufPtr());
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "add dataset buffer failed.";
        return RET_ERROR;
      }
    }
    inputs_mem_managed_by_tensor[i] = true;
  }
  return RET_OK;
}

int AclModelManager::CreateNoShareTaskBufAndWorkBuf() {
  size_t input_size = svp_acl_mdl_get_num_inputs(acl_model_desc_);
  MS_CHECK_TRUE_MSG(input_size > kMinAclInputSize, RET_ERROR,
                    "acl model input size should be greater than " << kMinAclInputSize);
  size_t cur_num_buffers = svp_acl_mdl_get_dataset_num_buffers(acl_inputs_);
  MS_CHECK_TRUE_MSG(cur_num_buffers > 0, RET_ERROR, "acl model input size should be greater than 0");
  for (size_t i = cur_num_buffers; i < input_size; i++) {
    void *data = nullptr;
    AclDataInfo acl_data_info(AclDataInfo::Input);
    int ret = GetAclDataInfo(&acl_data_info, acl_model_desc_, i);
    MS_CHECK_TRUE_MSG(ret == RET_OK, RET_ERROR, "get acl data info failed.");
    ret = AclMalloc(&data, acl_data_info.data_size);
    MS_CHECK_TRUE_MSG(ret == RET_OK, RET_ERROR, "svp acl rt malloc task and work buf failed.");
    ret = AddDatasetBuffer(acl_inputs_, acl_data_info.data_size, acl_data_info.stride, data);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "add dataset buffer failed.";
      ret = AclFree(&data);
      MS_CHECK_TRUE_MSG(ret == SVP_ACL_SUCCESS, RET_ERROR, "AclFree data failed");
      return RET_ERROR;
    }
    inputs_mem_managed_by_tensor[i] = false;
  }
  return RET_OK;
}

int AclModelManager::CopyTensorDataToAclInputs(const std::vector<mindspore::MSTensor> &input_tensors) {
  for (size_t i = 0; i + kNumOfInputOm < input_tensors.size(); i++) {
    MS_CHECK_TRUE_MSG(inputs_mem_managed_by_tensor.find(i) != inputs_mem_managed_by_tensor.end(), RET_ERROR,
                      "invalid input index");
    if (inputs_mem_managed_by_tensor[i]) {
      continue;
    }

    auto input_tensor = input_tensors.at(i);
    AclDataInfo acl_data_info{AclDataInfo::Input};
    int ret = GetAclDataInfo(&acl_data_info, acl_model_desc_, i);
    MS_CHECK_TRUE_MSG(ret == RET_OK, RET_ERROR, "get acl data info failed.");
    auto *input_tensor_data = input_tensor.MutableData();
    MS_CHECK_TRUE_MSG(input_tensor_data != nullptr, RET_ERROR, "input tensor data is nullptr.");

    MS_CHECK_TRUE_MSG(!input_tensor.Shape().empty(), RET_ERROR, "input tensor shape is empty");
    size_t input_tensor_last_dim = input_tensor.Shape().back();
    size_t type_size = GetDataTypeSize(acl_model_desc_, i, acl_data_info.data_mode);
    auto input_tensor_stride = input_tensor_last_dim * type_size;

    auto *acl_data_buffer = svp_acl_mdl_get_dataset_buffer(acl_inputs_, i);
    MS_CHECK_TRUE_MSG(acl_data_buffer != nullptr, RET_ERROR, "get acl data buffer failed.");
    auto *acl_data = svp_acl_get_data_buffer_addr(acl_data_buffer);
    MS_CHECK_TRUE_MSG(acl_data != nullptr, RET_ERROR, "acl data is nullptr.");

    MS_CHECK_TRUE_MSG(acl_data_info.stride != 0, RET_ERROR, "acl stride cannot be 0");
    int64_t loop_times = acl_data_info.data_size * actual_batch_size_ / acl_data_info.stride;
    if (acl_model_type_ == AclModelType::kRecurrent) {
      if (i == 0) {  // e.g: tensor shape is (3, 1, 29), acl dims is (1024, 1, 29)
        loop_times = loop_times / acl_data_info.dim_info.dims[0] * custom_config_manager_ptr_->GTotalT();
      } else if (i == 1) {  // e.g: tensor shape is (3, 1, 1), acl dims is (1, 1, 1024)
        input_tensor_stride = custom_config_manager_ptr_->GTotalT() * type_size;
      }
    }

    // copy tensor data to acl inputs with the last dim(bytes) as a unit
    for (int64_t loop = 0; loop < loop_times; loop++) {
      memcpy(reinterpret_cast<uint8_t *>(acl_data) + loop * acl_data_info.stride,
             reinterpret_cast<uint8_t *>(input_tensor_data) + loop * input_tensor_stride, input_tensor_stride);
    }
  }
  return RET_OK;
}

int AclModelManager::CopyAclOutputsToTensorData(const std::vector<mindspore::MSTensor> &output_tensors) {
  for (size_t i = 0; i < output_tensors.size(); i++) {
    MS_CHECK_TRUE_MSG(outputs_mem_managed_by_tensor.find(i) != outputs_mem_managed_by_tensor.end(), RET_ERROR,
                      "invalid output index");
    if (i == 1 && custom_config_manager_ptr_->NeedDetectPostProcess() && acl_model_type_ == kRoi) {
      constexpr size_t kMinDetectOutputSize = 2;
      MS_CHECK_TRUE_MSG(output_tensors.size() >= kMinDetectOutputSize, RET_ERROR,
                        "detection net output size shouldn't be less than " << kMinDetectOutputSize);
      auto detect_boxes_tensor = output_tensors.at(i);
      int ret = DetectPostProcess(&detect_boxes_tensor);
      MS_CHECK_TRUE_MSG(ret == RET_OK, RET_ERROR, "detect net post process failed.");
      continue;
    }

    if (outputs_mem_managed_by_tensor[i]) {
      continue;
    }
    auto output_tensor = output_tensors.at(i);
    AclDataInfo acl_data_info{AclDataInfo::Output};
    int ret = GetAclDataInfo(&acl_data_info, acl_model_desc_, i);
    MS_CHECK_TRUE_MSG(ret == RET_OK, RET_ERROR, "get acl data info failed.");
    auto *output_tensor_data = output_tensor.MutableData();
    MS_CHECK_TRUE_MSG(output_tensor_data != nullptr, RET_ERROR, "output tensor data is nullptr.");

    MS_CHECK_TRUE_MSG(!output_tensor.Shape().empty(), RET_ERROR, "output tensor shape is empty");
    size_t output_tensor_last_dim = output_tensor.Shape().back();
    size_t type_size = GetDataTypeSize(acl_model_desc_, i, acl_data_info.data_mode);
    auto output_tensor_stride = output_tensor_last_dim * type_size;

    auto *acl_data_buffer = svp_acl_mdl_get_dataset_buffer(acl_outputs_, i);
    MS_CHECK_TRUE_MSG(acl_data_buffer != nullptr, RET_ERROR, "get acl data buffer failed.");
    auto *acl_data = svp_acl_get_data_buffer_addr(acl_data_buffer);
    MS_CHECK_TRUE_MSG(acl_data != nullptr, RET_ERROR, "acl data is nullptr.");

    MS_CHECK_TRUE_MSG(acl_data_info.stride != 0, RET_ERROR, "acl stride cannot be 0");
    int64_t loop_times = acl_data_info.data_size * actual_batch_size_ / acl_data_info.stride;
    if (acl_model_type_ == AclModelType::kRecurrent && i == 0) {  // RNN input_0's dims[0] isn't equal to gTotalT
      loop_times = loop_times / acl_data_info.dim_info.dims[0] * custom_config_manager_ptr_->GTotalT();
    }
    for (int64_t loop = 0; loop < loop_times; loop++) {
      memcpy(reinterpret_cast<uint8_t *>(output_tensor_data) + loop * output_tensor_stride,
             reinterpret_cast<uint8_t *>(acl_data) + loop * acl_data_info.stride, output_tensor_stride);
    }
  }
  return RET_OK;
}

int AclModelManager::FlushAclInputsAndOutputs() {
  // flush acl inputs
  auto dataset_buffer_size = svp_acl_mdl_get_dataset_num_buffers(acl_inputs_);
  for (size_t i = 0; i < dataset_buffer_size; i++) {
    if (!inputs_mem_managed_by_tensor[i]) {
      continue;
    }
    auto *acl_data_buffer = svp_acl_mdl_get_dataset_buffer(acl_inputs_, i);
    MS_CHECK_TRUE_MSG(acl_data_buffer != nullptr, RET_ERROR, "get acl data buffer failed.");
    auto input_size = svp_acl_get_data_buffer_size(acl_data_buffer);
    void *data = svp_acl_get_data_buffer_addr(acl_data_buffer);
    MS_CHECK_TRUE_MSG(data != nullptr, RET_ERROR, "acl data is nullptr");
    auto ret = svp_acl_rt_mem_flush(data, input_size);
    MS_CHECK_TRUE_MSG(ret == SVP_ACL_SUCCESS, RET_ERROR, "flush input tensor failed.");
  }

  // flush acl outputs
  dataset_buffer_size = svp_acl_mdl_get_dataset_num_buffers(acl_outputs_);
  for (size_t i = 0; i < dataset_buffer_size; i++) {
    if (!outputs_mem_managed_by_tensor[i]) {
      continue;
    }
    auto *acl_data_buffer = svp_acl_mdl_get_dataset_buffer(acl_outputs_, i);
    MS_CHECK_TRUE_MSG(acl_data_buffer != nullptr, RET_ERROR, "get acl data buffer failed.");
    auto output_size = svp_acl_get_data_buffer_size(acl_data_buffer);
    void *data = svp_acl_get_data_buffer_addr(acl_data_buffer);
    MS_CHECK_TRUE_MSG(data != nullptr, RET_ERROR, "acl data is nullptr");
    auto ret = svp_acl_rt_mem_invalidate(data, output_size);
    MS_CHECK_TRUE_MSG(ret == SVP_ACL_SUCCESS, RET_ERROR, "invalidate input tensor failed.");
  }
  return RET_OK;
}

int AclModelManager::AclModelRun(const std::vector<mindspore::MSTensor> &input_tensors) {
  std::unique_lock<std::mutex> lock(acl_run_mutex);
  int ret;
  ret = svp_acl_rt_set_device(acl_device_id_);
  MS_CHECK_TRUE_MSG(ret == SVP_ACL_SUCCESS, RET_ERROR, "svp acl rt set device failed.");
  if (acl_model_type_ != AclModelType::kRecurrent) {
    size_t index;
    ret = svp_acl_mdl_get_input_index_by_name(acl_model_desc_, SVP_ACL_DYNAMIC_TENSOR_NAME, &index);
    MS_CHECK_TRUE_MSG(ret == SVP_ACL_SUCCESS, RET_ERROR, "acl get input index by name failed.");

    ret = svp_acl_mdl_set_dynamic_batch_size(acl_model_id_, acl_inputs_, index, actual_batch_size_);
    MS_CHECK_TRUE_MSG(ret == SVP_ACL_SUCCESS, RET_ERROR, "acl set dynamic batch size failed.");
  } else {
    ret = svp_acl_mdl_set_total_t(acl_model_id_, acl_inputs_, custom_config_manager_ptr_->GTotalT());
    MS_CHECK_TRUE_MSG(ret == SVP_ACL_SUCCESS, RET_ERROR, "acl set total t failed.");
  }
  ret = svp_acl_mdl_execute(acl_model_id_, acl_inputs_, acl_outputs_);
  MS_CHECK_TRUE_MSG(ret == SVP_ACL_SUCCESS, RET_ERROR, "svp acl execute failed.");
  ret = svp_acl_rt_reset_device(acl_device_id_);
  MS_CHECK_TRUE_MSG(ret == SVP_ACL_SUCCESS, RET_ERROR, "acl reset device failed.");
  return RET_OK;
}

int AclModelManager::UnloadModel() {
  int ret = svp_acl_mdl_unload(acl_model_id_);
  MS_CHECK_TRUE_MSG(ret == SVP_ACL_SUCCESS, RET_ERROR, "svp acl unload model failed");
  if (acl_model_desc_ != nullptr) {
    ret = svp_acl_mdl_destroy_desc(acl_model_desc_);
    MS_CHECK_TRUE_MSG(ret == SVP_ACL_SUCCESS, RET_ERROR, "svp acl destroy model desc failed.");
    acl_model_desc_ = nullptr;
  }
  if (acl_model_ptr_ != nullptr) {
    ret = AclFree(&acl_model_ptr_);
    MS_CHECK_TRUE_MSG(ret == SVP_ACL_SUCCESS, RET_ERROR, "AclFree acl_model_ptr_ failed");
  }
  return RET_OK;
}

int AclModelManager::Init(const std::map<std::string, std::string> &dpico_config,
                          const std::map<std::string, std::string> &model_share_config,
                          const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &input_tensors,
                          const std::vector<mindspore::MSTensor> &output_tensors) {
  MS_CHECK_TRUE_MSG(acl_context_manager_ != nullptr, RET_ERROR, "acl_context_manager_ is nullptr.");
  MS_CHECK_TRUE_MSG(custom_config_manager_ptr_ != nullptr, RET_ERROR, "custom_config_manager_ptr_ is nullptr.");
  if (custom_config_manager_ptr_->Init(dpico_config) != RET_OK) {
    MS_LOG(ERROR) << "custom config manager init failed";
    return RET_ERROR;
  }
  if (acl_context_manager_->Init(custom_config_manager_ptr_->AclConfigFile()) != RET_OK) {
    MS_LOG(ERROR) << "acl context manager init failed.";
    return RET_ERROR;
  }
  if (GetAclModelType(primitive, &acl_model_type_) != RET_OK) {
    MS_LOG(ERROR) << "get acl model type failed.";
    return RET_ERROR;
  }
  if (LoadModel(input_tensors) != RET_OK) {
    MS_LOG(ERROR) << "load acl model failed.";
    return RET_ERROR;
  }
  if (CreateModelDesc() != RET_OK) {
    MS_LOG(ERROR) << "create model desc failed.";
    return RET_ERROR;
  }
  if (!custom_config_manager_ptr_->IsEnableMultiModelSharingMemPrepare(model_share_config)) {
    MS_LOG(INFO) << "MultiModelSharingMemPrepare function not open, do not need to model share.";
  } else {
    if (GetMaxTaskAndWorkBufSize() != RET_OK) {
      MS_LOG(ERROR) << "get max task and work buffer size failed.";
      return RET_ERROR;
    }
  }

  return RET_OK;
}

int AclModelManager::UpdateBatchSize(const std::vector<mindspore::MSTensor> &input_tensors) {
  auto input_tensor = input_tensors.front();
  auto input_shape = input_tensor.Shape();
  if (input_shape.empty()) {
    MS_LOG(ERROR) << "input shape is empty. " << input_tensor.Name();
    return RET_ERROR;
  }

  svp_acl_mdl_io_dims acl_mdl_input_0_dims;
  int ret = svp_acl_mdl_get_input_dims(acl_model_desc_, 0, &acl_mdl_input_0_dims);
  MS_CHECK_TRUE_MSG(ret == SVP_ACL_SUCCESS, RET_ERROR, "acl get input dims failed.");
  auto ms_input_batch = static_cast<size_t>(input_shape.front());
  if (input_shape.size() == 1 && ms_input_batch > 1) {
    MS_LOG(ERROR) << "When input dim is 1, batch size can't be more than 1";
    return RET_ERROR;
  }
  if (acl_model_type_ == AclModelType::kRecurrent) {
    custom_config_manager_ptr_->SetGTotalT(ms_input_batch);
  } else {
    MS_CHECK_TRUE_MSG(acl_mdl_input_0_dims.dim_count > 0 && acl_mdl_input_0_dims.dims[0] != 0, RET_ERROR,
                      "acl input 0 dims is invalid.");
    actual_batch_size_ = ms_input_batch / acl_mdl_input_0_dims.dims[0];
  }
  return RET_OK;
}

int AclModelManager::PrepareAclInputs(std::vector<mindspore::MSTensor> *input_tensors) {
  MS_CHECK_TRUE_MSG(acl_model_desc_ != nullptr, RET_ERROR, "acl model desc is nullptr.");
  MS_CHECK_TRUE_MSG(custom_allocator_ != nullptr, RET_ERROR, "custom allocator is nullptr.");
  MS_CHECK_TRUE_MSG(input_tensors != nullptr, RET_ERROR, "input_tensors is nullptr.");
  int ret;
  if (acl_inputs_ != nullptr) {
    ret = DestroyAclDataset(&acl_inputs_, inputs_mem_managed_by_tensor, custom_allocator_);
    MS_CHECK_TRUE_MSG(ret == RET_OK, RET_ERROR, "destroy acl input dataset failed.");
    inputs_mem_managed_by_tensor.clear();
  }
  acl_inputs_ = svp_acl_mdl_create_dataset();
  MS_CHECK_TRUE_MSG(acl_inputs_ != nullptr, RET_ERROR, "create acl model input dataset failed.");
  for (size_t i = 0; i < input_tensors->size() - kNumOfInputOm; i++) {
    AclDataInfo acl_data_info(AclDataInfo::Input);
    ret = GetAclDataInfo(&acl_data_info, acl_model_desc_, i);
    MS_CHECK_TRUE_MSG(ret == RET_OK, RET_ERROR, "get acl data info failed.");
    void *data = nullptr;
    if (acl_data_info.data_size * actual_batch_size_ != input_tensors->at(i).DataSize()) {
      inputs_mem_managed_by_tensor[i] = false;
      MS_LOG(INFO) << "The size of the last dimension of the input tensor "
                   << "does not align with 'internal_stride' value, will memcpy";
      ret = AclMalloc(&data, acl_data_info.data_size * actual_batch_size_);
      MS_CHECK_TRUE_MSG(ret == RET_OK, RET_ERROR, "svp acl rt malloc acl input buffer failed.");
    } else {
      inputs_mem_managed_by_tensor[i] = true;
      MS_LOG(INFO) << "The size of the last dimension of the input tensor "
                   << "is equal to 'internal_stride' value, will not memcpy";
      input_tensors->at(i).SetAllocator(custom_allocator_);
      data = input_tensors->at(i).MutableData();  // svp malloc memory for ms tensor
    }
    MS_CHECK_TRUE_MSG(data != nullptr, RET_ERROR, "get data ptr failed.");
    ret = AddDatasetBuffer(acl_inputs_, acl_data_info.data_size * actual_batch_size_, acl_data_info.stride, data);
    MS_CHECK_TRUE_MSG(ret == RET_OK, RET_ERROR, "add dataset buffer failed.");
  }
  if (acl_model_type_ == AclModelType::kRoi) {
    ret = AddDetectParamInput();
    MS_CHECK_TRUE_MSG(ret == RET_OK, RET_ERROR, "add detect param input failed.");
  }
  return RET_OK;
}

int AclModelManager::PrepareAclOutputs(std::vector<mindspore::MSTensor> *output_tensors) {
  MS_CHECK_TRUE_MSG(acl_model_desc_ != nullptr, RET_ERROR, "acl model desc is nullptr.");
  MS_CHECK_TRUE_MSG(custom_allocator_ != nullptr, RET_ERROR, "custom allocator is nullptr.");
  MS_CHECK_TRUE_MSG(output_tensors != nullptr, RET_ERROR, "output_tensors is nullptr.");
  int ret;
  if (acl_outputs_ != nullptr) {
    ret = DestroyAclDataset(&acl_outputs_, outputs_mem_managed_by_tensor, custom_allocator_);
    MS_CHECK_TRUE_MSG(ret == RET_OK, RET_ERROR, "destroy acl output dataset failed.");
    outputs_mem_managed_by_tensor.clear();
  }
  acl_outputs_ = svp_acl_mdl_create_dataset();
  MS_CHECK_TRUE_MSG(acl_outputs_ != nullptr, RET_ERROR, "create acl model output dataset failed.");
  size_t output_size = svp_acl_mdl_get_num_outputs(acl_model_desc_);
  MS_CHECK_TRUE_MSG(output_size == output_tensors->size(), RET_ERROR,
                    "acl output size should be equal to ms output tensor size");
  for (size_t i = 0; i < output_size; i++) {
    AclDataInfo acl_data_info(AclDataInfo::Output);
    ret = GetAclDataInfo(&acl_data_info, acl_model_desc_, i);
    MS_CHECK_TRUE_MSG(ret == RET_OK, RET_ERROR, "get acl data info failed.");
    void *data = nullptr;
    if (acl_data_info.data_size * actual_batch_size_ != output_tensors->at(i).DataSize()) {
      outputs_mem_managed_by_tensor[i] = false;
      MS_LOG(INFO) << "The size of the last dimension of the output tensor "
                   << "does not align with 'internal_stride' value, will memcpy";
      ret = AclMalloc(&data, acl_data_info.data_size * actual_batch_size_);
      MS_CHECK_TRUE_MSG(ret == RET_OK, RET_ERROR, "svp acl rt malloc acl output buffer failed.");
    } else {
      outputs_mem_managed_by_tensor[i] = true;
      MS_LOG(INFO) << "The size of the last dimension of the output tensor "
                   << "aligns with 'internal_stride' value, will not memcpy";
      output_tensors->at(i).SetAllocator(custom_allocator_);
      data = output_tensors->at(i).MutableData();  // svp malloc memory for ms tensor
    }
    MS_CHECK_TRUE_MSG(data != nullptr, RET_ERROR, "get data ptr failed.");
    ret = AddDatasetBuffer(acl_outputs_, acl_data_info.data_size * actual_batch_size_, acl_data_info.stride, data);
    MS_CHECK_TRUE_MSG(ret == RET_OK, RET_ERROR, "add dataset buffer failed.");
  }
  return RET_OK;
}

int AclModelManager::UpdateKernelConfig(const std::map<std::string, std::string> &dpico_config) {
  int ret = custom_config_manager_ptr_->UpdateConfig(dpico_config);
  MS_CHECK_TRUE_MSG(ret == RET_OK, RET_ERROR, "update custom config success");
  return RET_OK;
}

int AclModelManager::UpdateAclInputs(std::vector<mindspore::MSTensor> *input_tensors) {
  MS_CHECK_TRUE_MSG(input_tensors != nullptr, RET_ERROR, "input_tensors is nullptr.");
  for (size_t i = 0; i < input_tensors->size() - kNumOfInputOm; i++) {
    MS_CHECK_TRUE_MSG(inputs_mem_managed_by_tensor.find(i) != inputs_mem_managed_by_tensor.end(), RET_ERROR,
                      "invalid input index: " << i);
    auto input_tensor = input_tensors->at(i);
    auto data_buffer = svp_acl_mdl_get_dataset_buffer(acl_inputs_, i);
    MS_CHECK_TRUE_MSG(data_buffer != nullptr, RET_ERROR, "data_buffer is nullptr.");
    auto stride = svp_acl_mdl_get_input_default_stride(acl_model_desc_, i);
    if (!inputs_mem_managed_by_tensor[i]) {
      MS_LOG(INFO) << "input data isn't managed by tensor." << input_tensor.Name();
      void *tmp_buffer = svp_acl_get_data_buffer_addr(data_buffer);
      auto ret = AclFree(&tmp_buffer);
      MS_CHECK_TRUE_MSG(ret == SVP_ACL_SUCCESS, RET_ERROR, "execute AclFree failed");
      AclDataInfo acl_data_info(AclDataInfo::Input);
      ret = GetAclDataInfo(&acl_data_info, acl_model_desc_, i);
      MS_CHECK_TRUE_MSG(ret == RET_OK, RET_ERROR, "get acl data info failed.");
      void *data = nullptr;
      ret = AclMalloc(&data, acl_data_info.data_size * actual_batch_size_);
      MS_CHECK_TRUE_MSG(ret == RET_OK, RET_ERROR, "svp acl rt malloc acl input buffer failed.");
      auto svp_ret =
        svp_acl_update_data_buffer(data_buffer, data, acl_data_info.data_size * actual_batch_size_, stride);
      MS_CHECK_TRUE_MSG(svp_ret == SVP_ACL_SUCCESS, RET_ERROR,
                        "svp update data buffer failed. " << input_tensor.Name());
    } else {
      auto ret = svp_acl_update_data_buffer(data_buffer, input_tensors->at(i).MutableData(),
                                            input_tensors->at(i).DataSize(), stride);
      MS_CHECK_TRUE_MSG(ret == SVP_ACL_SUCCESS, RET_ERROR, "svp update data buffer failed. " << input_tensor.Name());
    }
  }

  if (acl_model_type_ == AclModelType::kRoi) {
    size_t detect_param_index = input_tensors->size() - kNumOfInputOm;
    auto data_buffer = svp_acl_mdl_get_dataset_buffer(acl_inputs_, detect_param_index);
    MS_CHECK_TRUE_MSG(data_buffer != nullptr, RET_ERROR, "data_buffer is nullptr.");
    void *data = svp_acl_get_data_buffer_addr(data_buffer);
    MS_CHECK_TRUE_MSG(data != nullptr, RET_ERROR, "detect param data is nullptr.");
    auto ret = AclFree(&data);
    MS_CHECK_TRUE_MSG(ret == SVP_ACL_SUCCESS, RET_ERROR, "execute AclFree failed");
    void *data_tmp = nullptr;
    size_t detect_param_stride = sizeof(float) * kDetectParamNum;
    ret = AclMalloc(&data_tmp, actual_batch_size_ * detect_param_stride);
    MS_CHECK_TRUE_MSG(ret == RET_OK, RET_ERROR, "svp acl rt malloc detect input buffer failed.");
    for (size_t loop = 0; loop < actual_batch_size_; loop++) {
      ret = SetDetectParams(reinterpret_cast<uint8_t *>(data_tmp) + loop * detect_param_stride);
      MS_CHECK_TRUE_MSG(ret == RET_OK, RET_ERROR, "set detect params failed.");
    }
    auto svp_ret =
      svp_acl_update_data_buffer(data_buffer, data_tmp, detect_param_stride * actual_batch_size_, detect_param_stride);
    MS_CHECK_TRUE_MSG(svp_ret == SVP_ACL_SUCCESS, RET_ERROR, "svp update data buffer for detect_param failed. ");
  }
  return RET_OK;
}

int AclModelManager::UpdateAclOutputs(std::vector<mindspore::MSTensor> *output_tensors) {
  MS_CHECK_TRUE_MSG(output_tensors != nullptr, RET_ERROR, "output_tensors is nullptr.");
  for (size_t i = 0; i < output_tensors->size(); i++) {
    MS_CHECK_TRUE_MSG(outputs_mem_managed_by_tensor.find(i) != outputs_mem_managed_by_tensor.end(), RET_ERROR,
                      "invalid output index: " << i);
    auto output_tensor = output_tensors->at(i);
    auto data_buffer = svp_acl_mdl_get_dataset_buffer(acl_outputs_, i);
    MS_CHECK_TRUE_MSG(data_buffer != nullptr, RET_ERROR, "data_buffer is nullptr.");
    auto stride = svp_acl_mdl_get_output_default_stride(acl_model_desc_, i);
    if (!outputs_mem_managed_by_tensor[i]) {
      MS_LOG(INFO) << "output data isn't managed by tensor." << output_tensor.Name();
      void *tmp_buffer = svp_acl_get_data_buffer_addr(data_buffer);
      auto ret = AclFree(&tmp_buffer);
      MS_CHECK_TRUE_MSG(ret == SVP_ACL_SUCCESS, RET_ERROR, "execute AclFree failed");
      AclDataInfo acl_data_info(AclDataInfo::Output);
      ret = GetAclDataInfo(&acl_data_info, acl_model_desc_, i);
      MS_CHECK_TRUE_MSG(ret == RET_OK, RET_ERROR, "get acl data info failed.");
      void *data = nullptr;
      ret = AclMalloc(&data, acl_data_info.data_size * actual_batch_size_);
      MS_CHECK_TRUE_MSG(ret == RET_OK, RET_ERROR, "svp acl rt malloc acl output buffer failed.");
      auto svp_ret =
        svp_acl_update_data_buffer(data_buffer, data, acl_data_info.data_size * actual_batch_size_, stride);
      MS_CHECK_TRUE_MSG(svp_ret == SVP_ACL_SUCCESS, RET_ERROR,
                        "svp update data buffer failed. " << output_tensor.Name());
    } else {
      auto ret = svp_acl_update_data_buffer(data_buffer, output_tensor.MutableData(), output_tensor.DataSize(), stride);
      MS_CHECK_TRUE_MSG(ret == SVP_ACL_SUCCESS, RET_ERROR, "svp update data buffer failed. " << output_tensor.Name());
    }
  }
  return RET_OK;
}

int AclModelManager::Execute(const std::vector<mindspore::MSTensor> &input_tensors,
                             const std::vector<mindspore::MSTensor> &output_tensors,
                             const std::map<std::string, std::string> &model_share_config) {
  int ret;
  if (custom_config_manager_ptr_->IsEnableMultiModelSharingMem(model_share_config)) {
    ret = CreateTaskBufAndWorkBuf();
  } else {
    ret = CreateNoShareTaskBufAndWorkBuf();
  }
  MS_CHECK_TRUE_MSG(ret == RET_OK, RET_ERROR, "create task buf and work buf failed.");

  ret = CopyTensorDataToAclInputs(input_tensors);
  MS_CHECK_TRUE_MSG(ret == RET_OK, RET_ERROR, "copy input tensor data to acl inputs failed.");

  ret = FlushAclInputsAndOutputs();
  MS_CHECK_TRUE_MSG(ret == RET_OK, RET_ERROR, "flush input and output tensor data failed.");

  ret = AclModelRun(input_tensors);
  MS_CHECK_TRUE_MSG(ret == RET_OK, RET_ERROR, "acl model run failed.");

  ret = CopyAclOutputsToTensorData(output_tensors);
  MS_CHECK_TRUE_MSG(ret == RET_OK, RET_ERROR, "copy acl outputs to output tensor data failed.");
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
