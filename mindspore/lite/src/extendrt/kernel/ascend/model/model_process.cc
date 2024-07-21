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

#include "extendrt/kernel/ascend/model/model_process.h"
#include <sys/time.h>
#include <utility>
#include <algorithm>
#include <map>
#include <thread>
#include "common/log_adapter.h"
#include "src/common/utils.h"
#include "src/common/log_util.h"
#include "src/litert/kernel/ascend/src/acl_mem_manager.h"
#include "src/extendrt/kernel/ascend/model/acl_allocator.h"
#include "transform/symbol/acl_base_symbol.h"
#include "transform/symbol/acl_mdl_symbol.h"
#include "transform/symbol/acl_rt_symbol.h"
#include "transform/symbol/acl_symbol.h"
#include "transform/symbol/symbol_utils.h"

namespace mindspore::kernel {
namespace acl {
namespace {
constexpr size_t kBatchSizeNum = 1;
constexpr size_t kImageSizeHwNum = 2;
constexpr char kINFOLogLevel = '1';
constexpr char kDEBUGLogLevel = '0';
}  // namespace
static TypeId TransToDataType(aclDataType data_type) {
  static const std::map<aclDataType, enum TypeId> data_type_map = {
    {ACL_FLOAT16, TypeId::kNumberTypeFloat16}, {ACL_FLOAT, TypeId::kNumberTypeFloat32},
    {ACL_DOUBLE, TypeId::kNumberTypeFloat64},  {ACL_INT8, TypeId::kNumberTypeInt8},
    {ACL_INT16, TypeId::kNumberTypeInt16},     {ACL_INT32, TypeId::kNumberTypeInt32},
    {ACL_INT64, TypeId::kNumberTypeInt64},     {ACL_UINT8, TypeId::kNumberTypeUInt8},
    {ACL_UINT16, TypeId::kNumberTypeUInt16},   {ACL_UINT32, TypeId::kNumberTypeUInt32},
    {ACL_UINT64, TypeId::kNumberTypeUInt64},   {ACL_BOOL, TypeId::kNumberTypeBool},
  };
  auto it = data_type_map.find(data_type);
  if (it == data_type_map.end()) {
    return TypeId::kNumberTypeEnd;
  } else {
    return it->second;
  }
}

static std::string ShapeToString(const std::vector<int64_t> &shape) {
  std::string result = "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    result += std::to_string(shape[i]);
    if (i + 1 < shape.size()) {
      result += ", ";
    }
  }
  result += "]";
  return result;
}

ModelProcess::~ModelProcess() {
  if (dynamic_dims_ != nullptr) {
    delete[] dynamic_dims_;
    dynamic_dims_ = nullptr;
  }
}

aclError ModelProcess::AclrtMemcpy(void *dst, size_t destMax, const void *src, size_t count, aclrtMemcpyKind kind) {
  struct timeval start_time;
  auto env = std::getenv("GLOG_v");
  if (env != nullptr && (env[0] == kDEBUGLogLevel)) {
    (void)gettimeofday(&start_time, nullptr);
  }
  auto ret = CALL_ASCEND_API(aclrtMemcpy, dst, destMax, src, count, kind);
  if (env != nullptr && (env[0] == kDEBUGLogLevel)) {
    struct timeval end_time;
    (void)gettimeofday(&end_time, nullptr);
    constexpr uint64_t kUSecondInSecond = 1000000;
    uint64_t cost =
      (kUSecondInSecond * static_cast<uint64_t>(end_time.tv_sec) + static_cast<uint64_t>(end_time.tv_usec)) -
      (kUSecondInSecond * static_cast<uint64_t>(start_time.tv_sec) + static_cast<uint64_t>(start_time.tv_usec));
    if (kind == ACL_MEMCPY_DEVICE_TO_HOST) {
      MS_LOG(DEBUG) << "Device to Host copy in " << cost << " us";
    } else if (kind == ACL_MEMCPY_HOST_TO_DEVICE) {
      MS_LOG(DEBUG) << "Host to Device copy in " << cost << " us";
    } else if (kind == ACL_MEMCPY_DEVICE_TO_DEVICE) {
      MS_LOG(DEBUG) << "Device to Device copy in " << cost << " us";
    }
  }
  return ret;
}

bool ModelProcess::PreInitModelResource() {
  model_desc_ = CALL_ASCEND_API(aclmdlCreateDesc);
  aclError acl_ret = CALL_ASCEND_API(aclmdlGetDesc, model_desc_, model_id_);
  if (acl_ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "Read model desc failed, ret = " << acl_ret;
    return false;
  }
  dynamic_shape_options_.batch_size = GetDynamicBatch();
  dynamic_shape_options_.image_size = GetDynamicImage();
  dynamic_shape_options_.dynamic_dims = GetDynamicDims();
  if (!CheckAndSetDynFlag()) {
    MS_LOG(ERROR) << "Check and set dynamic flag failed";
    return false;
  }
  if (!InitInputsBuffer()) {
    MS_LOG(ERROR) << "Create input buffer failed.";
    return false;
  }
  if (!InitOutputsBuffer()) {
    MS_LOG(ERROR) << "Create output buffer failed.";
    return false;
  }
  if (is_dynamic_input_) {
    data_input_num_ = input_infos_.size();
    return true;
  }

  data_input_num_ = input_infos_.size();
  if (IsDynamicShape() && data_input_num_ > 0) {
    data_input_num_ -= 1;
  }

  dynamic_shape_options_.input_format = GetInputFormat();
  dynamic_shape_options_.input_shapes = GetInputShape();

  if (!dyn_shape_proc_.Init(dynamic_shape_options_)) {
    MS_LOG(ERROR) << "Init DynShapeProcess failed.";
    return false;
  }
  return true;
}

std::set<uint64_t> ModelProcess::GetDynamicBatch() {
  if (model_desc_ == nullptr) {
    MS_LOG(ERROR) << " Model desc is nullptr.";
    return std::set<uint64_t>();
  }
  aclmdlBatch dynamic_batch;
  if (aclmdlGetDynamicBatch(model_desc_, &dynamic_batch) != ACL_SUCCESS) {
    MS_LOG(ERROR) << "Failed to get dynamic batch.";
    return std::set<uint64_t>();
  }
  size_t batch_count = dynamic_batch.batchCount;
  if (batch_count > ACL_MAX_BATCH_NUM) {
    MS_LOG(ERROR) << "Real batch count " << batch_count << " is larger than max " << ACL_MAX_BATCH_NUM;
    return std::set<uint64_t>();
  }
  std::set<uint64_t> batch;
  for (size_t i = 0; i < dynamic_batch.batchCount; ++i) {
    batch.insert(dynamic_batch.batch[i]);
  }
  return batch;
}

std::pair<aclmdlIODims *, size_t> ModelProcess::GetDynamicDims() {
  if (model_desc_ == nullptr) {
    MS_LOG(ERROR) << " Model desc is nullptr.";
    return std::make_pair(nullptr, 0);
  }
  size_t gear_conut = 0;
  auto ret = aclmdlGetInputDynamicGearCount(model_desc_, -1, &gear_conut);
  if (ret != ACL_SUCCESS) {
    MS_LOG(ERROR) << "aclmdlGetInputDynamicGearCount failed.";
    return std::make_pair(nullptr, 0);
  }
  MS_LOG(INFO) << "gear_conut is: " << gear_conut;
  if (gear_conut == 0) {
    MS_LOG(INFO) << "gear_conut is zero";
    return std::make_pair(nullptr, 0);
  }
  dynamic_dims_ = new aclmdlIODims[gear_conut];
  if (dynamic_dims_ == nullptr) {
    MS_LOG(ERROR) << "new aclmldIODims failed.";
    return std::make_pair(nullptr, 0);
  }
  if (aclmdlGetInputDynamicDims(model_desc_, -1, dynamic_dims_, gear_conut) != ACL_SUCCESS) {
    MS_LOG(ERROR) << "aclmdlGetInputDynamicDims failed.";
    delete[] dynamic_dims_;
    dynamic_dims_ = nullptr;
    return std::make_pair(nullptr, 0);
  }
  return std::make_pair(dynamic_dims_, gear_conut);
}

std::set<std::pair<uint64_t, uint64_t>> ModelProcess::GetDynamicImage() {
  if (model_desc_ == nullptr) {
    MS_LOG(ERROR) << " Model desc is nullptr.";
    return std::set<std::pair<uint64_t, uint64_t>>();
  }
  aclmdlHW dynamic_hw;
  if (aclmdlGetDynamicHW(model_desc_, 0, &dynamic_hw) != ACL_SUCCESS) {
    MS_LOG(ERROR) << "Failed to get dynamic hw.";
    return std::set<std::pair<uint64_t, uint64_t>>();
  }
  size_t hw_count = dynamic_hw.hwCount;
  if (hw_count > ACL_MAX_HW_NUM) {
    MS_LOG(ERROR) << "Real hw count " << hw_count << " is larger than max " << ACL_MAX_HW_NUM;
    return std::set<std::pair<uint64_t, uint64_t>>();
  }
  std::set<std::pair<uint64_t, uint64_t>> image;
  for (size_t i = 0; i < dynamic_hw.hwCount; ++i) {
    image.insert(std::pair<uint64_t, uint64_t>(dynamic_hw.hw[i][0], dynamic_hw.hw[i][1]));
  }
  return image;
}

std::vector<Format> ModelProcess::GetInputFormat() {
  if (model_desc_ == nullptr) {
    MS_LOG(ERROR) << " Model desc is nullptr.";
    return std::vector<Format>();
  }
  std::vector<Format> input_formats;
  static const std::map<aclFormat, enum Format> acl_format_map = {
    {ACL_FORMAT_NCHW, NCHW}, {ACL_FORMAT_NHWC, NHWC}, {ACL_FORMAT_ND, NCHW}};
  for (size_t i = 0; i < data_input_num_; ++i) {
    aclFormat format = aclmdlGetInputFormat(model_desc_, i);
    auto iter = acl_format_map.find(format);
    if (iter != acl_format_map.end()) {
      input_formats.emplace_back(iter->second);
    } else {
      MS_LOG(INFO) << "aclFormat " << format << " not found in map, please double check and add...using default format";
      input_formats.emplace_back(DEFAULT_FORMAT);
    }
    MS_LOG(DEBUG) << "Format of Input " << i << " is " << static_cast<int32_t>(format);
  }
  return input_formats;
}

std::vector<Format> ModelProcess::GetOutputFormat() {
  if (model_desc_ == nullptr) {
    MS_LOG(ERROR) << " Model desc is nullptr.";
    return std::vector<Format>();
  }
  std::vector<Format> output_formats;
  static const std::map<aclFormat, enum Format> acl_format_map = {
    {ACL_FORMAT_NCHW, NCHW}, {ACL_FORMAT_NHWC, NHWC}, {ACL_FORMAT_ND, NCHW}};
  for (size_t i = 0; i < output_infos_.size(); ++i) {
    aclFormat format = aclmdlGetOutputFormat(model_desc_, i);
    auto iter = acl_format_map.find(format);
    if (iter != acl_format_map.end()) {
      output_formats.emplace_back(iter->second);
    } else {
      MS_LOG(INFO) << "aclFormat " << format << " not found in map, please double check and add...using default format";
      output_formats.emplace_back(DEFAULT_FORMAT);
    }
    MS_LOG(DEBUG) << "Format of Output " << i << " is " << static_cast<int32_t>(format);
  }
  return output_formats;
}

const std::vector<TypeId> ModelProcess::GetOutputDataType() {
  std::vector<TypeId> data_types;
  for (size_t i = 0; i < output_infos_.size(); ++i) {
    TypeId data_type = TransToDataType(output_infos_[i].data_type);
    data_types.emplace_back(data_type);
  }
  return data_types;
}

const std::vector<ShapeVector> ModelProcess::GetOutputShape() {
  std::vector<ShapeVector> shapes;
  for (size_t i = 0; i < output_infos_.size(); ++i) {
    shapes.emplace_back(output_infos_[i].dims);
  }
  return shapes;
}

const std::vector<ShapeVector> ModelProcess::GetInputShape() {
  std::vector<ShapeVector> shapes;
  for (size_t i = 0; i < data_input_num_; ++i) {
    shapes.push_back(input_infos_[i].dims);
  }
  return shapes;
}

const std::vector<TypeId> ModelProcess::GetInputDataType() {
  std::vector<TypeId> data_types;
  for (size_t i = 0; i < data_input_num_; ++i) {
    TypeId data_type = TransToDataType(input_infos_[i].data_type);
    data_types.emplace_back(data_type);
  }
  return data_types;
}

bool ModelProcess::CheckAndSetDynFlag() {
  aclError ret;
  size_t input_size = CALL_ASCEND_API(aclmdlGetNumInputs, model_desc_);
  for (size_t i = 0; i < input_size; ++i) {
    auto buffer_size = CALL_ASCEND_API(aclmdlGetInputSizeByIndex, model_desc_, i);
    aclmdlIODims input_dims;
    ret = aclmdlGetInputDimsV2(model_desc_, i, &input_dims);
    if (ret != ACL_ERROR_NONE) {
      MS_LOG(ERROR) << "Get input dims failed";
      return false;
    }
    for (size_t j = 0; j < input_dims.dimCount; ++j) {
      if (input_dims.dims[j] < 0) {
        if (buffer_size == 0) {
          is_dynamic_input_ = true;
          MS_LOG(INFO) << "The input of model is dynamic.";
          break;
        } else {
          if (!IsDynamicShape()) {
            is_dynamic_shape_range_ = true;
            MS_LOG(INFO) << "The input of model is dynamic shape range";
          }
        }
      }
    }
    if (is_dynamic_input_ || is_dynamic_shape_range_) {
      break;
    }
  }
  size_t output_size = CALL_ASCEND_API(aclmdlGetNumOutputs, model_desc_);
  for (size_t i = 0; i < output_size; ++i) {
    aclmdlIODims output_dims;
    ret = CALL_ASCEND_API(aclmdlGetOutputDims, model_desc_, i, &output_dims);
    if (ret != ACL_ERROR_NONE) {
      MS_LOG(ERROR) << "Get output dims failed";
      return false;
    }
    for (size_t j = 0; j < output_dims.dimCount; ++j) {
      if (output_dims.dims[j] < 0) {
        is_dynamic_output_ = true;
        MS_LOG(INFO) << "The output of model is dynamic.";
        return true;
      }
    }
  }
  return true;
}

bool ModelProcess::InitInputsBuffer() {
  aclError ret;
  inputs_ = CALL_ASCEND_API(aclmdlCreateDataset);
  if (inputs_ == nullptr) {
    MS_LOG(ERROR) << "Create input dataset failed";
    return false;
  }
  size_t input_size = CALL_ASCEND_API(aclmdlGetNumInputs, model_desc_);
  MS_LOG(INFO) << "input_size = " << input_size;
  for (size_t i = 0; i < input_size; ++i) {
    aclmdlIODims dims;
    // To get correct dims with static AIPP configured, same result as aclmdlGetInputDims without static AIPP
    if (is_dynamic_output_) {  // There is a bug for aclmdlGetInputDimsV2 when output is dynamic shape.
      ret = CALL_ASCEND_API(aclmdlGetInputDims, model_desc_, i, &dims);
    } else {
      ret = aclmdlGetInputDimsV2(model_desc_, i, &dims);
    }
    if (ret != ACL_ERROR_NONE) {
      MS_LOG(ERROR) << "Get input shape failed, ret = " << ret;
      return false;
    }
    auto buffer_size = CALL_ASCEND_API(aclmdlGetInputSizeByIndex, model_desc_, i);
    void *data_mem_buffer = nullptr;
    if (!is_dynamic_input_ && !CreateDataBuffer(&data_mem_buffer, buffer_size, inputs_)) {
      MS_LOG(ERROR) << "Add input data buffer failed, buffer size " << buffer_size;
      return false;
    }
    aclDataType data_type = CALL_ASCEND_API(aclmdlGetInputDataType, model_desc_, i);
    std::vector<int64_t> shape(dims.dims, dims.dims + dims.dimCount);
    std::string input_name = CALL_ASCEND_API(aclmdlGetInputNameByIndex, model_desc_, i);
    if (!is_dynamic_input_) {
      aclFormat input_format = aclmdlGetInputFormat(model_desc_, i);
      aclTensorDesc *desc = CALL_ASCEND_API(aclCreateTensorDesc, data_type, dims.dimCount, dims.dims, input_format);
      ret = aclmdlSetDatasetTensorDesc(inputs_, desc, i);
      if (ret != ACL_ERROR_NONE) {
        MS_LOG(ERROR) << "aclmdlSetDatasetTensorDesc failed, ret = " << ret;
        return false;
      }
    }
    if (input_name.empty()) {
      MS_LOG(WARNING) << "Get name of input " << i << " failed.";
    }
    MS_LOG(INFO) << "Name of input " << i << " is " << input_name;
    input_infos_.emplace_back(
      AclTensorInfo{data_mem_buffer, data_mem_buffer, buffer_size, buffer_size, data_type, shape, input_name});
  }
  MS_LOG(INFO) << "Create model inputs success";
  return true;
}

bool ModelProcess::InitOutputsBuffer() {
  aclError ret;
  outputs_ = CALL_ASCEND_API(aclmdlCreateDataset);
  if (outputs_ == nullptr) {
    MS_LOG(ERROR) << "Create output dataset failed";
    return false;
  }
  size_t output_size = CALL_ASCEND_API(aclmdlGetNumOutputs, model_desc_);
  MS_LOG(INFO) << "Output_size = " << output_size;
  for (size_t i = 0; i < output_size; ++i) {
    aclmdlIODims dims;
    ret = CALL_ASCEND_API(aclmdlGetOutputDims, model_desc_, i, &dims);
    if (ret != ACL_ERROR_NONE) {
      MS_LOG(ERROR) << "Get output shape failed";
      return true;
    }
    bool is_dynamic_output = false;
    for (size_t dim_idx = 0; dim_idx < dims.dimCount; dim_idx++) {
      is_dynamic_output = (dims.dims[dim_idx] < 0) ? true : false;
    }
    size_t buffer_size = 0;
    if (!is_dynamic_output) {
      buffer_size = CALL_ASCEND_API(aclmdlGetOutputSizeByIndex, model_desc_, i);
    }
    void *data_mem_buffer = nullptr;
    if (!CreateDataBuffer(&data_mem_buffer, buffer_size, outputs_)) {
      MS_LOG(ERROR) << "Add output data buffer failed, buffer size " << buffer_size;
      return false;
    }
    aclFormat format = aclmdlGetOutputFormat(model_desc_, i);
    MS_LOG(DEBUG) << "The output format of om is " << format;
    aclDataType data_type = CALL_ASCEND_API(aclmdlGetOutputDataType, model_desc_, i);
    std::vector<int64_t> shape(dims.dims, dims.dims + dims.dimCount);
    if (is_dynamic_output) {
      shape = std::vector<int64_t>({-1});
    }
    std::string output_name = CALL_ASCEND_API(aclmdlGetOutputNameByIndex, model_desc_, i);
    if (output_name.empty()) {
      MS_LOG(WARNING) << "Get name of output " << i << " failed.";
    }
    MS_LOG(INFO) << "Name of om output " << i << " is " << output_name << "Buffer size " << buffer_size;
    output_infos_.emplace_back(
      AclTensorInfo{data_mem_buffer, data_mem_buffer, buffer_size, buffer_size, data_type, shape, output_name});
  }
  MS_LOG(INFO) << "Create model output success.";
  return true;
}

bool ModelProcess::CreateDataBuffer(void **data_mem_buffer, size_t buffer_size, aclmdlDataset *dataset) {
  aclError ret;
  auto free_data_buffer = [this](void *dataMemBuffer) {
    if (!is_run_on_device_) {
      (void)CALL_ASCEND_API(aclrtFree, dataMemBuffer);
    } else {
      (void)CALL_ASCEND_API(aclrtFreeHost, dataMemBuffer);
    }
  };
  // The model with dynamic input do not need to malloc the memory of output
  if (buffer_size != 0) {
    if (data_mem_buffer == nullptr) {
      MS_LOG(ERROR) << "Data mem buffer is nullptr.";
      return false;
    }
    if (!is_run_on_device_) {
      ret = CALL_ASCEND_API(aclrtMalloc, data_mem_buffer, buffer_size, ACL_MEM_MALLOC_HUGE_FIRST);
      if (ret != ACL_ERROR_NONE) {
        MS_LOG(ERROR) << "Malloc device buffer failed , buffer size " << buffer_size;
        return false;
      }
    } else {
      ret = CALL_ASCEND_API(aclrtMallocHost, data_mem_buffer, buffer_size);
      if (ret != ACL_ERROR_NONE) {
        MS_LOG(ERROR) << "Malloc host buffer failed , buffer size " << buffer_size;
        return false;
      }
    }
  }
  auto data_buffer = CALL_ASCEND_API(aclCreateDataBuffer, *data_mem_buffer, buffer_size);
  if (data_buffer == nullptr) {
    MS_LOG(ERROR) << "Create Data Buffer failed";
    if (data_mem_buffer != nullptr) {
      free_data_buffer(*data_mem_buffer);
    }
    CALL_ASCEND_API(aclDestroyDataBuffer, data_buffer);
    return false;
  }
  ret = CALL_ASCEND_API(aclmdlAddDatasetBuffer, dataset, data_buffer);
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "add data buffer failed";
    if (data_mem_buffer != nullptr) {
      free_data_buffer(*data_mem_buffer);
    }
    CALL_ASCEND_API(aclDestroyDataBuffer, data_buffer);
    return false;
  }
  return true;
}

void ModelProcess::DestroyInputsBuffer() {
  for (const auto &item : input_infos_) {
    if (item.device_data != nullptr) {
      if (!is_run_on_device_) {
        CALL_ASCEND_API(aclrtFree, item.device_data);
      } else {
        CALL_ASCEND_API(aclrtFreeHost, item.device_data);
      }
    }
    if (item.dynamic_acl_tensor_desc != nullptr) {
      CALL_ASCEND_API(aclDestroyTensorDesc, item.dynamic_acl_tensor_desc);
    }
  }
  input_infos_.clear();

  if (inputs_ == nullptr) {
    return;
  }
  for (size_t i = 0; i < CALL_ASCEND_API(aclmdlGetDatasetNumBuffers, inputs_); i++) {
    auto dataBuffer = CALL_ASCEND_API(aclmdlGetDatasetBuffer, inputs_, i);
    CALL_ASCEND_API(aclDestroyDataBuffer, dataBuffer);
  }
  CALL_ASCEND_API(aclmdlDestroyDataset, inputs_);
  inputs_ = nullptr;
}

void ModelProcess::DestroyOutputsBuffer() {
  if (!is_dynamic_output_) {
    for (const auto &item : output_infos_) {
      if (item.device_data != nullptr) {
        if (!is_run_on_device_) {
          CALL_ASCEND_API(aclrtFree, item.device_data);
        } else {
          CALL_ASCEND_API(aclrtFreeHost, item.device_data);
        }
      }
    }
  }
  output_infos_.clear();

  if (outputs_ == nullptr) {
    return;
  }
  for (size_t i = 0; i < CALL_ASCEND_API(aclmdlGetDatasetNumBuffers, outputs_); i++) {
    auto dataBuffer = CALL_ASCEND_API(aclmdlGetDatasetBuffer, outputs_, i);
    CALL_ASCEND_API(aclDestroyDataBuffer, dataBuffer);
  }
  CALL_ASCEND_API(aclmdlDestroyDataset, outputs_);
  outputs_ = nullptr;
}

bool ModelProcess::PrepareMutiModelShare(const void *om_data, size_t om_data_size) {
  size_t work_size = 0;
  size_t weight_size = 0;
  auto acl_ret = CALL_ASCEND_API(aclmdlQuerySizeFromMem, om_data, om_data_size, &work_size, &weight_size);
  if (acl_ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "Call aclmdlQuerySizeFromMem failed, ret = " << acl_ret;
    return false;
  }
  MS_LOG(INFO) << "work_size: " << work_size << " weight_size: " << weight_size;
  std::thread::id thread_id = std::this_thread::get_id();
  auto ret = AclMemManager::GetInstance().UpdateWorkspace(work_size, device_id_, thread_id);
  if (ret != lite::RET_OK) {
    MS_LOG(ERROR) << "update workspace failed, ret = " << ret;
    return false;
  }
  auto model_path = options_->model_path;
  ret = AclMemManager::GetInstance().UpdateWeightspace(model_path, weight_size, device_id_);
  if (ret != lite::RET_OK) {
    MS_LOG(ERROR) << "update weightspace failed, ret = " << ret;
    return false;
  }
  return true;
}

bool ModelProcess::Load(const void *om_data, size_t om_data_size) {
  if (loaded_) {
    MS_LOG(INFO) << "Model has been loaded";
    return true;
  }
  MS_LOG(INFO) << "Start load model model.";
  // model load model
  MS_LOG(INFO) << "multi_model_sharing_mem_prepare: " << options_->multi_model_sharing_mem_prepare;
  MS_LOG(INFO) << "multi_model_sharing_mem: " << options_->multi_model_sharing_mem;
  if (options_->multi_model_sharing_mem_prepare) {
    auto ret = PrepareMutiModelShare(om_data, om_data_size);
    return ret;
  } else if (options_->multi_model_sharing_mem) {
    MS_LOG(INFO) << "using sharing mem by model group.";
    std::thread::id thread_id = std::this_thread::get_id();
    size_t work_size = 0;
    size_t weight_size = 0;
    auto acl_ret = CALL_ASCEND_API(aclmdlQuerySizeFromMem, om_data, om_data_size, &work_size, &weight_size);
    if (acl_ret != ACL_ERROR_NONE) {
      MS_LOG(ERROR) << "Call aclmdlQuerySizeFromMem failed, ret = " << acl_ret;
      return false;
    }
    AclModelMemInfo acl_work_mem_info;
    AclModelMemInfo acl_weight_mem_info;
    if (options_->share_workspace) {
      auto ret = AclMemManager::GetInstance().GetModelWorkMem(&acl_work_mem_info, device_id_, thread_id);
      if (ret != lite::RET_OK) {
        MS_LOG(ERROR) << "Get work mem failed!";
        return ret;
      }
      acl_weight_mem_info.mem_size = weight_size;
      acl_ret = CALL_ASCEND_API(aclrtMalloc, &(acl_weight_mem_info.mem_addr), acl_weight_mem_info.mem_size,
                                ACL_MEM_MALLOC_HUGE_FIRST);
      if (acl_ret != ACL_ERROR_NONE) {
        MS_LOG(ERROR) << "Call aclrtMalloc failed, err_code = " << acl_ret;
        return lite::RET_ERROR;
      }
    } else if (options_->share_weightspace) {
      auto model_path = options_->model_path;
      auto ret = AclMemManager::GetInstance().GetModelWeightMem(&acl_weight_mem_info, model_path, device_id_);
      if (ret != lite::RET_OK) {
        MS_LOG(ERROR) << "Get weight mem failed!";
        return ret;
      }
      acl_work_mem_info.mem_size = work_size;
      acl_ret = CALL_ASCEND_API(aclrtMalloc, &(acl_work_mem_info.mem_addr), acl_work_mem_info.mem_size,
                                ACL_MEM_MALLOC_HUGE_FIRST);
      if (acl_ret != ACL_ERROR_NONE) {
        MS_LOG(ERROR) << "Call aclrtMalloc failed, err_code = " << acl_ret;
        return lite::RET_ERROR;
      }
    } else if (options_->share_weightspace_workspace) {
      auto model_path = options_->model_path;
      auto ret = AclMemManager::GetInstance().GetModelWeightMem(&acl_weight_mem_info, model_path, device_id_);
      if (ret != lite::RET_OK) {
        MS_LOG(ERROR) << "Get weight mem failed!";
        return ret;
      }
      ret = AclMemManager::GetInstance().GetModelWorkMem(&acl_work_mem_info, device_id_, thread_id);
      if (ret != lite::RET_OK) {
        MS_LOG(ERROR) << "Get work mem failed!";
        return ret;
      }
    } else {
      MS_LOG(ERROR) << "Please specify the sharing type!";
      return false;
    }
    acl_ret = aclmdlLoadFromMemWithMem(om_data, om_data_size, &model_id_, acl_work_mem_info.mem_addr, work_size,
                                       acl_weight_mem_info.mem_addr, weight_size);
    if (acl_ret != ACL_ERROR_NONE) {
      MS_LOG(ERROR) << "Call aclmdlLoadFromMemWithMem failed, ret = " << acl_ret;
      return lite::RET_ERROR;
    }
    is_sharing_workspace_ = true;
  } else {
    auto acl_ret = CALL_ASCEND_API(aclmdlLoadFromMem, om_data, om_data_size, &model_id_);
    if (acl_ret != ACL_ERROR_NONE) {
      MS_LOG(ERROR) << "Call aclmdlLoadFromMem failed, ret = " << acl_ret;
      return false;
    }
  }
  // model init model resource
  if (!PreInitModelResource()) {
    (void)CALL_ASCEND_API(aclmdlUnload, model_id_);
    MS_LOG(ERROR) << "Pre init model resource failed.";
    return false;
  }
  loaded_ = true;
  MS_LOG(INFO) << "Load model model success.";
  return true;
}

bool ModelProcess::UnLoad() {
  if (!loaded_) {
    MS_LOG(INFO) << "Model has not been loaded or has been unloaded";
    return true;
  }
  loaded_ = false;
  auto ret = CALL_ASCEND_API(aclmdlUnload, model_id_);
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "Unload model failed, ret = " << ret;
    return false;
  }
  if (model_desc_ != nullptr) {
    ret = CALL_ASCEND_API(aclmdlDestroyDesc, model_desc_);
    if (ret != ACL_ERROR_NONE) {
      MS_LOG(ERROR) << "Unload model failed, ret = " << ret;
      return false;
    }
    model_desc_ = nullptr;
  }
  DestroyInputsBuffer();
  DestroyOutputsBuffer();
  if (weight_ptr_ != nullptr) {
    CALL_ASCEND_API(aclrtFree, weight_ptr_);
    weight_ptr_ = nullptr;
  }
  MS_LOG(INFO) << "End unload model " << model_id_;
  return true;
}

bool ModelProcess::IsDynamicShape() { return IsDynamicBatchSize() || IsDynamicImageSize() || IsDynamicDims(); }

bool ModelProcess::IsDynamicBatchSize() { return !dynamic_shape_options_.batch_size.empty(); }

bool ModelProcess::IsDynamicImageSize() { return !dynamic_shape_options_.image_size.empty(); }

bool ModelProcess::IsDynamicDims() { return dynamic_shape_options_.dynamic_dims.second != 0; }

bool ModelProcess::ResetInputSize(const std::vector<ShapeVector> &new_shapes) {
  for (size_t index = 0; index < new_shapes.size(); index++) {
    std::vector<int64_t> shape = new_shapes[index];
    size_t elem_count = 1;
    for (size_t i = 0; i < shape.size(); i++) {
      if (shape[i] < 0) {
        elem_count = 0;
        break;
      }
      elem_count *= shape[i];
    }
    input_infos_[index].dims = shape;
    auto data_type = CALL_ASCEND_API(aclmdlGetInputDataType, model_desc_, index);
    auto new_buffer_size = elem_count * CALL_ASCEND_API(aclDataTypeSize, data_type);
    if (!is_dynamic_input_) {
      input_infos_[index].buffer_size = new_buffer_size;
    } else if (new_buffer_size > input_infos_[index].buffer_size) {
      is_dynamic_resize_input_ = true;
      input_infos_[index].buffer_size = new_buffer_size;
    }
  }
  return true;
}

bool ModelProcess::ResetOutputSize() {
  if (model_desc_ == nullptr) {
    MS_LOG(ERROR) << " Model desc is nullptr.";
    return false;
  }
  aclDataType data_type;
  aclError ret;
  size_t output_size = CALL_ASCEND_API(aclmdlGetNumOutputs, model_desc_);
  for (size_t index = 0; index < output_size; index++) {
    struct aclmdlIODims dims;
    ret = CALL_ASCEND_API(aclmdlGetCurOutputDims, model_desc_, index, &dims);
    if (ret != ACL_ERROR_NONE) {
      MS_LOG(ERROR) << "get output dim error.";
      return false;
    }
    std::vector<int64_t> shape(dims.dims, dims.dims + dims.dimCount);
    size_t elem_count = 1;
    for (size_t i = 0; i < dims.dimCount; i++) {
      if (dims.dims[i] < 0) {
        elem_count = 0;
        break;
      }
      elem_count *= dims.dims[i];
    }
    data_type = CALL_ASCEND_API(aclmdlGetOutputDataType, model_desc_, index);
    output_infos_[index].dims = shape;
    output_infos_[index].buffer_size = elem_count * CALL_ASCEND_API(aclDataTypeSize, data_type);
  }
  return true;
}

bool ModelProcess::Resize(const std::vector<ShapeVector> &new_shapes) {
  if (!loaded_) {
    MS_LOG(ERROR) << "Model has not been loaded";
    return false;
  }
  auto input_shapes = GetInputShape();
  if (input_shapes.size() != new_shapes.size()) {
    MS_LOG(ERROR) << "Invalid new input size " << new_shapes.size() << ", expect input size " << input_shapes.size();
    return false;
  }
  bool input_shape_changed = false;
  for (size_t i = 0; i < new_shapes.size(); i++) {
    auto new_shape = new_shapes[i];
    if (std::any_of(new_shape.begin(), new_shape.end(), [](auto dim) { return dim < 0; })) {
      MS_LOG(ERROR) << "New shape of input " << i << " cannot be dynamic, new shape: " << new_shape;
      return false;
    }
    if (input_shapes[i] != new_shape) {
      input_shape_changed = true;
    }
  }
  if (!input_shape_changed) {
    return true;
  }
  if (is_dynamic_input_) {
    return ResizeDynamicInputShape(new_shapes);
  }
  if (is_dynamic_shape_range_) {
    return ResizeDynamicInputShapeRange(new_shapes);
  }
  if (!IsDynamicShape()) {
    MS_LOG(ERROR) << "Not support dynamic input";
    return false;
  }
  if (!ResizeDynamicBatchAndImageSize(new_shapes)) {
    MS_LOG(ERROR) << "Resize dynamic batch and image size failed";
    return false;
  }

  return true;
}

bool ModelProcess::ResizeDynamicInputShape(const std::vector<ShapeVector> &new_shapes) {
  MS_LOG(INFO) << "Start to resize dynamic input shape";
  // If it is not the first time to resize input shape, the old addr need to be free
  ResetInputSize(new_shapes);
  FreeResourceInput(input_infos_);
  if (is_dynamic_resize_input_) {
    inputs_ = CALL_ASCEND_API(aclmdlCreateDataset);
    if (inputs_ == nullptr) {
      MS_LOG(ERROR) << "Create input dataset failed";
      return false;
    }
  }
  for (size_t i = 0; i < new_shapes.size(); ++i) {
    if (is_dynamic_resize_input_) {
      void *data_buf = nullptr;
      if (!CreateDataBuffer(&data_buf, input_infos_[i].buffer_size, inputs_)) {
        MS_LOG(ERROR) << "Add input data buffer failed";
        return false;
      }
      auto data_type = CALL_ASCEND_API(aclmdlGetInputDataType, model_desc_, i);
      std::string input_name = CALL_ASCEND_API(aclmdlGetInputNameByIndex, model_desc_, i);
      if (input_name.empty()) {
        MS_LOG(ERROR) << "Get name of input " << i << " failed.";
        return false;
      }
      MS_LOG(INFO) << "Name of input " << i << " is " << input_name;
      input_infos_[i].cur_device_data = data_buf;
      input_infos_[i].device_data = data_buf;
      input_infos_[i].data_type = data_type;
      input_infos_[i].name = input_name;
      auto data_buffer = CALL_ASCEND_API(aclmdlGetDatasetBuffer, inputs_, i);
      input_infos_[i].dynamic_acl_data_buffer = data_buffer;
    }

    aclTensorDesc *input_desc =
      CALL_ASCEND_API(aclCreateTensorDesc, ACL_FLOAT, new_shapes[i].size(), &new_shapes[i][0], ACL_FORMAT_NCHW);
    auto ret = aclmdlSetDatasetTensorDesc(inputs_, input_desc, i);
    input_infos_[i].dynamic_acl_tensor_desc = input_desc;
    if (ret != ACL_ERROR_NONE) {
      MS_LOG(ERROR) << "Acl set dataset tensor desc failed";
      return false;
    }
  }
  is_dynamic_resize_input_ = false;
  MS_LOG(INFO) << "Resize dynamic input shape success";
  return true;
}

bool ModelProcess::ResizeDynamicInputShapeRange(const std::vector<ShapeVector> &new_shapes) {
  MS_LOG(INFO) << "Start to resize dynamic input shape range";
  for (size_t i = 0; i < new_shapes.size(); ++i) {
    std::vector<int64_t> shape = new_shapes[i];
    auto buffer_size = CALL_ASCEND_API(aclmdlGetInputSizeByIndex, model_desc_, i);
    auto data_type = CALL_ASCEND_API(aclmdlGetInputDataType, model_desc_, i);
    size_t elem_count = 1;
    for (size_t j = 0; j < shape.size(); ++j) {
      if (shape[j] < 0) {
        MS_LOG(ERROR) << "The resize shape has the dim less than 0";
        return false;
      }
      elem_count *= shape[j];
    }
    auto new_buffer_size = elem_count * CALL_ASCEND_API(aclDataTypeSize, data_type);
    if (new_buffer_size > buffer_size) {
      MS_LOG(ERROR) << "The resize shape is over shape range";
      return false;
    }
    input_infos_[i].dims = shape;
    aclTensorDesc *input_desc =
      CALL_ASCEND_API(aclCreateTensorDesc, ACL_FLOAT, new_shapes[i].size(), &new_shapes[i][0], ACL_FORMAT_NCHW);
    auto ret = aclmdlSetDatasetTensorDesc(inputs_, input_desc, i);
    if (ret != ACL_ERROR_NONE) {
      MS_LOG(ERROR) << "Acl set dataset tensor desc failed";
      return false;
    }
  }
  MS_LOG(INFO) << "Resize dynamic input shape range success";
  return true;
}
bool ModelProcess::ResizeDynamicBatchAndImageSize(const std::vector<ShapeVector> &new_shapes) {
  if (model_desc_ == nullptr || inputs_ == nullptr) {
    MS_LOG(ERROR) << "Model is not inited";
    return false;
  }
  size_t index;
  auto ret = CALL_ASCEND_API(aclmdlGetInputIndexByName, model_desc_, ACL_DYNAMIC_TENSOR_NAME, &index);
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "Get index of dynamic tensor failed";
    return false;
  }
  if (IsDynamicBatchSize()) {
    int32_t batch_size = 0;
    if (!dyn_shape_proc_.CheckAndGetBatchSize(new_shapes, &batch_size)) {
      MS_LOG(ERROR) << "Failed to check batch size";
      return false;
    }
    MS_LOG(INFO) << "Set Batch size(" << batch_size << ") of input " << index << ".";
    ret = CALL_ASCEND_API(aclmdlSetDynamicBatchSize, model_id_, inputs_, index, batch_size);
    if (ret != ACL_ERROR_NONE) {
      MS_LOG(ERROR) << "Set dynamic batch size failed, model_id is " << model_id_;
      return false;
    }
  } else if (IsDynamicImageSize()) {
    int32_t height = 0;
    int32_t width = 0;
    if (!dyn_shape_proc_.CheckAndGetImageSize(new_shapes, &height, &width)) {
      MS_LOG(ERROR) << "Failed to check image size";
      return false;
    }
    MS_LOG(INFO) << "Set Image size(" << height << "," << width << ") of input " << index << ".";
    ret = aclmdlSetDynamicHWSize(model_id_, inputs_, index, height, width);
    if (ret != ACL_ERROR_NONE) {
      MS_LOG(ERROR) << "Set dynamic batch size failed, model_id is " << model_id_;
      return false;
    }
  } else if (IsDynamicDims()) {
    aclmdlIODims dynamic_dims;
    if (!dyn_shape_proc_.CheckAndGetDynamicDims(new_shapes, &dynamic_dims)) {
      MS_LOG(ERROR) << "CheckAndGetDynamicDims failed.";
      return false;
    }
    ret = aclmdlSetInputDynamicDims(model_id_, inputs_, index, &dynamic_dims);
    if (ret != ACL_ERROR_NONE) {
      MS_LOG(ERROR) << "aclmdlSetInputDynamicDims failed.";
      return false;
    }
  } else {
    MS_LOG(ERROR) << "Not support dynamic input";
    return false;
  }
  if (!ResetInputSize(new_shapes)) {
    MS_LOG(ERROR) << "Reset input size failed";
    return false;
  }
  if (!ResetOutputSize()) {
    MS_LOG(ERROR) << "Reset output size failed";
    return false;
  }
  return true;
}

bool ModelProcess::CheckInputTensors(const std::vector<KernelTensor *> &input_tensors) {
  if (data_input_num_ != input_tensors.size()) {
    MS_LOG(ERROR) << "Expect input size to be " << data_input_num_ << ", but got " << input_tensors.size();
    return false;
  }
  for (size_t i = 0; i < input_tensors.size(); ++i) {
    auto &tensor = input_tensors[i];
    auto &info = input_infos_[i];
    if (tensor->GetShapeVector() != info.dims) {
      MS_LOG(WARNING) << "Note: input " << i << " shape not match, required " << ShapeToString(info.dims) << ", given "
                      << ShapeToString(tensor->GetShapeVector()) << "."
                      << "Please check input shape has been modified by DVPP method.";
    }
    if (tensor->dtype_id() != TransToDataType(info.data_type)) {
      MS_LOG(ERROR) << "Note: input " << i << " data type not match, required "
                    << static_cast<int>(TransToDataType(info.data_type)) << ", given "
                    << static_cast<int>(tensor->dtype_id());
      return false;
    }
    auto device_data = tensor->GetData();
    auto host_data = tensor->GetHostData();
    if (device_data != nullptr && device_data->addr != nullptr) {
      if (!is_dynamic_input_ && !is_dynamic_shape_range_ && device_data->size != info.buffer_size) {
        MS_LOG(ERROR) << "Input " << i << " data size not match, required size " << info.buffer_size << ", given count "
                      << device_data->size;
        return false;
      }
    } else if (host_data != nullptr && host_data->addr != nullptr) {
      if (!is_dynamic_input_ && !is_dynamic_shape_range_ && host_data->size != info.buffer_size) {
        MS_LOG(ERROR) << "Input " << i << " data size not match, required size " << info.buffer_size << ", given count "
                      << host_data->size;
        return false;
      }
    } else {
      MS_LOG(ERROR) << "Failed to get data from input " << i;
      return false;
    }
  }
  return true;
}

bool ModelProcess::CheckOutputTensors(const std::vector<KernelTensor *> &outputs) {
  if (outputs.size() != output_infos_.size()) {
    MS_LOG(ERROR) << "Actual tensor count not match, required count " << output_infos_.size() << ", given count "
                  << outputs.size();
    return false;
  }
  if (is_dynamic_output_) {
    MS_LOG(INFO) << "This Model has dynamic output shape.";
    return true;
  }
  for (size_t i = 0; i < outputs.size(); ++i) {
    auto &tensor = outputs[i];
    auto &info = output_infos_[i];
    if (tensor->GetShapeVector() != info.dims) {
      MS_LOG(WARNING) << "Note: output " << i << " shape not match, required " << ShapeToString(info.dims) << ", given "
                      << ShapeToString(tensor->GetShapeVector()) << "."
                      << "Please check output shape.";
    }
    if (tensor->dtype_id() != TransToDataType(info.data_type)) {
      MS_LOG(ERROR) << "Note: output " << i << " data type not match, required "
                    << static_cast<int>(TransToDataType(info.data_type)) << ", given "
                    << static_cast<int>(tensor->dtype_id());
      return false;
    }
    auto device_data = tensor->GetData();
    auto host_data = tensor->GetHostData();
    if (device_data != nullptr && device_data->addr != nullptr) {
      if (device_data->size != info.buffer_size) {
        MS_LOG(ERROR) << "Output " << i << " device data size not match, required size " << info.buffer_size
                      << ", given count " << tensor->GetData()->size;
        return false;
      }
    } else if (host_data != nullptr && host_data->addr != nullptr) {
      if (host_data->size != info.buffer_size) {
        MS_LOG(ERROR) << "Output " << i << " host data size not match, required size " << info.buffer_size
                      << ", given count " << tensor->GetData()->size;
        return false;
      }
    } else {
      MS_LOG(ERROR) << "Failed to get data from output " << i;
      return false;
    }
  }
  return true;
}

bool ModelProcess::CheckAndInitInput(const std::vector<KernelTensor *> &inputs) {
  // check inputs
  if (!CheckInputTensors(inputs)) {
    MS_LOG(ERROR) << "Check input tensor failed.";
    return false;
  }
  aclError ret;
  // copy inputs
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto &info = input_infos_[i];
    auto input = inputs[i];
    void *input_buffer = nullptr;
    auto device_data = input->GetData();
    auto host_data = input->GetHostData();
    if (device_data && device_data->addr) {
      auto input_device_id = input->device_id();
      if (input_device_id == IntToUint(device_id_)) {
        input_buffer = device_data->addr;
      } else {
        // memcpy device data from src device to current device.
        auto data_copy_size = inputs[i]->size();
        if (AscendAllocatorPlugin::GetInstance().CopyDeviceDataToDevice(device_data->addr, info.device_data,
                                                                        data_copy_size, info.buffer_size,
                                                                        input_device_id, device_id_) != kSuccess) {
          MS_LOG(ERROR) << "Copy input data from device to current device failed.";
          return false;
        }
        input_buffer = info.device_data;
      }
    } else {
      auto data = host_data->addr;
      auto size = host_data->size;
      if (!is_run_on_device_) {
        ret = AclrtMemcpy(info.device_data, info.buffer_size, data, size, ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_ERROR_NONE) {
          MS_LOG(ERROR) << "Acl memcpy input " << i << " data to device failed, src input size: " << size
                        << ", dst device buffer size: " << info.buffer_size;
          return false;
        }
        input_buffer = info.device_data;
      } else {
        input_buffer = data;
      }
    }
    auto data_buffer = CALL_ASCEND_API(aclmdlGetDatasetBuffer, inputs_, i);
    if (data_buffer == nullptr) {
      MS_LOG(ERROR) << "Failed to get dataset buffer of input " << i;
      return false;
    }
    ret = aclUpdateDataBuffer(data_buffer, input_buffer, info.buffer_size);
    if (ret != ACL_ERROR_NONE) {
      MS_LOG(ERROR) << "Failed to update Data Buffer of input " << i << ", buffer size: " << info.buffer_size
                    << ", input shape: " << input->GetShapeVector();
      return false;
    }
  }
  return true;
}

void ModelProcess::CheckAndInitDynOutputDeviceBuf(const KernelTensor *output, const AclTensorInfo &output_info,
                                                  void **output_device_buffer, size_t *output_buf_size,
                                                  size_t output_idx) {
  auto device_data = output->GetData();
  auto host_data = output->GetHostData();
  if ((host_data == nullptr) || (dyn_out_sys_buf_addr_.find(host_data->addr) != dyn_out_sys_buf_addr_.end()) ||
      (host_data->size == 0)) {
    MS_LOG(DEBUG) << "host_data->addr: " << host_data->addr
                  << ", user not defined dynamic output buffer on host, using system defined buffer";
    user_defined_output_buf_[output_idx] = false;
  }
  if (user_defined_output_buf_[output_idx]) {
    *output_device_buffer = output_info.device_data;
    auto addr = (host_data != nullptr) ? host_data->addr : device_data->addr;
    auto size = (host_data != nullptr) ? host_data->size : device_data->size;
    *output_buf_size = size;
    MS_LOG(DEBUG) << "found user buffer with addr: " << addr << " with size: " << size
                  << ". init output device addr: " << output_info.device_data;
  }
}

bool ModelProcess::CheckAndInitOutput(const std::vector<KernelTensor *> &outputs) {
  // check outputs
  if (!CheckOutputTensors(outputs)) {
    MS_LOG(ERROR) << "Check output tensor failed.";
    return false;
  }
  aclError ret;
  // copy outputs
  for (size_t i = 0; i < outputs.size(); ++i) {
    auto &info = output_infos_[i];
    auto output = outputs[i];
    void *output_device_buffer = nullptr;
    auto device_data = output->GetData();
    auto host_data = output->GetHostData();
    auto output_device_id = output->device_id();
    auto output_device_buffer_size = info.buffer_size;
    bool is_dynamic = is_dynamic_input_ || is_dynamic_shape_range_ || is_dynamic_output_;
    if (device_data && device_data->addr) {
      output_device_buffer = (output_device_id == IntToUint(device_id_)) ? device_data->addr : info.device_data;
      if (is_dynamic) {
        output_device_buffer_size = device_data->size;  // device data buffer size is needed for memory alloc
      }
      MS_LOG(DEBUG) << "user defined output device data addr: " << output_device_buffer
                    << ", with size: " << output_device_buffer_size;
    } else if (host_data && host_data->addr && is_run_on_device_) {
      output_device_buffer = host_data->addr;
    } else {
      output_device_buffer = info.device_data;
      if (is_dynamic) {
        output_device_buffer = nullptr;  // in dynamic output shape, setting nullptr allows acl to alloc memory
        output_device_buffer_size = 0;
        CheckAndInitDynOutputDeviceBuf(output, info, &output_device_buffer, &output_device_buffer_size, i);
      }
    }
    auto data_buffer = CALL_ASCEND_API(aclmdlGetDatasetBuffer, outputs_, i);
    if (data_buffer == nullptr) {
      MS_LOG(ERROR) << "Failed to get dataset buffer of output " << i;
      return false;
    }
    ret = aclUpdateDataBuffer(data_buffer, output_device_buffer, output_device_buffer_size);
    if (ret != ACL_ERROR_NONE) {
      MS_LOG(ERROR) << "Failed to update Data Buffer of output " << i << ", buffer size: " << info.buffer_size
                    << ", output shape: " << output->GetShapeVector();
      return false;
    }
  }
  return true;
}

bool ModelProcess::ResetDynamicOutputTensor(const std::vector<KernelTensor *> &outputs) {
  dyn_out_sys_buf_addr_.clear();
  for (size_t i = 0; i < output_infos_.size(); ++i) {
    auto &output = outputs[i];
    auto &output_info = output_infos_[i];

    // get actual output tensor info
    aclTensorDesc *tensor_info = aclmdlGetDatasetTensorDesc(outputs_, i);
    size_t output_desc_size = aclGetTensorDescSize(tensor_info);
    if (output_desc_size == 0) {
      MS_LOG(ERROR) << "dynamic output size from acl inference result is 0, please check graph or inputs";
      return false;
    }
    aclDataBuffer *data_buffer = CALL_ASCEND_API(aclmdlGetDatasetBuffer, outputs_, i);
    void *acl_device_data = aclGetDataBufferAddr(data_buffer);

    // update host address and size
    auto host_data = output->GetHostData();
    auto device_data = output->GetData();
    if (device_data && device_data->addr) {
      MS_LOG(DEBUG) << "data on device, no need to update system allocated buffer";
      auto output_device_id = output->device_id();
      output->SetHostData(nullptr);
      output->SetData(std::make_shared<kernel::Address>(acl_device_data, output_desc_size));
      if (output_device_id != IntToUint(device_id_)) {
        MS_LOG(DEBUG) << "output across device, tensor on device " << output_device_id << " with addr "
                      << device_data->addr << ", infer on device " << device_id_ << " with addr " << acl_device_data;
        output->SetData(std::make_shared<kernel::Address>(device_data->addr, output_desc_size));
        output_info.cur_device_data = acl_device_data;
      }
    } else {
      if (!user_defined_output_buf_[i]) {
        // data_buf_ptr is passed to tensor ref data and will be freed in destructor
        void *data_buf_ptr = kernel::AscendAllocatorPlugin::GetInstance().MallocHost(output_desc_size);
        output->SetHostData(std::make_shared<kernel::Address>(data_buf_ptr, output_desc_size));
        output->SetData(nullptr);
        (void)dyn_out_sys_buf_addr_.insert(output->GetHostData()->addr);
        MS_LOG(DEBUG) << "no user provided output buffer, memory alloc by system with addr: "
                      << output->GetHostData()->addr << ", size: " << output_desc_size;
      } else {
        if (host_data == nullptr) {
          MS_LOG(ERROR) << "critical error! found user defined buffer nullptr";
          return false;
        }
        MS_LOG(DEBUG) << "found user provided buffer addr: " << host_data->addr << ", size: " << host_data->size
                      << " no need to update system allocated buffer";
      }
    }

    // update acl tensor info
    size_t dim_nums = CALL_ASCEND_API(aclGetTensorDescNumDims, tensor_info);
    ShapeVector shape;
    for (size_t j = 0; j < dim_nums; ++j) {
      int64_t shape_j = aclGetTensorDescDim(tensor_info, j);
      shape.emplace_back(shape_j);
    }
    output->SetShapeVector(shape);
    output_info.device_data = acl_device_data;
    output_info.cur_device_data = acl_device_data;
    output_info.buffer_size = output_desc_size;
    output_info.malloc_buffer_size = output_desc_size;
  }
  return true;
}

bool ModelProcess::PredictFromHost(const std::vector<KernelTensor *> &inputs,
                                   const std::vector<KernelTensor *> &outputs) {
  if (!loaded_) {
    MS_LOG(ERROR) << "Model has not been loaded";
    return false;
  }
  if (!CheckAndInitInput(inputs)) {
    MS_LOG(ERROR) << "Check or init input failed";
    return false;
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    if (user_defined_output_buf_.size() < outputs.size()) {
      user_defined_output_buf_.push_back(true);
    } else {
      user_defined_output_buf_[i] = true;
    }
  }
  if (!CheckAndInitOutput(outputs)) {
    MS_LOG(ERROR) << "Check output tensor failed";
    return false;
  }

  aclError acl_ret;
  struct timeval start_time;
  auto env = std::getenv("GLOG_v");
  bool output_timecost = (env != nullptr && (env[0] == kINFOLogLevel || env[0] == kDEBUGLogLevel));
  if (output_timecost) {
    (void)gettimeofday(&start_time, nullptr);
  }

  if (is_sharing_workspace_) {
    MS_LOG(DEBUG) << "Need to lock before aclmdlExecute.";
    AclMemManager::GetInstance().Lock();
  }
  acl_ret = CALL_ASCEND_API(aclmdlExecute, model_id_, inputs_, outputs_);
  if (is_sharing_workspace_) {
    MS_LOG(DEBUG) << "Unlock after aclmdlExecute.";
    AclMemManager::GetInstance().Unlock();
  }
  if (output_timecost) {
    struct timeval end_time;
    (void)gettimeofday(&end_time, nullptr);
    constexpr uint64_t kUSecondInSecond = 1000000;
    uint64_t cost =
      (kUSecondInSecond * static_cast<uint64_t>(end_time.tv_sec) + static_cast<uint64_t>(end_time.tv_usec)) -
      (kUSecondInSecond * static_cast<uint64_t>(start_time.tv_sec) + static_cast<uint64_t>(start_time.tv_usec));
    MS_LOG(INFO) << "Model execute in " << cost << " us";
  }

  if (acl_ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "Execute Model Failed, ret = " << acl_ret << ", detail:" << CALL_ASCEND_API(aclGetRecentErrMsg);
    return false;
  }
  if (is_dynamic_output_) {
    bool ret = ResetDynamicOutputTensor(outputs);
    if (!ret) {
      return false;
    }
  }
  if (!GetOutputs(outputs)) {
    MS_LOG(ERROR) << "Build outputs failed";
    return false;
  }
  // The device_data is malloced by acl, user need to free the addr
  if (is_dynamic_output_) {
    FreeResourceOutput(&output_infos_, outputs);
  }
  MS_LOG(INFO) << "Execute model success";
  return true;
}

void ModelProcess::FreeResourceInput(std::vector<AclTensorInfo> acl_tensor_info) {
  for (const auto &item : acl_tensor_info) {
    if (item.dynamic_acl_tensor_desc != nullptr) {
      CALL_ASCEND_API(aclDestroyTensorDesc, item.dynamic_acl_tensor_desc);
    }
    if (is_dynamic_resize_input_) {
      if (item.device_data != nullptr) {
        if (!is_run_on_device_) {
          CALL_ASCEND_API(aclrtFree, item.device_data);
        } else {
          CALL_ASCEND_API(aclrtFreeHost, item.device_data);
        }
      }
      if (item.dynamic_acl_data_buffer != nullptr) {
        CALL_ASCEND_API(aclDestroyDataBuffer, item.dynamic_acl_data_buffer);
      }
    }
  }
  if (is_dynamic_resize_input_) {
    CALL_ASCEND_API(aclmdlDestroyDataset, inputs_);
    inputs_ = nullptr;
  }
}

void ModelProcess::FreeResourceOutput(std::vector<AclTensorInfo> *acl_tensor_info,
                                      const std::vector<KernelTensor *> &outputs) {
  for (size_t i = 0; i < acl_tensor_info->size(); i++) {
    auto &item = (*acl_tensor_info)[i];
    auto &output = outputs[i];
    auto device_data = output->GetData();
    if ((device_data && device_data->addr) || user_defined_output_buf_[i]) {
      MS_LOG(DEBUG) << "found data managed by the user, skipping resource release";
      continue;
    }
    if (item.device_data != nullptr) {
      MS_LOG(DEBUG) << "freeing device buffer at addr: " << item.device_data;
      if (!is_run_on_device_) {
        CALL_ASCEND_API(aclrtFree, item.device_data);
      } else {
        CALL_ASCEND_API(aclrtFreeHost, item.device_data);
      }
      item.device_data = nullptr;
    }
    if (item.dynamic_acl_data_buffer != nullptr) {
      CALL_ASCEND_API(aclDestroyDataBuffer, item.dynamic_acl_data_buffer);
    }
    if (item.dynamic_acl_tensor_desc != nullptr) {
      CALL_ASCEND_API(aclDestroyTensorDesc, item.dynamic_acl_tensor_desc);
    }
  }
}

bool ModelProcess::GetOutputs(const std::vector<KernelTensor *> &outputs) {
  aclrtMemcpyKind kind = is_run_on_device_ ? ACL_MEMCPY_HOST_TO_HOST : ACL_MEMCPY_DEVICE_TO_HOST;
  for (size_t i = 0; i < output_infos_.size(); ++i) {
    auto &output = outputs[i];
    auto &output_info = output_infos_[i];
    if (output_info.cur_device_data == nullptr) {
      MS_LOG(WARNING) << "Output device add is nullptr.";
      continue;
    }
    auto host_data = output->GetHostData();
    auto output_device_id = output->device_id();
    if (host_data && host_data->addr && !is_run_on_device_) {
      if (host_data->size != output_info.buffer_size) {
        MS_LOG(ERROR) << "Specified output host data size " << host_data->size << " != execute output data size "
                      << output_info.buffer_size << ", output shape: " << output_info.dims;
        return false;
      }
      MS_LOG(DEBUG) << "copying to host with addr: " << host_data->addr << " with size: " << output_info.buffer_size;
      auto ret =
        AclrtMemcpy(host_data->addr, host_data->size, output_info.cur_device_data, output_info.buffer_size, kind);
      if (ret != ACL_ERROR_NONE) {
        MS_LOG(ERROR) << "Memcpy output " << i << " from " << (is_run_on_device_ ? "host" : "device")
                      << " to host failed, memory size " << output_info.buffer_size << ", ret: " << ret;
        return false;
      }
    } else if (output_device_id != IntToUint(device_id_)) {
      // memcpy output data from current device to output device.
      if (AscendAllocatorPlugin::GetInstance().CopyDeviceDataToDevice(
            output_info.cur_device_data, output->GetData()->addr, output->size(), output_info.buffer_size, device_id_,
            output_device_id) != kSuccess) {
        MS_LOG(ERROR) << "Copy output data from device to current device failed.";
        return false;
      }
    }
  }
  return true;
}
}  // namespace acl
}  // namespace mindspore::kernel
