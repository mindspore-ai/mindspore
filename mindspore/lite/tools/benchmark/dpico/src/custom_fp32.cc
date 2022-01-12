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

#include "src/custom_fp32.h"
#include <map>
#include <memory>
#include <iomanip>
#include <algorithm>
#include "schema/model_generated.h"
#include "include/registry/register_kernel.h"
#include "include/api/context.h"
#include "src/custom_allocator.h"

using mindspore::schema::PrimitiveType_Custom;
constexpr int RET_OK = 0;        /**< No error occurs. */
constexpr int RET_ERROR = -1;    /**< Common error code. */
constexpr int RET_NULL_PTR = -2; /**< NULL pointer returned.*/

namespace mindspore {
namespace lite {
namespace {
constexpr size_t kBitNumOfOneByte = 8;

constexpr int TOP_LEFT_X = 0;
constexpr int TOP_LEFT_Y = 1;
constexpr int BOTTOM_RIGHT_X = 2;
constexpr int BOTTOM_RIGHT_Y = 3;
constexpr int SCORE = 4;
constexpr int CLASS_ID = 5;
constexpr int BBOX_SIZE = 6;

constexpr int NMS_THR = 0;
constexpr int SCORE_THR = 1;
constexpr int MIN_HEIGHT = 2;
constexpr int MIN_WIDTH = 3;

int DetermineInputIndexInOm(const svp_acl_mdl_desc *model_desc, const std::string &tensor_name, size_t *input_index) {
  MS_CHECK_FALSE_MSG(model_desc == nullptr || input_index == nullptr, SVP_ACL_ERROR_INVALID_PARAM,
                     "The function's parameter is nullptr.");
  std::string om_tensor_name = tensor_name;

  // post match
  std::vector<std::string> patterns{"_nh2nc", "_nh2nc", "_post"};
  bool has_matched = false;
  for (const auto &pattern : patterns) {
    if (tensor_name.size() >= pattern.size() && tensor_name.rfind(pattern) == tensor_name.size() - pattern.size()) {
      om_tensor_name = tensor_name.substr(0, tensor_name.size() - pattern.size());
      has_matched = true;
      break;
    }
  }

  // pre match
  if (!has_matched) {
    std::string pattern = "duplicate_";
    auto find_index = tensor_name.find(pattern);
    if (find_index != std::string::npos) {  // todo return size_t
      om_tensor_name = tensor_name.substr(find_index + pattern.size());
    }
  }
  return svp_acl_mdl_get_input_index_by_name(model_desc, om_tensor_name.c_str(), input_index);
}
}  // namespace

size_t CustomCPUKernel::num_of_om_model_ = 0;
dpico::CustomInterface CustomCPUKernel::custom_infershape_ = dpico::CustomInterface();
std::shared_ptr<Allocator> CustomCPUKernel::custom_allocator_ = std::make_shared<dpico::CustomAllocator>();
DpicoConfigParamExtractor CustomCPUKernel::dpico_config_param_extractor_ = DpicoConfigParamExtractor();
DpicoContextManager CustomCPUKernel::dpico_context_manager_ = DpicoContextManager();
DpicoAicpuThreadManager CustomCPUKernel::dpico_aicpu_thread_manager_ = DpicoAicpuThreadManager();

Result CustomCPUKernel::DetermineBatchSize() {
  MS_CHECK_FALSE_MSG(model_desc_ == nullptr, FAILED, "the om hasn't been loaded.");
  if (inputs_.size() < kMinInputSize) {
    MS_LOG(ERROR) << "inputs' num is invalid, which now is less than 2";
    return FAILED;
  }
  std::vector<int> batch_sizes;
  for (size_t index = 0; index < inputs_.size() - 1; ++index) {
    auto lite_shape = inputs_[index].Shape();
    if (lite_shape.empty() || std::any_of(lite_shape.begin(), lite_shape.end(), [](int64_t val) { return val <= 0; })) {
      MS_LOG(ERROR) << "lite shape is invalid, which contains negative.";
      return FAILED;
    }
    svp_acl_mdl_io_dims om_input_info;
    auto ret = GetInputDims(index, &om_input_info);
    if (ret != SUCCESS) {
      MS_LOG(ERROR) << "get input shape from om failed";
      return FAILED;
    }
    auto om_shape = om_input_info.dims;
    if (lite_shape.size() != om_input_info.dim_count) {
      MS_LOG(ERROR) << "lite shape size is different that of om.";
      return FAILED;
    }
    if (std::any_of(om_shape, om_shape + om_input_info.dim_count, [](int64_t val) { return val <= 0; })) {
      MS_LOG(ERROR) << "lite shape is invalid, which contains negative.";
      return FAILED;
    }
    for (size_t dim = 1; dim < lite_shape.size(); ++dim) {
      if (lite_shape[dim] != om_shape[dim]) {
        MS_LOG(ERROR) << "lite shape cannot match om shape.";
        return FAILED;
      }
    }
    if (lite_shape.front() % om_shape[0] != 0) {
      MS_LOG(ERROR) << "lite shape cannot match om shape.";
      return FAILED;
    }
    auto batch_size = lite_shape.front() / om_shape[0];
    if (batch_size > static_cast<int64_t>(INT_MAX)) {
      MS_LOG(ERROR) << "batch size is out of range INT_MAX.";
      return FAILED;
    }
    batch_sizes.push_back(static_cast<int>(batch_size));
  }
  batch_size_ = batch_sizes.front();
  if (std::any_of(batch_sizes.begin(), batch_sizes.end(),
                  [this](int val) { return static_cast<size_t>(val) != batch_size_; })) {
    MS_LOG(ERROR) << "all inputs's batch size is different.";
    return FAILED;
  }
  return SUCCESS;
}

int CustomCPUKernel::LoadModelAndInitResource() {
  OmNetType net_type{OmNetType_CNN};
  int ret = JudgeOmNetType(*primitive_, &net_type);
  if (ret != SUCCESS) {
    MS_LOG(ERROR) << "get model attr failed";
    return FAILED;
  }
  is_detection_net_ = net_type == OmNetType_ROI;
  is_recurrent_net_ = net_type == OmNetType_RECURRENT;
  ret = PrepareDevice();
  if (ret != SUCCESS) {
    MS_LOG(ERROR) << "sample init resource failed";
    return FAILED;
  }
  ret = LoadModel();
  if (ret != SUCCESS) {
    MS_LOG(ERROR) << "execute LoadModel failed";
    return FAILED;
  }
  dpico_aicpu_thread_manager_.CreateAicpuThread(model_id_);
  return SUCCESS;
}
Result CustomCPUKernel::InitInputsLinkMap() {
  for (size_t i = 0; i < inputs_.size() - 1; ++i) {
    std::string tensor_name = inputs_[i].Name();
    size_t index;
    auto status = DetermineInputIndexInOm(model_desc_, tensor_name, &index);
    if (status != SVP_ACL_SUCCESS) {
      MS_LOG(WARNING) << "svp_acl_mdl_get_input_index_by_name fail! ret = " << status
                      << "\ninput num except the last input om model is " << (inputs_.size() - 1)
                      << "\ntensor name is below:";
      for (size_t tensor_index = 0; tensor_index < inputs_.size() - 1; tensor_index++) {
        std::string tensor_name_inner = inputs_[tensor_index].Name();
        MS_LOG(INFO) << "    tensor index: " << tensor_index << ", tensor name: " << tensor_name_inner.c_str();
      }
      MS_LOG(INFO) << "om bottom name is below:";
      auto om_inputs_num = svp_acl_mdl_get_num_inputs(model_desc_);
      for (size_t j = 0; j < om_inputs_num; i++) {
        std::string om_bottom_name = svp_acl_mdl_get_input_name_by_index(model_desc_, j);
        MS_LOG(INFO) << "    om bottom index: " << static_cast<int>(j)
                     << ", om bottom name: " << om_bottom_name.c_str();
      }
      index = i;
      MS_LOG(WARNING) << "    can't find same node, use tensor index " << i;
    }
    MS_LOG(DEBUG) << "input tensor index: " << i << " input om index: " << index;
    inputs_link_map_[i] = index;
  }
  return SUCCESS;
}

Result CustomCPUKernel::InitOutputsLinkMap() {
  std::map<std::string, int> output_tensor_names;
  for (size_t i = 0; i < outputs_.size(); i++) {
    output_tensor_names[outputs_[i].Name()] = i;
  }
  for (size_t i = 0; i < outputs_.size(); ++i) {
    std::string om_index_name = svp_acl_mdl_get_output_name_by_index(model_desc_, i);
    std::string om_index_name_duplicate = om_index_name + "_duplicate";
    if (output_tensor_names.find(om_index_name_duplicate) != output_tensor_names.end()) {
      outputs_link_map_[output_tensor_names[om_index_name_duplicate]] = i;
      MS_LOG(DEBUG) << "output tensor index: " << output_tensor_names[om_index_name_duplicate]
                    << " output om index: " << i;
    } else if (output_tensor_names.find(om_index_name) != output_tensor_names.end()) {
      outputs_link_map_[output_tensor_names[om_index_name]] = i;
      MS_LOG(DEBUG) << "output tensor index: " << output_tensor_names[om_index_name] << " output om index: " << i;
    } else {
      if (outputs_link_map_.find(i) == outputs_link_map_.end()) {
        MS_LOG(WARNING) << "    can't find same node, use tensor index " << i;
        outputs_link_map_[i] = i;
      } else {
        MS_LOG(ERROR) << "can't find correspond node. " << om_index_name;
        return FAILED;
      }
    }
  }
  return SUCCESS;
}

Result CustomCPUKernel::FlushInputsData() {
  for (size_t i = 0; i < inputs_.size() - 1; i++) {
    if (!inputs_mem_aligned_flag_[i]) {
      continue;
    }
    auto input = inputs_[i];
    svp_acl_error ret = svp_acl_rt_mem_flush(input.MutableData(), input.DataSize());
    if (ret != SUCCESS) {
      MS_LOG(ERROR) << "svp acl rt mem flush input: " << input.Name() << " failed.";
      return FAILED;
    }
  }
  return SUCCESS;
}

Result CustomCPUKernel::InvalidateOutputsData() {
  for (size_t i = 0; i < outputs_.size(); i++) {
    if (!outputs_mem_aligned_flag_[i]) {
      continue;
    }
    auto output = outputs_[i];
    svp_acl_error ret = svp_acl_rt_mem_invalidate(output.MutableData(), output.DataSize());
    if (ret != SUCCESS) {
      MS_LOG(ERROR) << "svp acl rt mem invalidate output: " << output.Name() << " failed.";
      return FAILED;
    }
  }
  return SUCCESS;
}
int CustomCPUKernel::Prepare() {
  if (prepared_) {
    return RET_OK;
  }

  if (inputs_.size() < kMinInputSize || outputs_.size() < 1) {
    return RET_ERROR;
  }
  dpico_config_param_extractor_.InitDpicoConfigParam(*this);
  if (!load_flag_) {
    LoadModelAndInitResource();
  }
  if (!InferDone(outputs_)) {
    return RET_OK;
  }

  if (inputs_[0].Shape().size() < 1) {
    return RET_ERROR;
  }

  if (!is_recurrent_net_) {
    if (DetermineBatchSize() != SUCCESS) {
      MS_LOG(ERROR) << "cannot determine batch size.";
      return RET_ERROR;
    }
  } else {
    auto g_total_t = dpico_config_param_extractor_.GetGTotalT();
    if (g_total_t != 0) {
      recurrent_total_t = g_total_t;
      if (recurrent_total_t > (size_t)inputs_[0].Shape().at(0)) {
        MS_LOG(ERROR) << "recurrent_total_t " << recurrent_total_t << " is bigger than batch "
                      << inputs_[0].Shape().at(0) << ", now this condition is not supported";
        return RET_ERROR;
      }
    } else {
      recurrent_total_t = inputs_[0].Shape().at(0);
    }
  }

  int ret = InitInputsLinkMap();
  if (ret != SUCCESS) {
    MS_LOG(ERROR) << "init inputs link map failed.";
    return FAILED;
  }

  ret = InitOutputsLinkMap();
  if (ret != SUCCESS) {
    MS_LOG(ERROR) << "init outputs link map failed.";
    return FAILED;
  }

  ret = CreateInputs();
  if (ret != SUCCESS) {
    MS_LOG(ERROR) << "execute CreateInputs failed";
    return RET_ERROR;
  }

  ret = CreateOutputs();
  if (ret != SUCCESS) {
    MS_LOG(ERROR) << "execute CreateOutputs failed";
    return RET_ERROR;
  }

  prepared_ = true;
  return RET_OK;
}

int CustomCPUKernel::ReSize() {
  if (prepared_) {
    DestroyInput();
    DestroyOutput();
    inputs_data_in_npu_.clear();
    prepared_ = false;
  }
  Prepare();
  return RET_OK;
}

Result CustomCPUKernel::MallocOutputsData() {
  MS_CHECK_TRUE_MSG(custom_allocator_ != nullptr, FAILED, "custom_allocator_ is nullptr.");
  for (auto output : outputs_) {
    auto output_data = output.MutableData();
    MS_CHECK_TRUE_MSG(output_data != nullptr, FAILED, "output_data is nullptr.");
  }
  return SUCCESS;
}
Result CustomCPUKernel::UpdateInputDataset() {
  for (size_t i = 0; i < inputs_.size(); i++) {
    if (inputs_mem_aligned_flag_[i]) {
      svp_acl_data_buffer *data_buffer = svp_acl_mdl_get_dataset_buffer(input_dataset_, i);
      if (data_buffer == nullptr) {
        MS_LOG(ERROR) << "data_buffer is nullptr.";
        return FAILED;
      }
      auto stride = svp_acl_mdl_get_input_default_stride(model_desc_, i);
      svp_acl_error ret =
        svp_acl_update_data_buffer(data_buffer, inputs_[i].MutableData(), inputs_[i].DataSize(), stride);
      if (ret != SUCCESS) {
        MS_LOG(ERROR) << "svp_acl_update_data_buffer input:" << inputs_[i].Name() << " failed.";
        return FAILED;
      }
    }
  }
  return SUCCESS;
}

Result CustomCPUKernel::UpdateOutputDataset() {
  for (size_t i = 0; i < outputs_.size(); i++) {
    if (outputs_mem_aligned_flag_[i]) {
      svp_acl_data_buffer *data_buffer = svp_acl_mdl_get_dataset_buffer(output_dataset_, i);
      if (data_buffer == nullptr) {
        MS_LOG(ERROR) << "data_buffer is nullptr.";
        return FAILED;
      }
      auto stride = svp_acl_mdl_get_output_default_stride(model_desc_, i);
      svp_acl_error ret =
        svp_acl_update_data_buffer(data_buffer, outputs_[i].MutableData(), outputs_[i].DataSize(), stride);
      if (ret != SUCCESS) {
        MS_LOG(ERROR) << "svp_acl_update_data_buffer output:" << outputs_[i].Name() << " failed.";
        return FAILED;
      }
    }
  }
  return SUCCESS;
}

Result CustomCPUKernel::PreExecute() {
  if (!InferDone(outputs_)) {
    if (custom_infershape_.Infer(&inputs_, &outputs_, primitive_, this) != kSuccess) {
      MS_LOG(ERROR) << "infershape failed when running.";
      return FAILED;
    }
    auto ret = ReSize();
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "reSize fail.";
      return FAILED;
    }
  }
  if (MallocOutputsData() != SUCCESS) {
    MS_LOG(ERROR) << "malloc output data fail.";
    return FAILED;
  }
  if (UpdateInputDataset() != SUCCESS) {
    MS_LOG(ERROR) << "update input dataset fail.";
    return FAILED;
  }
  if (UpdateOutputDataset() != SUCCESS) {
    MS_LOG(ERROR) << "update output dataset fail.";
    return FAILED;
  }
  return SUCCESS;
}

int CustomCPUKernel::Execute() {
  enum { TENSOR_FIRST_INDEX = 0, OM_FIRST_INDEX = 0 };
  if (PreExecute() != SUCCESS) {
    MS_LOG(ERROR) << "pre-execute failed.";
    return RET_ERROR;
  }
  int ret = FlushInputsData();
  if (ret != SUCCESS) {
    MS_LOG(ERROR) << "flush inputs data failed";
    return RET_ERROR;
  }
  ret = InvalidateOutputsData();
  if (ret != SUCCESS) {
    MS_LOG(ERROR) << "invalidate outputs data failed";
    return RET_ERROR;
  }
  ret = CopyTensorsToNpuWithStride();
  if (ret != SUCCESS) {
    MS_LOG(ERROR) << "CopyTensorsToNpuWithStride failed";
    return RET_ERROR;
  }
  if (is_detection_net_) {
    UpdateDetParas();
  }
  svp_acl_rt_set_current_context(dpico_context_manager_.GetSvpContext());

  ret = DeviceExecute();
  if (ret != SUCCESS) {
    MS_LOG(ERROR) << "execute inference failed";
    return RET_ERROR;
  }

  ret = InvalidateOutputsData();
  if (ret != SUCCESS) {
    MS_LOG(ERROR) << "invalidate outputs data failed";
    return RET_ERROR;
  }

  if (is_detection_net_) {
    if (dpico_config_param_extractor_.GetDpicoDetectionPostProcess() == 1) {
      OutputModelResult();
      WriteOutputToTensor(TENSOR_FIRST_INDEX, OM_FIRST_INDEX);
    } else {
      DumpModelOutputResultToTensor();
    }
  } else {
    DumpModelOutputResultToTensor();
  }
  return RET_OK;
}

CustomCPUKernel::~CustomCPUKernel() {
  num_of_om_model_--;
  dpico_aicpu_thread_manager_.DestroyAicpuThread();
  UnloadModel();
  DestroyInput();
  DestroyOutput();
  TerminateDevice();
}

std::shared_ptr<mindspore::kernel::Kernel> CustomCreateKernel(const std::vector<MSTensor> &inputs,
                                                              const std::vector<MSTensor> &outputs,
                                                              const mindspore::schema::Primitive *primitive,
                                                              const mindspore::Context *ctx) {
  MS_CHECK_FALSE_MSG(primitive == nullptr, nullptr, "primitive is nullptr");
  MS_CHECK_FALSE_MSG(ctx == nullptr, nullptr, "ctx is nullptr");

  if (primitive->value_type() != mindspore::schema::PrimitiveType_Custom) {
    MS_LOG(ERROR) << "Primitive type is not PrimitiveType_Custom";
    return nullptr;
  }

  auto op = primitive->value_as_Custom();
  MS_CHECK_FALSE_MSG(op == nullptr, nullptr, "op is nullptr");
  MS_CHECK_FALSE_MSG(op->attr() == nullptr, nullptr, "op atrr is nullptr");
  if (op->attr()->size() < 1) {
    MS_LOG(ERROR) << "There are at least 1 attribute of Custom";
    return nullptr;
  }

  auto kernel = std::make_shared<CustomCPUKernel>(inputs, outputs, primitive, ctx);
  MS_CHECK_FALSE_MSG(kernel == nullptr, nullptr, "new custom kernel is nullptr");
  return kernel;
}

Result CustomCPUKernel::LoadModel() {
  auto model_buf = reinterpret_cast<char *>(inputs_[inputs_.size() - 1].MutableData());
  auto model_size = inputs_[inputs_.size() - 1].DataSize();
  svp_acl_error ret = SVP_ACL_SUCCESS;
  ret = svp_acl_rt_malloc(&model_mem_ptr_, model_size, SVP_ACL_MEM_MALLOC_NORMAL_ONLY);
  if (ret != SVP_ACL_SUCCESS) {
    MS_LOG(ERROR) << "malloc device buffer failed. size is " << model_size;
    return FAILED;
  }
  memcpy(model_mem_ptr_, model_buf, model_size);
  ret = svp_acl_mdl_load_from_mem(static_cast<uint8_t *>(model_mem_ptr_), model_size, &model_id_);
  if (ret != SVP_ACL_SUCCESS) {
    svp_acl_rt_free(model_mem_ptr_);
    MS_LOG(ERROR) << "load model from file failed, model file";
    return FAILED;
  }
  load_flag_ = true;
  MS_LOG(INFO) << "load model success";

  // obtain om's desc.
  model_desc_ = svp_acl_mdl_create_desc();
  if (model_desc_ == nullptr) {
    svp_acl_rt_free(model_mem_ptr_);
    MS_LOG(ERROR) << "create model description failed";
    return FAILED;
  }
  auto status = svp_acl_mdl_get_desc(model_desc_, model_id_);
  if (status != SVP_ACL_SUCCESS) {
    svp_acl_rt_free(model_mem_ptr_);
    MS_LOG(ERROR) << "get model description failed";
    return FAILED;
  }
  MS_LOG(INFO) << "create model description success";
  return SUCCESS;
}

Result CustomCPUKernel::CreateInput(void *inputDataBuffer, size_t bufferSize, int stride) {
  MS_CHECK_FALSE_MSG(inputDataBuffer == nullptr, FAILED, "inputdatabuffer is nullptr");
  svp_acl_data_buffer *inputData = svp_acl_create_data_buffer(inputDataBuffer, bufferSize, stride);
  MS_CHECK_FALSE_MSG(inputData == nullptr, FAILED, "can't create data buffer, create input failed");

  svp_acl_error ret = svp_acl_mdl_add_dataset_buffer(input_dataset_, inputData);
  if (ret != SVP_ACL_SUCCESS) {
    MS_LOG(ERROR) << "add input dataset buffer failed";
    svp_acl_destroy_data_buffer(inputData);
    inputData = nullptr;
    return FAILED;
  }

  return SUCCESS;
}

Result CustomCPUKernel::GetInputDims(int index, svp_acl_mdl_io_dims *dims) {
  if (dims == nullptr) {
    return FAILED;
  }
  svp_acl_error ret = svp_acl_mdl_get_input_dims(model_desc_, index, dims);
  if (ret != SVP_ACL_SUCCESS) {
    MS_LOG(ERROR) << "svp_acl_mdl_get_input_dims error!";
    return FAILED;
  }
  return SUCCESS;
}

size_t CustomCPUKernel::GetInputDataSize(int index) {
  svp_acl_data_type dataType = svp_acl_mdl_get_input_data_type(model_desc_, index);
  return svp_acl_data_type_size(dataType) / kBitNumOfOneByte;
}

Result CustomCPUKernel::GetStrideParam(size_t *devSize, int index, size_t *stride, svp_acl_mdl_io_dims *dims) {
  MS_CHECK_FALSE_MSG(devSize == nullptr || stride == nullptr || dims == nullptr, FAILED, "nullptr found");
  Result ret = GetInputDims(index, dims);
  if (ret != SUCCESS) {
    MS_LOG(ERROR) << "GetModelInputDims error!";
    return FAILED;
  }
  *stride = svp_acl_mdl_get_input_default_stride(model_desc_, index);
  if (*stride == 0) {
    MS_LOG(ERROR) << "svp_acl_mdl_get_input_default_stride error!";
    return FAILED;
  }
  *devSize = svp_acl_mdl_get_input_size_by_index(model_desc_, index);
  if (*devSize == 0) {
    MS_LOG(ERROR) << "svp_acl_mdl_get_input_size_by_index error!";
    return FAILED;
  }
  return SUCCESS;
}

void CustomCPUKernel::DestroyInput() {
  if (input_dataset_ == nullptr) {
    return;
  }
  for (size_t i = 0; i < svp_acl_mdl_get_dataset_num_buffers(input_dataset_); ++i) {
    svp_acl_data_buffer *dataBuffer = svp_acl_mdl_get_dataset_buffer(input_dataset_, i);
    if (i < (inputs_.size() - 1) && !inputs_mem_aligned_flag_[i]) {
      void *tmp = svp_acl_get_data_buffer_addr(dataBuffer);
      if (tmp != nullptr) {
        svp_acl_rt_free(tmp);
        tmp = nullptr;
      }
    }
    if (svp_acl_destroy_data_buffer(dataBuffer) != SVP_ACL_SUCCESS) {
      MS_LOG(ERROR) << "svp acl destroy data buffer failed.";
      return;
    }
    dataBuffer = nullptr;
  }
  inputs_mem_aligned_flag_.clear();
  if (svp_acl_mdl_destroy_dataset(input_dataset_) != SVP_ACL_SUCCESS) {
    MS_LOG(ERROR) << "svp acl destroy input data dataset failed";
    return;
  }
  input_dataset_ = nullptr;
}

void CustomCPUKernel::DestroyOutput() {
  if (output_dataset_ == nullptr) {
    return;
  }
  for (size_t i = 0; i < svp_acl_mdl_get_dataset_num_buffers(output_dataset_); ++i) {
    svp_acl_data_buffer *dataBuffer = svp_acl_mdl_get_dataset_buffer(output_dataset_, i);
    if (!outputs_mem_aligned_flag_[i]) {
      void *data = svp_acl_get_data_buffer_addr(dataBuffer);
      if (data != nullptr) {
        svp_acl_rt_free(data);
        data = nullptr;
      }
    }
    if (svp_acl_destroy_data_buffer(dataBuffer) != SVP_ACL_SUCCESS) {
      MS_LOG(ERROR) << "svp acl destroy data buffer failed.";
      return;
    }
    dataBuffer = nullptr;
  }
  outputs_mem_aligned_flag_.clear();
  if (svp_acl_mdl_destroy_dataset(output_dataset_) != SVP_ACL_SUCCESS) {
    MS_LOG(ERROR) << "svp acl destroy output data dataset failed";
    return;
  }
  output_dataset_ = nullptr;
}

Result CustomCPUKernel::CreateOutputs() {
  MS_CHECK_TRUE_MSG(model_desc_ != nullptr, FAILED, "no model description, create output failed");
  MS_CHECK_TRUE_MSG(outputs_mem_aligned_flag_.empty(), FAILED, "outputs_mem_aligned_flag_ should be empty.");
  output_dataset_ = svp_acl_mdl_create_dataset();
  if (output_dataset_ == nullptr) {
    MS_LOG(ERROR) << "can't create dataset, create output failed";
    return FAILED;
  }
  size_t outputSize = svp_acl_mdl_get_num_outputs(model_desc_);
  for (size_t i = 0; i < outputSize; ++i) {
    MS_CHECK_TRUE_MSG(outputs_link_map_.find(i) != outputs_link_map_.end(), FAILED,
                      "can't find correspond output index");
    MS_LOG(DEBUG) << "output tensor index: " << i << " output om index: " << outputs_link_map_[i];
    size_t stride = svp_acl_mdl_get_output_default_stride(model_desc_, outputs_link_map_[i]);
    if (stride == 0) {
      MS_LOG(ERROR) << "output default stride is " << stride;
      return FAILED;
    }
    size_t buffer_size = svp_acl_mdl_get_output_size_by_index(model_desc_, outputs_link_map_[i]);
    if (buffer_size == 0) {
      MS_LOG(ERROR) << "output size is " << buffer_size;
      return FAILED;
    }

    svp_acl_data_buffer *outputData = nullptr;
    void *outputBuffer = nullptr;
    if (outputs_[i].DataSize() != buffer_size * batch_size_) {
      MS_LOG(DEBUG) << "output tensor needs stride. " << outputs_[i].Name();
      outputs_mem_aligned_flag_[i] = false;
      svp_acl_error ret =
        svp_acl_rt_malloc_cached(&outputBuffer, buffer_size * batch_size_, SVP_ACL_MEM_MALLOC_NORMAL_ONLY);
      if (ret != SVP_ACL_SUCCESS) {
        MS_LOG(ERROR) << "can't malloc buffer, size is " << buffer_size << ", create output failed";
        return FAILED;
      }
      outputData = svp_acl_create_data_buffer(outputBuffer, buffer_size * batch_size_, stride);
    } else {
      outputs_mem_aligned_flag_[i] = true;
      outputs_[i].SetAllocator(custom_allocator_);
      outputData = svp_acl_create_data_buffer(outputs_[i].MutableData(), buffer_size * batch_size_, stride);
    }
    MS_CHECK_FALSE_MSG(outputData == nullptr, FAILED, "create data buffer failed.");
    svp_acl_error ret = svp_acl_mdl_add_dataset_buffer(output_dataset_, outputData);
    if (ret != SVP_ACL_SUCCESS) {
      MS_LOG(ERROR) << "can't add data buffer, create output failed";
      if (!outputs_mem_aligned_flag_[i]) {
        svp_acl_rt_free(outputBuffer);
      }
      svp_acl_destroy_data_buffer(outputData);
      return FAILED;
    }
  }

  MS_LOG(INFO) << "create model output success";
  return SUCCESS;
}

bool Cmp(const std::vector<float> &veci, const std::vector<float> &vecj) {
  if (veci[CLASS_ID] < vecj[CLASS_ID]) {
    return true;
  } else if (veci[CLASS_ID] == vecj[CLASS_ID]) {
    return veci[SCORE] > vecj[SCORE];
  }
  return false;
}

void CustomCPUKernel::PrintResultToTensor(const std::vector<std::vector<float>> &boxValue) {
  std::vector<int> clsNum;
  float cId = boxValue[0][CLASS_ID];
  int validNum = 0;
  for (size_t loop = 0; loop < boxValue.size(); loop++) {
    if (boxValue[loop][CLASS_ID] == cId) {
      validNum++;
    } else {
      clsNum.push_back(validNum);
      cId = boxValue[loop][CLASS_ID];
      validNum = 1;
    }
  }
  clsNum.push_back(validNum);
  int idx = 0;
  int sumNum = 0;
  MS_LOG(INFO) << "current class valid box number is: " << clsNum[idx];
  sumNum += clsNum[idx];
  int totalBoxNum = boxValue.size();
  const size_t min_detect_output_size = 2;
  if (outputs_.size() < min_detect_output_size) {
    return;
  }
  float *box_value_in_tensor = reinterpret_cast<float *>(outputs_[1].MutableData());
  if (box_value_in_tensor == nullptr) {
    return;
  }
  if (outputs_[1].ElementNum() < totalBoxNum * BBOX_SIZE) {
    return;
  }
  for (int loop = 0; loop < totalBoxNum; loop++) {
    if (loop == sumNum) {
      idx++;
      MS_LOG(INFO) << "current class valid box number is: " << clsNum[idx];
      sumNum += clsNum[idx];
    }
    MS_LOG(INFO) << "lx: " << boxValue[loop][TOP_LEFT_X] << ", ly: " << boxValue[loop][TOP_LEFT_Y]
                 << ", rx: " << boxValue[loop][BOTTOM_RIGHT_X] << ", ry: " << boxValue[loop][BOTTOM_RIGHT_Y]
                 << ", score: " << boxValue[loop][SCORE] << "; class id: " << boxValue[loop][CLASS_ID];
    box_value_in_tensor[loop] = boxValue[loop][TOP_LEFT_X];
    box_value_in_tensor[totalBoxNum + loop] = boxValue[loop][TOP_LEFT_Y];
    box_value_in_tensor[2 * totalBoxNum + loop] = boxValue[loop][BOTTOM_RIGHT_X];
    box_value_in_tensor[3 * totalBoxNum + loop] = boxValue[loop][BOTTOM_RIGHT_Y];
    box_value_in_tensor[4 * totalBoxNum + loop] = boxValue[loop][SCORE];
    box_value_in_tensor[5 * totalBoxNum + loop] = boxValue[loop][CLASS_ID];
  }
  outputs_[1].SetShape({1, totalBoxNum * BBOX_SIZE});
}

void CustomCPUKernel::OutputModelResult() {
  // yolo/ssd output 0 is num, output 1 is bbox
  enum InputOutputId { INPUT_IMG_ID = 0, OUTPUT_NUM_ID = 0, OUTPUT_BBOX_ID = 1 };
  // get valid box number
  svp_acl_mdl_io_dims aclDims;
  std::vector<int> validBoxNum;
  svp_acl_mdl_get_output_dims(model_desc_, OUTPUT_NUM_ID, &aclDims);
  svp_acl_data_buffer *dataBuffer = svp_acl_mdl_get_dataset_buffer(output_dataset_, OUTPUT_NUM_ID);
  auto outData = reinterpret_cast<float *>(svp_acl_get_data_buffer_addr(dataBuffer));
  for (uint32_t loop = 0; loop < static_cast<uint32_t>(aclDims.dims[aclDims.dim_count - 1]); loop++) {
    validBoxNum.push_back(*(outData + loop));
  }
  int totalValidNum = 0;
  for (size_t loop = 0; loop < validBoxNum.size(); loop++) {
    totalValidNum += validBoxNum[loop];
  }
  if (totalValidNum == 0) {
    MS_LOG(INFO) << "total valid num is zero";
    return;
  }

  // get x y score
  svp_acl_data_buffer *dataBufferValue = svp_acl_mdl_get_dataset_buffer(output_dataset_, OUTPUT_BBOX_ID);
  auto outDataValue = reinterpret_cast<float *>(svp_acl_get_data_buffer_addr(dataBufferValue));
  svp_acl_mdl_get_output_dims(model_desc_, OUTPUT_BBOX_ID, &aclDims);
  if (aclDims.dim_count <= 0) {
    MS_LOG(ERROR) << "aclrtOutputDims error";
    return;
  }

  svp_acl_error ret = svp_acl_mdl_get_input_dims(model_desc_, INPUT_IMG_ID, &aclDims);
  if (ret != SVP_ACL_SUCCESS || aclDims.dim_count <= 2) {
    MS_LOG(ERROR) << "svp_acl_mdl_get_input_dims error!";
    return;
  }
  // input data shape is nchw, 2 is stand h
  int imgHeight = aclDims.dims[aclDims.dim_count - 2];
  int imgWidth = aclDims.dims[aclDims.dim_count - 1];
  MS_LOG(INFO) << "input image width[" << imgWidth << "]; height[" << imgHeight << "]";

  size_t wStrideOffset = svp_acl_mdl_get_output_default_stride(model_desc_, OUTPUT_BBOX_ID) / sizeof(float);
  // box include 6 part which is lx, ly, rx, ry, score, class idq
  std::vector<std::vector<float>> bboxes;
  for (int inx = 0; inx < totalValidNum; inx++) {
    float classId = (*(outDataValue + inx + CLASS_ID * wStrideOffset));
    if (classId == 0.0f) {
      continue;  // skip class 0 back ground
    }
    std::vector<float> bbox(BBOX_SIZE, 0.0f);
    for (size_t loop = 0; loop < BBOX_SIZE; loop++) {
      bbox[loop] = (*(outDataValue + inx + loop * wStrideOffset));
    }
    bboxes.push_back(bbox);
  }
  std::sort(bboxes.begin(), bboxes.end(), Cmp);
  PrintResultToTensor(bboxes);
  MS_LOG(INFO) << "output data success";
  return;
}

void CustomCPUKernel::WriteOutputToTensor(size_t index, size_t output_tensor_index) {
  svp_acl_data_buffer *dataBuffer = svp_acl_mdl_get_dataset_buffer(output_dataset_, index);
  void *data = svp_acl_get_data_buffer_addr(dataBuffer);
  size_t stride = svp_acl_get_data_buffer_stride(dataBuffer);
  svp_acl_data_type dataType = svp_acl_mdl_get_output_data_type(model_desc_, index);
  size_t dataSize = svp_acl_data_type_size(dataType);
  svp_acl_mdl_io_dims dims;
  int ret = svp_acl_mdl_get_output_dims(model_desc_, index, &dims);
  if (ret != SUCCESS) {
    MS_LOG(ERROR) << "get output dims fail";
    return;
  }
  int outNum = 1 * batch_size_;
  for (size_t i = 0; i < dims.dim_count - 1; ++i) {
    outNum *= dims.dims[i];
  }
  if (is_recurrent_net_) {
    if (index == 0) {
      if (dims.dims[0] == 0) {
        return;
      }
      outNum = outNum / dims.dims[0] * recurrent_total_t;
    }
  }
  if (dims.dim_count < 1) {
    return;
  }
  uint32_t w = dims.dims[dims.dim_count - 1];
  auto outTensorData = reinterpret_cast<char *>(outputs_[output_tensor_index].MutableData());
  if (outTensorData == nullptr) {
    return;
  }
  int valid_size_per_stride = ceil(w * dataSize / kBitNumOfOneByte);
  size_t om_valid_data_size = static_cast<size_t>(outNum * valid_size_per_stride);
  if (om_valid_data_size != outputs_[output_tensor_index].DataSize()) {
    MS_LOG(ERROR) << "outnum * valid_size_per_stride in om is not equal to corresponding output tensor";
    return;
  }
  size_t out_data_buffer_size = svp_acl_get_data_buffer_size(dataBuffer);
  ret = svp_acl_rt_mem_flush(data, out_data_buffer_size);
  if (ret != SUCCESS) {
    MS_LOG(ERROR) << "flush error";
    return;
  }
  for (int n = 0; n < outNum; n++) {
    auto outData = reinterpret_cast<char *>(data) + n * stride;
    memcpy(outTensorData + n * valid_size_per_stride, outData, valid_size_per_stride);
  }
  return;
}

void CustomCPUKernel::DumpModelOutputResultToTensor() {
  size_t outputNum = svp_acl_mdl_get_dataset_num_buffers(output_dataset_);
  if (outputs_mem_aligned_flag_.size() != outputNum) {
    MS_LOG(ERROR) << "outputs_mem_aligned_flag_ should be equal to tensor output num.";
    return;
  }
  for (size_t i = 0; i < outputNum; ++i) {
    if (outputs_link_map_.find(i) == outputs_link_map_.end()) {
      MS_LOG(ERROR) << "can't find correspond output index";
      return;
    }
    if (outputs_mem_aligned_flag_[i]) {
      continue;
    }
    WriteOutputToTensor(outputs_link_map_[i], i);
  }
  MS_LOG(INFO) << "dump data success";
}

Result CustomCPUKernel::DeviceExecute() {
  int ret = svp_acl_mdl_execute(model_id_, input_dataset_, output_dataset_);
  if (ret != SVP_ACL_SUCCESS) {
    MS_LOG(ERROR) << "execute model failed, modelId is " << model_id_;
    return FAILED;
  }

  MS_LOG(INFO) << "model execute success";
  return SUCCESS;
}

Result CustomCPUKernel::CreateBuf(int index) {
  void *bufPtr = nullptr;
  size_t bufSize = 0;
  size_t bufStride = 0;
  svp_acl_mdl_io_dims inDims;
  svp_acl_error ret = GetStrideParam(&bufSize, index, &bufStride, &inDims);
  if (ret != SUCCESS) {
    MS_LOG(ERROR) << "get stride param failed";
    return FAILED;
  }

  ret = svp_acl_rt_malloc(&bufPtr, bufSize, SVP_ACL_MEM_MALLOC_NORMAL_ONLY);
  if (ret != SVP_ACL_SUCCESS) {
    MS_LOG(ERROR) << "malloc device buffer failed. size is " << bufSize;
    return FAILED;
  }
  ret = CreateInput(bufPtr, bufSize, bufStride);
  if (ret != SUCCESS) {
    MS_LOG(ERROR) << "execute CreateInput failed";
    svp_acl_rt_free(bufPtr);
    return FAILED;
  }
  return SUCCESS;
}

Result CustomCPUKernel::CreateTaskBufAndWorkBuf() {
  // 2 is stand taskbuf and workbuf
  if (svp_acl_mdl_get_num_inputs(model_desc_) <= 2) {
    MS_LOG(ERROR) << "input dataset Num is error.";
    return FAILED;
  }
  int datasetSize = svp_acl_mdl_get_dataset_num_buffers(input_dataset_);
  if (datasetSize == 0) {
    MS_LOG(ERROR) << "input dataset Num is 0.";
    return FAILED;
  }
  for (size_t loop = datasetSize; loop < svp_acl_mdl_get_num_inputs(model_desc_); loop++) {
    Result ret = CreateBuf(loop);
    if (ret != SUCCESS) {
      MS_LOG(ERROR) << "execute Create taskBuffer and workBuffer failed";
      return FAILED;
    }
  }
  return SUCCESS;
}

void CustomCPUKernel::UnloadModel() {
  if (load_flag_) {
    svp_acl_error ret = svp_acl_mdl_unload(model_id_);
    if (ret != SVP_ACL_SUCCESS) {
      MS_LOG(ERROR) << "unload model failed, modelId is " << model_id_;
    } else {
      MS_LOG(INFO) << "unload model success, modelId is " << model_id_;
    }
    load_flag_ = false;
  }

  if (model_desc_ != nullptr) {
    (void)svp_acl_mdl_destroy_desc(model_desc_);
    model_desc_ = nullptr;
  }

  if (model_mem_ptr_ != nullptr) {
    svp_acl_rt_free(model_mem_ptr_);
    model_mem_ptr_ = nullptr;
  }
}

Result CustomCPUKernel::CopyTensorsToNpuWithStride() {
  for (size_t index = 0; index < inputs_.size() - 1; ++index) {
    if (inputs_mem_aligned_flag_[index]) {
      MS_LOG(DEBUG) << "`aligned input tensor no need to copy memory. ";
      continue;
    }
    auto tensor = inputs_[index];
    size_t devSize;
    size_t stride;
    svp_acl_mdl_io_dims dims;
    Result ret = GetStrideParam(&devSize, index, &stride, &dims);
    if (ret != SUCCESS) {
      MS_LOG(ERROR) << "get stride param erro";
      return ret;
    }
    size_t dataSize = GetInputDataSize(index);
    auto tensor_data = tensor.MutableData();
    if (tensor_data == nullptr) {
      return FAILED;
    }
    uint32_t loopTimes = dims.dims[0] * batch_size_;
    if (is_recurrent_net_ && index == 0) {
      loopTimes = recurrent_total_t;
      if (dims.dim_count < 3) {
        MS_LOG(ERROR) << "recurrent first input must be not less than 3";
        return FAILED;
      }
      MS_LOG(INFO) << "lstm net, the index 0 real dims in om is " << dims.dims[0] << ", " << dims.dims[1] << ", "
                   << dims.dims[2];
      MS_LOG(INFO) << "lstm net, the index 0 valid dims in om is " << recurrent_total_t << ", " << dims.dims[1] << ", "
                   << dims.dims[2];
    }
    for (size_t loop = 1; loop < dims.dim_count - 1; loop++) {
      loopTimes *= dims.dims[loop];
    }
    devSize = loopTimes * stride;
    int dimValue = dims.dims[dims.dim_count - 1];
    if (is_recurrent_net_ && index == 1) {
      dimValue = recurrent_total_t;
      if (dims.dim_count == 2) {
        MS_LOG(INFO) << "lstm net, the index 1 real dims in om is " << dims.dims[0] << "," << dims.dims[1];
        MS_LOG(INFO) << "lstm net, the index 1 valid dims in om is " << dims.dims[0] << ", " << dimValue;
      } else if (dims.dim_count == 3) {
        MS_LOG(INFO) << "lstm net, the index 1 real dims in om is " << dims.dims[0] << ", " << dims.dims[1] << ", "
                     << dims.dims[2];
        MS_LOG(INFO) << "lstm net, the index 1 valid dims in om is " << dims.dims[0] << ", " << dims.dims[1] << ", "
                     << dimValue;
      } else {
        MS_LOG(ERROR) << "unsupported dims count";
        return FAILED;
      }
    }
    for (uint32_t loop = 0; loop < loopTimes; loop++) {
      memcpy((static_cast<char *>(inputs_data_in_npu_[index]) + loop * stride),
             (static_cast<char *>(tensor_data) + loop * dimValue * dataSize), dimValue * dataSize);
    }
  }
  return SUCCESS;
}

void *CustomCPUKernel::GetDeviceBufferOfTensor(const svp_acl_mdl_io_dims &dims, const size_t &stride) {
  void *input_buff = nullptr;
  svp_acl_error ret = SVP_ACL_SUCCESS;
  uint32_t loopTimes = dims.dims[0] * batch_size_;
  for (size_t loop = 1; loop < dims.dim_count - 1; loop++) {
    loopTimes *= dims.dims[loop];
  }
  uint32_t devSize = loopTimes * stride;
  ret = svp_acl_rt_malloc(&input_buff, devSize, SVP_ACL_MEM_MALLOC_NORMAL_ONLY);
  if (ret != SVP_ACL_SUCCESS) {
    MS_LOG(ERROR) << "malloc device buffer failed. size is " << devSize;
    return nullptr;
  }
  return input_buff;
}

Result CustomCPUKernel::PrepareDevice() {
  if (dpico_context_manager_.InitContext(dpico_config_param_extractor_.GetDpicoDumpConfigFile()) != SUCCESS) {
    MS_LOG(ERROR) << "dpico init resource failed.";
    return FAILED;
  }

  // create stream
  auto ret = svp_acl_rt_create_stream(&stream_);
  if (ret != SUCCESS) {
    MS_LOG(ERROR) << "acl create stream failed";
    return FAILED;
  }
  MS_LOG(INFO) << "create stream success";

  // get run mode
  svp_acl_rt_run_mode runMode;
  ret = svp_acl_rt_get_run_mode(&runMode);
  if (ret != SUCCESS) {
    MS_LOG(ERROR) << "acl get run mode failed";
    return FAILED;
  }
  bool is_acl_device = (runMode == SVP_ACL_DEVICE);
  if (!is_acl_device) {
    MS_LOG(ERROR) << "acl run mode failed";
    return FAILED;
  }
  MS_LOG(INFO) << "get run mode success";
  return SUCCESS;
}

Result CustomCPUKernel::CreateInputs() {
  input_dataset_ = svp_acl_mdl_create_dataset();
  MS_CHECK_FALSE_MSG(input_dataset_ == nullptr, FAILED, "can't create dataset, create input failed");
  for (size_t loop = 0; loop < inputs_.size() - 1; ++loop) {
    MS_CHECK_TRUE_MSG(inputs_link_map_.find(loop) != inputs_link_map_.end(), FAILED,
                      "can't find correspond input index");
    size_t index = inputs_link_map_[loop];
    MS_LOG(DEBUG) << "input tensor index: " << loop << " input om index: " << index;
    MS_LOG(INFO) << "start to process inputs: " << loop;
    size_t devSize;
    size_t stride;
    svp_acl_mdl_io_dims inputDims;
    Result ret = GetStrideParam(&devSize, index, &stride, &inputDims);
    if (ret != SUCCESS) {
      MS_LOG(ERROR) << "get stride param error";
      return ret;
    }
    void *picDevBuffer = nullptr;
    if (inputs_[loop].DataSize() != devSize * batch_size_) {
      MS_LOG(DEBUG) << "input tensor needs stride. " << inputs_[index].Name();
      inputs_mem_aligned_flag_[loop] = false;
      picDevBuffer = GetDeviceBufferOfTensor(inputDims, stride);
    } else {
      inputs_mem_aligned_flag_[loop] = true;
      inputs_[index].SetAllocator(custom_allocator_);
      picDevBuffer = reinterpret_cast<void *>(inputs_[loop].MutableData());
    }

    if (picDevBuffer == nullptr) {
      MS_LOG(ERROR) << "get pic device buffer failed,index is " << loop;
      return FAILED;
    }
    inputs_data_in_npu_.push_back(picDevBuffer);
    ret = CreateInput(reinterpret_cast<uint8_t *>(picDevBuffer), devSize * batch_size_, stride);
    if (ret != SUCCESS) {
      MS_LOG(ERROR) << "execute CreateInput failed";
      if (!inputs_mem_aligned_flag_[loop]) {
        svp_acl_rt_free(picDevBuffer);
      }
      return FAILED;
    }
  }
  if (is_detection_net_) {
    int ret = SetDetParas();
    if (ret != SUCCESS) {
      MS_LOG(ERROR) << "SetDetParas failed";
      return FAILED;
    }
  }
  auto ret = CreateTaskBufAndWorkBuf();
  if (ret != SUCCESS) {
    MS_LOG(ERROR) << "CreateTaskBufAndWorkBuf inference failed";
    return FAILED;
  }

  if (is_recurrent_net_) {
    auto status = svp_acl_mdl_set_total_t(model_id_, input_dataset_, recurrent_total_t);
    if (status != SVP_ACL_SUCCESS) {
      MS_LOG(ERROR) << "svp_acl_mdl_set_total_t failed";
      return FAILED;
    }
  }
  if (batch_size_ > 1) {
    size_t index;
    auto status = svp_acl_mdl_get_input_index_by_name(model_desc_, SVP_ACL_DYNAMIC_TENSOR_NAME, &index);
    if (status != SVP_ACL_SUCCESS) {
      MS_LOG(ERROR) << "svp_acl_mdl_get_input_index_by_name fail! ret = " << ret;
      return FAILED;
    }
    status = svp_acl_mdl_set_dynamic_batch_size(model_id_, input_dataset_, index, batch_size_);
    if (status != SVP_ACL_SUCCESS) {
      MS_LOG(ERROR) << "svp_acl_mdl_set_dynamic_batch_size fail! ret = " << ret;
      return FAILED;
    }
  }
  return SUCCESS;
}

void CustomCPUKernel::TerminateDevice() {
  svp_acl_error ret;
  if (stream_ != nullptr) {
    ret = svp_acl_rt_destroy_stream(stream_);
    if (ret != SVP_ACL_SUCCESS) {
      MS_LOG(ERROR) << "destroy stream failed";
    }
    stream_ = nullptr;
  }
  MS_LOG(INFO) << "end to destroy stream";

  if (num_of_om_model_ == 0) {
    dpico_context_manager_.DestroyContext();
  }
}

void CustomCPUKernel::UpdateDetParas() {
  dpico_config_param_extractor_.UpdateDpicoConfigParam(*this);
  det_param_buf_float_[NMS_THR] = dpico_config_param_extractor_.GetNmsThreshold();
  det_param_buf_float_[SCORE_THR] = dpico_config_param_extractor_.GetScoreThreshold();
  det_param_buf_float_[MIN_HEIGHT] = dpico_config_param_extractor_.GetMinHeight();
  det_param_buf_float_[MIN_WIDTH] = dpico_config_param_extractor_.GetMinWidth();
}

Result CustomCPUKernel::SetDetParas() {
  void *bufPtr = nullptr;
  size_t bufferSize = 4u * sizeof(float);
  svp_acl_error ret = svp_acl_rt_malloc(&bufPtr, bufferSize, SVP_ACL_MEM_MALLOC_NORMAL_ONLY);
  if (ret != SVP_ACL_SUCCESS) {
    MS_LOG(ERROR) << "malloc device buffer failed";
    return FAILED;
  }
  det_param_buf_float_ = reinterpret_cast<float *>(bufPtr);
  det_param_buf_float_[NMS_THR] = dpico_config_param_extractor_.GetNmsThreshold();
  det_param_buf_float_[SCORE_THR] = dpico_config_param_extractor_.GetScoreThreshold();
  det_param_buf_float_[MIN_HEIGHT] = dpico_config_param_extractor_.GetMinHeight();
  det_param_buf_float_[MIN_WIDTH] = dpico_config_param_extractor_.GetMinWidth();

  // det para is 4 * sizeof(float) = 16 = default stride
  svp_acl_data_buffer *inputData = svp_acl_create_data_buffer(bufPtr, bufferSize, bufferSize);
  if (inputData == nullptr) {
    (void)svp_acl_rt_free(bufPtr);
    MS_LOG(ERROR) << "can't create data buffer, create input failed";
    return FAILED;
  }

  ret = svp_acl_mdl_add_dataset_buffer(input_dataset_, inputData);
  if (ret != SVP_ACL_SUCCESS) {
    MS_LOG(ERROR) << "add input dataset buffer failed";
    (void)svp_acl_rt_free(bufPtr);
    (void)svp_acl_destroy_data_buffer(inputData);
    inputData = nullptr;
    return FAILED;
  }
  return SUCCESS;
}

namespace {
const auto kFloat32 = DataType::kNumberTypeFloat32;
const auto kInt8 = DataType::kNumberTypeInt8;
const auto kUInt8 = DataType::kNumberTypeUInt8;
}  // namespace
REGISTER_CUSTOM_KERNEL(CPU, DPICO, kFloat32, DPICO, CustomCreateKernel)
REGISTER_CUSTOM_KERNEL(CPU, DPICO, kInt8, DPICO, CustomCreateKernel)
REGISTER_CUSTOM_KERNEL(CPU, DPICO, kUInt8, DPICO, CustomCreateKernel)
}  // namespace lite
}  // namespace mindspore
