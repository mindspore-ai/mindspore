/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "src/mslite_model_state.h"
#include <unistd.h>
#include <algorithm>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include "triton/backend/backend_input_collector.h"
#include "triton/backend/backend_output_responder.h"

namespace triton {
namespace backend {
namespace mslite {
TRITONSERVER_Error *ModelState::Create(TRITONBACKEND_Model *triton_model, ModelState **state) {
  try {
    *state = new ModelState(triton_model);
  } catch (const BackendModelException &ex) {
    RETURN_ERROR_IF_TRUE(ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
                         std::string("unexpected nullptr in BackendModelException"));
    RETURN_IF_ERROR(ex.err_);
  }

  RETURN_IF_ERROR((*state)->ParseModelConfig());
  RETURN_IF_ERROR((*state)->ParseModelParameterConfig());
  RETURN_IF_ERROR((*state)->InitMSContext());
  return nullptr;  // success
}

TRITONSERVER_Error *ModelState::InitMSContext() {
  // Init ms context.
  ms_context_ = std::make_shared<mindspore::Context>();
  RETURN_ERROR_IF_TRUE(ms_context_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
                       std::string("New mindspore-lite context failed."));
  auto &device_list = ms_context_->MutableDeviceInfo();

  if (device_type_ == "ascend") {
    auto ascend_device_info = std::make_shared<mindspore::AscendDeviceInfo>();
    RETURN_ERROR_IF_TRUE(ascend_device_info == nullptr, TRITONSERVER_ERROR_INTERNAL,
                         std::string("New AscendDeviceInfo failed for mindspore-lite context."));
    ascend_device_info->SetDeviceID(device_id_);
    device_list.push_back(ascend_device_info);
  }
  auto device_info = std::make_shared<mindspore::CPUDeviceInfo>();
  RETURN_ERROR_IF_TRUE(device_info == nullptr, TRITONSERVER_ERROR_INTERNAL,
                       std::string("New CPUDeviceInfo failed for mindspore-lite context."));
  device_list.push_back(device_info);
  return nullptr;
}

std::shared_ptr<mindspore::Model> ModelState::BuildMSModel() {
  auto ms_model = std::make_shared<mindspore::Model>();
  if (ms_model == nullptr) {
    LOG_MESSAGE(TRITONSERVER_LOG_ERROR, "New mslite model failed.");
    return nullptr;
  }
  auto model_path = JoinPath({this->RepositoryPath(), std::to_string(this->Version())});
  auto model_name = this->Name() + "." + model_type_;
  auto model_file = JoinPath({model_path, model_name});
  auto model_type = model_type_ == "mindir" ? mindspore::kMindIR : mindspore::kMindIR_Lite;
  LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string("Begin to init mslite model file: ") + model_file).c_str());
  auto build_ret = ms_model->Build(model_file, model_type, ms_context_);
  if (build_ret != mindspore::kSuccess) {
    LOG_MESSAGE(TRITONSERVER_LOG_ERROR, "Build mslite model failed.");
    return nullptr;
  }
  return ms_model;
}

TRITONSERVER_Error *ModelState::ParseModelParameterConfig() {
  // parse dynamic batching config. the max_batch_size is parsed in the inner method ParseModelConfig.
  // dynamic batch may means that the max batch is not zero.
  RETURN_IF_ERROR(this->SupportsFirstDimBatching(&support_dynamic_batch_));

  // for ascend, use the dynamic_batching to judge the real state of supporting dynamic batch.
  common::TritonJson::Value dynamic_batching;
  support_dynamic_batch_ &= model_config_.Find("dynamic_batching", &dynamic_batching);
  if (support_dynamic_batch_) {
    RETURN_IF_ERROR(backend::ParseShape(dynamic_batching, "preferred_batch_size", &preferred_batch_size_));
    std::sort(preferred_batch_size_.begin(), preferred_batch_size_.end());
    RETURN_ERROR_IF_FALSE(std::all_of(preferred_batch_size_.begin(), preferred_batch_size_.end(),
                                      [&](int64_t dim) { return dim <= max_batch_size_; }),
                          TRITONSERVER_ERROR_INVALID_ARG,
                          std::string("the preferred batch size should not be larger than the max batch size."));
  }

  // parse parameters.
  common::TritonJson::Value parameters;
  if (model_config_.Find("parameters", &parameters)) {
    (void)GetParameterValue(parameters, "model_type", &model_type_);
    (void)GetParameterValue(parameters, "device_type", &device_type_);
    std::string device_id;
    (void)GetParameterValue(parameters, "device_id", &device_id);
    if (!device_id.empty()) {
      try {
        device_id_ = std::stoi(device_id);
      } catch (const BackendModelInstanceException &ex) {
        RETURN_ERROR_IF_TRUE(ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
                             std::string("The device_id is not valid.") + device_id);
        RETURN_IF_ERROR(ex.err_);
      }
    }
  }
  return nullptr;  // success
}

TRITONSERVER_Error *ModelInstanceState::Create(ModelState *model_state,
                                               TRITONBACKEND_ModelInstance *triton_model_instance,
                                               ModelInstanceState **state) {
  try {
    *state = new ModelInstanceState(model_state, triton_model_instance);
  } catch (const BackendModelInstanceException &ex) {
    RETURN_ERROR_IF_TRUE(ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
                         std::string("unexpected nullptr in BackendModelInstanceException"));
    RETURN_IF_ERROR(ex.err_);
  }

  (*state)->ms_model_ = model_state->BuildMSModel();
  RETURN_ERROR_IF_TRUE((*state)->ms_model_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
                       std::string("Init mslite model failed."));
  return nullptr;  // success
}

TRITONSERVER_Error *ModelInstanceState::ProcessInputs(TRITONBACKEND_Request **requests, const uint32_t request_count,
                                                      std::vector<TRITONBACKEND_Response *> *responses) {
  // To instruct ProcessTensor to "gather" the entire batch of input
  // tensors into a single contiguous buffer in CPU memory, set the
  // "allowed input types" to be the CPU ones (see tritonserver.h in
  // the triton-inference-server/core repo for allowed memory types).
  BackendInputCollector collector(requests, request_count, responses, model_state_->TritonMemoryManager(),
                                  model_state_->EnablePinnedInput(), nullptr);

  std::vector<std::pair<TRITONSERVER_MemoryType, int64_t>> allowed_input_types = {{TRITONSERVER_MEMORY_CPU_PINNED, 0},
                                                                                  {TRITONSERVER_MEMORY_CPU, 0}};

  uint32_t input_count = 0;
  RETURN_IF_ERROR(TRITONBACKEND_RequestInputCount(requests[0], &input_count));

  auto model_inputs = ms_model_->GetInputs();
  RETURN_ERROR_IF_FALSE(model_inputs.size() == input_count, TRITONSERVER_ERROR_INVALID_ARG,
                        std::string("The input count is not equal to model inputs: ") +
                          std::to_string(model_inputs.size()) + std::string(", while the input count of request is: ") +
                          std::to_string(input_count));

  inputs_.clear();
  for (uint32_t idx = 0; idx < input_count; idx++) {
    TRITONBACKEND_Input *input;
    RETURN_IF_ERROR(TRITONBACKEND_RequestInputByIndex(requests[0], idx, &input));
    const char *input_name;
    TRITONSERVER_DataType input_datatype;
    const int64_t *input_shape;
    uint32_t input_dims_count;
    batched_size_ = 0;
    for (uint32_t r = 0; r < request_count; r++) {
      RETURN_IF_ERROR(TRITONBACKEND_InputProperties(input, &input_name, &input_datatype, &input_shape,
                                                    &input_dims_count, nullptr, nullptr));
      batched_size_ += *input_shape;
    }
    RETURN_ERROR_IF_TRUE(model_state_->MaxBatchSize() > 0 && batched_size_ > model_state_->MaxBatchSize(),
                         TRITONSERVER_ERROR_INVALID_ARG,
                         std::string("The input batch size is larger than the max batch size."));
    auto data_type = GetMSDataTypeFromTritonServerDataType(input_datatype);
    RETURN_ERROR_IF_FALSE(data_type == model_inputs.at(idx).DataType(), TRITONSERVER_ERROR_INVALID_ARG,
                          std::string("The input data type is not equal to model input."));

    std::vector<int64_t> batched_shape(input_dims_count);
    std::memcpy(batched_shape.data(), input_shape, input_dims_count * sizeof(int64_t));
    batched_shape.at(0) = batched_size_;
    if (support_dynamic_batch_) {
      auto pad_batch_itr = std::lower_bound(preferred_batch_size_.begin(), preferred_batch_size_.end(), batched_size_);
      batched_shape.at(0) = pad_batch_itr != preferred_batch_size_.end() ? *pad_batch_itr : max_batch_size_;
      LOG_MESSAGE(TRITONSERVER_LOG_INFO,
                  std::string("The batched size will be pad to " + std::to_string(batched_shape.at(0))).c_str());
      need_resize_ = model_inputs.at(idx).Shape() != batched_shape;
    }

    const char *input_buffer = nullptr;
    size_t input_buffer_byte_size;
    TRITONSERVER_MemoryType input_buffer_memory_type;
    int64_t input_buffer_memory_type_id;
    RETURN_IF_ERROR(collector.ProcessTensor(
      input_name, nullptr /* existing_buffer */, 0 /* existing_buffer_byte_size */, allowed_input_types, &input_buffer,
      &input_buffer_byte_size, &input_buffer_memory_type, &input_buffer_memory_type_id));
    RETURN_ERROR_IF_TRUE(input_buffer == nullptr || input_buffer_byte_size == 0, TRITONSERVER_ERROR_INTERNAL,
                         std::string("Process input tensor data failed."));

    auto input_tensor = mindspore::MSTensor(input_name, data_type, {}, nullptr, 0);
    input_tensor.SetShape(batched_shape);
    if (!support_dynamic_batch_ && input_tensor.DataSize() == input_buffer_byte_size) {
      // speed up with non-memcpy in static shape,
      // while the batched data from several requests may cause a host-to-device memcpy error.
      LOG_MESSAGE(TRITONSERVER_LOG_INFO, std::string("use the original data without copy.").c_str());
      input_tensor.SetData(reinterpret_cast<void *>(const_cast<char *>(input_buffer)), false);
    } else {
      LOG_MESSAGE(TRITONSERVER_LOG_INFO, std::string("malloc data because the data size is not equal").c_str());
      auto input_data = input_tensor.MutableData();
      auto data_size = input_tensor.DataSize();
      RETURN_ERROR_IF_TRUE(input_data == nullptr || input_buffer_byte_size > data_size, TRITONSERVER_ERROR_INTERNAL,
                           std::string("Process input tensor data failed."));

      std::memset(input_data, 0, input_tensor.DataSize());
      std::memcpy(input_data, input_buffer, input_buffer_byte_size);
    }
    inputs_.push_back(input_tensor);
  }
  collector.Finalize();
  LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string("Process inputs tensor finished.").c_str()));
  return nullptr;  // success
}

TRITONSERVER_Error *ModelInstanceState::ProcessOutputs(TRITONBACKEND_Request **requests, const uint32_t request_count,
                                                       std::vector<TRITONBACKEND_Response *> *responses) {
  BackendOutputResponder responder(requests, request_count, responses, model_state_->TritonMemoryManager(),
                                   model_state_->MaxBatchSize() > 1, model_state_->EnablePinnedOutput(), nullptr);

  for (auto output : outputs_) {
    auto shape = output.Shape();
    auto data_type = GetTritonServerDataTypeFromMSDataType(output.DataType());
    RETURN_ERROR_IF_TRUE(data_type == TRITONSERVER_TYPE_INVALID, TRITONSERVER_ERROR_INTERNAL,
                         std::string("The output data type is invalid."));
    auto data = output.MutableData();
    RETURN_ERROR_IF_TRUE(data == nullptr, TRITONSERVER_ERROR_INTERNAL, std::string("The output data is nullptr."));
    responder.ProcessTensor(output.Name(), TRITONSERVER_TYPE_FP32, shape, reinterpret_cast<char *>(data),
                            TRITONSERVER_MEMORY_CPU, 0);
  }
  responder.Finalize();
  return nullptr;  // success
}

void ModelInstanceState::ProcessRequests(TRITONBACKEND_Request **requests, const uint32_t request_count) {
  LOG_MESSAGE(TRITONSERVER_LOG_INFO,
              (std::string("Begin to process ") + std::to_string(request_count) + std::string(" requests.")).c_str());
  uint64_t exec_start_ns = 0;
  SET_TIMESTAMP(exec_start_ns);

  for (size_t i = 0; i < request_count; i++) {
    // If we get a nullptr request then something is badly wrong. Fail
    // and release all requests.
    if (requests[i] == nullptr) {
      RequestsRespondWithError(
        requests, request_count,
        TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL,
                              std::string("null request given to MSLite backend for '" + Name() + "'").c_str()));
      return;
    }
  }

  // At this point we accept ownership of 'requests', which means that
  // even if something goes wrong we must still return success from
  // this function. If something does go wrong in processing a
  // particular request then we send an error response just for the
  // specific request.
  std::vector<TRITONBACKEND_Response *> responses;
  responses.reserve(request_count);
  for (size_t i = 0; i < request_count; i++) {
    TRITONBACKEND_Response *response;
    auto err = TRITONBACKEND_ResponseNew(&response, requests[i]);
    if (err == nullptr) {
      responses.emplace_back(response);
    } else {
      responses.emplace_back(nullptr);
      LOG_MESSAGE(TRITONSERVER_LOG_ERROR, "Fail to create response");
      TRITONSERVER_ErrorDelete(err);
    }
  }

  RESPOND_ALL_AND_SET_NULL_IF_ERROR(responses, request_count, ProcessInputs(requests, request_count, &responses));

  if (need_resize_) {
    std::vector<std::vector<int64_t>> shapes;
    std::transform(inputs_.begin(), inputs_.end(), std::back_inserter(shapes),
                   [](const mindspore::MSTensor &input) { return input.Shape(); });
    std::stringstream oss;
    auto print_shape = [&oss](const std::vector<int64_t> &shape) {
      oss << "[";
      std::for_each(shape.begin(), shape.end(), [&oss](int64_t dim) { oss << (std::to_string(dim) + ", "); });
      oss << "], ";
    };
    oss << "The inputs needs to resize to [";
    std::for_each(shapes.begin(), shapes.end(), print_shape);
    oss << "]";
    LOG_MESSAGE(TRITONSERVER_LOG_INFO, oss.str().c_str());
    std::cout << oss.str() << std::endl;

    auto ret = ms_model_->Resize(ms_model_->GetInputs(), shapes);
    if (ret != mindspore::kSuccess) {
      RESPOND_ALL_AND_SET_NULL_IF_ERROR(
        responses, request_count, TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, "Fail to resize mslite model."));
      return;
    }
  }

  uint64_t compute_start_ns = 0;
  SET_TIMESTAMP(compute_start_ns);
  outputs_ = !outputs_.empty() ? outputs_ : ms_model_->GetOutputs();
  auto ret = ms_model_->Predict(inputs_, &outputs_);
  if (ret != mindspore::kSuccess) {
    RESPOND_ALL_AND_SET_NULL_IF_ERROR(
      responses, request_count, TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, "Fail to predict mslite model."));
    return;
  }

  uint64_t compute_end_ns = 0;
  SET_TIMESTAMP(compute_end_ns);
  LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string("Compute model ") + Name() + " cost " +
                                      std::to_string((compute_end_ns - compute_start_ns) / 1000) + " us")
                                       .c_str());

  RESPOND_ALL_AND_SET_NULL_IF_ERROR(responses, request_count, ProcessOutputs(requests, request_count, &responses));

  uint64_t exec_end_ns = 0;
  SET_TIMESTAMP(exec_end_ns);

  // Send all the responses that haven't already been sent because of
  // an earlier error. Note that the responses are not set to nullptr
  // here as we need that indication below to determine if the request
  // we successful or not.
  for (auto &response : responses) {
    if (response != nullptr) {
      LOG_IF_ERROR(TRITONBACKEND_ResponseSend(response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, nullptr),
                   "failed to send MSLite backend response");
    }
  }

  // Report statistics for each request.
  for (uint32_t r = 0; r < request_count; ++r) {
    auto &request = requests[r];
    LOG_IF_ERROR(TRITONBACKEND_ModelInstanceReportStatistics(TritonModelInstance(), request,
                                                             (responses[r] != nullptr) /* success */, exec_start_ns,
                                                             compute_start_ns, compute_end_ns, exec_end_ns),
                 "failed reporting request statistics");

    LOG_IF_ERROR(TRITONBACKEND_RequestRelease(request, TRITONSERVER_REQUEST_RELEASE_ALL), "failed releasing request");
  }

  // Report the entire batch statistics.
  LOG_IF_ERROR(TRITONBACKEND_ModelInstanceReportBatchStatistics(TritonModelInstance(), batched_size_, exec_start_ns,
                                                                compute_start_ns, compute_end_ns, exec_end_ns),
               "failed reporting batch request statistics");

  LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string("TRITONBACKEND_ModelExecute: model ") + Name() + " released " +
                                      std::to_string(request_count) + " requests")
                                       .c_str());

  LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string("Execute model ") + Name() + " cost " +
                                      std::to_string((exec_end_ns - exec_start_ns) / 1000) + " us")
                                       .c_str());
}
}  // namespace mslite
}  // namespace backend
}  // namespace triton
