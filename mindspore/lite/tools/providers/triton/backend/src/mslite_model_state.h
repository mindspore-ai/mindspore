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

#ifndef MINDSPORE_LITE_TOOLS_PROVIDERS_TRITON_BACKEND_SRC_MSLITE_MODEL_STATE_H_
#define MINDSPORE_LITE_TOOLS_PROVIDERS_TRITON_BACKEND_SRC_MSLITE_MODEL_STATE_H_

#include <memory>
#include <string>
#include <vector>
#include <future>
#include "src/mslite_utils.h"
#include "triton/backend/backend_model.h"
#include "triton/backend/backend_model_instance.h"
#include "triton/core/tritonserver.h"
#include "include/api/context.h"
#include "include/api/model.h"
#include "triton/backend/backend_input_collector.h"
#include "triton/backend/backend_output_responder.h"

namespace triton {
namespace backend {
namespace mslite {
// ModelState
//
// State associated with a model that is using this backend. An object
// of this class is createdBLSExecutor and associated with each
// TRITONBACKEND_Model.
//
class ModelState : public BackendModel {
 public:
  static TRITONSERVER_Error *Create(TRITONBACKEND_Model *triton_model, ModelState **state);
  virtual ~ModelState() = default;

  TRITONSERVER_Error *ParseModelParameterConfig();
  TRITONSERVER_Error *InitMSContext();
  // Init mslite model
  std::shared_ptr<mindspore::Model> BuildMSModel();

  std::vector<int64_t> PreferredBatchSize() { return preferred_batch_size_; }
  bool SupportDynamicBatch() { return support_dynamic_batch_; }

 private:
  explicit ModelState(TRITONBACKEND_Model *triton_model) : BackendModel(triton_model) {}

  std::shared_ptr<mindspore::Context> ms_context_ = nullptr;
  bool support_dynamic_batch_ = false;
  std::vector<int64_t> preferred_batch_size_;
  std::string model_type_ = "mindir";
  std::string device_type_ = "";
  int device_id_ = 0;
};

//
// ModelInstanceState
//
// State associated with a model instance. An object of this class is
// created and associated with each TRITONBACKEND_ModelInstance.
//
class ModelInstanceState : public BackendModelInstance {
 public:
  static TRITONSERVER_Error *Create(ModelState *model_state, TRITONBACKEND_ModelInstance *triton_model_instance,
                                    ModelInstanceState **state);
  virtual ~ModelInstanceState() = default;

  void ProcessRequests(TRITONBACKEND_Request **requests, const uint32_t request_count);

 private:
  ModelInstanceState(ModelState *model_state, TRITONBACKEND_ModelInstance *triton_model_instance)
      : BackendModelInstance(reinterpret_cast<BackendModel *>(model_state), triton_model_instance),
        model_state_(model_state) {
    max_batch_size_ = model_state->MaxBatchSize();
    preferred_batch_size_ = model_state->PreferredBatchSize();
    support_dynamic_batch_ = model_state->SupportDynamicBatch();
  }

  TRITONSERVER_Error *ProcessInputs(TRITONBACKEND_Request **requests, const uint32_t request_count,
                                    std::vector<TRITONBACKEND_Response *> *responses);

  TRITONSERVER_Error *ProcessOutputs(TRITONBACKEND_Request **requests, const uint32_t request_count,
                                     std::vector<TRITONBACKEND_Response *> *responses);

  ModelState *model_state_;
  std::shared_ptr<mindspore::Model> ms_model_;

  std::vector<mindspore::MSTensor> inputs_;
  std::vector<mindspore::MSTensor> outputs_;

  int64_t batched_size_ = 0;
  int64_t max_batch_size_ = 0;
  std::vector<int64_t> preferred_batch_size_;
  bool support_dynamic_batch_ = false;
  bool need_resize_ = false;
};
}  // namespace mslite
}  // namespace backend
}  // namespace triton
#endif  // MINDSPORE_LITE_TOOLS_PROVIDERS_TRITON_BACKEND_SRC_MSLITE_MODEL_STATE_H_
