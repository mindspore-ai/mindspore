/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "src/cxx_api/model_pool/model_worker.h"
#include "src/common/log_adapter.h"
#include "src/runtime/numa_adapter.h"
#include "src/common/common.h"
#include "nnacl/op_base.h"
namespace mindspore {
namespace {
const int kNumInitBatch = 2000;
}
bool ModelWorker::IsAvailable() {
  bool expected = true;
  return available_.compare_exchange_strong(expected, false);
}

void ModelWorker::Run(int node_id, const std::shared_ptr<PredictTaskQueue> &predict_task_queue) {
  predict_task_queue_ = predict_task_queue;
  while (!predict_task_queue->IsPredictTaskDone()) {
    auto task = predict_task_queue->GetPredictTask(node_id, this);
    if (task == nullptr) {
      break;
    }
    available_ = false;
    auto inputs = task->inputs;
    auto *outputs = task->outputs;
    auto before = task->before;
    auto after = task->after;
    auto status = Predict(*inputs, outputs, before, after);
    if (status != kSuccess) {
      MS_LOG(ERROR) << "model predict failed.";
      task->ready = true;
      predict_task_queue->ActiveTask(task);
      continue;
    }
    task->ready = true;
    predict_task_queue->ActiveTask(task);
  }
}

Status ModelWorker::ResizeInit() {
  auto inputs = model_->GetInputs();
  std::vector<std::vector<int64_t>> new_input_shape;
  for (size_t input_idx = 0; input_idx < inputs.size(); input_idx++) {
    new_input_shape.push_back(inputs[input_idx].Shape());
    if (new_input_shape[input_idx][0] == -1) {
      // only support resize for batch dim
      new_input_shape[input_idx][0] = kNumInitBatch;
    } else {
      // If the batch dimension is not -1, no resize processing is performed
      return kSuccess;
    }
  }
  auto status = model_->Resize(inputs, new_input_shape);
  if (status != kSuccess) {
    MS_LOG(ERROR) << "model resize failed in init. ret=" << status;
    return kLiteError;
  }
  inputs = model_->GetInputs();
  for (auto &input : inputs) {
    input.MutableData();
  }
  std::vector<MSTensor> out;
  status = model_->Predict(inputs, &out);
  if (status != kSuccess) {
    MS_LOG(ERROR) << "init resize failed. ret=" << status;
    return kLiteError;
  }
  return kSuccess;
}

Status ModelWorker::Init(const char *model_buf, size_t size, const std::shared_ptr<Context> &model_context) {
  MS_CHECK_TRUE_MSG(model_buf != nullptr, kLiteError, "model_buf is nullptr in model worker.");
  MS_CHECK_TRUE_MSG(model_context != nullptr, kLiteError, "model_context is nullptr in model worker.");
  model_ = std::make_shared<Model>();
  if (model_ == nullptr) {
    MS_LOG(ERROR) << "model is nullptr.";
    return kLiteNullptr;
  }
  mindspore::ModelType model_type = kMindIR_Lite;
  auto status = model_->Build(model_buf, size, model_type, model_context);
  if (status != kSuccess) {
    MS_LOG(ERROR) << "model build failed in ModelPool Init";
    return status;
  }
  origin_worker_inputs_ = model_->GetInputs();
  origin_worker_outputs_ = model_->GetOutputs();
  if (origin_worker_outputs_.empty() || origin_worker_outputs_.empty()) {
    MS_LOG(ERROR) << "model worker get empty input/output.";
    return kLiteError;
  }
  if (need_init_resize_) {
    status = ResizeInit();
    if (status != kSuccess) {
      MS_LOG(ERROR) << "init resize failed. ret=" << status;
      return kLiteError;
    }
  }
  return kSuccess;
}

std::vector<MSTensor> ModelWorker::GetInputs() { return origin_worker_inputs_; }

std::vector<MSTensor> ModelWorker::GetOutputs() { return origin_worker_outputs_; }

std::pair<std::vector<std::vector<int64_t>>, bool> ModelWorker::GetModelResize(
  const std::vector<MSTensor> &model_inputs, const std::vector<MSTensor> &inputs) {
  std::vector<std::vector<int64_t>> dims;
  bool need_resize = false;
  for (size_t i = 0; i < model_inputs.size(); i++) {
    for (size_t j = 0; j < model_inputs[i].Shape().size(); j++) {
      if (model_inputs[i].Shape()[j] != inputs[i].Shape()[j]) {
        need_resize = true;
      }
    }
    dims.push_back(inputs[i].Shape());
  }
  return std::make_pair(dims, need_resize);
}

Status ModelWorker::CopyOutputTensor(std::vector<MSTensor> model_outputs, std::vector<MSTensor> *user_outputs) {
  user_outputs->clear();
  user_outputs->insert(user_outputs->end(), model_outputs.begin(), model_outputs.end());
  std::vector<MSTensor> new_outputs;
  auto output_size = user_outputs->size();
  for (size_t i = 0; i < output_size; i++) {
    auto copy_tensor = mindspore::MSTensor::CreateTensor(user_outputs->at(i).Name(), user_outputs->at(i).DataType(),
                                                         user_outputs->at(i).Shape(), user_outputs->at(i).MutableData(),
                                                         user_outputs->at(i).DataSize());
    if (copy_tensor == nullptr) {
      MS_LOG(ERROR) << "model thread copy output tensor failed.";
      return kLiteError;
    }
    new_outputs.push_back(*copy_tensor);
    delete copy_tensor;
  }
  user_outputs->clear();
  user_outputs->insert(user_outputs->end(), new_outputs.begin(), new_outputs.end());
  return kSuccess;
}

Status ModelWorker::Predict(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs,
                            const MSKernelCallBack &before, const MSKernelCallBack &after) {
  std::lock_guard<std::mutex> worker_lock(mtx_worker_);
  available_ = false;
  auto model_input = model_->GetInputs();
  if (model_input.size() != inputs.size()) {
    MS_LOG(ERROR) << "model input size is: " << model_input.size() << ", but get input size is: " << inputs.size();
    return kLiteError;
  }
  auto resize_pair = GetModelResize(model_input, inputs);
  if (resize_pair.second) {
    // model need resize
    auto dims = resize_pair.first;
    auto status = model_->Resize(model_->GetInputs(), dims);
    if (status != kSuccess) {
      MS_LOG(ERROR) << "model pool resize failed.";
      return kLiteError;
    }
  }
  auto model_output = model_->GetOutputs();
  for (size_t i = 0; i < outputs->size(); i++) {
    if (outputs->at(i).MutableData() != nullptr) {
      /* user set graph-output-tensor from outside */
      model_output[i].SetData(outputs->at(i).MutableData());
      model_output[i].SetAllocator(nullptr);
      need_copy_output_ = false;
    }
  }
  for (size_t i = 0; i < inputs.size(); i++) {
    model_input[i].SetData(const_cast<MSTensor &>(inputs[i]).MutableData());
    model_input[i].SetShape(inputs[i].Shape());
  }
  auto status = model_->Predict(model_input, &model_output, before, after);
  if (status != kSuccess) {
    MS_LOG(ERROR) << "model predict failed.";
    return status;
  }
  for (size_t i = 0; i < model_input.size(); i++) {
    model_input[i].SetData(nullptr);
  }
  if (need_copy_output_) {
    status = CopyOutputTensor(model_output, outputs);
    if (status != kSuccess) {
      return kLiteError;
    }
  } else {
    model_output = model_->GetOutputs();
    for (size_t i = 0; i < outputs->size(); i++) {
      outputs->at(i).SetShape(model_output[i].Shape());
      model_output[i].SetData(nullptr);
      model_output[i].SetAllocator(nullptr);
    }
  }
  available_ = true;
  predict_task_queue_->ActiveTaskQueue();
  return kSuccess;
}
}  // namespace mindspore
