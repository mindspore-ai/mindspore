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
#include "src/runtime/cxx_api/model_pool/model_worker.h"
#include "src/common/log_adapter.h"
#include "src/runtime/numa_adapter.h"
#include "src/common/common.h"
#include "nnacl/op_base.h"
namespace mindspore {
bool ModelWorker::IsAvailable() {
  bool expected = true;
  return available_.compare_exchange_strong(expected, false);
}

void ModelWorker::WaitCreateWorkerDone() {
  std::unique_lock<std::mutex> create_work_lock(create_work_done_mutex_);
  while (!create_work_done_) {
    create_work_done_condition_.wait(create_work_lock);
  }
  return;
}

void ModelWorker::CreateThreadWorker(const char *model_buf, size_t size,
                                     const std::shared_ptr<WorkerConfig> &worker_config,
                                     const std::shared_ptr<PredictTaskQueue> &predict_task_queue,
                                     bool *create_success) {
  worker_config_ = worker_config;
  MS_LOG(DEBUG) << "worker bind core id list: " << worker_config_->context->GetThreadAffinityCoreList();
  MS_LOG(DEBUG) << "worker thread num: " << worker_config_->context->GetThreadNum();
  predict_task_queue_ = predict_task_queue;
  numa::NUMAAdapter::GetInstance()->Bind(worker_config_->numa_id);
  auto status = Init(model_buf, size);
  if (status != kSuccess) {
    MS_LOG(ERROR) << "init failed in model worker.";
    {
      std::unique_lock<std::mutex> create_work_lock(create_work_done_mutex_);
      *create_success = false;
      create_work_done_ = true;
    }
    create_work_done_condition_.notify_one();
  }
  Run();
}

void ModelWorker::Run() {
  auto numa_node_id = worker_config_->numa_id;
  int task_queue_id = numa_node_id != -1 ? numa_node_id : 0;
  {
    // The scope of the lock is only for this variable
    std::unique_lock<std::mutex> create_work_lock(create_work_done_mutex_);
    create_work_done_ = true;
  }
  create_work_done_condition_.notify_one();
  MS_LOG(INFO) << "model worker is initialized.";
  while (!predict_task_queue_->IsPredictTaskDone()) {
    auto task = predict_task_queue_->GetPredictTask(task_queue_id, this);
    if (task == nullptr) {
      MS_LOG(DEBUG) << "task queue is empty, wait task ...";
      available_ = true;
      continue;
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
      predict_task_queue_->ActiveTask(task);
      continue;
    }
    task->ready = true;
    predict_task_queue_->ActiveTask(task);
  }
  MS_LOG(INFO) << "task queue all tasks completed.";
}

Status ModelWorker::Init(const char *model_buf, size_t size) {
  MS_CHECK_TRUE_MSG(model_buf != nullptr, kLiteError, "model_buf is nullptr in model worker.");
  model_ = std::make_shared<Model>();
  if (model_ == nullptr) {
    MS_LOG(ERROR) << "model is nullptr.";
    return kLiteNullptr;
  }
  mindspore::ModelType model_type = kMindIR_Lite;
  if (!worker_config_->config_path.empty()) {
    auto status = model_->LoadConfig(worker_config_->config_path);
    if (status != kSuccess) {
      MS_LOG(ERROR) << "model load config failed.";
      return kLiteError;
    }
  }
  for (auto &section : worker_config_->config_info) {
    for (auto &config : section.second) {
      auto status = model_->UpdateConfig(section.first, std::make_pair(config.first, config.second));
      if (status != kSuccess) {
        MS_LOG(ERROR) << "Update Config failed, status=" << status;
        return status;
      }
    }
  }
  MS_LOG(INFO) << "ms model init.";
  auto status = model_->Build(model_buf, size, model_type, worker_config_->context);
  if (status != kSuccess) {
    MS_LOG(ERROR) << "model build failed in ModelPool Init";
    return status;
  }
  MS_LOG(INFO) << "ms model init done.";
  origin_worker_inputs_ = model_->GetInputs();
  origin_worker_outputs_ = model_->GetOutputs();
  if (origin_worker_outputs_.empty() || origin_worker_outputs_.empty()) {
    MS_LOG(ERROR) << "model worker get empty input/output.";
    return kLiteError;
  }
  return kSuccess;
}

Status ModelWorker::UpdateConfig(const std::string &section, const std::pair<std::string, std::string> &config) {
  std::lock_guard<std::mutex> worker_lock(mtx_worker_);
  MS_LOG(DEBUG) << "UpdateConfig now.";
  return model_->UpdateConfig(section, config);
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
    available_ = true;
    return kLiteError;
  }
  auto resize_pair = GetModelResize(model_input, inputs);
  if (resize_pair.second) {
    // model need resize
    auto dims = resize_pair.first;
    auto status = model_->Resize(model_->GetInputs(), dims);
    if (status != kSuccess) {
      MS_LOG(ERROR) << "model pool resize failed.";
      available_ = true;
      return kLiteError;
    }
  }
  bool need_copy_output = true;
  auto model_output = model_->GetOutputs();
  for (size_t i = 0; i < outputs->size(); i++) {
    if (outputs->at(i).Data() != nullptr) {
      /* user set graph-output-tensor from outside */
      model_output[i].SetData(outputs->at(i).MutableData());
      model_output[i].SetAllocator(nullptr);
      need_copy_output = false;
    }
  }
  for (size_t i = 0; i < inputs.size(); i++) {
    model_input[i].SetData(const_cast<MSTensor &>(inputs[i]).MutableData());
    model_input[i].SetShape(inputs[i].Shape());
  }
  auto status = model_->Predict(model_input, &model_output, before, after);
  if (status != kSuccess) {
    MS_LOG(ERROR) << "model predict failed.";
    available_ = true;
    return status;
  }
  for (size_t i = 0; i < model_input.size(); i++) {
    model_input[i].SetData(nullptr);
  }
  if (need_copy_output) {
    status = CopyOutputTensor(model_output, outputs);
    if (status != kSuccess) {
      available_ = true;
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
