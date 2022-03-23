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
#include "src/common/log.h"
#include "src/common/utils.h"
#include "src/common/common.h"
namespace mindspore {
namespace {
const int kNumInitBatch = 2000;
}
void ModelWorker::Run(int node_id, const std::shared_ptr<PredictTaskQueue> &predict_task_queue) {
  while (!predict_task_queue->IsPredictTaskDone()) {
    auto task = predict_task_queue->GetPredictTask(node_id);
    if (task == nullptr) {
      break;
    }
    auto inputs = task->inputs;
    auto *outputs = task->outputs;
    auto before = task->before;
    auto after = task->after;
    auto status = Predict(*inputs, outputs, before, after);
    if (status != kSuccess) {
      MS_LOG(ERROR) << "model predict failed.";
      task->ready = true;
      predict_task_queue->ActiveTask();
      continue;
    }
    if (need_copy_output_) {
      std::vector<MSTensor> new_outputs;
      auto output_size = outputs->size();
      for (size_t i = 0; i < output_size; i++) {
        auto copy_tensor =
          mindspore::MSTensor::CreateTensor(outputs->at(i).Name(), outputs->at(i).DataType(), outputs->at(i).Shape(),
                                            outputs->at(i).MutableData(), outputs->at(i).DataSize());
        if (copy_tensor == nullptr) {
          MS_LOG(ERROR) << "model thread copy output tensor failed.";
          task->ready = true;
          predict_task_queue->ActiveTask();
          continue;
        }
        new_outputs.push_back(*copy_tensor);
        delete copy_tensor;
      }
      outputs->clear();
      outputs->insert(outputs->end(), new_outputs.begin(), new_outputs.end());
    }
    task->ready = true;
    predict_task_queue->ActiveTask();
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
    auto data = malloc(input.DataSize());
    input.SetData(data);
  }
  std::vector<MSTensor> out;
  status = model_->Predict(inputs, &out);
  if (status != kSuccess) {
    MS_LOG(ERROR) << "init resize failed. ret=" << status;
    return kLiteError;
  }
  return kSuccess;
}

Status ModelWorker::Init(const char *model_buf, size_t size, const std::shared_ptr<Context> &model_context,
                         int node_id) {
  model_ = std::make_shared<Model>();
  mindspore::ModelType model_type = kMindIR_Lite;
  if (node_id != -1) {
    model_->UpdateConfig(lite::kConfigServerInference, {lite::kConfigNUMANodeId, std::to_string(node_id)});
  }
  auto status = model_->Build(model_buf, size, model_type, model_context);
  if (status != kSuccess) {
    MS_LOG(ERROR) << "model build failed in ModelPool Init";
    return status;
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

std::vector<MSTensor> ModelWorker::GetInputs() {
  if (model_ == nullptr) {
    MS_LOG(ERROR) << "model is nullptr in model worker.";
    return {};
  }
  auto inputs = model_->GetInputs();
  return inputs;
}

std::vector<MSTensor> ModelWorker::GetOutputs() {
  if (model_ == nullptr) {
    MS_LOG(ERROR) << "model is nullptr in model worker.";
    return {};
  }
  auto outputs = model_->GetOutputs();
  return outputs;
}

std::pair<std::vector<std::vector<int64_t>>, bool> ModelWorker::GetModelResize(
  const std::vector<MSTensor> &model_inputs, const std::vector<MSTensor> &inputs) {
  std::unique_lock<std::mutex> model_lock(mtx_model_);
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

Status ModelWorker::Predict(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs,
                            const MSKernelCallBack &before, const MSKernelCallBack &after) {
  auto model_input = model_->GetInputs();
  if (model_input.size() != inputs.size()) {
    MS_LOG(ERROR) << "model input size is: " << model_input.size() << ", but get input size is: " << inputs.size();
    return kLiteError;
  }
  auto resize_pair = GetModelResize(model_input, inputs);
  if (resize_pair.second) {
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
  auto status = model_->Predict(inputs, &model_output, before, after);
  if (status != kSuccess) {
    MS_LOG(ERROR) << "model predict failed.";
    return status;
  }
  if (need_copy_output_) {
    outputs->clear();
    outputs->insert(outputs->end(), model_output.begin(), model_output.end());
  } else {
    model_output = model_->GetOutputs();
    for (size_t i = 0; i < outputs->size(); i++) {
      outputs->at(i).SetShape(model_output[i].Shape());
      model_output[i].SetData(nullptr);
      model_output[i].SetAllocator(nullptr);
    }
  }
  return kSuccess;
}
}  // namespace mindspore
