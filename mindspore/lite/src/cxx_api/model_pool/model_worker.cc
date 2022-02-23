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
namespace mindspore {
void ModelThread::Run() {
  while (!PredictTaskQueue::GetInstance()->IsPredictTaskDone()) {
    auto task = PredictTaskQueue::GetInstance()->GetPredictTask();
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
      PredictTaskQueue::GetInstance()->ActiveTask();
      continue;
    }
    if (is_copy_output_) {
      std::vector<MSTensor> new_outputs;
      auto output_size = outputs->size();
      for (size_t i = 0; i < output_size; i++) {
        auto copy_tensor =
          mindspore::MSTensor::CreateTensor(outputs->at(i).Name(), outputs->at(i).DataType(), outputs->at(i).Shape(),
                                            outputs->at(i).MutableData(), outputs->at(i).DataSize());
        if (copy_tensor == nullptr) {
          MS_LOG(ERROR) << "model thread copy output tensor failed.";
          task->ready = true;
          PredictTaskQueue::GetInstance()->ActiveTask();
          continue;
        }
        new_outputs.push_back(*copy_tensor);
        delete copy_tensor;
      }
      outputs->clear();
      outputs->insert(outputs->end(), new_outputs.begin(), new_outputs.end());
    }
    task->ready = true;
    PredictTaskQueue::GetInstance()->ActiveTask();
  }
}

Status ModelThread::Init(const char *model_buf, size_t size, const std::shared_ptr<Context> &model_context,
                         const Key &dec_key, const std::string &dec_mode) {
  model_ = std::make_shared<Model>();
  mindspore::ModelType model_type = kMindIR;
  auto status = model_->Build(model_buf, size, model_type, model_context, dec_key, dec_mode);
  if (status != kSuccess) {
    MS_LOG(ERROR) << "model build failed in ModelPool Init";
    return status;
  }
  return kSuccess;
}

std::vector<MSTensor> ModelThread::GetInputs() {
  if (model_ == nullptr) {
    MS_LOG(ERROR) << "model is nullptr in ModelThread.";
    return {};
  }
  auto inputs = model_->GetInputs();
  return inputs;
}

std::vector<MSTensor> ModelThread::GetOutputs() {
  if (model_ == nullptr) {
    MS_LOG(ERROR) << "model is nullptr in ModelThread.";
    return {};
  }
  auto outputs = model_->GetOutputs();
  return outputs;
}

std::pair<std::vector<std::vector<int64_t>>, bool> ModelThread::GetModelResize(
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

Status ModelThread::Predict(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs,
                            const MSKernelCallBack &before, const MSKernelCallBack &after) {
  // model
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
      is_copy_output_ = false;
    }
  }
  auto status = model_->Predict(inputs, &model_output, before, after);
  if (status != kSuccess) {
    MS_LOG(ERROR) << "model predict failed.";
    return status;
  }
  if (is_copy_output_) {
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
