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

#include "src/cxx_api/model_pool/predict_task_queue.h"
namespace mindspore {
PredictTaskQueue::~PredictTaskQueue() {
  predict_task_done_ = true;
  task_push_cond_.notify_all();
}

void PredictTaskQueue::SetTaskQueueNum(int num) {
  predict_task_.resize(num);
  waite_worker_num_.resize(num, 0);
}

void PredictTaskQueue::WaitUntilPredictActive(const std::shared_ptr<PredictTask> &task) {
  std::unique_lock<std::mutex> result_lock(mtx_predict_task_);
  while (!task->ready) {
    task_pop_cond_.wait(result_lock);
  }
  return;
}

void PredictTaskQueue::ActiveTask() { task_pop_cond_.notify_all(); }

PredictTaskQueue *PredictTaskQueue::GetInstance() {
  static PredictTaskQueue instance;
  return &instance;
}

void PredictTaskQueue::PushPredictTask(std::shared_ptr<PredictTask> task, int node_id) {
  std::unique_lock<std::mutex> task_lock(mtx_predict_task_);
  predict_task_.at(node_id).push(task);
  task_push_cond_.notify_all();
}

std::shared_ptr<PredictTask> PredictTaskQueue::GetPredictTask(int node_id) {
  std::unique_lock<std::mutex> task_lock(mtx_predict_task_);
  while (predict_task_.at(node_id).empty() && !predict_task_done_) {
    task_push_cond_.wait(task_lock);
  }
  if (predict_task_done_) {
    return nullptr;
  }
  auto predict_task = predict_task_.at(node_id).front();
  predict_task_.at(node_id).pop();
  return predict_task;
}

int PredictTaskQueue::GetTaskNum(int node_id) {
  std::unique_lock<std::mutex> task_lock(mtx_predict_task_);
  return predict_task_.at(node_id).size();
}
}  // namespace mindspore
