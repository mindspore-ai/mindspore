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

#include "src/runtime/cxx_api/model_pool/predict_task_queue.h"
#include "src/common/log_adapter.h"
namespace mindspore {
PredictTaskQueue::~PredictTaskQueue() {
  if (predict_task_ != nullptr) {
#ifdef USE_HQUEUE
    for (size_t i = 0; i < task_queue_num_; i++) {
      predict_task_[i].Clean();
    }
#endif
    delete[] predict_task_;
    predict_task_ = nullptr;
  }
  if (idle_worker_num_ != nullptr) {
    delete[] idle_worker_num_;
    idle_worker_num_ = nullptr;
  }
}

void PredictTaskQueue::SetPredictTaskDone() {
  std::unique_lock<std::mutex> task_lock(mtx_predict_task_);
  predict_task_done_ = true;
  task_push_cond_.notify_all();
}

Status PredictTaskQueue::InitTaskQueue(size_t num, size_t max_queue_size) {
  if (num == 0) {
    MS_LOG(ERROR) << "task queue size should greater than 0";
    return kLiteError;
  }
#ifdef USE_HQUEUE
  task_queue_num_ = num;
  predict_task_ = new (std::nothrow) HQueue<PredictTask>[num]();
  if (predict_task_ == nullptr) {
    MS_LOG(ERROR) << "new predict task failed.";
    return kLiteNullptr;
  }
  for (size_t i = 0; i < num; i++) {
    if (!predict_task_[i].Init(max_queue_size + 1)) {
      MS_LOG(ERROR) << "HQueue init failed.";
      return kLiteError;
    }
  }
#else
  predict_task_ = new (std::nothrow) std::queue<PredictTask *>[num]();
  if (predict_task_ == nullptr) {
    MS_LOG(ERROR) << "new predict task failed.";
    return kLiteNullptr;
  }
#endif
  idle_worker_num_ = new (std::nothrow) std::atomic_int[num]();
  if (idle_worker_num_ == nullptr) {
    MS_LOG(ERROR) << "new wait worker num list failed.";
    return kLiteError;
  }
  return kSuccess;
}

void PredictTaskQueue::WaitUntilPredictActive(PredictTask *task, int node_id) {
  std::unique_lock<std::mutex> result_lock(task->task_done_mutex);
  while (!task->ready) {
    task->task_done_condition.wait(result_lock);
  }
  task->ready = false;
  idle_worker_num_[node_id] += 1;
  return;
}

void PredictTaskQueue::ActiveTask(PredictTask *task) {
  std::unique_lock<std::mutex> result_lock(task->task_done_mutex);
  task->task_done_condition.notify_one();
}

void PredictTaskQueue::ActiveTaskQueue() {
  std::unique_lock<std::mutex> task_lock(mtx_predict_task_);
  task_push_cond_.notify_all();
}

void PredictTaskQueue::PushPredictTask(PredictTask *task, int node_id) {
  idle_worker_num_[node_id] -= 1;
#ifdef USE_HQUEUE
  while (!predict_task_[node_id].Enqueue(task)) {
  }
  std::unique_lock<std::mutex> task_lock(mtx_predict_task_);
#else
  std::unique_lock<std::mutex> task_lock(mtx_predict_task_);
  predict_task_[node_id].push(task);
#endif
  task_push_cond_.notify_all();
}

PredictTask *PredictTaskQueue::GetPredictTask(int node_id, ModelWorker *worker) {
#ifdef USE_HQUEUE
  if (!predict_task_[node_id].Empty() && worker->IsAvailable()) {
    return predict_task_[node_id].Dequeue();
  } else {
    std::unique_lock<std::mutex> task_lock(mtx_predict_task_);
    while ((predict_task_[node_id].Empty() || (!worker->IsAvailable())) && (!predict_task_done_)) {
      task_push_cond_.wait(task_lock);
    }
  }
  return predict_task_[node_id].Dequeue();
#else
  std::unique_lock<std::mutex> task_lock(mtx_predict_task_);
  while ((predict_task_[node_id].empty() || (!worker->IsAvailable())) && (!predict_task_done_)) {
    task_push_cond_.wait(task_lock);
  }
  if (predict_task_done_) {
    return nullptr;
  }
  auto predict_task = predict_task_[node_id].front();
  predict_task_[node_id].pop();
  return predict_task;
#endif
}
}  // namespace mindspore
