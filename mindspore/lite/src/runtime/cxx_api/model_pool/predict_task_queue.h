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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_CXX_API_MODEL_POOL_PREDICT_TASK_QUEUE_H_
#define MINDSPORE_LITE_SRC_RUNTIME_CXX_API_MODEL_POOL_PREDICT_TASK_QUEUE_H_

#include <queue>
#include <mutex>
#include <memory>
#include <vector>
#include <condition_variable>
#include "include/api/types.h"
#include "include/api/status.h"
#include "src/runtime/cxx_api/model_pool/model_worker.h"
#include "thread/hqueue.h"
#ifndef USE_HQUEUE
#define USE_HQUEUE
#endif
namespace mindspore {
class ModelWorker;
struct PredictTask {
  PredictTask(const std::vector<MSTensor> *in = nullptr, std::vector<MSTensor> *out = nullptr,
              MSKernelCallBack before = nullptr, MSKernelCallBack after = nullptr, bool ready = false)
      : inputs(in), outputs(out), before(before), after(after), ready(ready) {}
  const std::vector<MSTensor> *inputs;
  std::vector<MSTensor> *outputs;
  MSKernelCallBack before;
  MSKernelCallBack after;
  std::atomic_bool ready;
  std::condition_variable task_done_condition;
  std::mutex task_done_mutex;
};

class PredictTaskQueue {
 public:
  PredictTaskQueue() = default;
  ~PredictTaskQueue();

  void PushPredictTask(PredictTask *task, int node_id);
  void WaitUntilPredictActive(PredictTask *task, int node_id);
  PredictTask *GetPredictTask(int node_id, ModelWorker *worker);
  void ActiveTask(PredictTask *task);
  void ActiveTaskQueue();
  Status InitTaskQueue(size_t num, size_t max_queue_size);

  bool IsPredictTaskDone() const { return predict_task_done_; }
  void SetPredictTaskDone();
  int GetWaitModelNum(int node_id) const { return idle_worker_num_[node_id]; }
  void DecreaseWaitModelNum(int num, int node_id) { idle_worker_num_[node_id] -= num; }
  void IncreaseWaitModelNum(int num, int node_id) { idle_worker_num_[node_id] += num; }

 private:
  // use an array to save predict tasks, different numa nodes correspond to different arrays
#ifdef USE_HQUEUE
  HQueue<PredictTask> *predict_task_;
#else
  std::queue<PredictTask *> *predict_task_;
#endif
  size_t task_queue_num_ = -1;
  std::atomic_int *idle_worker_num_;
  std::mutex mtx_predict_task_;
  std::condition_variable task_pop_cond_;
  std::condition_variable task_push_cond_;
  bool predict_task_done_ = false;
};
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_RUNTIME_CXX_API_MODEL_POOL_PREDICT_TASK_QUEUE_H_
