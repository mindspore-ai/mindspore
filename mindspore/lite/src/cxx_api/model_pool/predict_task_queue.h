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
#ifndef MINDSPORE_LITE_SRC_CXX_API_MODEL_POOL_PREDICT_TASK_QUEUE_H_
#define MINDSPORE_LITE_SRC_CXX_API_MODEL_POOL_PREDICT_TASK_QUEUE_H_

#include <queue>
#include <mutex>
#include <memory>
#include <vector>
#include <condition_variable>
#include "include/api/types.h"
#include "include/api/status.h"
namespace mindspore {
struct PredictTask {
  PredictTask(const std::vector<MSTensor> *in, std::vector<MSTensor> *out, MSKernelCallBack before,
              MSKernelCallBack after, bool ready = false)
      : inputs(in), outputs(out), before(before), after(after), ready(ready) {}
  const std::vector<MSTensor> *inputs;
  std::vector<MSTensor> *outputs;
  MSKernelCallBack before;
  MSKernelCallBack after;
  bool ready;
};

class PredictTaskQueue {
 public:
  PredictTaskQueue() = default;
  ~PredictTaskQueue() = default;

  void PushPredictTask(std::shared_ptr<PredictTask> task, int node_id);
  void WaitUntilPredictActive(const std::shared_ptr<PredictTask> &task);
  std::shared_ptr<PredictTask> GetPredictTask(int node_id);
  void ActiveTask();
  int GetTaskNum(int node_id);
  void SetTaskQueueNum(int num);

  bool IsPredictTaskDone() const { return predict_task_done_; }
  void SetPredictTaskDone();
  int GetWaitModelNum(int node_id) const { return waite_worker_num_.at(node_id); }
  void DecreaseWaitModelNum(int num, int node_id) { waite_worker_num_.at(node_id) -= num; }
  void IncreaseWaitModelNum(int num, int node_id) { waite_worker_num_.at(node_id) += num; }

 private:
  std::vector<std::queue<std::shared_ptr<PredictTask>>> predict_task_;
  std::vector<int> waite_worker_num_;
  std::mutex mtx_predict_task_;
  std::condition_variable task_pop_cond_;
  std::condition_variable task_push_cond_;
  bool predict_task_done_ = false;
};
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_CXX_API_MODEL_POOL_PREDICT_TASK_QUEUE_H_
