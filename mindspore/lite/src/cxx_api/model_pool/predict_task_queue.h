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
  static PredictTaskQueue *GetInstance();
  ~PredictTaskQueue();

  void PushPredictTask(std::shared_ptr<PredictTask> task);
  void WaitUntilPredictActive(std::shared_ptr<PredictTask> task);
  std::shared_ptr<PredictTask> GetPredictTask();
  void ActiveTask();
  bool IsPredictTaskDone() { return predict_task_done_; }
  int GetTaskNum();
  int GetWaitModelNum() { return waite_model_num_; }

 private:
  PredictTaskQueue() = default;
  std::queue<std::shared_ptr<PredictTask>> predict_task_;
  int waite_model_num_ = 0;

  std::mutex mtx_predict_task_;
  std::condition_variable task_pop_cond_;
  std::condition_variable task_push_cond_;
  bool predict_task_done_ = false;
};
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_CXX_API_MODEL_POOL_PREDICT_TASK_QUEUE_H_
