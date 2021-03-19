/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_SRC_TRAIN_TRAIN_LOOP_H_
#define MINDSPORE_LITE_SRC_TRAIN_TRAIN_LOOP_H_
#include <vector>
#include <string>
#include <tuple>
#include <memory>
#include <unordered_map>
#include "include/errorcode.h"
#include "include/train/train_loop.h"
#include "include/train/metrics.h"
#include "include/datasets.h"
#include "include/iterator.h"
#include "src/common/log_adapter.h"

namespace mindspore {
namespace lite {

class TrainLoop : virtual public session::TrainLoop {
 public:
  explicit TrainLoop(session::TrainSession *session) : train_session_(session) {}

  session::TrainSession *train_session() override { return train_session_; }

  int Reset() override {
    epoch_ = 0;
    return RET_OK;
  }

  virtual ~TrainLoop();

  int Init(std::vector<mindspore::session::Metrics *> metrics) override {
    metrics_ = metrics;
    return RET_OK;
  }

  int SetKernelCallBack(const KernelCallBack &before, const KernelCallBack &after) override {
    before_cb_ = before;
    after_cb_ = after;
    return RET_OK;
  }

  int Train(int epochs, dataset::Dataset *dataset, std::vector<session::TrainLoopCallBack *> cbs,
            LoadDataFunc load_func = nullptr) override;
  int Eval(dataset::Dataset *dataset, std::vector<session::TrainLoopCallBack *> cbs, LoadDataFunc load_func = nullptr,
           int max_steps = 0) override;

  std::vector<mindspore::session::Metrics *> GetMetrics() override { return metrics_; }

 protected:
  static int LoadData(std::vector<tensor::MSTensor *> inputs, dataset::MSTensorVec *dataset_vec);
  static int LoadPartialData(std::vector<tensor::MSTensor *> inputs, dataset::MSTensorVec *dataset_vec);

  session::TrainSession *train_session_ = nullptr;
  unsigned int epoch_ = 0;
  KernelCallBack before_cb_ = nullptr;
  KernelCallBack after_cb_ = nullptr;
  int batch_size;
  std::vector<mindspore::session::Metrics *> metrics_;
};
}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_TRAIN_TRAIN_LOOP_H_
