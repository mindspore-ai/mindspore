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

#include "src/train/train_loop.h"
#include <sys/stat.h>
#include <vector>
#include <memory>
#include <algorithm>
#include "include/errorcode.h"
#include "include/train/train_session.h"
#include "include/iterator.h"
#include "src/common/log_adapter.h"

namespace mindspore {
namespace lite {

using dataset::Dataset;
using dataset::Iterator;
using dataset::MSTensorVec;
using session::RET_CONTINUE;
using session::RET_EXIT;
using session::RET_STOP_TRAINING;

TrainLoop::~TrainLoop() {
  if (train_session_ != nullptr) delete train_session_;
}

int TrainLoop::Train(int epochs, Dataset *ds, std::vector<session::TrainLoopCallBack *> cbs, LoadDataFunc load_func) {
  train_session_->Train();
  session::TrainLoopCallBackData cb_data(true, epoch_, train_session_, this);

  if (load_func == nullptr) load_func = TrainLoop::LoadData;

  for (auto cb : cbs) cb->Begin(cb_data);

  for (int i = 0; i < epochs; i++) {
    cb_data.epoch_ = epoch_++;
    for (auto cb : cbs) cb->EpochBegin(cb_data);

    std::shared_ptr<Iterator> iter = ds->CreateIterator();
    MSTensorVec row_vec;
    int s = 0;

    iter->GetNextRow(&row_vec);
    while (row_vec.size() != 0) {
      auto ret = load_func(cb_data.session_->GetInputs(), &row_vec);
      if (ret != RET_OK) break;

      cb_data.step_ = s++;
      for (auto cb : cbs) cb->StepBegin(cb_data);

      train_session_->RunGraph(before_cb_, after_cb_);
      for (auto cb : cbs) cb->StepEnd(cb_data);
      iter->GetNextRow(&row_vec);
    }
    iter->Stop();
    int break_loop = false;
    for (auto cb : cbs) {
      int ret = cb->EpochEnd(cb_data);
      if (ret != RET_CONTINUE) {
        if (ret == RET_EXIT) {
          MS_LOG(ERROR) << "Error in TrainLoop callback";
          return RET_ERROR;
        }
        if (ret == RET_STOP_TRAINING) {
          break_loop = true;
        }
      }
    }
    if (break_loop) {
      break;
    }
  }

  for (auto cb : cbs) cb->End(cb_data);
  return RET_OK;
}

int TrainLoop::Eval(Dataset *ds, std::vector<session::TrainLoopCallBack *> cbs, LoadDataFunc load_func, int max_steps) {
  train_session_->Eval();
  session::TrainLoopCallBackData cb_data(false, epoch_, train_session_, this);

  if (load_func == nullptr) load_func = TrainLoop::LoadData;

  for (auto metric : metrics_) metric->Clear();
  for (auto cb : cbs) cb->Begin(cb_data);
  for (auto cb : cbs) cb->EpochBegin(cb_data);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  MSTensorVec row_vec;
  int s = 0;

  iter->GetNextRow(&row_vec);
  while (row_vec.size() != 0) {
    if (s >= max_steps) break;
    auto ret = load_func(cb_data.session_->GetInputs(), &row_vec);
    if (ret != RET_OK) break;

    cb_data.step_ = ++s;
    for (auto cb : cbs) cb->StepBegin(cb_data);

    train_session_->RunGraph(before_cb_, after_cb_);
    for (auto cb : cbs) cb->StepEnd(cb_data);

    auto outputs = cb_data.session_->GetPredictions();
    for (auto metric : metrics_) metric->Update(cb_data.session_->GetInputs(), outputs);
    iter->GetNextRow(&row_vec);
  }
  iter->Stop();
  for (auto cb : cbs) cb->EpochEnd(cb_data);
  for (auto cb : cbs) cb->End(cb_data);

  return RET_OK;
}

int TrainLoop::LoadData(std::vector<tensor::MSTensor *> inputs, dataset::MSTensorVec *row_vec) {
  auto num_of_inputs = inputs.size();
  if ((num_of_inputs == 0) || (row_vec == nullptr) || (num_of_inputs != row_vec->size())) {
    return RET_STOP_TRAINING;
  }

  for (unsigned int i = 0; i < num_of_inputs; i++) {
    unsigned char *input_data = reinterpret_cast<unsigned char *>(inputs.at(i)->MutableData());
    const unsigned char *row_data = reinterpret_cast<const unsigned char *>(row_vec->at(i).MutableData());
    auto data_size = row_vec->at(i).DataSize();
    if (data_size != inputs.at(i)->Size()) {
      MS_LOG(WARNING) << "Model Input tensor " << i << " size (" << inputs.at(i)->Size()
                      << ") does not match dataset size (" << data_size << ")\n";
      return RET_STOP_TRAINING;
    }
    std::copy(row_data, row_data + data_size, input_data);
  }
  return RET_OK;
}

int TrainLoop::LoadPartialData(std::vector<tensor::MSTensor *> inputs, dataset::MSTensorVec *row_vec) {
  auto num_of_inputs = inputs.size();
  if ((num_of_inputs == 0) || (row_vec == nullptr) || (num_of_inputs < row_vec->size())) {
    return RET_STOP_TRAINING;
  }

  for (unsigned int i = 0; i < row_vec->size(); i++) {
    unsigned char *input_data = reinterpret_cast<unsigned char *>(inputs.at(i)->MutableData());
    const unsigned char *row_data = reinterpret_cast<const unsigned char *>(row_vec->at(i).MutableData());
    auto data_size = row_vec->at(i).DataSize();
    if (data_size != inputs.at(i)->Size()) {
      MS_LOG(WARNING) << "Model Input tensor " << i << " size (" << inputs.at(i)->Size()
                      << ") does not match dataset size (" << data_size << ")\n";
      return RET_STOP_TRAINING;
    }
    std::copy(row_data, row_data + data_size, input_data);
  }
  return RET_OK;
}

}  // namespace lite

session::TrainLoop *session::TrainLoop::CreateTrainLoop(session::TrainSession *train_session) {
  auto loop = new (std::nothrow) lite::TrainLoop(train_session);
  return loop;
}

}  // namespace mindspore
