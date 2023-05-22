/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_MINDSPORE_CCSRC_PIPELINE_PYNATIVE_FORWARD_FORWARD_TASK_H_
#define MINDSPORE_MINDSPORE_CCSRC_PIPELINE_PYNATIVE_FORWARD_FORWARD_TASK_H_

#include <functional>
#include <utility>
#include "runtime/pynative/async/task.h"
#include "pipeline/pynative/base.h"
#include "backend/common/session/session_basic.h"

namespace mindspore {
namespace pynative {
class FrontendTask : public AsyncTask {
 public:
  FrontendTask(std::function<void(const FrontendOpRunInfoPtr &op_run_info)> run_func, FrontendOpRunInfoPtr op_run_info)
      : AsyncTask(kFrontendTask), run_func_(std::move(run_func)), op_run_info_(std::move(op_run_info)) {}
  ~FrontendTask() override = default;
  void Run() override;
  void SetException(const std::exception_ptr &e) override;

 private:
  std::function<void(const FrontendOpRunInfoPtr &op_run_info)> run_func_;
  FrontendOpRunInfoPtr op_run_info_;
};

using BackendOpRunInfoPtr = session::BackendOpRunInfoPtr;
class BackendTask : public AsyncTask {
 public:
  BackendTask(
    std::function<void(const FrontendOpRunInfoPtr &op_run_info, const BackendOpRunInfoPtr &backend_op_run_info)>
      run_func,
    FrontendOpRunInfoPtr op_run_info, BackendOpRunInfoPtr backend_op_run_info)
      : AsyncTask(kBackendTask),
        run_func_(std::move(run_func)),
        op_run_info_(std::move(op_run_info)),
        backend_op_run_info_(std::move(backend_op_run_info)) {}
  ~BackendTask() override = default;
  void Run() override;
  void SetException(const std::exception_ptr &e) override;

 private:
  std::function<void(const FrontendOpRunInfoPtr &op_run_info, const BackendOpRunInfoPtr &backend_op_run_info)>
    run_func_;
  FrontendOpRunInfoPtr op_run_info_;
  BackendOpRunInfoPtr backend_op_run_info_;
};
}  // namespace pynative
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_PIPELINE_PYNATIVE_FORWARD_FORWARD_TASK_H_
