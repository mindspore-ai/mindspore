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
#include <vector>
#include <memory>
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

class SliceOpFrontendTask : public AsyncTask {
 public:
  SliceOpFrontendTask(
    std::function<void(const std::vector<ValuePtr> &input_values, const std::vector<SliceOpInfoPtr> &slice_op_infos,
                       bool requires_grad, const stub::StubNodePtr &stub_output)>
      run_func,
    std::vector<ValuePtr> input_values, std::vector<SliceOpInfoPtr> slice_op_infos, bool requires_grad,
    const stub::StubNodePtr &stub_output)
      : AsyncTask(kFrontendTask),
        run_func_(std::move(run_func)),
        input_values_(std::move(input_values)),
        slice_op_infos_(std::move(slice_op_infos)),
        requires_grad_(requires_grad),
        stub_output_(stub_output) {}
  ~SliceOpFrontendTask() override = default;
  void Run() override;
  void SetException(const std::exception_ptr &e) override;

 private:
  std::function<void(const std::vector<ValuePtr> &input_values, const std::vector<SliceOpInfoPtr> &slice_op_infos,
                     bool requires_grad, const stub::StubNodePtr &stub_output)>
    run_func_;
  std::vector<ValuePtr> input_values_;
  std::vector<SliceOpInfoPtr> slice_op_infos_;
  bool requires_grad_{false};
  stub::StubNodePtr stub_output_;
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

class ViewKernelBackendTask : public AsyncTask {
 public:
  ViewKernelBackendTask(
    std::function<void(const FrontendOpRunInfoPtr &op_run_info, const KernelTaskType &task_type)> run_func,
    FrontendOpRunInfoPtr op_run_info, const KernelTaskType &task_type)
      : AsyncTask(kBackendTask),
        run_func_(std::move(run_func)),
        op_run_info_(std::move(op_run_info)),
        task_type_(task_type) {}
  ~ViewKernelBackendTask() override = default;
  void Run() override;

 private:
  std::function<void(const FrontendOpRunInfoPtr &op_run_info, const KernelTaskType &task_type)> run_func_;
  FrontendOpRunInfoPtr op_run_info_;
  KernelTaskType task_type_;
};

class AllocViewMemBackendTask : public AsyncTask {
 public:
  AllocViewMemBackendTask(
    std::function<void(const FrontendOpRunInfoPtr &op_run_info, const tensor::TensorPtr &input_tensor,
                       const size_t &input_idx, bool need_wait)>
      run_func,
    FrontendOpRunInfoPtr op_run_info, const tensor::TensorPtr &input_tensor, const size_t &input_idx, bool need_wait)
      : AsyncTask(kBackendTask),
        run_func_(std::move(run_func)),
        op_run_info_(std::move(op_run_info)),
        input_tensor_(input_tensor),
        input_idx_(input_idx),
        need_wait_(need_wait) {}
  ~AllocViewMemBackendTask() override = default;
  void Run() override;
  void SetException(const std::exception_ptr &e) override;

 private:
  std::function<void(const FrontendOpRunInfoPtr &op_run_info, const tensor::TensorPtr &input_tensor,
                     const size_t &input_idx, bool need_wait)>
    run_func_;
  FrontendOpRunInfoPtr op_run_info_;
  tensor::TensorPtr input_tensor_;
  size_t input_idx_{0};
  bool need_wait_{false};
};

class ContiguousBackendTask : public AsyncTask {
 public:
  ContiguousBackendTask(std::function<void(const tensor::TensorPtr &tensor)> run_func, const tensor::TensorPtr &tensor)
      : AsyncTask(kBackendTask), run_func_(std::move(run_func)), tensor_(tensor) {}
  ~ContiguousBackendTask() override = default;
  void Run() override;

 private:
  std::function<void(const tensor::TensorPtr &tensor)> run_func_;
  tensor::TensorPtr tensor_;
};
}  // namespace pynative
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_PIPELINE_PYNATIVE_FORWARD_FORWARD_TASK_H_
