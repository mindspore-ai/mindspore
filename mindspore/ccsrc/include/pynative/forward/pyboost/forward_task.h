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

#ifndef MINDSPORE_MINDSPORE_CCSRC_INCLUDE_PYNATIVE_FORWARD_PYBOOST_FORWARD_TASK_H_
#define MINDSPORE_MINDSPORE_CCSRC_INCLUDE_PYNATIVE_FORWARD_PYBOOST_FORWARD_TASK_H_

#include <functional>
#include <utility>
#include <vector>
#include <memory>
#include "runtime/pipeline/task/task.h"
#include "pynative/utils/base.h"
#include "include/backend/common/kernel_graph/session_basic.h"
#include "include/utils/visible.h"

namespace mindspore {
namespace pynative {

class PYNATIVE_EXPORT PyboostTask : public runtime::AsyncTask {
 public:
  PyboostTask(std::function<void(const PyboostOpRunInfoPtr &op_run_info)> run_func, PyboostOpRunInfoPtr op_run_info)
      : AsyncTask(runtime::kFrontendTask), run_func_(std::move(run_func)), op_run_info_(std::move(op_run_info)) {}
  ~PyboostTask() override = default;
  void Run() override;
  void SetException(const std::exception_ptr &e) override;

 private:
  std::function<void(const PyboostOpRunInfoPtr &op_run_info)> run_func_;
  PyboostOpRunInfoPtr op_run_info_;
};

class PYNATIVE_EXPORT FrontendTask : public runtime::AsyncTask {
 public:
  FrontendTask(std::function<void(const FrontendOpRunInfoPtr &op_run_info)> run_func, FrontendOpRunInfoPtr op_run_info)
      : AsyncTask(runtime::kFrontendTask), run_func_(std::move(run_func)), op_run_info_(std::move(op_run_info)) {}
  ~FrontendTask() override = default;
  void Run() override;
  void SetException(const std::exception_ptr &e) override;

 protected:
  std::function<void(const FrontendOpRunInfoPtr &op_run_info)> run_func_;
  FrontendOpRunInfoPtr op_run_info_;
};

class PYNATIVE_EXPORT PassthroughFrontendTask : public runtime::AsyncTask {
 public:
  explicit PassthroughFrontendTask(std::function<void(void)> run_func)
      : AsyncTask(runtime::kFrontendTask), run_func_(std::move(run_func)) {}
  explicit PassthroughFrontendTask(std::function<void(void)> run_func, std::function<void()> set_exception_func)
      : AsyncTask(runtime::kFrontendTask),
        run_func_(std::move(run_func)),
        set_exception_func_(std::move(set_exception_func)) {}
  ~PassthroughFrontendTask() override = default;
  void Run() override;
  void SetException(const std::exception_ptr &e) override;

 private:
  std::function<void(void)> run_func_;
  std::function<void()> set_exception_func_;
};

class PYNATIVE_EXPORT PyboostPromiseTask : public PyboostTask {
 public:
  PyboostPromiseTask(std::function<void(const PyboostOpRunInfoPtr &op_run_info)> run_func,
                     std::function<void()> set_exception_func, PyboostOpRunInfoPtr op_run_info)
      : PyboostTask(std::move(run_func), std::move(op_run_info)), set_exception_func_(std::move(set_exception_func)) {}

  void SetException(const std::exception_ptr &e) override {
    if (set_exception_func_ == nullptr) {
      MS_LOG(ERROR) << "set_exception_func_ is null";
      return;
    }
    set_exception_func_();
  }

 private:
  std::function<void()> set_exception_func_;
};

class PYNATIVE_EXPORT ViewPyboostPromiseTask : public runtime::AsyncTask {
 public:
  explicit ViewPyboostPromiseTask(std::function<void(void)> run_func, std::function<void()> set_exception_func)
      : AsyncTask(runtime::kFrontendTask),
        run_func_(std::move(run_func)),
        set_exception_func_(std::move(set_exception_func)) {}
  ~ViewPyboostPromiseTask() override = default;
  void Run() override;
  void SetException(const std::exception_ptr &e) override;

 private:
  std::function<void(void)> run_func_;
  std::function<void()> set_exception_func_;
};

class PYNATIVE_EXPORT FrontendPromiseTask : public FrontendTask {
 public:
  FrontendPromiseTask(std::function<void(const FrontendOpRunInfoPtr &op_run_info)> run_func,
                      std::function<void()> set_exception_func, FrontendOpRunInfoPtr op_run_info)
      : FrontendTask(std::move(run_func), std::move(op_run_info)), set_exception_func_(std::move(set_exception_func)) {}

  void SetException(const std::exception_ptr &e) override {
    if (set_exception_func_ == nullptr) {
      MS_LOG(ERROR) << "set_exception_func_ is null";
      return;
    }
    set_exception_func_();
  }

 private:
  std::function<void()> set_exception_func_;
};

class SliceOpFrontendTask : public runtime::AsyncTask {
 public:
  SliceOpFrontendTask(
    std::function<void(const std::vector<ValuePtr> &input_values, const std::vector<SliceOpInfoPtr> &slice_op_infos,
                       bool requires_grad, const stub::StubNodePtr &stub_output, size_t stream_id)>
      run_func,
    std::vector<ValuePtr> input_values, std::vector<SliceOpInfoPtr> slice_op_infos, bool requires_grad,
    const stub::StubNodePtr &stub_output, size_t stream_id)
      : AsyncTask(runtime::kFrontendTask),
        run_func_(std::move(run_func)),
        input_values_(std::move(input_values)),
        slice_op_infos_(std::move(slice_op_infos)),
        requires_grad_(requires_grad),
        stub_output_(stub_output),
        stream_id_(stream_id) {}
  ~SliceOpFrontendTask() override = default;
  void Run() override;
  void SetException(const std::exception_ptr &e) override;

 private:
  std::function<void(const std::vector<ValuePtr> &input_values, const std::vector<SliceOpInfoPtr> &slice_op_infos,
                     bool requires_grad, const stub::StubNodePtr &stub_output, size_t stream_id)>
    run_func_;
  std::vector<ValuePtr> input_values_;
  std::vector<SliceOpInfoPtr> slice_op_infos_;
  bool requires_grad_{false};
  stub::StubNodePtr stub_output_;
  size_t stream_id_{kDefaultStreamIndex};
};

using BackendOpRunInfoPtr = session::BackendOpRunInfoPtr;
class BackendTask : public runtime::AsyncTask {
 public:
  BackendTask(
    std::function<void(const FrontendOpRunInfoPtr &op_run_info, const BackendOpRunInfoPtr &backend_op_run_info)>
      run_func,
    FrontendOpRunInfoPtr op_run_info, BackendOpRunInfoPtr backend_op_run_info)
      : AsyncTask(runtime::kBackendTask),
        run_func_(std::move(run_func)),
        op_run_info_(std::move(op_run_info)),
        backend_op_run_info_(std::move(backend_op_run_info)) {}
  ~BackendTask() override = default;
  void Run() override;

 private:
  std::function<void(const FrontendOpRunInfoPtr &op_run_info, const BackendOpRunInfoPtr &backend_op_run_info)>
    run_func_;
  FrontendOpRunInfoPtr op_run_info_;
  BackendOpRunInfoPtr backend_op_run_info_;
};

class ViewKernelBackendTask : public runtime::AsyncTask {
 public:
  ViewKernelBackendTask(
    std::function<void(const FrontendOpRunInfoPtr &op_run_info, const runtime::KernelTaskType &task_type)> run_func,
    FrontendOpRunInfoPtr op_run_info, const runtime::KernelTaskType &task_type)
      : AsyncTask(runtime::kBackendTask),
        run_func_(std::move(run_func)),
        op_run_info_(std::move(op_run_info)),
        task_type_(task_type) {}
  ~ViewKernelBackendTask() override = default;
  void Run() override;

 private:
  std::function<void(const FrontendOpRunInfoPtr &op_run_info, const runtime::KernelTaskType &task_type)> run_func_;
  FrontendOpRunInfoPtr op_run_info_;
  runtime::KernelTaskType task_type_;
};

class AllocViewMemBackendTask : public runtime::AsyncTask {
 public:
  AllocViewMemBackendTask(
    std::function<void(const FrontendOpRunInfoPtr &op_run_info, const tensor::TensorPtr &input_tensor,
                       const size_t &input_idx, bool need_wait)>
      run_func,
    FrontendOpRunInfoPtr op_run_info, const tensor::TensorPtr &input_tensor, const size_t &input_idx, bool need_wait)
      : AsyncTask(runtime::kBackendTask),
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

class ContiguousBackendTask : public runtime::AsyncTask {
 public:
  ContiguousBackendTask(std::function<void(const tensor::TensorPtr &tensor)> run_func, const tensor::TensorPtr &tensor)
      : AsyncTask(runtime::kBackendTask), run_func_(std::move(run_func)), tensor_(tensor) {}
  ~ContiguousBackendTask() override = default;
  void Run() override;

 private:
  std::function<void(const tensor::TensorPtr &tensor)> run_func_;
  tensor::TensorPtr tensor_;
};
}  // namespace pynative
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_INCLUDE_PYNATIVE_FORWARD_PYBOOST_FORWARD_TASK_H_
