/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include "backend/session/executor.h"
#include "backend/session/executor_manager.h"
#include <algorithm>
#include <exception>
#include "runtime/device/kernel_runtime_manager.h"
#include "utils/comm_manager.h"
#include "utils/scoped_long_running.h"
#include "pybind_api/ir/tensor_py.h"
#if (ENABLE_CPU && (ENABLE_D || ENABLE_GPU))
#include "ps/ps_cache/ps_cache_manager.h"
#endif

using mindspore::tensor::TensorPy;
namespace mindspore {
namespace session {
namespace {
void UpdateOutputTensors(const VectorRef *outputs,
                         const std::map<tensor::TensorPtr, session::KernelWithIndex> &tensor_to_node) {
  MS_EXCEPTION_IF_NULL(outputs);
  for (auto item : *outputs) {
    if (utils::isa<VectorRefPtr>(item)) {
      auto vector_ref = utils::cast<VectorRef>(item);
      UpdateOutputTensors(&vector_ref, tensor_to_node);
    } else if (utils::isa<tensor::TensorPtr>(item)) {
      auto tensor = utils::cast<tensor::TensorPtr>(item);
      MS_EXCEPTION_IF_NULL(tensor);
      auto iter = tensor_to_node.find(tensor);
      if (iter != tensor_to_node.end()) {
        auto &node = iter->second.first;
        auto &output_index = iter->second.second;
        auto address = AnfAlgo::GetMutableOutputAddr(node, output_index);
        tensor->set_device_address(address);

        if (AnfAlgo::IsDynamicShape(node)) {
          auto updated_shape = AnfAlgo::GetOutputInferShape(node, output_index);
          ShapeVector int_shape;
          std::transform(updated_shape.begin(), updated_shape.end(), std::back_inserter(int_shape), SizeToInt);
          tensor->set_shape(int_shape);
        }
      }
      if (tensor->NeedSyncDeviceToHostImmediately()) {
        tensor->data_sync(false);
        tensor->set_device_address(nullptr);
        tensor->set_sync_status(kNeedSyncHostToDevice);
      }
    }
  }
}

void NotifyOutputTensors(const VectorRef *outputs) {
  MS_EXCEPTION_IF_NULL(outputs);
  for (auto item : *outputs) {
    if (utils::isa<VectorRefPtr>(item)) {
      auto vector_ref = utils::cast<VectorRef>(item);
      NotifyOutputTensors(&vector_ref);
    } else if (utils::isa<tensor::TensorPtr>(item)) {
      auto tensor = utils::cast<tensor::TensorPtr>(item);
      MS_EXCEPTION_IF_NULL(tensor);
      tensor->SetNeedWait(false);
    }
  }
}

bool TensorInVector(const VectorRef *outputs) {
  MS_EXCEPTION_IF_NULL(outputs);
  for (auto item : *outputs) {
    if (utils::isa<VectorRefPtr>(item)) {
      auto vector_ref = utils::cast<VectorRef>(item);
      if (TensorInVector(&vector_ref)) {
        return true;
      }
    } else if (utils::isa<tensor::TensorPtr>(item)) {
      return true;
    }
  }
  return false;
}
}  // namespace

void CompileNodesTask::Run() {
  MS_EXCEPTION_IF_NULL(session_);
  MS_EXCEPTION_IF_NULL(segment_);
  graph_id_ = session_->CompileGraphImpl(segment_->nodes_, output_nodes_);
}

void CompileGraphTask::Run() {
  MS_EXCEPTION_IF_NULL(session_);
  graph_id_ = session_->CompileGraphImpl(NOT_NULL(func_graph_));
}

void BuildGraphTask::Run() {
  MS_EXCEPTION_IF_NULL(session_);
  session_->BuildGraphImpl(graph_id_);
}

void RunGraphTask::Run() {
  MS_EXCEPTION_IF_NULL(session_);
  MS_LOG(INFO) << "Start run graph " << graph_id_;
  auto graph = session_->GetGraph(graph_id_);
  if (graph == nullptr) {
    MS_LOG(ERROR) << "Invalid graph id " << graph_id_;
    return;
  }
  graph->ResetGraphRunningStatus();
  try {
    session_->RunGraphImpl(graph_id_, input_tensors_, &outputs_);
    UpdateOutputTensors(&outputs_, tensor_to_node_);
  } catch (const std::exception &e) {
    ExecutorManager::Instance().OnEvent(ExecutorEvent::kException);
    MsException::Instance().SetException();
  }
  MS_LOG(INFO) << "End run graph " << graph_id_;
  graph->OnRunGraphFinished();
  for (auto &tensor : input_need_lock_tensors_) {
    tensor->SetNeedWait(false);
  }
  NotifyOutputTensors(&outputs_);
  ExecutorManager::Instance().OnEvent(ExecutorEvent::kRunGraphFinished);
}

void RunOpTask::Run() {
  MS_EXCEPTION_IF_NULL(session_);
  session_->RunOpImpl(graph_info_, op_run_info_, input_tensors_, &outputs_, tensors_mask_);
}

void RunOpsInGraphTask::Run() {
  MS_EXCEPTION_IF_NULL(session_);
  session_->RunOpsInGraphImpl(graph_id_, input_tensors_, &outputs_);
}

void CreateCommGroupTask::Run() { result_ = CommManager::GetInstance().CreateGroupSync(group_name_, ranks_); }

void DestroyCommGroupTask::Run() { result_ = CommManager::GetInstance().DestroyGroup(group_name_); }

Executor::Executor(const std::string &device_name, uint32_t device_id) {
  device_name_ = device_name;
  device_id_ = device_id;
  worker_ = std::make_shared<std::thread>(&Executor::WorkerLoop, this);
}

Executor::~Executor() { WorkerJoin(); }

void Executor::WorkerJoin() {
  // Avoid worker thread join itself which will cause deadlock
  if (worker_->joinable() && worker_->get_id() != std::this_thread::get_id()) {
    {
      std::lock_guard<std::mutex> lock(task_mutex_);
      auto task = std::make_shared<ExitTask>();
      ready_tasks_.push(task);
      task_cond_var_.notify_all();
    }
    worker_->join();
  }
}

void Executor::WorkerLoop() {
  while (true) {
    std::shared_ptr<Task> task;
    {
      std::unique_lock<std::mutex> lock(task_mutex_);
      task_cond_var_.wait(lock, [this] { return !ready_tasks_.empty(); });
      task = ready_tasks_.front();
      ready_tasks_.pop();
    }
    if (task->type_ == kExit) {
      OnWorkerExit();
      return;
    }
    try {
      task->Run();
    } catch (const std::exception &e) {
      ExecutorManager::Instance().OnEvent(ExecutorEvent::kException);
      MsException::Instance().SetException();
    }
    {
      std::lock_guard<std::mutex> lock(done_task_mutex_);
      done_tasks_.emplace_back(task);
    }
    if (task->type_ != kRunGraph || task->sync_run_) {
      std::lock_guard<std::mutex> lock(task_mutex_);
      sync_run_task_finished_ = true;
      sync_cond_var_.notify_all();
    }
  }
}

std::vector<std::shared_ptr<RunGraphTask>> Executor::GetNewReadyTasks() {
  std::vector<std::shared_ptr<RunGraphTask>> new_ready_tasks;
  std::lock_guard<std::mutex> lock(pending_task_mutex_);
  for (auto iter = pending_tasks_.begin(); iter != pending_tasks_.end();) {
    auto task = *iter;
    if (IsTaskReady(task)) {
      new_ready_tasks.emplace_back(task);
      pending_tasks_.erase(iter++);
    } else {
      iter++;
    }
  }
  return new_ready_tasks;
}

void Executor::OnEvent(const ExecutorEvent &event) {
  if (event == ExecutorEvent::kRunGraphFinished) {
    OnRunGraphFinished();
  } else if (event == ExecutorEvent::kClear) {
    OnClear();
  } else if (event == ExecutorEvent::kException) {
    OnException();
  }
}

void Executor::OnClear() {
  WorkerJoin();
  ClearDoneTasks();
}

void Executor::OnException() {
  std::vector<std::shared_ptr<Task>> new_done_tasks;
  {
    std::lock_guard<std::mutex> lock(task_mutex_);
    while (!ready_tasks_.empty()) {
      new_done_tasks.emplace_back(ready_tasks_.front());
      ready_tasks_.pop();
    }
  }
  {
    std::lock_guard<std::mutex> lock(pending_task_mutex_);
    std::copy(pending_tasks_.begin(), pending_tasks_.end(), std::back_inserter(new_done_tasks));
    pending_tasks_.clear();
  }
  {
    std::lock_guard<std::mutex> lock(done_task_mutex_);
    (void)done_tasks_.insert(done_tasks_.end(), new_done_tasks.begin(), new_done_tasks.end());
  }
}

void Executor::OnRunGraphFinished() {
  auto new_ready_tasks = GetNewReadyTasks();
  std::lock_guard<std::mutex> lock(task_mutex_);
  for (auto &task : new_ready_tasks) {
    ready_tasks_.push(task);
  }
  if (!new_ready_tasks.empty()) {
    task_cond_var_.notify_all();
  }
  reenter_cond_var_.notify_all();
}

bool Executor::IsTaskReady(const std::shared_ptr<RunGraphTask> &task) {
  MS_EXCEPTION_IF_NULL(task);
  for (auto &input : task->input_need_wait_tensors_) {
    MS_EXCEPTION_IF_NULL(input);
    if (input->NeedWait()) {
      return false;
    }
  }
  auto session = task->session_;
  MS_EXCEPTION_IF_NULL(session);
  auto graph = session->GetGraph(task->graph_id_);
  if (graph != nullptr) {
    return graph->IsPreGraphFinished();
  }
  return true;
}

void Executor::ClearDoneTasks() {
  std::lock_guard<std::mutex> lock(done_task_mutex_);
  done_tasks_.clear();
}

void Executor::RunTask(const std::shared_ptr<Task> &task, bool sync, bool long_run) {
  {
    std::lock_guard<std::mutex> lock(task_mutex_);
    sync_run_task_finished_ = false;
    ready_tasks_.push(task);
  }
  task_cond_var_.notify_all();
  if (sync && !sync_run_task_finished_) {
    std::unique_lock<std::mutex> lock(task_mutex_);
    if (long_run) {
      mindspore::ScopedLongRunning long_running;
      sync_cond_var_.wait(lock, [this] { return sync_run_task_finished_; });
    } else {
      sync_cond_var_.wait(lock, [this] { return sync_run_task_finished_; });
    }
  }
  ClearDoneTasks();
  MsException::Instance().CheckException();
}

GraphId Executor::CompileGraph(const SessionPtr &session, const GraphSegmentPtr &segment,
                               const AnfNodePtrList &outputs) {
  auto task = std::make_shared<CompileNodesTask>();
  task->session_ = session;
  task->segment_ = segment;
  task->output_nodes_ = outputs;
  RunTask(task, true);
  return task->graph_id_;
}

GraphId Executor::CompileGraph(const SessionPtr &session, NotNull<FuncGraphPtr> func_graph) {
  auto task = std::make_shared<CompileGraphTask>();
  task->session_ = session;
  task->func_graph_ = func_graph.get();
  RunTask(task, true);
  return task->graph_id_;
}

void Executor::BuildGraph(const SessionPtr &session, GraphId graphId) {
  auto task = std::make_shared<BuildGraphTask>();
  task->session_ = session;
  task->graph_id_ = graphId;
  RunTask(task, true);
}

void Executor::RunGraph(const SessionPtr &session, const GraphId &graph_id,
                        const std::vector<tensor::TensorPtr> &inputs, VectorRef *outputs) {
  MS_EXCEPTION_IF_NULL(session);
  MS_EXCEPTION_IF_NULL(outputs);
  auto task = std::make_shared<RunGraphTask>();
  task->session_ = session;
  task->graph_id_ = graph_id;
  task->input_tensors_ = inputs;
  session->CreateOutputTensors(graph_id, inputs, outputs, &task->tensor_to_node_);
  task->outputs_ = *outputs;
  task->sync_run_ = true;
  RunTask(task, true, true);
}

void Executor::WaitLockedInputs(const SessionPtr &session, const std::shared_ptr<RunGraphTask> &task) {
  bool need_lock = false;
  for (auto &tensor : task->input_tensors_) {
    if (tensor->NeedWait()) {
      if (tensor->IsGraphOutput()) {
        task->input_need_wait_tensors_.emplace_back(tensor);
      } else {
        need_lock = true;
      }
    }
  }
  if (need_lock) {
    mindspore::ScopedLongRunning long_running;
    for (auto &tensor : task->input_tensors_) {
      if (tensor->NeedWait() && !tensor->IsGraphOutput()) {
        MsException::Instance().CheckException();
        tensor->Wait();
      }
    }
    MsException::Instance().CheckException();
  }
  // need lock input parameters for optimizer
  for (auto &tensor : task->input_need_lock_tensors_) {
    tensor->SetNeedWait(true);
  }
}

void Executor::RunGraphAsync(const SessionPtr &session, const GraphId &graph_id,
                             const std::vector<tensor::TensorPtr> &inputs, VectorRef *outputs) {
  MS_EXCEPTION_IF_NULL(session);
  MS_EXCEPTION_IF_NULL(outputs);
  auto task = std::make_shared<RunGraphTask>();
  task->session_ = session;
  task->graph_id_ = graph_id;
  task->input_tensors_ = inputs;
  task->input_need_lock_tensors_ = session->GetInputNeedLockTensors(graph_id, inputs);
  auto graph = session->GetGraph(task->graph_id_);
  if (graph != nullptr && !graph->IsPostGraphFinished()) {
    mindspore::ScopedLongRunning long_running;
    std::unique_lock<std::mutex> lock(reenter_mutex_);
    reenter_cond_var_.wait(lock, [&graph] { return graph->IsPostGraphFinished(); });
    MsException::Instance().CheckException();
  }
  session->CreateOutputTensors(graph_id, inputs, outputs, &task->tensor_to_node_);
  // maintain a copy of output vector
  task->outputs_ = *outputs;
  // sync run graph without output tensor(int dataset graph)
  if (!TensorInVector(outputs)) {
    task->sync_run_ = true;
    RunTask(task, true, true);
    return;
  }
  WaitLockedInputs(session, task);
  {
    std::lock_guard<std::mutex> lock(pending_task_mutex_);
    if (!IsTaskReady(task)) {
      pending_tasks_.push_back(task);
      return;
    }
  }
  RunTask(task, false);
}

void Executor::RunOp(const SessionPtr &session, OpRunInfo *op_run_info, const GraphInfo &graph_info,
                     std::vector<tensor::TensorPtr> *input_tensors, VectorRef *outputs,
                     const std::vector<int64_t> &tensors_mask) {
  MS_EXCEPTION_IF_NULL(session);
  auto ms_context = MsContext::GetInstance();
  auto target = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  if (target == kGPUDevice) {
    for (auto &tensor : *input_tensors) {
      if (tensor->NeedWait()) {
        tensor->Wait();
      }
    }
    {
      // Release GIL before calling into (potentially long-running) C++ code
      if (Py_IsInitialized()) {
        py::gil_scoped_release release;
        session->RunOpImpl(graph_info, op_run_info, input_tensors, outputs, tensors_mask);
      } else {
        session->RunOpImpl(graph_info, op_run_info, input_tensors, outputs, tensors_mask);
      }
    }
  } else {
    auto task = std::make_shared<RunOpTask>();
    task->session_ = session;
    task->op_run_info_ = op_run_info;
    task->graph_info_ = graph_info;
    task->input_tensors_ = input_tensors;
    task->tensors_mask_ = tensors_mask;
    for (auto &tensor : *input_tensors) {
      if (tensor->NeedWait()) {
        tensor->Wait();
      }
    }
    RunTask(task, true, true);
    *outputs = task->outputs_;
  }
}

void Executor::RunOpsInGraph(const SessionPtr &session, const GraphId &graph_id,
                             const std::vector<tensor::TensorPtr> &inputs, VectorRef *outputs) {
  MS_EXCEPTION_IF_NULL(session);
  MS_EXCEPTION_IF_NULL(outputs);
  auto task = std::make_shared<RunOpsInGraphTask>();
  task->session_ = session;
  task->graph_id_ = graph_id;
  task->input_tensors_ = inputs;
  RunTask(task, true, true);
  *outputs = task->outputs_;
}

bool Executor::CreateCommGroup(const std::string &group_name, std::vector<uint32_t> ranks) {
  auto task = std::make_shared<CreateCommGroupTask>();
  task->group_name_ = group_name;
  task->ranks_ = ranks;
  RunTask(task, true);
  return task->result_;
}

bool Executor::DestroyCommGroup(const std::string &group_name) {
  auto task = std::make_shared<DestroyCommGroupTask>();
  task->group_name_ = group_name;
  RunTask(task, true);
  return task->result_;
}

void Executor::OnWorkerExit() {
  if (device_name_ == kAscendDevice) {
    device::KernelRuntimeManager::Instance().ReleaseKernelRuntime(kAscendDevice, device_id_);
  }
}
}  // namespace session
}  // namespace mindspore
