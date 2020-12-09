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
#include "backend/session/executor.h"
#include "runtime/device/kernel_runtime_manager.h"
#include "backend/session/executor_manager.h"
#include "utils/comm_manager.h"
#include "utils/scoped_long_running.h"

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
      tensor->SetNeedWait(false);
      auto iter = tensor_to_node.find(tensor);
      if (iter != tensor_to_node.end()) {
        auto &node = iter->second.first;
        auto &output_index = iter->second.second;
        auto address = AnfAlgo::GetMutableOutputAddr(node, output_index);
        tensor->set_device_address(address);
      }
      if (tensor->NeedSyncDeviceToHostImmediately()) {
        tensor->data_sync();
        tensor->set_device_address(nullptr);
        tensor->set_sync_status(kNeedSyncHostToDevice);
      }
    }
  }
}
}  // namespace
void CompileNodesTask::Run() {
  MS_EXCEPTION_IF_NULL(session_);
  graph_id_ = session_->CompileGraph(nodes_, output_nodes_);
}

void CompileGraphTask::Run() {
  MS_EXCEPTION_IF_NULL(session_);
  graph_id_ = session_->CompileGraph(NOT_NULL(func_graph_));
}

void BuildGraphTask::Run() {
  MS_EXCEPTION_IF_NULL(session_);
  session_->BuildGraph(graph_id_);
}

void RunGraphTask::Run() {
  MS_EXCEPTION_IF_NULL(session_);
  session_->RunGraph(graph_id_, input_tensors_, &outputs_);
  UpdateOutputTensors(&outputs_, tensor_to_node_);
  ExecutorManager::Instance().OnRunGraphFinished();
}

void BuildOpTask::Run() {
  MS_EXCEPTION_IF_NULL(session_);
  session_->BuildOp(*op_run_info_, graph_info_, input_tensors_, tensors_mask_);
}

void RunOpTask::Run() {
  MS_EXCEPTION_IF_NULL(session_);
  session_->RunOp(*op_run_info_, graph_info_, input_tensors_, &outputs_);
}

void CreateCommGroupTask::Run() { result_ = CommManager::GetInstance().CreateGroupSync(group_name_, ranks_); }

void DestroyCommGroupTask::Run() { result_ = CommManager::GetInstance().DestroyGroup(group_name_); }

Executor::Executor(const std::string &device_name, uint32_t device_id) {
  device_name_ = device_name;
  device_id_ = device_id;
  worker_ = std::make_shared<std::thread>(&Executor::WorkerLoop, this);
}

Executor::~Executor() { WorkerJoin(); }

void Executor::CheckException() {
  if (exception_ptr_ != nullptr) {
    auto exception_ptr = exception_ptr_;
    exception_ptr_ = nullptr;
    std::rethrow_exception(exception_ptr);
  }
}

void Executor::WorkerJoin() {
  if (worker_->joinable() && worker_->get_id() != std::this_thread::get_id()) {
    {
      std::unique_lock<std::mutex> lock(task_mutex_);
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
      exception_ptr_ = std::current_exception();
    }
    task = nullptr;
    sync_cond_var_.notify_all();
  }
}

std::vector<std::shared_ptr<RunGraphTask>> Executor::GetNewReadyTasks() {
  std::vector<std::shared_ptr<RunGraphTask>> new_ready_tasks;
  std::unique_lock<std::mutex> lock(pending_task_mutex_);
  for (auto iter = pending_tasks_.begin(); iter != pending_tasks_.end();) {
    auto task = *iter;
    if (IsAllInputsReady(task->input_tensors_)) {
      new_ready_tasks.emplace_back(task);
      pending_tasks_.erase(iter++);
    } else {
      iter++;
    }
  }
  return new_ready_tasks;
}

void Executor::OnRunGraphFinished() {
  auto new_ready_tasks = GetNewReadyTasks();
  std::unique_lock<std::mutex> lock(task_mutex_);
  for (auto &task : new_ready_tasks) {
    ready_tasks_.push(task);
  }
  if (new_ready_tasks.size() > 0) {
    task_cond_var_.notify_all();
  }
}

bool Executor::IsAllInputsReady(const std::vector<tensor::TensorPtr> &inputs) {
  for (auto &input : inputs) {
    MS_EXCEPTION_IF_NULL(input);
    if (input->NeedWait()) {
      return false;
    }
  }
  return true;
}

GraphId Executor::CompileGraphAsync(const SessionPtr &session, const AnfNodePtrList &lst,
                                    const AnfNodePtrList &outputs) {
  CheckException();
  std::unique_lock<std::mutex> lock(task_mutex_);
  auto task = std::make_shared<CompileNodesTask>();
  task->session_ = session;
  task->nodes_ = lst;
  task->output_nodes_ = outputs;
  ready_tasks_.push(task);
  task_cond_var_.notify_all();
  sync_cond_var_.wait(lock);
  CheckException();
  return task->graph_id_;
}

GraphId Executor::CompileGraphAsync(const SessionPtr &session, NotNull<FuncGraphPtr> func_graph) {
  CheckException();
  std::unique_lock<std::mutex> lock(task_mutex_);
  auto task = std::make_shared<CompileGraphTask>();
  task->session_ = session;
  task->func_graph_ = func_graph;
  ready_tasks_.push(task);
  task_cond_var_.notify_all();
  sync_cond_var_.wait(lock);
  CheckException();
  return task->graph_id_;
}

void Executor::BuildGraphAsync(const SessionPtr &session, GraphId graphId) {
  CheckException();
  std::unique_lock<std::mutex> lock(task_mutex_);
  auto task = std::make_shared<BuildGraphTask>();
  task->session_ = session;
  task->graph_id_ = graphId;
  ready_tasks_.push(task);
  task_cond_var_.notify_all();
  sync_cond_var_.wait(lock);
  CheckException();
}

void Executor::RunGraphAsync(const SessionPtr &session, const GraphId &graph_id,
                             const std::vector<tensor::TensorPtr> &inputs, VectorRef *outputs) {
  CheckException();
  auto task = std::make_shared<RunGraphTask>();
  task->session_ = session;
  task->graph_id_ = graph_id;
  task->input_tensors_ = inputs;
  MS_EXCEPTION_IF_NULL(session);
  session->CreateOutputTensors(graph_id, inputs, outputs, &task->tensor_to_node_);
  // maintain a copy of output vector
  task->outputs_ = *outputs;

  bool ready = IsAllInputsReady(inputs);
  if (!ready) {
    std::unique_lock<std::mutex> lock(pending_task_mutex_);
    pending_tasks_.push_back(task);
    return;
  }
  std::unique_lock<std::mutex> lock(task_mutex_);
  ready_tasks_.push(task);
  task_cond_var_.notify_all();
  mindspore::ScopedLongRunning long_running;
  sync_cond_var_.wait(lock);
  CheckException();
}

void Executor::BuildOpAsync(const SessionPtr &session, OpRunInfo *op_run_info, const GraphInfo &graph_info,
                            const std::vector<tensor::TensorPtr> &input_tensors, const std::vector<int> &tensors_mask) {
  CheckException();
  std::unique_lock<std::mutex> lock(task_mutex_);
  auto task = std::make_shared<BuildOpTask>();
  task->session_ = session;
  task->op_run_info_ = op_run_info;
  task->graph_info_ = graph_info;
  task->input_tensors_ = input_tensors;
  task->tensors_mask_ = tensors_mask;
  ready_tasks_.push(task);
  task_cond_var_.notify_all();
  sync_cond_var_.wait(lock);
  CheckException();
}

void Executor::RunOpAsync(const SessionPtr &session, OpRunInfo *op_run_info, const GraphInfo &graph_info,
                          const std::vector<tensor::TensorPtr> &input_tensors, VectorRef *outputs) {
  CheckException();
  std::unique_lock<std::mutex> lock(task_mutex_);
  auto task = std::make_shared<RunOpTask>();
  task->session_ = session;
  task->op_run_info_ = op_run_info;
  task->graph_info_ = graph_info;
  task->input_tensors_ = input_tensors;
  ready_tasks_.push(task);
  task_cond_var_.notify_all();
  sync_cond_var_.wait(lock);
  CheckException();
  *outputs = task->outputs_;
}

bool Executor::CreateCommGroup(const std::string &group_name, std::vector<uint32_t> ranks) {
  std::unique_lock<std::mutex> lock(task_mutex_);
  auto task = std::make_shared<CreateCommGroupTask>();
  task->group_name_ = group_name;
  task->ranks_ = ranks;
  ready_tasks_.push(task);
  task_cond_var_.notify_all();
  sync_cond_var_.wait(lock);
  return task->result_;
}

bool Executor::DestroyCommGroup(const std::string &group_name) {
  std::unique_lock<std::mutex> lock(task_mutex_);
  auto task = std::make_shared<DestroyCommGroupTask>();
  task->group_name_ = group_name;
  ready_tasks_.push(task);
  task_cond_var_.notify_all();
  sync_cond_var_.wait(lock);
  return task->result_;
}

void Executor::OnWorkerExit() {
  if (device_name_ == kAscendDevice) {
    device::KernelRuntimeManager::Instance().ReleaseKernelRuntime(kAscendDevice, device_id_);
  }
}
}  // namespace session
}  // namespace mindspore
