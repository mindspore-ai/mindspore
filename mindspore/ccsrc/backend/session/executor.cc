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

namespace mindspore {
namespace session {
namespace {
void UpdateOutputTensors(VectorRef *outputs,
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
      if (tensor->need_sync()) {
        tensor->data_sync();
        tensor->set_need_sync(false);
      }
    }
  }
}

BaseRef TransformBaseRefListToTuple(const BaseRef &base_ref) {
  if (utils::isa<VectorRef>(base_ref)) {
    auto ref_list = utils::cast<VectorRef>(base_ref);
    py::tuple output_tensors(ref_list.size());
    for (size_t i = 0; i < ref_list.size(); ++i) {
      auto output = TransformBaseRefListToTuple(ref_list[i]);  // use pyObjectRef
      if (utils::isa<tensor::TensorPtr>(output)) {
        auto tensor_ptr = utils::cast<tensor::TensorPtr>(output);
        MS_EXCEPTION_IF_NULL(tensor_ptr);
        output_tensors[i] = tensor_ptr;
      } else if (utils::isa<PyObjectRef>(output)) {
        py::object obj = utils::cast<PyObjectRef>(output).object_;
        py::tuple tensor_tuple = py::cast<py::tuple>(obj);
        output_tensors[i] = tensor_tuple;
      } else {
        MS_LOG(EXCEPTION) << "The output is not a base ref list or a tensor!";
      }
    }
    return output_tensors;  // turn tuple to py::object and store in PyObjectRef
  } else if (utils::isa<tensor::TensorPtr>(base_ref)) {
    return base_ref;
  } else {
    MS_LOG(EXCEPTION) << "The output is not a base ref list or a tensor!";
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

Executor::Executor(const std::string &device_name, uint32_t device_id) {
  device_name_ = device_name;
  device_id_ = device_id;
  worker_ = std::make_shared<std::thread>(&Executor::WorkerLoop, this);
}

void Executor::WorkerJoin() {
  StopWorker();
  worker_->join();
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
    task->Run();
    if (task->type_ == kCompileNodes) {
      compile_cond_var_.notify_all();
    } else if (task->type_ == kCompileGraph) {
      compile_cond_var_.notify_all();
    } else if (task->type_ == kBuildGraph) {
      build_cond_var_.notify_all();
    } else if (task->type_ == kRunGraph) {
      run_cond_var_.notify_all();
    } else if (task->type_ == kBuildOp) {
      build_op_cond_var_.notify_all();
    } else if (task->type_ == kRunOp) {
      run_op_cond_var_.notify_all();
    }
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
  std::unique_lock<std::mutex> lock(task_mutex_);
  auto task = std::make_shared<CompileNodesTask>();
  task->session_ = session;
  task->nodes_ = lst;
  task->output_nodes_ = outputs;
  ready_tasks_.push(task);
  task_cond_var_.notify_all();
  compile_cond_var_.wait(lock);
  return task->graph_id_;
}

GraphId Executor::CompileGraphAsync(const SessionPtr &session, NotNull<FuncGraphPtr> func_graph) {
  std::unique_lock<std::mutex> lock(task_mutex_);
  auto task = std::make_shared<CompileGraphTask>();
  task->session_ = session;
  task->func_graph_ = func_graph;
  ready_tasks_.push(task);
  task_cond_var_.notify_all();
  compile_cond_var_.wait(lock);
  return task->graph_id_;
}

void Executor::BuildGraphAsync(const SessionPtr &session, GraphId graphId) {
  std::unique_lock<std::mutex> lock(task_mutex_);
  auto task = std::make_shared<BuildGraphTask>();
  task->session_ = session;
  task->graph_id_ = graphId;
  ready_tasks_.push(task);
  task_cond_var_.notify_all();
  build_cond_var_.wait(lock);
}

void Executor::RunGraphAsync(const SessionPtr &session, const GraphId &graph_id,
                             const std::vector<tensor::TensorPtr> &inputs, VectorRef *outputs) {
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
  py::gil_scoped_release release;
  run_cond_var_.wait(lock);
}

void Executor::BuildOpAsync(const SessionPtr &session, OpRunInfo *op_run_info, const GraphInfo &graph_info,
                            const std::vector<tensor::TensorPtr> &input_tensors, const std::vector<int> &tensors_mask) {
  std::unique_lock<std::mutex> lock(task_mutex_);
  auto task = std::make_shared<BuildOpTask>();
  task->session_ = session;
  task->op_run_info_ = op_run_info;
  task->graph_info_ = graph_info;
  task->input_tensors_ = input_tensors;
  task->tensors_mask_ = tensors_mask;
  ready_tasks_.push(task);
  task_cond_var_.notify_all();
  build_op_cond_var_.wait(lock);
}

py::tuple Executor::RunOpAsync(const SessionPtr &session, OpRunInfo *op_run_info, const GraphInfo &graph_info,
                               const std::vector<tensor::TensorPtr> &input_tensors) {
  std::unique_lock<std::mutex> lock(task_mutex_);
  auto task = std::make_shared<RunOpTask>();
  task->session_ = session;
  task->op_run_info_ = op_run_info;
  task->graph_info_ = graph_info;
  task->input_tensors_ = input_tensors;
  ready_tasks_.push(task);
  task_cond_var_.notify_all();
  run_op_cond_var_.wait(lock);

  // Trans output to tuple
  auto output_tensors = TransformBaseRefListToTuple(task->outputs_);
  if (!utils::isa<PyObjectRef>(output_tensors) ||
      !py::isinstance<py::tuple>(utils::cast<PyObjectRef>(output_tensors).object_)) {
    MS_EXCEPTION(NotSupportError) << "The output tensors should be a tuple !";
  }
  py::object tuple_obj = utils::cast<PyObjectRef>(output_tensors).object_;
  py::tuple tuple_tensors = py::cast<py::tuple>(tuple_obj);
  return tuple_tensors;
}

void Executor::StopWorker() {
  std::unique_lock<std::mutex> lock(task_mutex_);
  auto task = std::make_shared<ExitTask>();
  ready_tasks_.push(task);
  task_cond_var_.notify_all();
}

void Executor::OnWorkerExit() {
  if (device_name_ == kAscendDevice) {
    device::KernelRuntimeManager::Instance().ReleaseKernelRuntime(kAscendDevice, device_id_);
  }
}
}  // namespace session
}  // namespace mindspore
