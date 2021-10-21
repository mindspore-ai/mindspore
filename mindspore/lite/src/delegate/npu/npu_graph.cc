/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "src/delegate/npu/npu_graph.h"
#include <queue>
#include "src/delegate/npu/npu_subgraph.h"
#include "src/delegate/delegate_utils.h"
#include "src/delegate/npu/op/transpose_npu.h"
#include "src/delegate/npu/transpose_kernel.h"
namespace mindspore {
NPUGraph::~NPUGraph() {
  for (auto *kernel : all_kernels_) {
    delete kernel;
  }
  for (auto *op : npu_ops_) {
    delete op;
  }
  for (auto tensor : insert_tensors_) {
    MSTensor::DestroyTensorPtr(tensor);
  }
}

void NPUGraph::set_input(mindspore::MSTensor in_tensor, int index) {
  MS_ASSERT(index < inputs_.size());
  auto origin_tensor = this->inputs_[index];
  for (auto kernel : all_kernels_) {
    for (size_t i = 0; i < kernel->inputs().size(); i++) {
      if (kernel->inputs()[i] == origin_tensor) {
        kernel->set_input(in_tensor, i);
      }
    }
  }
  this->inputs_[index] = in_tensor;
}

void NPUGraph::set_output(mindspore::MSTensor out_tensor, int index) {
  MS_ASSERT(index < outputs_.size());
  auto origin_tensor = this->outputs_[index];
  for (auto kernel : all_kernels_) {
    for (size_t i = 0; i < kernel->outputs().size(); i++) {
      if (kernel->outputs()[i] == origin_tensor) {
        kernel->set_output(out_tensor, i);
      }
    }
  }
  this->outputs_[index] = out_tensor;
}

int NPUGraph::Init() {
  all_kernels_.clear();
  std::map<const NPUOp *, bool> is_visited;
  std::map<const NPUOp *, bool> is_searched;
  std::queue<NPUOp *> candidate_in_ops;
  std::queue<NPUOp *> valid_in_ops;
  // Initialization
  for (auto op : npu_ops_) {
    is_visited[op] = false;
    is_searched[op] = false;
    if (op->in_ops().empty()) {
      candidate_in_ops.push(op);
    }
  }
  while (!candidate_in_ops.empty()) {
    // 1. Find out all input ops except transpose, and handle transpose ops independently.
    auto ret = FindValidSubgraphInOps(&valid_in_ops, &candidate_in_ops, &is_visited);
    if (ret != RET_OK) {
      MS_LOG(DEBUG) << "Fail to find valid input ops or handle transpose ops.";
      return RET_ERROR;
    }
    if (valid_in_ops.empty()) {
      MS_LOG(INFO) << "Can not find input ops except transpose.";
      break;
    }
    // 2. Find out all ready ops based on valid input ops, but these ops maybe not belong to the same subgraph.
    auto ready_ops = FindReadySubgraphOps(valid_in_ops, &candidate_in_ops, &is_visited);
    // 3. Create subgraph(s). Input ops with connection will be built into a same subgraph.
    ret = CreateSubgraphFromReadyOps(&valid_in_ops, ready_ops, &is_searched);
    if (ret != RET_OK) {
      MS_LOG(DEBUG) << "Fail to create subgraph(s) from ready ops.";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

std::vector<NPUOp *> NPUGraph::FindPreOps(NPUOp *cur_op) {
  std::vector<NPUOp *> in_ops;
  for (auto in_tensor : cur_op->inputs()) {
    for (auto op : npu_ops_) {
      if (find(op->outputs().begin(), op->outputs().end(), in_tensor) != op->outputs().end()) {
        in_ops.emplace_back(op);
      }
    }
  }
  return in_ops;
}

std::vector<NPUOp *> NPUGraph::FindNextOps(NPUOp *cur_op) {
  std::vector<NPUOp *> out_ops;
  for (auto out_tensor : cur_op->outputs()) {
    for (auto op : npu_ops_) {
      if (find(op->inputs().begin(), op->inputs().end(), out_tensor) != op->inputs().end()) {
        out_ops.emplace_back(op);
      }
    }
  }
  return out_ops;
}

int NPUGraph::FindPreNextOps() {
  for (auto op : npu_ops_) {
    auto in_ops = FindPreOps(op);
    op->set_in_ops(in_ops);
    auto out_ops = FindNextOps(op);
    op->set_out_ops(out_ops);
  }
  return RET_OK;
}

int NPUGraph::FindValidSubgraphInOps(std::queue<NPUOp *> *valid_in_ops, std::queue<NPUOp *> *candidate_in_ops,
                                     std::map<const NPUOp *, bool> *is_visited) {
  while (!candidate_in_ops->empty()) {
    auto cur_op = candidate_in_ops->front();
    candidate_in_ops->pop();
    if ((*is_visited)[cur_op]) {
      continue;
    }
    if (cur_op->type() == schema::PrimitiveType_Transpose) {
      auto transpose_kernel = CreateNPUTransposeKernel(cur_op);
      if (transpose_kernel == nullptr) {
        MS_LOG(DEBUG) << "New NPU transpose kernel failed.";
        return RET_ERROR;
      }
      all_kernels_.emplace_back(transpose_kernel);
      (*is_visited)[cur_op] = true;
      for (auto out_op : cur_op->out_ops()) {
        if (out_op->type() == schema::PrimitiveType_Transpose) {
          candidate_in_ops->push(out_op);
        } else {
          auto input_ready = std::all_of(out_op->in_ops().begin(), out_op->in_ops().end(),
                                         [&](NPUOp *in_op) { return (*is_visited)[in_op] == true; });
          if (input_ready) {
            valid_in_ops->push(out_op);
          }
        }
      }
    } else {
      valid_in_ops->push(cur_op);
    }
  }
  return RET_OK;
}

std::vector<NPUOp *> NPUGraph::FindReadySubgraphOps(std::queue<NPUOp *> op_queue,
                                                    std::queue<NPUOp *> *next_candidate_ops,
                                                    std::map<const NPUOp *, bool> *is_visited) {
  std::vector<NPUOp *> subgraph_ops;
  while (!op_queue.empty()) {
    auto cur_op = op_queue.front();
    op_queue.pop();
    if ((*is_visited)[cur_op]) {
      continue;
    }
    subgraph_ops.emplace_back(cur_op);
    (*is_visited)[cur_op] = true;
    auto out_ops = cur_op->out_ops();
    for (auto out_op : out_ops) {
      if ((*is_visited)[out_op]) {
        continue;
      }
      auto input_ready = std::all_of(out_op->in_ops().begin(), out_op->in_ops().end(),
                                     [&](NPUOp *in_op) { return (*is_visited)[in_op] == true; });
      if (out_op->type() == schema::PrimitiveType_Transpose) {
        next_candidate_ops->push(out_op);
      } else if (input_ready) {
        op_queue.push(out_op);
      }
    }
  }
  return subgraph_ops;
}

void FindConnectedOps(NPUOp *head_op, std::vector<NPUOp *> ready_ops, std::vector<NPUOp *> *connected_ops,
                      std::map<const NPUOp *, bool> *is_searched) {
  std::queue<NPUOp *> bfs_ops;
  bfs_ops.push(head_op);
  while (!bfs_ops.empty()) {
    auto cur_op = bfs_ops.front();
    bfs_ops.pop();
    if ((*is_searched)[cur_op]) {
      continue;
    }
    for (auto in_op : cur_op->in_ops()) {
      if (std::find(ready_ops.begin(), ready_ops.end(), in_op) == ready_ops.end() || (*is_searched)[in_op]) {
        continue;
      }
      bfs_ops.push(in_op);
    }
    for (auto out_op : cur_op->out_ops()) {
      if (std::find(ready_ops.begin(), ready_ops.end(), out_op) == ready_ops.end() || (*is_searched)[out_op]) {
        continue;
      }
      bfs_ops.push(out_op);
    }
    (*is_searched)[cur_op] = true;
    connected_ops->emplace_back(cur_op);
  }
  return;
}

int NPUGraph::CreateSubgraphFromReadyOps(std::queue<NPUOp *> *valid_in_ops, std::vector<NPUOp *> ready_ops,
                                         std::map<const NPUOp *, bool> *is_searched) {
  while (!valid_in_ops->empty()) {
    std::vector<NPUOp *> connected_ops;
    auto op = valid_in_ops->front();
    valid_in_ops->pop();
    if ((*is_searched)[op]) {
      continue;
    }
    if (!valid_in_ops->empty()) {
      // use BFS to find out connected input ops
      FindConnectedOps(op, ready_ops, &connected_ops, is_searched);
    } else {
      // if current input op is the only input op, there is no need to confirm the connectivity
      for (auto ready_op : ready_ops) {
        if (!(*is_searched)[ready_op]) {
          connected_ops.emplace_back(ready_op);
          (*is_searched)[ready_op] = true;
        }
      }
    }
    auto subgraph_kernel = CreateNPUSubgraphKernel(connected_ops);
    if (subgraph_kernel == nullptr) {
      MS_LOG(DEBUG) << "Create NPU subgraph kernel failed.";
      return RET_ERROR;
    }
    all_kernels_.emplace_back(subgraph_kernel);
  }
  return RET_OK;
}

kernel::Kernel *NPUGraph::CreateNPUSubgraphKernel(std::vector<NPUOp *> npu_ops) {
  auto subgraph = new (std::nothrow) NPUSubGraph(npu_ops, npu_manager_);
  if (subgraph == nullptr) {
    MS_LOG(ERROR) << "New NPU Subgraph failed.";
    return nullptr;
  }
  subgraph->set_inputs(lite::GetGraphInTensors(npu_ops));
  // The output of NPUGraph should be assigned to the corresponding NPUSubgraph
  auto subgraph_outputs = lite::GetGraphOutTensors(npu_ops);
  for (auto graph_output : this->outputs()) {
    for (auto subgraph_op : npu_ops) {
      auto subgraph_op_outputs = subgraph_op->outputs();
      if (find(subgraph_op_outputs.begin(), subgraph_op_outputs.end(), graph_output) != subgraph_op_outputs.end() &&
          find(subgraph_outputs.begin(), subgraph_outputs.end(), graph_output) == subgraph_outputs.end()) {
        subgraph_outputs.emplace_back(graph_output);
        break;
      }
    }
  }
  subgraph->set_outputs(subgraph_outputs);
  auto ret = subgraph->Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "NPU Subgraph Init failed.";
    return nullptr;
  }
  return subgraph;
}

kernel::Kernel *NPUGraph::CreateNPUTransposeKernel(NPUOp *op) {
  if (op->type() != schema::PrimitiveType_Transpose) {
    MS_LOG(ERROR) << "Check npu transpose op failed.";
    return nullptr;
  }
  auto transpose_op = static_cast<TransposeNPUOp *>(op);
  auto transpose_kernel = new (std::nothrow)
    TransposeNPUKernel(transpose_op->inputs(), transpose_op->outputs(), transpose_op->GetPerm(), transpose_op->name());
  if (transpose_kernel == nullptr) {
    MS_LOG(ERROR) << "New npu transpose kernel failed.";
    return nullptr;
  }
  return transpose_kernel;
}

int NPUGraph::Prepare() {
  for (int i = 0; i < all_kernels_.size(); i++) {
    auto ret = all_kernels_[i]->Prepare();
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "NPU Subgraph " << all_kernels_[i]->name() << " prepare failed.";
      return RET_ERROR;
    }
    for (auto output : all_kernels_[i]->outputs()) {
      if (find(outputs_.begin(), outputs_.end(), output) == outputs_.end()) {
        if (output.MutableData() == nullptr) {
          MS_LOG(ERROR) << "NPU Subgraph " << all_kernels_[i]->name() << " prepare malloc output tensor failed.";
          return RET_ERROR;
        }
      }
    }
  }
  return RET_OK;
}

int NPUGraph::Execute() {
  for (int i = 0; i < all_kernels_.size(); i++) {
    // 1. malloc graph output data
    for (auto output : all_kernels_[i]->outputs()) {
      if (find(outputs_.begin(), outputs_.end(), output) != outputs_.end()) {
        if (output.MutableData() == nullptr) {
          MS_LOG(ERROR) << "NPU Subgraph " << output.Name() << " execute malloc output tensor failed.";
          return RET_ERROR;
        }
      }
    }
    // 2. execute
    auto ret = all_kernels_[i]->Execute();
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "NPU Subgraph " << all_kernels_[i]->name() << " execute failed.";
      return RET_ERROR;
    }
  }
  return RET_OK;
}
}  // namespace mindspore
