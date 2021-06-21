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
#include "src/delegate/npu/npu_graph_utils.h"
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
  for (auto *tensor : insert_tensors_) {
    delete tensor;
  }
}

void NPUGraph::set_input(tensor::MSTensor *in_tensor, int index) {
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

void NPUGraph::set_output(tensor::MSTensor *out_tensor, int index) {
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
  for (auto op : npu_ops_) {
    is_visited[op] = false;
  }

  while (npu_ops_.size() > 0) {
    auto head_op_iter = std::find_if(npu_ops_.begin(), npu_ops_.end(), [&](const NPUOp *op) {
      if (is_visited[op]) {
        return false;
      }
      return true;
    });
    if (head_op_iter == npu_ops_.end()) {
      break;
    }
    auto head_op = *head_op_iter;
    if (head_op->type() != schema::PrimitiveType_Transpose) {
      // If npu_kernel does not equal nullptr, this kernel can be supported by delegate
      auto npu_ops = FindSubgraphOps(head_op, &is_visited);
      auto subgraph_kernel = CreateNPUSubgraphKernel(npu_ops);
      if (subgraph_kernel == nullptr) {
        MS_LOG(DEBUG) << "Create NPU subgraph kernel failed.";
        return RET_ERROR;
      }
      all_kernels_.push_back(subgraph_kernel);
    } else {
      auto transpose_kernel = CreateNPUTransposeKernel(head_op);
      if (transpose_kernel == nullptr) {
        MS_LOG(DEBUG) << "New NPU transpose kernel failed.";
        return RET_ERROR;
      }
      all_kernels_.push_back(transpose_kernel);
      is_visited[head_op] = true;
    }
  }
  return RET_OK;
}

std::vector<NPUOp *> NPUGraph::FindPreOps(NPUOp *cur_op) {
  std::vector<NPUOp *> in_ops;
  for (auto in_tensor : cur_op->inputs()) {
    for (auto op : npu_ops_) {
      if (find(op->outputs().begin(), op->outputs().end(), in_tensor) != op->outputs().end()) {
        in_ops.push_back(op);
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
        out_ops.push_back(op);
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

std::vector<NPUOp *> NPUGraph::FindSubgraphOps(NPUOp *head_op, std::map<const NPUOp *, bool> *is_visited) {
  std::vector<NPUOp *> subgraph_ops;
  subgraph_ops.push_back(head_op);
  (*is_visited)[head_op] = true;
  std::queue<NPUOp *> op_queue;
  op_queue.emplace(head_op);
  while (!op_queue.empty()) {
    auto cur_op = op_queue.front();
    op_queue.pop();
    auto out_ops = cur_op->out_ops();
    for (auto out_op : out_ops) {
      if ((*is_visited)[out_op] == true) {
        continue;
      }
      auto input_ready = std::all_of(out_op->in_ops().begin(), out_op->in_ops().end(),
                                     [&](NPUOp *in_op) { return (*is_visited)[in_op] == true; });
      if (input_ready && out_op->type() != schema::PrimitiveType_Transpose) {
        subgraph_ops.push_back(out_op);
        (*is_visited)[out_op] = true;
        op_queue.push(out_op);
      }
    }
  }
  return subgraph_ops;
}

kernel::Kernel *NPUGraph::CreateNPUSubgraphKernel(std::vector<NPUOp *> npu_ops) {
  auto subgraph = new (std::nothrow) NPUSubGraph(npu_ops, npu_manager_);
  if (subgraph == nullptr) {
    MS_LOG(ERROR) << "New NPU Subgraph failed.";
    return nullptr;
  }
  subgraph->set_inputs(NPUGraphUtils::GetGraphInTensors(npu_ops));
  subgraph->set_outputs(NPUGraphUtils::GetGraphOutTensors(npu_ops));
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
        output->MutableData();
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
        output->MutableData();
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
