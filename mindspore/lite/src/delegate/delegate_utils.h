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
#ifndef MINDSPORE_LITE_SRC_DELEGATE_DELEGATE_UTILS
#define MINDSPORE_LITE_SRC_DELEGATE_DELEGATE_UTILS
#include <vector>
#include "include/ms_tensor.h"
#include "include/api/delegate.h"
#include "src/common/log_adapter.h"
#include "include/errorcode.h"

namespace mindspore::lite {
bool IsSubGraphInputTensor(const std::vector<mindspore::MSTensor> &inputs, mindspore::MSTensor input);

template <typename T>
std::vector<mindspore::MSTensor> GetGraphInTensors(std::vector<T *> ops, std::vector<size_t> *input_index) {
  std::vector<mindspore::MSTensor> inputs;
  auto is_op_output = [&](mindspore::MSTensor tensor) -> bool {
    for (auto op : ops) {
      auto out_tensors = op->outputs();
      if (find(out_tensors.begin(), out_tensors.end(), tensor) != out_tensors.end()) {
        return true;
      }
    }
    return false;
  };

  for (auto op : ops) {
    for (auto in_tensor : op->inputs()) {
      if (in_tensor.Data() == nullptr && !is_op_output(in_tensor)) {
        // remove the repeated input.
        size_t idx = std::find(inputs.begin(), inputs.end(), in_tensor) - inputs.begin();
        if (idx == inputs.size()) {
          inputs.push_back(in_tensor);
        }
        if (input_index != nullptr) {
          input_index->push_back(idx);
        }
      }
    }
  }
  return inputs;
}

template <typename T>
std::vector<mindspore::MSTensor> GetGraphOutTensors(const std::vector<T *> &ops) {
  std::vector<mindspore::MSTensor> outputs;
  auto is_op_input = [&](const mindspore::MSTensor tensor) -> bool {
    for (auto op : ops) {
      auto in_tensors = op->inputs();
      if (find(in_tensors.begin(), in_tensors.end(), tensor) != in_tensors.end()) {
        return true;
      }
    }
    return false;
  };

  for (auto op : ops) {
    for (auto out_tensor : op->outputs()) {
      if (!is_op_input(out_tensor)) {
        outputs.push_back(out_tensor);
      }
    }
  }

  for (auto op : ops) {
    for (auto out_op : op->out_ops()) {
      if (find(ops.begin(), ops.end(), out_op) == ops.end()) {
        // visit the out op that is not in the subgraph
        for (auto tensor : op->outputs()) {
          if (find(out_op->inputs().begin(), out_op->inputs().end(), tensor) != out_op->inputs().end() &&
              find(outputs.begin(), outputs.end(), tensor) == outputs.end()) {
            // find the connected tensor
            outputs.push_back(tensor);
            break;
          }
        }
      }
    }
  }
  return outputs;
}

template <typename T>
std::vector<mindspore::MSTensor> GraphInTensors(const std::vector<T *> &ops, DelegateModel<schema::Primitive> *model,
                                                KernelIter from, KernelIter end) {
  auto in_tensors = GetGraphInTensors(ops, nullptr);
  std::vector<mindspore::MSTensor> all_in_tensors;
  for (auto op : ops) {
    for (auto in_tensor : op->inputs()) {
      if (in_tensor.Data() != nullptr && find(in_tensors.begin(), in_tensors.end(), in_tensor) == in_tensors.end()) {
        all_in_tensors.push_back(in_tensor);
      }
    }
  }

  for (auto iter = model->BeginKernelIterator(); iter != model->EndKernelIterator(); iter++) {
    if (iter >= from && iter <= end) {
      continue;
    }
    // The output of other kernels is the input of the current subgraph kernel.
    for (auto out_tensor : (*iter)->outputs()) {
      if (std::find(all_in_tensors.begin(), all_in_tensors.end(), out_tensor) != all_in_tensors.end()) {
        in_tensors.push_back(out_tensor);
      }
    }
  }
  return in_tensors;
}

template <typename T>
std::vector<mindspore::MSTensor> GraphOutTensors(const std::vector<T *> &ops, DelegateModel<schema::Primitive> *model,
                                                 KernelIter from, KernelIter end) {
  auto out_tensors = GetGraphOutTensors(ops);
  std::vector<mindspore::MSTensor> all_out_tensors;
  for (auto op : ops) {
    for (auto out_tensor : op->outputs()) {
      if (find(out_tensors.begin(), out_tensors.end(), out_tensor) == out_tensors.end()) {
        all_out_tensors.push_back(out_tensor);
      }
    }
  }

  for (auto iter = model->BeginKernelIterator(); iter != model->EndKernelIterator(); iter++) {
    if (iter >= from && iter <= end) {
      continue;
    }
    // The input of other kernels is the output of the current subgraph kernel.
    for (auto in_tensor : (*iter)->inputs()) {
      if (find(all_out_tensors.begin(), all_out_tensors.end(), in_tensor) != all_out_tensors.end()) {
        out_tensors.push_back(in_tensor);
      }
    }
  }
  return out_tensors;
}

template <typename T>
std::vector<T *> FindPreOps(T *cur_op, std::vector<T *> all_ops) {
  std::vector<T *> in_ops;
  for (auto in_tensor : cur_op->inputs()) {
    for (auto op : all_ops) {
      if (find(op->outputs().begin(), op->outputs().end(), in_tensor) != op->outputs().end()) {
        in_ops.push_back(op);
      }
    }
  }
  return in_ops;
}

template <typename T>
std::vector<T *> FindNextOps(T *cur_op, std::vector<T *> all_ops) {
  std::vector<T *> out_ops;
  for (auto out_tensor : cur_op->outputs()) {
    for (auto op : all_ops) {
      if (find(op->inputs().begin(), op->inputs().end(), out_tensor) != op->inputs().end()) {
        out_ops.push_back(op);
      }
    }
  }
  return out_ops;
}

template <typename T>
void FindPreNextOps(std::vector<T *> all_ops) {
  for (auto op : all_ops) {
    auto in_ops = FindPreOps(op, all_ops);
    op->set_in_ops(in_ops);
    auto out_ops = FindNextOps(op, all_ops);
    op->set_out_ops(out_ops);
  }
}

template <typename T>
int GetGraphInOutOps(const std::vector<mindspore::MSTensor> &inputs, const std::vector<mindspore::MSTensor> &outputs,
                     std::vector<T *> *in_ops, std::vector<T *> *out_ops, const std::vector<T *> &all_ops) {
  for (auto in_tensor : inputs) {
    for (auto op : all_ops) {
      if (find(op->inputs().begin(), op->inputs().end(), in_tensor) != op->inputs().end() &&
          find(in_ops->begin(), in_ops->end(), op) == in_ops->end()) {
        in_ops->push_back(op);
      }
    }
  }
  if (in_ops->empty()) {
    MS_LOG(ERROR) << "Can't find the input ops for npu sub graph.";
    return RET_ERROR;
  }

  for (auto out_tensor : outputs) {
    for (auto op : all_ops) {
      if (find(op->outputs().begin(), op->outputs().end(), out_tensor) != op->outputs().end() &&
          find(out_ops->begin(), out_ops->end(), op) == out_ops->end()) {
        out_ops->push_back(op);
      }
    }
  }
  if (out_ops->empty()) {
    MS_LOG(ERROR) << "Can't find the output ops for npu sub graph.";
    return RET_ERROR;
  }
  return RET_OK;
}
}  // namespace mindspore::lite

#endif  // MINDSPORE_LITE_SRC_DELEGATE_DELEGATE_UTILS
