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

#include "src/delegate/npu/npu_graph_utils.h"
namespace mindspore {
std::vector<mindspore::MSTensor> NPUGraphUtils::GetGraphInTensors(std::vector<NPUOp *> ops) {
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
        inputs.push_back(in_tensor);
      }
    }
  }
  return inputs;
}

std::vector<mindspore::MSTensor> NPUGraphUtils::GetGraphOutTensors(std::vector<NPUOp *> ops) {
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
}  // namespace mindspore
