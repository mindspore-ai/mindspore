/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "src/graph_execution.h"
#include <utility>
#include <vector>
#include <memory>

namespace mindspore {
namespace predict {
GraphExecution::GraphExecution(const Context &ctx) : graph(nullptr), _ctx(ctx) {}
GraphExecution::GraphExecution(const Context &ctx, Graph *staticGraph) : _ctx(ctx) {
  graph = staticGraph;
  if (graph != nullptr) {
    depends = graph->depends;
    readyQue = graph->readyQue;
    outputTensors = graph->GetOutputs();
    inputTensors = graph->GetInputs();
  }
}

GraphExecution::~GraphExecution() = default;

int GraphExecution::TransInputDataToNc4hw4(const Tensor &src, Tensor *dst) {
  MS_ASSERT(dst != nullptr);
  if (dst->GetData() == nullptr) {
    auto ret = dst->MallocData(nullptr, MSConst_WEIGHT_REFCOUNT);
    if (ret != RET_OK) {
      MS_LOGE("Malloc inputTensors failed: %d", ret);
      return ret;
    }
  }
  auto ret = NchwToNc4hw4(&src, dst);
  if (ret != RET_OK) {
    MS_LOGE("NchwToNc4hw4 failed");
    return ret;
  }
  return RET_OK;
}

int GraphExecution::SetInputTensors(const std::vector<Tensor *> &inputs) {
  size_t num = inputs.size();
  if (num != inputTensors.size()) {
    MS_LOGE("input num %zu != model input num %zu", num, inputTensors.size());
    return RET_INPUT_TENSOR_ERROR;
  }

  for (size_t i = 0; i < num; i++) {
    MS_ASSERT(inputs[i] != nullptr);
    // The input Tensor desc must be equivalent with the model tensor
    if (inputs[i]->GetData() == nullptr) {
      MS_LOGE("input tensor data is null!");
      return RET_INPUT_TENSOR_ERROR;
    }
    if (inputTensors[i] == nullptr) {
      MS_LOGE("inputTensors[%zu] is nullptr", i);
      return RET_ERROR;
    }

    if (!inputs[i]->CompareShape(*inputTensors[i])) {
      MS_LOGE("tensor shape in graph and executor are different!");
      return RET_INPUT_TENSOR_ERROR;
    }

    if (inputs[i]->GetDataType() != inputTensors[i]->GetDataType()) {
      MS_LOGE("tensor datatype in graph and executor are different!");
      return RET_INPUT_TENSOR_ERROR;
    }

    if (inputs[i]->GetFormat() != Format_NCHW) {
      MS_LOGE("input format not support. only nchw is supported now");
      return RET_INPUT_TENSOR_ERROR;
    }

    if (inputs[i]->GetFormat() == inputTensors[i]->GetFormat()) {
      auto data = inputs[i]->GetData();
      if (data == nullptr) {
        MS_LOGE("data of input tensor is null!");
        return RET_INPUT_TENSOR_ERROR;
      }
      inputTensors[i]->SetData(data);
    } else if (inputTensors[i]->GetFormat() == Format_NC4HW4) {
      auto ret = TransInputDataToNc4hw4(*inputs[i], inputTensors[i]);
      if (ret != RET_OK) {
        MS_LOGE("TransInputDataToNc4hw4 failed");
        return ret;
      }
    } else {
      MS_LOGE("graphDef inputTensors format is invalid: %d", inputTensors[i]->GetFormat());
      return RET_ERROR;
    }
  }
  return RET_OK;
}

int GraphExecution::MallocOutput() {
  for (auto tensor : outputTensors) {
    auto ret = tensor->MallocData();
    if (ret != RET_OK) {
      MS_LOGE("malloc output data failed");
      return RET_ERROR;
    }
  }
  return RET_OK;
}

void GraphExecution::FreeTensors(std::vector<Tensor *> *tensors) {
  for (auto &tensor : (*tensors)) {
    delete tensor;
  }
  tensors->clear();
}

void GraphExecution::FreeOutputMap(std::map<NODE_ID, std::vector<Tensor *>> *map) {
  MS_ASSERT(map != nullptr);
  for (auto &m : *map) {
    FreeTensors(&(m.second));
  }
  map->clear();
}

int GraphExecution::CopyOutputTensors(const std::vector<Tensor *> &refOutputs, std::vector<Tensor *> *outputs) {
  for (auto tensor : refOutputs) {
    if (tensor == nullptr) {
      MS_LOGE("tensor in refOutputs is nullptr");
      return RET_INPUT_TENSOR_ERROR;
    }
    std::unique_ptr<Tensor> t(new Tensor(*tensor));
    if (t == nullptr) {
      MS_LOGE("new Tensor failed.");
      if (outputs != nullptr) {
        FreeTensors(outputs);
      }
      return RET_ERROR;
    }

    if (tensor->GetFormat() == Format_NC4HW4) {
      t->SetFormat(Format_NCHW);
      auto ret = t->MallocData();
      if (ret != RET_OK) {
        MS_LOGE("malloc data failed.")
        FreeTensors(outputs);
        return ret;
      }

      ret = Nc4hw4ToNchw(tensor, t.get());
      if (ret != RET_OK) {
        MS_LOGE("Nc4hw4ToNchw failed");
        return ret;
      }
      tensor->FreeData();
    } else {
      t->SetData(tensor->GetData());
      tensor->SetData(nullptr);
    }
    outputs->push_back(t.release());
  }
  return RET_OK;
}

std::map<NODE_ID, std::vector<Tensor *>> GraphExecution::GetAllOutput() {
  std::map<NODE_ID, std::vector<Tensor *>> outputs{};
  for (auto &outputNode : graph->GetOutputsMap()) {
    std::vector<Tensor *> outputNodeTensors{};
    auto ret = this->CopyOutputTensors(outputNode.second, &outputNodeTensors);
    if (ret != RET_OK) {
      MS_LOGE("copy output failed.");
      FreeOutputMap(&outputs);
      return outputs;
    }
    outputs.emplace(std::pair<NODE_ID, std::vector<Tensor *>>(outputNode.first, outputNodeTensors));
  }
  return outputs;
}

std::vector<Tensor *> GraphExecution::GetOutput(const NODE_ID &nodeName) {
  std::vector<Tensor *> outputNodeTensors{};
  auto iter = graph->GetOutputsMap().find(nodeName);
  if (iter == graph->GetOutputsMap().end()) {
    MS_LOGE("node name is not in output.");
    return outputNodeTensors;
  }
  auto ret = this->CopyOutputTensors(iter->second, &outputNodeTensors);
  if (ret != RET_OK) {
    MS_LOGE("copy output failed.");
  }
  return outputNodeTensors;
}

std::vector<Tensor *> GraphExecution::GetInput() {
  std::vector<Tensor *> inputs{};
  for (auto refInput : graph->GetInputs()) {
    if (refInput == nullptr) {
      MS_LOGE("tensor from graph->GetInputs() is nullptr");
      return inputs;
    }
    std::unique_ptr<Tensor> t(new Tensor(refInput->GetDataType(), refInput->GetDims(), Format_NCHW, nullptr));
    if (t == nullptr) {
      MS_LOGE("new Tensor failed.")
      FreeTensors(&inputs);
      return inputs;
    }
    inputs.push_back(t.release());
  }
  return inputs;
}

void GraphExecution::ResetInputData() {
  for (auto tensor : inputTensors) {
    if (tensor == nullptr) {
      MS_LOGW("tensor in inputTensors is nullptr");
      continue;
    }
    if (tensor->GetFormat() == Format_NC4HW4) {
      if (tensor->GetData() != nullptr) {
        free(tensor->GetData());
        tensor->SetData(nullptr);
      }
      continue;
    }
    tensor->SetData(nullptr);
  }
}

void GraphExecution::FreeAllTensors() { graph->FreeAllTensors(); }

int GraphExecution::Run(const std::vector<Tensor *> &inputs) {
  if (inputs.empty()) {
    MS_LOGE("input is empty");
    return RET_ERROR;
  }

  int ret;

  if (readyQue.empty()) {
    MS_LOGE("readyQue is empty");
    return RET_ERROR;
  }

  ret = SetInputTensors(inputs);
  if (ret != RET_OK) {
    MS_LOGE("SetInputTensors failed: %d", ret);
    ResetInputData();
    return ret;
  }
  ret = MallocOutput();
  if (ret != RET_OK) {
    MS_LOGE("MallocOutput failed: %d", ret);
    ResetInputData();
    return ret;
  }

  while (!readyQue.empty()) {
    auto *node = readyQue.front();
    readyQue.pop_front();

    ret = node->Run(_ctx);
    if (ret != RET_OK) {
      MS_LOGE("node (%s) failed to run op (%s). error code:%d", node->ID().c_str(), node->Type().c_str(), ret);
      ResetInputData();
      FreeAllTensors();
      return ret;
    }

    for (auto outNode : node->GetAllOutEdges()) {
      auto nodeDepend = depends.find(outNode);
      nodeDepend->second.erase(node);
      if (nodeDepend->second.empty()) {
        depends.erase(nodeDepend);
        readyQue.push_back(outNode);
      }
    }
  }

  ResetInputData();

  return RET_OK;
}
}  // namespace predict
}  // namespace mindspore
