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

#ifndef PREDICT_SRC_GRAPH_EXECUTION_H_
#define PREDICT_SRC_GRAPH_EXECUTION_H_

#include <map>
#include <deque>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "common/mslog.h"
#include "src/graph.h"
#include "include/errorcode.h"
#include "schema/inner/ms_generated.h"
#include "src/operator/cpu/include/op_func_comm.h"
#include "src/node.h"

namespace mindspore {
namespace predict {
class GraphExecution {
 public:
  explicit GraphExecution(const Context &ctx);
  GraphExecution(const Context &ctx, Graph *staticGraph);
  virtual ~GraphExecution();

  virtual std::vector<Tensor *> GetInput();
  virtual int SetInputTensors(const std::vector<Tensor *> &inputs);

  virtual int Run(const std::vector<Tensor *> &inputs);

  virtual std::map<NODE_ID, std::vector<Tensor *>> GetAllOutput();
  virtual std::vector<Tensor *> GetOutput(const NODE_ID &nodeName);

 private:
  void ResetInputData();
  int MallocOutput();
  void FreeTensors(std::vector<Tensor *> *tensors);
  int TransInputDataToNc4hw4(const Tensor &src, Tensor *dst);
  int CopyOutputTensors(const std::vector<Tensor *> &refOutputs, std::vector<Tensor *> *outputs);
  void FreeOutputMap(std::map<NODE_ID, std::vector<Tensor *>> *map);
  void FreeAllTensors();

 protected:
  Graph *graph;
  const Context &_ctx;
  std::vector<Tensor *> inputTensors;
  std::vector<Tensor *> outputTensors;
  std::unordered_map<Node *, std::unordered_set<Node *>> depends;  // records the dependencies
  std::deque<Node *> readyQue;  // the nodes which can execute without any dependencies
};
}  // namespace predict
}  // namespace mindspore

#endif  // PREDICT_SRC_GRAPH_EXECUTION_H_
