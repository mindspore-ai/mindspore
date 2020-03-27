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

#ifndef PREDICT_SRC_GRAPH_H_
#define PREDICT_SRC_GRAPH_H_

#include <map>
#include <deque>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "common/utils.h"
#include "common/graph_util.h"
#include "include/tensor.h"
#include "src/node.h"

#define MSPREDICT_API __attribute__((visibility("default")))

namespace mindspore {
namespace predict {
class SubGraph {
 public:
  SubGraph();
  ~SubGraph();
  static SubGraph *CreateSubGraph(const SubGraphDef &subGraphDef, const Context &ctx);
  int Build(const SubGraphDef &subGraphDef, const Context &ctx);
  bool IsInputIndex(uint32_t i);
  bool IsOutputIndex(uint32_t i);

  const std::vector<uint32_t> *GetInputIndices() const;
  const std::vector<uint32_t> *GetOutputIndices() const;

  std::vector<Tensor *> GetInputs();
  std::vector<Tensor *> GetOutputs();
  std::map<NODE_ID, std::vector<Tensor *>> &GetOutputsMap();
  void FreeAllTensors();

  Node *GetNode(const NODE_ID &id);

  std::unordered_map<Node *, std::unordered_set<Node *>> GetDepends();

 private:
  int ConverterIndex(const flatbuffers::Vector<uint32_t> &srcIndex, std::vector<uint32_t> *dstIndex);

  int ConverterAllTensor(const flatbuffers::Vector<flatbuffers::Offset<TensorDef>> &srcTensors);

  int ConverterNodes(const flatbuffers::Vector<flatbuffers::Offset<NodeDef>> &opDefs, const Context &ctx);

  int ConverterEdges(const SubGraphDef &subGraphDef);

  int InitOutputsMap();

 protected:
  std::unordered_map<NODE_ID, Node *> nodes;
  std::vector<uint32_t> inputIndices;
  std::vector<uint32_t> outputIndices;
  std::vector<Tensor *> allTensors;  // weight + input + output
  std::map<NODE_ID, std::vector<Tensor *>> outputsMap;
};

class MSPREDICT_API Graph {
 public:
  Graph();
  ~Graph();
  static Graph *CreateFromBuf(const char *buf, size_t size, const Context &ctx);

  std::vector<Tensor *> GetInputs();
  std::vector<Tensor *> GetOutputs();

  std::map<NODE_ID, std::vector<Tensor *>> &GetOutputsMap();

  void FreeAllTensors();

  int Build(const GraphDef &def, const Context &ctx);
  std::vector<SubGraph *> *Subgraphs();

 protected:
  friend class GraphExecution;

  std::vector<SubGraph *> subgraphs;
  std::unordered_map<Node *, std::unordered_set<Node *>> depends;  // records the dependencies
  std::deque<Node *> readyQue;  // the nodes which can execute without any dependencies
};
}  // namespace predict
}  // namespace mindspore

#endif  // PREDICT_SRC_GRAPH_H_
