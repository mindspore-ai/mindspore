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
#ifndef MINDSPORE_INCLUDE_API_CELL_H
#define MINDSPORE_INCLUDE_API_CELL_H
#include <string>
#include <vector>
#include <map>
#include <memory>
#include "include/api/status.h"
#include "include/api/types.h"
#include "include/api/graph.h"

namespace mindspore {
class InputAndOutput;
class Context;
using Input = InputAndOutput;
using Output = InputAndOutput;

class MS_API CellBase {
 public:
  CellBase() = default;
  virtual ~CellBase() = default;
  virtual std::vector<Output> Construct(const std::vector<Input> &inputs) { return {}; }
  virtual std::shared_ptr<CellBase> Clone() const = 0;
  virtual Status Run(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs) { return kSuccess; }
  std::vector<Output> operator()(const std::vector<Input> &inputs) const;
};

template <class T>
class MS_API Cell : public CellBase {
 public:
  virtual ~Cell() = default;
  std::shared_ptr<CellBase> Clone() const override { return std::make_shared<T>(static_cast<const T &>(*this)); }
};

class MS_API GraphCell final : public Cell<GraphCell> {
 public:
  class GraphImpl;

  GraphCell() = default;
  ~GraphCell() override = default;

  explicit GraphCell(const Graph &graph);
  explicit GraphCell(Graph &&graph);
  explicit GraphCell(const std::shared_ptr<Graph> &graph);

  void SetContext(const std::shared_ptr<Context> &context);
  const std::shared_ptr<Graph> &GetGraph() const { return graph_; }
  Status Run(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs) override;
  std::vector<MSTensor> GetInputs();
  std::vector<MSTensor> GetOutputs();
  Status Load(uint32_t device_id);

 private:
  friend class Model;

  std::shared_ptr<Graph> graph_;
  std::shared_ptr<GraphImpl> executor_;
};

class MS_API InputAndOutput {
 public:
  InputAndOutput();
  ~InputAndOutput() = default;

  InputAndOutput(const std::shared_ptr<CellBase> &cell, const std::vector<InputAndOutput> &prev, int32_t index);

  int32_t GetIndex() const { return index_; }
  void SetIndex(int32_t index) { index_ = index; }

 private:
  std::shared_ptr<CellBase> cell_;
  std::vector<InputAndOutput> prev_;
  int32_t index_ = 0;
};
}  // namespace mindspore
#endif  // MINDSPORE_INCLUDE_API_CELL_H
