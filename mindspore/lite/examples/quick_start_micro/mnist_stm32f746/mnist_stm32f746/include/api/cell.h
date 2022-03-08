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

class MS_API ParameterCell final : public Cell<ParameterCell> {
 public:
  ParameterCell() = default;
  ~ParameterCell() override = default;

  ParameterCell(const ParameterCell &);
  ParameterCell &operator=(const ParameterCell &);

  ParameterCell(ParameterCell &&);
  ParameterCell &operator=(ParameterCell &&);

  explicit ParameterCell(const MSTensor &);
  ParameterCell &operator=(const MSTensor &);

  explicit ParameterCell(MSTensor &&);
  ParameterCell &operator=(MSTensor &&);

  MSTensor GetTensor() const { return tensor_; }

 private:
  MSTensor tensor_;
};

class MS_API OpCellBase : public CellBase {
 public:
  explicit OpCellBase(const std::string &name) : name_(name) {}
  ~OpCellBase() override = default;
  const std::string &GetOpType() const { return name_; }

 protected:
  std::string name_;
};

template <class T>
class MS_API OpCell : public OpCellBase, public std::enable_shared_from_this<T> {
 public:
  explicit OpCell(const std::string &name) : OpCellBase(name) {}
  ~OpCell() override = default;
  std::shared_ptr<CellBase> Clone() const override { return std::make_shared<T>(static_cast<const T &>(*this)); }
};

class MS_API GraphCell final : public Cell<GraphCell> {
 public:
  class GraphImpl;

  GraphCell() = default;
  ~GraphCell() override = default;

  explicit GraphCell(const Graph &);
  explicit GraphCell(Graph &&);
  explicit GraphCell(const std::shared_ptr<Graph> &);

  const std::shared_ptr<Graph> &GetGraph() const { return graph_; }
  Status Run(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs) override;
  std::vector<MSTensor> GetInputs();
  std::vector<MSTensor> GetOutputs();

 private:
  friend class Model;
  friend class ModelImpl;
  Status Load(uint32_t device_id);

  std::shared_ptr<Graph> graph_;
  std::shared_ptr<GraphImpl> executor_;
};

class MS_API InputAndOutput {
 public:
  InputAndOutput();
  ~InputAndOutput() = default;

  // no explicit
  InputAndOutput(const MSTensor &);  // NOLINT(runtime/explicit)
  InputAndOutput(MSTensor &&);       // NOLINT(runtime/explicit)

  InputAndOutput(const std::shared_ptr<CellBase> &, const std::vector<InputAndOutput> &, int32_t index);

  int32_t GetIndex() const { return index_; }
  void SetIndex(int32_t index) { index_ = index; }

 private:
  std::shared_ptr<CellBase> cell_;
  std::vector<InputAndOutput> prev_;
  int32_t index_;
};
}  // namespace mindspore
#endif  // MINDSPORE_INCLUDE_API_CELL_H
