/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_COMMON_DRAW_GRAPHVIZ_GRAPH_H_
#define MINDSPORE_LITE_SRC_COMMON_DRAW_GRAPHVIZ_GRAPH_H_

#include <utility>
#include <vector>
#include <string>
#include <unordered_map>
#include "src/common/log_adapter.h"
#include "src/tensor.h"
#include "include/errorcode.h"

namespace mindspore::lite {
constexpr int kNodeTypeCNode = 0;
constexpr int kNodeTypeInput = 1;
constexpr int kNodeTypeOutput = 2;
constexpr int kNodeTypeWeight = 3;
class GVNode;

class Edge {
 public:
  Edge(std::string name, const GVNode *from, const size_t &from_port, std::string info)
      : name_(std::move(name)), from_(from), from_port_(from_port), info_(std::move(info)) {}

  std::string From() const;
  std::string name() const;
  void AppendOutput(const GVNode *to, size_t port);
  std::string Code() const;

 private:
  std::string name_;
  const GVNode *from_{nullptr};
  const size_t from_port_{};
  std::vector<const GVNode *> tos_{};
  std::vector<size_t> to_ports_{};
  std::string info_;
};

class GVNode {
 public:
  static GVNode *CreateCNode(const std::string &id, const std::string &label, size_t input_size,
                             const std::vector<std::string> &output_names, const std::vector<std::string> &output_infos,
                             bool highlight = false);
  static GVNode *CreateInput(const std::string &id, const std::vector<std::string> &output_names,
                             const std::vector<std::string> &output_infos, bool highlight = false);
  static GVNode *CreateOutput(const std::string &id, size_t input_size, bool highlight = false);
  static GVNode *CreateWeight(const std::string &id, const std::string &label,
                              const std::vector<std::string> &output_names,
                              const std::vector<std::string> &output_infos, bool highlight = false);
  virtual ~GVNode();

  int type() const { return this->type_; }
  std::string prefix() const { return this->prefix_; }
  std::string name() const { return this->id_; }
  size_t input_size() const { return input_size_; }
  size_t output_size() const { return output_size_; }
  const std::vector<Edge *> &inputs() const { return inputs_; }
  const std::vector<Edge *> &outputs() const { return outputs_; }
  void AppendInput(Edge *edge) { this->inputs_.emplace_back(edge); }
  std::string Code() const;

 protected:
  GVNode(std::string id, std::string label, int type, size_t input_size, size_t output_size, bool highlight = false)
      : id_(std::move(id)),
        label_(std::move(label)),
        type_(type),
        input_size_(input_size),
        output_size_(output_size),
        highlight_(highlight) {}
  void Init(const std::vector<std::string> &output_names, const std::vector<std::string> &output_infos);
  size_t FindCols() const;

 private:
  std::string id_;
  std::string label_;
  int type_;
  std::string prefix_;
  std::string color_ = "white";
  size_t input_size_{0};
  size_t output_size_{0};
  std::string shape_;
  bool highlight_{false};
  std::vector<Edge *> inputs_{};
  std::vector<Edge *> outputs_{};
};

class GVGraph {
 public:
  explicit GVGraph(std::string name) : name_{std::move(name)} {};
  virtual ~GVGraph();

  void AppendNode(GVNode *node);
  int Link(const std::string &from_name, size_t from_port, const std::string &to_name, size_t to_port);
  std::string Code() const;

 private:
  std::string name_;
  std::vector<GVNode *> nodes_;
  std::unordered_map<std::string, GVNode *> node_map_;
};
}  // namespace mindspore::lite

#endif  // MINDSPORE_LITE_SRC_COMMON_DRAW_GRAPHVIZ_GRAPH_H_
