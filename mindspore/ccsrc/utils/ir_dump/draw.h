/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_UTILS_IR_DUMP_DRAW_H_
#define MINDSPORE_CCSRC_UTILS_IR_DUMP_DRAW_H_

#include <fstream>
#include <memory>
#include <string>
#include "ir/anf.h"
#include "utils/any.h"
#include "utils/file_utils.h"
#include "base/base.h"
#include "include/utils/visible.h"

namespace mindspore {
namespace draw {

class Graphviz {
 public:
  Graphviz(const std::string &name, const std::string &filename) : name_(name), filename_(filename), fout_(filename_) {}

  explicit Graphviz(const std::string &name) : name_(name) {}

  virtual ~Graphviz() {}

  virtual void Start() {}
  virtual void End() {}

  virtual std::string Shape(const AnfNodePtr &node);
  std::string Color(const AnfNodePtr &node) const;
  std::ostringstream &buffer() { return buffer_; }
  std::ostringstream buffer_;

 protected:
  std::string name_;
  std::string filename_;
  std::ofstream fout_;
};

class BaseDigraph : public Graphviz {
 public:
  BaseDigraph(const std::string &name, const std::string &filename) : Graphviz(name, filename) {}
  explicit BaseDigraph(const std::string &name) : Graphviz(name) {}
  ~BaseDigraph() override = default;

  virtual void Node(const AnfNodePtr &node, int id) = 0;
  virtual void Edge(const AnfNodePtr &start, const AnfNodePtr &end, int idx, int id_start) = 0;

  void Start() override;
  void End() override;
  virtual void Edge(const AnfNodePtr &start, const FuncGraphPtr &end, int id_start);
  void FuncGraphParameters(const FuncGraphPtr &key);
  void SubGraph(const FuncGraphPtr &key, const std::shared_ptr<BaseDigraph> &gsub);

  const std::string &name() const { return name_; }

 protected:
  void Head(const AnfNodePtr &node, int id = 0);
  void Tail(const AnfNodePtr &node, int idx, int id = 0);
  void Tail(const FuncGraphPtr &func_graph);
};

class Digraph : public BaseDigraph {
 public:
  Digraph(const std::string &name, const std::string &filename) : BaseDigraph(name, filename) {}
  explicit Digraph(const std::string &name) : BaseDigraph(name) {}
  ~Digraph() override;

  void Node(const AnfNodePtr &node, int id) override;
  void Edge(const AnfNodePtr &start, const AnfNodePtr &end, int idx, int id_start) override;
};

class ModelDigraph : public BaseDigraph {
 public:
  ModelDigraph(const std::string &name, const std::string &filename) : BaseDigraph(name, filename) {}
  explicit ModelDigraph(const std::string &name) : BaseDigraph(name) {}
  ~ModelDigraph() override;

  std::string Shape(const AnfNodePtr &node) override;
  void Node(const AnfNodePtr &node, int id) override;
  void Edge(const AnfNodePtr &start, const AnfNodePtr &end, int idx, int id_start) override;
};

// API to draw
COMMON_EXPORT void Draw(const std::string &filename, const FuncGraphPtr &func_graph);
COMMON_EXPORT void DrawUserFuncGraph(const std::string &filename, const FuncGraphPtr &func_graph);

}  // namespace draw
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_UTILS_IR_DUMP_DRAW_H_
