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

#ifndef MINDSPORE_PI_JIT_FUNC_INLINER_H_
#define MINDSPORE_PI_JIT_FUNC_INLINER_H_

#include <map>
#include <memory>
#include <string>
#include "pipeline/jit/pi/graph_compiler/pi_ir/ir_mutator.h"
#include "pipeline/jit/pi/graph_compiler/pi_ir/ir_visitor.h"
#include "utils/convert_utils_base.h"

namespace mindspore {
namespace pijit {
class FuncInlineDetector : public ir::IRVisitor {
 public:
  explicit FuncInlineDetector(const ir::FunctionNodePtr &func) : func_(func), index_(0), cur_root_node_(nullptr) {}
  virtual ~FuncInlineDetector() = default;
  void Run();

  void Visit_(const ir::FunctionNodePtr &node) override;
  void Visit_(const ir::CallNodePtr &node) override;
  size_t GetRootNodeIndex(const ir::CallNodePtr &node) const;
  const ir::NodePtr &GetRootNode(const ir::CallNodePtr &node) const;

 private:
  bool CanBeInlined(const ir::NodePtr &node) const;
  void EvolvingFunction(const ir::FunctionNodePtr &func_node, const ir::NodePtrList args) const;

  const ir::FunctionNodePtr func_;
  size_t index_;
  ir::NodePtr cur_root_node_;
  std::map<ir::CallNodePtr, size_t> node_2_index_;
  std::map<ir::CallNodePtr, ir::NodePtr> node_2_root_;
};

using FuncInlineDetectorPtr = std::shared_ptr<FuncInlineDetector>;

class FuncLocalVarRenamer : public ir::IRVisitor {
 public:
  explicit FuncLocalVarRenamer(const ir::FunctionNodePtr &func) : func_(func) {}
  virtual ~FuncLocalVarRenamer() = default;
  void Run();

  void Visit_(const ir::ParameterPtr &node) override;
  void Visit_(const ir::ValuePtr &node) override;

 private:
  const ir::FunctionNodePtr func_;
};

using FuncLocalVarRenamerPtr = std::shared_ptr<FuncLocalVarRenamer>;

class FuncParameterEliminator : public ir::IRMutator {
 public:
  explicit FuncParameterEliminator(const ir::FunctionNodePtr &func, const ir::NodePtrList &args)
      : func_(func), args_(args) {}
  virtual ~FuncParameterEliminator() = default;
  void Run();

  ir::NodePtr Mutate_(const ir::ParameterPtr &node) override;
  ir::NodePtr Mutate_(const ir::LoadValueNodePtr &node) override;
  ir::NodePtr Mutate_(const ir::StoreNodePtr &node) override;

 private:
  const ir::FunctionNodePtr func_;
  const ir::NodePtrList args_;
  std::map<std::string, ir::NodePtr> name_2_node_;
};

using FuncParameterEliminatorPtr = std::shared_ptr<FuncParameterEliminator>;

// FuncInliner to convert ir graph to function graph
class FuncInliner : public ir::IRMutator {
 public:
  explicit FuncInliner(const ir::FunctionNodePtr &func)
      : func_(func), detector_(std::make_shared<FuncInlineDetector>(func)), inserted_nodes_cnt_(0) {}
  virtual ~FuncInliner() = default;
  void Run();
  void InsertSubFunction();

  // overloadable Mutate function.
  ir::NodePtr Mutate_(const ir::CallNodePtr &node) override;

 private:
  const ir::FunctionNodePtr func_;
  const FuncInlineDetectorPtr detector_;
  size_t inserted_nodes_cnt_;
  std::map<size_t, ir::FunctionNodePtr> index_2_function_;
};

using FuncInlinerPtr = std::shared_ptr<FuncInliner>;
}  // namespace pijit
}  // namespace mindspore

#endif  // MINDSPORE_PI_JIT_FUNC_INLINER_H_
