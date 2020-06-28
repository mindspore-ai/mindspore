/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
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

#ifndef MINDSPORE_CCSRC_VM_TRANSFORM_H_
#define MINDSPORE_CCSRC_VM_TRANSFORM_H_

#include <string>
#include <memory>
#include <functional>
#include <utility>
#include <unordered_map>
#include <vector>

#include "vm/vm.h"
#include "ir/anf.h"
#include "operator/ops.h"
#include "vm/segment_runner.h"
#include "vm/backend.h"

// mindspore namespace is the top level namespace of MindSpore project.
// Other namespace should be a sub namespace of mindspore namespace in the ME project.
namespace mindspore {
extern const char kMsVm[];
extern const char kGeVm[];

// compile namespace
// A sub namespace in ME to support compile related definition.
namespace compile {
extern std::vector<PrimitivePtr> nonlinear_ops;
const std::vector<PrimitivePtr> &GetMsNonlinearOps();

using VmEvalFunc = std::function<BaseRef(const VectorRef &)>;
using VmEvalFuncPtr = std::shared_ptr<std::function<BaseRef(const VectorRef &)>>;

class CompileGraph {
 public:
  explicit CompileGraph(const BackendPtr &backend, const std::vector<PrimitivePtr> &cut_list = nonlinear_ops);

  ~CompileGraph() = default;

  InstSet Run(const FuncGraphPtr &func_graph);
  InstSet GenMultiGraphsSinkInst(const FuncGraphPtr &graph);
  bool IsCut(const AnfNodePtr &node);
  void Push(const AnfNodePtr &node);
  void Tie(const AnfNodePtr &n1, const AnfNodePtr &n2) { slots_[n2] = slots_[n1]; }
  void Ret(int nargs);
  void GenMultiGraphsRun(const FuncGraphPtr &graph);
  int Ref(const AnfNodePtr &node);
  VectorRef SplitNodes(const FuncGraphPtr &func_graph);

  void set_height(int h) {
    height_ = h;
    if (height_ > max_height_) {
      max_height_ = height_;
    }
  }

  void Reset() {
    height_ = 0;
    max_height_ = 0;
    slots_.clear();
    inst_.clear();
  }

 private:
  VectorRef SplitNodesWithTarget(const std::vector<AnfNodePtr> &input_nodes, const FuncGraphPtr &graph);
  void PushParameters(const FuncGraphPtr &func_graph);
  bool SplitGraph(const FuncGraphPtr &func_graph);
  int LinConvert(const FuncGraphPtr &func_graph, const AnfNodePtrList &node_list, const std::string &target = "");
  int InterpretNode(const FuncGraphPtr &func_graph, const CNodePtr &node);
  int AddCall(const FuncGraphPtr &graph, const CNodePtr &node);
  void AddSinkSwitch(const CNodePtr &node);
  void AddPadStack(int param_height);
  void AddTailCall(const AnfNodePtr &fn, size_t size);
  void AddPartial(const CNodePtr &node);
  void AddMakeTuple(const CNodePtr &node);
  void AddSwitch(const CNodePtr &node);
  void AddReturn(const CNodePtr &node);
  void AddPrimitive(const CNodePtr &node, const PrimitivePtr &prim);
  void AddInput(const AnfNodePtr &node);
  void AddExternal(const LinConvertResult &result);
  void AddInst(const Instruction &inst, const int &arg);
  void AddInst(const Instruction &inst, const ValuePtr &arg);
  void AddInst(const Instruction &inst, const VectorRef &args);

  BackendPtr backend_;
  LinkFuncType lin_convert_;
  bool is_gevm_convert_;
  int height_{0};
  int max_height_{0};
  std::vector<PrimitivePtr> cut_list_;
  std::unordered_map<AnfNodePtr, int> slots_;
  InstSet inst_;
};

using CompileGraphPtr = std::shared_ptr<CompileGraph>;

// CompileGraphs is used to Convert a graph cluster into instruction lists.
class CompileGraphs {
 public:
  explicit CompileGraphs(const BackendPtr &backend, const std::vector<PrimitivePtr> &cut_list = nonlinear_ops);

  ~CompileGraphs() = default;

  void Reset() {
    insts_.clear();
    mapping_.clear();
  }

  void Compile(const FuncGraphPtr &func_graph);
  FinalVMPtr Link(const FuncGraphPtr &func_graph);
  FinalVMPtr CompileAndLink(const FuncGraphPtr &func_graph);
  static bool ContainMixedTarget(const FuncGraphPtr &graph);

 private:
  InstSet insts_;
  std::unordered_map<FuncGraphPtr, int> mapping_;
  CompileGraphPtr transform_;
  BackendPtr backend_;
};

BackendPtr CreateBackend();

}  // namespace compile
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_VM_TRANSFORM_H_
