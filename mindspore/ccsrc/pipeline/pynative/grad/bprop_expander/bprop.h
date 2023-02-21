/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_BPROP_EXPANDER_BPROP_H_
#define MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_BPROP_EXPANDER_BPROP_H_

#include <map>
#include <vector>
#include <utility>
#include <string>
#include <memory>
#include "ir/anf.h"
#include "pipeline/pynative/grad/bprop_expander/bprop_irbuilder.h"

namespace mindspore {
namespace expander {
namespace bprop {
using UserType = mindspore::HashMap<AnfNodePtr, std::vector<std::pair<std::weak_ptr<CNode>, int>>>;
class BpropExpander {
 public:
  BpropExpander() {}
  BpropExpander(CNodePtrList *outputs, UserType *users) : outputs_(outputs), users_(users) {}
  ~BpropExpander() = default;
  bool Run(const CNodePtr &cnode);
  const std::vector<size_t> &GetUnusedInputs(const CNodePtr &cnode) const;

 protected:
  bool RunBprop(const CNodePtr &cnode);
  virtual void ExtractInputs(const CNodePtr &cnode, const BpropIRBuilder *ir_builder);
  virtual std::unique_ptr<BpropIRBuilder> CreateIRBuilder(const std::string &name, const CNodePtr &cnode);
  const BpropHandle *GetBpropHandle(const std::string &name) const {
    return BpropIRBuilderFactory::Instance().GetBuilder(name);
  }
  virtual void PostProcess() const;
  virtual void DumpResult(const std::string &name) const;
  NodePtrList input_nodes_;
  // outputs_ must be CNodePtrList, but output_nodes_ may not necessary. output_nodes_ are used to
  // create bprop func_graph in graph_mode.
  NodePtrList output_nodes_;
  CNodePtrList *outputs_{nullptr};
  UserType *users_{nullptr};
};

class BpropExpanderInGraphMode : public BpropExpander {
 public:
  BpropExpanderInGraphMode() {}
  ~BpropExpanderInGraphMode() = default;
  FuncGraphPtr GetGraph() { return fg_; }

 protected:
  FuncGraphPtr fg_{nullptr};
  void ExtractInputs(const CNodePtr &cnode, const BpropIRBuilder *ir_builder) override;
  std::unique_ptr<BpropIRBuilder> CreateIRBuilder(const std::string &name, const CNodePtr &cnode) override;
  void PostProcess() const override;
  void DumpResult(const std::string &name) const override;
};

#ifdef _MSC_VER
class WinBpropRegister {
 public:
  WinBpropRegister();
  ~WinBpropRegister() {}
  void DoNothing() const {}
};
#endif
}  // namespace bprop
}  // namespace expander

using expander::bprop::BpropExpander;
#ifdef _MSC_VER
using expander::bprop::WinBpropRegister;
#endif
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_BPROP_EXPANDER_BPROP_H_
