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

 private:
  bool RunBprop(const CNodePtr &cnode);
  NodePtrList ExtractInputs(const CNodePtr &cnode, const BpropIRBuilder *ir_builder);
  const BpropHandle *GetBpropHandle(const std::string &name) const {
    return BpropIRBuilderFactory::Instance().GetBuilder(name);
  }
  void PostProcess(const NodePtrList &inputs) const;
  void DumpResult(const std::string &name, const NodePtrList &inputs) const;

 private:
  CNodePtrList *outputs_{nullptr};
  UserType *users_{nullptr};
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
