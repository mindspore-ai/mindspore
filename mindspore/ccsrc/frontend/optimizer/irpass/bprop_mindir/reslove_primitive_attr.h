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
#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_BPROP_MINDIR_RESLOVE_PRIMITIVE_ATTR_H
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_BPROP_MINDIR_RESLOVE_PRIMITIVE_ATTR_H

#include <string>
#include <memory>
#include "ir/anf.h"
#include "frontend/optimizer/optimizer.h"
#include "frontend/optimizer/irpass.h"
#include "frontend/optimizer/anf_visitor.h"

namespace mindspore {
namespace opt {
namespace irpass {
// {MindIRFuncGraph::getitem{ {Call{getattr(Primitive, get_attr_dict)}}, Attr_Name}} -> ValueNode
class ReslovePrimitiveAttr : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    Reset();
    if (!IsMatch(node)) {
      return nullptr;
    }
    auto value = primitive_->GetAttr(attr_name_);
    if (value == nullptr) {
      MS_LOG(EXCEPTION) << "The primitive[" << primitive_->ToString() << "] has no attribute name:" << attr_name_
                        << "Primitive info:" << primitive_->DumpText();
    }
    return std::make_shared<ValueNode>(value);
  }

  void Reset() {
    primitive_ = nullptr;
    attr_name_.clear();
  }

 private:
  // {MindIRFuncGraph::getitem{ {Call{getattr(Primitive, get_attr_dict)}}, Attr_Name}}
  bool IsMatch(const AnfNodePtr &node) { return IsCNodeMinIRMetaGraphGetItem(node); }
  // {MindIRFuncGraph::getitem{ {Call{getattr(Primitive, get_attr_dict)}}, Attr_Name}}
  bool IsCNodeMinIRMetaGraphGetItem(const AnfNodePtr &node);
  // Check Is Attr_Name
  bool IsStringAttrValueNode(const AnfNodePtr &node);
  // Check is Call{getattr(Primitive, get_attr_dict)}
  bool IsCallPrimitiveAttrDictNode(const AnfNodePtr &node);
  // getattr(Primitive, get_attr_dict)}
  bool IsGetAttrDictFuncNode(const CNode *node);
  Primitive *primitive_;
  std::string attr_name_;
};
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_BPROP_MINDIR_
