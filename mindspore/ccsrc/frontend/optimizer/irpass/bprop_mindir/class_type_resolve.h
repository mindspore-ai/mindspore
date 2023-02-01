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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_BPROP_MINDIR_CLASS_TYPE_RESOLVE_H
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_BPROP_MINDIR_CLASS_TYPE_RESOLVE_H

#include <string>
#include <memory>
#include <utility>
#include <vector>

#include "ir/value.h"
#include "frontend/optimizer/optimizer.h"
#include "frontend/optimizer/irpass.h"
#include "frontend/optimizer/anf_visitor.h"

namespace mindspore {
namespace opt {
namespace irpass {
class ClassTypeResolve : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    if (!IsValueNode<parse::ClassType>(node) && !IsValueNode<ValueDictionary>(node) &&
        !IsValueNode<ValueSequence>(node) && !IsValueNode<parse::NameSpace>(node)) {
      return nullptr;
    }
    bool need_convert = false;
    auto value_node = node->cast<ValueNodePtr>();
    auto value = value_node->value();
    auto new_value = ConvertValue(value, &need_convert);
    if (need_convert) {
      return NewValueNode(new_value);
    }
    return nullptr;
  }

 private:
  ValuePtr ConvertValueSequence(const ValueSequencePtr &value, bool *need_convert);
  ValuePtr ConvertValue(const ValuePtr &value, bool *need_convert);
};
ValuePtr ClassTypeResolve::ConvertValue(const ValuePtr &value, bool *need_convert) {
  if (value->isa<parse::ClassType>()) {
    auto class_type = value->cast<parse::ClassTypePtr>()->name();
    (*need_convert) = true;
    return std::make_shared<MindIRClassType>(class_type);
  }
  if (value->isa<parse::NameSpace>()) {
    auto name_space = value->cast<parse::NameSpacePtr>()->name();
    (*need_convert) = true;
    return std::make_shared<MindIRNameSpace>(name_space);
  }

  if (value->isa<ValueDictionary>()) {
    auto dic = value->cast<ValueDictionaryPtr>();
    auto dic_pairs = dic->value();
    std::vector<std::pair<ValuePtr, ValuePtr>> convert_dict;
    for (const auto &item : dic_pairs) {
      (void)convert_dict.emplace_back(std::make_pair(item.first, ConvertValue(item.second, need_convert)));
    }
    if (need_convert) {
      return std::make_shared<ValueDictionary>(convert_dict);
    }
  }

  if (value->isa<ValueSequence>()) {
    MS_EXCEPTION_IF_NULL(value);
    auto seq_value = value->cast<ValueSequencePtr>();
    return ConvertValueSequence(seq_value, need_convert);
  }
  return value;
}

ValuePtr ClassTypeResolve::ConvertValueSequence(const ValueSequencePtr &seq_value, bool *need_convert) {
  auto vec_seq = std::vector<ValuePtr>();
  for (size_t i = 0; i < seq_value->size(); ++i) {
    (void)vec_seq.emplace_back(ConvertValue((*seq_value)[i], need_convert));
  }
  if (!need_convert) {
    return seq_value;
  }
  if (seq_value->isa<ValueTuple>()) {
    return std::make_shared<ValueTuple>(vec_seq);
  }
  return std::make_shared<ValueList>(vec_seq);
}
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_BPROP_MINDIR_CLASS_TYPE_RESOLVE_H
