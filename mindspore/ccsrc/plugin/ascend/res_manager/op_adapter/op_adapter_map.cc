/**
 * Copyright 2019-2024 Huawei Technologies Co., Ltd
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
#include "plugin/ascend/res_manager/op_adapter/op_adapter_map.h"
#include <map>
#include <memory>
#include <set>
#include "graph/operator.h"
#include "primitive/other_ops.h"
#include "primitive/framework_ops.h"
#include "primitive/sequence_op_name.h"
#include "plugin/ascend/res_manager/op_adapter/op_adapter_desc.h"

namespace mindspore::device::ascend {
namespace {
mindspore::HashMap<std::string, OpAdapterDescPtr> adpt_map_ = {
  {kNameCustomOp, std::make_shared<OpAdapterDesc>(std::make_shared<OpAdapter<Operator>>(""))}};
}  // namespace

template <>
mindspore::HashMap<std::string, mindspore::HashMap<int, std::string>> OpAdapter<::ge::Operator>::cus_input_map_{};
template <>
mindspore::HashMap<std::string, std::map<int, std::string>> OpAdapter<::ge::Operator>::cus_output_map_{};

mindspore::HashMap<std::string, OpAdapterDescPtr> &OpAdapterMap::get() { return adpt_map_; }

OpAdapterPtr FindAdapter(const AnfNodePtr node, bool train) {
  MS_EXCEPTION_IF_NULL(node);
  if (node->isa<CNode>()) {
    auto cnode = node->cast<CNodePtr>();

    std::string name = kNameCustomOp;
    if (!IsCustomCNode(cnode)) {
      name = GetCNodeTargetFuncName(cnode);
    }

    // Convert TupleGetItem to control edge when it has monad.
    if (name == kNameTupleGetItem) {
      if (HasAbstractMonad(node)) {
        name = kNameUpdateState;
      }
    }

    return FindAdapter(name, train);
  }

  if (node->isa<ValueNode>()) {
    return OpAdapterMap::get()[kNameConst]->Get(train);
  }
  if (node->isa<Parameter>()) {
    return OpAdapterMap::get()[kNameParam]->Get(train);
  }
  return OpAdapterPtr(nullptr);
}

OpAdapterPtr FindAdapter(const std::string &name, bool train) {
  auto it = OpAdapterMap::get().find(name);
  if (it != OpAdapterMap::get().end()) {
    return it->second->Get(train);
  }

  std::set<std::string> cpu_only_ops{kRealMakeTupleOpName, kRealTupleGetItemOpName, kShapeCalcOpName};
  auto iter = cpu_only_ops.find(name);
  // If ops in cpu only list or ops is scalar ops or is sequence ops
  if (iter != cpu_only_ops.end() || name.find("Scalar") != std::string::npos ||
      name.find("Sequence") != std::string::npos || name.find("Tuple") != std::string::npos ||
      name.find("List") != std::string::npos) {
    MS_LOG(INFO) << "Can't find OpAdapter for " << name;
    return nullptr;
  }
  MS_LOG(WARNING) << "Can't find OpAdapter for " << name;
  return nullptr;
}

}  // namespace mindspore::device::ascend
