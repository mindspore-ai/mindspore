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
 * WITHOUT WARRANTIES OR CONDITIONS OF gNY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "vm/segment_runner.h"

#include <algorithm>
#include <functional>
#include <memory>
#include <set>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <string>

#include "utils/log_adapter.h"
#include "ir/manager.h"
#include "ir/func_graph_cloner.h"
#include "operator/ops.h"

namespace mindspore {
const char kMsConvert[] = "ms";
const char kMsVm[] = "vm";
const char kGeVm[] = "ge";

namespace compile {
// cached conversion
ConvertCache g_ConvertCache;
void ClearConvertCache() { g_ConvertCache.clear(); }

// Return the list of nodes whose values are required beyond this segment.
// Arguments:
//   lst: list of nodes (the segment)
//   users: dict mapping each node to its users (globally)
//   seen: set of nodes that are part of the segment
AnfNodePtrList GetOutput(const AnfNodePtrList& lst, const NodeUsersMap& users, const std::vector<AnfNodePtr>& seen) {
  AnfNodePtrList output;
  if (users.size() == 0) {
    return output;
  }

  (void)std::transform(
    std::begin(lst), std::end(lst), std::back_inserter(output), [&users, &seen](AnfNodePtr n) -> AnfNodePtr {
      auto usersn = users.find(n);
      bool is_referred_out_of_segment = std::any_of(
        std::begin(usersn->second), std::end(usersn->second), [&seen](const std::pair<AnfNodePtr, int>& u) -> bool {
          return std::find(std::begin(seen), std::end(seen), u.first) == std::end(seen);
        });
      if (n->isa<CNode>() && is_referred_out_of_segment) {
        return n;
      }
      return nullptr;
    });

  // remove nullptr
  for (auto it = output.begin(); it != output.end();) {
    if (*it == nullptr) {
      it = output.erase(it);
    } else {
      ++it;
    }
  }

  return output;
}

std::tuple<FuncGraphPtr, AnfNodePtrList, AnfNodePtrList> TransformSegmentToAnfGraph(const AnfNodePtrList& lst) {
  auto fg = std::make_shared<FuncGraph>();
  AnfNodePtrList inputs;
  AnfNodePtrToAnfNodePtrMap eqv;
  if (lst.empty()) {
    MS_LOG(EXCEPTION) << "Input anf node list is empty";
  }

  auto ref = [&eqv, &inputs, &fg](const AnfNodePtr& a) -> AnfNodePtr {
    if (a->isa<ValueNode>() && !IsValueNode<FuncGraph>(a)) {
      eqv[a] = a;
    } else if (eqv.find(a) == eqv.end()) {
      inputs.push_back(a);
      eqv[a] = fg->add_parameter();
    }

    return eqv[a];
  };

  // Merge CNodes into a AnfGraph that represents a linear instruction segment
  for (auto n : lst) {
    if (!n->isa<CNode>()) {
      MS_LOG(EXCEPTION) << "Inst is not CNode";
    }
    auto& inps = n->cast<CNodePtr>()->inputs();

    if (inps.empty()) {
      MS_LOG(EXCEPTION) << "Input is empty";
    }
    if (!IsValueNode<Primitive>(inps[0])) {
      MS_LOG(EXCEPTION) << "Input[0] Must be a Primitive valuenode";
    }
    auto fn = inps[0];

    std::vector<AnfNodePtr> args{fn};
    (void)std::transform(std::begin(inps) + 1, std::end(inps), std::back_inserter(args), ref);

    eqv[n] = fg->NewCNode(args);
  }

  std::vector<AnfNodePtr> eqv_keys;
  (void)std::transform(std::begin(eqv), std::end(eqv), std::back_inserter(eqv_keys),
                       [](const std::pair<AnfNodePtr, AnfNodePtr>& elem) -> AnfNodePtr { return elem.first; });

  auto outputs = GetOutput(lst, lst[0]->func_graph()->manager()->node_users(), eqv_keys);
  std::vector<AnfNodePtr> output_args;
  output_args.push_back(NewValueNode(prim::kPrimMakeTuple));
  (void)std::transform(std::begin(outputs), std::end(outputs), std::back_inserter(output_args),
                       [&eqv](const AnfNodePtr& o) -> AnfNodePtr { return eqv[o]; });

  // Set output for AnfGraph
  auto fg_output = fg->NewCNode(output_args);
  fg->set_output(fg_output);

  return std::make_tuple(fg, inputs, outputs);
}

// Converts the list of nodes to a runnable form.
// All the nodes in the list must represent linear flow (no calls, branches, ...)
// Returns:
//  (fn, inputs, outputs):
//  - fn: A callable function
//  - inputs: the list of inputs nodes whose values should be
//             provided to the function
//  - outputs: the list of output nodes corresponding to the
//             outputs of the function
// Notes:
//   This implementation will convert the nodes into a subgraph
//   that will run using the MsVM.
template <typename T>
LinConvertResult Convert(const AnfNodePtrList& lst) {
  auto cached = g_ConvertCache.find(lst);
  if (cached != g_ConvertCache.end()) {
    return cached->second;
  }

  LinConvertResult result;

  FuncGraphPtr fg = nullptr;
  AnfNodePtrList inputs;
  AnfNodePtrList outputs;

  std::tie(fg, inputs, outputs) = TransformSegmentToAnfGraph(lst);

  // Clone in case g contains subgraphs that have a different manager
  fg = BasicClone(fg);

  std::shared_ptr<VMImpl> vm = std::make_shared<T>();

  result.run =
    std::make_shared<RunFunc>([fg, vm](const VectorRef& args) -> VectorRef { return vm->RunGraph(fg, args); });
  result.inputs = inputs;
  result.outputs = outputs;
  result.graph_id = UINT32_MAX;

  (void)g_ConvertCache.emplace(lst, result);
  return result;
}

LinkFuncType MsVmConvert = Convert<VM>;
LinkFuncType GeVmConvert = Convert<GeVM>;

std::unordered_map<std::string, LinkFuncType> backends = {{kMsVm, MsVmConvert}, {kGeVm, GeVmConvert}};

std::set<std::string> backend_list = {
  kMsConvert,
  kMsVm,
  kGeVm,
};

}  // namespace compile
}  // namespace mindspore
