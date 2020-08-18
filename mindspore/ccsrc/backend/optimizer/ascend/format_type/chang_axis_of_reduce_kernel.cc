/**
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
#include "backend/optimizer/ascend/format_type/chang_axis_of_reduce_kernel.h"

#include <string>
#include <memory>
#include <vector>
#include <map>

#include "utils/utils.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "backend/kernel_compiler/common_utils.h"

namespace mindspore {
namespace opt {
namespace {
using ConvertFunction = std::function<void(const CNodePtr &)>;

void ConvertReduceAttrFraczAnd6HD(const CNodePtr &cnode);
const size_t kAxis_H = 2;
const size_t kAxis_W = 3;
const size_t kAxis_6HD_H = 1;
const size_t kAxis_6HD_W = 2;
const std::map<std::string, ConvertFunction> kReduceConvertMap = {{kOpFormat_FRAC_Z, ConvertReduceAttrFraczAnd6HD},
                                                                  {kOpFormat_C1HWNCoC0, ConvertReduceAttrFraczAnd6HD}};
void SafeCheckFunction(const CNodePtr &cnode, const std::vector<int> &reduce_axis) {
  if (reduce_axis.empty()) {
    MS_LOG(EXCEPTION) << "The node " << cnode->DebugString() << "'s reduce axis got a empty vector";
  }
  if (AnfAlgo::GetInputTensorNum(cnode) != AnfAlgo::GetOutputTensorNum(cnode) &&
      AnfAlgo::GetInputTensorNum(cnode) != 1) {
    MS_LOG(EXCEPTION) << "the kind of reduce node [" << cnode->DebugString()
                      << "] is not single input or single output ";
  }
  for (auto elem : reduce_axis) {
    if (elem > 4) {
      MS_LOG(INFO) << "reduce axis is larger than 4 dims reduce axis : [" << elem << "]";
    }
  }
}

void ConvertReduceAttrFraczAnd6HD(const CNodePtr &cnode) {
  auto axis = kernel::GetReduceAttrAxis(cnode);
  std::vector<int> convert_axis;
  SafeCheckFunction(cnode, axis);
  auto format = AnfAlgo::GetInputFormat(cnode, 0);
  if (format != kOpFormat_FRAC_Z || format != kOpFormat_C1HWNCoC0) {
    MS_LOG(EXCEPTION) << "The node [" << cnode->DebugString() << "] format " << format << " is not 5hd";
  }
  for (auto elem : axis) {
    switch (elem) {
      case kAxis_H:
        convert_axis.emplace_back(kAxis_6HD_H);
        break;
      case kAxis_W:
        convert_axis.emplace_back(kAxis_6HD_W);
        break;
      default:
        MS_LOG(INFO) << "reduce axis is axis : [" << elem << "]"
                     << " but the format is not supported this reduce axis";
    }
  }
  AnfAlgo::SetNodeAttr(kAttrAxis, MakeValue(convert_axis), cnode);
}
}  // namespace

const BaseRef ChangeAxisOfReduceKernel::DefinePattern() const {
  VarPtr X = std::make_shared<Var>();
  VarPtr Xs = std::make_shared<SeqVar>();
  return VectorRef({X, Xs});
}

const AnfNodePtr ChangeAxisOfReduceKernel::Process(const FuncGraphPtr &, const AnfNodePtr &node,
                                                   const EquivPtr &) const {
  if (node == nullptr || !node->isa<CNode>() || !AnfAlgo::IsRealKernel(node)) {
    return nullptr;
  }
  if (AnfAlgo::GetOpPattern(node) != kernel::kReducePattern) {
    return nullptr;
  }
  auto convert_map = kReduceConvertMap.find(AnfAlgo::GetInputFormat(node, 0));
  if (convert_map == kReduceConvertMap.end()) {
    return nullptr;
  }
  convert_map->second(node->cast<CNodePtr>());
  return nullptr;
}
}  // namespace opt
}  // namespace mindspore
