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

#include "plugin/device/ascend/optimizer/mindir/slice_grad_unify_mindir.h"

#include <vector>
#include <memory>
#include <string>
#include <algorithm>

#include "include/common/utils/utils.h"
#include "utils/ms_context.h"
#include "utils/trace_base.h"
#include "backend/common/optimizer/helper.h"
#include "include/backend/kernel_info.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore {
namespace opt {
namespace {
constexpr size_t kSliceGradInputTensorNum = 4;
constexpr size_t kSliceGradCangjieInputTensorNum = 2;
constexpr auto kMSliceGrad = "m_slice_grad";
constexpr auto kRPad = "r_pad";
constexpr auto kX1 = "X1";
constexpr auto kXs = "Xs";

std::vector<int64_t> GetInputXShape(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  return common::AnfAlgo::GetPrevNodeOutputInferShape(node, 1);
}

std::vector<int64_t> GetTupleValue(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto value_node = node->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(value_node);
  MS_EXCEPTION_IF_NULL(value_node->value());
  return GetValue<std::vector<int64_t>>(value_node->value());
}

AnfNodePtr BuildPad(const PatternMap &m, const AnfNodePtr &pad) {
  auto node = m.Get(kMSliceGrad);
  MS_EXCEPTION_IF_NULL(node);
  auto slice_grad = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(slice_grad);
  auto input_num = common::AnfAlgo::GetInputTensorNum(slice_grad);
  if (input_num != kSliceGradInputTensorNum && input_num != kSliceGradCangjieInputTensorNum) {
    MS_LOG(EXCEPTION) << "The input tensor size[" << input_num
                      << "] of node " + slice_grad->DebugString() + " is not equal to " << kSliceGradInputTensorNum
                      << " or " << kSliceGradCangjieInputTensorNum << trace::DumpSourceLines(node);
  }
  MS_EXCEPTION_IF_NULL(pad);
  pad->set_scope(slice_grad->scope());
  pad->set_abstract(slice_grad->abstract());

  // set attr paddings
  auto x_shape = GetInputXShape(slice_grad);
  std::vector<int64_t> begins;
  std::vector<int64_t> sizes;
  if (input_num == kSliceGradInputTensorNum) {
    begins = GetTupleValue(slice_grad->input(kIndex3));
    sizes = GetTupleValue(slice_grad->input(kIndex4));
  } else {
    // if frontend is Cangjie and mode is pynative, input 2, 3 is already converted to attr
    begins = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(slice_grad, kAttrBegin);
    sizes = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(slice_grad, kAttrSize);
  }
  if (x_shape.size() != begins.size() || begins.size() != sizes.size()) {
    MS_LOG(EXCEPTION)
      << "For SliceGrad, x_shape dim number should be equal to len(begin) and len(size), but got x_shape dim: "
      << x_shape.size() << ", len(begin): " << begins.size() << ", len(size): " << sizes.size()
      << trace::DumpSourceLines(node);
  }
  std::vector<std::vector<int64_t>> paddings;
  for (size_t i = 0; i < x_shape.size(); ++i) {
    paddings.emplace_back(std::vector<int64_t>{begins[i], (x_shape[i] - begins[i] - sizes[i])});
  }
  common::AnfAlgo::SetNodeAttr(kAttrPaddings, MakeValue(paddings), pad);
  common::AnfAlgo::SetNodeAttr(kAttrInputNames, MakeValue(std::vector<std::string>{"x"}), pad);

  return pad;
}
}  // namespace

bool SliceGradUnifyMindIR::CheckMatchedDAG(const PatternMap &, const FuncGraphPtr &, const AnfNodePtr &node) const {
  MS_EXCEPTION_IF_NULL(node);
  auto slice_grad = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(slice_grad);
  auto input_num = common::AnfAlgo::GetInputTensorNum(slice_grad);
  auto x_shape = GetInputXShape(slice_grad);
  if (input_num == kSliceGradInputTensorNum) {
    auto begin_value = GetValueNode(slice_grad->input(kIndex3));
    auto size_value = GetValueNode(slice_grad->input(kIndex4));
    if (IsDynamic(x_shape) || begin_value == nullptr || size_value == nullptr || !begin_value->isa<ValueSequence>() ||
        !size_value->isa<ValueSequence>()) {
      return false;
    }
  }
  return true;
}

void SliceGradUnifyMindIR::DefineSrcPattern(SrcPattern *src_pattern) {
  (*src_pattern).AddVar(kX1).AddSeqVar(kXs).AddCNode(kMSliceGrad, {std::make_shared<Primitive>("SliceGrad"), kX1, kXs});
}

void SliceGradUnifyMindIR::DefineDstPattern(DstPattern *dst_pattern) {
  (*dst_pattern).AddCNode(kRPad, {std::make_shared<Primitive>(kPadDOpName), kX1}, BuildPad);
}
}  // namespace opt
}  // namespace mindspore
