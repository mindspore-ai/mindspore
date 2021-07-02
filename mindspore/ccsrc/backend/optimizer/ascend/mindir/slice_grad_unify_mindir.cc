/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "backend/optimizer/ascend/mindir/slice_grad_unify_mindir.h"

#include <vector>
#include <memory>
#include <string>
#include <algorithm>

#include "utils/utils.h"
#include "utils/ms_context.h"
#include "backend/optimizer/common/helper.h"
#include "runtime/device/kernel_info.h"
#include "backend/session/anf_runtime_algorithm.h"

namespace mindspore {
namespace opt {
namespace {
constexpr size_t kSliceGradInputTensorNum = 4;
constexpr size_t kSliceGradCangjieInputTensorNum = 2;

std::vector<int64_t> GetInputXShape(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  std::vector<int64_t> shapes;
  auto shape_size_t = AnfAlgo::GetPrevNodeOutputInferShape(node, 1);
  std::transform(shape_size_t.begin(), shape_size_t.end(), std::back_inserter(shapes), SizeToLong);
  return shapes;
}

std::vector<int64_t> GetTupleValue(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto value_node = node->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(value_node);
  MS_EXCEPTION_IF_NULL(value_node->value());
  return GetValue<std::vector<int64_t>>(value_node->value());
}
}  // namespace

const BaseRef SliceGradUnifyMindIR::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  VectorRef slice_grad({std::make_shared<Primitive>("SliceGrad"), Xs});
  return slice_grad;
}

const AnfNodePtr SliceGradUnifyMindIR::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                               const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);

  auto slice_grad = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(slice_grad);
  auto input_num = AnfAlgo::GetInputTensorNum(slice_grad);
  if (input_num != kSliceGradInputTensorNum && input_num != kSliceGradCangjieInputTensorNum) {
    MS_LOG(EXCEPTION) << "The input tensor size[" << input_num
                      << "] of node " + slice_grad->DebugString() + " is not equal to " << kSliceGradInputTensorNum
                      << " or " << kSliceGradCangjieInputTensorNum;
  }
  std::vector<AnfNodePtr> pad_inputs = {NewValueNode(std::make_shared<Primitive>(kPadOpName)),
                                        slice_grad->input(kIndex1)};
  auto pad = graph->NewCNode(pad_inputs);
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
    begins = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(slice_grad, kAttrBegin);
    sizes = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(slice_grad, kAttrSize);
  }
  if (x_shape.size() != begins.size() || begins.size() != sizes.size()) {
    MS_LOG(EXCEPTION) << "For SliceGrad, x's shape dim number should be equal to len(begin) and len(size).";
  }
  std::vector<std::vector<int64_t>> paddings;
  for (size_t i = 0; i < x_shape.size(); ++i) {
    paddings.emplace_back(std::vector<int64_t>{begins[i], x_shape[i] - begins[i] - sizes[i]});
  }
  AnfAlgo::SetNodeAttr(kAttrPaddings, MakeValue(paddings), pad);
  AnfAlgo::SetNodeAttr(kAttrInputNames, MakeValue(std::vector<std::string>{"x"}), pad);

  return pad;
}
}  // namespace opt
}  // namespace mindspore
