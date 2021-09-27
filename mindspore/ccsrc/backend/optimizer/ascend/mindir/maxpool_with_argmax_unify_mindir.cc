/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "backend/optimizer/ascend/mindir/maxpool_with_argmax_unify_mindir.h"
#include <memory>
#include <vector>
#include "backend/optimizer/common/helper.h"
#include "runtime/device/kernel_info.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "base/core_ops.h"
#include "utils/utils.h"

namespace mindspore {
namespace opt {
namespace {
constexpr size_t kMaxPoolGradWithArgmaxInputTensorNum = 3;
constexpr size_t kMaxPoolGradWithArgmaxInputNum = 4;
constexpr size_t kMaxPoolWithArgmaxShape = 4;
constexpr size_t kAlignBytes = 16;

bool IsC(const BaseRef &n) {
  if (utils::isa<AnfNodePtr>(n)) {
    AnfNodePtr in = utils::cast<AnfNodePtr>(n);
    MS_EXCEPTION_IF_NULL(in);
    return in->isa<ValueNode>();
  }
  return false;
}

CNodePtr GetMaxPoolWithArgmax(const CNodePtr &maxpool_grad_with_argmax) {
  CheckCNodeInputSize(maxpool_grad_with_argmax, kMaxPoolGradWithArgmaxInputTensorNum);
  auto tuple_getitem0_anf = maxpool_grad_with_argmax->input(kIndex3);
  MS_EXCEPTION_IF_NULL(tuple_getitem0_anf);
  return tuple_getitem0_anf->cast<CNodePtr>();
}
}  // namespace

const BaseRef MaxPoolWithArgmaxUnifyMindIR::DefinePattern() const {
  VarPtr X = std::make_shared<Var>();
  VectorRef pattern({prim::kPrimMaxPoolWithArgmax, X});
  return pattern;
}

const AnfNodePtr MaxPoolWithArgmaxUnifyMindIR::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                       const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  auto maxpool_with_argmax = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(maxpool_with_argmax);

  TypeId argmax_dtype = kNumberTypeUInt16;
  auto ksize = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(maxpool_with_argmax, kAttrKernelSize);
  auto output_shape = AnfAlgo::GetOutputInferShape(maxpool_with_argmax, 0);
  auto argmax_shape = output_shape;
  if (argmax_shape.size() != kMaxPoolWithArgmaxShape || ksize.size() != kMaxPoolWithArgmaxShape) {
    MS_LOG(EXCEPTION) << "argmax or kernel_size's shape size not equal to 4";
  }
  argmax_shape[kDim2] = LongToSize(ksize[kDim1] * ksize[kDim2]);
  argmax_shape[kDim3] = (output_shape[kDim2] * output_shape[kDim3] + kAlignBytes - 1) / kAlignBytes + 1;
  auto types = {AnfAlgo::GetOutputInferDataType(maxpool_with_argmax, 0), argmax_dtype};
  auto shapes = {output_shape, argmax_shape};
  AnfAlgo::SetOutputInferTypeAndShape(types, shapes, maxpool_with_argmax.get());

  return maxpool_with_argmax;
}

const BaseRef MaxPoolGradWithArgmaxUnifyMindIR::DefinePattern() const {
  VarPtr X = std::make_shared<Var>();
  VarPtr Y = std::make_shared<Var>();
  VarPtr index0 = std::make_shared<CondVar>(IsC);
  VectorRef maxpool_with_argmax({prim::kPrimMaxPoolWithArgmax, X});
  VectorRef tuple_getitem0 = VectorRef({prim::kPrimTupleGetItem, maxpool_with_argmax, index0});
  VectorRef maxpool_grad_with_argmax({prim::kPrimMaxPoolGradWithArgmax, X, Y, tuple_getitem0});
  return maxpool_grad_with_argmax;
}

const AnfNodePtr MaxPoolGradWithArgmaxUnifyMindIR::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                           const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  auto maxpool_grad_with_argmax = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(maxpool_grad_with_argmax);
  auto tuple_getitem0_anf = GetMaxPoolWithArgmax(maxpool_grad_with_argmax);
  MS_EXCEPTION_IF_NULL(tuple_getitem0_anf);

  TypeId argmax_dtype = kNumberTypeUInt16;
  auto ksize = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(maxpool_grad_with_argmax, kAttrKernelSize);
  auto argmax_shape = AnfAlgo::GetOutputInferShape(tuple_getitem0_anf, 0);
  if (argmax_shape.size() != kMaxPoolWithArgmaxShape || ksize.size() != kMaxPoolWithArgmaxShape) {
    MS_LOG(EXCEPTION) << "argmax or kernel_size's shape size not equal to 4";
  }
  argmax_shape[kDim3] = (argmax_shape[kDim2] * argmax_shape[kDim3] + kAlignBytes - 1) / kAlignBytes + 1;
  argmax_shape[kDim2] = LongToSize(ksize[kDim1] * ksize[kDim2]);
  AnfAlgo::SetOutputInferTypeAndShape({argmax_dtype}, {argmax_shape}, tuple_getitem0_anf.get());

  return maxpool_grad_with_argmax;
}
}  // namespace opt
}  // namespace mindspore
