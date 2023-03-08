/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/optimizer/mindir/maxpool_to_maxpool_with_argmax.h"

#include <vector>
#include <memory>
#include <string>

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
constexpr size_t kMaxPoolInputNum = 2;
constexpr size_t kMaxPoolAttrAxisNum = 4;
constexpr size_t kMaxPoolGradInputNum = 4;
constexpr size_t kMaxPoolWithArgmaxOutputNum = 2;

CNodePtr GetMaxPool(const CNodePtr &maxpool_grad) {
  MS_EXCEPTION_IF_NULL(maxpool_grad);
  if (maxpool_grad->inputs().size() != kMaxPoolGradInputNum) {
    MS_LOG(EXCEPTION) << "MaxPoolGrad's input number should be " << (kMaxPoolGradInputNum - 1) << ", but got "
                      << (maxpool_grad->inputs().size() - 1) << trace::DumpSourceLines(maxpool_grad);
  }
  auto maxpool_anf = maxpool_grad->input(kIndex2);
  MS_EXCEPTION_IF_NULL(maxpool_anf);
  return maxpool_anf->cast<CNodePtr>();
}
}  // namespace

CNodePtr MaxPool2MaxPoolWithArgmax::CreateMaxPoolWithArgmax(const FuncGraphPtr &graph, const CNodePtr &maxpool) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(maxpool);
  if (maxpool->inputs().size() != kMaxPoolInputNum) {
    MS_LOG(EXCEPTION) << "MaxPool's input number should be " << (kMaxPoolInputNum - 1) << ", but got "
                      << (maxpool->inputs().size() - 1) << trace::DumpSourceLines(maxpool);
  }
  std::vector<AnfNodePtr> maxpool_argmax_inputs = {NewValueNode(std::make_shared<Primitive>(kMaxPoolWithArgmaxOpName)),
                                                   maxpool->input(kIndex1)};
  auto maxpool_argmax = opt::NewCNode(maxpool_argmax_inputs, graph, {maxpool});
  MS_EXCEPTION_IF_NULL(maxpool_argmax);
  maxpool_argmax->set_scope(maxpool->scope());

  // MaxPoolWithArgmax's second output is argmax, whose datatype is uint16 and with same shape as first output
  TypeId argmax_dtype = kNumberTypeUInt16;
  auto types = {common::AnfAlgo::GetOutputInferDataType(maxpool, 0UL), argmax_dtype};
  auto out_shape = AnfAlgo::GetOutputDetailShape(maxpool, 0UL);
  std::vector<BaseShapePtr> shapes = {out_shape, out_shape};
  common::AnfAlgo::SetOutputTypeAndDetailShape(types, shapes, maxpool_argmax.get());
  return maxpool_argmax;
}

CNodePtr MaxPool2MaxPoolWithArgmax::CreateMaxPoolGradWithArgmax(
  const FuncGraphPtr &graph, const CNodePtr &maxpool_grad,
  const std::vector<AnfNodePtr> &maxpool_argmax_outputs) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(maxpool_grad);
  if (maxpool_grad->inputs().size() != kMaxPoolGradInputNum) {
    MS_LOG(EXCEPTION) << "MaxPoolGrad's input number should be " << (kMaxPoolGradInputNum - 1) << ", but got "
                      << (maxpool_grad->inputs().size() - 1) << trace::DumpSourceLines(maxpool_grad);
  }
  // MaxPoolGrad's inputs are {input, output, grad_input}, MaxPoolGradWithArgmax's inputs are
  // {input, grad_input, argmax_output}
  std::vector<AnfNodePtr> maxpool_grad_argmax_inputs = {
    NewValueNode(std::make_shared<Primitive>(kMaxPoolGradWithArgmaxOpName)), maxpool_grad->input(kIndex1),
    maxpool_grad->input(kIndex3), maxpool_argmax_outputs[kIndex1]};
  auto maxpool_grad_argmax = opt::NewCNode(maxpool_grad_argmax_inputs, graph, {maxpool_grad});
  MS_EXCEPTION_IF_NULL(maxpool_grad_argmax);
  maxpool_grad_argmax->set_scope(maxpool_grad->scope());
  maxpool_grad_argmax->set_abstract(maxpool_grad->abstract());
  return maxpool_grad_argmax;
}

void MaxPool2MaxPoolWithArgmax::SetNodeAttrs(const CNodePtr &maxpool, const CNodePtr &maxpool_grad,
                                             const CNodePtr &maxpool_argmax,
                                             const CNodePtr &maxpool_grad_argmax) const {
  auto strides = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(maxpool, kAttrStrides);
  auto ksize = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(maxpool, kAttrKernelSize);
  if (strides.size() != kMaxPoolAttrAxisNum) {
    MS_LOG(EXCEPTION) << "MaxPool's attr strides has wrong axis number, should be " << kMaxPoolAttrAxisNum
                      << ", but got " << strides.size() << trace::DumpSourceLines(maxpool);
  }
  if (ksize.size() != kMaxPoolAttrAxisNum) {
    MS_LOG(EXCEPTION) << "MaxPool's attr ksize has wrong axis number, should be " << kMaxPoolAttrAxisNum << ", but got "
                      << ksize.size() << trace::DumpSourceLines(maxpool);
  }
  // note that strides and ksize change from (1, 1, x, y) to (1, x, y, 1)
  strides[kIndex1] = strides[kIndex2];
  strides[kIndex2] = strides[kIndex3];
  strides[kIndex3] = 1;

  ksize[kIndex1] = ksize[kIndex2];
  ksize[kIndex2] = ksize[kIndex3];
  ksize[kIndex3] = 1;

  common::AnfAlgo::CopyNodeAttrs(maxpool, maxpool_argmax);
  common::AnfAlgo::CopyNodeAttrs(maxpool_grad, maxpool_grad_argmax);
  common::AnfAlgo::SetNodeAttr(kAttrStrides, MakeValue(strides), maxpool_argmax);
  common::AnfAlgo::SetNodeAttr(kAttrStrides, MakeValue(strides), maxpool_grad_argmax);
  common::AnfAlgo::SetNodeAttr(kAttrKernelSize, MakeValue(ksize), maxpool_argmax);
  common::AnfAlgo::SetNodeAttr(kAttrKernelSize, MakeValue(ksize), maxpool_grad_argmax);
}

std::vector<std::string> MaxPool2MaxPoolWithArgmax::MustExistPrimitiveName() const {
  std::vector<std::string> ret;
  ret.emplace_back(prim::kPrimMaxPool->name());
  ret.emplace_back(prim::kPrimMaxPoolGrad->name());
  return ret;
}

const BaseRef MaxPool2MaxPoolWithArgmax::DefinePattern() const {
  VarPtr X = std::make_shared<Var>();
  VarPtr Y = std::make_shared<Var>();
  VectorRef maxpool({prim::kPrimMaxPool, X});
  VectorRef pattern({prim::kPrimMaxPoolGrad, X, maxpool, Y});
  return pattern;
}

const AnfNodePtr MaxPool2MaxPoolWithArgmax::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                    const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);

  auto maxpool_grad = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(maxpool_grad);
  auto maxpool = GetMaxPool(maxpool_grad);
  MS_EXCEPTION_IF_NULL(maxpool);
  if (common::AnfAlgo::IsDynamicShape(maxpool)) {
    // maxpoolwithargmax and maxpoolgradwithargmax don't support dynamic shape, so add the judgement;
    // can delete the judgement after supported;
    return nullptr;
  }

  auto maxpool_argmax = CreateMaxPoolWithArgmax(graph, maxpool);
  std::vector<AnfNodePtr> maxpool_argmax_outputs;
  CreateMultipleOutputsOfAnfNode(graph, maxpool_argmax, kMaxPoolWithArgmaxOutputNum, &maxpool_argmax_outputs);
  auto maxpool_grad_argmax = CreateMaxPoolGradWithArgmax(graph, maxpool_grad, maxpool_argmax_outputs);
  SetNodeAttrs(maxpool, maxpool_grad, maxpool_argmax, maxpool_grad_argmax);

  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  (void)manager->Replace(maxpool, maxpool_argmax_outputs[0]);
  return maxpool_grad_argmax;
}
}  // namespace opt
}  // namespace mindspore
