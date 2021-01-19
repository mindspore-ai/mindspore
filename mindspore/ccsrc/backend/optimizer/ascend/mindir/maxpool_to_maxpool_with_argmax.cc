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

#include "backend/optimizer/ascend/mindir/maxpool_to_maxpool_with_argmax.h"

#include <vector>
#include <memory>

#include "utils/utils.h"
#include "utils/ms_context.h"
#include "backend/optimizer/common/helper.h"
#include "runtime/device/kernel_info.h"
#include "backend/session/anf_runtime_algorithm.h"

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
    MS_LOG(EXCEPTION) << "MaxPoolGrad's input number should be " << kMaxPoolGradInputNum - 1 << ", but got "
                      << maxpool_grad->inputs().size() - 1;
  }
  auto maxpool_anf = maxpool_grad->input(2);
  MS_EXCEPTION_IF_NULL(maxpool_anf);
  return maxpool_anf->cast<CNodePtr>();
}

CNodePtr CreateMaxPoolWithArgmax(const FuncGraphPtr &graph, const CNodePtr &maxpool) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(maxpool);
  if (maxpool->inputs().size() != kMaxPoolInputNum) {
    MS_LOG(EXCEPTION) << "MaxPool's input number should be " << kMaxPoolInputNum - 1 << ", but got "
                      << maxpool->inputs().size() - 1;
  }
  std::vector<AnfNodePtr> maxpool_argmax_inputs = {NewValueNode(std::make_shared<Primitive>(kMaxPoolWithArgmaxOpName)),
                                                   maxpool->input(1)};
  auto maxpool_argmax = graph->NewCNode(maxpool_argmax_inputs);
  MS_EXCEPTION_IF_NULL(maxpool_argmax);
  maxpool_argmax->set_scope(maxpool->scope());

  // MaxPoolWithArgmax's second output is argmax, whose datatype is uint16 and with same shape as first output
  TypeId argmax_dtype = kNumberTypeUInt16;
  auto types = {AnfAlgo::GetOutputInferDataType(maxpool, 0), argmax_dtype};
  auto out_shape = AnfAlgo::GetOutputInferShape(maxpool, 0);
  auto shapes = {out_shape, out_shape};
  AnfAlgo::SetOutputInferTypeAndShape(types, shapes, maxpool_argmax.get());
  return maxpool_argmax;
}

CNodePtr CreateMaxPoolGradWithArgmax(const FuncGraphPtr &graph, const CNodePtr &maxpool_grad,
                                     const std::vector<AnfNodePtr> &maxpool_argmax_outputs) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(maxpool_grad);
  if (maxpool_grad->inputs().size() != kMaxPoolGradInputNum) {
    MS_LOG(EXCEPTION) << "MaxPoolGrad's input number should be " << kMaxPoolGradInputNum - 1 << ", but got "
                      << maxpool_grad->inputs().size() - 1;
  }
  // MaxPoolGrad's inputs are {input, output, grad_input}, MaxPoolGradWithArgmax's inputs are
  // {input, grad_input, argmax_output}
  std::vector<AnfNodePtr> maxpool_grad_argmax_inputs = {
    NewValueNode(std::make_shared<Primitive>(kMaxPoolGradWithArgmaxOpName)), maxpool_grad->input(1),
    maxpool_grad->input(3), maxpool_argmax_outputs[1]};
  auto maxpool_grad_argmax = graph->NewCNode(maxpool_grad_argmax_inputs);
  MS_EXCEPTION_IF_NULL(maxpool_grad_argmax);
  maxpool_grad_argmax->set_scope(maxpool_grad->scope());
  maxpool_grad_argmax->set_abstract(maxpool_grad->abstract());
  return maxpool_grad_argmax;
}

void SetNodeAttrs(const CNodePtr &maxpool, const CNodePtr &maxpool_grad, const CNodePtr &maxpool_argmax,
                  const CNodePtr &maxpool_grad_argmax) {
  auto strides = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(maxpool, kAttrStrides);
  auto ksize = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(maxpool, kAttrKernelSize);
  if (strides.size() != kMaxPoolAttrAxisNum) {
    MS_LOG(EXCEPTION) << "MaxPool's attr strides has wrong axis number, should be " << kMaxPoolAttrAxisNum
                      << ", but got " << strides.size();
  }
  if (ksize.size() != kMaxPoolAttrAxisNum) {
    MS_LOG(EXCEPTION) << "MaxPool's attr ksize has wrong axis number, should be " << kMaxPoolAttrAxisNum << ", but got "
                      << ksize.size();
  }
  // note that strides and ksize change from (1, 1, x, y) to (1, x, y, 1)
  for (size_t i = 1; i <= 2; ++i) {
    strides[i] = strides[i + 1];
    ksize[i] = ksize[i + 1];
  }
  strides[3] = 1;
  ksize[3] = 1;

  AnfAlgo::CopyNodeAttrs(maxpool, maxpool_argmax);
  AnfAlgo::CopyNodeAttrs(maxpool_grad, maxpool_grad_argmax);
  AnfAlgo::SetNodeAttr(kAttrStrides, MakeValue(strides), maxpool_argmax);
  AnfAlgo::SetNodeAttr(kAttrStrides, MakeValue(strides), maxpool_grad_argmax);
  AnfAlgo::SetNodeAttr(kAttrKernelSize, MakeValue(ksize), maxpool_argmax);
  AnfAlgo::SetNodeAttr(kAttrKernelSize, MakeValue(ksize), maxpool_grad_argmax);
}
}  // namespace

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
