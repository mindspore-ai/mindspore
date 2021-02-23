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

#include "backend/optimizer/ascend/mindir/conv2d_unify_mindir.h"

#include <vector>
#include <string>
#include <memory>
#include <utility>

#include "utils/utils.h"
#include "utils/ms_context.h"
#include "utils/check_convert_utils.h"
#include "backend/optimizer/common/helper.h"
#include "runtime/device/kernel_info.h"
#include "backend/session/anf_runtime_algorithm.h"

namespace mindspore {
namespace opt {
namespace {
constexpr size_t kConv2DBackpropInputNum = 4;
constexpr size_t kConv2DAxisNum = 4;
constexpr auto kAttrOffsetA = "offset_a";
constexpr auto kAttrPadList = "pad_list";
constexpr auto kAttrMode = "mode";
constexpr auto kAttrChannelMultiplier = "channel_multiplier";
constexpr auto kAttrPerm = "perm";
constexpr auto kAttrInputSizes = "input_sizes";
constexpr auto kAttrInputSize = "input_size";

bool NeedUpdate(const CNodePtr &conv2d, std::vector<size_t> in_shape, std::vector<size_t> out_shape) {
  MS_EXCEPTION_IF_NULL(conv2d);
  auto group = LongToSize(AnfAlgo::GetNodeAttr<int64_t>(conv2d, kAttrGroup));
  if (group == 1) {
    return false;
  }

  auto primitive_ptr = GetCNodePrimitive(conv2d);
  MS_EXCEPTION_IF_NULL(primitive_ptr);
  auto data_format_ptr = primitive_ptr->GetAttr(kAttrFormat);
  MS_EXCEPTION_IF_NULL(data_format_ptr);
  int64_t data_format;
  bool result = CheckAndConvertUtils::GetDataFormatEnumValue(data_format_ptr, &data_format);
  if (!result || data_format != Format::NCHW) {
    MS_LOG(EXCEPTION) << "Conv2D only supports NCHW when group > 1";
  }
  if (in_shape.size() != kConv2DAxisNum || out_shape.size() != kConv2DAxisNum) {
    MS_LOG(EXCEPTION) << "Conv2D's input and output should have 4 axis, but got input axis num: " << in_shape.size()
                      << "output axis num: " << out_shape.size();
  }
  auto in_channel = in_shape[1];
  auto out_channel = out_shape[1];
  if (group != in_channel || group != out_channel) {
    MS_LOG(EXCEPTION) << "Conv2D's attr group should be equal to in_channel and out_channel when group > 1, but got "
                      << "group: " << group << " in_channel: " << in_channel << " out_channel: " << out_channel;
  }
  return true;
}

ValueNodePtr CreatePermValueNode(const FuncGraphPtr &func_graph, const std::vector<int64_t> &perm) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto kernel_graph = func_graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  std::vector<ValuePtr> axis_values{};
  abstract::AbstractBasePtrList abs{};
  for (const auto &axis : perm) {
    axis_values.push_back(MakeValue(axis));
    abs.push_back(std::make_shared<abstract::AbstractScalar>(axis));
  }
  auto perm_value_tuple = std::make_shared<ValueTuple>(axis_values);
  MS_EXCEPTION_IF_NULL(perm_value_tuple);
  auto abstract = std::make_shared<abstract::AbstractTuple>(abs);
  MS_EXCEPTION_IF_NULL(abstract);
  auto perm_value = kernel_graph->NewValueNode(abstract, perm_value_tuple);
  MS_EXCEPTION_IF_NULL(perm_value);
  kernel_graph->AddValueNodeToGraph(perm_value);
  return perm_value;
}

CNodePtr CreateTranspose(const FuncGraphPtr &graph, const CNodePtr &conv2d, const AnfNodePtr &input_node,
                         bool need_trans_output) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(conv2d);
  MS_EXCEPTION_IF_NULL(input_node);
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto perm = std::vector<int64_t>{1, 0, 2, 3};
  std::vector<AnfNodePtr> transpose_inputs;
  if (ms_context->get_param<int>(MS_CTX_EXECUTION_MODE) == kPynativeMode) {
    transpose_inputs = {NewValueNode(std::make_shared<Primitive>(kTransposeOpName)), input_node};
  } else {
    transpose_inputs = {NewValueNode(std::make_shared<Primitive>(kTransposeOpName)), input_node,
                        CreatePermValueNode(graph, perm)};
  }
  auto transpose = graph->NewCNode(transpose_inputs);
  MS_EXCEPTION_IF_NULL(transpose);
  transpose->set_scope(conv2d->scope());

  if (need_trans_output) {
    auto types = {AnfAlgo::GetOutputInferDataType(input_node, 0)};
    auto out_shape = AnfAlgo::GetOutputInferShape(input_node, 0);
    if (out_shape.size() != kConv2DAxisNum) {
      MS_LOG(EXCEPTION) << "Conv2D's output axis number should be " << kConv2DAxisNum << ", but got "
                        << out_shape.size();
    }
    std::swap(out_shape[0], out_shape[1]);
    auto shapes = {out_shape};
    AnfAlgo::SetOutputInferTypeAndShape(types, shapes, transpose.get());
  } else {
    transpose->set_abstract(conv2d->abstract());
  }

  auto input_names = std::vector<std::string>{"x", "perm"};
  auto output_names = std::vector<std::string>{"output"};
  AnfAlgo::SetNodeAttr(kAttrInputNames, MakeValue(input_names), transpose);
  AnfAlgo::SetNodeAttr(kAttrOutputNames, MakeValue(output_names), transpose);
  if (ms_context->get_param<int>(MS_CTX_EXECUTION_MODE) == kPynativeMode) {
    AnfAlgo::SetNodeAttr(kAttrPerm, MakeValue(perm), transpose);
  }
  return transpose;
}

CNodePtr CreateDepthwiseConv2D(const FuncGraphPtr &graph, const CNodePtr &conv2d, const CNodePtr &transpose) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(conv2d);
  CheckCNodeInputSize(conv2d, kConvInputTensorNum);
  std::vector<AnfNodePtr> depth_conv_inputs = {NewValueNode(std::make_shared<Primitive>(kDepthwiseConv2dNativeOpName)),
                                               conv2d->input(1), transpose};
  auto depth_conv = graph->NewCNode(depth_conv_inputs);
  MS_EXCEPTION_IF_NULL(depth_conv);
  depth_conv->set_abstract(conv2d->abstract());
  depth_conv->set_scope(conv2d->scope());
  return depth_conv;
}

CNodePtr CreateDepthwiseConv2DBackpropInput(const FuncGraphPtr &graph, const CNodePtr &conv2d_backin,
                                            const CNodePtr &transpose) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(conv2d_backin);

  CNodePtr depth_conv_backin = nullptr;
  if (conv2d_backin->inputs().size() == kConv2DBackpropInputNum) {
    std::vector<AnfNodePtr> depth_conv_backin_inputs = {
      NewValueNode(std::make_shared<Primitive>(kDepthwiseConv2dNativeBackpropInputOpName)), conv2d_backin->input(3),
      transpose, conv2d_backin->input(1)};
    depth_conv_backin = graph->NewCNode(depth_conv_backin_inputs);
  } else {
    // In nn.Conv2DTranspose, Conv2DBackpropInput is a forward op and the input_sizes input will be convert to attr
    // in pynative mode.
    std::vector<AnfNodePtr> depth_conv_backin_inputs = {
      NewValueNode(std::make_shared<Primitive>(kDepthwiseConv2dNativeBackpropInputOpName)), transpose,
      conv2d_backin->input(1)};
    depth_conv_backin = graph->NewCNode(depth_conv_backin_inputs);
    AnfAlgo::CopyNodeAttr(kAttrInputSizes, kAttrInputSize, conv2d_backin, depth_conv_backin);
  }
  MS_EXCEPTION_IF_NULL(depth_conv_backin);
  depth_conv_backin->set_abstract(conv2d_backin->abstract());
  depth_conv_backin->set_scope(conv2d_backin->scope());
  return depth_conv_backin;
}

CNodePtr CreateDepthwiseConv2DBackpropFilter(const FuncGraphPtr &graph, const CNodePtr &conv2d_backfil) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(conv2d_backfil);
  if (conv2d_backfil->inputs().size() != kConv2DBackpropInputNum) {
    MS_LOG(EXCEPTION) << "Conv2DBackpropFilter's input number should be " << kConv2DBackpropInputNum - 1 << ", but got "
                      << conv2d_backfil->inputs().size() - 1;
  }
  auto filter_size_node = conv2d_backfil->input(3);
  MS_EXCEPTION_IF_NULL(filter_size_node);
  auto filter_size_vnode = filter_size_node->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(filter_size_vnode);
  auto filter_size = GetValue<std::vector<int64_t>>(filter_size_vnode->value());
  // swap axis 0 and 1 of filter shape, but don't swap twice since some node share same filter_size valuenode
  // when the filter_size value is same.
  if (filter_size[0] != 1) {
    std::swap(filter_size[0], filter_size[1]);
    conv2d_backfil->input(3)->cast<ValueNodePtr>()->set_value(MakeValue(filter_size));
  }
  std::vector<AnfNodePtr> depth_conv_backfil_inputs = {
    NewValueNode(std::make_shared<Primitive>(kDepthwiseConv2dNativeBackpropFilterOpName)), conv2d_backfil->input(2),
    conv2d_backfil->input(3), conv2d_backfil->input(1)};
  auto depth_conv_backfil = graph->NewCNode(depth_conv_backfil_inputs);
  MS_EXCEPTION_IF_NULL(depth_conv_backfil);
  depth_conv_backfil->set_scope(conv2d_backfil->scope());

  auto types = {AnfAlgo::GetOutputInferDataType(conv2d_backfil, 0)};
  std::vector<size_t> out_shape = AnfAlgo::GetOutputInferShape(conv2d_backfil, 0);
  if (out_shape.size() != kConv2DAxisNum) {
    MS_LOG(EXCEPTION) << "Conv2DBackpropFilter's output axis number should be " << kConv2DAxisNum << ", but got "
                      << out_shape.size();
  }
  std::swap(out_shape[0], out_shape[1]);
  auto shapes = {out_shape};
  AnfAlgo::SetOutputInferTypeAndShape(types, shapes, depth_conv_backfil.get());
  return depth_conv_backfil;
}

void SetCommonAttrs(const CNodePtr &conv2d, const CNodePtr &depth_conv) {
  AnfAlgo::CopyNodeAttr(kAttrKernelSize, conv2d, depth_conv);
  AnfAlgo::CopyNodeAttr(kAttrDilation, conv2d, depth_conv);
  AnfAlgo::CopyNodeAttr(kAttrFormat, conv2d, depth_conv);
  AnfAlgo::CopyNodeAttr(kAttrPadList, conv2d, depth_conv);
  AnfAlgo::CopyNodeAttr(kAttrPadMode, conv2d, depth_conv);
  AnfAlgo::SetNodeAttr(kAttrMode, MakeValue(3), depth_conv);
  AnfAlgo::SetNodeAttr(kAttrChannelMultiplier, MakeValue(1), depth_conv);
}

void SetConv2DAttrs(const CNodePtr &conv2d, const CNodePtr &depth_conv) {
  SetCommonAttrs(conv2d, depth_conv);
  AnfAlgo::CopyNodeAttr(kAttrInputNames, conv2d, depth_conv);
  AnfAlgo::CopyNodeAttr(kAttrStride, conv2d, depth_conv);
  if (AnfAlgo::HasNodeAttr(kAttrOffsetA, conv2d)) {
    AnfAlgo::CopyNodeAttr(kAttrOffsetA, conv2d, depth_conv);
  } else {
    AnfAlgo::SetNodeAttr(kAttrOffsetA, MakeValue(0), depth_conv);
  }
}

void SetConv2DBackpropInputAttrs(const CNodePtr &conv2d_backin, const CNodePtr &depth_conv_backin) {
  SetCommonAttrs(conv2d_backin, depth_conv_backin);
  auto input_names = std::vector<std::string>{"input_size", "filter", "dout"};
  AnfAlgo::SetNodeAttr(kAttrInputNames, MakeValue(input_names), depth_conv_backin);
  auto stride = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(conv2d_backin, kAttrStride);
  if (stride.size() == 2) {
    stride.insert(stride.begin(), 2, 1);
  }
  AnfAlgo::SetNodeAttr(kAttrStride, MakeValue(stride), depth_conv_backin);
}

void SetConv2DBackpropFilterAttrs(const CNodePtr &conv2d_backfil, const CNodePtr &depth_conv_backfil) {
  SetCommonAttrs(conv2d_backfil, depth_conv_backfil);
  auto input_names = std::vector<std::string>{"input", "filter_size", "dout"};
  AnfAlgo::SetNodeAttr(kAttrInputNames, MakeValue(input_names), depth_conv_backfil);
  auto stride = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(conv2d_backfil, kAttrStride);
  if (stride.size() == 2) {
    stride.insert(stride.begin(), 2, 1);
  }
  AnfAlgo::SetNodeAttr(kAttrStride, MakeValue(stride), depth_conv_backfil);
}
}  // namespace

const BaseRef Conv2DUnifyMindIR::DefinePattern() const {
  VarPtr X = std::make_shared<Var>();
  VarPtr W = std::make_shared<Var>();
  VectorRef pattern({prim::kPrimConv2D, X, W});
  return pattern;
}

const AnfNodePtr Conv2DUnifyMindIR::Process(const FuncGraphPtr &graph, const AnfNodePtr &node, const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);

  auto conv2d = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(conv2d);
  auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(conv2d, 0);
  auto output_shape = AnfAlgo::GetOutputInferShape(conv2d, 0);
  if (!NeedUpdate(conv2d, input_shape, output_shape)) {
    return nullptr;
  }
  CheckCNodeInputSize(conv2d, kConvInputTensorNum);
  auto transpose = CreateTranspose(graph, conv2d, conv2d->input(2), true);
  auto depth_conv = CreateDepthwiseConv2D(graph, conv2d, transpose);
  SetConv2DAttrs(conv2d, depth_conv);
  return depth_conv;
}

const BaseRef Conv2DBackpropInputUnifyMindIR::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  VectorRef pattern({prim::kPrimConv2DBackpropInput, Xs});
  return pattern;
}

const AnfNodePtr Conv2DBackpropInputUnifyMindIR::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                         const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);

  auto conv2d_backin = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(conv2d_backin);
  auto input_shape = AnfAlgo::GetOutputInferShape(conv2d_backin, 0);
  auto output_shape = AnfAlgo::GetPrevNodeOutputInferShape(conv2d_backin, 0);
  if (!NeedUpdate(conv2d_backin, input_shape, output_shape)) {
    return nullptr;
  }

  auto input_size = conv2d_backin->inputs().size();
  // In pynative mode, input_sizes input will be convert to attr if Conv2DBackpropInput is a forward op.
  if (input_size != kConv2DBackpropInputNum && input_size != kConv2DBackpropInputNum - 1) {
    MS_LOG(EXCEPTION) << "Conv2DBackpropInput's input number should be " << kConv2DBackpropInputNum - 1 << " or "
                      << kConv2DBackpropInputNum - 2 << ", but got " << input_size - 1;
  }
  auto transpose = CreateTranspose(graph, conv2d_backin, conv2d_backin->input(2), true);
  auto depth_conv_backin = CreateDepthwiseConv2DBackpropInput(graph, conv2d_backin, transpose);
  SetConv2DBackpropInputAttrs(conv2d_backin, depth_conv_backin);
  return depth_conv_backin;
}

const BaseRef Conv2DBackpropFilterUnifyMindIR::DefinePattern() const {
  VarPtr dout = std::make_shared<Var>();
  VarPtr input = std::make_shared<Var>();
  VarPtr filter_size = std::make_shared<Var>();
  VectorRef pattern({prim::kPrimConv2DBackpropFilter, dout, input, filter_size});
  return pattern;
}

const AnfNodePtr Conv2DBackpropFilterUnifyMindIR::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                          const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);

  auto conv2d_backfil = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(conv2d_backfil);
  auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(conv2d_backfil, 1);
  auto output_shape = AnfAlgo::GetPrevNodeOutputInferShape(conv2d_backfil, 0);
  if (!NeedUpdate(conv2d_backfil, input_shape, output_shape)) {
    return nullptr;
  }

  auto depth_conv_backfil = CreateDepthwiseConv2DBackpropFilter(graph, conv2d_backfil);
  SetConv2DBackpropFilterAttrs(conv2d_backfil, depth_conv_backfil);
  auto transpose = CreateTranspose(graph, conv2d_backfil, depth_conv_backfil, false);

  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  (void)manager->Replace(conv2d_backfil, transpose);
  return transpose;
}
}  // namespace opt
}  // namespace mindspore
