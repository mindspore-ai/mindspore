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

#include "plugin/device/ascend/optimizer/mindir/conv2d_unify_mindir.h"

#include <vector>
#include <string>
#include <memory>
#include <utility>
#include <algorithm>

#include "include/common/utils/utils.h"
#include "utils/ms_context.h"
#include "utils/check_convert_utils.h"
#include "utils/trace_base.h"
#include "include/backend/optimizer/helper.h"
#include "include/backend/kernel_info.h"
#include "plugin/device/ascend/optimizer/ascend_helper.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore {
namespace opt {
namespace {
constexpr size_t kConv2DBackpropInputNum = 3;
constexpr size_t kConv2DAxisNum = 4;
constexpr size_t kConv2DFilterSize = 4;
constexpr auto kAttrOffsetA = "offset_a";
constexpr auto kAttrPadList = "pad_list";
constexpr auto kAttrMode = "mode";
constexpr auto kAttrChannelMultiplier = "channel_multiplier";
constexpr auto kAttrInputSizes = "input_sizes";
constexpr auto kAttrInputSize = "input_size";

bool NeedUpdate(const CNodePtr &conv2d, ShapeVector in_shape, ShapeVector out_shape) {
  MS_EXCEPTION_IF_NULL(conv2d);
  auto group = common::AnfAlgo::GetNodeAttr<int64_t>(conv2d, kAttrGroup);
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
    MS_LOG(EXCEPTION) << "Conv2D only supports NCHW when group > 1" << trace::DumpSourceLines(conv2d);
  }
  if (in_shape.size() != kConv2DAxisNum || out_shape.size() != kConv2DAxisNum) {
    MS_LOG(EXCEPTION) << "Conv2D's input and output should have 4 axis, but got input axis num: " << in_shape.size()
                      << "output axis num: " << out_shape.size() << trace::DumpSourceLines(conv2d);
  }
  auto in_channel = in_shape[kDim1];
  auto out_channel = out_shape[kDim1];
  if (group != in_channel || group != out_channel) {
    return false;
  }
  return true;
}

CNodePtr CreateTranspose(const FuncGraphPtr &graph, const CNodePtr &conv2d, const AnfNodePtr &input_node,
                         bool need_trans_output, const PatternProcessPass &pass) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(conv2d);
  MS_EXCEPTION_IF_NULL(input_node);
  auto perm = std::vector<int64_t>{1, 0, 2, 3};
  std::vector<AnfNodePtr> transpose_inputs = {NewValueNode(std::make_shared<Primitive>(kTransposeOpName)), input_node,
                                              CreatePermValueNode(graph, perm)};
  auto transpose = pass.NewCNode(transpose_inputs, graph);
  MS_EXCEPTION_IF_NULL(transpose);
  transpose->set_scope(conv2d->scope());

  if (need_trans_output) {
    auto types = {common::AnfAlgo::GetOutputInferDataType(input_node, 0UL)};
    auto out_shape = common::AnfAlgo::GetOutputInferShape(input_node, 0UL);
    if (out_shape.size() != kConv2DAxisNum) {
      MS_LOG(EXCEPTION) << "Conv2D's output axis number should be " << kConv2DAxisNum << ", but got "
                        << out_shape.size() << trace::DumpSourceLines(conv2d);
    }
    std::swap(out_shape[kDim0], out_shape[kDim1]);
    auto shapes = {out_shape};
    common::AnfAlgo::SetOutputInferTypeAndShape(types, shapes, transpose.get());
  } else {
    transpose->set_abstract(conv2d->abstract());
  }

  auto input_names = std::vector<std::string>{"x", "perm"};
  auto output_names = std::vector<std::string>{"output"};
  common::AnfAlgo::SetNodeAttr(kAttrInputNames, MakeValue(input_names), transpose);
  common::AnfAlgo::SetNodeAttr(kAttrOutputNames, MakeValue(output_names), transpose);
  return transpose;
}

void SetCommonAttrs(const CNodePtr &conv2d, const CNodePtr &depth_conv) {
  common::AnfAlgo::CopyNodeAttr(kAttrKernelSize, conv2d, depth_conv);
  common::AnfAlgo::CopyNodeAttr(kAttrDilation, conv2d, depth_conv);
  common::AnfAlgo::CopyNodeAttr(kAttrFormat, conv2d, depth_conv);
  common::AnfAlgo::CopyNodeAttr(kAttrPadList, conv2d, depth_conv);
  common::AnfAlgo::CopyNodeAttr(kAttrPadMode, conv2d, depth_conv);
  constexpr auto kMode = 3;
  common::AnfAlgo::SetNodeAttr(kAttrMode, MakeValue(kMode), depth_conv);
  common::AnfAlgo::SetNodeAttr(kAttrChannelMultiplier, MakeValue(1), depth_conv);
}

void SetConv2DAttrs(const CNodePtr &conv2d, const CNodePtr &depth_conv) {
  SetCommonAttrs(conv2d, depth_conv);
  common::AnfAlgo::CopyNodeAttr(kAttrInputNames, conv2d, depth_conv);
  common::AnfAlgo::CopyNodeAttr(kAttrStride, conv2d, depth_conv);
  if (common::AnfAlgo::HasNodeAttr(kAttrOffsetA, conv2d)) {
    common::AnfAlgo::CopyNodeAttr(kAttrOffsetA, conv2d, depth_conv);
  } else {
    common::AnfAlgo::SetNodeAttr(kAttrOffsetA, MakeValue(0), depth_conv);
  }
}

void SetConv2DBackpropInputAttrs(const CNodePtr &conv2d_backin, const CNodePtr &depth_conv_backin) {
  SetCommonAttrs(conv2d_backin, depth_conv_backin);
  auto input_names = std::vector<std::string>{"input_size", "filter", "dout"};
  common::AnfAlgo::SetNodeAttr(kAttrInputNames, MakeValue(input_names), depth_conv_backin);
  auto stride = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(conv2d_backin, kAttrStride);
  constexpr size_t kStrideSize = 2;
  if (stride.size() == kStrideSize) {
    (void)stride.insert(stride.begin(), kStrideSize, 1);
  }
  common::AnfAlgo::SetNodeAttr(kAttrStride, MakeValue(stride), depth_conv_backin);
}

void SetConv2DBackpropFilterAttrs(const CNodePtr &conv2d_backfil, const CNodePtr &depth_conv_backfil) {
  SetCommonAttrs(conv2d_backfil, depth_conv_backfil);
  auto input_names = std::vector<std::string>{"input", "filter_size", "dout"};
  common::AnfAlgo::SetNodeAttr(kAttrInputNames, MakeValue(input_names), depth_conv_backfil);
  auto stride = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(conv2d_backfil, kAttrStride);
  constexpr size_t kStrideSize = 2;
  if (stride.size() == kStrideSize) {
    (void)stride.insert(stride.begin(), kStrideSize, 1);
  }
  common::AnfAlgo::SetNodeAttr(kAttrStride, MakeValue(stride), depth_conv_backfil);
}
}  // namespace

CNodePtr Conv2DUnifyMindIR::CreateDepthwiseConv2D(const FuncGraphPtr &graph, const CNodePtr &conv2d,
                                                  const CNodePtr &transpose) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(conv2d);
  CheckCNodeInputSize(conv2d, kConvInputTensorNum);
  std::vector<AnfNodePtr> depth_conv_inputs = {NewValueNode(std::make_shared<Primitive>(kDepthwiseConv2dNativeOpName)),
                                               conv2d->input(kIndex1), transpose};
  auto depth_conv = NewCNode(depth_conv_inputs, graph);
  MS_EXCEPTION_IF_NULL(depth_conv);
  depth_conv->set_abstract(conv2d->abstract());
  depth_conv->set_scope(conv2d->scope());
  return depth_conv;
}

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
  auto input_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(conv2d, 0UL);
  auto output_shape = common::AnfAlgo::GetOutputInferShape(conv2d, 0UL);
  if (!NeedUpdate(conv2d, input_shape, output_shape)) {
    return nullptr;
  }
  CheckCNodeInputSize(conv2d, kConvInputTensorNum);
  auto transpose = CreateTranspose(graph, conv2d, conv2d->input(kIndex2), true, *this);
  auto depth_conv = CreateDepthwiseConv2D(graph, conv2d, transpose);
  SetConv2DAttrs(conv2d, depth_conv);
  return depth_conv;
}

CNodePtr Conv2DBackpropInputUnifyMindIR::CreateDepthwiseConv2DBackpropInput(const FuncGraphPtr &graph,
                                                                            const CNodePtr &conv2d_backin,
                                                                            const CNodePtr &transpose) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(conv2d_backin);

  CNodePtr depth_conv_backin = nullptr;
  if (AnfUtils::GetInputTensorNum(conv2d_backin) == kConv2DBackpropInputNum) {
    std::vector<AnfNodePtr> depth_conv_backin_inputs = {
      NewValueNode(std::make_shared<Primitive>(kDepthwiseConv2dNativeBackpropInputOpName)),
      conv2d_backin->input(kIndex3), transpose, conv2d_backin->input(kIndex1)};
    depth_conv_backin = NewCNode(depth_conv_backin_inputs, graph);
  } else {
    // In nn.Conv2DTranspose, Conv2DBackpropInput is a forward op and the input_sizes input will be convert to attr
    // in pynative mode.
    std::vector<AnfNodePtr> depth_conv_backin_inputs = {
      NewValueNode(std::make_shared<Primitive>(kDepthwiseConv2dNativeBackpropInputOpName)), transpose,
      conv2d_backin->input(kIndex1)};
    depth_conv_backin = NewCNode(depth_conv_backin_inputs, graph);
    common::AnfAlgo::CopyNodeAttr(kAttrInputSizes, kAttrInputSize, conv2d_backin, depth_conv_backin);
  }
  MS_EXCEPTION_IF_NULL(depth_conv_backin);
  depth_conv_backin->set_abstract(conv2d_backin->abstract());
  depth_conv_backin->set_scope(conv2d_backin->scope());
  return depth_conv_backin;
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
  auto input_shape = common::AnfAlgo::GetOutputInferShape(conv2d_backin, 0UL);
  auto output_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(conv2d_backin, 0UL);
  if (!NeedUpdate(conv2d_backin, input_shape, output_shape)) {
    return nullptr;
  }

  auto input_size = AnfUtils::GetInputTensorNum(conv2d_backin);
  // In pynative mode, input_sizes input will be convert to attr if Conv2DBackpropInput is a forward op.
  if (input_size != kConv2DBackpropInputNum && input_size != kConv2DBackpropInputNum - 1) {
    MS_LOG(EXCEPTION) << "Conv2DBackpropInput's input number should be " << kConv2DBackpropInputNum << " or "
                      << (kConv2DBackpropInputNum - 1) << ", but got " << input_size << trace::DumpSourceLines(node);
  }
  auto transpose = CreateTranspose(graph, conv2d_backin, conv2d_backin->input(kIndex2), true, *this);
  auto depth_conv_backin = CreateDepthwiseConv2DBackpropInput(graph, conv2d_backin, transpose);
  SetConv2DBackpropInputAttrs(conv2d_backin, depth_conv_backin);
  return depth_conv_backin;
}

CNodePtr Conv2DBackpropFilterUnifyMindIR::CreateDepthwiseConv2DBackpropFilter(const FuncGraphPtr &graph,
                                                                              const CNodePtr &conv2d_backfil) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(conv2d_backfil);
  if (AnfUtils::GetInputTensorNum(conv2d_backfil) != kConv2DBackpropInputNum) {
    MS_LOG(EXCEPTION) << "Conv2DBackpropFilter's input number should be " << kConv2DBackpropInputNum << ", but got "
                      << AnfUtils::GetInputTensorNum(conv2d_backfil) << trace::DumpSourceLines(conv2d_backfil);
  }
  auto filter_size_node = conv2d_backfil->input(kIndex3);
  MS_EXCEPTION_IF_NULL(filter_size_node);
  auto filter_size_vnode = filter_size_node->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(filter_size_vnode);
  auto filter_size = GetValue<std::vector<int64_t>>(filter_size_vnode->value());
  if (filter_size.size() < kConv2DFilterSize) {
    MS_LOG(EXCEPTION) << "Filter size input of node[" << conv2d_backfil->fullname_with_scope()
                      << "] should be 4-D, but got " << filter_size;
  }
  std::swap(filter_size[0], filter_size[1]);
  auto new_filter_size_vnode = CreateShapeValueNode(graph, filter_size);

  std::vector<AnfNodePtr> depth_conv_backfil_inputs = {
    NewValueNode(std::make_shared<Primitive>(kDepthwiseConv2dNativeBackpropFilterOpName)),
    conv2d_backfil->input(kIndex2), new_filter_size_vnode, conv2d_backfil->input(kIndex1)};
  auto depth_conv_backfil = NewCNode(depth_conv_backfil_inputs, graph);
  MS_EXCEPTION_IF_NULL(depth_conv_backfil);
  depth_conv_backfil->set_scope(conv2d_backfil->scope());

  auto types = {common::AnfAlgo::GetOutputInferDataType(conv2d_backfil, 0UL)};
  auto out_shape = common::AnfAlgo::GetOutputInferShape(conv2d_backfil, 0UL);
  if (out_shape.size() != kConv2DAxisNum) {
    MS_LOG(EXCEPTION) << "Conv2DBackpropFilter's output axis number should be " << kConv2DAxisNum << ", but got "
                      << out_shape.size() << trace::DumpSourceLines(conv2d_backfil);
  }
  std::swap(out_shape[0], out_shape[1]);
  auto shapes = {out_shape};
  common::AnfAlgo::SetOutputInferTypeAndShape(types, shapes, depth_conv_backfil.get());
  return depth_conv_backfil;
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
  auto input_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(conv2d_backfil, 1UL);
  auto output_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(conv2d_backfil, 0UL);
  if (!NeedUpdate(conv2d_backfil, input_shape, output_shape)) {
    return nullptr;
  }

  auto depth_conv_backfil = CreateDepthwiseConv2DBackpropFilter(graph, conv2d_backfil);
  SetConv2DBackpropFilterAttrs(conv2d_backfil, depth_conv_backfil);
  auto transpose = CreateTranspose(graph, conv2d_backfil, depth_conv_backfil, false, *this);

  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  (void)manager->Replace(conv2d_backfil, transpose);
  return transpose;
}
}  // namespace opt
}  // namespace mindspore
