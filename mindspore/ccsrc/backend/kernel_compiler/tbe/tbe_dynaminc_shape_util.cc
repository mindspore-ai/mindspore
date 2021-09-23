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

#include "backend/kernel_compiler/tbe/tbe_dynaminc_shape_util.h"
#include <memory>
#include <string>
#include <utility>
#include <map>
#include <algorithm>
#include <vector>
#include "backend/session/anf_runtime_algorithm.h"

namespace mindspore {
namespace kernel {
namespace tbe {
namespace {
constexpr int64_t k16 = 16;
constexpr int64_t k4 = 4;
constexpr int kDims2 = 2;
enum k2Axis : int { kN = 0, kC, kH, kW, kNchwDims };
enum k3Axis : int { N_ncdhw = 0, C_ncdhw, D_ncdhw, H_ncdhw, W_ncdhw, kNcdhwDims };
RangePair PaddingRangeTo5D(const RangePair &ori_range) {
  RangePair dst_range(kNcdhwDims, std::pair<int64_t, int64_t>(1, 1));
  switch (ori_range.size()) {
    case N_ncdhw:
      return ori_range;
    case C_ncdhw:
      dst_range[C_ncdhw] = ori_range[N_ncdhw];
      break;
    case D_ncdhw:
      dst_range[C_ncdhw] = ori_range[N_ncdhw];
      dst_range[D_ncdhw] = ori_range[C_ncdhw];
      break;
    case H_ncdhw:
      dst_range[C_ncdhw] = ori_range[N_ncdhw];
      dst_range[D_ncdhw] = ori_range[C_ncdhw];
      dst_range[H_ncdhw] = ori_range[D_ncdhw];
      break;
    case W_ncdhw:
      dst_range[C_ncdhw] = ori_range[N_ncdhw];
      dst_range[D_ncdhw] = ori_range[C_ncdhw];
      dst_range[H_ncdhw] = ori_range[D_ncdhw];
      dst_range[W_ncdhw] = ori_range[H_ncdhw];
      break;
    default:
      MS_LOG(EXCEPTION) << "Unexpected shape size = " << ori_range.size();
  }
  return dst_range;
}

RangePair PaddingRangeTo4D(const RangePair &ori_range) {
  RangePair dst_range(kNchwDims, std::pair<int64_t, int64_t>(1, 1));
  switch (ori_range.size()) {
    case kN:
      return dst_range;
    case kC:
      dst_range[kC] = ori_range[kN];
      break;
    case kH:
      dst_range[kC] = ori_range[kN];
      dst_range[kH] = ori_range[kC];
      break;
    case kW:
      dst_range[kC] = ori_range[kN];
      dst_range[kH] = ori_range[kC];
      dst_range[kW] = ori_range[kH];
      break;
    case kNchwDims:
      (void)std::copy(ori_range.begin(), ori_range.end(), dst_range.begin());
      break;
    default:
      MS_LOG(EXCEPTION) << "Unexpected range size: " << ori_range.size();
  }
  return dst_range;
}

RangePair NchwRange(const RangePair &range) { return range; }

RangePair NhwcRange(const RangePair &range) {
  RangePair dst_range;
  dst_range.push_back(range[kN]);
  dst_range.push_back(range[kH]);
  dst_range.push_back(range[kW]);
  dst_range.push_back(range[kC]);
  return dst_range;
}

RangePair HwchRange(const RangePair &range) {
  RangePair dst_range;
  dst_range.push_back(range[kH]);
  dst_range.push_back(range[kW]);
  dst_range.push_back(range[kC]);
  dst_range.push_back(range[kN]);
  return dst_range;
}

RangePair Nc1hwc0Range(const RangePair &range) {
  RangePair dst_range;
  const std::pair<int64_t, int64_t> c0 = {k16, k16};
  const std::pair<int64_t, int64_t> c1 = {(range[kC].first + k16 - 1) / k16, (range[kC].second + k16 - 1) / k16};
  dst_range.push_back(range[kN]);
  dst_range.push_back(c1);
  dst_range.push_back(range[kH]);
  dst_range.push_back(range[kW]);
  dst_range.push_back(c0);
  return dst_range;
}

RangePair Nc1hwc04Range(const RangePair &range) {
  RangePair dst_range;
  const std::pair<int64_t, int64_t> c0 = {k4, k4};
  const std::pair<int64_t, int64_t> c1 = {1, 1};
  dst_range.push_back(range[kN]);
  dst_range.push_back(c1);
  dst_range.push_back(range[kH]);
  dst_range.push_back(range[kW]);
  dst_range.push_back(c0);
  return dst_range;
}

RangePair FracNZRange(const RangePair &range) {
  RangePair dst_range;
  if (range.size() < kDims2) {
    MS_LOG(EXCEPTION) << "Format FracNZ can not support range size: " << range.size();
  } else {
    (void)std::copy(range.begin(), range.end() - kDims2, std::back_inserter(dst_range));
  }
  const std::pair<int64_t, int64_t> c0 = {k16, k16};
  const std::pair<int64_t, int64_t> w1 = {(range[range.size() - 1].first - 1) / k16 + 1,
                                          (range[range.size() - 1].second - 1) / k16 + 1};
  const std::pair<int64_t, int64_t> h1 = {(range[range.size() - kDims2].first - 1) / k16 + 1,
                                          (range[range.size() - kDims2].second - 1) / k16 + 1};
  dst_range.push_back(w1);
  dst_range.push_back(h1);
  dst_range.push_back(c0);
  dst_range.push_back(c0);
  return dst_range;
}

RangePair FracZRange(const RangePair &range) {
  RangePair dst_range;
  const std::pair<int64_t, int64_t> c0 = {k16, k16};
  const std::pair<int64_t, int64_t> cout16 = {((range[kN].first + k16 - 1) / k16) * k16,
                                              ((range[kN].second + k16 - 1) / k16) * k16};
  const std::pair<int64_t, int64_t> cin16 = {((range[kC].first + k16 - 1) / k16) * k16,
                                             ((range[kC].second + k16 - 1) / k16) * k16};
  const std::pair<int64_t, int64_t> r0 = {range[kH].first * range[kW].first * cin16.first / k16,
                                          range[kH].second * range[kW].second * cin16.second / k16};
  const std::pair<int64_t, int64_t> r1 = {cout16.first / k16, cout16.second / k16};
  dst_range.push_back(r0);
  dst_range.push_back(r1);
  dst_range.push_back(c0);
  dst_range.push_back(c0);
  return dst_range;
}

RangePair FracZC04Range(const RangePair &range) {
  RangePair dst_range;
  const std::pair<int64_t, int64_t> c0 = {k4, k4};
  const std::pair<int64_t, int64_t> c16 = {k16, k16};
  const std::pair<int64_t, int64_t> first_dim = {(c0.first * range[kH].first * range[kW].first + k16 - 1) / k16,
                                                 (c0.second * range[kH].second * range[kW].second + k16 - 1) / k16};
  const std::pair<int64_t, int64_t> no = {(range[kN].first + k16 - 1) / k16, (range[kN].second + k16 - 1) / k16};
  dst_range.push_back(first_dim);
  dst_range.push_back(no);
  dst_range.push_back(c16);
  dst_range.push_back(c16);
  return dst_range;
}

RangePair FracZNLSTMCRange(const RangePair &range) {
  RangePair dst_range;
  const std::pair<int64_t, int64_t> c0 = {k4, k4};
  const std::pair<int64_t, int64_t> c16 = {k4, k4};
  const std::pair<int64_t, int64_t> h = {range[kN].first / c0.first, range[kN].second / c0.second};
  const std::pair<int64_t, int64_t> i = {range[kC].first - h.first, range[kC].second - h.second};
  const std::pair<int64_t, int64_t> first_dim = {(i.first + k16 - 1) / k16 + (h.first + k16 - 1) / k16,
                                                 (i.second + k16 - 1) / k16 + (h.second + k16 - 1) / k16};
  const std::pair<int64_t, int64_t> second = {c0.first * ((h.first + k16 - 1) / k16),
                                              c0.second * ((h.second + k16 - 1) / k16)};
  dst_range.push_back(first_dim);
  dst_range.push_back(second);
  dst_range.push_back(c16);
  dst_range.push_back(c16);
  return dst_range;
}

RangePair C1hwncoc0Range(const RangePair &range) {
  RangePair dst_range;
  const std::pair<int64_t, int64_t> c0 = {k16, k16};
  const std::pair<int64_t, int64_t> r1 = {(range[kC].first - 1) / k16 + 1, (range[kC].second - 1) / k16 + 1};
  dst_range.push_back(r1);
  dst_range.push_back(range[kH]);
  dst_range.push_back(range[kW]);
  dst_range.push_back(range[kN]);
  dst_range.push_back(c0);
  dst_range.push_back(c0);
  return dst_range;
}

RangePair NcdhwRange(const RangePair &range) { return range; }

RangePair NdhwcRange(const RangePair &range) {
  RangePair dst_range;
  dst_range.push_back(range[N_ncdhw]);
  dst_range.push_back(range[D_ncdhw]);
  dst_range.push_back(range[H_ncdhw]);
  dst_range.push_back(range[W_ncdhw]);
  dst_range.push_back(range[C_ncdhw]);
  return range;
}

RangePair Ndc1hwc0Range(const RangePair &range) {
  RangePair dst_range;
  const std::pair<int64_t, int64_t> c0 = {k16, k16};
  const std::pair<int64_t, int64_t> c1 = {(range[C_ncdhw].first + k16 - 1) / k16,
                                          (range[C_ncdhw].second + k16 - 1) / k16};
  dst_range.push_back(range[N_ncdhw]);
  dst_range.push_back(range[D_ncdhw]);
  dst_range.push_back(c1);
  dst_range.push_back(range[H_ncdhw]);
  dst_range.push_back(range[W_ncdhw]);
  dst_range.push_back(c0);
  return dst_range;
}

RangePair FracZ3DRange(const RangePair &range) {
  RangePair dst_range;
  const std::pair<int64_t, int64_t> c0 = {k16, k16};
  const std::pair<int64_t, int64_t> c1 = {(range[C_ncdhw].first + k16 - 1) / k16,
                                          (range[C_ncdhw].second + k16 - 1) / k16};
  const std::pair<int64_t, int64_t> n1 = {(range[N_ncdhw].first + k16 - 1) / k16,
                                          (range[N_ncdhw].second + k16 - 1) / k16};
  const int64_t r1_0 = range[D_ncdhw].first * c1.first * range[H_ncdhw].first * range[W_ncdhw].first;
  const int64_t r1_1 = range[D_ncdhw].second * c1.second * range[H_ncdhw].second * range[W_ncdhw].second;
  const std::pair<int64_t, int64_t> r1 = {r1_0, r1_1};
  dst_range.push_back(r1);
  dst_range.push_back(n1);
  dst_range.push_back(c1);
  dst_range.push_back(c0);
  return dst_range;
}

RangePair DynamicShapeRangeTrans(const RangePair &ori_range, const std::string &format) {
  using RangeTransfer = std::function<RangePair(const RangePair &)>;
  const std::map<std::string, RangeTransfer> format_range_map{
    {kOpFormat_NCHW, NchwRange},
    {kOpFormat_NHWC, NhwcRange},
    {kOpFormat_HWCN, HwchRange},
    {kOpFormat_NC1HWC0, Nc1hwc0Range},
    {kOpFormat_NC1HWC0_C04, Nc1hwc04Range},
    {kOpFormat_FRAC_Z, FracZRange},
    {kOpFormat_FRACTAL_Z_C04, FracZC04Range},
    {kOpFormat_C1HWNCoC0, C1hwncoc0Range},
    {kOpFormat_NCDHW, NcdhwRange},
    {kOpFormat_NDHWC, NdhwcRange},
    {kOpFormat_NDC1HWC0, Ndc1hwc0Range},
    {kOpFormat_FRACTAL_Z_3D, FracZ3DRange},
  };

  if (format == kOpFormat_ND || format == kOpFormat_DEFAULT) {
    return ori_range;
  }
  if (format == kOpFormat_FRACTAL_ZN_LSTM) {
    return FracZNLSTMCRange(ori_range);
  }
  if (format == kOpFormat_FRAC_NZ) {
    return FracNZRange(ori_range);
  }
  auto temp_range = ori_range;
  if (ori_range.size() < kNchwDims && k3DFormatSet.find(format) == k3DFormatSet.end()) {
    MS_LOG(DEBUG) << "A special format:" << format << " with a range size less than 4, so padding the range firstly";
    temp_range = PaddingRangeTo4D(ori_range);
  }
  if (ori_range.size() < kNcdhwDims && k3DFormatSet.find(format) != k3DFormatSet.end()) {
    MS_LOG(DEBUG) << "A special format:" << format << " with a range size less than 4, so padding the range firstly";
    temp_range = PaddingRangeTo5D(ori_range);
  }
  auto iter = format_range_map.find(format);
  if (iter == format_range_map.end()) {
    MS_LOG(WARNING) << "Can not find a supported format: " << format << ", using default range";
    return ori_range;
  }
  return iter->second(temp_range);
}
}  // namespace

bool TbeDynamicShapeUtil::IsDynamicShapeNode(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  auto input_num = AnfAlgo ::GetInputTensorNum(cnode);
  for (size_t i = 0; i < input_num; ++i) {
    auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(cnode, i);
    if (std::any_of(input_shape.begin(), input_shape.end(), [](const size_t &dim) { return dim < 0; })) {
      MS_LOG(INFO) << "Node(" << cnode->fullname_with_scope() << ") is dynamic shape node.";
      return true;
    }
  }
  auto output_num = AnfAlgo ::GetOutputTensorNum(cnode);
  for (size_t i = 0; i < output_num; ++i) {
    auto output_shape = AnfAlgo::GetOutputInferShape(cnode, i);
    if (std::any_of(output_shape.begin(), output_shape.end(), [](const size_t &dim) { return dim < 0; })) {
      MS_LOG(INFO) << "Node(" << cnode->fullname_with_scope() << ") is dynamic shape node.";
      return true;
    }
  }
  return false;
}

bool TbeDynamicShapeUtil::IsDynamicShapeNode(const AnfNodePtr &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  if (anf_node->isa<CNode>()) {
    auto cnode = anf_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    return IsDynamicShapeNode(cnode);
  }
  return false;
}

void TbeDynamicShapeUtil::SetDynamicShapeAttr(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  auto is_dyanmic_shape = IsDynamicShapeNode(cnode);
  AnfAlgo::SetNodeAttr(kAttrIsDynamicShape, MakeValue(is_dyanmic_shape), cnode);
}

bool TbeDynamicShapeUtil::GetDynamicShapeAttr(const AnfNodePtr &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  if (anf_node->isa<CNode>()) {
    auto cnode = anf_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    return GetDynamicShapeAttr(cnode);
  }
  return false;
}

bool TbeDynamicShapeUtil::GetDynamicShapeAttr(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  auto is_dynamic_shape = AnfAlgo::HasNodeAttr(kAttrIsDynamicShape, cnode);
  if (!is_dynamic_shape) {
    return false;
  }
  is_dynamic_shape = AnfAlgo::GetNodeAttr<bool>(cnode, kAttrIsDynamicShape);
  return is_dynamic_shape;
}

std::shared_ptr<OpInfo> TbeDynamicShapeUtil::FindOp(const std::string &op_name, const AnfNodePtr &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  if (anf_node->isa<CNode>()) {
    auto cnode = anf_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    return FindOp(op_name, cnode);
  }
  return nullptr;
}

std::shared_ptr<OpInfo> TbeDynamicShapeUtil::FindOp(const std::string &op_name, const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  auto is_dynamic_shape = GetDynamicShapeAttr(cnode);
  return mindspore::kernel::OpLib::FindOp(op_name, OpImplyType::kTBE, is_dynamic_shape);
}

RangePair TbeDynamicShapeUtil::GetInputDynamicRange(const AnfNodePtr &anf_node, size_t index,
                                                    const std::string &def_format) {
  MS_EXCEPTION_IF_NULL(anf_node);
  auto kernel_info = dynamic_cast<device::KernelInfo *>(anf_node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto format =
    kernel_info->select_kernel_build_info() == nullptr ? def_format : AnfAlgo::GetInputFormat(anf_node, index);
  auto input_range_min = AnfAlgo::GetInputMinShape(anf_node, index);
  auto input_range_max = AnfAlgo::GetInputMaxShape(anf_node, index);
  if (input_range_min.size() != input_range_max.size()) {
    MS_EXCEPTION(ArgumentError) << "Input range size is not equal, min size: " << input_range_min.size()
                                << "max size: " << input_range_max.size();
  }
  if (input_range_min.empty() && input_range_max.empty()) {
    RangePair ret = {{1, 1}};
    return DynamicShapeRangeTrans(ret, format);
  }
  RangePair ret;
  for (size_t i = 0; i < input_range_min.size(); ++i) {
    ret.emplace_back(input_range_min[i], input_range_max[i]);
  }
  return DynamicShapeRangeTrans(ret, format);
}

RangePair TbeDynamicShapeUtil::GetOutputDynamicRange(const AnfNodePtr &anf_node, size_t index,
                                                     const std::string &def_format) {
  MS_EXCEPTION_IF_NULL(anf_node);
  auto kernel_info = dynamic_cast<device::KernelInfo *>(anf_node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto format =
    kernel_info->select_kernel_build_info() == nullptr ? def_format : AnfAlgo::GetOutputFormat(anf_node, index);
  auto output_range_min = AnfAlgo::GetOutputMinShape(anf_node, index);
  auto output_range_max = AnfAlgo::GetOutputMaxShape(anf_node, index);
  if (output_range_min.size() != output_range_max.size()) {
    MS_EXCEPTION(ArgumentError) << "Onput range size is not equal, min size: " << output_range_min.size()
                                << "max size: " << output_range_max.size();
  }
  if (output_range_max.empty() && output_range_min.empty()) {
    RangePair ret = {{1, 1}};
    return DynamicShapeRangeTrans(ret, format);
  }
  RangePair ret;
  for (size_t i = 0; i < output_range_min.size(); ++i) {
    ret.emplace_back(output_range_min[i], output_range_max[i]);
  }
  return DynamicShapeRangeTrans(ret, format);
}
}  // namespace tbe
}  // namespace kernel
}  // namespace mindspore
