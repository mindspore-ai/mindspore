/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/optimizer/ir_fusion/avgpool_3d_fusion.h"
#include <vector>
#include <memory>
#include <string>
#include <algorithm>
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore {
namespace opt {
namespace {
constexpr size_t kAvgPool3DInputNum = 1;
constexpr size_t k5DInferDims = 5;
constexpr int64_t kC0 = 16;
constexpr size_t kDHWDimNum = 3;
constexpr size_t kNCDHWDimNum = 5;

int64_t GetInterSection(int64_t start_1, int64_t end_1, int64_t start_2, int64_t end_2) {
  if (end_1 <= start_2) {
    return 0;
  }
  if (start_1 >= end_2) {
    return 0;
  }
  if (start_1 < start_2) {
    start_1 = start_2;
  }
  if (end_1 > end_2) {
    end_1 = end_2;
  }
  return end_1 - start_1;
}

bool GetKernelSize(const AnfNodePtr &node, int64_t *kd, int64_t *kh, int64_t *kw) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(kd);
  MS_EXCEPTION_IF_NULL(kh);
  MS_EXCEPTION_IF_NULL(kw);
  if (common::AnfAlgo::HasNodeAttr("kernel_size", node->cast<CNodePtr>())) {
    auto kernel_size = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(node, "kernel_size");
    if (kernel_size.size() == 1) {
      *kd = kernel_size[kDim0];
      *kh = kernel_size[kDim0];
      *kw = kernel_size[kDim0];
    } else if (kernel_size.size() == kDHWDimNum) {
      *kd = kernel_size[kDim0];
      *kh = kernel_size[kDim1];
      *kw = kernel_size[kDim2];
    } else if (kernel_size.size() == kNCDHWDimNum) {
      // NCDHW
      *kd = kernel_size[kDim2];
      *kh = kernel_size[kDim3];
      *kw = kernel_size[kDim4];
    } else {
      MS_LOG(EXCEPTION) << "Unknown kernel size " << kernel_size.size() << trace::DumpSourceLines(node);
    }
    return true;
  }
  return false;
}

bool GetStrideSize(const AnfNodePtr &node, int64_t *sd, int64_t *sh, int64_t *sw) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(sd);
  MS_EXCEPTION_IF_NULL(sh);
  MS_EXCEPTION_IF_NULL(sw);
  if (common::AnfAlgo::HasNodeAttr("strides", node->cast<CNodePtr>())) {
    auto stride_size = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(node, "strides");
    if (stride_size.size() == 1) {
      *sd = stride_size[kDim0];
      *sh = stride_size[kDim0];
      *sw = stride_size[kDim0];
    } else if (stride_size.size() == kDHWDimNum) {
      *sd = stride_size[kDim0];
      *sh = stride_size[kDim1];
      *sw = stride_size[kDim2];
    } else if (stride_size.size() == kNCDHWDimNum) {
      // NCDHW
      *sd = stride_size[kDim2];
      *sh = stride_size[kDim3];
      *sw = stride_size[kDim4];
    } else {
      MS_LOG(EXCEPTION) << "Unknown strides size " << stride_size.size() << trace::DumpSourceLines(node);
    }
    return true;
  }
  return false;
}

void GetAttrs(const AnfNodePtr &node, std::vector<int64_t> *pad_list, bool *count_include_pad, bool *ceil_mode,
              int64_t *divisor_override) {
  MS_EXCEPTION_IF_NULL(node);
  if (!common::AnfAlgo::HasNodeAttr("pad_list", node->cast<CNodePtr>())) {
    MS_LOG(EXCEPTION) << "AvgPool3D should has attr pad_list" << trace::DumpSourceLines(node);
  }
  *pad_list = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(node, "pad_list");
  if (common::AnfAlgo::HasNodeAttr("count_include_pad", node->cast<CNodePtr>())) {
    *count_include_pad = common::AnfAlgo::GetNodeAttr<bool>(node, "count_include_pad");
  }
  if (common::AnfAlgo::HasNodeAttr("ceil_mode", node->cast<CNodePtr>())) {
    *ceil_mode = common::AnfAlgo::GetNodeAttr<bool>(node, "ceil_mode");
  }
  if (common::AnfAlgo::HasNodeAttr("divisor_override", node->cast<CNodePtr>())) {
    *divisor_override = common::AnfAlgo::GetNodeAttr<int64_t>(node, "divisor_override");
  }
}

bool IsVectorImpl(int64_t fh, int64_t fw, int64_t kh, int64_t kw, const std::vector<int64_t> &pad_list) {
  if (std::any_of(pad_list.begin(), pad_list.end(), [](int64_t item) { return item != 0; })) {
    return false;
  }
  if (fh != kh || fw != kw) {
    return false;
  }
  return true;
}

bool IsZeroPads(const std::vector<int64_t> &pad_list) {
  return std::all_of(pad_list.begin(), pad_list.end(), [](int64_t item) { return item == 0; });
}

AnfNodePtr ConstructFilter(const FuncGraphPtr &func_graph, const std::vector<int64_t> &pad_list, int64_t fc, int64_t kd,
                           int64_t kh, int64_t kw, bool ceil_mode, int64_t divisor_override) {
  MS_EXCEPTION_IF_NULL(func_graph);
  // assist tensor 1
  int64_t c1 = (fc + kC0 - 1) / kC0;
  ShapeVector assist_shape = {c1 * kd * kh * kw, 1, kC0, kC0};  // frac_z_3d
  ShapeVector infer_shape = {1, fc, kd, kh, kw};
  float val = 1.0 / (kd * kh * kw);
  if (divisor_override != 0) {
    val = 1.0 / divisor_override;
  } else if (!IsZeroPads(pad_list) || ceil_mode) {
    val = 1.0;
  }
  // create value node
  int64_t cnt = c1 * kd * kh * kw;
  return ConstructFilterValueNode(func_graph, val, assist_shape, infer_shape, cnt);
}

AnfNodePtr ConstructMultiplier(const FuncGraphPtr &func_graph, int64_t fn, int64_t fc, int64_t fd, int64_t fh,
                               int64_t fw, int64_t dd, int64_t dh, int64_t dw, int64_t kd, int64_t kh, int64_t kw,
                               int64_t sd, int64_t sh, int64_t sw, const std::vector<int64_t> &pad_list,
                               bool count_include_pad) {
  MS_EXCEPTION_IF_NULL(func_graph);
  //  assist tensor 2
  std::vector<int64_t> assist_shape = {fn, fc, dd, dh, dw};  // NCDHW
  tensor::TensorPtr tensor = std::make_shared<tensor::Tensor>(kNumberTypeFloat16, assist_shape);
  MS_EXCEPTION_IF_NULL(tensor);
  auto tensor_data = static_cast<float16 *>(tensor->data_c());
  auto pad_d = pad_list[kDim0] + pad_list[kDim1];
  auto pad_h = pad_list[kDim2] + pad_list[kDim3];
  auto pad_w = pad_list[kDim4] + pad_list[kDim5];
  auto len_d = fd + pad_d;
  auto len_h = fh + pad_h;
  auto len_w = fw + pad_w;
  for (int64_t nn = 0; nn < fn; nn++) {
    for (int64_t cc = 0; cc < fc; cc++) {
      int64_t start_d = 0;
      for (int64_t di = 0; di < dd; di++) {
        auto v_kd = start_d + kd <= len_d ? kd : len_d - start_d;
        int64_t start_h = 0;
        for (int64_t hi = 0; hi < dh; hi++) {
          auto v_kh = start_h + kh <= len_h ? kh : len_h - start_h;
          int64_t start_w = 0;
          for (int64_t wi = 0; wi < dw; wi++) {
            auto v_kw = start_w + kw < len_w ? kw : len_w - start_w;
            auto vaild_d = GetInterSection(start_d, start_d + kd, pad_list[kDim0], pad_list[kDim0] + fd);
            auto vaild_h = GetInterSection(start_h, start_h + kh, pad_list[kDim2], pad_list[kDim2] + fh);
            auto vaild_w = GetInterSection(start_w, start_w + kw, pad_list[kDim4], pad_list[kDim4] + fw);
            auto vaild_data = vaild_d * vaild_h * vaild_w;
            auto vaild_kernel = v_kd * v_kh * v_kw;
            auto valid_dividend = count_include_pad ? vaild_kernel : vaild_data;
            if (valid_dividend == 0) {
              MS_LOG(EXCEPTION) << "Dividend 'valid_dividend' should not be 0.";
            }
            float val = 1.0 / valid_dividend;
            *tensor_data = float16(val);
            ++tensor_data;
            start_w += sw;
          }
          start_h += sh;
        }
        start_d += sd;
      }
    }
  }
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat16, assist_shape);
  auto kernel_graph = func_graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto value_node = kernel_graph->NewValueNode(x_abstract, tensor);
  kernel_graph->AddValueNodeToGraph(value_node);
  common::AnfAlgo::SetOutputInferTypeAndShape({kNumberTypeFloat16}, {assist_shape}, value_node.get());
  return value_node;
}
}  // namespace

AnfNodePtr ConstructFilterValueNode(const FuncGraphPtr &func_graph, float val, const ShapeVector &assist_shape,
                                    const ShapeVector &infer_shape, int64_t cnt) {
  tensor::TensorPtr assist_tensor = std::make_shared<tensor::Tensor>(kNumberTypeFloat16, assist_shape);
  MS_EXCEPTION_IF_NULL(assist_tensor);
  TensorTypePtr tensor_type = std::make_shared<TensorType>(kFloat16);
  tensor::DeviceInfo device_info{kOpFormat_FRACTAL_Z_3D, tensor_type, kOpFormat_FRACTAL_Z_3D};
  assist_tensor->set_device_info(device_info);
  auto tensor_data = static_cast<float16 *>(assist_tensor->data_c());
  for (int64_t i = 0; i < cnt; ++i) {
    for (int64_t j = 0; j < kC0; ++j) {
      for (int64_t k = 0; k < kC0; ++k) {
        float t = j == k ? val : 0;
        *tensor_data = float16(t);
        ++tensor_data;
      }
    }
  }

  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat16, assist_shape);
  auto kernel_graph = func_graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto value_node = kernel_graph->NewValueNode(x_abstract, assist_tensor);
  kernel_graph->AddValueNodeToGraph(value_node);
  common::AnfAlgo::SetOutputInferTypeAndShape({kNumberTypeFloat16}, {infer_shape}, value_node.get());
  return value_node;
}

const BaseRef AvgPool3DFusion::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  return VectorRef({prim::kPrimAvgPool3DD, Xs});
}

const AnfNodePtr AvgPool3DFusion::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                          const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  auto avg_pool_3d_node = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(avg_pool_3d_node);
  if (common::AnfAlgo::IsDynamicShape(avg_pool_3d_node)) {
    MS_LOG(EXCEPTION) << "AvgPool3D don't support dynamic shape in ascend yet, node: "
                      << avg_pool_3d_node->fullname_with_scope();
  }
  if (avg_pool_3d_node->size() != kAvgPool3DInputNum + 1) {
    MS_LOG(INFO) << "The node " << avg_pool_3d_node->DebugString() << " is not equal to " << kAvgPool3DInputNum
                 << " inputs. Can not do fusion.";
    return nullptr;
  }
  auto dims_in = common::AnfAlgo::GetPrevNodeOutputInferShape(avg_pool_3d_node, 0);
  auto dims_out = common::AnfAlgo::GetOutputInferShape(avg_pool_3d_node, 0);
  if (dims_in.size() < k5DInferDims || dims_out.size() < k5DInferDims) {
    MS_LOG(EXCEPTION) << "AvgPool3D's in_out infer shape dims can not be less " << k5DInferDims
                      << ", but got in_shape is " << dims_in.size() << "-D, out_shape is " << dims_out.size()
                      << trace::DumpSourceLines(node);
  }
  auto fn = dims_in[kDim0];
  auto fc = dims_in[kDim1];
  auto fd = dims_in[kDim2];
  auto fh = dims_in[kDim3];
  auto fw = dims_in[kDim4];
  auto dout = dims_out[kDim2];
  auto dh = dims_out[kDim3];
  auto dw = dims_out[kDim4];
  // kernel size
  int64_t kd;
  int64_t kh;
  int64_t kw;
  if (!GetKernelSize(avg_pool_3d_node, &kd, &kh, &kw)) {
    MS_LOG(EXCEPTION) << "Get kernel size failed" << trace::DumpSourceLines(node);
  }
  // strides
  int64_t sd;
  int64_t sh;
  int64_t sw;
  if (!GetStrideSize(avg_pool_3d_node, &sd, &sh, &sw)) {
    MS_LOG(EXCEPTION) << "Get stride size failed" << trace::DumpSourceLines(node);
  }
  std::vector<int64_t> pad_list;
  bool count_include_pad = true;
  bool ceil_mode = false;
  int64_t divisor_override = 0;
  GetAttrs(avg_pool_3d_node, &pad_list, &count_include_pad, &ceil_mode, &divisor_override);
  if (IsVectorImpl(fh, fw, kh, kw, pad_list)) {
    MS_LOG(INFO) << "No need fusion";
    return nullptr;
  }
  std::vector<AnfNodePtr> new_inputs{NewValueNode(std::make_shared<Primitive>(prim::kPrimAvgPool3DD->name()))};
  (void)new_inputs.insert(new_inputs.cend(), avg_pool_3d_node->inputs().cbegin() + 1,
                          avg_pool_3d_node->inputs().cend());
  // assist node 1
  auto filter_node = ConstructFilter(func_graph, pad_list, fc, kd, kh, kw, ceil_mode, divisor_override);
  new_inputs.push_back(filter_node);
  MS_EXCEPTION_IF_NULL(filter_node);
  // assist node 2
  if ((!IsZeroPads(pad_list) || ceil_mode) && divisor_override == 0) {
    auto multiplier = ConstructMultiplier(func_graph, fn, fc, fd, fh, fw, dout, dh, dw, kd, kh, kw, sd, sh, sw,
                                          pad_list, count_include_pad);
    new_inputs.push_back(multiplier);
  }
  auto new_3d = NewCNode(new_inputs, func_graph);
  MS_EXCEPTION_IF_NULL(new_3d);
  new_3d->set_scope(avg_pool_3d_node->scope());
  new_3d->set_abstract(avg_pool_3d_node->abstract());
  common::AnfAlgo::CopyNodeAttrs(avg_pool_3d_node, new_3d);
  return new_3d;
}
}  // namespace opt
}  // namespace mindspore
