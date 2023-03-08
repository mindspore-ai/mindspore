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

#include "plugin/device/ascend/optimizer/ir_fusion/avgpool_3d_grad_fusion.h"
#include <vector>
#include <memory>
#include <string>
#include <algorithm>
#include "plugin/device/ascend/optimizer/ir_fusion/avgpool_3d_fusion.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore {
namespace opt {
namespace {
constexpr size_t kAvgPool3DGradInputNum = 1;
constexpr int64_t kC0 = 16;

void GetAttrs(const AnfNodePtr &node, std::vector<int64_t> *kernel_size, std::vector<int64_t> *strides,
              std::vector<int64_t> *pad_list, std::vector<int64_t> *origin_input_shape, bool *ceil_mode,
              bool *count_include_pad, int64_t *divisor_override, std::string *format_str) {
  MS_EXCEPTION_IF_NULL(node);
  // attr kernel size
  if (!common::AnfAlgo::HasNodeAttr("kernel_size", node->cast<CNodePtr>())) {
    MS_LOG(EXCEPTION) << "AvgPool3D should has attr kernel_size" << trace::DumpSourceLines(node);
  }
  *kernel_size = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(node, "kernel_size");
  // attr strides
  if (!common::AnfAlgo::HasNodeAttr("strides", node->cast<CNodePtr>())) {
    MS_LOG(EXCEPTION) << "AvgPool3D should has attr strides" << trace::DumpSourceLines(node);
  }
  *strides = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(node, "strides");
  // sttr pad_list
  if (!common::AnfAlgo::HasNodeAttr("pad_list", node->cast<CNodePtr>())) {
    MS_LOG(EXCEPTION) << "AvgPool3D should has attr pad_list" << trace::DumpSourceLines(node);
  }
  *pad_list = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(node, "pad_list");
  // attr origin input shape
  if (!common::AnfAlgo::HasNodeAttr("origin_input_shape", node->cast<CNodePtr>())) {
    MS_LOG(EXCEPTION) << "AvgPool3D should has attr origin_input_shape" << trace::DumpSourceLines(node);
  }
  *origin_input_shape = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(node, "origin_input_shape");
  // attr count include pad
  if (common::AnfAlgo::HasNodeAttr("count_include_pad", node->cast<CNodePtr>())) {
    *count_include_pad = common::AnfAlgo::GetNodeAttr<bool>(node, "count_include_pad");
  }
  // attr ceil mode
  if (common::AnfAlgo::HasNodeAttr("ceil_mode", node->cast<CNodePtr>())) {
    *ceil_mode = common::AnfAlgo::GetNodeAttr<bool>(node, "ceil_mode");
  }
  // attr divisor override
  if (common::AnfAlgo::HasNodeAttr("divisor_override", node->cast<CNodePtr>())) {
    *divisor_override = common::AnfAlgo::GetNodeAttr<int64_t>(node, "divisor_override");
  }
  if (common::AnfAlgo::HasNodeAttr("format", node->cast<CNodePtr>())) {
    *format_str = common::AnfAlgo::GetNodeAttr<std::string>(node, "format");
  }
}

bool IsVectorImpl(const std::vector<int64_t> &fp_shape, const std::vector<int64_t> &k_size,
                  const std::vector<int64_t> &pad_list) {
  // NCDHW
  auto fd = fp_shape[kDim2];
  auto fh = fp_shape[kDim3];
  auto fw = fp_shape[kDim4];
  auto kd = k_size[kDim2];
  auto kh = k_size[kDim3];
  auto kw = k_size[kDim4];
  bool flag1 = kd >= fd + pad_list[kDim0] + pad_list[kDim1];
  bool flag2 = kh >= fh + pad_list[kDim2] + pad_list[kDim3];
  bool flag3 = kw >= fw + pad_list[kDim4] + pad_list[kDim5];
  if (flag1 && flag2 && flag3) {
    return true;
  }
  return false;
}

bool IsZeroPads(const std::vector<int64_t> &pad_list) {
  return std::all_of(pad_list.begin(), pad_list.end(), [](int64_t item) { return item == 0; });
}

AnfNodePtr ConstructFilter(const FuncGraphPtr &func_graph, const std::vector<int64_t> &pad_list, int64_t fc, int64_t kd,
                           int64_t kh, int64_t kw, int64_t divisor_override, bool ceil_mode) {
  MS_EXCEPTION_IF_NULL(func_graph);
  //  assist tensor 1
  int64_t c1 = (fc + kC0 - 1) / kC0;
  ShapeVector assist_shape = {c1 * kd * kh * kw, 1, kC0, kC0};  // frac_z_3d
  ShapeVector infer_shape = {1, fc, kd, kh, kw};
  float val = 1.0;
  if (divisor_override != 0) {
    val = 1.0 / divisor_override;
  } else if (IsZeroPads(pad_list) && !ceil_mode) {
    val = 1.0 / static_cast<float>(kd * kh * kw);
  }
  // create value node
  int64_t cnt = c1 * kd * kh * kw;
  return ConstructFilterValueNode(func_graph, val, assist_shape, infer_shape, cnt);
}

AnfNodePtr ConstructMultiplier(const FuncGraphPtr &func_graph, const ShapeVector &ori_shape,
                               const std::vector<int64_t> &ori_input_shape, const std::vector<int64_t> &kernel_size,
                               const std::vector<int64_t> &strides, const std::vector<int64_t> &pad_list,
                               bool count_include_pad) {
  MS_EXCEPTION_IF_NULL(func_graph);
  //  assist tensor 2
  std::vector<int64_t> assist_shape = ori_shape;  // NCDHW
  tensor::TensorPtr tensor = std::make_shared<tensor::Tensor>(kNumberTypeFloat16, assist_shape);
  MS_EXCEPTION_IF_NULL(tensor);
  auto tensor_data = static_cast<float16 *>(tensor->data_c());
  auto pad_d = pad_list[kDim0] + pad_list[kDim1];
  auto pad_h = pad_list[kDim2] + pad_list[kDim3];
  auto pad_w = pad_list[kDim4] + pad_list[kDim5];
  auto len_d = ori_input_shape[kDim2] + pad_d;
  auto len_h = ori_input_shape[kDim3] + pad_h;
  auto len_w = ori_input_shape[kDim4] + pad_w;
  for (int64_t nn = 0; nn < ori_shape[kDim0]; nn++) {
    for (int64_t cc = 0; cc < ori_shape[kDim1]; cc++) {
      int64_t start_d = 0;
      for (int64_t di = 0; di < ori_shape[kDim2]; di++) {
        int64_t start_h = 0;
        for (int64_t hi = 0; hi < ori_shape[kDim3]; hi++) {
          int64_t start_w = 0;
          for (int64_t wi = 0; wi < ori_shape[kDim4]; wi++) {
            int64_t valid_d = 0;
            int64_t valid_h = 0;
            int64_t valid_w = 0;
            if (count_include_pad) {
              valid_d = start_d + kernel_size[kDim2] <= len_d ? kernel_size[kDim2] : len_d - start_d;
              valid_h = start_h + kernel_size[kDim3] <= len_h ? kernel_size[kDim3] : len_h - start_h;
              valid_w = start_w + kernel_size[kDim4] <= len_w ? kernel_size[kDim4] : len_w - start_w;
            } else {
              valid_d = std::min(start_d + kernel_size[kDim2], pad_list[kDim0] + ori_input_shape[kDim2]) -
                        std::max(pad_list[kDim0], start_d);
              valid_h = std::min(start_h + kernel_size[kDim3], pad_list[kDim2] + ori_input_shape[kDim3]) -
                        std::max(pad_list[kDim2], start_h);
              valid_w = std::min(start_w + kernel_size[kDim4], pad_list[kDim4] + ori_input_shape[kDim4]) -
                        std::max(pad_list[kDim4], start_w);
            }
            auto valid_data = valid_d * valid_h * valid_w;
            if (valid_data == 0) {
              MS_LOG(EXCEPTION) << "Divisor 'valid_data' should not be 0.";
            }
            float val = 1.0 / valid_data;
            *tensor_data = float16(val);
            ++tensor_data;
            start_w += strides[kDim4];
          }
          start_h += strides[kDim3];
        }
        start_d += strides[kDim2];
      }
    }
  }
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat16, assist_shape);
  auto kernel_graph = func_graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto value_node = kernel_graph->NewValueNode(x_abstract, tensor);
  kernel_graph->AddValueNodeToGraph(value_node);
  common::AnfAlgo::SetOutputInferTypeAndShape({kNumberTypeFloat16}, {ori_shape}, value_node.get());
  return value_node;
}
}  // namespace

const BaseRef AvgPool3DGradFusion::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  return VectorRef({prim::kPrimAvgPool3DGradD, Xs});
}

const AnfNodePtr AvgPool3DGradFusion::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                              const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  auto avg_pool_3d_grad_node = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(avg_pool_3d_grad_node);
  if (common::AnfAlgo::IsDynamicShape(avg_pool_3d_grad_node)) {
    MS_LOG(EXCEPTION) << "AvgPool3DGrad don't support dynamic shape in ascend yet, node: "
                      << avg_pool_3d_grad_node->fullname_with_scope();
  }
  if (avg_pool_3d_grad_node->size() != kAvgPool3DGradInputNum + 1) {
    MS_LOG(INFO) << "The node " << avg_pool_3d_grad_node->DebugString() << " is not equal to " << kAvgPool3DGradInputNum
                 << " inputs. Can not do fusion.";
    return nullptr;
  }
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> strides;
  std::vector<int64_t> pad_list;
  std::vector<int64_t> origin_input_shape;
  bool ceil_mode = false;
  bool count_include_pad = true;
  int64_t divisor_override = 0;
  std::string format_str;
  GetAttrs(avg_pool_3d_grad_node, &kernel_size, &strides, &pad_list, &origin_input_shape, &ceil_mode,
           &count_include_pad, &divisor_override, &format_str);
  if (IsVectorImpl(origin_input_shape, kernel_size, pad_list)) {
    MS_LOG(INFO) << "No need fusion";
    return nullptr;
  }
  std::vector<AnfNodePtr> new_inputs{NewValueNode(std::make_shared<Primitive>(prim::kPrimAvgPool3DGradD->name()))};
  (void)new_inputs.insert(new_inputs.cend(), avg_pool_3d_grad_node->inputs().cbegin() + 1,
                          avg_pool_3d_grad_node->inputs().cend());
  // assist node 1
  auto kd = kernel_size[kDim2];
  auto kh = kernel_size[kDim3];
  auto kw = kernel_size[kDim4];
  auto fc = origin_input_shape[kDim1];
  auto filter_node = ConstructFilter(func_graph, pad_list, fc, kd, kh, kw, divisor_override, ceil_mode);
  new_inputs.push_back(filter_node);
  MS_EXCEPTION_IF_NULL(filter_node);

  // after input to attr, the first input should be the 'grads', the index is 0;
  auto dims_in = common::AnfAlgo::GetPrevNodeOutputInferShape(avg_pool_3d_grad_node, 0);

  // assist node 2
  if (divisor_override == 0 && (!IsZeroPads(pad_list) || ceil_mode)) {
    auto multiplier =
      ConstructMultiplier(func_graph, dims_in, origin_input_shape, kernel_size, strides, pad_list, count_include_pad);
    new_inputs.push_back(multiplier);
  }
  auto new_3d_grad = NewCNode(new_inputs, func_graph);
  MS_EXCEPTION_IF_NULL(new_3d_grad);
  new_3d_grad->set_scope(avg_pool_3d_grad_node->scope());
  new_3d_grad->set_abstract(avg_pool_3d_grad_node->abstract());
  common::AnfAlgo::CopyNodeAttrs(avg_pool_3d_grad_node, new_3d_grad);
  return new_3d_grad;
}
}  // namespace opt
}  // namespace mindspore
