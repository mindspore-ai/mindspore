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

#include "backend/optimizer/ascend/ir_fusion/avgpool_3d_grad_fusion.h"
#include <vector>
#include <memory>
#include <string>
#include <algorithm>
#include "backend/optimizer/ascend/ir_fusion/avgpool_3d_fusion.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "backend/optimizer/common/helper.h"
#include "base/core_ops.h"
#include "utils/utils.h"

namespace mindspore {
namespace opt {
namespace {
constexpr size_t kAvgPool3DGradInputNum = 1;
constexpr size_t k5DInferDims = 5;
constexpr size_t kKernelDims = 3;
constexpr size_t kStridesDims = 3;
constexpr size_t kOrigShapeDims = 5;
constexpr size_t kShapeDims = 6;
constexpr size_t kPadDims = 6;
constexpr int64_t kC0 = 16;

void GetAttrs(const AnfNodePtr &node, std::vector<int64_t> *kernel_size, std::vector<int64_t> *strides,
              std::vector<int64_t> *pad_list, std::vector<int64_t> *origin_input_shape, bool *ceil_mode,
              bool *count_include_pad, int64_t *divisor_override, std::string *format_str) {
  MS_EXCEPTION_IF_NULL(node);
  // attr kernel size
  if (!AnfAlgo::HasNodeAttr("kernel_size", node->cast<CNodePtr>())) {
    MS_LOG(EXCEPTION) << "AvgPool3D should has attr kernel_size";
  }
  *kernel_size = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(node, "kernel_size");
  // attr strides
  if (!AnfAlgo::HasNodeAttr("strides", node->cast<CNodePtr>())) {
    MS_LOG(EXCEPTION) << "AvgPool3D should has attr strides";
  }
  *strides = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(node, "strides");
  // sttr pad_list
  if (!AnfAlgo::HasNodeAttr("pad_list", node->cast<CNodePtr>())) {
    MS_LOG(EXCEPTION) << "AvgPool3D should has attr pad_list";
  }
  *pad_list = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(node, "pad_list");
  // attr origin input shape
  if (!AnfAlgo::HasNodeAttr("origin_input_shape", node->cast<CNodePtr>())) {
    MS_LOG(EXCEPTION) << "AvgPool3D should has attr origin_input_shape";
  }
  *origin_input_shape = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(node, "origin_input_shape");
  // attr count include pad
  if (AnfAlgo::HasNodeAttr("count_include_pad", node->cast<CNodePtr>())) {
    *count_include_pad = AnfAlgo::GetNodeAttr<bool>(node, "count_include_pad");
  }
  // attr ceil mode
  if (AnfAlgo::HasNodeAttr("ceil_mode", node->cast<CNodePtr>())) {
    *ceil_mode = AnfAlgo::GetNodeAttr<bool>(node, "ceil_mode");
  }
  // attr divisor override
  if (AnfAlgo::HasNodeAttr("divisor_override", node->cast<CNodePtr>())) {
    *divisor_override = AnfAlgo::GetNodeAttr<int64_t>(node, "divisor_override");
  }
  if (AnfAlgo::HasNodeAttr("format", node->cast<CNodePtr>())) {
    *format_str = AnfAlgo::GetNodeAttr<std::string>(node, "format");
  }
}

bool IsVectorImpl(const std::vector<int64_t> &fp_shape, const std::vector<int64_t> &k_size,
                  const std::vector<int64_t> &pad_list) {
  // NCDHW
  auto fd = fp_shape[kDim2];
  auto fh = fp_shape[kDim3];
  auto fw = fp_shape[kDim4];
  auto kd = k_size[kDim0];
  auto kh = k_size[kDim1];
  auto kw = k_size[kDim2];
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
  std::vector<int64_t> assist_shape = {c1 * kd * kh * kw, 1, kC0, kC0};  // frac_z_3d
  std::vector<size_t> infer_shape = {IntToSize(1), LongToSize(fc), LongToSize(kd), LongToSize(kh), LongToSize(kw)};
  float val = 1.0;
  if (divisor_override) {
    val = 1.0 / divisor_override;
  } else if (IsZeroPads(pad_list) && !ceil_mode) {
    val = 1.0 / (kd * kh * kw);
  }
  // create value node
  int64_t cnt = c1 * kd * kh * kw;
  return ConstructFilterValueNode(func_graph, val, assist_shape, infer_shape, cnt);
}

AnfNodePtr ConstructMultiplier(const FuncGraphPtr &func_graph, const std::vector<size_t> &ori_shape,
                               const std::vector<int64_t> &ori_input_shape, const std::vector<int64_t> &kernel_size,
                               const std::vector<int64_t> &strides, const std::vector<int64_t> &pad_list,
                               bool count_include_pad) {
  MS_EXCEPTION_IF_NULL(func_graph);
  //  assist tensor 2
  std::vector<int64_t> grad_shape;
  (void)std::transform(ori_shape.begin(), ori_shape.end(), std::back_inserter(grad_shape), SizeToLong);
  std::vector<int64_t> assist_shape = grad_shape;  // NCDHW
  tensor::TensorPtr tensor = std::make_shared<tensor::Tensor>(kNumberTypeFloat16, assist_shape);
  MS_EXCEPTION_IF_NULL(tensor);
  auto tensor_data = reinterpret_cast<float16 *>(tensor->data_c());
  auto pad_d = pad_list[kDim0] + pad_list[kDim1];
  auto pad_h = pad_list[kDim2] + pad_list[kDim3];
  auto pad_w = pad_list[kDim4] + pad_list[kDim5];
  auto len_d = ori_input_shape[kDim2] + pad_d;
  auto len_h = ori_input_shape[kDim3] + pad_h;
  auto len_w = ori_input_shape[kDim4] + pad_w;
  for (int64_t nn = 0; nn < grad_shape[kDim0]; nn++) {
    for (int64_t cc = 0; cc < grad_shape[kDim1]; cc++) {
      int64_t start_d = 0;
      for (int64_t di = 0; di < grad_shape[kDim2]; di++) {
        int64_t start_h = 0;
        for (int64_t hi = 0; hi < grad_shape[kDim3]; hi++) {
          int64_t start_w = 0;
          for (int64_t wi = 0; wi < grad_shape[kDim4]; wi++) {
            int64_t valid_d = 0;
            int64_t valid_h = 0;
            int64_t valid_w = 0;
            if (count_include_pad) {
              valid_d = start_d + kernel_size[kDim0] <= len_d ? kernel_size[kDim0] : len_d - start_d;
              valid_h = start_h + kernel_size[kDim1] <= len_h ? kernel_size[kDim1] : len_h - start_h;
              valid_w = start_w + kernel_size[kDim2] <= len_w ? kernel_size[kDim2] : len_w - start_w;
            } else {
              valid_d = std::min(start_d + kernel_size[kDim0], pad_list[kDim0] + ori_input_shape[kDim2]) -
                        std::max(pad_list[kDim0], start_d);
              valid_h = std::min(start_h + kernel_size[kDim1], pad_list[kDim2] + ori_input_shape[kDim3]) -
                        std::max(pad_list[kDim2], start_h);
              valid_w = std::min(start_w + kernel_size[kDim2], pad_list[kDim4] + ori_input_shape[kDim4]) -
                        std::max(pad_list[kDim4], start_w);
            }
            auto valid_data = valid_d * valid_h * valid_w;
            if (valid_data == 0) {
              MS_LOG(EXCEPTION) << "Divisor 'valid_data' should not be 0.";
            }
            float val = 1.0 / valid_data;
            *tensor_data = float16(val);
            ++tensor_data;
            start_w += strides[kDim2];
          }
          start_h += strides[kDim1];
        }
        start_d += strides[kDim0];
      }
    }
  }
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat16, assist_shape);
  auto kernel_graph = func_graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto value_node = kernel_graph->NewValueNode(x_abstract, tensor);
  kernel_graph->AddValueNodeToGraph(value_node);
  AnfAlgo::SetOutputInferTypeAndShape({kNumberTypeFloat16}, {ori_shape}, value_node.get());
  return value_node;
}
}  // namespace

const BaseRef AvgPool3DGradFusion::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  return VectorRef({prim::kPrimAvgPool3DGrad, Xs});
}

const AnfNodePtr AvgPool3DGradFusion::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                              const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  auto avg_pool_3d_grad_node = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(avg_pool_3d_grad_node);
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
  std::vector<AnfNodePtr> new_inputs{NewValueNode(std::make_shared<Primitive>(prim::kPrimAvgPool3DGrad->name()))};
  (void)new_inputs.insert(new_inputs.end(), avg_pool_3d_grad_node->inputs().begin() + 1,
                          avg_pool_3d_grad_node->inputs().end());
  // assist node 1
  auto kd = kernel_size[kDim0];
  auto kh = kernel_size[kDim1];
  auto kw = kernel_size[kDim2];
  auto fc = origin_input_shape[kDim1];
  auto filter_node = ConstructFilter(func_graph, pad_list, fc, kd, kh, kw, divisor_override, ceil_mode);
  new_inputs.push_back(filter_node);
  MS_EXCEPTION_IF_NULL(filter_node);

  // after input to attr, the first input should be the 'grads', the index is 0;
  auto dims_in = AnfAlgo::GetPrevNodeOutputInferShape(avg_pool_3d_grad_node, 0);

  // assist node 2
  if (divisor_override == 0 && (!IsZeroPads(pad_list) || ceil_mode)) {
    auto multiplier =
      ConstructMultiplier(func_graph, dims_in, origin_input_shape, kernel_size, strides, pad_list, count_include_pad);
    new_inputs.push_back(multiplier);
  }
  auto new_3d_grad = func_graph->NewCNode(new_inputs);
  MS_EXCEPTION_IF_NULL(new_3d_grad);
  new_3d_grad->set_scope(avg_pool_3d_grad_node->scope());
  new_3d_grad->set_abstract(avg_pool_3d_grad_node->abstract());
  AnfAlgo::CopyNodeAttrs(avg_pool_3d_grad_node, new_3d_grad);
  const int64_t dim_one = SizeToLong(1);
  AnfAlgo::SetNodeAttr("kernel_size", MakeValue(std::vector<int64_t>{dim_one, dim_one, kd, kh, kw}), new_3d_grad);
  AnfAlgo::SetNodeAttr(
    "strides", MakeValue(std::vector<int64_t>{dim_one, dim_one, strides[kDim0], strides[kDim1], strides[kDim2]}),
    new_3d_grad);
  return new_3d_grad;
}
}  // namespace opt
}  // namespace mindspore
