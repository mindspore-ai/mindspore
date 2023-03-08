/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/optimizer/ir_fusion/avgpool_fusion.h"
#include <vector>
#include <memory>
#include <string>
#include <algorithm>
#include <functional>
#include "include/common/utils/utils.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore {
namespace opt {
namespace {
constexpr size_t kAvgPoolInputDimSize = 4;
constexpr int64_t kMaxStride = 63;
constexpr int64_t kAvgKernelSizeHW = 255;
constexpr int64_t kCin = 16;
constexpr int64_t kCout = 16;
constexpr int64_t kDivNumTwo = 2;

bool CheckAttrValue(const AnfNodePtr &avgpool, int64_t input_c, int64_t output_w, int64_t ksize_h, int64_t ksize_w,
                    int64_t stride_h, int64_t stride_w, bool is_dynamic) {
  if (input_c < 0) {
    MS_LOG(DEBUG) << "Input C dim is " << input_c << ", skip fusion node [" << avgpool->fullname_with_scope() << "].";
    return false;
  }

  if (!is_dynamic && output_w == 1) {
    MS_LOG(INFO) << "Avgpool node [" << avgpool->fullname_with_scope() << "] is global or output_w=1, skip fusion.";
    return false;
  }
  if (stride_h > kMaxStride || stride_w > kMaxStride) {
    MS_LOG(INFO) << "Stride of node [" << avgpool->fullname_with_scope() << "] is greater than 63, skip fusion.";
    return false;
  }
  if (is_dynamic && (ksize_h * ksize_w > kAvgKernelSizeHW)) {
    MS_LOG(INFO) << "Stride_h * stride_w is greater than 255, skip fusion node [" << avgpool->fullname_with_scope()
                 << "].";
    return false;
  }
  auto input_dtype = common::AnfAlgo::GetPrevNodeOutputInferDataType(avgpool, 0);
  if (input_dtype == kNumberTypeInt8) {
    MS_LOG(INFO) << "Input dtype is int8, skip fusion node [" << avgpool->fullname_with_scope() << "].";
    return false;
  }

  return true;
}

bool GetAttrValue(const AnfNodePtr &avgpool, const std::vector<int64_t> &input_shape,
                  const std::vector<int64_t> &output_shape, std::string *pad_mode, std::string *format,
                  int64_t *input_c, int64_t *ksize_h, int64_t *ksize_w, int64_t *stride_h, int64_t *stride_w,
                  bool *is_dynamic) {
  MS_EXCEPTION_IF_NULL(avgpool);
  *pad_mode = common::AnfAlgo::GetNodeAttr<std::string>(avgpool, kAttrPadMode);
  *format = common::AnfAlgo::GetNodeAttr<std::string>(avgpool, kAttrFormat);
  auto ksizes = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(avgpool, kAttrKernelSize);
  auto strides = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(avgpool, kAttrStrides);
  if (ksizes.size() != kAvgPoolInputDimSize || strides.size() != kAvgPoolInputDimSize) {
    MS_LOG(INFO) << "Size of kernel_size or strides should be 4, but got kernel size: " << ksizes.size()
                 << ", strides size: " << strides.size() << ", node: " << avgpool->fullname_with_scope();
    return false;
  }
  int64_t output_w;
  if (*format == kOpFormat_NHWC) {
    *input_c = input_shape[kDim3];
    output_w = output_shape[kDim2];
    *ksize_h = ksizes[kDim1];
    *ksize_w = ksizes[kDim2];
    *stride_h = strides[kDim1];
    *stride_w = strides[kDim2];
  } else if (*format == kOpFormat_NCHW) {
    *input_c = input_shape[kDim1];
    output_w = output_shape[kDim3];
    *ksize_h = ksizes[kDim2];
    *ksize_w = ksizes[kDim3];
    *stride_h = strides[kDim2];
    *stride_w = strides[kDim3];
  } else {
    MS_LOG(INFO) << "Format attr should be NCHW or NHWC, but got " << format << ", skip fusion node ["
                 << avgpool->fullname_with_scope() << "].";
    return false;
  }
  *is_dynamic = common::AnfAlgo::IsDynamicShape(avgpool);
  return CheckAttrValue(avgpool, *input_c, output_w, *ksize_h, *ksize_w, *stride_h, *stride_w, *is_dynamic);
}

bool CheckOptimization(int64_t ksize_h, int64_t ksize_w, int64_t stride_h, int64_t stride_w, size_t input_num) {
  if (ksize_h == 1 && ksize_w == 1 && stride_h == 1 && stride_w == 1 && input_num == 1) {
    return true;
  }
  return false;
}

void GenerateCoffeData(const std::vector<int64_t> &coffe_shape, const std::vector<int64_t> &window,
                       const std::vector<int64_t> &stride, const std::vector<int64_t> &pad, int64_t in_h, int64_t in_w,
                       float16 *tensor_data) {
  MS_EXCEPTION_IF_NULL(tensor_data);
  for (int64_t m = 0; m < coffe_shape[kDim0]; ++m) {
    for (int64_t n = 0; n < coffe_shape[kDim1]; ++n) {
      for (int64_t i = 0; i < coffe_shape[kDim2]; ++i) {
        for (int64_t j = 0; j < coffe_shape[kDim3]; ++j) {
          for (int64_t k = 0; k < coffe_shape[kDim4]; ++k) {
            int64_t h_start = i * stride[kDim0] - pad[kDim0];
            int64_t w_start = j * stride[kDim1] - pad[kDim2];
            int64_t h_end = std::min(h_start + window[kDim0], in_h);
            int64_t w_end = std::min(w_start + window[kDim1], in_w);
            h_start = std::max(h_start, int64_t(0));
            w_start = std::max(w_start, int64_t(0));
            float area = std::max((h_end - h_start) * (w_end - w_start), int64_t(1));
            area = 1.0 / area;
            *tensor_data = float16(area);
            ++tensor_data;
          }
        }
      }
    }
  }
}
}  // namespace

ValueNodePtr AvgPoolFusion::ConstructFilterValueNode(const KernelGraphPtr &graph, float factor,
                                                     const std::vector<int64_t> &assist_shape) const {
  MS_EXCEPTION_IF_NULL(graph);
  tensor::TensorPtr assist_tensor = std::make_shared<tensor::Tensor>(kNumberTypeFloat16, assist_shape);
  auto tensor_data = reinterpret_cast<float16 *>(assist_tensor->data_c());
  int64_t assist_size = std::accumulate(assist_shape.begin(), assist_shape.end(), 1, std::multiplies<int64_t>());
  for (int64_t i = 0; i < assist_size; ++i) {
    *tensor_data = float16(factor);
    ++tensor_data;
  }

  auto assist_abstract = std::make_shared<abstract::AbstractTensor>(kFloat16, assist_shape);
  auto value_node = graph->NewValueNode(assist_abstract, assist_tensor);
  common::AnfAlgo::SetOutputInferTypeAndShape({kNumberTypeFloat16}, {assist_shape}, value_node.get());
  value_node->set_fracz_group(assist_shape[0]);
  return value_node;
}

ValueNodePtr AvgPoolFusion::ConstructFilterValueNodeDynamic(const KernelGraphPtr &graph, float factor,
                                                            const std::vector<int64_t> &assist_shape,
                                                            const std::vector<int64_t> &host_shape) const {
  MS_EXCEPTION_IF_NULL(graph);
  tensor::TensorPtr assist_tensor = std::make_shared<tensor::Tensor>(kNumberTypeFloat16, assist_shape);
  TensorTypePtr tensor_type = std::make_shared<TensorType>(kFloat16);
  tensor::DeviceInfo device_info{kOpFormat_FRACTAL_Z, tensor_type, kOpFormat_FRACTAL_Z};
  assist_tensor->set_device_info(device_info);
  auto tensor_data = reinterpret_cast<float16 *>(assist_tensor->data_c());
  for (int64_t i = 0; i < assist_shape[kDim0]; ++i) {
    for (int64_t j = 0; j < assist_shape[kDim1]; ++j) {
      for (int64_t k = 0; k < assist_shape[kDim2]; ++k) {
        for (int64_t l = 0; l < assist_shape[kDim3]; ++l) {
          *tensor_data = k == l ? float16(factor) : float16(0);
          ++tensor_data;
        }
      }
    }
  }

  auto assist_abstract = std::make_shared<abstract::AbstractTensor>(kFloat16, assist_shape);
  auto value_node = graph->NewValueNode(assist_abstract, assist_tensor);
  common::AnfAlgo::SetOutputInferTypeAndShape({kNumberTypeFloat16}, {host_shape}, value_node.get());
  return value_node;
}

ValueNodePtr AvgPoolFusion::ConstructCoffeValueNode(const KernelGraphPtr &graph, const std::string &format,
                                                    const std::vector<int64_t> &avg_in_shape,
                                                    const std::vector<int64_t> &avg_out_shape,
                                                    const std::vector<int64_t> &window,
                                                    const std::vector<int64_t> &stride) const {
  MS_EXCEPTION_IF_NULL(graph);
  std::vector<int64_t> dilation = {1, 1};
  bool is_nhwc = format == kOpFormat_NHWC;
  int64_t out_h = is_nhwc ? avg_out_shape[kDim1] : avg_out_shape[kDim2];
  int64_t out_w = is_nhwc ? avg_out_shape[kDim2] : avg_out_shape[kDim3];
  int64_t out_c = is_nhwc ? avg_out_shape[kDim3] : avg_out_shape[kDim1];
  int64_t in_h = is_nhwc ? avg_in_shape[kDim1] : avg_in_shape[kDim2];
  int64_t in_w = is_nhwc ? avg_in_shape[kDim2] : avg_in_shape[kDim3];
  int64_t out_c0 = 16;
  int64_t out_c1 = (out_c + out_c0 - 1) / out_c0;
  int64_t pad_row = (out_h - 1) * stride[0] + ((window[0] - 1) * dilation[0] + 1) - in_h;
  int64_t pad_col = (out_w - 1) * stride[1] + ((window[1] - 1) * dilation[1] + 1) - in_w;
  int64_t pad_top = pad_row / kDivNumTwo;
  int64_t pad_bottom = pad_row - pad_top;
  int64_t pad_left = pad_col / kDivNumTwo;
  int64_t pad_right = pad_col - pad_left;
  pad_top = std::max(pad_top, int64_t(0));
  pad_bottom = std::max(pad_bottom, int64_t(0));
  pad_left = std::max(pad_left, int64_t(0));
  pad_right = std::max(pad_right, int64_t(0));
  std::vector<int64_t> pad = {pad_top, pad_bottom, pad_left, pad_right};

  std::vector<int64_t> coffe_shape = {1, out_c1, out_h, out_w, out_c0};
  tensor::TensorPtr coffe_tensor = std::make_shared<tensor::Tensor>(kNumberTypeFloat16, coffe_shape);
  TensorTypePtr tensor_type = std::make_shared<TensorType>(kFloat16);
  tensor::DeviceInfo device_info{kOpFormat_NC1HWC0, tensor_type, kOpFormat_NC1HWC0};
  coffe_tensor->set_device_info(device_info);
  auto tensor_data = reinterpret_cast<float16 *>(coffe_tensor->data_c());
  GenerateCoffeData(coffe_shape, window, stride, pad, in_h, in_w, tensor_data);
  auto coffe_abstract = std::make_shared<abstract::AbstractTensor>(kFloat16, coffe_shape);
  auto value_node = graph->NewValueNode(coffe_abstract, coffe_tensor);
  std::vector<int64_t> host_shape = {1, out_c, out_h, out_w};
  common::AnfAlgo::SetOutputInferTypeAndShape({kNumberTypeFloat16}, {host_shape}, value_node.get());
  graph->AddValueNodeToGraph(value_node);
  return value_node;
}

AnfNodePtr AvgPoolFusion::AddMul(const KernelGraphPtr &graph, const CNodePtr &avgpool, const AnfNodePtr &coffe) const {
  MS_EXCEPTION_IF_NULL(graph);
  std::vector<AnfNodePtr> mul_inputs = {NewValueNode(std::make_shared<Primitive>(kMulOpName)), avgpool, coffe};
  auto mul = NewCNode(mul_inputs, graph);
  mul->set_abstract(avgpool->abstract());
  mul->set_scope(avgpool->scope());
  return mul;
}

const BaseRef AvgPoolFusion::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  return VectorRef({prim::kPrimAvgPool, Xs});
}

const AnfNodePtr AvgPoolFusion::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                        const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_graph = func_graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto avgpool = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(avgpool);
  if (common::AnfAlgo::HasNodeAttr(kAttrVisited, avgpool)) {
    return nullptr;
  }
  common::AnfAlgo::SetNodeAttr(kAttrVisited, MakeValue(true), avgpool);

  ShapeVector input_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(avgpool, 0);
  ShapeVector output_shape = common::AnfAlgo::GetOutputInferShape(avgpool, 0);
  if (input_shape.size() != kAvgPoolInputDimSize || output_shape.size() != kAvgPoolInputDimSize) {
    MS_LOG(INFO) << "Node [" << avgpool->fullname_with_scope() << "]'s input or output dim size is not 4, skip fusion.";
    return nullptr;
  }

  std::string pad_mode;
  std::string format;
  int64_t input_c = 0;
  int64_t ksize_h = 0;
  int64_t ksize_w = 0;
  int64_t stride_h = 0;
  int64_t stride_w = 0;
  bool is_dynamic = false;
  if (!GetAttrValue(avgpool, input_shape, output_shape, &pad_mode, &format, &input_c, &ksize_h, &ksize_w, &stride_h,
                    &stride_w, &is_dynamic)) {
    return nullptr;
  }

  size_t input_num = common::AnfAlgo::GetInputTensorNum(avgpool);
  if (CheckOptimization(ksize_h, ksize_w, stride_h, stride_w, input_num)) {
    return avgpool->input(1);
  }
  int64_t input_c1 = (input_c + kCout - 1) / kCout;
  int64_t matrix_size = is_dynamic ? input_c1 * ksize_h * ksize_w * kCin * kCout : input_c * ksize_h * ksize_w;
  if (matrix_size < 0) {
    MS_LOG(INFO) << "Filter matrix size is negative, skip fusion node [" << avgpool->fullname_with_scope() << "].";
    return nullptr;
  }

  std::vector<int64_t> assit_shape = {input_c, 1, ksize_h, ksize_w};
  std::vector<int64_t> assit_shape_dynamic = {input_c1 * ksize_h * ksize_w, 1, kCin, kCout};
  ValueNodePtr filter_node = nullptr;
  AnfNodePtr ret_node = avgpool;
  if (pad_mode == "VALID") {
    filter_node = is_dynamic ? ConstructFilterValueNodeDynamic(kernel_graph, 1.0, assit_shape_dynamic, assit_shape)
                             : ConstructFilterValueNode(kernel_graph, 1.0 / (ksize_h * ksize_w), assit_shape);
  } else if (pad_mode == "SAME") {
    filter_node = is_dynamic ? ConstructFilterValueNodeDynamic(kernel_graph, 1.0, assit_shape_dynamic, assit_shape)
                             : ConstructFilterValueNode(kernel_graph, 1.0, assit_shape);
    if (!is_dynamic) {
      auto coffe_node =
        ConstructCoffeValueNode(kernel_graph, format, input_shape, output_shape, std::vector<int64_t>{ksize_h, ksize_w},
                                std::vector<int64_t>{stride_h, stride_w});
      auto mul = AddMul(kernel_graph, avgpool, coffe_node);
      ret_node = mul;
    }
  } else {
    MS_LOG(INFO) << "Pad mode should be VALID or SAME, but got " << pad_mode << ", skip fusion node ["
                 << avgpool->fullname_with_scope() << "].";
    return nullptr;
  }

  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  if (input_num <= 1) {
    manager->AddEdge(avgpool, filter_node);
  } else {
    manager->SetEdge(avgpool, 1, filter_node);
  }
  kernel_graph->AddValueNodeToGraph(filter_node);
  return ret_node;
}
}  // namespace opt
}  // namespace mindspore
