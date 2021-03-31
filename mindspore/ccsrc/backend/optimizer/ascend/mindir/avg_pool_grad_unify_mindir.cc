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

#include "backend/optimizer/ascend/mindir/avg_pool_grad_unify_mindir.h"

#include <vector>
#include <memory>
#include <algorithm>
#include <functional>
#include <string>

#include "utils/utils.h"
#include "utils/ms_context.h"
#include "utils/check_convert_utils.h"
#include "backend/optimizer/common/helper.h"
#include "runtime/device/kernel_info.h"
#include "backend/session/anf_runtime_algorithm.h"

namespace mindspore {
namespace opt {
namespace {
constexpr size_t kAvgPoolGradInputNum = 3;
constexpr size_t kShapeDimNum = 4;
constexpr float kKernelMatrixInitNum = 1.0;
constexpr size_t kFloat32Len = 4;  // size of float32

std::vector<int64_t> GetInputXShape(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  std::vector<int64_t> shapes;
  auto shape_size_t = AnfAlgo::GetPrevNodeOutputInferShape(node, 0);
  std::transform(shape_size_t.begin(), shape_size_t.end(), std::back_inserter(shapes), SizeToLong);
  return shapes;
}

int64_t windowed_output_size(int64_t input_size, int64_t ksize, int64_t stride, PadMode pad_mode, int64_t *pad_before,
                             int64_t *pad_after) {
  int64_t output = 0;
  *pad_before = 0;
  *pad_after = 0;
  if (stride == 0) {
    MS_LOG(EXCEPTION) << "The stride of AvgPoolGrad should not be 0.";
    return 0;
  }
  if (pad_mode == PadMode::VALID) {
    output = (input_size - ksize + stride) / stride;
  } else if (pad_mode == PadMode::SAME) {
    output = (input_size + stride - 1) / stride;
    int64_t pad_need = std::max(int64_t(0), (output - 1) * stride + ksize - input_size);
    *pad_before = pad_need / 2;
    *pad_after = pad_need - *pad_before;
  } else {
    MS_LOG(EXCEPTION) << "The pad mode of AvgPoolGrad should be SAME or VALID.";
  }
  return output;
}

std::vector<std::vector<float>> GetAssistInputMatrix(const std::vector<int64_t> &x_shape, int64_t pad_top,
                                                     int64_t pad_bottom, int64_t pad_left, int64_t pad_right) {
  // `assist_input_matrix` is a 2d matrix with input_shape after padding,
  // the value of element which is padded is 0, else are 1.
  // For each element of output, it is mapped for slide window: `[h*h_stride : h*h_stride + h_ksize,
  // w*w_stride : w*w_stride + w_ksize]` of `assist_input_matrix`, so the sum of slide window is the
  // number of input that associate with output element.
  std::vector<std::vector<float>> assist_input_matrix;
  std::vector<int64_t> in_shape_after_padding_2d = {x_shape[2] + pad_top + pad_bottom,
                                                    x_shape[3] + pad_left + pad_right};
  std::vector<float> tmp_zero_vector(in_shape_after_padding_2d[1], 0.0);
  std::vector<float> tmp_one_vector(in_shape_after_padding_2d[1], 1.0);
  for (int64_t i = 0; i < in_shape_after_padding_2d[1]; ++i) {
    if (i < pad_left || i >= (in_shape_after_padding_2d[1] - pad_right)) {
      tmp_one_vector[i] = 0.0;
    }
  }
  for (int64_t i = 0; i < in_shape_after_padding_2d[0]; ++i) {
    if (i < pad_top || i >= (in_shape_after_padding_2d[0] - pad_bottom)) {
      assist_input_matrix.emplace_back(tmp_zero_vector);
    } else {
      assist_input_matrix.emplace_back(tmp_one_vector);
    }
  }
  return assist_input_matrix;
}

ValueNodePtr CreateMeanMatrixValueNode(const FuncGraphPtr &func_graph, const std::vector<int64_t> &x_shape,
                                       const std::vector<int64_t> &k_size, const std::vector<int64_t> &stride,
                                       const PadMode pad_mode, const TypeId x_dtype) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto kernel_graph = func_graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  if (x_shape.size() != kShapeDimNum || k_size.size() != kShapeDimNum || stride.size() != kShapeDimNum) {
    MS_LOG(EXCEPTION) << "The dim of x_shape or kernel_size or strides of AvgPoolGrad should be 4.";
  }
  int64_t pad_top, pad_bottom, pad_left, pad_right;
  int64_t h_output = windowed_output_size(x_shape[2], k_size[2], stride[2], pad_mode, &pad_top, &pad_bottom);
  int64_t w_output = windowed_output_size(x_shape[3], k_size[3], stride[3], pad_mode, &pad_left, &pad_right);
  auto assist_input_matrix = GetAssistInputMatrix(x_shape, pad_top, pad_bottom, pad_left, pad_right);

  // calculate output
  std::vector<float> hw_output(h_output * w_output, 0.0);
  for (int64_t h = 0; h < h_output; ++h) {
    for (int64_t w = 0; w < w_output; ++w) {
      float curr_sum = 0;
      for (int64_t i = h * stride[2]; i < h * stride[2] + k_size[2]; ++i) {
        for (int64_t j = w * stride[3]; j < w * stride[3] + k_size[3]; ++j) {
          curr_sum += assist_input_matrix[i][j];
        }
      }
      if (curr_sum > 0) {
        hw_output[h * w_output + w] = 1.0 / curr_sum;
      }
    }
  }

  // make output tensor
  std::vector<int64_t> output_shape = {x_shape[0], x_shape[1], h_output, w_output};
  auto output_size = std::accumulate(output_shape.begin(), output_shape.end(), int64_t(1), std::multiplies<int64_t>());
  std::vector<float> output(output_size, 0.0);
  for (int64_t i = 0; i < output_shape[0] * output_shape[1]; ++i) {
    size_t src_size = hw_output.size() * kFloat32Len;
    size_t dst_size = output_shape[2] * output_shape[3] * kFloat32Len;
    auto ret = memcpy_s(&output[i * hw_output.size()], dst_size, &hw_output[0], src_size);
    if (ret != 0) {
      MS_LOG(EXCEPTION) << "memcpy_s error, errorno(" << ret << ")";
      return nullptr;
    }
  }
  auto output_tensor = std::make_shared<tensor::Tensor>(x_dtype, output_shape, &output[0], kNumberTypeFloat32);
  MS_EXCEPTION_IF_NULL(output_tensor);
  auto abstract = std::make_shared<abstract::AbstractTensor>(TypeIdToType(x_dtype), output_shape);
  MS_EXCEPTION_IF_NULL(abstract);
  auto mean_matrix_vnode = kernel_graph->NewValueNode(abstract, output_tensor);
  MS_EXCEPTION_IF_NULL(mean_matrix_vnode);
  kernel_graph->AddValueNodeToGraph(mean_matrix_vnode);
  return mean_matrix_vnode;
}

ValueNodePtr CreateKernelMatrixValueNode(const FuncGraphPtr &func_graph, const std::vector<int64_t> &x_shape,
                                         const std::vector<int64_t> &k_size, const TypeId x_dtype) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto kernel_graph = func_graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  if (x_shape.size() != kShapeDimNum || k_size.size() != kShapeDimNum) {
    MS_LOG(EXCEPTION) << "The dim of x_shape or kernel_size of AvgPoolGrad should be 4.";
  }
  std::vector<int64_t> kernel_shape = {1, x_shape[1], k_size[2], k_size[3]};
  auto data_size = std::accumulate(kernel_shape.begin(), kernel_shape.end(), int64_t(1), std::multiplies<int64_t>());
  std::vector<float> data(data_size, kKernelMatrixInitNum);
  auto kernel_matrix_tensor = std::make_shared<tensor::Tensor>(x_dtype, kernel_shape, &data[0], kNumberTypeFloat32);
  MS_EXCEPTION_IF_NULL(kernel_matrix_tensor);
  auto abstract = std::make_shared<abstract::AbstractTensor>(TypeIdToType(x_dtype), kernel_shape);
  MS_EXCEPTION_IF_NULL(abstract);
  auto kernel_matrix_vnode = kernel_graph->NewValueNode(abstract, kernel_matrix_tensor);
  MS_EXCEPTION_IF_NULL(kernel_matrix_vnode);
  kernel_graph->AddValueNodeToGraph(kernel_matrix_vnode);
  return kernel_matrix_vnode;
}
}  // namespace

const BaseRef AvgPoolGradUnifyMindIR::DefinePattern() const {
  VarPtr X1 = std::make_shared<Var>();
  VarPtr X2 = std::make_shared<Var>();
  VarPtr G = std::make_shared<Var>();
  VectorRef pattern({prim::kPrimAvgPoolGrad, X1, X2, G});
  return pattern;
}

const AnfNodePtr AvgPoolGradUnifyMindIR::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                 const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  auto avgpool_grad = CheckAnfNodeIfCNodeAndInputSize(node, kAvgPoolGradInputNum);

  auto x_shape = GetInputXShape(avgpool_grad);
  auto x_dtype = AnfAlgo::GetPrevNodeOutputInferDataType(avgpool_grad, 0);
  auto k_size = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(avgpool_grad, kAttrKernelSize);
  auto stride = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(avgpool_grad, kAttrStrides);
  auto pad_mode = PadMode(AnfAlgo::GetNodeAttr<int64_t>(avgpool_grad, kAttrPadMode));

  auto x_shape_vnode = CreateShapeValueNode(graph, x_shape);
  auto mean_matrix_vnode = CreateMeanMatrixValueNode(graph, x_shape, k_size, stride, pad_mode, x_dtype);
  auto kernel_matrix_vnode = CreateKernelMatrixValueNode(graph, x_shape, k_size, x_dtype);

  std::vector<AnfNodePtr> avgpool_grad_vm_inputs = {NewValueNode(std::make_shared<Primitive>(kAvgPoolGradVmOpName)),
                                                    x_shape_vnode, avgpool_grad->input(3), mean_matrix_vnode,
                                                    kernel_matrix_vnode};
  auto avgpool_grad_vm = graph->NewCNode(avgpool_grad_vm_inputs);
  MS_EXCEPTION_IF_NULL(avgpool_grad_vm);
  avgpool_grad_vm->set_scope(avgpool_grad->scope());
  avgpool_grad_vm->set_abstract(avgpool_grad->abstract());
  AnfAlgo::CopyNodeAttr(kAttrKernelSize, avgpool_grad, avgpool_grad_vm);
  AnfAlgo::CopyNodeAttr(kAttrStrides, avgpool_grad, avgpool_grad_vm);
  AnfAlgo::CopyNodeAttr(kAttrPadMode, avgpool_grad, avgpool_grad_vm);
  AnfAlgo::CopyNodeAttr(kAttrFormat, avgpool_grad, avgpool_grad_vm);
  auto input_names = std::vector<std::string>{"x_origin", "grad", "mean_matrix", "kernel_matrix"};
  AnfAlgo::SetNodeAttr(kAttrInputNames, MakeValue(input_names), avgpool_grad_vm);
  auto output_names = std::vector<std::string>{"output"};
  AnfAlgo::SetNodeAttr(kAttrOutputNames, MakeValue(output_names), avgpool_grad_vm);
  return avgpool_grad_vm;
}
}  // namespace opt
}  // namespace mindspore
