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

#include "src/delegate/tensorrt/tensorrt_delegate.h"
#include <cuda_runtime.h>
#include <vector>
#include <fstream>
#include <string>
#include "src/delegate/delegate_utils.h"
#include "src/delegate/tensorrt/op/activation_tensorrt.h"
#include "src/delegate/tensorrt/op/shape_tensorrt.h"
#include "src/delegate/tensorrt/op/gather_tensorrt.h"
#include "src/delegate/tensorrt/op/shuffle_tensorrt.h"
#include "src/delegate/tensorrt/op/concate_tensorrt.h"
#include "src/delegate/tensorrt/op/convolution_tensorrt.h"
#include "src/delegate/tensorrt/op/elementwise_tensorrt.h"
#include "src/delegate/tensorrt/op/reduce_tensorrt.h"
#include "src/delegate/tensorrt/op/softmax_tensorrt.h"
#include "src/delegate/tensorrt/op/unary_tensorrt.h"
#include "src/delegate/tensorrt/op/matmul_tensorrt.h"
#include "src/delegate/tensorrt/op/scale_tensorrt.h"

namespace mindspore::lite {
bool IsHardwareSupport() {
  int driver_version = 0;
  cudaDriverGetVersion(&driver_version);
  if (driver_version == 0) {
    MS_LOG(WARNING) << "No nvidia GPU driver.";
    return false;
  }
  return true;
}

Status TensorRTDelegate::Init() {
  if (!IsHardwareSupport()) {
    return mindspore::kLiteNotSupport;
  }
  op_func_lists_.clear();
  op_func_lists_ = {
    {schema::PrimitiveType_Activation, GetTensorRTOp<ActivationTensorRT>},
    {schema::PrimitiveType_Unsqueeze, GetTensorRTOp<ShuffleTensorRT>},
    {schema::PrimitiveType_Squeeze, GetTensorRTOp<ShuffleTensorRT>},
    {schema::PrimitiveType_Reshape, GetTensorRTOp<ShuffleTensorRT>},
    {schema::PrimitiveType_Concat, GetTensorRTOp<ConcateTensorRT>},
    {schema::PrimitiveType_Conv2DFusion, GetTensorRTOp<ConvolutionTensorRT>},
    {schema::PrimitiveType_SubFusion, GetTensorRTOp<ElementWiseTensorRT>},
    {schema::PrimitiveType_DivFusion, GetTensorRTOp<ElementWiseTensorRT>},
    {schema::PrimitiveType_PowFusion, GetTensorRTOp<ElementWiseTensorRT>},
    {schema::PrimitiveType_AddFusion, GetTensorRTOp<ElementWiseTensorRT>},
    {schema::PrimitiveType_Transpose, GetTensorRTOp<ShuffleTensorRT>},
    {schema::PrimitiveType_ReduceFusion, GetTensorRTOp<ReduceTensorRT>},
    {schema::PrimitiveType_Sqrt, GetTensorRTOp<UnaryTensorRT>},
    {schema::PrimitiveType_MatMul, GetTensorRTOp<MatMulTensorRT>},
    {schema::PrimitiveType_ScaleFusion, GetTensorRTOp<ScaleTensorRT>},
  };
  return mindspore::kSuccess;
}

Status TensorRTDelegate::Build(DelegateModel<schema::Primitive> *model) {
  KernelIter from, end;
  std::vector<TensorRTOp *> tensorrt_ops;
  int graph_index = 0;
  for (KernelIter iter = model->BeginKernelIterator(); iter != model->EndKernelIterator(); iter++) {
    kernel::Kernel *kernel = *iter;
    auto tensorrt_op = FindTensorRTOp(kernel, model->GetPrimitive(kernel));
    if (tensorrt_op != nullptr) {
      // If tensorrt_ops does not equal nullptr, this kernel can be supported by delegate
      if (tensorrt_ops.size() == 0) {
        from = iter;
      }
      tensorrt_ops.push_back(tensorrt_op);
      end = iter;
    } else {
      if (tensorrt_ops.size() > 0) {
        auto tensorrt_subgraph = CreateTensorRTGraph(tensorrt_ops, model, from, end);
        if (tensorrt_subgraph == nullptr) {
          MS_LOG(ERROR) << "Create TensorRT Graph failed.";
          return mindspore::kLiteNullptr;
        }
        tensorrt_subgraph->set_name("TensorRtGraph" + std::to_string(graph_index++));
        iter = model->Replace(from, end + 1, tensorrt_subgraph);
        tensorrt_ops.clear();
      }
    }
  }
  if (tensorrt_ops.size() > 0) {
    auto tensorrt_subgraph = CreateTensorRTGraph(tensorrt_ops, model, from, end);
    if (tensorrt_subgraph == nullptr) {
      MS_LOG(DEBUG) << "Create TensorRT Graph failed.";
      return mindspore::kLiteNullptr;
    }
    tensorrt_subgraph->set_name("TensorRtGraph" + std::to_string(graph_index++));
    model->Replace(from, end + 1, tensorrt_subgraph);
    tensorrt_ops.clear();
  }
  return mindspore::kSuccess;
}

TensorRTOp *TensorRTDelegate::FindTensorRTOp(kernel::Kernel *kernel, const schema::Primitive *primitive) {
  auto in_tensors = kernel->inputs();
  auto out_tensors = kernel->outputs();
  auto name = kernel->name();
  auto node_type = primitive->value_type();
  if (op_func_lists_.find(node_type) != op_func_lists_.end()) {
    return op_func_lists_[node_type](primitive, in_tensors, out_tensors, name);
  } else {
    MS_LOG(WARNING) << "Unsupported op type for TensorRT. kernel->name:" << kernel->name()
                    << " type:" << schema::EnumNamePrimitiveType(primitive->value_type());
    return nullptr;
  }
}

TensorRTSubGraph *TensorRTDelegate::CreateTensorRTGraph(const std::vector<TensorRTOp *> &ops,
                                                        DelegateModel<schema::Primitive> *model, KernelIter from,
                                                        KernelIter end) {
  auto in_tensors = GraphInTensors<TensorRTOp>(ops, model, from, end);
  auto out_tensors = GraphOutTensors<TensorRTOp>(ops, model, from, end);
  auto *tensorrt_graph = new (std::nothrow) TensorRTSubGraph(ops, in_tensors, out_tensors);
  if (tensorrt_graph == nullptr) {
    MS_LOG(ERROR) << "new tensorrt_graph failed.";
    return nullptr;
  }
  // 1. For every op, find pre and next ops
  FindPreNextOps<TensorRTOp>(ops);

  // 2. Init TensorRT SubGraph.
  auto ret = tensorrt_graph->Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "TensorRTGraph init failed.";
    return nullptr;
  }

  // 3. Build TensorRT Model.
  ret = tensorrt_graph->BuildTensorRTGraph();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "TensorRTGraph build failed.";
    return nullptr;
  }

  return tensorrt_graph;
}
}  // namespace mindspore::lite
