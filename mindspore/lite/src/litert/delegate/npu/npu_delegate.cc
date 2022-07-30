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

#include "src/litert/delegate/npu/npu_delegate.h"
#include <queue>
#include "include/errorcode.h"
#include "src/common/prim_util.h"
#include "src/litert/delegate/npu/pass/npu_pass_utils.h"
#include "src/litert/delegate/npu/op/npu_op.h"
#include "src/litert/delegate/npu/op/activation_npu.h"
#include "src/litert/delegate/npu/op/argmax_npu.h"
#include "src/litert/delegate/npu/op/arithmetic_npu.h"
#include "src/litert/delegate/npu/op/arithmetic_self_npu.h"
#include "src/litert/delegate/npu/op/avg_pooling_npu.h"
#include "src/litert/delegate/npu/op/batchnorm_npu.h"
#include "src/litert/delegate/npu/op/cast_npu.h"
#include "src/litert/delegate/npu/op/concat_npu.h"
#include "src/litert/delegate/npu/op/convolution_npu.h"
#include "src/litert/delegate/npu/op/crop_and_resize_npu.h"
#include "src/litert/delegate/npu/op/deconvolution_npu.h"
#include "src/litert/delegate/npu/op/eltwise_npu.h"
#include "src/litert/delegate/npu/op/expand_dims_npu.h"
#include "src/litert/delegate/npu/op/fullconnection_npu.h"
#include "src/litert/delegate/npu/op/gather_npu.h"
#include "src/litert/delegate/npu/op/instance_norm_npu.h"
#include "src/litert/delegate/npu/op/matmul_npu.h"
#include "src/litert/delegate/npu/op/max_pooling_npu.h"
#include "src/litert/delegate/npu/op/pad_npu.h"
#include "src/litert/delegate/npu/op/reduce_npu.h"
#include "src/litert/delegate/npu/op/reshape_npu.h"
#include "src/litert/delegate/npu/op/resize_npu.h"
#include "src/litert/delegate/npu/op/scale_npu.h"
#include "src/litert/delegate/npu/op/slice_npu.h"
#include "src/litert/delegate/npu/op/softmax_npu.h"
#include "src/litert/delegate/npu/op/split_npu.h"
#include "src/litert/delegate/npu/op/squeeze_npu.h"
#include "src/litert/delegate/npu/op/strided_slice_npu.h"
#include "src/litert/delegate/npu/op/tile_npu.h"
#include "src/litert/delegate/npu/op/transpose_npu.h"
#include "src/litert/delegate/npu/op/unsqueeze_npu.h"
#include "src/litert/delegate/npu/op/abs_npu.h"
#include "src/litert/delegate/npu/op/flatten_npu.h"
#include "src/litert/delegate/npu/op/broadcast_to_npu.h"
#include "src/litert/delegate/npu/npu_graph.h"
#include "src/litert/delegate/delegate_utils.h"
#include "src/litert/delegate/npu/pass/npu_transform_pass.h"
#include "src/litert/delegate/npu/pass/npu_insert_transform_pass.h"
#include "src/litert/delegate/npu/pass/npu_fusion_pass.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

namespace mindspore::lite {
NPUDelegate::~NPUDelegate() {
  if (npu_manager_ != nullptr) {
    npu_manager_->Reset();
    delete npu_manager_;
    npu_manager_ = nullptr;
  }
  if (pass_manager_ != nullptr) {
    pass_manager_->Clear();
    delete pass_manager_;
    pass_manager_ = nullptr;
  }
}

Status NPUDelegate::AddPasses() {
  auto transform_pass = new (std::nothrow) NPUTransformPass();
  if (transform_pass == nullptr) {
    MS_LOG(ERROR) << "New NPUTransformPass failed.";
    return mindspore::kLiteNullptr;
  }
  pass_manager_->AddPass(transform_pass);

  auto insert_transform_pass = new (std::nothrow) NPUInsertTransformPass();
  if (insert_transform_pass == nullptr) {
    MS_LOG(ERROR) << "New NPUInsertTransformPass failed.";
    return mindspore::kLiteNullptr;
  }
  pass_manager_->AddPass(insert_transform_pass);

  auto fusion_pass = new (std::nothrow) NPUFusionPass();
  if (fusion_pass == nullptr) {
    MS_LOG(ERROR) << "New NPUFusionPass failed.";
    return mindspore::kLiteNullptr;
  }
  pass_manager_->AddPass(fusion_pass);
  return mindspore::kSuccess;
}

Status NPUDelegate::Init() {
  npu_manager_ = new (std::nothrow) NPUManager(frequency_);
  if (npu_manager_ == nullptr) {
    MS_LOG(ERROR) << "New npu manager failed.";
    return mindspore::kLiteNullptr;
  }
  if (!npu_manager_->IsSupportNPU()) {
    MS_LOG(DEBUG) << "Checking that npu is unsupported.";
    delete npu_manager_;
    npu_manager_ = nullptr;
    return mindspore::kLiteNotSupport;
  }
  pass_manager_ = new (std::nothrow) NPUPassManager();
  if (pass_manager_ == nullptr) {
    delete npu_manager_;
    npu_manager_ = nullptr;
    MS_LOG(ERROR) << "New npu pass manager failed.";
    return mindspore::kLiteNullptr;
  }

  auto ret = AddPasses();
  if (ret != mindspore::kSuccess) {
    MS_LOG(ERROR) << "add passes for npu pass manager failed.";
    return ret;
  }

  op_func_lists_.clear();
  op_func_lists_ = {
    {schema::PrimitiveType_Activation, GetNPUOp<ActivationNPUOp>},
    {schema::PrimitiveType_ArgMaxFusion, GetNPUOp<ArgmaxNPUOp>},
    {schema::PrimitiveType_MulFusion, GetNPUOp<ArithmeticNPUOp>},
    {schema::PrimitiveType_AddFusion, GetNPUOp<ArithmeticNPUOp>},
    {schema::PrimitiveType_SubFusion, GetNPUOp<ArithmeticNPUOp>},
    {schema::PrimitiveType_DivFusion, GetNPUOp<ArithmeticNPUOp>},
    {schema::PrimitiveType_FloorMod, GetNPUOp<ArithmeticNPUOp>},
    {schema::PrimitiveType_FloorDiv, GetNPUOp<ArithmeticNPUOp>},
    {schema::PrimitiveType_LogicalAnd, GetNPUOp<ArithmeticNPUOp>},
    {schema::PrimitiveType_LogicalOr, GetNPUOp<ArithmeticNPUOp>},
    {schema::PrimitiveType_Maximum, GetNPUOp<ArithmeticNPUOp>},
    {schema::PrimitiveType_Minimum, GetNPUOp<ArithmeticNPUOp>},
    {schema::PrimitiveType_NotEqual, GetNPUOp<ArithmeticNPUOp>},
    {schema::PrimitiveType_Equal, GetNPUOp<ArithmeticNPUOp>},
    {schema::PrimitiveType_Less, GetNPUOp<ArithmeticNPUOp>},
    {schema::PrimitiveType_LessEqual, GetNPUOp<ArithmeticNPUOp>},
    {schema::PrimitiveType_Greater, GetNPUOp<ArithmeticNPUOp>},
    {schema::PrimitiveType_GreaterEqual, GetNPUOp<ArithmeticNPUOp>},
    {schema::PrimitiveType_Ceil, GetNPUOp<ArithmeticSelfNPUOp>},
    {schema::PrimitiveType_Cos, GetNPUOp<ArithmeticSelfNPUOp>},
    {schema::PrimitiveType_Floor, GetNPUOp<ArithmeticSelfNPUOp>},
    {schema::PrimitiveType_Log, GetNPUOp<ArithmeticSelfNPUOp>},
    {schema::PrimitiveType_LogicalNot, GetNPUOp<ArithmeticSelfNPUOp>},
    {schema::PrimitiveType_Neg, GetNPUOp<ArithmeticSelfNPUOp>},
    {schema::PrimitiveType_Reciprocal, GetNPUOp<ArithmeticSelfNPUOp>},
    {schema::PrimitiveType_Round, GetNPUOp<ArithmeticSelfNPUOp>},
    {schema::PrimitiveType_Rsqrt, GetNPUOp<ArithmeticSelfNPUOp>},
    {schema::PrimitiveType_Sin, GetNPUOp<ArithmeticSelfNPUOp>},
    {schema::PrimitiveType_Sqrt, GetNPUOp<ArithmeticSelfNPUOp>},
    {schema::PrimitiveType_Square, GetNPUOp<ArithmeticSelfNPUOp>},
    {schema::PrimitiveType_ExpFusion, GetNPUOp<ArithmeticSelfNPUOp>},
    {schema::PrimitiveType_AvgPoolFusion, GetNPUOp<AvgPoolingNPUOp>},
    {schema::PrimitiveType_MaxPoolFusion, GetNPUOp<MaxPoolingNPUOp>},
    {schema::PrimitiveType_FusedBatchNorm, GetNPUOp<BatchnormNPUOp>},
    {schema::PrimitiveType_Cast, GetNPUOp<CastNPUOp>},
    {schema::PrimitiveType_Concat, GetNPUOp<ConcatNPUOp>},
    {schema::PrimitiveType_Conv2dTransposeFusion, GetNPUOp<DeconvolutionNPUOp>},
    {schema::PrimitiveType_CropAndResize, GetNPUOp<CropAndResizeNPUOp>},
    {schema::PrimitiveType_Eltwise, GetNPUOp<EltwiseNPUOp>},
    {schema::PrimitiveType_ExpandDims, GetNPUOp<ExpandDimsNPUOp>},
    {schema::PrimitiveType_FullConnection, GetNPUOp<FullconnectionNPUOp>},
    {schema::PrimitiveType_Gather, GetNPUOp<GatherNPUOp>},
    {schema::PrimitiveType_InstanceNorm, GetNPUOp<InstanceNormNPUOp>},
    {schema::PrimitiveType_MatMulFusion, GetNPUOp<MatMulNPUOp>},
    {schema::PrimitiveType_PadFusion, GetNPUOp<PadNPUOp>},
    {schema::PrimitiveType_ReduceFusion, GetNPUOp<ReduceNPUOp>},
    {schema::PrimitiveType_Reshape, GetNPUOp<ReshapeNPUOp>},
    {schema::PrimitiveType_Resize, GetNPUOp<ResizeNPUOp>},
    {schema::PrimitiveType_ScaleFusion, GetNPUOp<ScaleNPUOp>},
    {schema::PrimitiveType_SliceFusion, GetNPUOp<SliceNPUOp>},
    {schema::PrimitiveType_Softmax, GetNPUOp<SoftmaxNPUOp>},
    {schema::PrimitiveType_Split, GetNPUOp<SplitNPUOp>},
    {schema::PrimitiveType_Squeeze, GetNPUOp<SqueezeNPUOp>},
    {schema::PrimitiveType_StridedSlice, GetNPUOp<StridedSliceNPUOp>},
    {schema::PrimitiveType_TileFusion, GetNPUOp<TileNPUOp>},
    {schema::PrimitiveType_Transpose, GetNPUOp<TransposeNPUOp>},
    {schema::PrimitiveType_Unsqueeze, GetNPUOp<UnsqueezeNPUOp>},
    {schema::PrimitiveType_Abs, GetNPUOp<AbsNPUOp>},
    {schema::PrimitiveType_Flatten, GetNPUOp<FlattenNPUOp>},
    {schema::PrimitiveType_BroadcastTo, GetNPUOp<BroadcastToNPUOp>},
  };
  return mindspore::kSuccess;
}

Status NPUDelegate::Build(DelegateModel<schema::Primitive> *model) {
  KernelIter from, end;
  std::vector<NPUOp *> npu_ops;
  int graph_index = 0;
  for (auto iter = model->BeginKernelIterator(); iter != model->EndKernelIterator(); iter++) {
    kernel::Kernel *kernel = *iter;
    auto npu_op = GetOP(kernel, model->GetPrimitive(kernel));
    if (npu_op != nullptr) {
      // If npu_op does not equal nullptr, this kernel can be supported by delegate
      if (npu_ops.empty()) {
        from = iter;
      }
      npu_ops.push_back(npu_op);
      end = iter;
    } else {
      if (!npu_ops.empty()) {
        auto npu_graph_kernel = CreateNPUGraph(npu_ops, model, from, end);
        if (npu_graph_kernel == nullptr) {
          MS_LOG(ERROR) << "Create NPU Graph failed.";
          return mindspore::kLiteNullptr;
        }
        npu_graph_kernel->set_name("NpuGraph" + std::to_string(graph_index++));
        iter = model->Replace(from, end + 1, npu_graph_kernel);
        npu_ops.clear();
      }
    }
  }
  if (!npu_ops.empty()) {
    auto npu_graph_kernel = CreateNPUGraph(npu_ops, model, from, end);
    if (npu_graph_kernel == nullptr) {
      MS_LOG(ERROR) << "Create NPU Graph failed.";
      return mindspore::kLiteNullptr;
    }
    npu_graph_kernel->set_name("NpuGraph" + std::to_string(graph_index++));
    model->Replace(from, end + 1, npu_graph_kernel);
    npu_ops.clear();
  }
  auto ret = npu_manager_->LoadOMModel();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "NPU client load model failed.";
    return mindspore::kLiteError;
  }
  return mindspore::kSuccess;
}

NPUOp *NPUDelegate::GetOP(kernel::Kernel *kernel, const schema::Primitive *primitive) {
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "primitive is NULL!";
    return nullptr;
  }
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "kernel is NULL!";
    return nullptr;
  }
  auto name = kernel->name();
  NPUOp *npu_op = nullptr;
  auto node_type = primitive->value_type();
  if (node_type == schema::PrimitiveType_Conv2DFusion) {
    npu_op = GetNPUConvOp(primitive, kernel->inputs(), kernel->outputs(), name);
  } else if (node_type == schema::PrimitiveType_FullConnection) {
    npu_op = GetNPUFCOp(primitive, kernel->inputs(), kernel->outputs(), name);
  } else {
    if (op_func_lists_.find(node_type) != op_func_lists_.end()) {
      npu_op = op_func_lists_[node_type](primitive, kernel->inputs(), kernel->outputs(), name);
    } else {
      MS_LOG(DEBUG) << "Unsupported op type for NPU: " << node_type;
      return nullptr;
    }
  }
  if (npu_op == nullptr) {
    MS_LOG(DEBUG) << "Get NPU op failed, op name: " << name;
    return nullptr;
  }

  for (int i = 0; i < kernel->inputs().size(); i++) {
    mindspore::MSTensor tensor = kernel->inputs()[i];
    if (tensor.DataType() == DataType::kNumberTypeFloat16 && tensor.Data() == nullptr) {
      tensor.SetDataType(DataType::kNumberTypeFloat32);
    }
  }
  for (int i = 0; i < kernel->outputs().size(); i++) {
    mindspore::MSTensor tensor = kernel->outputs()[i];
    if (tensor.DataType() == DataType::kNumberTypeFloat16) {
      tensor.SetDataType(DataType::kNumberTypeFloat32);
    }
  }

  MS_LOG(DEBUG) << "kernel: [" << kernel->name().c_str() << "] op success. "
                << "op_type: " << PrimitiveCurVersionTypeName(kernel->type()) << ", "
                << "arch: " << kKirinNPU;
  return npu_op;
}

kernel::Kernel *NPUDelegate::CreateNPUGraph(const std::vector<NPUOp *> &ops, DelegateModel<schema::Primitive> *model,
                                            KernelIter from, KernelIter end) {
  auto in_tensors = GetGraphInTensors(ops, nullptr);
  auto out_tensors = GraphOutTensors<NPUOp>(ops, model, from, end);
  auto graph_kernel = new (std::nothrow) NPUGraph(ops, npu_manager_, in_tensors, out_tensors);
  if (graph_kernel == nullptr) {
    MS_LOG(DEBUG) << "New NPU Graph failed.";
    return nullptr;
  }
  // 1. For every op, find pre and next ops
  auto ret = graph_kernel->FindPreNextOps();
  if (ret != RET_OK) {
    delete graph_kernel;
    MS_LOG(DEBUG) << "NPU Graph find input and output ops for every op failed.";
    return nullptr;
  }
  // 2. Pass
  ret = pass_manager_->RunPass(graph_kernel);
  if (ret != RET_OK) {
    delete graph_kernel;
    MS_LOG(DEBUG) << "NPU Graph run pass failed. This function mainly solves the problem that the format is "
                     "inconsistent and requires interpolation transpose operators.";
    return nullptr;
  }
  // 3. NPUGraph init, create subgraph_kernel and transpose_kernel
  ret = graph_kernel->Init();
  if (ret != RET_OK) {
    delete graph_kernel;
    MS_LOG(DEBUG) << "NPU subgraph Init failed.";
    return nullptr;
  }
  return graph_kernel;
}
}  // namespace mindspore::lite
