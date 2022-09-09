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

#include "src/litert/delegate/coreml/coreml_delegate.h"
#include "include/errorcode.h"
#include "src/common/prim_util.h"
#include "src/litert/delegate/coreml/op/coreml_op.h"
#include "src/litert/delegate/coreml/op/activation_coreml.h"
#include "src/litert/delegate/coreml/op/transpose_coreml.h"
#include "src/litert/delegate/coreml/op/convolution_coreml.h"
#include "src/litert/delegate/coreml/op/deconvolution_coreml.h"
#include "src/litert/delegate/coreml/op/avg_pooling_coreml.h"
#include "src/litert/delegate/coreml/op/max_pooling_coreml.h"
#include "src/litert/delegate/coreml/op/arithmetic_coreml.h"
#include "src/litert/delegate/coreml/op/resize_coreml.h"
#include "src/litert/delegate/coreml/op/reshape_coreml.h"
#include "src/litert/delegate/coreml/op/matmul_coreml.h"
#include "src/litert/delegate/coreml/op/concat_coreml.h"
#include "src/litert/delegate/coreml/op/unsqueeze_coreml.h"
#include "src/litert/delegate/coreml/op/gather_coreml.h"
#include "src/litert/delegate/coreml/op/shape_coreml.h"
#include "src/litert/delegate/coreml/op/softmax_coreml.h"
#include "src/litert/delegate/coreml/op/flatten_coreml.h"
#include "src/litert/delegate/coreml/op/arithmetic_self_coreml.h"
#include "src/litert/delegate/coreml/op/split_coreml.h"
#include "src/litert/delegate/coreml/op/strided_slice_coreml.h"
#include "src/litert/delegate/coreml/coreml_graph.h"
#include "src/litert/delegate/delegate_utils.h"
#include "src/litert/delegate/coreml/pass/coreml_format_trans_pass.h"
#include "src/litert/delegate/coreml/pass/coreml_trans_extend_pass.h"
#include "src/litert/delegate/coreml/pass/coreml_fusion_pass.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

namespace mindspore {
namespace lite {
CoreMLDelegateImpl::~CoreMLDelegateImpl() {
  if (pass_manager_ != nullptr) {
    pass_manager_->Clear();
    delete pass_manager_;
    pass_manager_ = nullptr;
  }
}

bool CoreMLDelegateImpl::IsSupportCoreML() const {
  if (@available(iOS 11, *)) {
    return true;
  }
  return false;
}

Status CoreMLDelegateImpl::AddPasses() {
  auto format_trans_pass = new (std::nothrow) CoreMLFormatTransPass();
  if (format_trans_pass == nullptr) {
    MS_LOG(ERROR) << "New CoreMLFormatTransPass failed.";
    return mindspore::kLiteNullptr;
  }
  pass_manager_->AddPass(format_trans_pass);

  auto trans_extend_pass = new (std::nothrow) CoreMLTransExtendPass();
  if (trans_extend_pass == nullptr) {
    MS_LOG(ERROR) << "New CoreMLTransExtendPass failed.";
    return mindspore::kLiteNullptr;
  }
  pass_manager_->AddPass(trans_extend_pass);

  auto fusion_pass = new (std::nothrow) CoreMLFusionPass();
  if (fusion_pass == nullptr) {
    MS_LOG(ERROR) << "New CoreMLFusionPass failed.";
    return mindspore::kLiteNullptr;
  }
  pass_manager_->AddPass(fusion_pass);
  return mindspore::kSuccess;
}

Status CoreMLDelegateImpl::Init() {
  if (!IsSupportCoreML()) {
    MS_LOG(WARNING) << "Current device not support CoreML.";
    return mindspore::kLiteNotSupport;
  }
  pass_manager_ = new (std::nothrow) CoreMLPassManager();
  if (pass_manager_ == nullptr) {
    MS_LOG(ERROR) << "New coreml pass manager failed.";
    return mindspore::kLiteNullptr;
  }
  auto ret = AddPasses();
  if (ret != mindspore::kSuccess) {
    MS_LOG(ERROR) << "add passes for coreml pass manager failed.";
    return ret;
  }
  op_func_lists_.clear();
  op_func_lists_ = {
    {schema::PrimitiveType_Activation, GetCoreMLOp<ActivationCoreMLOp>},
    {schema::PrimitiveType_Transpose, GetCoreMLOp<TransposeCoreMLOp>},
    {schema::PrimitiveType_Conv2DFusion, GetCoreMLOp<ConvolutionCoreMLOp>},
    {schema::PrimitiveType_Conv2dTransposeFusion, GetCoreMLOp<DeconvolutionCoreMLOp>},
    {schema::PrimitiveType_AvgPoolFusion, GetCoreMLOp<AvgPoolingCoreMLOp>},
    {schema::PrimitiveType_MaxPoolFusion, GetCoreMLOp<MaxPoolingCoreMLOp>},
    {schema::PrimitiveType_AddFusion, GetCoreMLOp<ArithmeticCoreMLOp>},
    {schema::PrimitiveType_MulFusion, GetCoreMLOp<ArithmeticCoreMLOp>},
    {schema::PrimitiveType_Reshape, GetCoreMLOp<ReshapeCoreMLOp>},
    {schema::PrimitiveType_Resize, GetCoreMLOp<ResizeCoreMLOp>},
    {schema::PrimitiveType_Concat, GetCoreMLOp<ConcatCoreMLOp>},
    {schema::PrimitiveType_Shape, GetCoreMLOp<ShapeCoreMLOp>},
    {schema::PrimitiveType_Gather, GetCoreMLOp<GatherCoreMLOp>},
    {schema::PrimitiveType_Unsqueeze, GetCoreMLOp<UnsqueezeCoreMLOp>},
    {schema::PrimitiveType_MatMulFusion, GetCoreMLOp<MatMulCoreMLOp>},
    {schema::PrimitiveType_Softmax, GetCoreMLOp<SoftmaxCoreMLOp>},
    {schema::PrimitiveType_Flatten, GetCoreMLOp<FlattenCoreMLOp>},
    {schema::PrimitiveType_ExpFusion, GetCoreMLOp<ArithmeticSelfCoreMLOp>},
    {schema::PrimitiveType_Sqrt, GetCoreMLOp<ArithmeticSelfCoreMLOp>},
    {schema::PrimitiveType_Split, GetCoreMLOp<SplitCoreMLOp>},
  };
  return mindspore::kSuccess;
}

Status CoreMLDelegateImpl::Build(DelegateModel<schema::Primitive> *model) {
  KernelIter from, end;
  std::vector<CoreMLOp *> coreml_ops;
  for (KernelIter iter = model->BeginKernelIterator(); iter != model->EndKernelIterator(); iter++) {
    kernel::Kernel *kernel = *iter;
    auto coreml_op = GetOP(kernel, model->GetPrimitive(kernel));
    if (coreml_op != nullptr) {
      // If coreml_op does not equal nullptr, this kernel can be supported by delegate
      if (coreml_ops.size() == 0) {
        from = iter;
      }
      coreml_ops.push_back(coreml_op);
      end = iter;
    } else {
      if (!coreml_ops.empty()) {
        auto coreml_graph_kernel = CreateCoreMLGraph(coreml_ops, model, from, end);
        if (coreml_graph_kernel == nullptr) {
          MS_LOG(ERROR) << "Create CoreML Graph failed.";
          return mindspore::kLiteNullptr;
        }
        iter = model->Replace(from, end + 1, coreml_graph_kernel);
        coreml_ops.clear();
      }
    }
  }
  if (!coreml_ops.empty()) {
    auto coreml_graph_kernel = CreateCoreMLGraph(coreml_ops, model, from, end);
    if (coreml_graph_kernel == nullptr) {
      MS_LOG(ERROR) << "Create CoreML Graph failed.";
      return mindspore::kLiteNullptr;
    }
    model->Replace(from, end + 1, coreml_graph_kernel);
    coreml_ops.clear();
  }
  MS_LOG(ERROR) << "CoreML graph build success!";
  return mindspore::kSuccess;
}

CoreMLOp *CoreMLDelegateImpl::GetOP(kernel::Kernel *kernel, const schema::Primitive *primitive) {
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "primitive is NULL!";
    return nullptr;
  }
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "kernel is NULL!";
    return nullptr;
  }
  auto name = kernel->name();
  CoreMLOp *coreml_op = nullptr;
  auto node_type = primitive->value_type();
  if (op_func_lists_.find(node_type) != op_func_lists_.end()) {
    coreml_op = op_func_lists_[node_type](primitive, kernel->inputs(), kernel->outputs(), name);
  } else {
    MS_LOG(DEBUG) << "Unsupported op type for CoreML.";
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

  if (coreml_op != nullptr) {
    MS_LOG(DEBUG) << "kernel: [" << kernel->name().c_str() << "] op success. "
                  << "op_type: " << PrimitiveCurVersionTypeName(kernel->type());
  }
  return coreml_op;
}

kernel::Kernel *CoreMLDelegateImpl::CreateCoreMLGraph(const std::vector<CoreMLOp *> &ops,
                                                      DelegateModel<schema::Primitive> *model, KernelIter from,
                                                      KernelIter end) {
  auto in_tensors = GetGraphInTensors(ops, nullptr);
  auto out_tensors = GraphOutTensors<CoreMLOp>(ops, model, from, end);
  auto graph_kernel = new (std::nothrow) CoreMLGraph(ops, in_tensors, out_tensors);
  if (graph_kernel == nullptr) {
    MS_LOG(ERROR) << "New CoreML Graph failed.";
    return nullptr;
  }
  graph_kernel->set_name("CoreMLGraph" + std::to_string(graph_index_++));

  // 1. For every op, find pre and next ops
  FindPreNextOps<CoreMLOp>(ops);

  // 2. Run pass
  auto ret = pass_manager_->RunPass(graph_kernel);
  if (ret != RET_OK) {
    delete graph_kernel;
    MS_LOG(ERROR) << "CoreML Graph run pass failed. This function mainly solves the problem that the format is "
                     "inconsistent and requires interpolation transpose operators.";
    return nullptr;
  }

  // 3. CoreMLGraph init, build and compile the MLModel
  ret = graph_kernel->Init();
  if (ret != RET_OK) {
    delete graph_kernel;
    MS_LOG(ERROR) << "CoreML subgraph Init failed.";
    return nullptr;
  }
  return graph_kernel;
}
}  // namespace lite

// the definition of open CoreMLDelegate class
CoreMLDelegate::CoreMLDelegate() : impl_(nullptr) {}

Status CoreMLDelegate::Init() {
  if (impl_ == nullptr) {
    impl_ = std::make_shared<lite::CoreMLDelegateImpl>();
    if (impl_ == nullptr) {
      MS_LOG(ERROR) << "new CoreMLDelegate inner implementation failed.";
      return kLiteError;
    }
  }
  Status ret = impl_->Init();
  return ret;
}

Status CoreMLDelegate::Build(DelegateModel<schema::Primitive> *model) {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "CoreMLDelegate implementation is null.";
    return kLiteNullptr;
  }
  Status ret = impl_->Build(model);
  return ret;
}
}  // namespace mindspore
