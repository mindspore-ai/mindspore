/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "src/extendrt/graph_compiler/infershape_helper.h"
#include <set>
#include <string>
#include <algorithm>
#include <vector>
#include <memory>
#include "src/common/ops/operator_populate/operator_populate_register.h"
#include "src/extendrt/graph_compiler/anfnode_tensor_adapter.h"
#include "src/litert/pass/format_pass/format_pass.h"
#include "tools/optimizer/graph/node_infershape.h"
#include "abstract/dshape.h"

#include "ops/adam.h"
#include "ops/apply_momentum.h"
#include "ops/batch_norm.h"
#include "ops/batch_to_space.h"
#include "ops/bias_add.h"
#include "ops/depth_to_space.h"
#include "ops/fused_batch_norm.h"
#include "ops/fusion/avg_pool_fusion.h"
#include "ops/fusion/conv2d_backprop_input_fusion.h"
#include "ops/fusion/conv2d_backprop_filter_fusion.h"
#include "ops/fusion/conv2d_fusion.h"
#include "ops/fusion/conv2d_transpose_fusion.h"
#include "ops/fusion/max_pool_fusion.h"
#include "ops/fusion/prelu_fusion.h"
#include "ops/grad/avg_pool_grad.h"
#include "ops/grad/batch_norm_grad.h"
#include "ops/grad/bias_add_grad.h"
#include "ops/grad/max_pool_grad.h"
#include "ops/grad/resize_grad.h"
#include "ops/instance_norm.h"
#include "ops/lrn.h"
#include "ops/op_utils.h"
#include "ops/resize.h"
#include "ops/roi_pooling.h"
#include "ops/sgd.h"
#include "ops/space_to_batch.h"
#include "ops/space_to_batch_nd.h"
#include "ops/space_to_depth.h"
#include "ops/grid_sampler_2d.h"

namespace mindspore::lite {
namespace {
static const std::set<std::string> FormatAwareOp = {ops::kNameAdam,
                                                    ops::kNameApplyMomentum,
                                                    ops::kNameAvgPoolFusion,
                                                    ops::kNameAvgPoolGrad,
                                                    ops::kNameBatchNorm,
                                                    ops::kNameBatchNormGrad,
                                                    ops::kNameBatchToSpace,
                                                    ops::kNameBiasAdd,
                                                    ops::kNameBiasAddGrad,
                                                    ops::kNameConv2DBackpropInputFusion,
                                                    ops::kNameConv2DBackpropFilterFusion,
                                                    ops::kNameConv2DFusion,
                                                    ops::kNameConv2dTransposeFusion,
                                                    ops::kNameDepthToSpace,
                                                    ops::kNameFusedBatchNorm,
                                                    ops::kNameGridSampler2D,
                                                    ops::kNameInstanceNorm,
                                                    ops::kNameLRN,
                                                    ops::kNameMaxPoolFusion,
                                                    ops::kNameMaxPoolGrad,
                                                    ops::kNamePReLUFusion,
                                                    ops::kNameResize,
                                                    ops::kNameResizeGrad,
                                                    ops::kNameROIPooling,
                                                    ops::kNameSGD,
                                                    ops::kNameSpaceToBatch,
                                                    ops::kNameSpaceToBatchND,
                                                    ops::kNameSpaceToDepth};

constexpr int kNCHW2NHWC = 0;
constexpr int kNHWC2NCHW = 1;
void TransposeShape(InferTensor *tensor, int transpose_type) {
  if (MS_UNLIKELY(tensor == nullptr)) {
    return;
  }
  auto shape = tensor->shape();
  constexpr int kNCHWDimSize = 4;
  if (shape.size() != kNCHWDimSize) {
    return;
  }
  std::vector<int> new_shape(kNCHWDimSize);
  if (transpose_type == kNCHW2NHWC) {
    new_shape[kNHWC_N] = shape[kNCHW_N];
    new_shape[kNHWC_H] = shape[kNCHW_H];
    new_shape[kNHWC_W] = shape[kNCHW_W];
    new_shape[kNHWC_C] = shape[kNCHW_C];
    tensor->set_shape(new_shape);
    tensor->set_format(NHWC);
    return;
  } else if (transpose_type == kNHWC2NCHW) {
    new_shape[kNCHW_N] = shape[kNHWC_N];
    new_shape[kNCHW_C] = shape[kNHWC_C];
    new_shape[kNCHW_H] = shape[kNHWC_H];
    new_shape[kNCHW_W] = shape[kNHWC_W];
    tensor->set_shape(new_shape);
    tensor->set_format(NCHW);
    return;
  }
}

void TransposeShape(std::vector<InferTensor *> *tensors, int transpose_type) {
  if (MS_UNLIKELY(tensors == nullptr)) {
    return;
  }
  for (auto *tensor : *tensors) {
    TransposeShape(tensor, transpose_type);
  }
}

int SyncInferRetToLiteTensor(const CompileNode &node, const int &infer_ret) {
  if (infer_ret == RET_INFER_INVALID) {
    for (auto *output : node.GetOutputs()) {
      output->set_shape({-1});
    }
  }
  auto cnode = node.GetCNode();
  MS_ASSERT(cnode != nullptr);
  auto abstract = cnode->abstract();
  if (utils::isa<mindspore::abstract::AbstractSequencePtr>(abstract)) {
    auto elements = utils::cast<mindspore::abstract::AbstractSequencePtr>(abstract)->elements();
    if (elements.size() != node.OutputSize()) {
      MS_LOG(INFO) << "The cnode output size: " << elements.size()
                   << " is not equal to lite tensors size: " << node.OutputSize();
      return RET_ERROR;
    }
    for (size_t i = 0; i < elements.size(); i++) {
      if (!TensorAdapter::SetDTAndShapeFromAbTensorToLiteTensor(elements[i], node.GetOutput(i))) {
        MS_LOG(INFO) << "Sync infershape result from Abstract to InferTensor failed, node : " << node.GetName();
        return RET_ERROR;
      }
    }
    return RET_OK;
  }
  if (utils::isa<mindspore::abstract::AbstractTensorPtr>(abstract)) {
    if (!TensorAdapter::SetDTAndShapeFromAbTensorToLiteTensor(abstract, node.GetOutput(0))) {
      MS_LOG(INFO) << "Sync infershape result from Abstract to InferTensor failed, node : " << node.GetName();
      return RET_ERROR;
    }
    return RET_OK;
  }
  MS_LOG(INFO) << "Unsupported abstract type: " << abstract;
  return RET_ERROR;
}

int SyncInferRetToCNodeNative(const CompileNode &node) {
  auto cnode = node.GetCNode();
  MS_ASSERT(cnode != nullptr);
  const auto &outputs = node.GetOutputs();
  if (outputs.empty()) {
    return RET_OK;
  }
  auto abstract = cnode->abstract();
  if (utils::isa<abstract::AbstractTuplePtr>(abstract)) {
    auto abs_tuple = utils::cast<abstract::AbstractTuplePtr>(abstract);
    MS_ASSERT(abs_tuple != nullptr);
    if (abs_tuple->elements().size() != outputs.size()) {
      MS_LOG(INFO) << "Node(" << node.GetName() << ") has " << outputs.size()
                   << " output tensor(s), but its AbstractTuple has " << abs_tuple->elements().size() << " element(s).";
      return RET_ERROR;
    }
    for (size_t i = 0; i < outputs.size(); i++) {
      if (!TensorAdapter::SetDTAndShapeFromLiteTensorToAbTensor(*outputs[i], abs_tuple->elements()[i])) {
        MS_LOG(INFO) << "Sync infershape result from InferTensor to Abstract failed, " << node.GetName();
        return RET_ERROR;
      }
    }
    cnode->set_abstract(abs_tuple);
    return RET_OK;
  }
  if (utils::isa<mindspore::abstract::AbstractTensorPtr>(abstract)) {
    if (outputs.size() != 1) {
      MS_LOG(INFO) << "Node(" << node.GetName() << ")'s abstract is an AbstractTensor but has " << outputs.size()
                   << " output tensor(s).";
      return RET_ERROR;
    }
    auto abs_tensor = utils::cast<abstract::AbstractTensorPtr>(abstract);
    MS_ASSERT(abs_tensor != nullptr);
    if (!TensorAdapter::SetDTAndShapeFromLiteTensorToAbTensor(*outputs[0], abs_tensor)) {
      MS_LOG(INFO) << "Sync infershape result from InferTensor to Abstract failed, " << node.GetName();
      return RET_ERROR;
    }
    cnode->set_abstract(abs_tensor);
    return RET_OK;
  }
  MS_LOG(INFO) << "Unsupported abstract type: " << abstract;
  return RET_ERROR;
}

int SyncInferRetToCNode(const CompileNode &node, const int &infer_ret) {
  const auto &outputs = node.GetOutputs();
  if (infer_ret == RET_INFER_INVALID) {
    for (auto *output : outputs) {
      output->set_shape({abstract::Shape::kShapeRankAny});
    }
  }
  auto ret = SyncInferRetToCNodeNative(node);
  if (infer_ret == RET_INFER_INVALID) {
    for (auto *output : outputs) {
      output->set_shape({-1});
    }
  }
  return ret;
}

int InferShapeByNNACL(const CompileNodePtr &node, OpParameter *op_parameter, Format format, InferContext *context) {
  if (format != NHWC && format != NCHW) {
    MS_LOG(INFO) << "NNACL infershape only support NCHW or NHWC format, got " << FormatEnumToString(format);
    return RET_ERROR;
  }
  auto inputs = node->GetInputs();
  auto outputs = node->GetOutputs();
  int infer_ret = RET_OK;
  for (auto *input : inputs) {
    auto shape = input->shape();
    if (std::any_of(shape.begin(), shape.end(), [](const int dim) { return dim < 0; })) {
      infer_ret = RET_INFER_INVALID;
      break;
    }
  }
  if (infer_ret != RET_INFER_INVALID) {
    if (format == NCHW) {
      TransposeShape(&inputs, kNCHW2NHWC);
      TransposeShape(&outputs, kNCHW2NHWC);
    }
    infer_ret = KernelInferShape(node->GetInputs(), node->GetOutputs(), op_parameter, context->allocator);
    if (format == NCHW) {
      TransposeShape(&inputs, kNHWC2NCHW);
      TransposeShape(&outputs, kNHWC2NCHW);
    }
  }
  if (infer_ret != RET_OK && infer_ret != RET_INFER_INVALID) {
    return infer_ret;
  }
  auto ret = SyncInferRetToCNode(*node, infer_ret);
  if (ret != RET_OK) {
    MS_LOG(INFO) << "Sync infershape result from InferTensor to Abstract failed: " << node->GetName();
    return ret;
  }
  return infer_ret;
}

int InferShapeByOps(const CompileNodePtr &node, Format format) {
  auto node_infer_shape = std::make_shared<opt::NodeInferShape>();
  if (node_infer_shape == nullptr) {
    MS_LOG(INFO) << "create NodeInferShape manager failed.";
    return false;
  }
  auto cnode = node->GetCNode();
  auto infer_ret = NodeFallBackInferShape(cnode, format);
  if (infer_ret != RET_OK && infer_ret != RET_INFER_INVALID) {
    return infer_ret;
  }

  auto ret = SyncInferRetToLiteTensor(*node, infer_ret);
  if (ret != RET_OK) {
    MS_LOG(INFO) << "Sync infershape result from Abstract to InferTensor failed: " << node->GetName();
    return ret;
  }
  return infer_ret;
}

inline void DumpInferResult(const CompileNode &node, int infer_ret) {
#ifdef Debug
  std::ostringstream oss;
  oss << "GraphFallBackInferShape(" << node.GetName() << ") InferShape ret: " << infer_ret << ", shape:";
  bool first_output = true;
  for (auto &output : node.GetOutputs()) {
    if (first_output) {
      first_output = false;
    } else {
      oss << ", ";
    }
    oss << ShapeVectorToStr(output->shape());
  }
  MS_LOG(INFO) << oss.str();
#endif
}
}  // namespace

int GraphFallBackInferShape(const FuncGraphPtr &graph, Format format, InferContext *context) { return RET_ERROR; }

int NodeFallBackInferShape(const CNodePtr &cnode, Format format) {
  if (cnode == nullptr) {
    MS_LOG(INFO) << "cnode is nullptr";
    return RET_ERROR;
  }
  auto node_infer_shape = std::make_shared<opt::NodeInferShape>();
  if (node_infer_shape == nullptr) {
    MS_LOG(INFO) << "create NodeInferShape manager failed.";
    return false;
  }
  auto anf_prim = GetValueNode<std::shared_ptr<Primitive>>(cnode->input(0));
  if (anf_prim == nullptr) {
    MS_LOG(INFO) << "primitive is nullptr";
    return RET_ERROR;
  }
  (void)anf_prim->AddAttr(ops::kFormat, MakeValue<int64_t>(static_cast<int64_t>(format)));
  //  return {-1} when infer-invalid currently. But we should support {-2} and {-1, -1, -1} in NNACL in future.
  auto infer_ret = node_infer_shape->InferShapeByOps(cnode, true);
  if (infer_ret != RET_OK && infer_ret != RET_INFER_INVALID) {
    return infer_ret;
  }
  return infer_ret;
}

namespace {
int OpsOrNNACLInferShape(const CompileNodePtr &node, OpParameter *op_parameter, InferContext *context,
                         Format infer_format = Format::DEFAULT_FORMAT) {
  if (op_parameter != nullptr) {
    infer_format = (infer_format == Format::DEFAULT_FORMAT) ? NHWC : infer_format;
    auto infer_ret = InferShapeByNNACL(node, op_parameter, infer_format, context);
    free(op_parameter);
    if (infer_ret != RET_OK && infer_ret != RET_INFER_INVALID) {
      MS_LOG(INFO) << "Infer kernel failed for op: " << node->GetName();
    }
    return infer_ret;
  } else {
    infer_format = (infer_format == Format::DEFAULT_FORMAT) ? NCHW : infer_format;
    auto infer_ret = InferShapeByOps(node, infer_format);
    if (infer_ret != RET_OK && infer_ret != RET_INFER_INVALID) {
      MS_LOG(INFO) << "Infer kernel failed for op: " << node->GetName();
    }
    return infer_ret;
  }
}
}  // namespace

int NodeFallBackInferShape(const CompileNodePtr &node, Format format, InferContext *context) {
  MSLITE_CHECK_PTR_RETURN(node, RET_PARAM_INVALID);
  auto base_operator = node->GetBaseOperator();
  MSLITE_CHECK_PTR_RETURN(base_operator, RET_NULL_PTR);
  auto op_parameter = OperatorPopulateRegistry::GetInstance()->CreatePopulateByOp(base_operator);
  auto iter = FormatAwareOp.find(node->GetType().TypeName());
  int infer_ret;
  // Format-not-aware op should infer in format indicated by format attr of mindir.
  if (iter != FormatAwareOp.end()) {
    infer_ret = OpsOrNNACLInferShape(node, op_parameter, context, format);
  } else {  // non-format-aware op not care about format, could infershape by NNACL or OPS
    infer_ret = OpsOrNNACLInferShape(node, op_parameter, context);
  }
  DumpInferResult(*node, infer_ret);
  return infer_ret;
}

int GraphFallBackInferShape(const CompileResultPtr &node_list, Format format, InferContext *context) {
  for (const auto &node : node_list->GetNodes()) {
    auto infer_ret = NodeFallBackInferShape(node, format, context);
    if (infer_ret != RET_OK && infer_ret != RET_INFER_INVALID) {
      MS_LOG(INFO) << "Infer kernel failed for op: " << node->GetName();
      return infer_ret;
    }
  }
  return RET_OK;
}
}  // namespace mindspore::lite
