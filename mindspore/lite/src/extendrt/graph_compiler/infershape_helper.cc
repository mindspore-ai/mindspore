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

namespace mindspore {
namespace lite {
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

bool SetDTAndShapeFromAbTensorToLiteTensor(const AbstractBasePtr &abstract, lite::Tensor *tensor) {
  if (!utils::isa<mindspore::abstract::AbstractTensorPtr>(abstract)) {
    MS_LOG(ERROR) << "The abstract should be tensor, but got abstract : " << abstract;
    return false;
  }
  ShapeVector shape_vector;
  TypeId data_type = kTypeUnknown;
  auto ret = TensorAdapter::GetDTAndShapeFromAbTensor(utils::cast<mindspore::abstract::AbstractTensorPtr>(abstract),
                                                      &data_type, &shape_vector);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "Get dtype and shape from abstract failed, abstract : " << abstract;
    return false;
  }
  std::vector<int32_t> int32_shape;
  std::transform(shape_vector.begin(), shape_vector.end(), std::back_inserter(int32_shape),
                 [](const auto &shape) { return static_cast<int32_t>(shape); });
  tensor->set_data_type(data_type);
  tensor->set_shape(int32_shape);
  tensor->set_format(NHWC);
  return true;
}

constexpr int kNCHW2NHWC = 0;
constexpr int kNHWC2NCHW = 1;
void TransposeShape(Tensor *tensor, int transpose_type) {
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

void TransposeShape(std::vector<Tensor *> *tensors, int transpose_type) {
  if (MS_UNLIKELY(tensors == nullptr)) {
    return;
  }
  for (auto *tensor : *tensors) {
    TransposeShape(tensor, transpose_type);
  }
}

int InferShapeByNNACL(CompileNode *node, OpParameter *op_parameter, Format format, InferContext *context) {
  if (format != NHWC && format != NCHW) {
    MS_LOG(ERROR) << "NNACL infershape only support NCHW or NHWC format, got " << FormatEnumToString(format);
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
  if (infer_ret == RET_INFER_INVALID) {
    for (auto *output : outputs) {
      output->set_shape({abstract::Shape::kShapeRankAny});
    }
  }
  return infer_ret;
}

int InferShapeByOps(CompileNode *node, Format format) {
  auto node_infer_shape = std::make_shared<opt::NodeInferShape>();
  if (node_infer_shape == nullptr) {
    MS_LOG(ERROR) << "create NodeInferShape manager failed.";
    return false;
  }
  auto cnode = node->GetCNode();
  if (cnode == nullptr) {
    MS_LOG(ERROR) << "cnode is nullptr";
    return lite::RET_ERROR;
  }
  auto anf_prim = GetValueNode<std::shared_ptr<Primitive>>(cnode->input(0));
  if (anf_prim == nullptr) {
    MS_LOG(ERROR) << "primitive is nullptr";
    return lite::RET_ERROR;
  }
  (void)anf_prim->AddAttr(ops::kFormat, MakeValue<int64_t>(static_cast<int64_t>(format)));
  auto ret = node_infer_shape->InferShapeByOps(cnode, true);
  if (ret != lite::RET_OK) {  // invalid is no need to sync output shape from abstract
    return ret;
  }

  auto abstract = cnode->abstract();
  if (utils::isa<mindspore::abstract::AbstractSequencePtr>(abstract)) {
    auto elements = utils::cast<mindspore::abstract::AbstractSequencePtr>(abstract)->elements();
    if (elements.size() != node->OutputSize()) {
      MS_LOG(ERROR) << "The cnode output size: " << elements.size()
                    << " is not equal to lite tensors size: " << node->OutputSize();
      return lite::RET_ERROR;
    }
    for (size_t i = 0; i < elements.size(); i++) {
      if (!SetDTAndShapeFromAbTensorToLiteTensor(elements[i], node->GetOutput(i))) {
        MS_LOG(ERROR) << "Set tensor info from abstract failed, abstract : " << elements[i];
        return lite::RET_ERROR;
      }
    }
    return lite::RET_OK;
  }
  if (utils::isa<mindspore::abstract::AbstractTensorPtr>(abstract)) {
    if (!SetDTAndShapeFromAbTensorToLiteTensor(abstract, node->GetOutput(0))) {
      MS_LOG(ERROR) << "Set tensor info from abstract failed, abstract : " << abstract;
      return lite::RET_ERROR;
    }
    return lite::RET_OK;
  }
  MS_LOG(ERROR) << "Unsupported abstract type: " << abstract;
  return lite::RET_ERROR;
}
}  // namespace

int FallBackInferShape(const CompileResultPtr &node_list, Format format, InferContext *context) {
  for (const auto &node : node_list->GetNodes()) {
    MSLITE_CHECK_PTR_RETURN(node, false);
    auto base_operator = node->GetBaseOperator();
    MSLITE_CHECK_PTR_RETURN(base_operator, false);
    auto op_parameter = lite::OperatorPopulateRegistry::GetInstance()->CreatePopulateByOp(base_operator);
    auto iter = FormatAwareOp.find(node->GetType().PBType());
    // Format-not-aware op should infer in format indicated by format attr of mindir.
    if (iter != FormatAwareOp.end()) {
      if (op_parameter != nullptr) {
        auto ret = InferShapeByNNACL(node, op_parameter, format, context);
        free(op_parameter);
        if (ret != lite::RET_OK && ret != lite::RET_INFER_INVALID) {
          MS_LOG(ERROR) << "Infer kernel failed for op: " << node->GetName();
          return ret;
        }
      } else {
        auto ret = InferShapeByOps(node, format);
        if (ret != lite::RET_OK && ret != lite::RET_INFER_INVALID) {
          MS_LOG(ERROR) << "Infer kernel failed for op: " << node->GetName();
          return ret;
        }
      }
    } else {  // non-format-aware op not care about format, could infershape by NNACL or OPS
      if (op_parameter != nullptr) {
        auto ret = InferShapeByNNACL(node, op_parameter, NHWC, context);
        free(op_parameter);
        if (ret != lite::RET_OK && ret != lite::RET_INFER_INVALID) {
          MS_LOG(ERROR) << "Infer kernel failed for op: " << node->GetName();
          return ret;
        }
      } else {
        auto ret = InferShapeByOps(node, NCHW);
        if (ret != lite::RET_OK && ret != lite::RET_INFER_INVALID) {
          MS_LOG(ERROR) << "Infer kernel failed for op: " << node->GetName();
          return ret;
        }
      }
    }
  }
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
