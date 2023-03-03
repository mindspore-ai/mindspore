/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#include "transform/graph_ir/op_declare/nn_norm_ops_declare.h"
#include <vector>
#include <string>

namespace mindspore::transform {
// SoftmaxV2
INPUT_MAP(SoftmaxV2) = {{1, INPUT_DESC(x)}};
ATTR_MAP(SoftmaxV2) = {
  {"axis", ATTR_DESC(axes, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
};
OUTPUT_MAP(SoftmaxV2) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Softmax, kNameSoftmax, ADPT_DESC(SoftmaxV2))
REG_ADPT_DESC(SoftmaxV2, kSoftmaxV2OpName, ADPT_DESC(SoftmaxV2))

// SoftmaxGrad
INPUT_MAP(SoftmaxGrad) = {{1, INPUT_DESC(softmax)}, {2, INPUT_DESC(grad_softmax)}};
OUTPUT_MAP(SoftmaxGrad) = {{0, OUTPUT_DESC(grad_x)}};
ATTR_MAP(SoftmaxGrad) = EMPTY_ATTR_MAP;
REG_ADPT_DESC(SoftmaxGrad, kNameSoftmaxGrad, ADPT_DESC(SoftmaxGrad))

// SoftmaxCrossEntropyWithLogits
INPUT_MAP(SoftmaxCrossEntropyWithLogits) = {{1, INPUT_DESC(features)}, {2, INPUT_DESC(labels)}};
ATTR_MAP(SoftmaxCrossEntropyWithLogits) = EMPTY_ATTR_MAP;
OUTPUT_MAP(SoftmaxCrossEntropyWithLogits) = {{0, OUTPUT_DESC(loss)}, {1, OUTPUT_DESC(backprop)}};
REG_ADPT_DESC(SoftmaxCrossEntropyWithLogits, prim::kPrimSoftmaxCrossEntropyWithLogits->name(),
              ADPT_DESC(SoftmaxCrossEntropyWithLogits))

// SmoothL1Loss
INPUT_MAP(SmoothL1LossV2) = {{1, INPUT_DESC(predict)}, {2, INPUT_DESC(label)}};
ATTR_MAP(SmoothL1LossV2) = {{"beta", ATTR_DESC(sigma, AnyTraits<float>())},
                            {"reduction", ATTR_DESC(reduction, AnyTraits<std::string>())}};
OUTPUT_MAP(SmoothL1LossV2) = {{0, OUTPUT_DESC(loss)}};
REG_ADPT_DESC(SmoothL1Loss, kNameSmoothL1Loss, ADPT_DESC(SmoothL1LossV2))
REG_ADPT_DESC(SmoothL1LossV2, prim::kPrimSmoothL1LossV2->name(), ADPT_DESC(SmoothL1LossV2))

// SmoothL1LossGrad
INPUT_MAP(SmoothL1LossGradV2) = {{1, INPUT_DESC(predict)}, {2, INPUT_DESC(label)}, {3, INPUT_DESC(dout)}};
ATTR_MAP(SmoothL1LossGradV2) = {{"beta", ATTR_DESC(sigma, AnyTraits<float>())},
                                {"reduction", ATTR_DESC(reduction, AnyTraits<std::string>())}};
OUTPUT_MAP(SmoothL1LossGradV2) = {{0, OUTPUT_DESC(gradient)}};
REG_ADPT_DESC(SmoothL1LossGrad, kNameSmoothL1LossGrad, ADPT_DESC(SmoothL1LossGradV2))
REG_ADPT_DESC(SmoothL1LossGradV2, prim::kPrimSmoothL1LossGradV2->name(), ADPT_DESC(SmoothL1LossGradV2))

// SigmoidCrossEntropyWithLogits
INPUT_MAP(SigmoidCrossEntropyWithLogits) = {{1, INPUT_DESC(predict)}, {2, INPUT_DESC(target)}};
ATTR_MAP(SigmoidCrossEntropyWithLogits) = EMPTY_ATTR_MAP;
OUTPUT_MAP(SigmoidCrossEntropyWithLogits) = {{0, OUTPUT_DESC(loss)}};
REG_ADPT_DESC(SigmoidCrossEntropyWithLogits, kNameSigmoidCrossEntropyWithLogits,
              ADPT_DESC(SigmoidCrossEntropyWithLogits))

// SigmoidCrossEntropyWithLogitsGrad
INPUT_MAP(SigmoidCrossEntropyWithLogitsGrad) = {
  {1, INPUT_DESC(predict)}, {2, INPUT_DESC(target)}, {3, INPUT_DESC(dout)}};
ATTR_MAP(SigmoidCrossEntropyWithLogitsGrad) = EMPTY_ATTR_MAP;
OUTPUT_MAP(SigmoidCrossEntropyWithLogitsGrad) = {{0, OUTPUT_DESC(gradient)}};
REG_ADPT_DESC(SigmoidCrossEntropyWithLogitsGrad, kNameSigmoidCrossEntropyWithLogitsGrad,
              ADPT_DESC(SigmoidCrossEntropyWithLogitsGrad))

// SigmoidCrossEntropyWithLogitsV2
INPUT_MAP(SigmoidCrossEntropyWithLogitsV2) = {
  {1, INPUT_DESC(predict)}, {2, INPUT_DESC(target)}, {3, INPUT_DESC(weight)}, {4, INPUT_DESC(pos_weight)}};
ATTR_MAP(SigmoidCrossEntropyWithLogitsV2) = {{"reduction", ATTR_DESC(reduction, AnyTraits<std::string>())}};
OUTPUT_MAP(SigmoidCrossEntropyWithLogitsV2) = {{0, OUTPUT_DESC(loss)}};
REG_ADPT_DESC(BCEWithLogitsLoss, kNameSigmoidCrossEntropyWithLogitsV2, ADPT_DESC(SigmoidCrossEntropyWithLogitsV2))
REG_ADPT_DESC(SigmoidCrossEntropyWithLogitsV2, kSigmoidCrossEntropyWithLogitsV2OpName,
              ADPT_DESC(SigmoidCrossEntropyWithLogitsV2))

// LogSoftmaxGrad
INPUT_MAP(LogSoftmaxGrad) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(grad)}};
ATTR_MAP(LogSoftmaxGrad) = {
  {"axis", ATTR_DESC(axis, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())}};
OUTPUT_MAP(LogSoftmaxGrad) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(LogSoftmaxGrad, prim::kPrimLogSoftmaxGrad->name(), ADPT_DESC(LogSoftmaxGrad))

// LogSoftmaxV2
INPUT_MAP(LogSoftmaxV2) = {{1, INPUT_DESC(logits)}};
ATTR_MAP(LogSoftmaxV2) = {
  {"axis", ATTR_DESC(axes, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())}};
OUTPUT_MAP(LogSoftmaxV2) = {{0, OUTPUT_DESC(logsoftmax)}};
REG_ADPT_DESC(LogSoftmax, prim::kPrimLogSoftmax->name(), ADPT_DESC(LogSoftmaxV2))
REG_ADPT_DESC(LogSoftmaxV2, kLogSoftmaxV2OpName, ADPT_DESC(LogSoftmaxV2))

// LayerNorm
INPUT_MAP(LayerNorm) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(gamma)}, {3, INPUT_DESC(beta)}};
ATTR_MAP(LayerNorm) = {{"begin_norm_axis", ATTR_DESC(begin_norm_axis, AnyTraits<int64_t>())},
                       {"begin_params_axis", ATTR_DESC(begin_params_axis, AnyTraits<int64_t>())},
                       {"epsilon", ATTR_DESC(epsilon, AnyTraits<float>())}};
OUTPUT_MAP(LayerNorm) = {{0, OUTPUT_DESC(y)}, {1, OUTPUT_DESC(mean)}, {2, OUTPUT_DESC(variance)}};
REG_ADPT_DESC(LayerNorm, prim::kPrimLayerNorm->name(), ADPT_DESC(LayerNorm))

// LayerNormGrad
INPUT_MAP(LayerNormGrad) = {
  {1, INPUT_DESC(x)}, {2, INPUT_DESC(dy)}, {3, INPUT_DESC(variance)}, {4, INPUT_DESC(mean)}, {5, INPUT_DESC(gamma)}};
ATTR_MAP(LayerNormGrad) = EMPTY_ATTR_MAP;
OUTPUT_MAP(LayerNormGrad) = {{0, OUTPUT_DESC(pd_x)}, {1, OUTPUT_DESC(pd_gamma)}, {2, OUTPUT_DESC(pd_beta)}};
REG_ADPT_DESC(LayerNormGrad, prim::kPrimLayerNormGrad->name(), ADPT_DESC(LayerNormGrad))

// LayerNormBetaGammaBackpropV2
INPUT_MAP(LayerNormBetaGammaBackpropV2) = {{1, INPUT_DESC(dy)}, {2, INPUT_DESC(res_for_gamma)}};
ATTR_MAP(LayerNormBetaGammaBackpropV2) = {{"shape_gamma", ATTR_DESC(shape_gamma, AnyTraits<std::vector<int64_t>>())}};
OUTPUT_MAP(LayerNormBetaGammaBackpropV2) = {{0, OUTPUT_DESC(pd_gamma)}, {1, OUTPUT_DESC(pd_beta)}};
REG_ADPT_DESC(LayerNormBetaGammaBackpropV2, kLayerNormBetaGammaBackpropV2OpName,
              ADPT_DESC(LayerNormBetaGammaBackpropV2))

// LayerNormXBackpropV2
INPUT_MAP(LayerNormXBackpropV2) = {
  {1, INPUT_DESC(dy)}, {2, INPUT_DESC(x)}, {3, INPUT_DESC(variance)}, {4, INPUT_DESC(mean)}, {5, INPUT_DESC(gamma)}};
ATTR_MAP(LayerNormXBackpropV2) = EMPTY_ATTR_MAP;
OUTPUT_MAP(LayerNormXBackpropV2) = {{0, OUTPUT_DESC(pd_x)}, {1, OUTPUT_DESC(res_for_gamma)}};
REG_ADPT_DESC(LayerNormXBackpropV2, kLayerNormXBackpropV2OpName, ADPT_DESC(LayerNormXBackpropV2))

// LRN
INPUT_MAP(LRN) = {{1, INPUT_DESC(x)}};
ATTR_MAP(LRN) = {{"depth_radius", ATTR_DESC(depth_radius, AnyTraits<int64_t>())},
                 {"bias", ATTR_DESC(bias, AnyTraits<float>())},
                 {"alpha", ATTR_DESC(alpha, AnyTraits<float>())},
                 {"beta", ATTR_DESC(beta, AnyTraits<float>())},
                 {"norm_region", ATTR_DESC(norm_region, AnyTraits<string>())}};
OUTPUT_MAP(LRN) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(LRN, kNameLRN, ADPT_DESC(LRN))

// LRNGrad
INPUT_MAP(LRNGrad) = {{1, INPUT_DESC(grads)}, {2, INPUT_DESC(x)}, {3, INPUT_DESC(y)}};
ATTR_MAP(LRNGrad) = {{"depth_radius", ATTR_DESC(depth_radius, AnyTraits<int64_t>())},
                     {"bias", ATTR_DESC(bias, AnyTraits<float>())},
                     {"alpha", ATTR_DESC(alpha, AnyTraits<float>())},
                     {"beta", ATTR_DESC(beta, AnyTraits<float>())}};
OUTPUT_MAP(LRNGrad) = {{0, OUTPUT_DESC(z)}};
REG_ADPT_DESC(LRNGrad, kNameLRNGrad, ADPT_DESC(LRNGrad))

// DropoutDoMask
INPUT_MAP(DropOutDoMask) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(mask)}, {3, INPUT_DESC(keep_prob)}};
ATTR_MAP(DropOutDoMask) = EMPTY_ATTR_MAP;
OUTPUT_MAP(DropOutDoMask) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(DropOutDoMask, kDropOutDoMaskOpName, ADPT_DESC(DropOutDoMask))
REG_ADPT_DESC(DropoutDoMask, kDropoutDoMaskOpName, ADPT_DESC(DropOutDoMask))

// DropOutDoMaskV3
INPUT_MAP(DropOutDoMaskV3) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(mask)}, {3, INPUT_DESC(keep_prob)}};
ATTR_MAP(DropOutDoMaskV3) = EMPTY_ATTR_MAP;
OUTPUT_MAP(DropOutDoMaskV3) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(DropOutDoMaskV3, kNameDropOutDoMaskV3, ADPT_DESC(DropOutDoMaskV3))

// DropOutDoMaskV3D
INPUT_MAP(DropOutDoMaskV3D) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(mask)}};
INPUT_ATTR_MAP(DropOutDoMaskV3D) = {{3, ATTR_DESC(keep_prob, AnyTraits<float>())}};
ATTR_MAP(DropOutDoMaskV3D) = EMPTY_ATTR_MAP;
OUTPUT_MAP(DropOutDoMaskV3D) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(DropOutDoMaskV3D, kNameDropOutDoMaskV3D, ADPT_DESC(DropOutDoMaskV3D))

// BinaryCrossEntropy
INPUT_MAP(BinaryCrossEntropy) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(y)}, {3, INPUT_DESC(weight)}};
ATTR_MAP(BinaryCrossEntropy) = {{"reduction", ATTR_DESC(reduction, AnyTraits<std::string>())}};
OUTPUT_MAP(BinaryCrossEntropy) = {{0, OUTPUT_DESC(output)}};
REG_ADPT_DESC(BinaryCrossEntropy, kNameBinaryCrossEntropy, ADPT_DESC(BinaryCrossEntropy))

// BinaryCrossEntropyGrad
INPUT_MAP(BinaryCrossEntropyGrad) = {
  {1, INPUT_DESC(x)}, {2, INPUT_DESC(y)}, {3, INPUT_DESC(grad_output)}, {4, INPUT_DESC(weight)}};
ATTR_MAP(BinaryCrossEntropyGrad) = {{"reduction", ATTR_DESC(reduction, AnyTraits<std::string>())}};
OUTPUT_MAP(BinaryCrossEntropyGrad) = {{0, OUTPUT_DESC(output)}};
REG_ADPT_DESC(BinaryCrossEntropyGrad, kNameBinaryCrossEntropyGrad, ADPT_DESC(BinaryCrossEntropyGrad))

// Centralization
INPUT_MAP(Centralization) = {{1, INPUT_DESC(x)}};
INPUT_ATTR_MAP(Centralization) = {
  {2, ATTR_DESC(axes, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())}};
ATTR_MAP(Centralization) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Centralization) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Centralization, kNameCentralization, ADPT_DESC(Centralization))

// Scale
INPUT_MAP(Scale) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(scale)}, {3, INPUT_DESC(bias)}};
ATTR_MAP(Scale) = {{"axis", ATTR_DESC(axis, AnyTraits<int64_t>())},
                   {"num_axes", ATTR_DESC(num_axes, AnyTraits<int64_t>())},
                   {"scale_from_blob", ATTR_DESC(scale_from_blob, AnyTraits<bool>())}};

OUTPUT_MAP(Scale) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Scale, kNameScale, ADPT_DESC(Scale))

// KlDivLossGrad
INPUT_MAP(KlDivLossGrad) = {{1, INPUT_DESC(grad)}, {2, INPUT_DESC(input)}, {3, INPUT_DESC(target)}};
ATTR_MAP(KlDivLossGrad) = {{"reduction", ATTR_DESC(reduction, AnyTraits<std::string>())},
                           {"log_target", ATTR_DESC(log_target, AnyTraits<bool>())}};
OUTPUT_MAP(KlDivLossGrad) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(KlDivLossGrad, kNameKlDivLossGrad, ADPT_DESC(KlDivLossGrad))

// InstanceNorm
INPUT_MAP(InstanceNorm) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(gamma)}, {3, INPUT_DESC(beta)}};
ATTR_MAP(InstanceNorm) = {{"data_format", ATTR_DESC(data_format, AnyTraits<std::string>())},
                          {"epsilon", ATTR_DESC(epsilon, AnyTraits<float>())}};
OUTPUT_MAP(InstanceNorm) = {{0, OUTPUT_DESC(y)}, {1, OUTPUT_DESC(mean)}, {2, OUTPUT_DESC(variance)}};
REG_ADPT_DESC(InstanceNorm, prim::kPrimInstanceNorm->name(), ADPT_DESC(InstanceNorm))

// MultilabelMarginLoss
INPUT_MAP(MultilabelMarginLoss) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(target)}};
ATTR_MAP(MultilabelMarginLoss) = {{"reduction", ATTR_DESC(reduction, AnyTraits<std::string>())}};
OUTPUT_MAP(MultilabelMarginLoss) = {{0, OUTPUT_DESC(y)}, {1, OUTPUT_DESC(is_target)}};
REG_ADPT_DESC(MultilabelMarginLoss, prim::kPrimMultilabelMarginLoss->name(), ADPT_DESC(MultilabelMarginLoss))

// Roll
INPUT_MAP(Roll) = {{1, INPUT_DESC(x)}};
ATTR_MAP(Roll) = {{"shift", ATTR_DESC(shifts, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())},
                  {"axis", ATTR_DESC(dims, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())}};
OUTPUT_MAP(Roll) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Roll, prim::kRoll, ADPT_DESC(Roll))

// Renorm
INPUT_MAP(Renorm) = {{1, INPUT_DESC(x)}};
ATTR_MAP(Renorm) = {{"p", ATTR_DESC(p, AnyTraits<float>())},
                    {"dim", ATTR_DESC(dim, AnyTraits<int64_t>())},
                    {"maxnorm", ATTR_DESC(maxnorm, AnyTraits<float>())}};
OUTPUT_MAP(Renorm) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Renorm, prim::kPrimRenorm->name(), ADPT_DESC(Renorm))

// SoftMarginLoss
INPUT_MAP(SoftMarginLoss) = {{1, INPUT_DESC(input_x)}, {2, INPUT_DESC(input_y)}};
ATTR_MAP(SoftMarginLoss) = {{"reduction", ATTR_DESC(reduction, AnyTraits<std::string>())}};
OUTPUT_MAP(SoftMarginLoss) = {{0, OUTPUT_DESC(output_z)}};
REG_ADPT_DESC(SoftMarginLoss, prim::kPrimSoftMarginLoss->name(), ADPT_DESC(SoftMarginLoss))

// ConfusionSoftmaxGrad
INPUT_MAP(ConfusionSoftmaxGrad) = {{1, INPUT_DESC(grad)}, {2, INPUT_DESC(x)}};
ATTR_MAP(ConfusionSoftmaxGrad) = EMPTY_ATTR_MAP;
OUTPUT_MAP(ConfusionSoftmaxGrad) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(ConfusionSoftmaxGrad, "ConfusionSoftmaxGrad", ADPT_DESC(ConfusionSoftmaxGrad))

// SoftmaxGradExt
INPUT_MAP(SoftmaxGradExt) = {{1, INPUT_DESC(grad)}, {2, INPUT_DESC(x1)}, {3, INPUT_DESC(x2)}};
OUTPUT_MAP(SoftmaxGradExt) = {{0, OUTPUT_DESC(y)}};
ATTR_MAP(SoftmaxGradExt) = {{"axis", ATTR_DESC(axes, AnyTraits<int64_t>(), AnyTraits<int64_t>())},
                            {"keep_dims", ATTR_DESC(keep_dims, AnyTraits<bool>(), AnyTraits<bool>())}};
REG_ADPT_DESC(SoftmaxGradExt, kSoftmaxGradExtOpName, ADPT_DESC(SoftmaxGradExt))
}  // namespace mindspore::transform
