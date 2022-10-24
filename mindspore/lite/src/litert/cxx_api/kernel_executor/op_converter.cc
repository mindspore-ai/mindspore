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

#include "src/litert/cxx_api/kernel_executor/op_converter.h"
#include "ops/relu.h"
#include "ops/fusion/activation.h"
#include "ops/fusion/add_fusion.h"
#include "ops/fusion/arg_max_fusion.h"
#include "ops/fusion/arg_min_fusion.h"
#include "ops/fusion/avg_pool_fusion.h"
#include "ops/fused_batch_norm.h"
#include "ops/batch_norm.h"
#include "ops/fusion/conv2d_fusion.h"
#include "ops/fusion/conv2d_transpose_fusion.h"
#include "ops/fusion/div_fusion.h"
#include "ops/fusion/mat_mul_fusion.h"
#include "ops/fusion/max_pool_fusion.h"
#include "ops/fusion/mul_fusion.h"
#include "ops/fusion/pad_fusion.h"
#include "ops/fusion/prelu_fusion.h"
#include "ops/fusion/topk_fusion.h"

namespace mindspore {
namespace lite {
std::shared_ptr<ops::BaseOperator> ReLUConverterCreators(const std::shared_ptr<ops::BaseOperator> &op) {
  auto op_converter = std::make_shared<ops::Activation>();
  op_converter->set_activation_type(ActivationType::RELU);
  return op_converter;
}

std::shared_ptr<ops::BaseOperator> SigmoidConverterCreators(const std::shared_ptr<ops::BaseOperator> &op) {
  auto op_converter = std::make_shared<ops::Activation>();
  op_converter->set_activation_type(ActivationType::SIGMOID);
  return op_converter;
}

template <class T, class T2>
std::shared_ptr<ops::BaseOperator> CommonConverterCreators(const std::shared_ptr<ops::BaseOperator> &op) {
  auto op_cast = std::dynamic_pointer_cast<T>(op);
  auto op_converter = std::make_shared<T2>();
  op_converter->SetAttrs(op_cast->attrs());
  return op_converter;
}

std::shared_ptr<ops::BaseOperator> ArgMinConverterCreators(const std::shared_ptr<ops::BaseOperator> &op) {
  auto op_converter = CommonConverterCreators<ops::ArgMin, ops::ArgMinFusion>(op);
  auto op_arg = std::dynamic_pointer_cast<ops::ArgMinFusion>(op_converter);
  op_arg->set_top_k(1);
  op_arg->set_keep_dims(true);
  return op_converter;
}

std::shared_ptr<ops::BaseOperator> ArgMaxConverterCreators(const std::shared_ptr<ops::BaseOperator> &op) {
  auto op_converter = CommonConverterCreators<ops::Argmax, ops::ArgMaxFusion>(op);
  auto op_arg = std::dynamic_pointer_cast<ops::ArgMaxFusion>(op_converter);
  op_arg->set_top_k(1);
  op_arg->set_keep_dims(true);
  return op_converter;
}

std::shared_ptr<ops::BaseOperator> TopKConverterCreators(const std::shared_ptr<ops::BaseOperator> &op) {
  auto op_converter = CommonConverterCreators<ops::TopK, ops::TopKFusion>(op);
  auto op_topk = std::dynamic_pointer_cast<ops::TopKFusion>(op_converter);
  op_topk->set_axis(-1);
  return op_converter;
}

static RegistryOpsConverter g_ReLUConverterCreatorRegistry("ReLU", ReLUConverterCreators);
static RegistryOpsConverter g_SigmoidConverterCreatorRegistry("Sigmoid", SigmoidConverterCreators);
static RegistryOpsConverter g_AddConverterCreatorRegistry("Add", CommonConverterCreators<ops::Add, ops::AddFusion>);
static RegistryOpsConverter g_ArgmaxConverterCreatorRegistry("Argmax", ArgMaxConverterCreators);
static RegistryOpsConverter g_ArgminConverterCreatorRegistry("Argmin", ArgMinConverterCreators);

static RegistryOpsConverter g_AvgPoolConverterCreatorRegistry(
  "AvgPool", CommonConverterCreators<ops::AvgPool, ops::AvgPoolFusion>);
static RegistryOpsConverter g_BatchNormConverterCreatorRegistry(
  "BatchNorm", CommonConverterCreators<ops::BatchNorm, ops::FusedBatchNorm>);
static RegistryOpsConverter g_Conv2DConverterCreatorRegistry("Conv2D",
                                                             CommonConverterCreators<ops::Conv2D, ops::Conv2DFusion>);
static RegistryOpsConverter g_Conv2DTransposeConverterCreatorRegistry(
  "Conv2DTranspose", CommonConverterCreators<ops::Conv2DTranspose, ops::Conv2dTransposeFusion>);
static RegistryOpsConverter g_DivConverterCreatorRegistry("Div", CommonConverterCreators<ops::Div, ops::DivFusion>);
static RegistryOpsConverter g_MatMulConverterCreatorRegistry("MatMul",
                                                             CommonConverterCreators<ops::MatMul, ops::MatMulFusion>);
static RegistryOpsConverter g_MaxPoolConverterCreatorRegistry(
  "MaxPool", CommonConverterCreators<ops::MaxPool, ops::MaxPoolFusion>);
static RegistryOpsConverter g_MulConverterCreatorRegistry("Mul", CommonConverterCreators<ops::Mul, ops::MulFusion>);
static RegistryOpsConverter g_PadConverterCreatorRegistry("Pad", CommonConverterCreators<ops::Pad, ops::PadFusion>);
static RegistryOpsConverter g_PReLUConverterCreatorRegistry("PReLU",
                                                            CommonConverterCreators<ops::PReLU, ops::PReLUFusion>);
static RegistryOpsConverter g_TopkConverterCreatorRegistry("TopK", TopKConverterCreators);
}  // namespace lite
}  // namespace mindspore
