/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "tools/optimizer/common/format_utils.h"
#include <memory>
#include <vector>
#include <string>
#include <unordered_map>
#include "ops/adam.h"
#include "ops/addn.h"
#include "ops/apply_momentum.h"
#include "ops/batch_norm.h"
#include "ops/batch_to_space.h"
#include "ops/bias_add.h"
#include "ops/cast.h"
#include "ops/concat.h"
#include "ops/crop.h"
#include "ops/depth_to_space.h"
#include "ops/fused_batch_norm.h"
#include "ops/fusion/activation.h"
#include "ops/fusion/add_fusion.h"
#include "ops/fusion/avg_pool_fusion.h"
#include "ops/fusion/conv2d_backprop_input_fusion.h"
#include "ops/fusion/conv2d_backprop_filter_fusion.h"
#include "ops/fusion/conv2d_fusion.h"
#include "ops/fusion/conv2d_transpose_fusion.h"
#include "ops/fusion/div_fusion.h"
#include "ops/fusion/max_pool_fusion.h"
#include "ops/fusion/mul_fusion.h"
#include "ops/fusion/pad_fusion.h"
#include "ops/fusion/pow_fusion.h"
#include "ops/fusion/prelu_fusion.h"
#include "ops/fusion/sub_fusion.h"
#include "ops/fusion/scale_fusion.h"
#include "ops/fusion/slice_fusion.h"
#include "ops/fusion/topk_fusion.h"
#include "ops/eltwise.h"
#include "ops/erf.h"
#include "ops/grad/activation_grad.h"
#include "ops/grad/avg_pool_grad.h"
#include "ops/grad/batch_norm_grad.h"
#include "ops/grad/bias_add_grad.h"
#include "ops/grad/max_pool_grad.h"
#include "ops/grad/resize_grad.h"
#include "ops/instance_norm.h"
#include "ops/lrn.h"
#include "ops/maximum.h"
#include "ops/op_utils.h"
#include "ops/quant_dtype_cast.h"
#include "ops/resize.h"
#include "ops/roi_pooling.h"
#include "ops/sgd.h"
#include "ops/space_to_batch.h"
#include "ops/space_to_batch_nd.h"
#include "ops/space_to_depth.h"
#include "ops/split.h"
#include "ops/strided_slice.h"
#include "tools/lite_exporter/fetch_content.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace opt {
static const std::unordered_map<std::string, std::vector<size_t>> NHWCOpMap = {
  {ops::kNameAdam, {10}},
  {ops::kNameApplyMomentum, {4}},
  {ops::kNameAvgPoolFusion, {1}},
  {ops::kNameAvgPoolGrad, {}},
  {ops::kNameBatchNorm, {1}},
  {ops::kNameBatchNormGrad, {1, 2}},
  {ops::kNameBatchToSpace, {1}},
  {ops::kNameBiasAdd, {1}},
  {ops::kNameBiasAddGrad, {1}},
  {ops::kNameConv2DBackpropInputFusion, {1}},
  {ops::kNameConv2DBackpropFilterFusion, {1, 2}},
  {ops::kNameConv2DFusion, {1}},
  {ops::kNameConv2dTransposeFusion, {1}},
  {ops::kNameDepthToSpace, {1}},
  {ops::kNameFusedBatchNorm, {1}},
  {ops::kNameInstanceNorm, {1}},
  {ops::kNameLRN, {1}},
  {ops::kNameMaxPoolFusion, {1}},
  {ops::kNameMaxPoolGrad, {}},
  {ops::kNamePReLUFusion, {1}},
  {ops::kNameResize, {1}},
  {ops::kNameResizeGrad, {}},
  {ops::kNameROIPooling, {1}},
  {ops::kNameSGD, {2}},
  {ops::kNameSpaceToBatch, {1}},
  {ops::kNameSpaceToBatchND, {1}},
  {ops::kNameSpaceToDepth, {1}}};

static const std::unordered_map<std::string, std::vector<size_t>> NCHWOpMap = {};

static const std::unordered_map<std::string, std::vector<size_t>> ToNCHWOpMap = {
  {ops::kNameAdam, {10}},
  {ops::kNameApplyMomentum, {4}},
  {ops::kNameAvgPoolFusion, {1}},
  {ops::kNameAvgPoolGrad, {}},
  {ops::kNameBatchNorm, {1}},
  {ops::kNameBatchNormGrad, {1, 2}},
  {ops::kNameBatchToSpace, {1}},
  {ops::kNameBiasAdd, {1}},
  {ops::kNameBiasAddGrad, {1}},
  {ops::kNameConv2DBackpropInputFusion, {1}},
  {ops::kNameConv2DBackpropFilterFusion, {1, 2}},
  {ops::kNameConv2DFusion, {1}},
  {ops::kNameConv2dTransposeFusion, {1}},
  {ops::kNameDepthToSpace, {1}},
  {ops::kNameFusedBatchNorm, {1}},
  {ops::kNameInstanceNorm, {1}},
  {ops::kNameLRN, {1}},
  {ops::kNameMaxPoolFusion, {1}},
  {ops::kNameMaxPoolGrad, {}},
  {ops::kNamePReLUFusion, {1}},
  {ops::kNameResize, {1}},
  {ops::kNameResizeGrad, {}},
  {ops::kNameROIPooling, {1}},
  {ops::kNameSGD, {2}},
  {ops::kNameSpaceToBatch, {1}},
  {ops::kNameSpaceToBatchND, {1}},
  {ops::kNameSpaceToDepth, {1}}};

// a certain op whose input's format is not fixed, bool value determines whether the op has axis attribute or not.
static const std::unordered_map<std::string, bool> DynamicFormatOpList = {{ops::kNameAddN, false},
                                                                          {ops::kNameCrop, true},
                                                                          {ops::kNameSplit, true},
                                                                          {ops::kNameConcat, true},
                                                                          {ops::kNameEltwise, false},
                                                                          {ops::kNameMaximum, false},
                                                                          {ops::kNameAddFusion, false},
                                                                          {ops::kNameDivFusion, false},
                                                                          {ops::kNameMulFusion, false},
                                                                          {ops::kNamePadFusion, false},
                                                                          {ops::kNamePowFusion, false},
                                                                          {ops::kNameActivation, false},
                                                                          {ops::kNameSliceFusion, true},
                                                                          {ops::kNameStridedSlice, true},
                                                                          {ops::kNameActivationGrad, false},
                                                                          {ops::kNameQuantDTypeCast, false},
                                                                          {ops::kNameCast, false},
                                                                          {ops::kNameSubFusion, false},
                                                                          {ops::kNameErf, false}};

const std::unordered_map<std::string, std::vector<size_t>> &GetNHWCOpMap() { return NHWCOpMap; }
const std::unordered_map<std::string, std::vector<size_t>> &GetNCHWOpMap() { return NCHWOpMap; }
const std::unordered_map<std::string, std::vector<size_t>> &GetToNCHWOpMap() { return ToNCHWOpMap; }
bool IsDynamicFormatOp(const std::string &op_type) {
  return DynamicFormatOpList.find(op_type) != DynamicFormatOpList.end();
}
bool IsDynamicFormatOpWithAxis(const std::string &op_type) {
  auto iter = DynamicFormatOpList.find(op_type);
  return iter != DynamicFormatOpList.end() && iter->second;
}

STATUS GetCastDstDataType(const CNodePtr &cnode, int *perm) {
  MS_CHECK_TRUE_RET(cnode != nullptr, lite::RET_NULL_PTR);
  MS_CHECK_TRUE_RET(perm != nullptr, lite::RET_NULL_PTR);
  if (cnode->size() != kInputSizeThree) {
    MS_LOG(ERROR) << "cast op input size must be three.";
    return lite::RET_ERROR;
  }
  if (utils::isa<CNodePtr>(cnode->input(kInputIndexTwo))) {
    return lite::RET_OK;
  }
  lite::DataInfo data_info;
  int status;
  if (utils::isa<ParameterPtr>(cnode->input(kInputIndexTwo))) {
    status = lite::FetchDataFromParameterNode(cnode, kInputIndexTwo, converter::kFmkTypeMs, &data_info, true);
  } else {
    status = lite::FetchDataFromValueNode(cnode, kInputIndexTwo, converter::kFmkTypeMs, false, &data_info, true);
  }
  if (status != lite::RET_OK) {
    MS_LOG(ERROR) << "fetch cast dst data type failed.";
    return lite::RET_ERROR;
  }
  if (data_info.data_type_ != kNumberTypeInt && data_info.data_type_ != kNumberTypeInt32) {
    MS_LOG(ERROR) << "cast data type is invalid.";
    return lite::RET_ERROR;
  }
  if (data_info.data_.size() < sizeof(int32_t)) {
    MS_LOG(ERROR) << "Data and datatype of data-info not match.";
    return false;
  }
  *perm = reinterpret_cast<int *>(data_info.data_.data())[0];
  return lite::RET_OK;
}

STATUS GetTransposePerm(const CNodePtr &cnode, std::vector<int> *perm) {
  MS_CHECK_TRUE_RET(cnode != nullptr, lite::RET_NULL_PTR);
  MS_CHECK_TRUE_RET(perm != nullptr, lite::RET_NULL_PTR);
  if (cnode->size() != kInputSizeThree) {
    MS_LOG(ERROR) << "transpose op input size must be three.";
    return lite::RET_ERROR;
  }
  if (utils::isa<CNodePtr>(cnode->input(kInputIndexTwo))) {
    return lite::RET_OK;
  }
  lite::DataInfo data_info;
  int status;
  if (utils::isa<ParameterPtr>(cnode->input(kInputIndexTwo))) {
    status = lite::FetchDataFromParameterNode(cnode, kInputIndexTwo, converter::kFmkTypeMs, &data_info, true);
  } else {
    status = lite::FetchDataFromValueNode(cnode, kInputIndexTwo, converter::kFmkTypeMs, false, &data_info, true);
  }
  if (status != lite::RET_OK) {
    MS_LOG(ERROR) << "fetch transpose perm data failed.";
    return lite::RET_ERROR;
  }
  if ((data_info.data_type_ != kNumberTypeInt && data_info.data_type_ != kNumberTypeInt32) ||
      data_info.shape_.size() != 1) {
    MS_LOG(ERROR) << "transpose perm data is invalid.";
    return lite::RET_ERROR;
  }
  perm->resize(data_info.shape_[0]);
  if (!data_info.data_.empty() &&
      memcpy_s(perm->data(), perm->size() * sizeof(int), data_info.data_.data(), data_info.data_.size()) != EOK) {
    MS_LOG(ERROR) << "memcpy data failed.";
    return lite::RET_ERROR;
  }
  return lite::RET_OK;
}

void RemoveIfMonad(const CNodePtr &cnode) {
  MS_ASSERT(cnode != nullptr);
  std::vector<AnfNodePtr> inputs{cnode->input(0)};
  for (size_t i = 1; i < cnode->size(); ++i) {
    if (utils::isa<ValueNodePtr>(cnode->input(i))) {
      auto value_node = cnode->input(i)->cast<ValueNodePtr>();
      auto value = value_node->value();
      if (value->isa<Monad>()) {
        continue;
      }
    }
    inputs.push_back(cnode->input(i));
  }
  cnode->set_inputs(inputs);
}

bool IsMonadNode(const AnfNodePtr &node) {
  if (node == nullptr) {
    MS_LOG(ERROR) << "input parameter is nullptr.";
    return false;
  }
  if (!utils::isa<ValueNodePtr>(node)) {
    return false;
  }
  auto value_node = node->cast<ValueNodePtr>();
  auto value = value_node->value();
  if (value->isa<Monad>()) {
    return true;
  }
  return false;
}

bool IsSpecialType(const CNodePtr &cnode) {
  return CheckPrimitiveType(cnode, prim::kPrimTupleGetItem) || CheckPrimitiveType(cnode, prim::kPrimDepend) ||
         CheckPrimitiveType(cnode, prim::kPrimMakeTuple) || CheckPrimitiveType(cnode, kPrimMakeTupleV2) ||
         CheckPrimitiveType(cnode, prim::kPrimReturn);
}

int DetermineCertainOutputFormat(const CNodePtr &cnode, int index, Format *format) {
  MS_CHECK_TRUE_MSG(cnode != nullptr && format != nullptr, RET_ERROR, "function's parameter is nullptr.");
  *format = mindspore::NHWC;
  auto prim = GetCNodePrimitive(cnode);
  MS_CHECK_TRUE_MSG(prim != nullptr, RET_ERROR, "get primitive failed");
  auto value_ptr = prim->GetAttr(kOutputsFormat);
  if (value_ptr != nullptr) {
    MS_CHECK_TRUE_MSG(value_ptr->isa<ValueSequeue>(), RET_ERROR, "outputs_format attr should be sequence.");
    auto formats = CastToInt(value_ptr);
    if (index >= 0 && static_cast<size_t>(index) < formats.size()) {
      MS_CHECK_TRUE_MSG(formats[index] >= NCHW && formats[index] <= NCW, RET_ERROR,
                        "format val is out of enum's range.");
      *format = static_cast<Format>(formats[index]);
    }
  }
  return RET_OK;
}

int DetermineCertainVarInputFormat(const CNodePtr &cnode, size_t index, Format *format) {
  MS_CHECK_TRUE_MSG(cnode != nullptr && format != nullptr, RET_ERROR, "function's parameter is nullptr.");
  auto var_input_info = GetRealCertainVarInput(cnode, index);
  if (var_input_info.first == nullptr) {
    MS_LOG(ERROR) << "cannot get the real var input.";
    return RET_ERROR;
  }
  auto real_input_cnode = var_input_info.first;
  auto item_index = var_input_info.second;
  return DetermineCertainOutputFormat(real_input_cnode, item_index, format);
}
}  // namespace opt
}  // namespace mindspore
