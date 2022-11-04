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

#include "mapper/resize_mapper.h"
#include <memory>
#include <utility>
#include <vector>
#include <unordered_map>
#include "common/anf_util.h"
#include "include/registry/converter_context.h"
#include "common/op_enum.h"
#include "ops/resize.h"
#include "op/resize_operator.h"

namespace mindspore {
namespace dpico {
namespace {
const std::unordered_map<CoordinateTransformMode, mapper::CoordinateTransMode> kCoordinateModeMap = {
  {CoordinateTransformMode::ASYMMETRIC, mapper::CoordinateTransMode::ASYMMETRIC},
  {CoordinateTransformMode::ALIGN_CORNERS, mapper::CoordinateTransMode::ALIGN_CORNERS},
  {CoordinateTransformMode::HALF_PIXEL, mapper::CoordinateTransMode::HALF_PIXEL},
  {CoordinateTransformMode::CROP_AND_RESIZE, mapper::CoordinateTransMode::TF_CROP_AND_RESIZE}};
const std::unordered_map<ResizeMethod, mapper::ResizeInterpolationMode> kInterpolationModeMap = {
  {ResizeMethod::LINEAR, mapper::ResizeInterpolationMode::RESIZE_LINEAR},
  {ResizeMethod::NEAREST, mapper::ResizeInterpolationMode::RESIZE_NEAREST},
  {ResizeMethod::CUBIC, mapper::ResizeInterpolationMode::RESIZE_CUBIC}};
const std::unordered_map<mindspore::NearestMode, mapper::NearestMode> kNearestModeMap = {
  {mindspore::NearestMode::ROUND_HALF_DOWN, mapper::NearestMode::ROUND_PREFER_FLOOR},
  {mindspore::NearestMode::ROUND_HALF_UP, mapper::NearestMode::ROUND_PREFER_CEIL},
  {mindspore::NearestMode::CEIL, mapper::NearestMode::NEAREST_CEIL},
  {mindspore::NearestMode::FLOOR, mapper::NearestMode::NEAREST_FLOOR}};

STATUS GetShapeAndFormat(const api::CNodePtr &cnode, const api::PrimitivePtr &prim, std::vector<int64_t> *input_shape,
                         Format *input_format) {
  MS_CHECK_TRUE_MSG(input_shape != nullptr, RET_ERROR, "input_shape is nullptr.");
  MS_CHECK_TRUE_MSG(input_format != nullptr, RET_ERROR, "input_format is nullptr.");
  auto abstract = GetCNodeInputAbstract(cnode, 1);
  if (abstract == nullptr) {
    MS_LOG(ERROR) << "get cnode input abstract failed.";
    return RET_ERROR;
  }
  if (FetchShapeFromAbstract(abstract, input_shape) != RET_OK) {
    MS_LOG(ERROR) << "fetch input shape failed. " << cnode->fullname_with_scope();
    return RET_ERROR;
  }
  if (prim->GetAttr(ops::kFormat) != nullptr) {
    *input_format = static_cast<Format>(api::GetValue<int64_t>(prim->GetAttr(ops::kFormat)));
  } else {
    MS_LOG(ERROR) << ops::kFormat << " attr is needed.";
    return RET_ERROR;
  }
  return RET_OK;
}
STATUS SetResizeDataInfo(const api::CNodePtr &cnode, const api::PrimitivePtr &prim,
                         mapper::ResizeOperator *resize_operator) {
  MS_CHECK_TRUE_MSG(resize_operator != nullptr, RET_ERROR, "resize_operator is nullptr.");
  if (cnode->inputs().size() != dpico::kDims3) {
    MS_LOG(DEBUG) << "only process two inputs. " << cnode->fullname_with_scope();
    return RET_OK;
  }
  auto input_node = cnode->input(kAxis2);
  MS_CHECK_TRUE_MSG(input_node != nullptr, RET_ERROR, "input_node is nullptr.");
  auto param_node = input_node->cast<api::ParameterPtr>();
  if (param_node == nullptr || !param_node->has_default()) {
    MS_LOG(ERROR) << "invalid parameter node. " << cnode->fullname_with_scope();
    return RET_ERROR;
  }
  auto tensor_info = param_node->default_param()->cast<api::TensorPtr>();
  if (tensor_info == nullptr || tensor_info->DataSize() == 0) {
    MS_LOG(ERROR) << "tensor_info is invalid. " << cnode->fullname_with_scope();
    return RET_ERROR;
  }
  std::vector<int64_t> input_shape;
  Format input_format;
  if (GetShapeAndFormat(cnode, prim, &input_shape, &input_format) != RET_OK) {
    MS_LOG(ERROR) << "get input shape and format failed. " << cnode->fullname_with_scope();
    return RET_ERROR;
  }
  if (input_shape.size() != kDims4) {
    MS_LOG(ERROR) << "resize cnode input shape should be 4 dims. " << cnode->fullname_with_scope();
    return RET_ERROR;
  }
  if (input_format != Format::NHWC && input_format != Format::NCHW) {
    MS_LOG(ERROR) << "resize cnode input format is invalid. " << cnode->fullname_with_scope();
    return RET_ERROR;
  }
  auto elem_cnt = tensor_info->DataSize();
  MS_CHECK_TRUE_MSG(static_cast<size_t>(elem_cnt) == kDims2, RET_ERROR,
                    "resize param element size should be 2. " << cnode->fullname_with_scope());
  if (tensor_info->data_type() == kNumberTypeInt32 || tensor_info->data_type() == kNumberTypeInt) {
    std::vector<int> size_vec;
    auto data = reinterpret_cast<int32_t *>(tensor_info->data());
    MS_CHECK_TRUE_MSG(data != nullptr, RET_ERROR, "data is nullptr.");
    if (input_format == Format::NCHW) {
      size_vec = {static_cast<int32_t>(input_shape.at(0)), static_cast<int32_t>(input_shape.at(kNCHW_C)), *data,
                  *(data + 1)};
    } else {
      size_vec = {static_cast<int32_t>(input_shape.at(0)), *data, *(data + 1),
                  static_cast<int32_t>(input_shape.at(kNHWC_C))};
    }
    resize_operator->SetSizeVec(size_vec);
  } else if (tensor_info->data_type() == kNumberTypeFloat32 || tensor_info->data_type() == kNumberTypeFloat) {
    std::vector<float> scale_vec;
    auto data = reinterpret_cast<float *>(tensor_info->data());
    MS_CHECK_TRUE_MSG(data != nullptr, RET_ERROR, "data is nullptr.");
    if (input_format == Format::NCHW) {
      scale_vec = {1.0, 1.0, *data, *(data + 1)};
    } else {
      scale_vec = {1.0, *data, *(data + 1), 1.0};
    }
    resize_operator->SetScaleVec(scale_vec);
  } else {
    MS_LOG(ERROR) << "unsupported param type. " << tensor_info->data_type();
    return RET_ERROR;
  }
  return RET_OK;
}
}  // namespace
STATUS ResizeMapper::Map(const api::CNodePtr &cnode, std::vector<BaseOperatorPtr> *base_operators,
                         const api::PrimitivePtr &prim, const api::CNodePtrList &output_cnodes) {
  if (base_operators == nullptr) {
    MS_LOG(ERROR) << "base_operators is nullptr.";
    return RET_ERROR;
  }
  auto resize_prim = api::utils::cast<api::SharedPtr<ops::Resize>>(prim);
  MS_ASSERT(resize_prim != nullptr);

  if (resize_prim->GetAttr(ops::kFmkType) != nullptr) {
    auto fmk_type = static_cast<converter::FmkType>(api::GetValue<int64_t>(resize_prim->GetAttr(ops::kFmkType)));
    if (fmk_type == converter::kFmkTypeCaffe) {
      MS_CHECK_TRUE_MSG(OpMapperRegistry::GetInstance()->GetOpMapper("Interp") != nullptr, RET_ERROR,
                        "mapper is nullptr.");
      auto status =
        OpMapperRegistry::GetInstance()->GetOpMapper("Interp")->Map(cnode, base_operators, prim, output_cnodes);
      if (status != RET_OK) {
        MS_LOG(ERROR) << cnode->fullname_with_scope() << " map to interp operator failed.";
        return RET_ERROR;
      } else {
        return RET_OK;
      }
    }
  }

  auto resize_operator = std::make_unique<mapper::ResizeOperator>();
  if (resize_operator == nullptr) {
    MS_LOG(ERROR) << "resize_operator is nullptr.";
    return RET_ERROR;
  }

  if (SetCommonAttr(cnode, resize_operator.get(), output_cnodes) != RET_OK) {
    MS_LOG(ERROR) << "set common attr failed. " << cnode->fullname_with_scope();
    return RET_ERROR;
  }

  resize_operator->SetOpType(mapper::OpType::RESIZE);
  if (prim->GetAttr(ops::kCubicCoeff) != nullptr) {
    resize_operator->SetCubicCoeff(resize_prim->get_cubic_coeff());
  }
  if (prim->GetAttr(ops::kExcludeOutside) != nullptr) {
    resize_operator->SetExcludeOutsideFlag(static_cast<bool>(resize_prim->get_exclude_outside()));
  }
  if (prim->GetAttr(ops::kExtrapolationValue) != nullptr) {
    resize_operator->SetExtrapolationValue(resize_prim->get_extrapolation_value());
  }
  if (prim->GetAttr(ops::kCoordinateTransformMode) != nullptr) {
    auto coordinate_transform_mode = static_cast<CoordinateTransformMode>(resize_prim->get_coordinate_transform_mode());
    if (kCoordinateModeMap.find(coordinate_transform_mode) == kCoordinateModeMap.end()) {
      MS_LOG(ERROR) << "unsupported coordinate transform mode:"
                    << std::to_string(static_cast<int>(coordinate_transform_mode)) << " "
                    << cnode->fullname_with_scope();
      return RET_ERROR;
    }
    resize_operator->SetCoordinateTransMode(kCoordinateModeMap.at(coordinate_transform_mode));
  }
  if (prim->GetAttr(ops::kMethod) != nullptr) {
    auto interpolation_mode = static_cast<ResizeMethod>(resize_prim->get_method());
    if (kInterpolationModeMap.find(interpolation_mode) == kInterpolationModeMap.end()) {
      MS_LOG(ERROR) << "unsupported interpolation mode:" << std::to_string(static_cast<int>(interpolation_mode)) << " "
                    << cnode->fullname_with_scope();
      return RET_ERROR;
    }
    resize_operator->SetInterpolationMode(kInterpolationModeMap.at(interpolation_mode));
  }
  if (prim->GetAttr(ops::kNearestMode) != nullptr) {
    auto nearest_mode = static_cast<mindspore::NearestMode>(resize_prim->get_nearest_mode());
    if (kNearestModeMap.find(nearest_mode) == kNearestModeMap.end()) {
      MS_LOG(ERROR) << "unsupported nearest mode:" << std::to_string(static_cast<int>(nearest_mode)) << " "
                    << cnode->fullname_with_scope();
      return RET_ERROR;
    }
    resize_operator->SetNearestMode(kNearestModeMap.at(nearest_mode));
  }
  if (SetResizeDataInfo(cnode, prim, resize_operator.get()) != RET_OK) {
    MS_LOG(ERROR) << "set resize data failed.";
    return RET_ERROR;
  }
  base_operators->push_back(std::move(resize_operator));
  return RET_OK;
}
REG_MAPPER(Resize, ResizeMapper)
}  // namespace dpico
}  // namespace mindspore
