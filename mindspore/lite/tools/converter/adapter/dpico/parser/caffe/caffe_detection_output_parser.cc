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

#include "parser/caffe/caffe_detection_output_parser.h"
#include <memory>
#include <vector>
#include "common/op_attr.h"
#include "parser/detection_output_param_holder.h"
#include "ops/custom.h"

namespace mindspore {
namespace lite {
STATUS SetParamType(mapper::DetectionOutputParam *param, const caffe::DetectionOutputParameter_ParamType &param_type) {
  if (param == nullptr) {
    MS_LOG(ERROR) << "input param is nullptr. ";
    return RET_ERROR;
  }
  switch (param_type) {
    case caffe::DetectionOutputParameter_ParamType_DecBBox:
      param->paramType = mapper::ProposalParamType::PROPOSAL_DECBBOX;
      break;
    case caffe::DetectionOutputParameter_ParamType_Sort:
      param->paramType = mapper::ProposalParamType::PROPOSAL_SORT;
      break;
    case caffe::DetectionOutputParameter_ParamType_Nms:
      param->paramType = mapper::ProposalParamType::PROPOSAL_NMS;
      break;
    case caffe::DetectionOutputParameter_ParamType_FilterBox:
      param->paramType = mapper::ProposalParamType::PROPOSAL_FILTERBOX;
      break;
    default:
      MS_LOG(ERROR) << "Unsupported Param Type: " << param_type;
      return RET_ERROR;
  }
  return RET_OK;
}
STATUS SetCodeType(mapper::DetectionOutputParam *param, const caffe::CodeType &code_type) {
  if (param == nullptr) {
    MS_LOG(ERROR) << "input param is nullptr. ";
    return RET_ERROR;
  }
  if (code_type == caffe::CodeType::CENTER_SIZE) {
    param->codeType = mapper::DecBboxCodeType::DECBBOX_CODE_TYPE_CENTER_SIZE;
  } else {
    MS_LOG(ERROR) << "Unsupported Code type: " << code_type;
    return RET_ERROR;
  }
  return RET_OK;
}
void SetParamAttr(mapper::DetectionOutputParam *param, const caffe::DetectionOutputParameter &iter) {
  if (iter.has_top_k()) {
    param->topK = iter.top_k();
  }
  if (iter.has_background_label_id()) {
    param->backgroundLabelId = iter.background_label_id();
  }
  if (iter.has_multi_class_sorting()) {
    param->multiClassSorting = iter.multi_class_sorting();
  }
  if (iter.has_share_location()) {
    param->shareLocation = iter.share_location();
  }
  if (iter.has_clip_bbox()) {
    param->clipBbox = iter.clip_bbox();
  }
  if (iter.has_calc_mode()) {
    param->calcMode = iter.calc_mode();
  }
  if (iter.has_report_flag()) {
    param->reportFlag = iter.report_flag();
  }
  if (iter.has_top()) {
    param->top = iter.top();
  }
  if (iter.has_share_variance()) {
    param->shareVariance = iter.share_variance();
  }
  if (!iter.variance().empty()) {
    param->varianceVec = std::vector<float>(iter.variance().begin(), iter.variance().end());
  }
  if (!iter.bias().empty()) {
    param->biasVec = std::vector<float>(iter.bias().begin(), iter.bias().end());
  }
}

ops::PrimitiveC *CaffeDetectionOutputParser::Parse(const caffe::LayerParameter &proto,
                                                   const caffe::LayerParameter &weight) {
  auto prim = std::make_unique<ops::Custom>();
  if (prim == nullptr) {
    MS_LOG(ERROR) << "prim is nullptr.";
    return nullptr;
  }
  prim->set_type("DetectionOutput");

  if (proto.has_num_anchors()) {
    prim->AddAttr(dpico::kNumAnchors, MakeValue<uint32_t>(proto.num_anchors()));
  }
  if (proto.has_num_bboxes_per_grid()) {
    prim->AddAttr(dpico::kNumBboxesPerGrid, MakeValue<uint32_t>(proto.num_bboxes_per_grid()));
  }
  if (proto.has_num_coords()) {
    prim->AddAttr(dpico::kNumCoords, MakeValue<uint32_t>(proto.num_coords()));
  }
  if (proto.has_num_classes()) {
    prim->AddAttr(dpico::kNumClasses, MakeValue<uint32_t>(proto.num_classes()));
  }
  if (proto.has_num_grids_height()) {
    prim->AddAttr(dpico::kNumGridsHeight, MakeValue<uint32_t>(proto.num_grids_height()));
  }
  if (proto.has_num_grids_width()) {
    prim->AddAttr(dpico::kNumGridsWidth, MakeValue<uint32_t>(proto.num_grids_width()));
  }

  const auto &detection_output_param = proto.detection_output_param();
  std::vector<DetectionOutputParamHolderPtr> detect_output_param_vec;
  for (const auto &iter : detection_output_param) {
    mapper::DetectionOutputParam param;
    if (SetParamType(&param, iter.param_type()) != RET_OK) {
      MS_LOG(ERROR) << "Set param type failed.";
      return nullptr;
    }
    if (SetCodeType(&param, iter.code_type()) != RET_OK) {
      MS_LOG(ERROR) << "Set code type failed.";
      return nullptr;
    }
    (void)SetParamAttr(&param, iter);
    auto param_hold_ptr = std::make_shared<DetectionOutputParamHolder>(param);
    if (param_hold_ptr == nullptr) {
      MS_LOG(ERROR) << "new DetectionOutputParamHolder failed.";
      return nullptr;
    }
    detect_output_param_vec.push_back(param_hold_ptr);
  }
  prim->AddAttr(dpico::kDetectionOutputParam, MakeValue(detect_output_param_vec));
  return prim.release();
}

CaffeNodeRegistrar g_caffeDetectionOutputParser("DetectionOutput", new CaffeDetectionOutputParser());
}  // namespace lite
}  // namespace mindspore
