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

#include "parser/detection_output_param_helper.h"
#include <memory>
#include <vector>
#include "mindapi/base/logging.h"
#include "common/op_attr.h"
#include "include/errorcode.h"
#include "ops/custom.h"
#include "./pico_caffe.pb.h"
#include "op/detection_output_operator.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
namespace mindspore {
namespace dpico {
namespace {
int GetProposalParamType(mapper::ProposalParamType *proposal_param_type,
                         const caffe::DetectionOutputParameter_ParamType &param_type) {
  if (proposal_param_type == nullptr) {
    MS_LOG(ERROR) << "input proposal_param_type is nullptr. ";
    return RET_ERROR;
  }
  switch (param_type) {
    case caffe::DetectionOutputParameter_ParamType_DecBBox:
      *proposal_param_type = mapper::ProposalParamType::PROPOSAL_DECBBOX;
      break;
    case caffe::DetectionOutputParameter_ParamType_Sort:
      *proposal_param_type = mapper::ProposalParamType::PROPOSAL_SORT;
      break;
    case caffe::DetectionOutputParameter_ParamType_Nms:
      *proposal_param_type = mapper::ProposalParamType::PROPOSAL_NMS;
      break;
    case caffe::DetectionOutputParameter_ParamType_FilterBox:
      *proposal_param_type = mapper::ProposalParamType::PROPOSAL_FILTERBOX;
      break;
    default:
      MS_LOG(ERROR) << "Unsupported Param Type: " << param_type;
      return RET_ERROR;
  }
  return RET_OK;
}
int GetCodeType(mapper::DecBboxCodeType *dec_bbox_code_type, const caffe::CodeType &code_type) {
  if (dec_bbox_code_type == nullptr) {
    MS_LOG(ERROR) << "input dec_bbox_code_type is nullptr. ";
    return RET_ERROR;
  }
  if (code_type == caffe::CodeType::CENTER_SIZE) {
    *dec_bbox_code_type = mapper::DecBboxCodeType::DECBBOX_CODE_TYPE_CENTER_SIZE;
  } else {
    MS_LOG(ERROR) << "Unsupported Code type: " << code_type;
    return RET_ERROR;
  }
  return RET_OK;
}

int SetAttrsByParam(const std::shared_ptr<ops::Custom> &custom_prim,
                    const caffe::DetectionOutputParameter &detection_param, int index) {
  if (custom_prim == nullptr) {
    MS_LOG(ERROR) << "custom_prim is nullptr.";
    return RET_ERROR;
  }
  if (detection_param.has_top_k()) {
    (void)custom_prim->AddAttr(kDetectionTopK + std::to_string(index), api::MakeValue(detection_param.top_k()));
  }
  if (detection_param.has_background_label_id()) {
    (void)custom_prim->AddAttr(kDetectionBackgroundLabelId + std::to_string(index),
                               api::MakeValue(detection_param.background_label_id()));
  }
  if (detection_param.has_multi_class_sorting()) {
    (void)custom_prim->AddAttr(kDetectionMultiClassSorting + std::to_string(index),
                               api::MakeValue<bool>(detection_param.multi_class_sorting()));
  }
  if (detection_param.has_share_location()) {
    (void)custom_prim->AddAttr(kDetectionShareLocation + std::to_string(index),
                               api::MakeValue<bool>(detection_param.share_location()));
  }
  if (detection_param.has_clip_bbox()) {
    (void)custom_prim->AddAttr(kDetectionClipBbox + std::to_string(index),
                               api::MakeValue<bool>(detection_param.clip_bbox()));
  }
  if (detection_param.has_calc_mode()) {
    (void)custom_prim->AddAttr(kDetectionCalcMode + std::to_string(index), api::MakeValue(detection_param.calc_mode()));
  }
  if (detection_param.has_report_flag()) {
    (void)custom_prim->AddAttr(kDetectionReportFlag + std::to_string(index),
                               api::MakeValue<bool>(detection_param.report_flag()));
  }
  if (detection_param.has_top()) {
    (void)custom_prim->AddAttr(kDetectionTop + std::to_string(index),
                               api::MakeValue<std::string>(detection_param.top()));
  }
  if (detection_param.has_share_variance()) {
    (void)custom_prim->AddAttr(kDetectionShareVariance + std::to_string(index),
                               api::MakeValue(detection_param.share_variance()));
  }
  if (!detection_param.variance().empty()) {
    (void)custom_prim->AddAttr(kDetectionVarianceVec + std::to_string(index),
                               api::MakeValue<std::vector<float>>(std::vector<float>(
                                 detection_param.variance().begin(), detection_param.variance().end())));
  }
  if (!detection_param.bias().empty()) {
    (void)custom_prim->AddAttr(kDetectionBiasVec + std::to_string(index),
                               api::MakeValue<std::vector<float>>(
                                 std::vector<float>(detection_param.bias().begin(), detection_param.bias().end())));
  }

  if (detection_param.has_param_type()) {
    mapper::ProposalParamType proposal_param_type;
    if (GetProposalParamType(&proposal_param_type, detection_param.param_type()) != RET_OK) {
      MS_LOG(ERROR) << "get detection proposal param type failed.";
      return RET_ERROR;
    }
    (void)custom_prim->AddAttr(kDetectionProposalParamType + std::to_string(index),
                               api::MakeValue(static_cast<int64_t>(proposal_param_type)));
  }

  if (detection_param.has_code_type()) {
    mapper::DecBboxCodeType dec_bbox_code_type;
    if (GetCodeType(&dec_bbox_code_type, detection_param.code_type()) != RET_OK) {
      MS_LOG(ERROR) << "get detection code type failed.";
      return RET_ERROR;
    }
    (void)custom_prim->AddAttr(kDetectionCodeType + std::to_string(index),
                               api::MakeValue(static_cast<int64_t>(dec_bbox_code_type)));
  }
  return RET_OK;
}
mapper::DetectionOutputParam GetParamFromAttrs(const api::SharedPtr<ops::Custom> &custom_prim, int index) {
  mapper::DetectionOutputParam detection_output_param;
  if (custom_prim == nullptr) {
    MS_LOG(ERROR) << "custom_prim is nullptr.";
    return detection_output_param;
  }
  auto top_k_ptr = custom_prim->GetAttr(kDetectionTopK + std::to_string(index));
  if (top_k_ptr != nullptr) {
    detection_output_param.topK = static_cast<uint32_t>(api::GetValue<int64_t>(top_k_ptr));
  }
  auto background_label_id_ptr = custom_prim->GetAttr(kDetectionBackgroundLabelId + std::to_string(index));
  if (background_label_id_ptr != nullptr) {
    detection_output_param.backgroundLabelId = static_cast<uint32_t>(api::GetValue<int64_t>(background_label_id_ptr));
  }
  auto multi_class_sorting_ptr = custom_prim->GetAttr(kDetectionMultiClassSorting + std::to_string(index));
  if (multi_class_sorting_ptr != nullptr) {
    detection_output_param.multiClassSorting = api::GetValue<bool>(multi_class_sorting_ptr);
  }
  auto share_location_ptr = custom_prim->GetAttr(kDetectionShareLocation + std::to_string(index));
  if (share_location_ptr != nullptr) {
    detection_output_param.shareLocation = api::GetValue<bool>(share_location_ptr);
  }
  auto clip_bbox_ptr = custom_prim->GetAttr(kDetectionClipBbox + std::to_string(index));
  if (clip_bbox_ptr != nullptr) {
    detection_output_param.clipBbox = api::GetValue<bool>(clip_bbox_ptr);
  }
  auto calc_mode_ptr = custom_prim->GetAttr(kDetectionCalcMode + std::to_string(index));
  if (calc_mode_ptr != nullptr) {
    detection_output_param.calcMode = static_cast<uint32_t>(api::GetValue<int64_t>(calc_mode_ptr));
  }
  auto report_flag_ptr = custom_prim->GetAttr(kDetectionReportFlag + std::to_string(index));
  if (report_flag_ptr != nullptr) {
    detection_output_param.reportFlag = api::GetValue<bool>(report_flag_ptr);
  }
  auto top_ptr = custom_prim->GetAttr(kDetectionTop + std::to_string(index));
  if (top_ptr != nullptr) {
    detection_output_param.top = api::GetValue<std::string>(top_ptr);
  }
  auto share_variance_ptr = custom_prim->GetAttr(kDetectionShareVariance + std::to_string(index));
  if (share_variance_ptr != nullptr) {
    detection_output_param.shareVariance = api::GetValue<bool>(share_variance_ptr);
  }
  auto variance_vec_ptr = custom_prim->GetAttr(kDetectionVarianceVec + std::to_string(index));
  if (variance_vec_ptr != nullptr) {
    detection_output_param.varianceVec = api::GetValue<std::vector<float>>(variance_vec_ptr);
  }
  auto bias_vec_ptr = custom_prim->GetAttr(kDetectionBiasVec + std::to_string(index));
  if (bias_vec_ptr != nullptr) {
    detection_output_param.biasVec = api::GetValue<std::vector<float>>(bias_vec_ptr);
  }
  auto proposal_param_type_ptr = custom_prim->GetAttr(kDetectionProposalParamType + std::to_string(index));
  if (proposal_param_type_ptr != nullptr) {
    detection_output_param.paramType =
      static_cast<mapper::ProposalParamType>(api::GetValue<int64_t>(proposal_param_type_ptr));
  }
  auto code_type_ptr = custom_prim->GetAttr(kDetectionCodeType + std::to_string(index));
  if (code_type_ptr != nullptr) {
    detection_output_param.codeType = static_cast<mapper::DecBboxCodeType>(api::GetValue<int64_t>(code_type_ptr));
  }
  return detection_output_param;
}
}  // namespace
int SetAttrsByDetectionOutputParam(const std::shared_ptr<ops::Custom> &custom_prim,
                                   const caffe::LayerParameter &proto) {
  int detection_output_param_size = proto.detection_output_param_size();
  if (custom_prim == nullptr) {
    MS_LOG(ERROR) << "custom_prim is nullptr.";
    return RET_ERROR;
  }
  (void)custom_prim->AddAttr(kDetectionOutputParamSize, api::MakeValue(detection_output_param_size));
  if (detection_output_param_size == 0) {
    MS_LOG(INFO) << "no detection param found";
    return RET_OK;
  }
  for (int i = 0; i < proto.detection_output_param_size(); i++) {
    const auto &detect_param = proto.detection_output_param(i);
    if (SetAttrsByParam(custom_prim, detect_param, i) != RET_OK) {
      MS_LOG(ERROR) << "set prim attrs from detection param failed.";
      return RET_ERROR;
    }
  }
  return RET_OK;
}
int SetAttrsByDecBboxParam(const std::shared_ptr<ops::Custom> &custom_prim, const caffe::LayerParameter &proto) {
  if (custom_prim == nullptr) {
    MS_LOG(ERROR) << "custom_prim is nullptr.";
    return RET_ERROR;
  }
  if (!proto.has_decbbox_param()) {
    MS_LOG(INFO) << "no decbbox param found";
    return RET_OK;
  }
  (void)custom_prim->AddAttr(kDetectionOutputParamSize, api::MakeValue(1));
  const auto &detect_param = proto.decbbox_param();
  if (SetAttrsByParam(custom_prim, detect_param, 0) != RET_OK) {
    MS_LOG(ERROR) << "set prim attrs from decbbox param failed.";
    return RET_ERROR;
  }
  return RET_OK;
}
int GetDetectionOutputParamFromAttrs(std::vector<mapper::DetectionOutputParam> *detection_params,
                                     const api::SharedPtr<ops::Custom> &custom_prim) {
  if (detection_params == nullptr) {
    MS_LOG(ERROR) << "input detection_params is nullptr.";
    return RET_ERROR;
  }
  int detection_output_param_size = 0;
  auto detection_output_param_size_attr = custom_prim->GetAttr(kDetectionOutputParamSize);
  if (detection_output_param_size_attr != nullptr) {
    detection_output_param_size = static_cast<int>(api::GetValue<int64_t>(detection_output_param_size_attr));
  }
  if (detection_output_param_size == 0) {
    MS_LOG(INFO) << "no detection param attr found.";
    return RET_OK;
  }
  for (int i = 0; i < detection_output_param_size; i++) {
    (void)detection_params->emplace_back(GetParamFromAttrs(custom_prim, i));
  }
  return RET_OK;
}
}  // namespace dpico
}  // namespace mindspore
