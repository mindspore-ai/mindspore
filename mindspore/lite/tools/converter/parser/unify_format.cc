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

#include "tools/converter/parser/unify_format.h"
#include <map>

namespace mindspore {
namespace lite {
namespace {
constexpr int kInputChannal = 3;
STATUS DecideMINDIRConvWeightSrcFormat(const CNodePtr &cnode, schema::QuantType quant_type,
                                       schema::Format *src_format) {
  MS_ASSERT(cnode != nullptr && src_format != nullptr);
  auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
  if (prim == nullptr) {
    MS_LOG(ERROR) << "Invalid anfnode, which don't have primitive.";
    return lite::RET_ERROR;
  }
  int64_t format = prim->GetAttr(ops::kFormat) != nullptr ? GetValue<int64_t>(prim->GetAttr(ops::kFormat)) : 0;
  if (format == schema::Format_NHWC) {
    *src_format = schema::Format_KHWC;
  } else if (format == schema::Format_NCHW) {
    *src_format = schema::Format_KCHW;
  } else {
    MS_LOG(ERROR) << "cnode format is invalid.";
    return RET_ERROR;
  }
  return RET_OK;
}

STATUS DecideTFConvWeightSrcFormat(const CNodePtr &cnode, schema::QuantType quant_type, schema::Format *src_format) {
  MS_ASSERT(cnode != nullptr && src_format != nullptr);
  auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
  if (prim == nullptr) {
    MS_LOG(ERROR) << "Invalid anfnode, which don't have primitive.";
    return lite::RET_ERROR;
  }
  bool is_depth_wise = prim->GetAttr(ops::kIsDepthWise) != nullptr && GetValue<bool>(prim->GetAttr(ops::kIsDepthWise));
  switch (quant_type) {
    case schema::QuantType_AwareTraining:
    case schema::QuantType_PostTraining:
    case schema::QuantType_WeightQuant:
    case schema::QuantType_QUANT_NONE: {
      if (opt::CheckPrimitiveType(cnode, prim::kPrimConv2DFusion)) {
        if (!is_depth_wise) {
          *src_format = schema::Format_HWCK;
        } else {
          *src_format = schema::Format_HWKC;
        }
      } else if (opt::CheckPrimitiveType(cnode, prim::kPrimConv2dTransposeFusion) && !is_depth_wise) {
        *src_format = schema::Format::Format_HWCK;
      } else {
        MS_LOG(ERROR) << "depthwise-conv2dTranspose need to check.";
        return RET_ERROR;
      }
    } break;
    default: {
      MS_LOG(ERROR) << "Unsupported op: " << cnode->fullname_with_scope();
      return lite::RET_ERROR;
    }
  }
  return RET_OK;
}

STATUS DecideTFLITEConvWeightSrcFormat(const CNodePtr &cnode, schema::QuantType quant_type,
                                       schema::Format *src_format) {
  MS_ASSERT(cnode != nullptr && src_format != nullptr);
  auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
  if (prim == nullptr) {
    MS_LOG(ERROR) << "Invalid anfnode, which don't have primitive.";
    return lite::RET_ERROR;
  }
  bool is_depth_wise = prim->GetAttr(ops::kIsDepthWise) != nullptr && GetValue<bool>(prim->GetAttr(ops::kIsDepthWise));
  switch (quant_type) {
    case schema::QuantType_AwareTraining:
    case schema::QuantType_PostTraining:
    case schema::QuantType_WeightQuant:
    case schema::QuantType_QUANT_NONE: {
      if (opt::CheckPrimitiveType(cnode, prim::kPrimConv2DFusion)) {
        if (!is_depth_wise) {
          *src_format = schema::Format_KHWC;
        } else {
          *src_format = schema::Format_CHWK;
        }
      } else if (opt::CheckPrimitiveType(cnode, prim::kPrimConv2dTransposeFusion) && !is_depth_wise) {
        *src_format = schema::Format_CHWK;
      } else {
        MS_LOG(ERROR) << "cannot decide weight format, current situation need to check.";
        return RET_NOT_SUPPORT;
      }
    } break;
    default: {
      MS_LOG(ERROR) << "Unsupported quantType: " << EnumNameQuantType(quant_type)
                    << ", node: " << cnode->fullname_with_scope();
      return RET_ERROR;
    }
  }
  return RET_OK;
}

STATUS DecideCAFFEConvWeightSrcFormat(const CNodePtr &cnode, schema::QuantType quant_type, schema::Format *src_format) {
  MS_ASSERT(cnode != nullptr && src_format != nullptr);
  *src_format = schema::Format_KCHW;
  return RET_OK;
}

STATUS DecideONNXConvWeightSrcFormat(const CNodePtr &cnode, schema::QuantType quant_type, schema::Format *src_format) {
  MS_ASSERT(cnode != nullptr && src_format != nullptr);
  auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
  if (prim == nullptr) {
    MS_LOG(ERROR) << "Invalid anfnode, which don't have primitive.";
    return lite::RET_ERROR;
  }
  bool is_depth_wise = prim->GetAttr(ops::kIsDepthWise) != nullptr && GetValue<bool>(prim->GetAttr(ops::kIsDepthWise));
  int64_t format = prim->GetAttr(ops::kFormat) != nullptr ? GetValue<int64_t>(prim->GetAttr(ops::kFormat)) : 0;
  switch (quant_type) {
    case schema::QuantType_AwareTraining: {
      if (opt::CheckPrimitiveType(cnode, prim::kPrimConv2DFusion)) {
        if (!is_depth_wise) {
          *src_format = schema::Format_KHWC;
        } else {
          *src_format = schema::Format_CHWK;
        }
      } else if (opt::CheckPrimitiveType(cnode, prim::kPrimConv2dTransposeFusion) && !is_depth_wise) {
        *src_format = schema::Format_KCHW;
      } else {
        MS_LOG(ERROR) << "Unsupported op: " << cnode->fullname_with_scope();
        return lite::RET_ERROR;
      }
    } break;
    case schema::QuantType_PostTraining:
    case schema::QuantType_WeightQuant:
    case schema::QuantType_QUANT_NONE: {
      if (opt::CheckPrimitiveType(cnode, prim::kPrimConv2DFusion) ||
          opt::CheckPrimitiveType(cnode, prim::kPrimConv2dTransposeFusion)) {
        if (format == schema::Format_NHWC) {
          *src_format = schema::Format_KHWC;
        } else if (format == schema::Format_NCHW) {
          *src_format = schema::Format_KCHW;
        } else {
          MS_LOG(ERROR) << "format is invalid, format is " << format;
          return RET_ERROR;
        }
      } else {
        MS_LOG(ERROR) << "d an unsupported op type, which need to check. the type is " << prim->name();
        return RET_NOT_SUPPORT;
      }
    } break;
    default: {
      MS_LOG(ERROR) << "Unsupported quantType: " << EnumNameQuantType(quant_type)
                    << ", node: " << cnode->fullname_with_scope();
      return lite::RET_ERROR;
    }
  }
  return RET_OK;
}
}  // namespace

STATUS UnifyFormatToNHWC::GetTransNodeFormatType(const CNodePtr &cnode, opt::TransTypePair *trans_info) {
  MS_ASSERT(cnode != nullptr && trans_info != nullptr);
  auto prim_node = cnode->input(0);
  auto prim = GetValueNode<PrimitivePtr>(prim_node);
  if (prim == nullptr) {
    return RET_OK;
  }
  auto &specify_nhwc_op_map = opt::GetNHWCOpMap();
  auto &specify_nchw_op_map = opt::GetNCHWOpMap();
  if (fmk_type_ == converter::kFmkTypeTflite) {
    if (specify_nchw_op_map.find(prim->name()) == specify_nchw_op_map.end()) {
      return lite::RET_OK;
    }
    trans_info->pre_ = opt::kNHWC2NCHW;
    trans_info->post_ = opt::kNCHW2NHWC;
  } else if (fmk_type_ == converter::kFmkTypeTf) {
    if (specify_nhwc_op_map.find(prim->name()) != specify_nhwc_op_map.end() && opt::GetFormat(cnode) == NCHW) {
      trans_info->pre_ = opt::kNCHW2NHWC;
      trans_info->post_ = opt::kNHWC2NCHW;
    }
    if (specify_nchw_op_map.find(prim->name()) != specify_nchw_op_map.end()) {
      trans_info->pre_ = opt::kNHWC2NCHW;
      trans_info->post_ = opt::kNCHW2NHWC;
    }
  } else {
    if (specify_nhwc_op_map.find(prim->name()) != specify_nhwc_op_map.end()) {
      if (fmk_type_ == converter::kFmkTypeOnnx && prim->GetAttr(ops::kFormat) != nullptr &&
          GetValue<int64_t>(prim->GetAttr(ops::kFormat)) == NHWC) {
        return lite::RET_OK;
      }
      trans_info->pre_ = opt::kNCHW2NHWC;
      trans_info->post_ = opt::kNHWC2NCHW;
    }
  }
  return lite::RET_OK;
}

void UnifyFormatToNHWC::SetSensitiveOps() {
  auto &sensitive_nhwc_ops = opt::GetNHWCOpMap();
  auto &sensitive_nchw_ops = opt::GetNCHWOpMap();
  sensitive_ops_.insert(sensitive_nhwc_ops.begin(), sensitive_nhwc_ops.end());
  sensitive_ops_.insert(sensitive_nchw_ops.begin(), sensitive_nchw_ops.end());
}

bool UnifyFormatToNHWC::DecideWhetherHandleGraphInput(const FuncGraphPtr &func_graph, const ShapeVector &shape) {
  MS_ASSERT(func_graph != nullptr);
  if (fmk_type_ == converter::kFmkTypeTf || fmk_type_ == converter::kFmkTypeTflite) {
    return false;
  }
  if (func_graph->get_inputs().size() == 1 && fmk_type_ == converter::kFmkTypeOnnx &&
      shape[opt::kInputIndexThree] == kInputChannal && shape[1] == -1) {
    return false;
  }
  return true;
}

bool UnifyFormatToNHWC::DecideWhetherInferShapeForNewNode() { return false; }

STATUS UnifyFormatToNHWC::DecideConvWeightSrcAndDstFormat(const CNodePtr &cnode, schema::Format *src_format,
                                                          schema::Format *dst_format) {
  MS_ASSERT(cnode != nullptr && src_format != nullptr && dst_format != nullptr);
  *dst_format = schema::Format_KHWC;
  std::map<converter::FmkType, std::function<int(const CNodePtr &, schema::QuantType, schema::Format *)>>
    decide_functions = {{converter::kFmkTypeMs, DecideMINDIRConvWeightSrcFormat},
                        {converter::kFmkTypeTf, DecideTFConvWeightSrcFormat},
                        {converter::kFmkTypeTflite, DecideTFLITEConvWeightSrcFormat},
                        {converter::kFmkTypeCaffe, DecideCAFFEConvWeightSrcFormat},
                        {converter::kFmkTypeOnnx, DecideONNXConvWeightSrcFormat}};
  auto iter = decide_functions.find(fmk_type_);
  if (iter == decide_functions.end()) {
    MS_LOG(ERROR) << "current fmk don't support, please check.";
    return RET_NOT_SUPPORT;
  }
  auto decide_func = iter->second;
  MS_ASSERT(decide_func != nullptr);
  if (decide_func(cnode, quant_type_, src_format) != RET_OK) {
    MS_LOG(ERROR) << "run decide function failed, cannot decide conv weight format.";
    return RET_ERROR;
  }
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
