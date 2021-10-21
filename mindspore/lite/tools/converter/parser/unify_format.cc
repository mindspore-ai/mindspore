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
#include <memory>
#include <vector>
#include "tools/common/tensor_util.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace lite {
namespace {
constexpr int kInputChannal = 3;
constexpr int kNumGatherIndiceSize_4 = 4;
constexpr int kNumGatherIndiceSize_2 = 2;
constexpr int kNumResizeInputShape = 2;
constexpr int kNumInputSize = 2;
constexpr int kNumIndex_0 = 0;
constexpr int kNumIndex_1 = 1;
constexpr int kNumIndex_2 = 2;
constexpr int kNumIndex_3 = 3;
STATUS DecideMINDIRConvWeightSrcFormat(const CNodePtr &cnode, schema::Format *src_format) {
  MS_ASSERT(cnode != nullptr && src_format != nullptr);
  auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
  if (prim == nullptr) {
    MS_LOG(ERROR) << "Invalid anfnode, which don't have primitive.";
    return lite::RET_ERROR;
  }
  int64_t format =
    prim->GetAttr(ops::kOriginalFormat) != nullptr ? GetValue<int64_t>(prim->GetAttr(ops::kOriginalFormat)) : 0;
  if (format == schema::Format_NHWC) {
    *src_format = schema::Format_KHWC;
  } else if (format == schema::Format_NCHW) {
    *src_format = schema::Format_KCHW;
  } else {
    MS_LOG(ERROR) << "cnode format is invalid. " << cnode->fullname_with_scope();
    return RET_ERROR;
  }
  return RET_OK;
}

STATUS DecideTFConvWeightSrcFormat(const CNodePtr &cnode, schema::Format *src_format) {
  MS_ASSERT(cnode != nullptr && src_format != nullptr);
  auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
  if (prim == nullptr) {
    MS_LOG(ERROR) << "Invalid anfnode, which don't have primitive.";
    return lite::RET_ERROR;
  }
  bool is_depth_wise = prim->GetAttr(ops::kIsDepthWise) != nullptr && GetValue<bool>(prim->GetAttr(ops::kIsDepthWise));
  if (opt::CheckPrimitiveType(cnode, prim::kPrimConv2DFusion)) {
    if (!is_depth_wise) {
      *src_format = schema::Format_HWCK;
    } else {
      *src_format = schema::Format_HWKC;
    }
  } else if (opt::CheckPrimitiveType(cnode, prim::kPrimConv2dTransposeFusion) && !is_depth_wise) {
    *src_format = schema::Format::Format_HWCK;
  } else {
    MS_LOG(ERROR) << "depthwise-conv2dTranspose need to check. " << cnode->fullname_with_scope();
    return RET_ERROR;
  }
  return RET_OK;
}

STATUS DecideTFLITEConvWeightSrcFormat(const CNodePtr &cnode, schema::Format *src_format) {
  MS_ASSERT(cnode != nullptr && src_format != nullptr);
  auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
  if (prim == nullptr) {
    MS_LOG(ERROR) << "Invalid anfnode, which don't have primitive.";
    return lite::RET_ERROR;
  }
  bool is_depth_wise = prim->GetAttr(ops::kIsDepthWise) != nullptr && GetValue<bool>(prim->GetAttr(ops::kIsDepthWise));
  if (opt::CheckPrimitiveType(cnode, prim::kPrimConv2DFusion)) {
    if (!is_depth_wise) {
      *src_format = schema::Format_KHWC;
    } else {
      *src_format = schema::Format_CHWK;
    }
  } else if (opt::CheckPrimitiveType(cnode, prim::kPrimConv2dTransposeFusion) && !is_depth_wise) {
    *src_format = schema::Format_CHWK;
  } else {
    MS_LOG(ERROR) << "cannot decide weight format, current situation need to check. " << cnode->fullname_with_scope();
    return RET_NOT_SUPPORT;
  }
  return RET_OK;
}

STATUS DecideCAFFEConvWeightSrcFormat(const CNodePtr &cnode, schema::Format *src_format) {
  MS_ASSERT(cnode != nullptr && src_format != nullptr);
  *src_format = schema::Format_KCHW;
  return RET_OK;
}

STATUS DecideONNXConvWeightSrcFormat(const CNodePtr &cnode, schema::Format *src_format) {
  MS_ASSERT(cnode != nullptr && src_format != nullptr);
  auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
  if (prim == nullptr) {
    MS_LOG(ERROR) << "Invalid anfnode, which don't have primitive.";
    return lite::RET_ERROR;
  }
  int64_t format =
    prim->GetAttr(ops::kOriginalFormat) != nullptr ? GetValue<int64_t>(prim->GetAttr(ops::kOriginalFormat)) : 0;
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
    MS_LOG(ERROR) << "unknown op, please check.";
    return RET_ERROR;
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
    if (specify_nhwc_op_map.find(prim->name()) != specify_nhwc_op_map.end() &&
        prim->GetAttr(ops::kOriginalFormat) != nullptr &&
        GetValue<int64_t>(prim->GetAttr(ops::kOriginalFormat)) == NCHW) {
      trans_info->pre_ = opt::kNCHW2NHWC;
      trans_info->post_ = opt::kNHWC2NCHW;
    }
    if (specify_nchw_op_map.find(prim->name()) != specify_nchw_op_map.end()) {
      trans_info->pre_ = opt::kNHWC2NCHW;
      trans_info->post_ = opt::kNCHW2NHWC;
    }
  } else {
    if (specify_nhwc_op_map.find(prim->name()) != specify_nhwc_op_map.end()) {
      if (fmk_type_ == converter::kFmkTypeOnnx && prim->GetAttr(ops::kOriginalFormat) != nullptr &&
          GetValue<int64_t>(prim->GetAttr(ops::kOriginalFormat)) == NHWC) {
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

bool UnifyFormatToNHWC::DecideWhetherHandleGraphInput(const FuncGraphPtr &func_graph, const ParameterPtr &input,
                                                      const ShapeVector &shape) {
  MS_ASSERT(func_graph != nullptr);
  MS_ASSERT(input != nullptr);
  if (shape.size() != opt::kInputSizeFour) {
    return false;
  }
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
  std::map<converter::FmkType, std::function<int(const CNodePtr &, schema::Format *)>> decide_functions = {
    {converter::kFmkTypeMs, DecideMINDIRConvWeightSrcFormat},
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
  if (decide_func(cnode, src_format) != RET_OK) {
    MS_LOG(ERROR) << "run decide function failed, cannot decide conv weight format.";
    return RET_ERROR;
  }
  return RET_OK;
}

STATUS UnifyFormatToNHWC::ConvertOnnxResizeForConstShape(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  auto resize_shape_node = cnode->input(kNumResizeInputShape)->cast<ParameterPtr>();
  auto shape_tensor = std::dynamic_pointer_cast<tensor::Tensor>(resize_shape_node->default_param());
  if (shape_tensor == nullptr) {
    MS_LOG(ERROR) << " shape tensor is nullptr.";
    return RET_ERROR;
  }
  MS_CHECK_TRUE_MSG(shape_tensor->data_c() != nullptr, RET_ERROR, "shape_tensor->data_c() is nullptr.");
  auto shape_data = static_cast<float *>(shape_tensor->data_c());
  std::vector<float> new_shape;
  MS_CHECK_TRUE_MSG(!shape_tensor->shape().empty(), RET_NULL_PTR, "out of range.");
  if (shape_tensor->shape().at(0) == kNumGatherIndiceSize_4) {
    new_shape = {shape_data[kNumIndex_0], shape_data[kNumIndex_2], shape_data[kNumIndex_3], shape_data[kNumIndex_1]};
  } else if (shape_tensor->shape().at(0) == kNumGatherIndiceSize_2) {
    return RET_OK;
  } else {
    return RET_ERROR;
  }
  auto new_shape_node = func_graph->add_parameter();
  MS_CHECK_TRUE_MSG(new_shape_node != nullptr, RET_NULL_PTR, "new_shape_node is nullptr.");
  auto tensor_info = CreateTensorInfo(nullptr, 0, shape_tensor->shape(), shape_tensor->data_type());
  if (tensor_info == nullptr) {
    MS_LOG(ERROR) << "create tensor info failed.";
    return RET_ERROR;
  }
  auto new_shape_data = static_cast<float *>(tensor_info->data_c());
  if (new_shape_data == nullptr) {
    MS_LOG(ERROR) << "data is nullptr";
    return RET_ERROR;
  }
  auto status = memcpy_s(new_shape_data, tensor_info->Size(), new_shape.data(), tensor_info->Size());
  if (status != RET_OK) {
    MS_LOG(ERROR) << "init parameter from tensor info failed";
    return RET_ERROR;
  }
  status = InitParameterFromTensorInfo(new_shape_node, tensor_info);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "init parameter from tensor info failed";
    return RET_ERROR;
  }
  manager_->SetEdge(cnode, kNumResizeInputShape, new_shape_node);
  return RET_OK;
}

STATUS UnifyFormatToNHWC::ConvertOnnxResizeForVariableShape(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  MS_CHECK_TRUE_MSG(func_graph != nullptr, RET_ERROR, "func_graph is nullptr.");
  MS_CHECK_TRUE_MSG(cnode != nullptr, RET_ERROR, "cnode is nullptr.");
  auto gather_name = cnode->fullname_with_scope() + "_gather";
  auto gather_input = cnode->input(kNumResizeInputShape);
  MS_CHECK_TRUE_MSG(gather_input != nullptr, RET_ERROR, "gather_input is nullptr.");
  auto abstract = cnode->input(kNumResizeInputShape)->abstract();
  MS_CHECK_TRUE_MSG(abstract != nullptr, RET_ERROR, "abstract is nullptr.");
  std::vector<int> gather_indices = {0, 2, 3, 1};  // NCHW to NHWC
  auto gather_cnode = opt::GenGatherNode(func_graph, gather_input, gather_indices, gather_name);
  if (gather_cnode == nullptr) {
    MS_LOG(ERROR) << "create gather cnode failed.";
    return RET_ERROR;
  }
  ShapeVector indices_shape = {kNumGatherIndiceSize_4};
  auto gather_prim = GetValueNode<PrimitivePtr>(gather_cnode->input(0));
  MS_CHECK_TRUE_MSG(gather_prim != nullptr, RET_NULL_PTR, "gather_prim is nullptr.");
  auto value_ptr = MakeValue<int64_t>(NHWC);
  MS_CHECK_TRUE_MSG(value_ptr != nullptr, RET_NULL_PTR, "value_ptr is nullptr.");
  (void)gather_prim->AddAttr(ops::kFormat, value_ptr);
  gather_cnode->set_abstract(abstract->Clone());
  auto shape_ptr = std::make_shared<abstract::Shape>(indices_shape);
  MS_CHECK_TRUE_MSG(shape_ptr != nullptr, RET_NULL_PTR, "shape_ptr is nullptr.");
  abstract->set_shape(shape_ptr);
  manager_->SetEdge(cnode, kNumIndex_2, gather_cnode);
  auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
  (void)prim->AddAttr(ops::kFormat, MakeValue<int64_t>(NHWC));
  return RET_OK;
}

STATUS UnifyFormatToNHWC::ResizeNodeProcess(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  MS_CHECK_TRUE_MSG(func_graph != nullptr, RET_ERROR, "func_graph is nullptr.");
  MS_CHECK_TRUE_MSG(cnode != nullptr, RET_ERROR, "cnode is nullptr.");
  if (fmk_type_ != converter::kFmkTypeOnnx) {
    return RET_OK;
  }
  if (cnode->inputs().size() > kNumInputSize && utils::isa<ParameterPtr>(cnode->input(kNumResizeInputShape))) {
    auto status = ConvertOnnxResizeForConstShape(func_graph, cnode);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "ConvertOnnxResizeForConstShape failed.";
      return RET_ERROR;
    }
  } else if (cnode->inputs().size() > kNumInputSize && utils::isa<CNodePtr>(cnode->input(kNumResizeInputShape))) {
    auto status = ConvertOnnxResizeForVariableShape(func_graph, cnode);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "ConvertResizeForVariableShape failed.";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

bool UnifyFormatToNHWC::ProcessResizeAndFormat(const FuncGraphPtr &func_graph) {
  MS_ASSERT(func_graph != nullptr);
  manager_->AddFuncGraph(func_graph);
  auto node_list = TopoSort(func_graph->get_return());
  int status;
  for (auto &node : node_list) {
    if (!utils::isa<CNodePtr>(node)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    if (opt::IsSpecialType(cnode)) {
      continue;
    }
    auto value_node = cnode->input(0)->cast<ValueNodePtr>();
    if (value_node == nullptr) {
      if (cnode->input(0)->cast<CNodePtr>() != nullptr) {
        continue;
      }
      MS_LOG(ERROR) << "cnode first input is invalid.";
      return false;
    }
    auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
    if (prim == nullptr) {
      continue;
    }
    if (prim->GetAttr(ops::kFormat) == nullptr && prim->GetAttr(ops::kOriginalFormat) != nullptr) {
      prim->AddAttr(mindspore::ops::kFormat, prim->GetAttr(ops::kOriginalFormat));
    }
    if (opt::CheckPrimitiveType(node, prim::kPrimIf) || opt::CheckPrimitiveType(node, prim::kPrimWhile)) {
      auto sub_func_graph = GetValueNode<FuncGraphPtr>(cnode->input(kNumIndex_1));
      if (sub_func_graph == nullptr) {
        lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
        return false;
      }
      (void)ProcessResizeAndFormat(sub_func_graph);
      sub_func_graph = GetValueNode<FuncGraphPtr>(cnode->input(kNumIndex_2));
      if (sub_func_graph == nullptr) {
        lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
        return false;
      }
      (void)ProcessResizeAndFormat(sub_func_graph);
      continue;
    }
    if (opt::CheckPrimitiveType(node, prim::kPrimResize)) {
      status = ResizeNodeProcess(func_graph, cnode);
      if (status != lite::RET_OK) {
        return false;
      }
    }
  }
  return true;
}

bool UnifyFormatToNHWC::Run(const FuncGraphPtr &func_graph) {
  MS_ASSERT(func_graph != nullptr);
  manager_ = Manage(func_graph, true);
  if (manager_ == nullptr) {
    MS_LOG(ERROR) << "manager is nullptr.";
    return false;
  }
  if (!ProcessResizeAndFormat(func_graph)) {
    MS_LOG(ERROR) << "ProcessResizeAndFormat failed.";
    return false;
  }
  if (!opt::ToFormatBase::Run(func_graph)) {
    MS_LOG(ERROR) << "run ToFormatBase failed.";
    return false;
  }
  return true;
}
}  // namespace lite
}  // namespace mindspore
