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

#include "plugin/device/ascend/kernel/tbe/tbe_kernel_select/tbe_select_utils.h"

#include <set>
#include <string>
#include "base/base.h"
#include "runtime/device/ms_device_shape_transfer.h"
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/utils/utils.h"
#include "plugin/device/ascend/kernel/tbe/tbe_dynamic_shape_util.h"

namespace mindspore::kernel {
namespace {
constexpr size_t kNcdhwShapeSize = 5;

bool CheckValidInputAndHiddenSize(const AnfNodePtr &node) {
  if (node->isa<Parameter>()) {
    auto param = node->cast<ParameterPtr>();
    return param->input_size() > 0 && param->hidden_size() > 0;
  }
  if (node->isa<CNode>()) {
    auto cnode = node->cast<CNodePtr>();
    return common::AnfAlgo::HasNodeAttr(kAttrInputSize, cnode) && common::AnfAlgo::HasNodeAttr(kAttrHiddenSize, cnode);
  }
  return false;
}
}  // namespace

bool HostCheck::CheckValidDeviceShape(const AnfNodePtr &node) {
  size_t real_input_num = common::AnfAlgo::GetInputTensorNum(node);
  for (size_t i = 0; i < real_input_num; i++) {
    auto format = AnfAlgo::GetInputFormat(node, i);
    if (!CheckValidInOutDeviceShape(node, i, false, format)) {
      MS_LOG(WARNING) << "TBE Host check input device shape failed, node:" << node->fullname_with_scope()
                      << ", format:" << format;
      return false;
    }
  }

  size_t real_output_num = common::AnfAlgo::GetOutputTensorNum(node);
  for (size_t i = 0; i < real_output_num; i++) {
    auto format = AnfAlgo::GetOutputFormat(node, i);
    if (!CheckValidInOutDeviceShape(node, i, true, format)) {
      MS_LOG(WARNING) << "TBE Host check output device shape failed, node:" << node->fullname_with_scope()
                      << ", format:" << format;
      return false;
    }
  }
  return true;
}

std::vector<int64_t> HostCheck::GetFinalInferShape(const AnfNodePtr &node, size_t index, bool is_output,
                                                   const std::string &format) {
  auto shape = is_output ? common::AnfAlgo::GetOutputDetailShape(node, index)
                         : common::AnfAlgo::GetPrevNodeOutputDetailShape(node, index);
  std::vector<int64_t> infer_shape;
  if (shape->isa<abstract::Shape>()) {
    auto shape_ptr = shape->cast<abstract::ShapePtr>();
    MS_EXCEPTION_IF_NULL(shape_ptr);
    infer_shape = shape_ptr->shape();
  }
  if (infer_shape.empty()) {
    return infer_shape;
  }

  if (trans::IsNeedPadding(format, infer_shape.size())) {
    auto reshape_type =
      is_output ? AnfAlgo::GetOutputReshapeType(node, index) : AnfAlgo::GetInputReshapeType(node, index);
    infer_shape = trans::PaddingShape(infer_shape, format, reshape_type, node);
  }

  auto temp_shape = infer_shape;
  if (!IsOneOfNoPaddingFormat(format) && format != kOpFormat_FRACTAL_ZN_LSTM && infer_shape.size() < kShape4dDims &&
      !IsOneOf3DFormat(format)) {
    MS_LOG(DEBUG) << "Get Device Shape using a shape size is less than 4 ,should be Padding shape by Default firstly";
    temp_shape = trans::PaddingShapeTo4dDefault(infer_shape, node);
  }
  if (infer_shape.size() != kNcdhwShapeSize && IsOneOf3DFormat(format)) {
    temp_shape = trans::PaddingShapeTo5dDefault(infer_shape, node);
  }
  return temp_shape;
}

bool HostCheck::CheckValidInOutDeviceShape(const AnfNodePtr &node, size_t index, bool is_output,
                                           const std::string &format) {
  auto infer_shape = GetFinalInferShape(node, index, is_output, format);
  if (infer_shape.empty()) {
    return true;
  }

  std::set<std::string> check_4D_format = {kOpFormat_NHWC,       kOpFormat_HWCN,      kOpFormat_FRAC_Z,
                                           kOpFormat_NC1HWC0,    kOpFormat_C1HWNCoC0, kOpFormat_FRACTAL_Z_C04,
                                           kOpFormat_NC1HWC0_C04};
  std::set<std::string> check_5D_format = {kOpFormat_NCDHW, kOpFormat_NDC1HWC0, kOpFormat_FRACTAL_Z_3D};
  if (check_4D_format.find(format) != check_4D_format.end()) {
    return infer_shape.size() == kShape4dDims;
  }
  if (check_5D_format.find(format) != check_5D_format.end()) {
    return infer_shape.size() == kShape5dDims;
  }

  if (format == kOpFormat_FRAC_NZ) {
    return infer_shape.size() >= kShape2dDims ||
           (infer_shape.size() == 1 && (infer_shape[0] == 1 || (infer_shape[0] % SizeToLong(kCubeSize) == 0)));
  }

  if (format == kOpFormat_FRACTAL_ZN_RNN) {
    return infer_shape.size() >= kShape2dDims && CheckValidInputAndHiddenSize(node);
  }

  if (format == kOpFormat_ND_RNN_BIAS) {
    return infer_shape.size() > 0 && CheckValidInputAndHiddenSize(node);
  }
  return true;
}

void GetSupportOriFormat(const CNodePtr &cnode, SupportFormat *support_format) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(support_format);
  auto input_num = common::AnfAlgo::GetInputTensorNum(cnode);
  auto output_num = common::AnfAlgo::GetOutputTensorNum(cnode);
  auto op_name = common::AnfAlgo::GetCNodeName(cnode);
  auto op_info = tbe::TbeDynamicShapeUtil::FindOp(op_name, cnode);
  MS_EXCEPTION_IF_NULL(op_info);
  // Note: in reduce/broadcast case: dynamic/optional input/output not care.
  if (op_info->inputs_ptr().size() != input_num || op_info->outputs_ptr().size() != output_num) {
    MS_LOG(INFO) << "Input/output maybe have optional or dynamic io, input num:" << input_num
                 << ", output_num:" << output_num << ", op_info->inputs size: " << op_info->inputs_ptr().size()
                 << ", op_info->output size: " << op_info->outputs_ptr().size();
  }
  SupportFormatItem input_item(input_num, kOpFormat_DEFAULT);
  (void)support_format->input_format.emplace_back(input_item);
  SupportFormatItem output_item(output_num, kOpFormat_DEFAULT);
  (void)support_format->output_format.emplace_back(output_item);
}

void PadScalarShape(ShapeVector *shape) {
  MS_EXCEPTION_IF_NULL(shape);
  if (shape->empty()) {
    (void)shape->emplace_back(1);
  }
}

void GenerateSupportFormat(const std::string &support_input_format, size_t input_num,
                           const std::string &support_output_format, size_t output_num, SupportFormat *support_format) {
  SupportFormatItem input_item(input_num, support_input_format);
  (void)support_format->input_format.emplace_back(input_item);
  SupportFormatItem output_item(output_num, support_output_format);
  (void)support_format->output_format.emplace_back(output_item);
}

void ConstructSupportDTypes(const std::vector<OpIOInfoPtr> &puts, size_t format_size,
                            std::vector<SupportDTypeItem> *support_dtypes) {
  MS_EXCEPTION_IF_NULL(support_dtypes);
  for (const auto &put : puts) {
    MS_EXCEPTION_IF_NULL(put);
    auto dtypes = put->dtypes();
    SupportDTypeItem support_dtype_item = {};
    for (size_t i = 0; i < format_size; ++i) {
      (void)support_dtype_item.insert(support_dtype_item.cbegin(), dtypes.cbegin(), dtypes.cend());
    }
    (void)support_dtypes->emplace_back(support_dtype_item);
  }
}

void ConstructSupportFormats(size_t put_size, const std::vector<SupportFormatItem> &support_format, size_t type_size,
                             std::vector<SupportFormatItem> *support_formats) {
  MS_EXCEPTION_IF_NULL(support_formats);
  for (size_t i = 0; i < put_size; ++i) {
    SupportFormatItem support_format_item = {};
    for (const auto &formats : support_format) {
      for (size_t j = 0; j < type_size; ++j) {
        (void)support_format_item.emplace_back(formats.at(i));
      }
    }
    (void)support_formats->emplace_back(support_format_item);
  }
}

void GenerateSupportFormatDType(const CNodePtr &cnode, const SupportFormat &support_format,
                                SupportFormatDType *support_format_dtype) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(support_format_dtype);
  auto op_info = tbe::TbeDynamicShapeUtil::FindOp(cnode);
  MS_EXCEPTION_IF_NULL(op_info);
  if (op_info->inputs_ptr().size() != support_format.input_format.at(0).size()) {
    MS_LOG(INFO) << "Op name: " << op_info->op_name() << " has optional input, but in the graph, this input not exist."
                 << "op info input num: " << op_info->inputs_ptr().size()
                 << "graph node input num: " << support_format.input_format.at(0).size();
  }
  auto type_size = op_info->outputs_ptr().at(0)->dtypes().size();
  auto format_size = support_format.input_format.size();
  auto input_size = op_info->inputs_ptr().size();
  auto output_size = op_info->outputs_ptr().size();
  ConstructSupportDTypes(op_info->inputs_ptr(), format_size, &support_format_dtype->input_dtypes);
  ConstructSupportDTypes(op_info->outputs_ptr(), format_size, &support_format_dtype->output_dtypes);
  ConstructSupportFormats(input_size, support_format.input_format, type_size, &support_format_dtype->input_formats);
  ConstructSupportFormats(output_size, support_format.output_format, type_size, &support_format_dtype->output_formats);
  if (support_format_dtype->input_dtypes.size() != op_info->inputs_ptr().size() ||
      support_format_dtype->output_dtypes.size() != op_info->outputs_ptr().size() ||
      support_format_dtype->output_dtypes.at(0).size() != support_format_dtype->output_formats.at(0).size()) {
    MS_LOG(ERROR) << "GenerateSupportFormatDType failed.";
  }
}
}  // namespace mindspore::kernel
