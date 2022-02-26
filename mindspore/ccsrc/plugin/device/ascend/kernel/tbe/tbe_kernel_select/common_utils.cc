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

#include "plugin/device/ascend/kernel/tbe/tbe_kernel_select/common_utils.h"
#include <set>
#include <string>
#include "base/base.h"
#include "runtime/device/ms_device_shape_transfer.h"
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore {
namespace kernel {
bool HostCheck::CheckValidDeviceShape(const AnfNodePtr &node) {
  size_t real_input_num = common::AnfAlgo::GetInputTensorNum(node);
  for (size_t i = 0; i < real_input_num; i++) {
    session::KernelWithIndex kernel_with_index = common::AnfAlgo::GetPrevNodeOutput(node, i);
    auto format = AnfAlgo::GetInputFormat(node, i);
    if (!CheckValidOutputDeviceShape(kernel_with_index.first, kernel_with_index.second, format)) {
      MS_LOG(WARNING) << "TBE Host check input device shape failed, node:" << node->fullname_with_scope()
                      << ", input node: " << kernel_with_index.first->DebugString() << ", format:" << format;
      return false;
    }
  }

  size_t real_output_num = common::AnfAlgo::GetOutputTensorNum(node);
  for (size_t i = 0; i < real_output_num; i++) {
    auto format = AnfAlgo::GetOutputFormat(node, i);
    if (!CheckValidOutputDeviceShape(node, i, format)) {
      MS_LOG(WARNING) << "TBE Host check output device shape failed, node:" << node->fullname_with_scope()
                      << ", format:" << format;
      return false;
    }
  }
  return true;
}

std::vector<int64_t> HostCheck::GetFinalInferShape(const AnfNodePtr &node, const size_t output_idx,
                                                   const std::string &format) {
  auto output_shape = common::AnfAlgo::GetOutputDetailShape(node, output_idx);
  std::vector<int64_t> infer_shape;
  if (output_shape->isa<abstract::Shape>()) {
    auto shape_ptr = output_shape->cast<abstract::ShapePtr>();
    MS_EXCEPTION_IF_NULL(shape_ptr);
    infer_shape = shape_ptr->shape();
  }
  if (infer_shape.empty()) {
    return infer_shape;
  }

  if (trans::IsNeedPadding(format, infer_shape.size())) {
    infer_shape = trans::PaddingShape(infer_shape, format, AnfAlgo::GetOutputReshapeType(node, output_idx));
  }

  auto temp_shape = infer_shape;
  if (kNoPaddingFormatSet.find(format) == kNoPaddingFormatSet.end() && format != kOpFormat_FRACTAL_ZN_LSTM &&
      infer_shape.size() < kShape4dDims && k3DFormatSet.find(format) == k3DFormatSet.end()) {
    MS_LOG(DEBUG) << "Get Device Shape using a shape size is less than 4 ,should be Padding shape by Default firstly";
    temp_shape = trans::PaddingShapeTo4dDefault(infer_shape);
  }
  if (infer_shape.size() != trans::kNcdhw && k3DFormatSet.find(format) != k3DFormatSet.end()) {
    temp_shape = trans::PaddingShapeTo5dDefault(infer_shape);
  }
  return temp_shape;
}

bool HostCheck::CheckValidOutputDeviceShape(const AnfNodePtr &node, const size_t output_idx,
                                            const std::string &format) {
  auto infer_shape = GetFinalInferShape(node, output_idx, format);
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
    return infer_shape.size() >= kShape2dDims;
  }

  if (format == kOpFormat_ND_RNN_BIAS) {
    return infer_shape.size() > 0;
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
