/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include <cstddef>
#include <optional>
#include "ir/anf.h"
#include "abstract/dshape.h"
#include "include/backend/kernel_info.h"
#include "transform/graph_ir/storage_format_config_factory.h"
#include "utils/log_adapter.h"

namespace mindspore::transform {
namespace {
// number of dims of format NCHW
constexpr size_t kNumDimsNCHW = 4;

inline ShapeVector GetShape(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto base_shape = node->abstract()->GetShape();
  if (base_shape->isa<abstract::Shape>()) {
    auto shape_ptr = base_shape->cast<abstract::ShapePtr>();
    MS_EXCEPTION_IF_NULL(shape_ptr);
    return shape_ptr->shape();
  }
  return {};
}

std::string GetParamStorageFormat(const ParameterPtr &param) {
  // param has default should have kernel info with one output
  MS_EXCEPTION_IF_NULL(param);
  std::shared_ptr<device::KernelInfo> kernel_info =
    std::dynamic_pointer_cast<device::KernelInfo>(param->kernel_info_ptr());
  MS_EXCEPTION_IF_NULL(kernel_info);
  kernel::KernelBuildInfoPtr build_info = kernel_info->GetMutableSelectKernelBuildInfo();
  MS_EXCEPTION_IF_NULL(build_info);
  return build_info->GetOutputFormat(0);
}
}  // namespace

template <int input_index>
std::optional<std::string> GetConv2DFormat(const AnfNodePtr & /* user_node */,
                                           const std::shared_ptr<GeTensorDesc> &desc) {
  if (desc == nullptr || desc->GetOriginShape().GetDimNum() != kNumDimsNCHW) {
    return std::nullopt;
  }
  return input_index == 0 ? kOpFormat_NC1HWC0 : kOpFormat_FRAC_Z;
}

std::optional<std::string> GetApplyMomentumFormat(const AnfNodePtr &user_node,
                                                  const std::shared_ptr<GeTensorDesc> & /*desc*/) {
  MS_EXCEPTION_IF_NULL(user_node);
  auto cnode = user_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);

  auto input_var = cnode->input(1);
  MS_EXCEPTION_IF_NULL(input_var);
  auto input_accum = cnode->input(2);
  MS_EXCEPTION_IF_NULL(input_accum);

  auto param_var = input_var->cast<ParameterPtr>();
  auto param_accum = input_accum->cast<ParameterPtr>();
  if (param_var == nullptr || param_accum == nullptr) {
    return std::nullopt;
  }

  auto var_fmt = GetParamStorageFormat(param_var);
  auto accum_fmt = GetParamStorageFormat(param_accum);
  if (var_fmt.empty() || accum_fmt == var_fmt) {
    return std::nullopt;
  }

  return var_fmt;
}

std::optional<std::string> GetBNTrainingUpdateFormat(const AnfNodePtr &user_node,
                                                     const std::shared_ptr<GeTensorDesc> & /*desc*/) {
  MS_EXCEPTION_IF_NULL(user_node);
  auto cnode = user_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);

  auto input_x = cnode->input(1);
  if (GetShape(input_x).size() != kNumDimsNCHW) {
    return std::nullopt;
  }

  return kOpFormat_NC1HWC0;
}

REGISTER_STORAGE_FORMAT_CONFIG(Conv2D)
  .set_index_format(0, GetConv2DFormat<0>, "")
  .set_index_format(1, GetConv2DFormat<1>, "");

REGISTER_STORAGE_FORMAT_CONFIG(ApplyMomentum).set_index_format(1, GetApplyMomentumFormat, "");

REGISTER_STORAGE_FORMAT_CONFIG(BNTrainingUpdate)
  .set_index_format(3, GetBNTrainingUpdateFormat, "")
  .set_index_format(4, GetBNTrainingUpdateFormat, "")
  .set_index_format(5, GetBNTrainingUpdateFormat, "")
  .set_index_format(6, GetBNTrainingUpdateFormat, "");

}  // namespace mindspore::transform
