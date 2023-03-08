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
#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_SELECT_TBE_KERNEL_TBE_SELECT_UTILS_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_SELECT_TBE_KERNEL_TBE_SELECT_UTILS_H_

#include <string>
#include <vector>
#include <memory>
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "kernel/oplib/opinfo.h"

namespace mindspore::kernel {
using SupportFormatItem = std::vector<std::string>;
struct SupportFormat {
  std::vector<SupportFormatItem> input_format;
  std::vector<SupportFormatItem> output_format;
};

using SupportDTypeItem = std::vector<std::string>;
struct SupportFormatDType {
  std::vector<SupportDTypeItem> input_dtypes;
  std::vector<SupportFormatItem> input_formats;
  std::vector<SupportDTypeItem> output_dtypes;
  std::vector<SupportFormatItem> output_formats;
};

struct KernelBuildInfoItem {
  std::vector<std::string> formats;
  std::vector<TypeId> device_types;
  std::vector<std::string> reshape_types;
};
class HostCheck {
 public:
  HostCheck() = default;
  ~HostCheck() = default;
  static bool CheckValidDeviceShape(const AnfNodePtr &node);

 private:
  static bool CheckValidInOutDeviceShape(const AnfNodePtr &node, size_t index, bool is_output,
                                         const std::string &format);
  static std::vector<int64_t> GetFinalInferShape(const AnfNodePtr &node, size_t index, bool is_output,
                                                 const std::string &format);
};
bool IsOpSupportDynamicImpl(const CNodePtr &cnode);

bool IsKernelDynamicImpl(const AnfNodePtr &node);

void GetSupportOriFormat(const CNodePtr &cnode, SupportFormat *support_format);

void PadScalarShape(ShapeVector *shape);

void GenerateSupportFormat(const std::string &support_input_format, size_t input_num,
                           const std::string &support_output_format, size_t output_num, SupportFormat *support_format);

void ConstructSupportDTypes(const std::vector<OpIOInfoPtr> &puts, size_t format_size,
                            std::vector<SupportDTypeItem> *support_dtypes);

void ConstructSupportFormats(size_t put_size, const std::vector<SupportFormatItem> &support_format, size_t type_size,
                             std::vector<SupportFormatItem> *support_formats);

void GenerateSupportFormatDType(const CNodePtr &cnode, const SupportFormat &support_format,
                                SupportFormatDType *support_format_dtype);

std::vector<std::shared_ptr<kernel::KernelBuildInfo>> FilterRaisedOrReducePrecisionMatchedKernelInfo(
  const CNodePtr &cnode, const std::vector<std::shared_ptr<kernel::KernelBuildInfo>> &kernel_info_list,
  bool *precision_reduce);
}  // namespace mindspore::kernel

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_SELECT_TBE_KERNEL_TBE_SELECT_UTILS_H_
