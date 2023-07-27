/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_TBE_KERNEL_SELECT_H
#define MINDSPORE_TBE_KERNEL_SELECT_H

#include <string>
#include <vector>
#include <memory>
#include "kernel/oplib/opinfo.h"
#include "kernel/kernel_build_info.h"
#include "plugin/device/ascend/kernel/tbe/tbe_kernel_select/tbe_select_utils.h"

namespace mindspore::kernel {
void TbeMetadataInfo(const CNodePtr &kernel_node, std::vector<std::shared_ptr<KernelBuildInfo>> *kernel_info_list);
bool TbeCheckIsSupportedSpec(const CNodePtr &kernel_node, const KernelBuildInfoPtr &select_kernel_build_info);
bool TbeCheckIsSupportedAny(const CNodePtr &kernel_node);

class TbeKernelSelect {
  using OpInfoPtr = std::shared_ptr<OpInfo>;

 public:
  TbeKernelSelect(CNodePtr kernel_node, std::vector<std::shared_ptr<KernelBuildInfo>> *kernel_info_list);
  ~TbeKernelSelect() { kernel_info_list_ = nullptr; }
  bool CheckOpSupported();
  std::vector<std::shared_ptr<KernelBuildInfo>> GetSupportFormatDTypes();
  bool TbeCheckIsSupportedSpec(const CNodePtr &kernel_node, const KernelBuildInfoPtr &select_kernel_build_info);
  bool TbeCheckIsSupportedAny(const CNodePtr &kernel_node);

 private:
  void FilterInvalidKernelInfo();
  bool FilterInvalidShape(const KernelBuildInfoPtr &kernel_build_info, const std::vector<int64_t> &dynamic_inputs);
  bool FilterUnsupportedMatMul(const KernelBuildInfoPtr &kernel_build_info);
  bool IsShapeMatchFormat(const ShapeVector &shape, const std::string &format);
  bool IsShapeMatchFormatRNN(const ShapeVector &shape, const std::string &format);
  bool TbeCheckSupported(const KernelBuildInfoPtr &kernel_build_info);
  std::vector<int64_t> GetNodeDynamicInputs();
  bool Initialize();
  bool GetKernelBuildInfoFromCache();
  void GenerateKernelBuildInfo(const SupportFormatDType &support_format_dtype);
  void ConstructIOKernelBuildInfo(const OpIOInfoPtr &op_io_info, const std::string &support_dtype,
                                  const std::string &support_format, int64_t dynamic_num,
                                  KernelBuildInfoItem *kernel_build_info_item, size_t *io_index,
                                  size_t *real_put_index) const;
  void ConstructKernelBuildInfo(const KernelBuildInfoItem &input_kernel_build_info,
                                const KernelBuildInfoItem &output_kernel_build_info);
  void AddKernelBuildInfoToCache();
  bool IsSupportFormatDTypeValid(const SupportFormatDType &support_format_dtype);
  void PrintSupportedFormatDtype(const SupportFormatDType &support_format_dtype);
  std::vector<std::shared_ptr<kernel::KernelBuildInfo>> GetSupportFormatDTypesWithFilter();

  OpInfoPtr op_info_ = nullptr;
  CNodePtr cnode_ptr_ = nullptr;
  std::vector<std::shared_ptr<KernelBuildInfo>> *kernel_info_list_ = nullptr;
  std::string node_name_;
  std::string full_name_;
  nlohmann::json kernel_json_;
  std::string kernel_hash_name_;
  inline static mindspore::HashMap<std::string, std::vector<std::shared_ptr<KernelBuildInfo>>> select_cache_ = {};
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_TBE_KERNEL_SELECT_H
