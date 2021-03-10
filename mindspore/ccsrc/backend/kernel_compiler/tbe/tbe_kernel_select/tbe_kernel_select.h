/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include "backend/kernel_compiler/oplib/opinfo.h"
#include "backend/kernel_compiler/kernel_build_info.h"
#include "backend/kernel_compiler/tbe/tbe_kernel_select/common_utils.h"

namespace mindspore {
namespace kernel {
void TbeMetadataInfo(const CNodePtr &kernel_node, std::vector<std::shared_ptr<KernelBuildInfo>> *kernel_info_list);

class TbeKernelSelect {
  using OpInfoPtr = std::shared_ptr<OpInfo>;
  using KernelBuildInfoIter = std::vector<std::shared_ptr<KernelBuildInfo>>::iterator;

 public:
  TbeKernelSelect(CNodePtr kernel_node, std::vector<std::shared_ptr<KernelBuildInfo>> *kernel_info_list);
  ~TbeKernelSelect() = default;
  void TbeMetadataInfoEx();

 private:
  void GetCommonPatternKernelInfo(const OpInfo &op_info);
  void GetDynamicFormatPatternKernelInfo(const OpInfo &op_info);
  void GetAgnosticPatternKernelInfo(const OpInfo &op_info);
  void GetBroadcastPatternKernelInfo(const OpInfo &op_info);
  void GetReducePatternKernelInfo(const OpInfo &op_info);
  void FilterInVaildKernelInfo(const OpInfo &op_info);
  bool FilterInVaildShape(const KernelBuildInfoIter &kernel_build_info_iter, bool is_dynamic_input);
  static bool IsShapeMatchFormat(const std::vector<size_t> &shape, const std::string &format);
  bool TbeCheckSupported(const KernelBuildInfoIter &kernel_build_info_iter);
  static void SetTbeBuildCommonInfo(const OpInfo &op_info, KernelBuildInfo::KernelBuildInfoBuilder *builder);
  std::vector<int64_t> GetNodeDynamicInputs();
  bool GenBuilderItem(bool is_input, size_t kernel_build_info_index, size_t real_io_tensor_num,
                      const std::vector<std::shared_ptr<OpIOInfo>> &ios_info,
                      const std::vector<int64_t> &dyn_input_sizes, std::vector<std::string> *formats,
                      std::vector<TypeId> *device_types, std::vector<std::string> *reshape_types);
  static void CreateNewOpInfo(const OpInfo &op_info, const SupportFormat &support_format, OpInfo *op_info_new);
  static void CreateNewOpIOInfo(const OpIOInfo &op_io_info,
                                const std::vector<std::vector<std::string>> &support_format_item, size_t index,
                                OpIOInfo *op_io_info_new);
  // op select(dynamic)
  void CreateNewOpInfo(const mindspore::kernel::OpInfo &op_info, mindspore::kernel::OpInfo *op_info_new);
  static void CreateNewOpIOInfo(const OpIOInfo &op_io_info, const std::vector<std::string> &support_dtype,
                                const std::vector<std::string> &support_format, OpIOInfo *op_io_info_new);
  static std::vector<std::string> SplitStrToVec(const std::string &op_select_json_item);
  std::string OpSelectFormat();

  static void PrintSupportedFormat(const SupportFormat &support_format);

 private:
  CNodePtr cnode_ptr_;
  std::vector<std::shared_ptr<KernelBuildInfo>> *kernel_info_list_;
  std::string node_name_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_TBE_KERNEL_SELECT_H
