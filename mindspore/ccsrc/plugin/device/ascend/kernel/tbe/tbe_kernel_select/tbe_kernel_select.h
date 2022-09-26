/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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
  void TbeMetadataInfoEx();
  bool FindKernelInfo(const KernelBuildInfoPtr &select_kernel_build_info);
  bool CheckIsAnyKernelInfo();

 private:
  void GetCommonPatternKernelInfo(const OpInfo &op_info);
  void GetDynamicFormatPatternKernelInfo(const OpInfo &op_info);
  void GetAgnosticPatternKernelInfo(const OpInfo &op_info);
  void GetBroadcastPatternKernelInfo(const OpInfo &op_info);
  void GetReducePatternKernelInfo(const OpInfo &op_info);
  void FilterInvalidKernelInfo();
  bool FilterInvalidShape(const KernelBuildInfoPtr &kernel_build_info, const std::vector<int64_t> &dynamic_inputs);
  bool IsShapeMatchFormat(const ShapeVector &shape, const std::string &format);
  bool IsShapeMatchFormatRNN(const ShapeVector &shape, const std::string &format);
  bool TbeCheckSupported(const KernelBuildInfoPtr &kernel_build_info);
  static void SetTbeBuildCommonInfo(const OpInfo &op_info, KernelBuildInfo::KernelBuildInfoBuilder *builder);
  std::vector<int64_t> GetNodeDynamicInputs();
  bool GenBuilderItem(bool is_input, size_t kernel_build_info_index, size_t real_io_tensor_num,
                      const std::vector<std::shared_ptr<OpIOInfo>> &ios_info,
                      const std::vector<int64_t> &dyn_input_sizes, std::vector<std::string> *formats,
                      std::vector<TypeId> *device_types, std::vector<std::string> *reshape_types,
                      std::vector<std::string> *value_depends);
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
  void GetKernelHashName();
  bool CheckCNode();

  CNodePtr cnode_ptr_;
  std::vector<std::shared_ptr<KernelBuildInfo>> *kernel_info_list_;
  std::string node_name_;
  std::string full_name_;
  nlohmann::json kernel_json;
  std::string kernel_hash_name;
  bool check_cnode;
  inline static mindspore::HashMap<std::string, std::vector<std::shared_ptr<KernelBuildInfo>>> select_cache_ = {};

 private:
  bool Initialize();
  bool GetKernelBuildInfoFromCache();
  void GenerateKernelBuildInfo(const SupportFormatDType &support_format_dtype);
  void ConstructIOKernelBuildInfo(const OpIOInfoPtr &op_io_info, const std::string &support_dtype,
                                  const std::string &support_format, int64_t dynamic_num,
                                  KernelBuildInfoItem *kernel_build_info_item, size_t *io_index,
                                  size_t *real_put_index) const;
  OpInfoPtr op_info_ = nullptr;

  void ConstructKernelBuildInfo(const KernelBuildInfoItem &input_kernel_build_info,
                                const KernelBuildInfoItem &output_kernel_build_info);
  void AddKernelBuildInfoToCache();

  void PrintSupportedFormatDtype(const SupportFormatDType &support_format_dtype);
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_TBE_KERNEL_SELECT_H
