/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_MEM_REUSE_MEM_REUSE_CHECKER_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_MEM_REUSE_MEM_REUSE_CHECKER_H_
#include <map>
#include <set>
#include <vector>
#include <string>
#include <memory>
#include <functional>
#include "mindspore/core/ir/anf.h"
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "backend/common/mem_reuse/mem_reuse.h"
#include "kernel/common_utils.h"
namespace mindspore {
namespace memreuse {
constexpr auto kSplitC = '/';
class BACKEND_EXPORT MemReuseChecker {
 public:
  static MemReuseChecker &GetInstance();
  MemReuseChecker(const MemReuseChecker &) = delete;
  MemReuseChecker &operator=(const MemReuseChecker &) = delete;
  void CheckSignalOps(const CNodePtr &c_node) const;
  void CheckWorkSpace(const std::vector<size_t> &max_list);
  void CheckOutRef(const KernelRefs &kernel_refs, const CNodePtr &c_node, size_t output_idx) const;
  bool CheckGraphOutputAssigned(const session::KernelGraph *graph) const;
  void CheckMemReuseIR(const KernelRefCountPtrList &total_refs_list, const KernelDefPtrMaps &kernel_def_ptr_list,
                       const KernelGraph *graph);
  int64_t CalculOriStatic(const KernelGraph *graph) const;
  int64_t CalculOriInput(const KernelGraph *graph) const;
  int64_t CalculOriValue(const KernelGraph *graph) const;
  int64_t CalculOriDy(const KernelGraph *graph) const;
  int64_t CalculOriWk(const KernelGraph *graph) const;
  std::string GetSplitName(const std::string &scope_name) const;
  int GetTensorIdx(const void *in) const;
  void SetMembuInfos(const KernelDef *op_def, const std::vector<MembufPtr> &membuf_ptr_list);
  void SetTensorFromAndToInfo(const KernelDef *op_def);
  void ExportMemOpIr(const KernelDef *def, std::ofstream &ofs, int def_idx) const;
  void ExportNormalOpIr(const std::vector<CNodePtr> &cnodes);
  void ExportNormalTensorIR(std::ofstream &ofs);
  void CheckNormalIR(const session::KernelGraph *graph);
  void ExportMembufInfoIR();
  void ExportEachMembufInfo(std::ofstream &ofs);
  void SetAddNewMembuInfos(const KernelDef *op_def, const std::vector<MembufPtr> &membuf_ptr_list, size_t op_idx);
  void ExportAddNewMmebufIR();
  void set_kernel_front_map(const std::map<KernelDefPtr, std::set<KernelDefPtr>> &kernel_front_map) {
    kernel_front_map_ = kernel_front_map;
  }
  void ExportKernelDependence();

 private:
  MemReuseChecker() = default;
  ~MemReuseChecker() {}
  bool IsAddNewMembuf_ = false;
  size_t total_re_wkspe_size_checker_{0};
  std::vector<std::vector<MembufPtr>> membuf_all_infos_;
  std::vector<const void *> nor_output_tensors_;
  std::vector<size_t> nor_tensor_sizes_;
  std::vector<const void *> nor_input_tensors_;
  std::map<const void *, size_t> ptr_idx_;
  std::map<const void *, size_t> ptr_refs_;
  std::map<void *, std::vector<const void *>> node_ins_;
  std::map<void *, std::vector<const void *>> node_ous_;
  std::vector<std::vector<MembufPtr>> add_new_mem_infos_;
  std::vector<std::string> add_new_names_;
  std::vector<size_t> add_new_op_indxs_;
  std::vector<uint32_t> add_new_stream_ids_;
  std::vector<std::string> all_split_names_;
  std::map<int, std::vector<string>> tensor_from_;
  std::map<int, std::vector<string>> tensor_to_;
  std::map<KernelDefPtr, std::set<KernelDefPtr>> kernel_front_map_;
  int64_t total_ori_static_size_ = 0;
  int64_t total_ori_input_size_ = 0;
  int64_t total_ori_value_size_ = 0;
  int64_t total_ori_dy_size_ = 0;
  int64_t total_ori_wkspace_size_ = 0;
};
}  // namespace memreuse
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_MEM_REUSE_MEM_REUSE_CHECKER_H_
