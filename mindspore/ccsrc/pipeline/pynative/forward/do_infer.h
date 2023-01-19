/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PIPELINE_PYNATIVE_FORWARD_DO_INFER_H_
#define MINDSPORE_CCSRC_PIPELINE_PYNATIVE_FORWARD_DO_INFER_H_

#include <vector>
#include <string>
#include <memory>
#include <set>
#include "pipeline/pynative/base.h"
#include "pipeline/pynative/pynative_cache.h"

namespace mindspore {
namespace pynative {
class InferOperation {
 public:
  InferOperation() = default;
  ~InferOperation() = default;
  void DoInfer(const FrontendOpRunInfoPtr &op_run_info);
  // Manage node abs cache.
  inline void ClearNodeAbsCache() { node_abs_cache_.clear(); }
  void SetNodeAbsCacheByValue(const FrontendOpRunInfoPtr &op_run_info);
  void SetNodeAbsCacheById(const std::string &id, const abstract::AbstractBasePtr &abs);
  inline const NodeAbsCache &node_abs_cache() const { return node_abs_cache_; }
  // Manage primitive output abstract cache.
  inline void ClearPrimAbsList() { prim_abs_list_.clear(); }
  // Manage constant flag primitive cache.
  inline void ClearConstFlagPrimCache() { no_const_flag_prims_.clear(); }
  py::object CallConstantFolding(const py::args &args) const;
  void set_only_single_op_run(bool only_single_op_run) { only_single_op_run_ = only_single_op_run; }

 private:
  void PynativeInfer(const FrontendOpRunInfoPtr &op_run_info) const;
  // Set abstract for each input value.
  void SetInputAbstract(const FrontendOpRunInfoPtr &op_run_info);
  AbstractBasePtr GetInputValueAbs(const FrontendOpRunInfoPtr &op_run_info, const ValuePtr &input_value,
                                   size_t input_index, bool marked_const);
  AbstractBasePtr GetInputTupleValueAbstract(const FrontendOpRunInfoPtr &op_run_info,
                                             const ValueSequencePtr &tuple_value, size_t input_index,
                                             bool marked_const);
  AbstractBasePtr GetAbstractByValue(const ValuePtr &value, size_t input_index, const std::string &input_id,
                                     bool marked_const);
  // Infer output abstract.
  void InferOutputAbstract(const FrontendOpRunInfoPtr &op_run_info);
  bool GetOutputAbstractByCache(const FrontendOpRunInfoPtr &op_run_info) const;
  void SaveOutputAbstractToCache(const FrontendOpRunInfoPtr &op_run_info);
  void SaveSpecifiedOutputToCache(const std::string &op_name, const ValuePtrList &value_list,
                                  const AbstractBasePtrList &abs_list);
  // Check whether primitive has constant flag or input position has been marked constant.
  std::vector<bool> CheckPrimitiveConstFlag(const FrontendOpRunInfoPtr &op_run_info);

  bool only_single_op_run_{true};
  // The primitive has no constant flag(const prim or const input) will be saved in this map.
  mindspore::HashSet<std::string> no_const_flag_prims_;
  // This map is used to get the input abstract of input value form cache.
  // It works when top cell forward run begin and is cleared when top cell forward run end.
  NodeAbsCache node_abs_cache_;
  // This map is used to cache op output abstract.
  PrimAbsCache prim_abs_list_;
};
using InferOperationPtr = std::shared_ptr<InferOperation>;
}  // namespace pynative
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PIPELINE_PYNATIVE_FORWARD_DO_INFER_H_
