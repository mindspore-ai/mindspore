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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPERATOR_COMPOSITE_TENSOR_INDEX_H_
#define MINDSPORE_CCSRC_FRONTEND_OPERATOR_COMPOSITE_TENSOR_INDEX_H_

#include <string>
#include <map>
#include <set>
#include <memory>
#include <limits>
#include <algorithm>
#include <tuple>
#include <vector>

#include "utils/hash_map.h"
#include "utils/misc.h"
#include "utils/any.h"
#include "ir/dtype.h"
#include "ir/meta_func_graph.h"
#include "utils/ms_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
// namespace to support composite operators definition
namespace prim {
enum class IndexHandleLevel { kHandleByConstFold, kHandleByFunc };

class TensorIndex : public MetaFuncGraph {
 public:
  explicit TensorIndex(const std::string &name) : MetaFuncGraph(name) {}
  ~TensorIndex() override = default;

 protected:
  AnfNodePtrList ParseSlice(const AnfNodePtr &index_node, const abstract::AbstractSlicePtr &abs_slice_ptr,
                            std::vector<int64_t> *init_by_one);
  IndexHandleLevel PreHandleIndex(const AbstractBasePtr &data, const abstract::AbstractTuplePtr &tuple_abs);
  AnfNodePtr IntIndexToTensor(const AnfNodePtr &data_node, const AnfNodePtr &index_node,
                              const AbstractBasePtr &int_index_abs, const std::vector<int64_t> &tuple_index_types,
                              size_t dim_index, int64_t expand_dims_mask);
  AnfNodePtr SequenceIndexToTensor(const AnfNodePtr &data_node, const AnfNodePtr &sequence_index_node,
                                   const std::vector<int64_t> &tuple_index_types,
                                   const AbstractBasePtr &sequence_index_abs, const size_t dim_index,
                                   int64_t expand_dims_mask, bool *empty_sequence);
  AnfNodePtr SliceIndexToTensor(const AnfNodePtr &data_node, const std::vector<int64_t> &tuple_index_types,
                                const AnfNodePtr &slice_index_node, const abstract::AbstractSlicePtr &slice_abs,
                                const size_t dim_index, const IndexHandleLevel index_handle_level,
                                int64_t expand_dims_mask);

  AnfNodePtr NoneIndexToTensor(const AnfNodePtr &data_node, const std::vector<int64_t> &tuple_index_types,
                               const AnfNodePtr &none_index_node, const size_t dim_index);
  AnfNodePtr EllipsisIndexToTensor(const AnfNodePtr &data_node, const std::vector<int64_t> &tuple_index_types,
                                   const AnfNodePtr &none_index_node, const size_t dim_index);
  std::vector<AnfNodePtr> NormalizeTupleIndex(const AnfNodePtr &data_node, const AnfNodePtr &index_node,
                                              const std::vector<int64_t> &tuple_index_types,
                                              const IndexHandleLevel index_handle_level, const bool has_ellipsis,
                                              const abstract::AbstractTuplePtr &tuple_abs_ptr);
  void RemakeTupleIndex(bool has_ellipsis, const std::vector<int64_t> &tuple_index_types, const AnfNodePtr &data_node,
                        const std::vector<AnfNodePtr> &new_normalized_tensors, size_t not_ellipsis_position_cnt,
                        size_t ellipsis_position);
  std::vector<CNodePtr> GetTupleIndexInfo(const AnfNodePtr &data_node, const AnfNodePtr &fancy_position_node,
                                          const std::vector<AnfNodePtr> &normalized_tensors,
                                          const mindspore::HashMap<std::string, ValuePtr> &attrs);
  AnfNodePtr ExpandDimsByTupleIndex(const AnfNodePtr &input_data_node, const abstract::AbstractTuplePtr &tuple_abs_ptr,
                                    const std::vector<int64_t> &tuple_index_types, size_t expand_dims_cnt);

  FuncGraphPtr res_graph_;
  ShapeVector data_shape_;
};

class TensorIndexGetitem : public TensorIndex {
 public:
  explicit TensorIndexGetitem(const std::string &name) : TensorIndex(name) {}
  ~TensorIndexGetitem() override = default;
  MS_DECLARE_PARENT(TensorIndexGetitem, MetaFuncGraph)
  FuncGraphPtr GenerateFuncGraph(const AbstractBasePtrList &args_spec_list) override;
  friend bool operator==(const TensorIndexGetitem &lhs, const TensorIndexGetitem &rhs) {
    return lhs.name_ == rhs.name_;
  }

 private:
  void GetItemBySlice(const AnfNodePtr &data_node, const AnfNodePtr &index_node, const AbstractBasePtr &data,
                      const abstract::AbstractSlicePtr &abs_slice_ptr);
  void GetItemByTuple(const AnfNodePtr &data_node, const AnfNodePtr &index_node, const AbstractBasePtr &data,
                      const abstract::AbstractTuplePtr &tuple_abs_ptr, const AbstractBasePtr &all_empty_tensor_index);
  std::tuple<AnfNodePtr, AnfNodePtr, AnfNodePtr> NormalizeStrideInfoFromTuple(
    const AnfNodePtr &data_node, const AnfNodePtr &index_node, const AbstractBasePtr &index_abs,
    const std::vector<int64_t> &tuple_index_types, size_t tuple_index);
  void ConstGetStrideInfoFromTuple(const AnfNodePtr &data_node, const AnfNodePtr &index_node,
                                   const std::vector<int64_t> &tuple_index_types, bool has_ellipsis,
                                   const abstract::AbstractTuplePtr &tuple_abs_ptr, size_t not_ellipsis_position_cnt,
                                   size_t ellipsis_position);
  void GetStrideInfoFromTuple(const AnfNodePtr &data_node, const AnfNodePtr &index_node,
                              const std::vector<int64_t> &tuple_index_types, const IndexHandleLevel index_handle_level,
                              bool has_ellipsis, const abstract::AbstractTuplePtr &tuple_abs_ptr,
                              size_t not_ellipsis_position_cnt, size_t ellipsis_position);
  AnfNodePtrList EllipsisIndexToSlice(const std::vector<int64_t> &tuple_index_types, const AnfNodePtr &data_node,
                                      const AnfNodePtr &begin_stride, const AnfNodePtr &end_stride,
                                      const AnfNodePtr &step_stride);
};

class TensorIndexSetitem : public TensorIndex {
 public:
  explicit TensorIndexSetitem(const std::string &name) : TensorIndex(name) {}
  ~TensorIndexSetitem() override = default;
  MS_DECLARE_PARENT(TensorIndexSetitem, MetaFuncGraph)
  FuncGraphPtr GenerateFuncGraph(const AbstractBasePtrList &args_spec_list) override;
  friend bool operator==(const TensorIndexSetitem &lhs, const TensorIndexSetitem &rhs) {
    return lhs.name_ == rhs.name_;
  }

 private:
  void SetItemBySlice(const AnfNodePtr &data_node, const AnfNodePtr &index_node, const AnfNodePtr &value_node,
                      const AbstractBasePtr &data, const abstract::AbstractSlicePtr &abs_slice_ptr,
                      const AbstractBasePtr &value);
  void SetItemByTuple(const AnfNodePtr &data_node, const AnfNodePtr &index_node, const AnfNodePtr &value_node,
                      const AnfNodePtr &fancy_position_node, const AbstractBasePtr &data,
                      const abstract::AbstractTuplePtr &abs_slice_ptr, const AbstractBasePtr &value,
                      const AbstractBasePtr &fancy_position);
};

class HandleEmptySlice : public TensorIndexGetitem {
 public:
  explicit HandleEmptySlice(const std::string &name) : TensorIndexGetitem(name) {}
  ~HandleEmptySlice() override = default;
  MS_DECLARE_PARENT(HandleEmptySlice, MetaFuncGraph)
  FuncGraphPtr GenerateFuncGraph(const AbstractBasePtrList &args_spec_list) override;
  friend bool operator==(const HandleEmptySlice &lhs, const HandleEmptySlice &rhs) { return lhs.name_ == rhs.name_; }
  void HandleEmptySliceByTupleIndex(const AnfNodePtr &data_node, const AnfNodePtr &index_node,
                                    const AbstractBasePtr &data, const abstract::AbstractTuplePtr &tuple_abs_ptr);
};

class HandleBoolTensor : public TensorIndex {
 public:
  explicit HandleBoolTensor(const std::string &name) : TensorIndex(name) {}
  ~HandleBoolTensor() override = default;
  MS_DECLARE_PARENT(HandleBoolTensor, MetaFuncGraph)
  FuncGraphPtr GenerateFuncGraph(const AbstractBasePtrList &args_spec_list) override;
  friend bool operator==(const HandleBoolTensor &lhs, const HandleBoolTensor &rhs) { return lhs.name_ == rhs.name_; }
};

class HandleScalarTensorIndex : public TensorIndex {
 public:
  explicit HandleScalarTensorIndex(const std::string &name) : TensorIndex(name) {}
  ~HandleScalarTensorIndex() override = default;
  MS_DECLARE_PARENT(HandleScalarTensorIndex, MetaFuncGraph)
  FuncGraphPtr GenerateFuncGraph(const AbstractBasePtrList &args_spec_list) override;
  friend bool operator==(const HandleScalarTensorIndex &lhs, const HandleScalarTensorIndex &rhs) {
    return lhs.name_ == rhs.name_;
  }
};

class PreSetitemByTuple : public TensorIndex {
 public:
  explicit PreSetitemByTuple(const std::string &name) : TensorIndex(name) {}
  ~PreSetitemByTuple() override = default;
  MS_DECLARE_PARENT(PreSetitemByTuple, MetaFuncGraph)
  FuncGraphPtr GenerateFuncGraph(const AbstractBasePtrList &args_spec_list) override;
  friend bool operator==(const PreSetitemByTuple &lhs, const PreSetitemByTuple &rhs) { return lhs.name_ == rhs.name_; }
  AnfNodePtr FormatIndex(const abstract::AbstractBasePtr &index_abs, const AnfNodePtr &data_node,
                         const AnfNodePtr &index_node, size_t cur_dim, const std::vector<int64_t> &tuple_index_types,
                         int64_t expand_dims_mask, bool *empty_sequence);
  void RemoveExpandedDims(const AnfNodePtr &data_node, const AnfNodePtr &index_node, const AnfNodePtr &value_node,
                          const std::vector<int64_t> &tuple_index_types, const IndexHandleLevel index_handle_level,
                          const bool has_ellipsis, const abstract::AbstractTuplePtr &tuple_abs_ptr,
                          int64_t expand_dims_mask);
};

}  // namespace prim
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_OPERATOR_COMPOSITE_TENSOR_INDEX_H_
