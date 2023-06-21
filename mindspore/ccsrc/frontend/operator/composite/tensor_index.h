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

#include "utils/hash_map.h"
#include "pipeline/jit/static_analysis/static_analysis.h"
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
  AnfNodePtr NormalizeSliceInfo(const AnfNodePtr &data_node, const AnfNodePtr &index_node,
                                const IndexHandleLevel &index_handle_level,
                                const abstract::AbstractSlicePtr &abs_slice_ptr, bool *empty);
  AnfNodePtrList NormalizeSlice(const AbstractBasePtrList &slice_info_abs, const AnfNodePtr &shape_node,
                                const AnfNodePtr &index_node);
  IndexHandleLevel PreHandleIndex(const AbstractBasePtr &data, const abstract::AbstractSlicePtr &abs_slice);

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
};

const int64_t kIndexMax = std::numeric_limits<int64_t>::max();
class Slice {
 public:
  Slice(const AbstractBasePtr &start_index, const AbstractBasePtr &stop_index, const AbstractBasePtr &step_index,
        int64_t dim_size) {
    CheckSliceType(start_index);
    CheckSliceType(stop_index);
    CheckSliceType(step_index);
    dim_size_ = dim_size;
    bool step_by_none_init = step_index->isa<abstract::AbstractNone>();
    if (step_by_none_init) {
      step_ = 1;
    } else {
      step_ = GetValue<int64_t>(step_index->BuildValue());
    }
    if (step_ == 0) {
      MS_EXCEPTION(ValueError) << "For 'StridedSlice', 'strides' cannot contain 0";
    }
    if (step_ < -kIndexMax) {
      step_ = -kIndexMax;
    }
    start_ = NormalizeIndex(start_index, step_, dim_size_);
    stop_ = NormalizeIndex(stop_index, -step_, dim_size_);
  }

  int64_t start() const { return start_; }
  int64_t stop() const { return stop_; }
  int64_t step() const { return step_; }

  static inline int64_t NormalizeIndex(int64_t index, int64_t dim_size) {
    int64_t new_index = index;
    if (new_index < 0) {
      MS_EXCEPTION_IF_ZERO("DimsSize should not be zero", dim_size);
      return new_index < -dim_size ? 0 : (dim_size + (new_index % dim_size)) % dim_size;  // NOLINT
    }
    return new_index < dim_size ? new_index : dim_size;
  }

  static inline int64_t NormalizeIndex(const AbstractBasePtr &index, int64_t step, int64_t dim_size) {
    int64_t normalized_index;
    if (index->isa<abstract::AbstractScalar>()) {
      normalized_index = NormalizeIndex(GetValue<int64_t>(index->BuildValue()), dim_size);
    } else if (index->isa<abstract::AbstractNone>()) {
      normalized_index = step > 0 ? 0 : dim_size;
    } else {
      MS_LOG(EXCEPTION) << "Slice index type must be int or none.";
    }
    return normalized_index;
  }

  static void CheckSliceType(const AbstractBasePtr &abs) {
    if (abs->isa<abstract::AbstractScalar>()) {
      if (abs->BuildType()->type_id() != kNumberTypeInt64) {
        MS_EXCEPTION(TypeError) << "The type of input of the MakeSlice operator must be int64 bot got "
                                << abs->ToString();
      }
    }
  }

 private:
  int64_t start_ = 0;
  int64_t stop_ = 0;
  int64_t step_ = 0;
  int64_t dim_size_ = 0;
};
}  // namespace prim
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_OPERATOR_COMPOSITE_TENSOR_INDEX_H_
