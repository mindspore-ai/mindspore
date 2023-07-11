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
#include <vector>

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
  AnfNodePtrList ParseSlice(const AnfNodePtr &index_node, const abstract::AbstractSlicePtr &abs_slice_ptr,
                            std::vector<int64_t> *init_by_one);
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
}  // namespace prim
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_OPERATOR_COMPOSITE_TENSOR_INDEX_H_
