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

#ifndef MINDSPORE_LITE_SRC_EXTENDRT_UTILS_CUSTOM_ASCEND_UTILS_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_UTILS_CUSTOM_ASCEND_UTILS_H_
#include <utility>
#include <string>
#include <vector>
#include <tuple>
#include <map>
#include <memory>

#include "ir/anf.h"
#include "ir/dtype/type.h"
#include "ir/func_graph.h"
#include "include/api/types.h"
#include "include/api/data_type.h"
#include "include/api/status.h"
#include "mindspore/ccsrc/kernel/kernel.h"
#include "include/common/utils/anfalgo.h"
#include "mindspore/core/ops/custom.h"
#include "mindspore/lite/src/common/common.h"

namespace mindspore {
struct DynKVCacheSaveInfo {
  bool batch_size_dyn = false;
  bool seq_length_dyn = false;
  std::string kv_cache_layout = lite::kKVCacheLayoutBNSD;
};

static inline std::vector<int64_t> SetKVCacheShape(bool dyn_batch, bool dyn_seq, const std::string &layout,
                                                   const std::vector<int64_t> &org_shape) {
  std::vector<int64_t> shape = org_shape;
  if (layout == lite::kKVCacheLayoutBNSD && org_shape.size() == kShape4dDims) {
    if (dyn_batch) {
      shape[kDim0] = abstract::Shape::kShapeDimAny;
    }
    if (dyn_seq) {
      shape[kDim2] = abstract::Shape::kShapeDimAny;
    }
  } else if (layout == lite::kKVCacheLayoutBSH && org_shape.size() == kShape3dDims) {
    if (dyn_batch) {
      shape[kDim0] = abstract::Shape::kShapeDimAny;
    }
    if (dyn_seq) {
      shape[kDim1] = abstract::Shape::kShapeDimAny;
    }
  }
  return shape;
}

class MS_API CustomAscendUtils {
 public:
  static bool CreateCustomFuncGraph(const FuncGraphPtr &func_graph, const Buffer &model_cache,
                                    const std::string &graph_name, const std::map<std::string, ValuePtr> &attr_map,
                                    const std::vector<std::string> &ref_datas,
                                    const DynKVCacheSaveInfo &dyn_kv_info = {});

  static bool ParseCustomFuncGraph(const FuncGraphPtr &func_graph, tensor::TensorPtr *model_cache,
                                   std::string *graph_name, std::map<std::string, ValuePtr> *attr_map,
                                   std::vector<std::pair<std::string, tensor::TensorPtr>> *ref_datas,
                                   DynKVCacheSaveInfo *dyn_kv_info = nullptr);

  static bool IsCustomFuncGraph(const FuncGraphPtr &func_graph);
  static ParameterPtr CreateOmParameter(const FuncGraphPtr &func_graph, const Buffer &om_data,
                                        const std::string &graph_name);

 private:
  std::vector<std::pair<AnfNodePtr, size_t>> outputs_;

  CNodePtr CreateCustomNode(const FuncGraphPtr &func_graph, const ParameterPtr &om_parameter,
                            const std::map<std::string, ValuePtr> &attr_map, const std::vector<std::string> &ref_datas);
  void SetCustomAttrs(const std::shared_ptr<ops::Custom> &prim, const std::map<std::string, ValuePtr> &attr_map);
  bool SetCustomOutputs(const FuncGraphPtr &func_graph, const CNodePtr &custom_node);
  bool ModifyGraphByCustomNode(const FuncGraphPtr &func_graph, const CNodePtr &custom_node);
  CNodePtr CreateMakeTupleGraphOutput(const FuncGraphPtr &func_graph, const CNodePtr &custom_node);
  static CNodePtr GetCustomNode(const FuncGraphPtr &func_graph);
  static bool IsParameterValueZero(const tensor::TensorPtr &tensor);

  static void SetZeroValueRefDatas(const ops::PrimitiveCPtr &primc,
                                   const std::vector<std::pair<std::string, tensor::TensorPtr>> &ref_infos);
  static bool GetZeroValueRefDatas(const ops::PrimitiveCPtr &primc,
                                   std::vector<std::pair<std::string, tensor::TensorPtr>> *ref_infos);
};
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_UTILS_CUSTOM_ASCEND_UTILS_H_
