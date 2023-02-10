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

#include "c_api/include/abstract.h"
#include "c_api/src/helper.h"
#include "c_api/src/common.h"
#include "c_api/src/utils.h"
#include "abstract/dshape.h"
#include "ir/dtype.h"

STATUS MSAssignAbstract(ResMgrHandle res_mgr, NodeHandle cur_node, ConstNodeHandle input_node) {
  if (res_mgr == nullptr || cur_node == nullptr || input_node == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] or [graph] or [inputs] are nullptr.";
    return RET_NULL_PTR;
  }
  auto node = GetSrcPtr<AnfNodePtr>(res_mgr, cur_node);
  if (node == nullptr) {
    MS_LOG(ERROR) << "Get source pointer failed.";
    return RET_NULL_PTR;
  }
  auto input = GetSrcPtr<AnfNodePtr>(res_mgr, input_node);
  if (input == nullptr) {
    MS_LOG(ERROR) << "Get source pointer failed.";
    return RET_NULL_PTR;
  }
  node->set_abstract(input->abstract());
  return RET_OK;
}

STATUS MSSetAbstract(ResMgrHandle res_mgr, NodeHandle node, TypeId type, const int64_t shape[], size_t shape_size) {
  if (res_mgr == nullptr || node == nullptr || shape == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] or [node] or [shape] are nullptr.";
    return RET_NULL_PTR;
  }
  auto node_impl = GetSrcPtr<AnfNodePtr>(res_mgr, node);
  if (node_impl == nullptr) {
    MS_LOG(ERROR) << "Get source pointer failed.";
    return RET_NULL_PTR;
  }
  auto type_ptr = mindspore::TypeIdToType(mindspore::TypeId(type));
  AbstractBasePtr abs = GetAbstract(type_ptr, shape, shape_size);
  node_impl->set_abstract(abs);
  return RET_OK;
}

STATUS MSSetMultiAbstract(ResMgrHandle res_mgr, NodeHandle node, TypeId type, const int64_t **shapes,
                          const size_t shape_sizes[], size_t abs_num) {
  if (res_mgr == nullptr || node == nullptr || shapes == nullptr || shape_sizes == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] or [node] or [shapes] or [shape_sizes] are nullptr.";
    return RET_NULL_PTR;
  }
  auto node_impl = GetSrcPtr<AnfNodePtr>(res_mgr, node);
  if (node_impl == nullptr) {
    MS_LOG(ERROR) << "Get source pointer failed.";
    return RET_NULL_PTR;
  }
  mindspore::AbstractBasePtrList abs_list{};
  for (size_t i = 0; i < abs_num; i++) {
    auto type_ptr = mindspore::TypeIdToType(mindspore::TypeId(type));
    AbstractBasePtr abs = GetAbstract(type_ptr, shapes[i], shape_sizes[i]);
    abs_list.push_back(abs);
  }
  node_impl->set_abstract(std::make_shared<AbstractTupleImpl>(abs_list));
  return RET_OK;
}
