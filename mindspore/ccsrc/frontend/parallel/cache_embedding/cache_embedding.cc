/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#include "frontend/parallel/cache_embedding/cache_embedding.h"
#include <random>
#include <vector>
#include <list>
#include <queue>
#include <utility>
#include <memory>
#include <string>
#include <algorithm>
#include "utils/hash_map.h"
#include "utils/hash_set.h"
#include "include/backend/optimizer/helper.h"
#include "frontend/optimizer/optimizer.h"
#include "ir/func_graph.h"
#include "utils/cache_embedding_hashmap_struct.h"
namespace mindspore {
namespace parallel {
using ParamMap = mindspore::HashMap<ParameterPtr, ParameterPtr>;
using ParamSet = mindspore::HashSet<ParameterPtr>;
using NodePairList = std::vector<std::pair<AnfNodePtr, AnfNodePtr>>;
using AnfMap = mindspore::HashMap<AnfNodePtr, AnfNodePtr>;
using AnfSet = mindspore::HashSet<AnfNodePtr>;

ParamMap AddCacheParameters(const FuncGraphPtr &graph, const ParamSet &parameter_cache_enable_set) {
  ParamMap cache_host_params_map;
  for (auto &param : parameter_cache_enable_set) {
    auto param_info = param->param_info();
    if (param_info && param_info->cache_enable()) {
      auto data_type = param->Type();
      auto data_element_type = data_type->cast<mindspore::TensorTypePtr>()->element();
      auto type_id = data_element_type->type_id();
      auto cache_shape = param_info->cache_shape();
      auto ori_param_name = param->name();
      auto new_tensor = std::make_shared<tensor::Tensor>(type_id, cache_shape);
      ParamInfoPtr new_param_info = std::make_shared<ParamInfo>();
      auto cache_name = ori_param_name + "_cache";
      new_param_info->set_name(cache_name);
      new_tensor->set_param_info(new_param_info);
      auto cache_param = graph->AddFvParameter(cache_name, new_tensor);
      cache_host_params_map[cache_param] = param;
    }
  }
  return cache_host_params_map;
}

bool CheckHostCacheParamSize(const ParamSet &parameter_cache_enable_set) {
  int64_t host_size = 0;
  int64_t cache_size = 0;
  for (auto &host_param : parameter_cache_enable_set) {
    auto tmp_host_size = host_param->abstract()->GetShapeTrack()->cast<abstract::ShapePtr>()->shape()[0];
    auto host_param_info = host_param->param_info();
    auto cache_shape = host_param_info->cache_shape();
    if (cache_shape.empty()) {
      MS_LOG(EXCEPTION) << "The value of cache_shape is empty.";
    }
    auto tmp_cache_size = cache_shape[0];
    if ((host_size != 0 && tmp_host_size != host_size) || (cache_size != 0 && tmp_cache_size != cache_size)) {
      MS_LOG(EXCEPTION)
        << "If EmbeddingLookup are cache enable, vocab_size and vocab_cache_size of different cells must be the same.";
    }
    cache_size = tmp_cache_size;
    host_size = tmp_host_size;
  }
  if (cache_size > host_size) {
    MS_LOG(WARNING) << "vocab_cache_size > vocab_size, there is no need use cache.";
    return false;
  }
  return true;
}

void ReplaceCacheParams(const FuncGraphPtr &graph, const ParamMap &map) {
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  for (auto &ele : map) {
    if (!manager->Replace(ele.second, ele.first)) {
      MS_LOG(EXCEPTION) << "host param: " << ele.second->name() << ", replace node failed.";
    }
  }
}

ParamSet MapKeysToSet(const ParamMap &map) {
  ParamSet set;
  for (auto &ele : map) {
    set.insert(ele.first);
  }
  return set;
}

ParamSet FindParamCacheEnable(const FuncGraphPtr &graph) {
  ParamSet parameter_cache_enable_set;
  auto parameters = graph->parameters();
  auto params_size = parameters.size();
  for (size_t i = 0; i < params_size; ++i) {
    auto param = parameters[i]->cast<ParameterPtr>();
    auto param_info = param->param_info();
    if (param_info && param_info->cache_enable()) {
      parameter_cache_enable_set.insert(param);
    }
  }
  return parameter_cache_enable_set;
}

CNodePtrList FindUniqueCacheEnable(const CNodePtrList &cnodes) {
  size_t cnodes_size = cnodes.size();
  CNodePtrList unique_cache_enable;
  for (size_t i = 0; i < cnodes_size; ++i) {
    if (IsPrimitiveCNode(cnodes[i], prim::kPrimUnique)) {
      auto unique_node = cnodes[i];
      auto unique_prim = GetCNodePrimitive(unique_node);
      MS_EXCEPTION_IF_NULL(unique_prim);
      auto attr_value = unique_prim->GetAttr(kAttrCacheEnable);
      if (attr_value != nullptr && GetValue<bool>(attr_value)) {
        unique_cache_enable.emplace_back(unique_node);
      }
    }
  }
  if (unique_cache_enable.size() > 1) {
    MS_LOG(EXCEPTION) << "Support only one of Unique op cache enable, but got " << unique_cache_enable.size();
  }
  return unique_cache_enable;
}

template <typename T>
void MemCopyFromHostToCache(void *hashmap_addr, void *host_addr, void *cache_addr, size_t host_max, size_t cache_max,
                            size_t hashmap_size, size_t col_size) {
  auto host_data = static_cast<char *>(host_addr);
  auto cache_data = static_cast<char *>(cache_addr);
  auto hashmap_data = static_cast<HashmapEntry<T> *>(hashmap_addr);
  // default param type float
  const size_t param_type_size = 4;
  size_t single_col_bytes = param_type_size * col_size;
  for (size_t i = 0; i < hashmap_size; ++i) {
    if (!hashmap_data[i].IsEmpty()) {
      size_t host_offset = single_col_bytes * static_cast<size_t>(hashmap_data[i].key_);
      size_t cache_offset = single_col_bytes * static_cast<size_t>(hashmap_data[i].value_);
      if (host_offset + single_col_bytes <= host_max) {
        auto ret =
          memcpy_s(cache_data + cache_offset, cache_max - cache_offset, host_data + host_offset, single_col_bytes);
        if (ret != EOK) {
          MS_LOG(EXCEPTION) << "Memcpy failed.";
        }
      }
    }
  }
  MS_LOG(INFO) << "Memcpy from cache to host success!";
}

void BindAndInitCacheTensor(const ParamMap &param_pair_list, const ParameterPtr &hashmap) {
  auto hashmap_tensor_value = hashmap->default_param();
  auto hashmap_tensor = hashmap_tensor_value->cast<std::shared_ptr<tensor::Tensor>>();
  auto hashmap_size = hashmap_tensor->shape_c()[0];
  auto hashmap_data_type = hashmap_tensor->data_type();
  for (auto &ele : param_pair_list) {
    auto host_tensor_value = ele.second->default_param();
    auto host_tensor = host_tensor_value->cast<std::shared_ptr<tensor::Tensor>>();
    auto cache_tensor_value = ele.first->default_param();
    auto cache_tensor = cache_tensor_value->cast<std::shared_ptr<tensor::Tensor>>();

    // bind host, cache, hashmap
    host_tensor->set_cache_enable(true);
    host_tensor->set_hashmap_tensor_ptr(hashmap_tensor);
    host_tensor->set_cache_tensor_ptr(cache_tensor);

    // init cache tensor data
    auto host_shape = host_tensor->shape_c();
    auto cache_shape = cache_tensor->shape_c();
    if (host_shape.size() != 2 && host_shape.size() != 2 && host_shape[1] != cache_shape[1]) {
      MS_LOG(EXCEPTION) << "Got host shape and cache shape invalid."
                        << "host shape:" << host_shape << ", cache shape:" << cache_shape;
    }
    auto host_data_max_size = static_cast<size_t>(host_tensor->Size());
    auto cache_data_max_size = static_cast<size_t>(cache_tensor->Size());
    if (hashmap_data_type == TypeId::kNumberTypeInt32) {
      MemCopyFromHostToCache<int32_t>(hashmap_tensor->data_c(), host_tensor->data_c(), cache_tensor->data_c(),
                                      host_data_max_size, cache_data_max_size, LongToSize(hashmap_size),
                                      LongToSize(host_shape[1]));
    } else if (hashmap_data_type == TypeId::kNumberTypeInt64) {
      MemCopyFromHostToCache<int32_t>(hashmap_tensor->data_c(), host_tensor->data_c(), cache_tensor->data_c(),
                                      host_data_max_size, cache_data_max_size, LongToSize(hashmap_size),
                                      LongToSize(host_shape[1]));
    } else {
      MS_LOG(ERROR) << "Hashmap dtype only suppotr int32, in64.";
    }
  }
}

template <typename T>
void InitHashMapData(void *data, const int64_t host_size, const int64_t cache_size, const size_t hashmap_size,
                     const size_t byte_size) {
  MS_LOG(INFO) << "Start init hashmap data.";
  MS_EXCEPTION_IF_NULL(data);
  HashmapEntry<T> *hashmap_data = static_cast<HashmapEntry<T> *>(data);
  MS_EXCEPTION_IF_NULL(hashmap_data);
  int ret = memset_s(hashmap_data, byte_size, 0, byte_size);
  if (ret != EOK) {
    MS_LOG(EXCEPTION) << "Memset failed.";
  }
  std::vector<T> host_range;
  host_range.reserve(static_cast<T>(host_size));
  for (int64_t i = 0; i < host_size; ++i) {
    host_range.emplace_back(static_cast<T>(i));
  }
#if defined(__APPLE__) || defined(_MSC_VER)
  std::random_device rd;
  std::mt19937 rng(rd());
  std::shuffle(host_range.begin(), host_range.end(), rng);
#else
  std::random_shuffle(host_range.begin(), host_range.end());
#endif
  size_t size = static_cast<size_t>(cache_size);
  size_t hashmap_count = 0;
  for (size_t i = 0; i < size; ++i) {
    auto random_key = host_range[i];
    auto entry = HashFunc(random_key, hashmap_size);
    size_t count = 1;
    while (!hashmap_data[entry].IsEmpty() && !hashmap_data[entry].IsKey(random_key)) {
      count += 1;
      entry = (entry + 1) % static_cast<T>(hashmap_size);
    }
    if (hashmap_data[entry].IsEmpty()) {
      hashmap_count++;
      hashmap_data[entry].key_ = random_key;
      hashmap_data[entry].value_ = SizeToInt(i);
      hashmap_data[entry].step_ = kInitStep;
      hashmap_data[entry].tag_ = SizeToInt(count);
    }
  }
  MS_LOG(INFO) << "Hashmap init success, with " << hashmap_count << " / " << hashmap_size;
}

AnfNodePtr InitHashMap(const FuncGraphPtr &func_graph, const int64_t host_size, const int64_t cache_size,
                       TypeId type_id) {
  // init new tensor
  size_t hashmap_size = static_cast<size_t>(cache_size * kEmptyRate);
  std::vector<int64_t> host_shape{static_cast<int64_t>(hashmap_size), 4};
  auto new_tensor = std::make_shared<tensor::Tensor>(type_id, host_shape);
  size_t byte_size = new_tensor->Size();
  if (type_id == TypeId::kNumberTypeInt64) {
    InitHashMapData<int64_t>(new_tensor->data_c(), host_size, cache_size, hashmap_size, byte_size);
  } else {
    InitHashMapData<int32_t>(new_tensor->data_c(), host_size, cache_size, hashmap_size, byte_size);
  }
  ParamInfoPtr new_param_info = std::make_shared<ParamInfo>();
  std::string hashmap_name = "cache_hashmap";
  new_param_info->set_name(hashmap_name);
  new_tensor->set_param_info(new_param_info);
  return func_graph->AddFvParameter(hashmap_name, new_tensor);
}

AnfNodePtr InitStep(const FuncGraphPtr &func_graph, TypeId type_id) {
  std::vector<int64_t> host_shape{1};
  auto new_tensor = std::make_shared<tensor::Tensor>(type_id, host_shape);
  ParamInfoPtr new_param_info = std::make_shared<ParamInfo>();
  std::string step_name = "cache_step";
  new_param_info->set_name(step_name);
  new_tensor->set_param_info(new_param_info);
  return func_graph->AddFvParameter(step_name, new_tensor);
}

AnfNodePtr CreateMapCacheIdx(const FuncGraphPtr &func_graph, const AnfNodePtr &indices,
                             const ParamMap &cache_host_params_map) {
  auto iter = cache_host_params_map.begin();
  int64_t cache_size = iter->first->abstract()->GetShapeTrack()->cast<abstract::ShapePtr>()->shape()[0];
  int64_t host_size = iter->second->abstract()->GetShapeTrack()->cast<abstract::ShapePtr>()->shape()[0];
  auto indices_type = indices->Type();
  auto indices_element_type = indices_type->cast<mindspore::TensorTypePtr>()->element();
  auto indices_type_id = indices_element_type->type_id();
  auto hashmap = InitHashMap(func_graph, host_size, cache_size, indices_type_id);
  auto step = InitStep(func_graph, indices_type_id);
  auto max_num = NewValueNode(MakeValue(host_size));
  auto hashmap_param = hashmap->cast<ParameterPtr>();
  BindAndInitCacheTensor(cache_host_params_map, hashmap_param);
  // add rank_id
  int64_t offset_value = 0;
  std::string rank_id_str = common::GetEnv("RANK_ID");
  if (!rank_id_str.empty()) {
    int64_t rank_id = atoi(rank_id_str.c_str());
    offset_value = rank_id * host_size;
  }
  auto offset = NewValueNode(MakeValue(offset_value));
  auto max_num_imm = std::make_shared<Int64Imm>(host_size);
  auto max_num_abstract_scalar = std::make_shared<abstract::AbstractScalar>(max_num_imm);
  max_num->set_abstract(max_num_abstract_scalar);
  auto offset_imm = std::make_shared<Int64Imm>(offset_value);
  auto offset_abstract_scalar = std::make_shared<abstract::AbstractScalar>(offset_imm);
  offset->set_abstract(offset_abstract_scalar);

  PrimitivePtr map_cache_primitive = prim::kPrimMapCacheIdx;
  map_cache_primitive->set_attr(kAttrPrimitiveTarget, MakeValue("CPU"));
  std::vector<AnfNodePtr> map_cache_nodes{NewValueNode(map_cache_primitive), hashmap, indices, step, max_num, offset};
  auto map_cache_idx = func_graph->NewCNode(map_cache_nodes);

  auto indices_ori_shp = indices->Shape();
  auto indices_shp = indices_ori_shp->cast<abstract::ShapePtr>();
  ShapeVector shape(indices_shp->shape().size(), -1);

  auto cache_idx = std::make_shared<abstract::AbstractTensor>(indices_element_type, indices_shp);
  auto old_emb_idx =
    std::make_shared<abstract::AbstractTensor>(indices_element_type, std::make_shared<abstract::Shape>(shape));
  auto miss_emb_idx =
    std::make_shared<abstract::AbstractTensor>(indices_element_type, std::make_shared<abstract::Shape>(shape));
  auto swap_emb_idx =
    std::make_shared<abstract::AbstractTensor>(indices_element_type, std::make_shared<abstract::Shape>(shape));

  std::vector<std::shared_ptr<abstract::AbstractBase>> elements = {cache_idx, old_emb_idx, miss_emb_idx, swap_emb_idx};
  auto abstract = std::make_shared<abstract::AbstractTuple>(elements);
  map_cache_idx->set_abstract(abstract);
  return map_cache_idx;
}

AnfNodePtr CreateTupleGetItem(const FuncGraphPtr &func_graph, const AnfNodePtr &input, size_t index) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto idx = NewValueNode(SizeToLong(index));
  MS_EXCEPTION_IF_NULL(idx);
  auto imm = std::make_shared<Int64Imm>(SizeToLong(index));
  auto abstract_scalar = std::make_shared<abstract::AbstractScalar>(imm);
  idx->set_abstract(abstract_scalar);
  auto tuple_getitem = func_graph->NewCNode({NewValueNode(prim::kPrimTupleGetItem), input, idx});
  auto input_abstract_tuple = dyn_cast<abstract::AbstractTuple>(input->abstract());
  auto tuple_getitem_abstract = input_abstract_tuple->elements()[index];
  tuple_getitem->set_abstract(tuple_getitem_abstract);
  return tuple_getitem;
}

void CreateTupleGetItems(const FuncGraphPtr &func_graph, const AnfNodePtr &input, std::vector<AnfNodePtr> *outputs) {
  auto input_abstract_tuple = dyn_cast<abstract::AbstractTuple>(input->abstract());
  auto size = input_abstract_tuple->elements().size();
  MS_EXCEPTION_IF_NULL(outputs);
  for (size_t i = 0; i < size; ++i) {
    (*outputs).emplace_back(CreateTupleGetItem(func_graph, input, i));
  }
}

AnfNodePtr CreateEmbeddingLookup(const FuncGraphPtr &graph, AnfNodePtr params, AnfNodePtr indices) {
  MS_EXCEPTION_IF_NULL(graph);
  PrimitivePtr emb_lookup_primitive = std::make_shared<Primitive>(kEmbeddingLookupOpName);
  emb_lookup_primitive->set_attr(kAttrPrimitiveTarget, MakeValue("CPU"));
  ValueNodePtr offset_value_node = NewValueNode(static_cast<int64_t>(0));
  std::vector<AnfNodePtr> emb_lookup_nodes{NewValueNode(emb_lookup_primitive), params, indices, offset_value_node};
  auto emb_lookup = graph->NewCNode(emb_lookup_nodes);
  return emb_lookup;
}

AnfNodePtr CreateCacheSwapTable(const FuncGraphPtr &graph, ParameterPtr cache_table, AnfNodePtr swap_cache_idx,
                                AnfNodePtr miss_value) {
  MS_EXCEPTION_IF_NULL(graph);
  PrimitivePtr cache_swap_table_primitive = std::make_shared<Primitive>(kCacheSwapTableOpName);
  std::vector<AnfNodePtr> cache_swap_table_nodes{NewValueNode(cache_swap_table_primitive), cache_table, swap_cache_idx,
                                                 miss_value};
  auto cache_swap_table = graph->NewCNode(cache_swap_table_nodes);
  return cache_swap_table;
}

AnfNodePtr CreateUpdateCache(const FuncGraphPtr &graph, ParameterPtr params, AnfNodePtr old_emb_idx,
                             AnfNodePtr old_value) {
  MS_EXCEPTION_IF_NULL(graph);
  PrimitivePtr update_cache_primitive = std::make_shared<Primitive>(kUpdateCacheOpName);
  update_cache_primitive->set_attr(kAttrPrimitiveTarget, MakeValue("CPU"));

  auto params_ori_shp = params->Shape();
  MS_EXCEPTION_IF_NULL(params_ori_shp);
  auto params_shp = params_ori_shp->cast<abstract::ShapePtr>();
  MS_EXCEPTION_IF_NULL(params_shp);
  auto params_shape = params_shp->shape();
  auto max_size = params_shape[0];
  auto max_size_node = NewValueNode(MakeValue(max_size));
  auto max_num_imm = std::make_shared<Int64Imm>(max_size);
  auto max_num_abstract_scalar = std::make_shared<abstract::AbstractScalar>(max_num_imm);
  max_size_node->set_abstract(max_num_abstract_scalar);

  std::vector<AnfNodePtr> update_cache_nodes{NewValueNode(update_cache_primitive), params, old_emb_idx, old_value,
                                             max_size_node};
  auto update_cache = graph->NewCNode(update_cache_nodes);
  return update_cache;
}

NodePairList CreateEmbSwapUpdate(const FuncGraphPtr &graph, ParamMap param_pair_list,
                                 const AnfNodePtrList &map_cache_idx_node_outputs) {
  MS_EXCEPTION_IF_NULL(graph);
  NodePairList node_pair_list;
  for (auto &ele : param_pair_list) {
    auto emb_lookup = CreateEmbeddingLookup(graph, ele.second, map_cache_idx_node_outputs[2]);
    auto cache_swap_table = CreateCacheSwapTable(graph, ele.first, map_cache_idx_node_outputs[3], emb_lookup);
    auto update_cache = CreateUpdateCache(graph, ele.second, map_cache_idx_node_outputs[1], cache_swap_table);
    node_pair_list.emplace_back(std::make_pair(cache_swap_table, update_cache));
  }
  return node_pair_list;
}

void CreateControlDepend(const FuncGraphPtr &main_graph, const AnfNodePtr &prior_node, const AnfNodePtr &behind_node) {
  // Create control depend
  MS_EXCEPTION_IF_NULL(main_graph);
  auto manager = main_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  AnfNodePtrList cd_inputs = {NewValueNode(prim::kPrimDepend), behind_node, prior_node};
  auto depend_cnode = main_graph->NewCNode(cd_inputs);
  if (!manager->Replace(behind_node, depend_cnode)) {
    MS_LOG(EXCEPTION) << behind_node->DebugString() << ", replace node failed.";
  }
}

AnfNodePtr CreateDepend(const FuncGraphPtr &graph, const std::vector<AnfNodePtr> &invalid_nodes,
                        const AnfNodePtr &patron_node) {
  MS_EXCEPTION_IF_NULL(graph);
  std::vector<AnfNodePtr> make_tuple_list{NewValueNode(prim::kPrimMakeTuple)};
  std::copy(invalid_nodes.begin(), invalid_nodes.end(), std::back_inserter(make_tuple_list));
  auto make_tuple = graph->NewCNode(make_tuple_list);
  std::vector<AnfNodePtr> depend_list{NewValueNode(prim::kPrimDepend), patron_node, make_tuple};
  auto depend_cnode = graph->NewCNode(depend_list);
  depend_cnode->set_abstract(patron_node->abstract());
  return depend_cnode;
}

CNodePtrList FindSparseGatherV2WithCache(const CNodePtrList &cnodes, const ParamSet &param_set) {
  size_t cnodes_size = cnodes.size();
  CNodePtrList sparse_gather_v2_with_cache;
  for (size_t i = 0; i < cnodes_size; ++i) {
    if (IsPrimitiveCNode(cnodes[i], prim::kPrimSparseGatherV2) ||
        IsPrimitiveCNode(cnodes[i], prim::kPrimEmbeddingLookup)) {
      auto load_node = cnodes[i]->input(1);
      if (IsPrimitiveCNode(load_node, prim::kPrimCast)) {
        load_node = load_node->cast<CNodePtr>()->input(1);
      }
      if (IsPrimitiveCNode(load_node, prim::kPrimLoad)) {
        auto param_node = load_node->cast<CNodePtr>()->input(1)->cast<ParameterPtr>();
        if (param_set.find(param_node) != param_set.end()) {
          sparse_gather_v2_with_cache.push_back(cnodes[i]);
        } else {
          MS_LOG(EXCEPTION) << "EmbeddingLookup can't not support cache and no cache in the same graph.";
        }
      }
    }
  }
  if (sparse_gather_v2_with_cache.empty()) {
    MS_LOG(EXCEPTION) << "Can not find SparseGatherV2 with cache param.";
  }

  auto indices = sparse_gather_v2_with_cache[0]->input(2);
  for (auto &ele : sparse_gather_v2_with_cache) {
    if (ele->input(2) != indices) {
      MS_LOG(EXCEPTION) << "SparseGatherV2 which with cache param  have different indices!.";
    }
  }
  return sparse_gather_v2_with_cache;
}

AnfNodePtr FindGatherV2FromSparseGatherV2(const FuncGraphPtr &graph, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(graph);
  AnfNodePtrList gatherv2_nodes;
  auto user_set = graph->manager()->node_users()[node];
  for (auto &ele : user_set) {
    if (IsPrimitiveCNode(ele.first, prim::kPrimGather)) {
      gatherv2_nodes.emplace_back(ele.first);
    }
  }
  if (gatherv2_nodes.size() != 1) {
    MS_LOG(EXCEPTION) << "SparseGatherV2 with cache can only used by one of gatherv2, but got "
                      << gatherv2_nodes.size();
  }
  return gatherv2_nodes[0];
}

AnfSet FindNoRefParams(const FuncGraphPtr &graph) {
  AnfSet no_ref_params;
  auto params = graph->parameters();
  for (auto &anf_param : params) {
    auto param = anf_param->cast<ParameterPtr>();
    if (!param->has_default()) {
      MS_LOG(INFO) << param->DebugString() << " has no default";
      no_ref_params.insert(anf_param);
    }
  }
  return no_ref_params;
}

void RemoveOriginParamFromSet(const CNodePtr &unique_node, AnfSet *no_ref_params) {
  std::queue<CNodePtr> que;
  que.push(unique_node);
  while (!que.empty()) {
    auto node = que.front();
    que.pop();
    auto node_inputs = node->inputs();
    for (auto &input : node_inputs) {
      if (input->isa<CNode>()) {
        que.push(input->cast<CNodePtr>());
      } else if (input->isa<Parameter>()) {
        size_t num = no_ref_params->erase(input);
        if (num > 0) {
          MS_LOG(INFO) << "Erase unique_node input from set success.";
          return;
        }
      }
    }
  }
  MS_LOG(EXCEPTION) << "Can not find any parameter that use by Unique.";
}

AnfNodePtr CreateOutputNodeParam(const FuncGraphPtr &graph, const AnfNodePtr &ori_input, const std::string &name) {
  auto ori_input_type = ori_input->Type();
  auto ori_input_element_type = ori_input_type->cast<mindspore::TensorTypePtr>()->element();
  auto ori_input_type_id = ori_input_element_type->type_id();
  auto ori_input_shp = ori_input->Shape();
  auto input_shp = ori_input_shp->cast<abstract::ShapePtr>();
  auto input_shape = input_shp->shape();
  auto new_tensor = std::make_shared<tensor::Tensor>(ori_input_type_id, input_shape);
  ParamInfoPtr new_param_info = std::make_shared<ParamInfo>();
  auto new_param_name = name + "_pipe";
  new_param_info->set_name(new_param_name);
  new_tensor->set_param_info(new_param_info);
  return graph->AddFvParameter(new_param_name, new_tensor);
}

AnfMap CreateOtherPipeParams(const FuncGraphPtr &graph, const AnfSet &no_ref_params) {
  AnfMap no_ref_pipe_param_map;
  for (auto &param : no_ref_params) {
    auto ori_param = param->cast<ParameterPtr>();
    auto ori_name = ori_param->name();
    auto new_param = CreateOutputNodeParam(graph, param, ori_name);
    no_ref_pipe_param_map[param] = new_param;
  }
  return no_ref_pipe_param_map;
}

AnfNodePtr CreateAssign(const FuncGraphPtr &graph, const AnfNodePtr &res_param, const AnfNodePtr &src_param,
                        bool is_dynamic = false) {
  auto assign_prim = prim::kPrimAssign;
  if (is_dynamic) {
    assign_prim = prim::kPrimDynamicAssign;
    assign_prim->set_attr(kAttrPrimitiveTarget, MakeValue("CPU"));
  }
  std::vector<AnfNodePtr> assign_nodes{NewValueNode(assign_prim), res_param, src_param};
  auto assign_status = graph->NewCNode(assign_nodes);
  return assign_status;
}

AnfNodePtr FindCNodeOutput(const FuncGraphPtr &graph, const AnfNodePtr &node, int64_t index) {
  auto manager = graph->manager();
  auto node_users = manager->node_users()[node];
  for (auto &node_user : node_users) {
    if (IsPrimitiveCNode(node_user.first, prim::kPrimTupleGetItem)) {
      auto cnode = node_user.first->cast<CNodePtr>();
      auto node_index = cnode->input(2);
      if (node_index->isa<ValueNode>()) {
        auto value_node = node_index->cast<ValueNodePtr>();
        MS_EXCEPTION_IF_NULL(value_node);
        auto item_idx = GetValue<int64_t>(value_node->value());
        if (item_idx == index) {
          return node_user.first;
        }
      }
    }
  }
  MS_LOG(EXCEPTION) << "Can't not find " << node->DebugString() << ", outputs:" << index;
}

void ReplaceNoRefToParams(const FuncGraphPtr &graph, const AnfMap &no_ref_pipe_param_map,
                          const AnfNodePtr &cache_idx_param, const AnfNodePtr &cache_idx,
                          const AnfNodePtr &sparse_gatherv2_indices) {
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  const auto &node_users = manager->node_users();
  // add other no ref pipe param and unique index dense
  for (auto &ele : no_ref_pipe_param_map) {
    const auto &user_set = node_users.at(ele.first);
    auto assign_status = CreateAssign(graph, ele.second, ele.first);
    for (const auto &user_node : user_set) {
      CreateControlDepend(graph, user_node.first, assign_status);
    }
    if (!manager->Replace(ele.first, ele.second)) {
      MS_LOG(EXCEPTION) << "pipe param: " << ele.first->DebugString() << ", replace node failed.";
    }
  }

  // add cache idx param
  auto dynamic_assgin_status = CreateAssign(graph, cache_idx_param, cache_idx, true);
  const auto &indices_user_set = node_users.at(sparse_gatherv2_indices);
  for (const auto &user_node : indices_user_set) {
    CreateControlDepend(graph, user_node.first, dynamic_assgin_status);
  }
  if (!manager->Replace(sparse_gatherv2_indices, cache_idx_param)) {
    MS_LOG(EXCEPTION) << "cache idx param: " << cache_idx_param->DebugString() << ", replace node failed.";
  }
}

void CacheEmbeddingForTrain(const FuncGraphPtr &graph, bool is_pipe, const CNodePtrList &cnodes,
                            const CNodePtr &unique_node, const ParamSet &param_cache_enable_set) {
  MS_EXCEPTION_IF_NULL(graph);
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  size_t cnodes_size = cnodes.size();
  auto cache_host_params_map = AddCacheParameters(graph, param_cache_enable_set);
  auto param_set = MapKeysToSet(cache_host_params_map);
  ReplaceCacheParams(graph, cache_host_params_map);
  graph->set_flag(GRAPH_FLAG_CACHE_ENABLE, true);
  MS_LOG(INFO) << "Graph is set cache enable.";

  CNodePtrList sparse_gatherv2_with_cache = FindSparseGatherV2WithCache(cnodes, param_set);
  auto unique_node_output_0 = CreateTupleGetItem(graph, unique_node, 0);
  auto map_cache_idx = CreateMapCacheIdx(graph, unique_node_output_0, cache_host_params_map);

  AnfNodePtrList map_cache_idx_node_outputs;
  CreateTupleGetItems(graph, map_cache_idx, &map_cache_idx_node_outputs);

  auto node_pair_list = CreateEmbSwapUpdate(graph, cache_host_params_map, map_cache_idx_node_outputs);
  AnfNodePtrList invalid_nodes;
  auto cache_idx = map_cache_idx_node_outputs[0];
  if (!is_pipe) {
    if (!manager->Replace(sparse_gatherv2_with_cache[0]->input(2), cache_idx)) {
      MS_LOG(EXCEPTION) << "MapCacheIdx output[0] replace node failed";
    }
    for (auto &ele : node_pair_list) {
      for (auto &gather_op : sparse_gatherv2_with_cache) {
        CreateControlDepend(graph, ele.first, gather_op);
      }
      invalid_nodes.emplace_back(ele.second);
    }
  } else {
    auto cache_idx_param = CreateOutputNodeParam(graph, unique_node->input(1), std::string("cache_idx"));
    auto unique_index_reverse = FindCNodeOutput(graph, unique_node, 1);
    auto unique_index_param = CreateOutputNodeParam(graph, unique_index_reverse, std::string("index_dense"));
    auto no_ref_params = FindNoRefParams(graph);
    RemoveOriginParamFromSet(unique_node, &no_ref_params);
    auto no_ref_param_map = CreateOtherPipeParams(graph, no_ref_params);
    no_ref_param_map[unique_index_reverse] = unique_index_param;
    ReplaceNoRefToParams(graph, no_ref_param_map, cache_idx_param, cache_idx, sparse_gatherv2_with_cache[0]->input(2));
    std::transform(node_pair_list.begin(), node_pair_list.end(), std::back_inserter(invalid_nodes),
                   [](const std::pair<AnfNodePtr, AnfNodePtr> &pair) { return pair.second; });
  }
  AnfNodePtr last_node = cnodes[cnodes_size - 1];
  CNodePtr return_node;
  if (last_node->isa<CNode>()) {
    return_node = last_node->cast<CNodePtr>();
  }
  MS_EXCEPTION_IF_NULL(return_node);
  if (!IsPrimitiveCNode(return_node, prim::kPrimReturn)) {
    MS_LOG(EXCEPTION) << "The last cnode after sorting, not return cnode.";
  }
  if (return_node->inputs().size() < 2) {
    MS_LOG(EXCEPTION) << "Number of return node inputs should be greater than or equal to 2.";
  }

  auto depend_node = CreateDepend(graph, invalid_nodes, return_node->input(1));
  if (!manager->Replace(return_node->input(1), depend_node)) {
    MS_LOG(EXCEPTION) << "Depend replace node failed";
  }
}

void CacheEmbeddingForEval(const FuncGraphPtr &graph, const CNodePtrList &cnodes, const CNodePtr &unique_node,
                           const ParamSet &param_cache_enable_set) {
  MS_EXCEPTION_IF_NULL(graph);
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  graph->set_flag(GRAPH_FLAG_CACHE_ENABLE, true);
  MS_LOG(INFO) << "Graph is set cache enable.";
  // replace GatherV2 to EmbeddingLookupCPU
  auto indices = unique_node->input(1);
  auto sparse_gatherv2_with_cache = FindSparseGatherV2WithCache(cnodes, param_cache_enable_set);
  for (auto &ele : sparse_gatherv2_with_cache) {
    auto anf_ele = ele->cast<AnfNodePtr>();
    auto gatherv2 = FindGatherV2FromSparseGatherV2(graph, anf_ele);
    auto embedding_lookup = CreateEmbeddingLookup(graph, ele->input(1), indices);
    if (!manager->Replace(gatherv2, embedding_lookup)) {
      MS_LOG(EXCEPTION) << "Depend replace node failed";
    }
  }
}

void AddCacheEmbedding(const FuncGraphPtr &graph, bool is_pipe) {
  MS_EXCEPTION_IF_NULL(graph);
  std::list<CNodePtr> orders = graph->GetOrderedCnodes();
  CNodePtrList cnodes(orders.cbegin(), orders.cend());
  bool training = graph->has_flag("training");
  auto param_cache_enable_set = FindParamCacheEnable(graph);
  if (param_cache_enable_set.empty()) {
    MS_LOG(INFO) << "Parameters are all not cache enable.";
    return;
  } else {
    MS_LOG(INFO) << "Parameters have cache enable.";
  }
  if (!CheckHostCacheParamSize(param_cache_enable_set)) {
    return;
  }
  for (auto &node : cnodes) {
    if (IsPrimitiveCNode(node, prim::kPrimNPUAllocFloatStatus)) {
      MS_LOG(EXCEPTION) << "Cache embedding haven't support loss scale yet.";
    }
  }
  auto unique_cache_enable = FindUniqueCacheEnable(cnodes);
  if (unique_cache_enable.empty()) {
    MS_LOG(WARNING) << "Parameters have cache enable, but not find Unique op cache enable.";
    return;
  }
  auto unique_node = unique_cache_enable[0];
  if (training) {
    // If training, create cache parameters corresponding to the host params with is cache_enable.
    // Replace the host params. Create hashmap then insert MapCacheIdx op after Unique with has 'cache_enable' attr.
    // Bind hashmap tensor ptr and cache tensor ptr to host tensor, so that we can flush values
    // from cache to host in each epoch end.
    // Create EmbeddingLookup(CPU), CacheSwapTable(Ascend), UpdateCache(CPU) for each pair of params, in order to
    // flush miss values to cache params and write back old values to host params.
    // If no use pipe in training, EmbeddingLookup and CacheSwapTable must execute before SparseGatherV2, so add
    // ControlDepend between them. And add Depend for UpdateCache op and ControlDepnd op to add nodes into graph.
    // If use pipe in training, create parameters for no ref param such as labels and MapCacheIdx output[0] and
    // Unique output[1], in each step, it will train the data from last step, so that can hide the time of Unique
    // and other cpu kernels. So in the first step, it's fake data.
    CacheEmbeddingForTrain(graph, is_pipe, cnodes, unique_node, param_cache_enable_set);
  } else {
    // If eval, Use EmbeddingLookup(CPU) op to replace GatherV2.
    // The network is the same as Host-Device mode.
    CacheEmbeddingForEval(graph, cnodes, unique_node, param_cache_enable_set);
  }
}
}  // namespace parallel
}  // namespace mindspore
