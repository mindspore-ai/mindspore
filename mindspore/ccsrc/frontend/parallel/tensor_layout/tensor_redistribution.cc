/**
 * Copyright 2019-2024 Huawei Technologies Co., Ltd
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

#include "frontend/parallel/tensor_layout/tensor_redistribution.h"
#include <functional>
#include <numeric>
#include <memory>
#include <set>
#include <utility>
#include <algorithm>
#include <string>
#include "frontend/parallel/status.h"
#include "frontend/parallel/ops_info/ops_utils.h"
#include "frontend/parallel/graph_util/graph_utils.h"
#include "frontend/parallel/tensor_layout/shape_util.h"
#include "frontend/parallel/step_parallel_utils.h"
#include "frontend/parallel/tensor_layout/prime_generator.h"
#include "frontend/parallel/tensor_layout/layout_utils.h"

namespace mindspore {
namespace parallel {
Status TensorRedistribution::MakeFromToLayout(const TensorLayout &from, const TensorLayout &to) {
  auto from_layout = from.LayoutForRedistribution();
  auto to_layout = to.LayoutForRedistribution();
  if (virtual_rank_ >= 0) {
    from_origin_ = from_layout;
    to_origin_ = to_layout;
    virtual_rank_list_ = {virtual_rank_};
    return SUCCESS;
  }
  if (from.GetVirtualRank().size() == to.GetVirtualRank().size()) {
    from_origin_ = from_layout;
    to_origin_ = to_layout;
    virtual_rank_list_ = from.GetVirtualRank();
    return SUCCESS;
  }
  if (from.GetVirtualRank().size() == 1) {
    auto device_matrix = from_layout.device_arrangement_origin().array();
    device_matrix.push_back(to.GetVirtualRank().size());
    virtual_rank_list_ = to.GetVirtualRank();
    to_origin_ = to_layout;
    if (!from_layout.tensor_map_before().empty()) {
      auto new_tensor_map = from_layout.tensor_map_before();
      std::for_each(new_tensor_map.begin(), new_tensor_map.end(), [](auto &inner_vec) {
        std::for_each(inner_vec.begin(), inner_vec.end(), [](auto &val) {
          if (val >= 0) {
            val++;
          }
        });
      });
      return from_origin_.InitFromExtendVector(device_matrix, new_tensor_map, from_layout.tensor_shape_before().array(),
                                               false, false);
    }
    auto new_map = from_layout.origin_tensor_map().array();
    std::transform(new_map.begin(), new_map.end(), new_map.begin(),
                   [](const auto &val) { return val >= 0 ? val + 1 : val; });
    return from_origin_.InitFromVector(device_matrix, new_map, from_layout.tensor_shape().array());
  }
  if (to.GetVirtualRank().size() == 1) {
    auto device_matrix = to_layout.device_arrangement_origin().array();
    device_matrix.push_back(from.GetVirtualRank().size());
    virtual_rank_list_ = from.GetVirtualRank();
    from_origin_ = from_layout;
    if (!to_layout.tensor_map_before().empty()) {
      auto new_tensor_map = to_layout.tensor_map_before();
      std::for_each(new_tensor_map.begin(), new_tensor_map.end(), [](auto &inner_vec) {
        std::for_each(inner_vec.begin(), inner_vec.end(), [](auto &val) {
          if (val >= 0) {
            val++;
          }
        });
      });
      return to_origin_.InitFromExtendVector(device_matrix, new_tensor_map, to_layout.tensor_shape_before().array(),
                                             false, false);
    }
    auto new_map = to_layout.origin_tensor_map().array();
    std::transform(new_map.begin(), new_map.end(), new_map.begin(),
                   [](const auto &val) { return val >= 0 ? val + 1 : val; });
    return to_origin_.InitFromVector(device_matrix, new_map, to_layout.tensor_shape().array());
  }
  MS_LOG(ERROR) << "The from layout sharding micro interleaved num:" << from.GetVirtualRank().size()
                << " dose not match the to layout sharding micro interleaved num:" << to.GetVirtualRank().size();
  return FAILED;
}

Status TensorRedistribution::Init(const TensorLayout &from, const TensorLayout &to, const RankList &dev_list) {
  if (MakeFromToLayout(from, to) != SUCCESS) {
    MS_LOG(ERROR) << "Make from_layout and to_layout failed.";
    return FAILED;
  }
  this->is_dynamic_shape_ = CheckDynamicShape(from, to);
  if (this->is_dynamic_shape_) {
    // Dynamic info of func_graph should be considered.
    MS_LOG(INFO) << "LayoutTransfer inited with dynamic shape.";
    this->from_origin_no_assembled_ = this->from_origin_;
    this->to_origin_no_assembled_ = this->to_origin_;
    Status ret = this->AssembleStaticTensorShape(this->from_origin_no_assembled_, this->to_origin_no_assembled_,
                                                 &this->from_origin_, &this->to_origin_);
    if (ret != Status::SUCCESS) {
      return ret;
    }
    this->is_assembled_static_shape_ = true;
  }
  const Shape from_origin_shape = from_origin_.tensor_shape().array();
  const Shape to_origin_shape = to_origin_.tensor_shape().array();
  bool is_from_dyn = std::find(from_origin_shape.begin(), from_origin_shape.end(), -1) != from_origin_shape.end();
  bool is_to_dyn = std::find(to_origin_shape.begin(), to_origin_shape.end(), -1) != to_origin_shape.end();
  if (!is_from_dyn && !is_to_dyn && from_origin_.tensor_shape().size() != to_origin_.tensor_shape().size()) {
    MS_LOG(ERROR) << "from shape size must be equal to to shape size! from shape size is "
                  << from_origin_.tensor_shape().size() << ", to shape size is " << to_origin_.tensor_shape().size();
    MS_LOG(ERROR) << "reshape from_origin_ " << from_origin_.ToString();
    MS_LOG(ERROR) << "reshape to_origin_ " << to_origin_.ToString();
    return Status::FAILED;
  }

  if (virtual_rank_list_.size() == 1) {
    dev_list_ = dev_list;
  } else {
    for (const auto &rank : dev_list) {
      for (size_t i = 0; i < virtual_rank_list_.size(); ++i) {
        dev_list_.push_back(int64_t(rank * virtual_rank_list_.size() + i));
      }
    }
  }
  from_ = from_origin_.SqueezeShape();
  to_ = to_origin_.SqueezeShape();

  this->is_inited_ = true;
  return Status::SUCCESS;
}

Status TensorRedistribution::CalculateFromTensorShape(Shape *from_shape, const Array &from_factors,
                                                      const Shape &to_shape, const Array &to_factors) {
  if (from_shape->size() != from_factors.GetDimSize() || to_shape.size() != to_factors.GetDimSize()) {
    MS_LOG(ERROR) << "Shape size is not equal to factor size.";
    return Status::FAILED;
  }
  int64_t to_layout_added_factor = GetLeastFactorWithoutConstDims(to_shape, to_factors);
  int64_t to_layout_const_size = GetTensorSize(to_shape);
  int64_t from_layout_const_size = GetTensorSize(*from_shape);
  if (to_layout_const_size > from_layout_const_size && to_layout_const_size % from_layout_const_size == 0) {
    to_layout_added_factor *= (to_layout_const_size / from_layout_const_size);
  }
  MS_LOG(INFO) << "from_shape=" << (*from_shape) << ", from_factors=" << from_factors.array()
               << ", to_shape=" << to_shape << ", to_factors=" << to_factors.array()
               << ", to_layout_added_factor=" << to_layout_added_factor;
  if (from_layout_const_size > to_layout_const_size && from_layout_const_size % to_layout_const_size == 0) {
    int64_t merged_const_factor = from_layout_const_size / to_layout_const_size;
    // Existed dim in from_layout already satisfy to_layout_added_factor.
    if (to_layout_added_factor > merged_const_factor && to_layout_added_factor % merged_const_factor == 0) {
      to_layout_added_factor /= merged_const_factor;
    }
    if (to_layout_added_factor == 1) {
      to_layout_added_factor = -1;
    }
  }
  bool strict_mode = UseStrictMode(*from_shape, to_shape);
  std::vector<int64_t> known_dims;
  (void)std::copy_if(from_shape->begin(), from_shape->end(), std::back_inserter(known_dims),
                     [](int64_t dim) -> bool { return dim != -1; });
  constexpr size_t INVALID_TENSOR_RANK = 9999;
  size_t last_dyn_dim = INVALID_TENSOR_RANK;
  auto last_dyn_dim_iter = std::find(from_shape->rbegin(), from_shape->rend(), -1);
  if (last_dyn_dim_iter != from_shape->rend()) {
    last_dyn_dim = from_shape->size() - (last_dyn_dim_iter - from_shape->rbegin()) - 1;
  }
  for (size_t i = 0; i < from_shape->size(); ++i) {
    if (from_shape->at(i) != -1) {
      continue;
    }
    int64_t prime_num = PrimeGenerator::GetInstance()->GetCoprimeNum(known_dims);
    if (prime_num == -1) {
      return Status::FAILED;
    }
    (*from_shape)[i] = prime_num * from_factors.GetDimByIdx(i);
    if (strict_mode && from_shape->at(i) < to_factors.GetDimByIdx(i) &&
        from_factors.GetDimByIdx(i) < to_factors.GetDimByIdx(i)) {
      int64_t common_factor = std::gcd(from_factors.GetDimByIdx(i), to_factors.GetDimByIdx(i));
      int64_t left_factor = to_factors.GetDimByIdx(i) / common_factor;
      (*from_shape)[i] *= left_factor;
      if (to_layout_added_factor >= left_factor && to_layout_added_factor % left_factor == 0) {
        to_layout_added_factor /= left_factor;
      }
      if (to_layout_added_factor < left_factor) {
        to_layout_added_factor = -1;
      }
    }
    if (strict_mode && from_shape->at(i) >= to_factors.GetDimByIdx(i) &&
        from_shape->at(i) % to_factors.GetDimByIdx(i) != 0) {
      (*from_shape)[i] *= to_factors.GetDimByIdx(i);
      if (to_layout_added_factor >= to_factors.GetDimByIdx(i) &&
          to_layout_added_factor % to_factors.GetDimByIdx(i) == 0) {
        to_layout_added_factor /= to_factors.GetDimByIdx(i);
      }
    }
    if (i == last_dyn_dim && to_layout_added_factor > 0) {
      if (from_shape->at(i) % to_layout_added_factor != 0) {
        (*from_shape)[i] *= to_layout_added_factor;
      }
      to_layout_added_factor = -1;
    }
    known_dims.emplace_back(from_shape->at(i));
    MS_LOG(DEBUG) << "Replace  " << i << " with value " << from_shape->at(i) << " prime " << prime_num;
    if (!RecordDimsChange(i, from_shape->at(i), &this->from_dims_replace_memo_)) {
      MS_LOG(ERROR) << "Index " << i << " conflicts.";
      return Status::FAILED;
    }
  }
  return Status::SUCCESS;
}

static std::vector<int64_t> EnumerateArray(int64_t base_n, size_t length = 100) {
  static std::map<int64_t, std::vector<int64_t>> enum_numbers;
  if (enum_numbers.find(base_n) != enum_numbers.end()) {
    return enum_numbers.at(base_n);
  }
  std::vector<int64_t> array(length);
  for (size_t i = 1; i < length + 1; ++i) {
    array[i - 1] = base_n * SizeToLong(i);
  }
  return array;
}

Status TensorRedistribution::CalculateToTensorShapeUsingEnumeration(const Shape &from_tsr_shape, Shape *to_tsr_shape,
                                                                    const Array &factors) {
  int64_t src_element_size = GetTensorSize(from_tsr_shape);
  int64_t dst_element_size = GetTensorSize(*to_tsr_shape);
  if (src_element_size % dst_element_size != 0) {
    MS_LOG(ERROR) << "Calculate to tensor shape failed. Tensor shape size is not matched.";
    return Status::FAILED;
  }
  const int64_t dyn_dim_val = -1;
  int64_t dyn_axis_cnt = std::count(to_tsr_shape->begin(), to_tsr_shape->end(), dyn_dim_val);
  int64_t left_size = src_element_size / dst_element_size;

  if (dyn_axis_cnt == 0) {
    if (left_size != 1) {
      MS_LOG(ERROR) << "Calculate to tensor shape failed. Tensor shape size is not matched.";
      return Status::FAILED;
    }
    return Status::SUCCESS;
  }

  if (dyn_axis_cnt == 1) {
    /**
     * Case1:
     * from: c1, -1(32), c3, c4; to: c1/2, -1(32)*c3, c4
     */
    auto iter = std::find(to_tsr_shape->begin(), to_tsr_shape->end(), dyn_dim_val);
    size_t index = static_cast<size_t>(iter - to_tsr_shape->begin());
    if (left_size % factors.GetDimByIdx(index) != 0) {
      MS_LOG(ERROR) << "Generate static shape failed, the shape cannot be divided by factor. dim=" << left_size
                    << ", factor=" << factors.GetDimByIdx(index);
      return Status::FAILED;
    }
    (*iter) = left_size;
    if (!RecordDimsChange(index, left_size, &this->to_dims_replace_memo_)) {
      MS_LOG(ERROR) << "Index " << iter - to_tsr_shape->begin() << " conflicts.";
      return Status::FAILED;
    }
    return Status::SUCCESS;
  } else {
    /**
     * Case2:
     * from: -1(16), c1, c2; to: -1(2), c1*c2/2, 2*-1(8)
     * Solution:
     * -1(16), c1*c2/2, 2
     *      A,       B, c1*c2/2, 2
     *      A, c1*c2/2, 2* B
     *
     * A*B=3*16 && A%2=0 && B%8=0
     */
    std::vector<std::vector<int64_t>> enum_numbers;
    for (size_t i = 0; i < to_tsr_shape->size(); ++i) {
      if (to_tsr_shape->at(i) == -1) {
        std::vector<int64_t> array = EnumerateArray(factors.GetDimByIdx(i));
        enum_numbers.emplace_back(array);
      }
    }
    std::vector<int64_t> candidates(enum_numbers.size());
    if (!SolveCombination(from_tsr_shape, 0, enum_numbers, 0, left_size, &candidates)) {
      MS_LOG(ERROR) << "Not supported for now.";
      return Status::FAILED;
    }
    size_t cnt = 0;
    for (size_t i = 0; i < to_tsr_shape->size(); ++i) {
      if (to_tsr_shape->at(i) == -1) {
        (*to_tsr_shape)[i] = candidates[cnt++];
        if (!RecordDimsChange(i, to_tsr_shape->at(i), &this->to_dims_replace_memo_)) {
          MS_LOG(ERROR) << "Index " << i << " conflicts.";
          return Status::FAILED;
        }
      }
    }
    return Status::SUCCESS;
  }
}

void CalculateToTensorShapeForOneDynamicAxis(const Shape &from_shape, const Shape &origin_to_shape, Shape *to_shape) {
  Shape from_shape_divisor(from_shape);
  size_t dynamic_axis = 0;
  for (size_t i = 0; i < origin_to_shape.size(); ++i) {
    int64_t dim_val = origin_to_shape[i];
    (*to_shape)[i] = dim_val;
    if (dim_val == -1) {
      dynamic_axis = i;
      continue;
    }
    for (int64_t &from_dim_val : from_shape_divisor) {
      if (dim_val == 1) {
        break;
      }
      int64_t f = std::gcd(dim_val, from_dim_val);
      from_dim_val /= f;
      dim_val /= f;
    }
  }
  (*to_shape)[dynamic_axis] = GetTensorSize(from_shape_divisor);
  MS_LOG(INFO) << "to_shape=" << (*to_shape) << ", from_shape_divisor=" << from_shape_divisor;
}

Status TensorRedistribution::CalculateToTensorShape(const Shape &from_shape, const Shape &origin_to_shape,
                                                    const Array &to_in_factors, Shape *to_shape) {
  MS_LOG(INFO) << "from_shape=" << from_shape << ", origin_to_shape=" << origin_to_shape
               << ", to_in_factors=" << to_in_factors.array();
  // Use forward and backward matching first, if failed, turn to enumeration.
  if (std::count(origin_to_shape.begin(), origin_to_shape.end(), -1) == 1) {
    CalculateToTensorShapeForOneDynamicAxis(from_shape, origin_to_shape, to_shape);
    return Status::SUCCESS;
  }
  bool flag_forward_match = ForwardMatching(from_shape, origin_to_shape, to_shape, to_in_factors);
  if (!flag_forward_match && !BackwardMatching(origin_to_shape, to_shape, to_in_factors)) {
    MS_LOG(DEBUG) << "Backward matching failed.";
    if (CalculateToTensorShapeUsingEnumeration(from_shape, to_shape, to_in_factors) != Status::SUCCESS) {
      MS_LOG(ERROR) << "Calculate to tensor shape failed trying to use enumeration method.";
      return Status::FAILED;
    }
  }
  return Status::SUCCESS;
}

Status TensorRedistribution::AssembleStaticTensorShape(const TensorLayout &from_in, const TensorLayout &to_in,
                                                       TensorLayout *new_from_layout, TensorLayout *new_to_layout) {
  Shape new_from_shape(from_in.tensor_shape().array());
  Shape original_to_shape = to_in.tensor_shape().array();
  Array from_in_factors;
  if (GetFactors(from_in, &from_in_factors) != Status::SUCCESS) {
    MS_LOG(ERROR) << "Get from_in factors failed.";
    return Status::FAILED;
  }
  Array to_in_factors;
  if (GetFactors(to_in, &to_in_factors) != Status::SUCCESS) {
    MS_LOG(ERROR) << "Get to_in factors failed.";
    return Status::FAILED;
  }
  if (CalculateFromTensorShape(&new_from_shape, from_in_factors, original_to_shape, to_in_factors) != Status::SUCCESS) {
    MS_LOG(ERROR) << "Failed to generate static shape for from_tensor layout: " << from_in.ToString();
    return Status::FAILED;
  }
  Shape new_to_shape(to_in_factors.GetDimSize(), 1);
  if (CalculateToTensorShape(new_from_shape, original_to_shape, to_in_factors, &new_to_shape)) {
    MS_LOG(ERROR) << "Failed to generate static shape for to_tensor layout: " << to_in.ToString() << std::endl
                  << "from_in layout: " << from_in.ToString() << std::endl
                  << "Already generate from_in shape: " << new_from_shape;
    return Status::FAILED;
  }
  size_t size = std::min(new_from_shape.size(), new_to_shape.size());
  if (GetTensorSize(new_from_shape) != GetTensorSize(new_to_shape)) {
    int64_t acc_scalar = 1;
    for (size_t i = 0; i < size; ++i) {
      if (new_from_shape.at(i) > new_to_shape.at(i) && new_from_shape.at(i) % new_to_shape.at(i) == 0) {
        int64_t scalar = new_from_shape.at(i) / new_to_shape.at(i);
        new_to_shape[i] = new_to_shape[i] * scalar;
        acc_scalar *= scalar;
      }
    }
    const Shape &f_in_tensor_shape = from_in.tensor_shape().array();
    auto last_dyn_dim_iter = std::find(f_in_tensor_shape.rbegin(), f_in_tensor_shape.rend(), -1);
    if (last_dyn_dim_iter != f_in_tensor_shape.rend()) {
      size_t last_dyn_dim =
        f_in_tensor_shape.size() - static_cast<size_t>(last_dyn_dim_iter - f_in_tensor_shape.rbegin()) - 1;
      new_from_shape[static_cast<size_t>(last_dyn_dim)] *= acc_scalar;
    }
  }

  // Unify shape from begin to end.
  UnifyFromAndToShape(&new_from_shape, &new_to_shape, from_in, to_in, &this->from_dims_replace_memo_);

  MS_LOG(INFO) << "new_from_shape=" << new_from_shape << ", new_to_shape=" << new_to_shape;
  if (new_from_layout->InitFromVector(from_in.device_arrangement().array(), from_in.tensor_map().array(),
                                      new_from_shape) != Status::SUCCESS) {
    MS_LOG(ERROR) << "Failed to init new from_tensor layout.";
    return Status::FAILED;
  }
  MS_LOG(DEBUG) << "Init new_from_tensor layout, origin:" << from_in.ToString()
                << ", new:" << new_from_layout->ToString();

  if (new_to_layout->InitFromVector(to_in.device_arrangement().array(), to_in.tensor_map().array(), new_to_shape) !=
      Status::SUCCESS) {
    MS_LOG(ERROR) << "Failed to init new to_tensor layout.";
    return Status::FAILED;
  }
  MS_LOG(DEBUG) << "Init new_to_layout layout, origin:" << to_in.ToString() << ", new:" << new_to_layout->ToString();

  return Status::SUCCESS;
}

bool IsVirtualDatasetNextInput(const CNodePtr &cnode, const CNodePtr &dst_cnode, size_t depth = 0) {
  if (depth >= MAX_RECURSIVE_DEPTH) {
    return false;
  }
  for (size_t j = 1; j < cnode->inputs().size(); ++j) {
    auto cur_cnode = cnode->input(j)->cast<CNodePtr>();
    if (cur_cnode == nullptr) {
      continue;
    }
    if (cur_cnode->UniqueId() == dst_cnode->UniqueId()) {
      return true;
    }
    if (IsVirtualDatasetNextInput(cur_cnode, dst_cnode, depth + 1)) {
      return true;
    }
  }
  return false;
}

CNodePtr UpdateShapeNodeInput(const CNodePtr &current_cnode, const CNodePtr &dst_cnode, size_t redistribution_index) {
  for (size_t i = redistribution_index; i < current_cnode->inputs().size(); ++i) {
    auto prev_cnode = current_cnode->input(i)->cast<CNodePtr>();
    if (prev_cnode == nullptr) {
      continue;
    }
    bool found = IsVirtualDatasetNextInput(prev_cnode, dst_cnode);
    if (found) {
      MS_LOG(INFO) << "change input to " << current_cnode->input(1)->fullname_with_scope();
      return prev_cnode;
    }
  }
  return nullptr;
}

std::pair<int64_t, AnfNodePtr> GetDimMapping(const AssembledDynamicDimsMapping &mapping, int64_t index) {
  for (const auto &iter : mapping) {
    if (SizeToLong(iter.second.first) == index) {
      return std::make_pair(iter.first, iter.second.second);
    }
  }
  MS_LOG(EXCEPTION) << "Cannot find index " << index << " in AssembledDynamicDimsMapping.";
}

void TensorRedistribution::UnifyAssembledMappingWithSqueezedFromShape() {
  AssembledDynamicDimsMapping new_mapping;
  for (const auto &iter : this->dynamic_dim_mapping_) {
    auto origin_tuple_get_item = iter.second.second;
    auto origin_tuple_get_item_cnode = origin_tuple_get_item->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(origin_tuple_get_item_cnode);
    auto func_graph = origin_tuple_get_item->func_graph();
    MS_EXCEPTION_IF_NULL(func_graph);
    auto prim_tuple_get_item = std::make_shared<Primitive>(TUPLE_GETITEM_OP);
    int64_t index = SizeToLong(iter.second.first) + 1;
    AnfNodePtrList inputs{NewValueNode(prim_tuple_get_item), origin_tuple_get_item_cnode->input(1),
                          NewValueNode(MakeValue(index))};
    auto tuple_get_item_cnode = func_graph->NewCNode(inputs);
    tuple_get_item_cnode->set_fullname_with_scope(iter.second.second->fullname_with_scope());
    prim_tuple_get_item->set_instance_name("tuple_getitem_for_value_" + std::to_string(iter.first));
    if (iter.second.second->isa<CNode>()) {
      auto raw_cnode = iter.second.second->cast<CNodePtr>();
      if (IsValueNode<Primitive>(raw_cnode->input(0))) {
        auto prim_node = raw_cnode->input(0)->cast<ValueNodePtr>();
        auto prim = GetValueNode<PrimitivePtr>(prim_node);
        prim_tuple_get_item->set_instance_name(prim->instance_name());
      }
    }
    new_mapping.insert({iter.first, {iter.second.first, tuple_get_item_cnode}});
    MS_LOG(WARNING) << "Adjust TupleGetItem for dim=" << iter.second.first << " to " << iter.second.first + 1
                    << " to replace value=" << iter.first;
  }
  this->dynamic_dim_mapping_ = new_mapping;
}

void TensorRedistribution::UnifyAssembledMappingWithSameSize(const std::set<int64_t> &index_mapping) {
  Shape from_shape = this->assembled_static_origin_from_.tensor_shape().array();
  Shape origin_slice_shape = this->assembled_static_origin_from_.slice_shape().array();
  AssembledDynamicDimsMapping new_mapping;
  for (int64_t i = SizeToLong(from_shape.size()) - 1; i >= 0; --i) {
    if (index_mapping.find(i) == index_mapping.end()) {
      continue;
    }
    auto dyn_dim = GetDimMapping(this->dynamic_dim_mapping_, i);
    int64_t real_dim_value = origin_slice_shape[i];
    new_mapping.insert({real_dim_value, {i, dyn_dim.second}});
    MS_LOG(INFO) << "insert at " << i << " with " << real_dim_value;
  }
  this->dynamic_dim_mapping_ = new_mapping;
}

void TensorRedistribution::UnifyAssembledMappingWithDiffSize(const std::set<int64_t> &index_mapping) {
  auto func_graph = this->next_cnode_->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);

  Shape from_shape = this->assembled_static_origin_from_.tensor_shape().array();
  Shape origin_slice_shape = this->assembled_static_origin_from_.slice_shape().array();
  Shape unified_from_shape = this->layout_transfer_.from_in().tensor_shape().array();
  Shape unified_slice_shape = this->layout_transfer_.from_in().slice_shape().array();

  AssembledDynamicDimsMapping new_mapping;
  // Assume length of unified_from_shape must be greater than from_shape.
  int64_t unified_offset = SizeToLong(unified_from_shape.size()) - 1;
  for (int64_t i = SizeToLong(from_shape.size()) - 1; i >= 0 && unified_offset >= 0; --i) {
    int64_t real_dim_value = origin_slice_shape[i];
    // It means it's a const dim.
    if (index_mapping.find(i) == index_mapping.end()) {
      MS_EXCEPTION_IF_CHECK_FAIL(real_dim_value >= unified_slice_shape[unified_offset] &&
                                   real_dim_value % unified_slice_shape[unified_offset] == 0,
                                 "Tensor layout tensor shape is illegal.");
      int64_t left_size = real_dim_value / unified_slice_shape[unified_offset];
      --unified_offset;
      if (left_size == 1) {
        continue;
      }
      while (left_size != 1 && unified_offset >= 0) {
        MS_EXCEPTION_IF_CHECK_FAIL(left_size % unified_slice_shape[unified_offset] == 0,
                                   "Tensor layout tensor shape is illegal, left_size is " + std::to_string(left_size) +
                                     ", factor is " + std::to_string(unified_slice_shape[unified_offset]));
        left_size = left_size / unified_slice_shape[unified_offset];
        --unified_offset;
      }
      continue;
    }
    auto dyn_dim = GetDimMapping(this->dynamic_dim_mapping_, i);
    // It means it's a dynamic dim.
    if (from_shape[i] == unified_from_shape[unified_offset]) {
      new_mapping.insert({real_dim_value, {unified_offset, dyn_dim.second}});
      MS_LOG(INFO) << "insert at " << unified_offset << " with " << real_dim_value;
      --unified_offset;
    } else if (from_shape[i] > unified_slice_shape[unified_offset] &&
               from_shape[i] % unified_slice_shape[unified_offset] == 0) {
      // left_size must be greater than 1.
      int64_t left_size = real_dim_value / unified_slice_shape[unified_offset];
      MS_EXCEPTION_IF_CHECK_FAIL(left_size >= 1, "left_size must be greater than or equal to 1.");
      int64_t divisor = real_dim_value / unified_slice_shape[unified_offset];
      if (GetPrimeFactor(unified_slice_shape[unified_offset]) != -1) {
        AnfNodePtr new_dim_node = CreateDiv(dyn_dim.second, divisor, func_graph, true, "assemble_dynamic_shape_op");
        new_mapping.insert({unified_slice_shape[unified_offset], {unified_offset, new_dim_node}});
        MS_LOG(INFO) << "insert at " << unified_offset << " with " << unified_slice_shape[unified_offset];
      } else {
        new_mapping.insert({unified_slice_shape[unified_offset], {unified_offset, dyn_dim.second}});
        MS_LOG(INFO) << "insert at " << unified_offset << " with " << unified_slice_shape[unified_offset];
      }
      --unified_offset;
      while (left_size != 1 && unified_offset >= 0) {
        left_size = left_size / unified_slice_shape[unified_offset];
        // If it's prime then add it to mapping.
        if (GetPrimeFactor(unified_slice_shape[unified_offset]) != -1) {
          new_mapping.insert({unified_slice_shape[unified_offset], {unified_offset, dyn_dim.second}});
          MS_LOG(INFO) << "insert at " << unified_offset << " with " << unified_slice_shape[unified_offset];
        } else {
          MS_LOG(INFO) << "skip at " << unified_offset << " for " << unified_slice_shape[unified_offset]
                       << ", because it's not a prime.";
        }
        --unified_offset;
      }
      if (left_size != 1 && unified_offset < 0) {
        MS_LOG(EXCEPTION) << "Tensor shape cannot be unified.";
      }
    } else {
      MS_LOG(EXCEPTION) << "Tensor shape cannot be unified.";
    }
  }
  this->dynamic_dim_mapping_ = new_mapping;
}

void TensorRedistribution::UnifyAssembledMapping() {
  // 12,10,2,2 -> 2,6,10,2,2, 12 and 10 are all dynamic.
  //  4, 6,2,2 -> 2,2, 6,2,2, 4 is static and 6 is dynamic.
  // After refactor, from_origin_ and layer_transfer_.from_in are both in static shape.
  // 1. If origin_from_shape.size > before_unified_from_shape, it means the shape is squeezed.
  //   Squeezed could be in head and also be in tail.
  // 2. If before_unified_from_shape < unified_from_shape, it means the shape is expanded.
  Shape origin_from_shape = this->from_origin_.tensor_shape().array();
  Shape origin_from_slice_shape = this->from_origin_.slice_shape().array();
  Shape before_unified_from_shape = this->assembled_static_origin_from_.tensor_shape().array();
  Shape before_unified_from_slice_shape = this->assembled_static_origin_from_.slice_shape().array();
  Shape unified_from_shape = this->layout_transfer_.from_in().tensor_shape().array();
  Shape unified_from_slice_shape = this->layout_transfer_.from_in().slice_shape().array();

  std::set<int64_t> index_mapping;
  for (const auto &iter : this->dynamic_dim_mapping_) {
    index_mapping.insert(iter.second.first);
  }
  MS_LOG(INFO) << "\norigin_from_shape=" << origin_from_shape << ", origin_from_slice_shape=" << origin_from_slice_shape
               << ", \nbefore_unified_from_shape=" << before_unified_from_shape
               << ", before_unified_from_slice_shape=" << before_unified_from_slice_shape
               << ", \nunified_from_shape=" << unified_from_shape
               << ", unified_from_slice_shape=" << unified_from_slice_shape;
  if (before_unified_from_shape.size() == origin_from_shape.size() - 1 &&
      (origin_from_shape.front() == 1 || origin_from_shape.back() == 1)) {
    // It means unified_from_shape and before_unified_from_shape are squeezed,
    // origin_from_shape has no squeezed info.
    MS_LOG(WARNING) << "before_unified_from_shape == origin_from_shape - 1.";
    this->UnifyAssembledMappingWithSqueezedFromShape();
    return;
  }
  if (unified_from_shape.size() == origin_from_shape.size()) {
    MS_LOG(WARNING) << "unified_from_shape == origin_from_shape.";
    this->UnifyAssembledMappingWithSameSize(index_mapping);
    return;
  }
  if (unified_from_shape.size() > before_unified_from_shape.size()) {
    // In this branch, it means the unified_from_shape is expanded,
    // or it's reshaped to another shape.
    MS_LOG(WARNING) << "unified_from_shape > before_unified_from_shape.";
    if (before_unified_from_shape.size() == origin_from_shape.size() - 1 &&
        (origin_from_shape.front() == 1 || origin_from_shape.back() == 1)) {
      // It means shape has been squeezed, so add one to index in mapping.
      this->UnifyAssembledMappingWithSqueezedFromShape();
    }
    this->UnifyAssembledMappingWithDiffSize(index_mapping);
    return;
  }
  MS_LOG(EXCEPTION) << "unified_from_shape.size() must be greater than before_unified_from_shape.size().";
}

void TensorRedistribution::CreateAssembledDynamicMapping(const CNodePtr &cur_cnode, const AnfNodePtr &pre_cnode,
                                                         const FuncGraphPtr &func_graph, int64_t redistribution_index) {
  MS_EXCEPTION_IF_NULL(func_graph);
  if (!this->IsAssembledStaticShape()) {
    return;
  }
  MS_LOG(INFO) << "Start to create assembled dynamic shape mapping for " << pre_cnode->fullname_with_scope() << "->"
               << cur_cnode->fullname_with_scope();
  this->dynamic_dim_mapping_.clear();

  AnfNodePtr shape_root = pre_cnode;
  if (pre_cnode->isa<CNode>() && IsPrimitiveCNode(pre_cnode, std::make_shared<Primitive>(VIRTUAL_DATA_SET))) {
    // Find VirtualDataset successor.
    auto shape_input = UpdateShapeNodeInput(cur_cnode, pre_cnode->cast<CNodePtr>(), redistribution_index);
    if (shape_input == nullptr) {
      MS_LOG(WARNING) << "Cannot find real input of shape node.";
    } else {
      shape_root = shape_input;
    }
  }
  const std::set<std::string> multi_output_op = {ARGMAXWITHVALUE, LAYER_NORM};
  if (pre_cnode->isa<CNode>() && IsSomePrimitiveList(pre_cnode->cast<CNodePtr>(), multi_output_op)) {
    shape_root = cur_cnode->input(redistribution_index);
    MS_LOG(INFO) << "Change shape_root to " << shape_root->fullname_with_scope();
  }

  ReplacementMemo from_layout_memo = this->from_dims_replace_memo_;
  Shape assembled_origin_slice_shape = this->from_origin_.slice_shape().array();
  MS_LOG(INFO) << "Start to create assembled dynamic shape mapping: " << pre_cnode->fullname_with_scope() << "->"
               << cur_cnode->fullname_with_scope() << ", shape_root=" << shape_root->fullname_with_scope()
               << ", assembled_origin_slice_shape=" << assembled_origin_slice_shape;
  // 1. New shape and set pre_cnode to its inputs.
  std::string instance_name = std::string(REDISTRIBUTION_OP) + "_" + pre_cnode->fullname_with_scope();
  auto shape_cnode = CreateShape(shape_root, func_graph, instance_name + "_get_shape");
  // 2. Create TupleGetItem node to get dim value and insert to mapping.
  for (const auto &iter : from_layout_memo) {
    int64_t dim = SizeToLong(iter.first);
    int64_t replacement = iter.second;
    MS_EXCEPTION_IF_CHECK_FAIL(replacement % assembled_origin_slice_shape[LongToSize(dim)] == 0,
                               "Slice shape is not matched.");
    MS_EXCEPTION_IF_CHECK_FAIL(LongToSize(dim) < assembled_origin_slice_shape.size(), "Slice shape is not matched.");
    replacement = assembled_origin_slice_shape[dim];
    auto prim_tuple_get_item = std::make_shared<Primitive>(TUPLE_GETITEM_OP);
    AnfNodePtrList inputs{NewValueNode(prim_tuple_get_item), shape_cnode, NewValueNode(MakeValue(dim))};
    auto tuple_get_item_cnode = func_graph->NewCNode(inputs);
    tuple_get_item_cnode->set_fullname_with_scope(std::string(REDISTRIBUTION_OP) + "_getitem");
    prim_tuple_get_item->set_instance_name(instance_name + "_getitem");
    this->dynamic_dim_mapping_.insert({replacement, {iter.first, tuple_get_item_cnode}});
    MS_LOG(INFO) << "Create TupleGetItem for dim=" << dim << " to replace value=" << replacement;
  }
}

void AppendOperatorVecStr(const OperatorVector &vec, std::string *res) {
  for (size_t i = 0; i < vec.size(); ++i) {
    res->append(vec.at(i).first);
    if (i != vec.size() - 1) {
      res->append(", ");
    }
  }
}

RedistributionOpListPtr TensorRedistribution::InferTensorRedistributionOperatorListUnExpand(bool is_cost_model) {
  MS_LOG(INFO) << "Start to infer tensor redistribution with unexpanded.";
  TensorLayout from_origin = this->from_origin_;
  TensorLayout to_origin = this->to_origin_;
  TensorLayout from_repeat = from_origin.TransferRepeatLayout();
  TensorLayout to_repeat = to_origin.TransferRepeatLayout();
  MS_LOG(DEBUG) << "reshape from_origin_ " << from_origin.ToString();
  MS_LOG(DEBUG) << "reshape to_origin_ " << to_origin.ToString();
  MS_LOG(DEBUG) << "reshape from_repeat " << from_repeat.ToString();
  MS_LOG(DEBUG) << "reshape to_repeat " << to_repeat.ToString();

  OperatorVector operator_vector;
  OutPutInfoVector output_info_vector;
  if (InferRedistribution(from_origin, from_repeat, &operator_vector, &output_info_vector, is_cost_model) ==
      Status::FAILED) {
    return nullptr;
  }
  std::string operator_vec_str;
  AppendOperatorVecStr(operator_vector, &operator_vec_str);
  MS_LOG(INFO) << "After InferRedistribution, operator_vector size: " << operator_vector.size()
               << ", operator_vector: " << operator_vec_str;
  if (from_repeat.slice_shape().array() != to_repeat.slice_shape().array()) {
    reshape_flag_ = true;
    ConstructOperator constructor;
    constructor.UpdateTensorShape(from_repeat.slice_shape().array());
    Arrangement shape = to_repeat.slice_shape();
    MS_LOG(INFO) << "from_repeat.slice_shape is not same with to_repeat.slice_shape: "
                 << "from_repeat.slice_shape=" << from_repeat.slice_shape().array()
                 << ", to_repeat.slice_shape=" << to_repeat.slice_shape().array() << ", reshape to "
                 << shape.ToString();
    if (constructor.ReshapeOP(shape.array()) == Status::FAILED) {
      return nullptr;
    } else {
      operator_vector.push_back(constructor.GetOperator());
      output_info_vector.emplace_back(std::make_pair(false, 0));
    }
  }
  if (InferRedistribution(to_repeat, to_origin, &operator_vector, &output_info_vector, is_cost_model) ==
      Status::FAILED) {
    return nullptr;
  }

  ConstructOperator constructor;
  if (from_origin_.base_slice_shape().array() != from_origin_.slice_shape().array()) {
    reshape_flag_ = true;
    constructor.UpdateTensorShape(from_origin_.base_slice_shape().array());
    Arrangement shape = from_origin_.slice_shape();
    MS_LOG(INFO) << "from_origin_.base_slice_shape is not same with from_origin_.slice_shape: "
                 << "from_origin_.base_slice_shape=" << from_origin_.base_slice_shape().array()
                 << ", from_origin_.slice_shape=" << from_origin_.slice_shape().array() << ", reshape to "
                 << shape.ToString();
    if (constructor.ReshapeOP(shape.array()) == Status::FAILED) {
      return nullptr;
    } else {
      (void)operator_vector.insert(operator_vector.cbegin(), constructor.GetOperator());
      (void)output_info_vector.insert(output_info_vector.cbegin(), std::make_pair(false, 0));
    }
  }

  if (to_origin_.slice_shape().array() != to_origin_.base_slice_shape().array()) {
    reshape_flag_ = true;
    constructor.UpdateTensorShape(to_origin_.slice_shape().array());
    Arrangement shape = to_origin_.base_slice_shape();
    MS_LOG(INFO) << "to_origin_.slice_shape is not same with to_origin_.base_slice_shape: "
                 << "to_origin_.slice_shape=" << to_origin_.slice_shape().array()
                 << ", to_origin_.base_slice_shape=" << to_origin_.base_slice_shape().array() << ", reshape to "
                 << shape.ToString();
    if (constructor.ReshapeOP(shape.array()) == Status::FAILED) {
      return nullptr;
    } else {
      (void)operator_vector.insert(operator_vector.cend(), constructor.GetOperator());
      (void)output_info_vector.insert(output_info_vector.cend(), std::make_pair(false, 0));
    }
  }

  operator_vec_str.clear();
  AppendOperatorVecStr(operator_vector, &operator_vec_str);
  MS_LOG(INFO) << "After InferRedistribution, operator_vector size: " << operator_vector.size()
               << ", operator_vector: " << operator_vec_str;
  return std::make_shared<std::pair<OperatorVector, OutPutInfoVector>>(
    std::make_pair(operator_vector, output_info_vector));
}

void GetRedistributionOperators(const RedistributionOperatorInfer &operator_infer, OperatorVector *operator_vector,
                                OutPutInfoVector *output_info_vector, OperatorList *operator_list) {
  for (const auto &op : operator_infer.operator_vector()) {
    (void)operator_vector->emplace_back(op);
  }
  for (auto info : operator_infer.output_info_vector()) {
    (void)output_info_vector->emplace_back(info);
  }
  for (const auto &opc : operator_infer.operator_list()) {
    (void)operator_list->emplace_back(opc);
  }
}

RedistributionOpListPtr TensorRedistribution::InferTensorRedistributionOperatorListForMultiDynamicReshape(
  bool is_cost_model) {
  MS_LOG(INFO) << "Start to infer tensor redistribution for multi dynamic axis reshape.";
  if (this->pre_cnode_ != nullptr && this->next_cnode_ != nullptr) {
    MS_LOG(DEBUG) << this->PrintRedistribution();
  }
  OperatorVector operator_vector;
  OutPutInfoVector output_info_vector;
  RedistributionOperatorInfer allgather_infer(this->construct_op_flag_);
  if (allgather_infer.Init(this->from_origin_no_assembled_, this->to_origin_no_assembled_.tensor_map(), this->dev_list_,
                           is_cost_model, this->is_dynamic_shape_) == Status::FAILED) {
    MS_LOG(EXCEPTION) << "Init operatorInfer failed.";
  }
  // 1. Do AllGather on dynamic axis, skip const axis?
  if (allgather_infer.MergePartialToFullForReshapeHasMultiDynamicAxis() != Status::SUCCESS) {
    MS_LOG(EXCEPTION) << "Insert AllGather for Reshape which has multi dynamic axis failed.";
  }
  GetRedistributionOperators(allgather_infer, &operator_vector, &output_info_vector, &this->operator_list_);
  // 2. Do Reshape. Const axis value should be divided later?
  ConstructOperator constructor;
  // Actually, no need to create virtual shape, store the original inputs and replace it later in replace op.
  Shape full_shape = this->to_origin_no_assembled_.tensor_shape().array();
  MS_LOG(INFO) << "before ReshapeOP, full_shape:" << full_shape;
  if (constructor.ReshapeOP(full_shape, true) == Status::FAILED) {
    MS_LOG(EXCEPTION) << "Cannot construct Reshape op for shape " << full_shape;
  }
  (void)operator_vector.emplace_back(constructor.GetOperator());
  (void)output_info_vector.emplace_back(std::make_pair(false, 0));
  // 3. Do Split, skip const axis?
  RedistributionOperatorInfer allsplit_infer(this->construct_op_flag_);
  if (allsplit_infer.Init(this->to_origin_no_assembled_, this->to_origin_no_assembled_.tensor_map(), this->dev_list_,
                          is_cost_model, this->is_dynamic_shape_) == Status::FAILED) {
    MS_LOG(ERROR) << "Init operatorInfer failed";
    return nullptr;
  }
  if (allsplit_infer.SegmentFullShapeToPartial() != Status::SUCCESS) {
    MS_LOG(EXCEPTION) << "Insert AllSplit for Reshape which has multi dynamic axis failed.";
  }
  GetRedistributionOperators(allsplit_infer, &operator_vector, &output_info_vector, &this->operator_list_);
  std::string operator_vec_str;
  AppendOperatorVecStr(operator_vector, &operator_vec_str);
  MS_LOG(INFO) << "After InferAllSplit, operator_vector size: " << operator_vector.size()
               << ", operator_vector: " << operator_vec_str;
  return std::make_shared<std::pair<OperatorVector, OutPutInfoVector>>(
    std::make_pair(operator_vector, output_info_vector));
}

RedistributionOpListPtr TensorRedistribution::InferTensorRedistributionOperatorList(bool is_cost_model) {
  MS_LOG(INFO) << "Start to infer tensor redistribution.";
  if (this->pre_cnode_ != nullptr && this->next_cnode_ != nullptr) {
    MS_LOG(DEBUG) << this->PrintRedistribution();
  }
  // Step 1: Match device arrangement between from_ and to_
  // RedistributionLayoutTransfer layout_transfer;
  // Step 0: Do dynamic shape to static shape conversion.
  // TensorRedistribution::Init() only save from and to tensor layout, and squeezed from and to layout.
  // We can change from_ and to_ in RedistributionLayoutTransfer object directly.
  // RedistributionLayoutTransfer::Init() will check whether is dynamic shape,
  // if the static shape cannot be created, reuse early process.
  Status status = this->layout_transfer_.Init(from_, to_);
  if (status != Status::SUCCESS) {
    return nullptr;
  }
  TensorLayout from_layout;
  TensorLayout to_layout;
  if (this->is_dynamic_shape_ && !this->is_assembled_static_shape_) {
    from_layout = this->layout_transfer_.from_in();
    to_layout = this->layout_transfer_.to_in();
  } else {
    // init a new layout_transfer
    // The function of assembled_static_origin_from_ is used to record layout before unify.
    // When device matrix or tensor shape is needed to unified, it could insert 1 in front of tensor shape
    // or split a dim into multi dim.
    this->assembled_static_origin_from_ = this->layout_transfer_.from_in();
    std::shared_ptr<ReshapeLayoutTransfer> ptr = this->layout_transfer_.UnifyDeviceArrangementAndTensorShape();
    if (ptr == nullptr) {
      MS_LOG(ERROR) << "Infer tensor layout return nullptr!";
      return nullptr;
    }
    this->layout_transfer_.Init(ptr->from_in(), ptr->to_in());
    if (!ptr->ExpandAble()) {
      expand_able_ = false;
      return InferTensorRedistributionOperatorListUnExpand(is_cost_model);
    }
    from_layout = ptr->from_in();
    to_layout = ptr->to_in();
  }
  MS_LOG(DEBUG) << "reshape from_layout " << from_layout.ToString();
  MS_LOG(DEBUG) << "reshape to_layout " << to_layout.ToString();
  MS_LOG(DEBUG) << "reshape from_origin_ " << from_origin_.ToString();
  MS_LOG(DEBUG) << "reshape to_origin_ " << to_origin_.ToString();
  MS_LOG(DEBUG) << "reshape from_ " << from_.ToString();
  MS_LOG(DEBUG) << "reshape to_ " << to_.ToString();

  // Step 2: Infer redistribution and insert operators
  OperatorVector operator_vector;
  OutPutInfoVector output_info_vector;
  if (InferRedistribution(from_layout, to_layout, &operator_vector, &output_info_vector, is_cost_model) !=
      Status::SUCCESS) {
    return nullptr;
  }
  //  Step 3: Infer reshape and insert operators
  if (InferReshape(from_layout, to_layout, &operator_vector, &output_info_vector) != Status::SUCCESS) {
    MS_LOG(ERROR) << "Construct Reshape operator failed!";
    return nullptr;
  }
  std::string operator_vec_str;
  AppendOperatorVecStr(operator_vector, &operator_vec_str);
  MS_LOG(INFO) << "After InferRedistribution, operator_vector size: " << operator_vector.size()
               << ", operator_vector: " << operator_vec_str;
  return std::make_shared<std::pair<OperatorVector, OutPutInfoVector>>(
    std::make_pair(operator_vector, output_info_vector));
}

std::vector<RedistributionOpListPtr> TensorRedistribution::InferTensorRedistributionOperatorVirtualGraphs() {
  std::vector<RedistributionOpListPtr> redis_list_vector;
  for (const auto &virtual_rank : virtual_rank_list_) {
    this->SetVirtualRank(virtual_rank);
    auto redis_list = this->InferTensorRedistributionOperatorList();
    if (!redis_list) {
      MS_LOG(INTERNAL_EXCEPTION) << "Infer tensor redistribution failed. from_layout:" << from_origin_.ToString()
                                 << ", to_layout:" << to_origin_.ToString();
    }
    redis_list_vector.push_back(redis_list);
  }
  return redis_list_vector;
}

bool IsSameShape(const Shape &src, const Shape &tgt) {
  if (src.size() != tgt.size()) {
    return false;
  }
  for (size_t i = 0; i < src.size(); ++i) {
    if (src[i] == -1 || tgt[i] == -1) {
      continue;
    }
    if (src[i] != tgt[i]) {
      return false;
    }
  }
  return true;
}

Shape AlignToLayoutShape(const Shape &to_origin_shape, const Shape &to_layout_shape) {
  Shape target_shape(to_origin_shape);
  auto cnt = std::count(target_shape.begin(), target_shape.end(), -1);
  if (cnt < SizeToInt(SIZE_TWO) || to_layout_shape[0] != 1 || to_layout_shape.size() - 1 != target_shape.size()) {
    return target_shape;
  }
  for (size_t i = 0; i < target_shape.size(); ++i) {
    if (target_shape[i] != -1) {
      continue;
    }
    target_shape[i] = to_layout_shape[i + 1];
  }
  return target_shape;
}

Status TensorRedistribution::OperatorListIsEmpty(ConstructOperator *constructor, OperatorVector *const operator_vector,
                                                 OutPutInfoVector *const output_info_vector) {
  if (from_origin_.base_slice_shape().array() != to_origin_.base_slice_shape().array() || keep_reshape_) {
    reshape_flag_ = true;
    constructor->UpdateTensorShape(from_origin_.base_slice_shape().array());
    Arrangement shape = to_origin_.base_slice_shape();
    MS_LOG(INFO) << "from_origin_.base_slice_shape is not same with to_origin_.base_slice_shape: "
                 << "from_origin_.base_slice_shape=" << from_origin_.base_slice_shape().array()
                 << ", to_origin_.base_slice_shape=" << to_origin_.base_slice_shape().array() << ", reshape to "
                 << shape.ToString();
    auto reshape_mode = ReshapeMode::FROM_ORIGIN_BASE_SLICE_TO_TO_ORIGIN_BASE_SLICE;
    reshape_mode = this->is_dynamic_shape_ ? reshape_mode : ReshapeMode::NO_RESHAPE;
    if (constructor->ReshapeOP(shape.array(), false, reshape_mode) == Status::FAILED) {
      return Status::FAILED;
    } else {
      (void)operator_vector->insert(operator_vector->cbegin(), constructor->GetOperator());
      (void)output_info_vector->insert(output_info_vector->cbegin(), std::make_pair(false, 0));
    }
  }
  return Status::SUCCESS;
}

Status TensorRedistribution::InferReshape(const TensorLayout &from_layout, const TensorLayout &to_layout,
                                          OperatorVector *const operator_vector,
                                          OutPutInfoVector *const output_info_vector) {
  MS_EXCEPTION_IF_NULL(operator_vector);
  MS_EXCEPTION_IF_NULL(output_info_vector);
  ConstructOperator constructor;
  if (operator_list_.empty()) {
    return OperatorListIsEmpty(&constructor, operator_vector, output_info_vector);
  }
  // 1. 需要知道哪个轴是动态的，哪个轴是常量，只比较常量轴，但是是否能保证from_origin_和from_layout的rank是一样的？
  // from_origin_是静态，那from_layout也一定是静态，如果from_origin_是动态，那from_layout也一定是动态
  // 先支持from_origin_和from_layout的rank一样的场景
  if (!IsSameShape(from_origin_.slice_shape().array(), from_layout.slice_shape().array())) {
    reshape_flag_ = true;
    constructor.UpdateTensorShape(from_origin_.slice_shape().array());
    Arrangement shape = from_layout.slice_shape();
    MS_LOG(INFO) << "from_origin.slice_shape is not same with from_layout.slice_shape: "
                 << "from_origin_.slice_shape=" << from_origin_.slice_shape().array()
                 << ", from_layout.slice_shape=" << from_layout.slice_shape().array() << ", reshape to "
                 << shape.ToString();
    auto reshape_mode = ReshapeMode::FROM_ORIGIN_SLICE_TO_FROM_LAYOUT_SLICE;
    reshape_mode = this->is_dynamic_shape_ ? reshape_mode : ReshapeMode::NO_RESHAPE;
    if (constructor.ReshapeOP(shape.array(), false, reshape_mode) == Status::FAILED) {
      return Status::FAILED;
    } else {
      // Before all-gather.
      (void)operator_vector->insert(operator_vector->cbegin(), constructor.GetOperator());
      (void)output_info_vector->insert(output_info_vector->cbegin(), std::make_pair(false, 0));
    }
  }

  if (from_origin_.base_slice_shape().array() != from_origin_.slice_shape().array()) {
    reshape_flag_ = true;
    constructor.UpdateTensorShape(from_origin_.base_slice_shape().array());
    Arrangement shape = from_origin_.slice_shape();
    MS_LOG(INFO) << "from_origin_.base_slice_shape is not same with from_origin_.slice_shape: "
                 << "from_origin_.base_slice_shape=" << from_origin_.base_slice_shape().array()
                 << ", from_origin_.slice_shape=" << from_origin_.slice_shape().array() << ", reshape to "
                 << shape.ToString();
    if (constructor.ReshapeOP(shape.array()) == Status::FAILED) {
      return Status::FAILED;
    } else {
      // Before all-gather.
      (void)operator_vector->insert(operator_vector->cbegin(), constructor.GetOperator());
      (void)output_info_vector->insert(output_info_vector->cbegin(), std::make_pair(false, 0));
    }
  }

  if (!IsSameShape(to_origin_.slice_shape().array(), to_layout.slice_shape().array())) {
    reshape_flag_ = true;
    constructor.UpdateTensorShape(to_layout.slice_shape().array());
    // If to_origin_ is all -1, it can not be reshape.
    Shape target_shape = to_origin_.slice_shape().array();
    size_t cnt = std::count(target_shape.begin(), target_shape.end(), -1);
    if (this->IsAssembledStaticShape() && cnt >= SIZE_TWO) {
      target_shape = AlignToLayoutShape(to_origin_.slice_shape().array(), to_layout.slice_shape().array());
      MS_LOG(INFO) << "update reshape target shape.";
    }
    MS_LOG(INFO) << "to_origin_.slice_shape is not same with to_layout.slice_shape: "
                 << "to_origin_.slice_shape=" << to_origin_.slice_shape().array()
                 << ", to_layout.slice_shape=" << to_layout.slice_shape().array() << ", reshape to " << target_shape;
    auto reshape_mode = ReshapeMode::TO_ORIGIN_SLICE_TO_TO_LAYOUT_SLICE;
    reshape_mode = this->is_dynamic_shape_ ? reshape_mode : ReshapeMode::NO_RESHAPE;
    if (constructor.ReshapeOP(target_shape, false, reshape_mode) == Status::FAILED) {
      return Status::FAILED;
    } else {
      // After all-gather.
      (void)operator_vector->insert(operator_vector->cend(), constructor.GetOperator());
      (void)output_info_vector->insert(output_info_vector->cend(), std::make_pair(false, 0));
    }
  }

  if (to_origin_.slice_shape().array() != to_origin_.base_slice_shape().array()) {
    reshape_flag_ = true;
    constructor.UpdateTensorShape(to_origin_.slice_shape().array());
    Arrangement shape = to_origin_.base_slice_shape();
    MS_LOG(INFO) << "to_origin_.slice_shape is not same with to_origin_.base_slice_shape: "
                 << "to_origin_.slice_shape=" << to_origin_.slice_shape().array()
                 << ", to_origin_.base_slice_shape=" << to_origin_.base_slice_shape().array() << ", reshape to "
                 << shape.ToString();
    if (constructor.ReshapeOP(shape.array()) == Status::FAILED) {
      return Status::FAILED;
    } else {
      // After all-gather.
      (void)operator_vector->insert(operator_vector->cend(), constructor.GetOperator());
      (void)output_info_vector->insert(output_info_vector->cend(), std::make_pair(false, 0));
    }
  }
  return Status::SUCCESS;
}

Status TensorRedistribution::InferRedistribution(const TensorLayout &from_layout, const TensorLayout &to_layout,
                                                 OperatorVector *const operator_vector,
                                                 OutPutInfoVector *const output_info_vector, bool is_cost_model) {
  MS_EXCEPTION_IF_NULL(operator_vector);
  MS_EXCEPTION_IF_NULL(output_info_vector);
  MS_LOG(DEBUG) << "Start to infer redistribution.";
  RedistributionOperatorInfer operator_infer(construct_op_flag_);
  if (virtual_rank_ >= 0) {
    operator_infer.SetVirtualRank(virtual_rank_);
  }
  if (operator_infer.Init(from_layout, to_layout.tensor_map(), dev_list_, is_cost_model, this->is_dynamic_shape_) ==
      Status::FAILED) {
    MS_LOG(ERROR) << "Init operatorInfer failed";
    return Status::FAILED;
  }
  if (operator_infer.InferRedistributionOperator() != Status::SUCCESS) {
    MS_LOG(ERROR) << "Infer redistribution failed";
    return Status::FAILED;
  } else {
    for (auto op : operator_infer.operator_vector()) {
      (void)operator_vector->insert(operator_vector->cend(), op);
    }
    for (auto info : operator_infer.output_info_vector()) {
      (void)output_info_vector->insert(output_info_vector->cend(), info);
    }
    for (auto opc : operator_infer.operator_list()) {
      (void)operator_list_.insert(operator_list_.cend(), opc);
    }
  }
  return Status::SUCCESS;
}

Status TensorRedistribution::RollbackToDynamicShape() {
  if (!this->IsAssembledStaticShape()) {
    return Status::FAILED;
  }
  for (auto &iter : this->from_dims_replace_memo_) {
    MS_LOG(DEBUG) << "from index=" << iter.first << ", value=" << iter.second << std::endl;
  }
  for (auto &iter : this->to_dims_replace_memo_) {
    MS_LOG(DEBUG) << "to index=" << iter.first << ", value=" << iter.second << std::endl;
  }
  MS_LOG(DEBUG) << "RollbackToDynamicShape: from_in_=" << this->from_origin_.ToString() << std::endl
                << "to_in_=" << this->to_origin_.ToString() << std::endl;
  return Status::SUCCESS;
}

Status TensorRedistribution::ComputeCost() {
  RedistributionOpListPtr redistribution_oplist_ptr = InferTensorRedistributionOperatorList(true);
  if (redistribution_oplist_ptr == nullptr) {
    MS_LOG(ERROR) << "Failure: InferTensorRedistribution failed";
    return Status::FAILED;
  }
  // Compute redistribution communication cost and computation cost
  for (auto &op_cost : operator_list_) {
    OperatorR op = op_cost.first;
    Shape slice_shape = op_cost.second;
    double prod =
      std::accumulate(slice_shape.begin(), slice_shape.end(), static_cast<double>(1.0), std::multiplies<double>());
    std::string str = op.first;
    if (str == PERMUTE_BY_AXIS && ComputePermuteCost(prod, op.second) != Status::SUCCESS) {
      return Status::FAILED;
    } else if (str == CONCAT_BY_AXIS && ComputeConcatCost(prod, op.second) != Status::SUCCESS) {
      return Status::FAILED;
    } else {
      // There is only computation cost in SplitByAxis.
      // computation cost = before_slice_shape
      computation_cost_ += prod;
      // This addition may be erroneous
      memory_cost_ += prod;
    }
  }
  if (reshape_flag()) {
    Shape prev_shape;
    if (expand_able_) {
      prev_shape = from_.slice_shape().array();
    } else {
      prev_shape = from_.tensor_shape().array();
    }
    double prev_prod =
      std::accumulate(prev_shape.begin(), prev_shape.end(), static_cast<double>(1.0), std::multiplies<double>());
    computation_cost_ += COST_FACTOR * prev_prod;
    memory_cost_ += COST_FACTOR * prev_prod;
  }
  return Status::SUCCESS;
}

Status TensorRedistribution::ComputePermuteCost(double input_size, const Shape &attrs) {
  // Since AlltoAll is a virtual operator, the expanded operators are used here to compute cost.
  // communication cost = all_gather + reduce_scatter = before_slice_shape + after_slice_shape
  if (attrs.size() < TRANSFER_PERMUTE_ARGS_SIZE) {
    MS_LOG(ERROR) << "attrs size should not be less than 5!";
    return Status::FAILED;
  }
  forward_comm_cost_ += input_size * ALLTOALL_SCALE_FACTOR;
  backward_comm_cost_ += input_size * ALLTOALL_SCALE_FACTOR;
  comm_cost_ += COST_FACTOR * input_size * ALLTOALL_SCALE_FACTOR;
  int64_t concat_dim = attrs[TRANSFER_PERMUTE_CONCAT_DIM_INDEX];
  if (concat_dim == 0) {
    // memory cost = all_gather
    computation_cost_ += input_size;
    memory_cost_ += input_size;
  } else {
    // memory cost = all_gather + split + concat
    int64_t dev_num = attrs[TRANSFER_PERMUTE_DEV_NUM_INDEX];
    computation_cost_ += (input_size + input_size * dev_num + input_size * dev_num);
    memory_cost_ += (input_size * dev_num + input_size * dev_num + input_size);
  }
  return Status::SUCCESS;
}

Status TensorRedistribution::ComputeConcatCost(double input_size, const Shape &attrs) {
  // communication cost = all_gather + reduce_scatter = before_slice_shape + after_slice_shape
  // computation cost = before_slice_shape
  if (attrs.size() < TRANSFER_CONCAT_ARGS_SIZE) {
    MS_LOG(ERROR) << "op.second size should not be less than 3!";
    return Status::FAILED;
  }
  double dev_num = attrs[TRANSFER_CONCAT_SPLIT_COUNT_INDEX];
  // here, communication cost = all_gather + reduce_scatter
  forward_comm_cost_ += input_size * dev_num * ALLGATHER_REDUCESCATTER_SCALE_FACTOR;
  backward_comm_cost_ += input_size * ALLGATHER_REDUCESCATTER_SCALE_FACTOR;
  comm_cost_ += input_size * (dev_num + 1.0) * ALLGATHER_REDUCESCATTER_SCALE_FACTOR;
  int64_t concat_dim = attrs[TRANSFER_CONCAT_TENSOR_DIM_INDEX];
  if (concat_dim == 0) {
    // computation cost = all_gather
    computation_cost_ += input_size;
    memory_cost_ += input_size * dev_num;
  } else {
    // computation cost = all_gather + split + concat
    computation_cost_ += (input_size + input_size * dev_num + input_size * dev_num);
    memory_cost_ += (input_size * dev_num + input_size * dev_num + input_size);
  }
  return Status::SUCCESS;
}
}  // namespace parallel
}  // namespace mindspore
