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

#include "frontend/parallel/allreduce_fusion/allreduce_node.h"
#include <queue>
#include "frontend/parallel/tensor_layout/tensor_layout.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace parallel {
Status AllreduceNode::AddNext(const AllreduceNodePtr &next_node) {
  if (next_node == nullptr) {
    MS_LOG(ERROR) << "next_node is nullptr!";
    return FAILED;
  }
  (void)next_.emplace_back(next_node);
  return SUCCESS;
}

Status AllreduceNode::AddPrev(const AllreduceNodePtr &prev_node, double dist, double *max) {
  if (prev_node == nullptr) {
    MS_LOG(ERROR) << "next_node is nullptr!";
    return FAILED;
  }
  if (dist <= 0) {
    MS_LOG(ERROR) << "dist must be positive! dist: " << dist;
    return FAILED;
  }
  (void)prev_.emplace_back(prev_node);
  double add_dist = prev_node->depend_feat_size() + dist;
  depend_feat_size_ += add_dist;
  if (depend_feat_size_ > *max) {
    *max = depend_feat_size_;
  }
  std::queue<AllreduceNodePtr> next_queue;
  for (auto &next : next_) {
    next_queue.push(next);
  }
  while (!next_queue.empty()) {
    auto ele = next_queue.front();
    ele->AddDependFeatSize(add_dist);
    if (ele->depend_feat_size() > *max) {
      *max = ele->depend_feat_size();
    }
    for (auto &next : ele->next()) {
      next_queue.push(next);
    }
    next_queue.pop();
  }
  return SUCCESS;
}

Status AllreduceNode::Init(const CNodePtr &cnode_ptr) {
  if (cnode_ptr == nullptr) {
    MS_LOG(ERROR) << "cnode_ptr is nullptr!";
    return FAILED;
  }
  cnode_ptr_ = cnode_ptr;
  return SUCCESS;
}

Status AllreduceNode::AddPara(const AnfNodePtr &node_ptr) {
  if (node_ptr == nullptr) {
    MS_LOG(ERROR) << "node_ptr is nullptr!";
    return FAILED;
  }
  if (!node_ptr->isa<Parameter>()) {
    MS_LOG(ERROR) << "node_ptr is not a ParameterPtr!";
    return FAILED;
  }
  auto para_ptr = node_ptr->cast<ParameterPtr>();
  MS_EXCEPTION_IF_NULL(para_ptr);
  auto layout_ptr = para_ptr->user_data<TensorLayout>();
  if (layout_ptr == nullptr) {
    MS_LOG(ERROR) << "layout_ptr is nullptr!";
    return FAILED;
  }
  auto emplace_return = paras_.emplace(node_ptr);
  if (emplace_return.second) {
    double para_size = static_cast<double>(layout_ptr->slice_shape().size());
    curr_para_size_ += para_size;
    para_size_map_[node_ptr] = para_size;
  } else {
    MS_LOG(INFO) << "node already exist!";
  }
  return SUCCESS;
}

Status AllreduceNode::RemovePara(const AnfNodePtr &node_ptr) {
  if (node_ptr == nullptr) {
    MS_LOG(ERROR) << "node_ptr is nullptr!";
    return FAILED;
  }
  auto erase_num = paras_.erase(node_ptr);
  if (erase_num == 0) {
    MS_LOG(ERROR) << "para not find!";
    return FAILED;
  }
  curr_para_size_ -= para_size_map_[node_ptr];
  return SUCCESS;
}

void AllreduceNode::ToString() const {
  MS_LOG(INFO) << "cnode: " << cnode_ptr_->DebugString() << "para size: " << paras_.size();
  for (auto &para : paras_) {
    MS_LOG(INFO) << "para name: " << para->fullname_with_scope() << " size: " << para_size_map_.at(para);
  }
  MS_LOG(INFO) << "depend_feat_size: " << depend_feat_size_ << " curr_para_size: " << curr_para_size_;
}
}  // namespace parallel
}  // namespace mindspore
