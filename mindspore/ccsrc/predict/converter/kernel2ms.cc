/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "predict/converter/kernel2ms.h"
#include <algorithm>
#include "ir/anf.h"
#include "predict/converter/lite_model/op_attr_packer.h"
#include "mindspore/ccsrc/operator/ops.h"

namespace mindspore {
namespace executor {
Kernel2Ms &Kernel2Ms::GetInstance() {
  static Kernel2Ms instance;
  return instance;
}

bool Kernel2Ms::SetMemResue() const {
  MS_LOG(INFO) << "MemResue start";
  return true;
}

bool Kernel2Ms::SetAllTensors(const TensorCachePtr &tensor_cache, SubGraphDefT *ms_graph) {
  if (tensor_cache == nullptr || ms_graph == nullptr) {
    return false;
  }
  const std::unordered_map<int, std::vector<ExTensorPtr>> &cachedTensors = tensor_cache->GetCachedTensor();
  size_t total_size = 0;
  if (cachedTensors.empty()) {
    return false;
  }
  for (auto &iter : cachedTensors) {
    auto ex_tensors = iter.second;
    total_size += ex_tensors.size();
  }
  ms_graph->allTensors.resize(total_size);
  for (auto &iter : cachedTensors) {
    for (auto &ex_tensor : iter.second) {
      std::unique_ptr<TensorDefT> ms_tensor(new TensorDefT());
      auto device_tensor_tmp = ex_tensor->device_tensor_ptr_;
      auto device_d_type = device_tensor_tmp->data_type();
      ms_tensor->dataType = predict::utils::GetMSDataType(device_d_type);
      auto device_shape = device_tensor_tmp->shape();
      ms_tensor->dims.clear();
      if (device_shape.empty()) {
        ms_tensor->dims.push_back(1);
      } else {
        ms_tensor->dims.assign(device_shape.begin(), device_shape.end());
      }
      std::string format_str = device_tensor_tmp->device_info().format_;
      ms_tensor->format = predict::utils::GetMsFormat(format_str);
      ms_tensor->offset = 0;
      auto stable = ex_tensor->stable_;
      if (stable == INPUTDATA || stable == CONSTANT || stable == WEIGHTS) {
        ms_tensor->refCount = MS_MAX_REFCOUNT;
      } else {
        ms_tensor->refCount = 0;
      }
      ms_graph->allTensors[IntToSize(ex_tensor->index_)] = std::move(ms_tensor);
    }
  }
  return true;
}

bool Kernel2Ms::SetGraphOutputIdx(const KernelGraphPtr &kernel_graph_ptr, const TensorCachePtr &tensor_cache,
                                  SubGraphDefT *ms_graph, AllOutputTensors *all_output_tensors) {
  MS_EXCEPTION_IF_NULL(tensor_cache);
  MS_EXCEPTION_IF_NULL(ms_graph);
  MS_EXCEPTION_IF_NULL(all_output_tensors);
  auto out_nodes = kernel_graph_ptr->outputs();
  if (out_nodes.empty()) {
    return false;
  }
  // maybe need to judge out_nodes is real && output must be CNode
  for (size_t i = 0; i < out_nodes.size(); ++i) {
    std::vector<AnfNodePtr> real_inputs_link;
    std::vector<size_t> real_output_idx_link;
    GetRealInpoutsPtr(out_nodes[i], &real_inputs_link, &real_output_idx_link);
    if (real_inputs_link.empty()) {
      MS_LOG(INFO) << "this graph output node is vitural node, has no real input";
      continue;
    }
    for (size_t k = 0; k < real_inputs_link.size(); ++k) {
      int key = node_indexs_[out_nodes[i].get()];
      auto ex_tensor_list = tensor_cache->findTensor(key);
      if (ex_tensor_list.empty()) {
        MS_LOG(INFO) << "SetGraphOutputIdx do not add Extensor ";
        continue;
      }
      auto ex_tensor = ex_tensor_list[real_output_idx_link[k]];
      ex_tensor_list.clear();
      ms_graph->outputIndex.push_back(ex_tensor->index_);
    }
  }
  return true;
}

bool Kernel2Ms::SetOpOutputIdx(const CNodePtr &c_node_ptr, const TensorPtr &output_tensor,
                               const TensorCachePtr &tensor_cache, int ref_count, size_t order_index, OpDefT *ms_node) {
  MS_EXCEPTION_IF_NULL(c_node_ptr);
  MS_EXCEPTION_IF_NULL(output_tensor);
  MS_EXCEPTION_IF_NULL(ms_node);
  MS_EXCEPTION_IF_NULL(tensor_cache);
  if (!predict::utils::FindNodeInMap(node_indexs_, c_node_ptr)) {
    MS_LOG(ERROR) << "can not find any pk_key in inited node_indexs map";
    return false;
  }
  int tensor_key = node_indexs_[c_node_ptr.get()];
  auto host_shape = AnfAlgo::GetOutputInferShape(c_node_ptr, order_index);
  std::vector<int> tensor_shape;
  (void)std::transform(host_shape.begin(), host_shape.end(), std::back_inserter(tensor_shape), SizeToInt);
  int outputIndex = tensor_cache->addExTensor(tensor_key, output_tensor, ref_count, tensor_shape, KERNEL);
  ms_node->outputIndex.push_back(outputIndex);
  return true;
}

void Kernel2Ms::GetRealInpoutsPtr(const AnfNodePtr &node, std::vector<AnfNodePtr> *real_inputs,
                                  std::vector<size_t> *real_output_idx) {
  MS_EXCEPTION_IF_NULL(real_inputs);
  MS_EXCEPTION_IF_NULL(real_output_idx);
  size_t default_idx = 0;
  if (node->isa<CNode>()) {
    auto c_node = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(c_node);
    std::string c_node_name = GetCNodeFuncName(c_node);
    if (c_node_name == prim::kPrimTupleGetItem->name()) {
      auto v_node = c_node->inputs()[kTupleGetItemIndex]->cast<ValueNodePtr>();
      MS_EXCEPTION_IF_NULL(v_node);
      default_idx = IntToSize(GetValue<int>(v_node->value()));
      real_inputs->push_back(c_node->inputs()[1]);
      real_output_idx->push_back(default_idx);
      return;
    } else if (c_node_name == prim::kPrimDepend->name()) {
      GetRealInpoutsPtr(c_node->inputs()[1], real_inputs, real_output_idx);
      return;
    } else if (c_node_name == prim::kPrimMakeTuple->name()) {
      for (auto &in : c_node->inputs()) {
        GetRealInpoutsPtr(in, real_inputs, real_output_idx);
      }
      return;
    } else {
      real_inputs->push_back(node);
      real_output_idx->push_back(default_idx);
    }
  } else if (node->isa<Parameter>()) {
    real_inputs->push_back(node);
    real_output_idx->push_back(default_idx);
  } else if (node->isa<ValueNode>()) {
    real_inputs->push_back(node);
    real_output_idx->push_back(default_idx);
  }
}

bool Kernel2Ms::SetOpInputIdx(const CNodePtr &c_node_ptr, const TensorCachePtr &tensor_cache, OpDefT *ms_node) {
  MS_EXCEPTION_IF_NULL(c_node_ptr);
  MS_EXCEPTION_IF_NULL(tensor_cache);
  MS_EXCEPTION_IF_NULL(ms_node);
  for (size_t i = 1; i < c_node_ptr->inputs().size(); ++i) {
    std::vector<AnfNodePtr> real_inputs;
    std::vector<size_t> real_output_idx;
    GetRealInpoutsPtr(c_node_ptr->inputs()[i], &real_inputs, &real_output_idx);
    if (real_inputs.empty()) {
      MS_LOG(INFO) << "kernel has no inputs: " << c_node_ptr.get() << " input size[%lu]" << c_node_ptr->inputs().size();
      continue;
    }
    for (size_t j = 0; j < real_inputs.size(); ++j) {
      int key = node_indexs_[real_inputs[j].get()];
      std::vector<ExTensorPtr> ex_tensor_list = tensor_cache->findTensor(key);
      if (ex_tensor_list.empty()) {
        continue;
      }
      ExTensorPtr ex_tensor_ptr = ex_tensor_list[real_output_idx[j]];
      ex_tensor_list.clear();
      ms_node->inputIndex.push_back(ex_tensor_ptr->index_);
    }
  }
  return true;
}

void Kernel2Ms::TransformGraphIndx() {
  // transform index && anfnodeptr
  if (node_indexs_.empty()) {
    MS_LOG(EXCEPTION) << "node_indexs_ not ininted";
  }
  for (auto &item : node_indexs_) {
    index_nodes_[item.second] = item.first;
  }
}

bool Kernel2Ms::InitGraphInputsIndx(const KernelGraphPtr &kernel_graph_ptr) {
  MS_EXCEPTION_IF_NULL(kernel_graph_ptr);
  auto input_nodes = kernel_graph_ptr->inputs();
  if (input_nodes.empty()) {
    return false;
  }
  for (const auto &input_node : input_nodes) {
    if (input_node->isa<Parameter>()) {
      if (!predict::utils::FindNodeInMap(node_indexs_, input_node)) {
        // init every parameter node
        node_indexs_[input_node.get()] = graph_index_;
        graph_index_++;
      }
    } else {
      MS_LOG(INFO) << "This node is anfnode, no need to handle, continue. node info: " << input_node->ToString();
      continue;
    }
  }
  MS_LOG(DEBUG) << "inputs GraphIndex: " << graph_index_;
  return true;
}

bool Kernel2Ms::InitGraphValueNodesIndx(const KernelGraphPtr &kernel_graph_ptr) {
  MS_EXCEPTION_IF_NULL(kernel_graph_ptr);
  if (kernel_graph_ptr->value_nodes().empty()) {
    return false;
  }
  for (auto &item : kernel_graph_ptr->value_nodes()) {
    if (item.first->isa<ValueNode>()) {
      auto value_node = item.first->cast<ValueNodePtr>();
      MS_EXCEPTION_IF_NULL(value_node);
      if (value_node == nullptr) {
        MS_LOG(WARNING) << "value_node is nullptr";
        return false;
      }
      if (value_node->value() == nullptr) {
        MS_LOG(ERROR) << "Constant value is  null.";
        return false;
      }
      if (!value_node->value()->isa<tensor::Tensor>()) {
        continue;
      }
      if (!predict::utils::FindNodeInMap(node_indexs_, item.first)) {
        // init node
        auto node_ptr = item.first;
        node_indexs_[node_ptr.get()] = graph_index_;
        graph_index_++;
      }
    }
  }
  return true;
}

bool Kernel2Ms::InitGraphOpsIndx(const KernelGraphPtr &kernel_graph_ptr) {
  MS_EXCEPTION_IF_NULL(kernel_graph_ptr);
  auto kernels = kernel_graph_ptr->execution_order();
  if (kernels.empty()) {
    MS_LOG(WARNING) << "this graph has no kernel";
    return false;
  }
  for (size_t i = 0; i < kernels.size(); ++i) {
    // for each kernel's inputs foreach real_input
    if (kernels[i]->isa<CNode>()) {
      if (!predict::utils::FindNodeInMap(node_indexs_, kernels[i])) {
        // init node
        node_indexs_[kernels[i].get()] = graph_index_;
        graph_index_++;
      }
    }
  }
  return true;
}

bool Kernel2Ms::InitGraphOutputsIndx(const KernelGraphPtr &kernel_graph_ptr) {
  MS_EXCEPTION_IF_NULL(kernel_graph_ptr);
  // graph output && their inputs should link together
  auto out_nodes = kernel_graph_ptr->outputs();
  if (out_nodes.empty()) {
    MS_LOG(ERROR) << "this graph has no outputs";
    return false;
  }
  for (auto &item : out_nodes) {
    if (!predict::utils::FindNodeInMap(node_indexs_, item)) {
      node_indexs_[item.get()] = graph_index_;
      graph_index_++;
    }
  }
  return true;
}

bool Kernel2Ms::InitGraphIndx(const KernelGraphPtr &kernel_graph_ptr) {
  MS_EXCEPTION_IF_NULL(kernel_graph_ptr);
  // only parameter
  if (!InitGraphInputsIndx(kernel_graph_ptr)) {
    return false;
  }
  // init value node
  if (!InitGraphValueNodesIndx(kernel_graph_ptr)) {
    return false;
  }
  // init op
  if (!InitGraphOpsIndx(kernel_graph_ptr)) {
    return false;
  }
  // init Graphoutput attention: out_put nodes have inputs
  return InitGraphOutputsIndx(kernel_graph_ptr);
}

bool Kernel2Ms::SetGraphInputTensors(const KernelGraphPtr &kernel_graph_ptr, const TensorCachePtr &tensor_cache,
                                     SubGraphDefT *ms_graph) {
  MS_EXCEPTION_IF_NULL(tensor_cache);
  MS_EXCEPTION_IF_NULL(ms_graph);
  MS_EXCEPTION_IF_NULL(kernel_graph_ptr);
  if (convert_mode_ == kConvertUnused) {
    return false;
  }
  if (kernel_graph_ptr->inputs().empty()) {
    return false;
  }
  for (const auto &input_node : kernel_graph_ptr->inputs()) {
    if (input_node->isa<Parameter>()) {
      ParameterPtr pk_node = std::dynamic_pointer_cast<Parameter>(input_node);
      TensorPtr device_tensor;
      if (convert_mode_ == kConvertCpuMode) {
        device_tensor = predict::utils::GetParaCpuTensor(input_node);
      } else {
        device_tensor = predict::utils::GetParaAscendTensor(input_node);
      }
      if (device_tensor == nullptr) {
        return false;
      }
      ExTensorType node_type;
      if (AnfAlgo::IsParameterWeight(pk_node)) {
        node_type = WEIGHTS;
      } else {
        node_type = INPUTDATA;
      }
      if (!predict::utils::FindNodeInMap(node_indexs_, input_node)) {
        MS_LOG(WARNING) << "can not find any pk_key in inited node_indexs map";
        return false;
      }
      auto pk_key = node_indexs_[input_node.get()];
      all_output_tensors_[pk_key].push_back(device_tensor);
      int nodeRefCount = SizeToInt(AnfAlgo::GetOutputTensorNum(input_node));
      int nodeInputIdx =
        tensor_cache->addExTensor(pk_key, device_tensor, nodeRefCount, device_tensor->shape(), node_type);
      if (!AnfAlgo::IsParameterWeight(pk_node)) {
        ms_graph->inputIndex.push_back(nodeInputIdx);
        all_input_idxs_.push_back(nodeInputIdx);
      } else {
        input_weight_idxs_.push_back(nodeInputIdx);
        all_input_idxs_.push_back(nodeInputIdx);
      }
    }
  }
  return true;
}

bool Kernel2Ms::SetGraphValueTensors(const KernelGraphPtr &kernel_graph_ptr, const TensorCachePtr &tensor_cache) {
  MS_EXCEPTION_IF_NULL(kernel_graph_ptr);
  MS_EXCEPTION_IF_NULL(tensor_cache);
  for (auto &item : kernel_graph_ptr->value_nodes()) {
    if (item.first->isa<ValueNode>()) {
      auto const_node = item.first->cast<ValueNodePtr>();
      auto tensor_constant = predict::utils::GetValueTensor(const_node);
      if (tensor_constant == nullptr) {
        continue;
      }
      if (!predict::utils::FindNodeInMap(node_indexs_, item.first)) {
        MS_LOG(WARNING) << "can not find any pk_key in inited node_indexs map";
        return false;
      }
      int constant_key = node_indexs_[(item.first).get()];
      all_output_tensors_[constant_key].push_back(tensor_constant);
      auto shape = tensor_constant->shape();
      (void)tensor_cache->addExTensor(constant_key, tensor_constant, 0, shape, CONSTANT);
    }
  }
  return true;
}

bool Kernel2Ms::SetGraphOpTensors(const KernelGraphPtr &kernel_graph_ptr, const TensorCachePtr &tensor_cache,
                                  SubGraphDefT *ms_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph_ptr);
  MS_EXCEPTION_IF_NULL(tensor_cache);
  MS_EXCEPTION_IF_NULL(ms_graph);
  auto kernels = kernel_graph_ptr->execution_order();
  if (kernels.empty()) {
    MS_LOG(ERROR) << "this graph has no kernels";
    return false;
  }
  for (auto &kernel : kernels) {
    if (!predict::utils::FindNodeInMap(node_indexs_, kernel)) {
      MS_LOG(ERROR) << "can not find any pk_key in inited node_indexs map";
      return false;
    }
    auto kernel_key = node_indexs_[kernel.get()];
    std::unique_ptr<OpDefT> ms_node(new OpDefT);
    ms_node->name = kernel->fullname_with_scope();
    ms_node->fmkType = mindspore::predict::FmkType_CAFFE;
    auto c_name = AnfAlgo::GetCNodeName(kernel);
    auto fun = predict::convert::OpAttrFactory::GetInstance()->GetPackFun(c_name);
    if (fun == nullptr) {
      MS_LOG(WARNING) << "get node [" << kernel->fullname_with_scope() << "] attr failed.";
    } else if (!fun(kernel, ms_node.get())) {
      MS_LOG(ERROR) << "set node [" << kernel->fullname_with_scope() << "] attr failed.";
      return false;
    }
    auto output_size = AnfAlgo::GetOutputTensorNum(kernel);
    int nodeRefCount = SizeToInt(output_size);
    for (size_t j = 0; j < output_size; ++j) {
      TensorPtr device_tensor;
      if (convert_mode_ == kConvertCpuMode) {
        device_tensor = predict::utils::GetKernelCpuTensor(kernel, j);
      } else if (convert_mode_ == kConvertAscendMode) {
        device_tensor = predict::utils::GetKernelAscendTensor(kernel, j);
      }
      if (device_tensor == nullptr) {
        return false;
      }
      all_output_tensors_[kernel_key].push_back(device_tensor);
      if (!SetOpOutputIdx(kernel, device_tensor, tensor_cache, nodeRefCount, j, ms_node.get())) {
        return false;
      }
    }
    tmp_op_nodes_.emplace_back(ms_node.release());
  }
  return true;
}

bool Kernel2Ms::KernelGraph2MsGraph(const KernelGraphPtr &kernel_graph_ptr) {
  MS_EXCEPTION_IF_NULL(kernel_graph_ptr);
  graph_index_ = 0;
  all_output_tensors_.clear();
  node_indexs_.clear();
  index_nodes_.clear();
  std::unique_ptr<SubGraphDefT> sub_ms_graph(new SubGraphDefT());
  if (!InitGraphIndx(kernel_graph_ptr)) {
    return false;
  }
  TransformGraphIndx();
  tensor_cache_ptr_ = std::make_shared<TensorCache>();
  // foreach node to init it's real output tensor
  if (!SetGraphInputTensors(kernel_graph_ptr, tensor_cache_ptr_, sub_ms_graph.get())) {
    return false;
  }
  // Get KernelGraph value node
  if (!SetGraphValueTensors(kernel_graph_ptr, tensor_cache_ptr_)) {
    return false;
  }
  // Get KernelGraph apply_kernel && add opNode
  if (!SetGraphOpTensors(kernel_graph_ptr, tensor_cache_ptr_, sub_ms_graph.get())) {
    return false;
  }
  // Get KernelGraph outputs
  if (!SetGraphOutputIdx(kernel_graph_ptr, tensor_cache_ptr_, sub_ms_graph.get(), &all_output_tensors_)) {
    return false;
  }
  auto kernels = kernel_graph_ptr->execution_order();
  for (size_t i = 0; i < kernels.size(); ++i) {
    auto ms_node = tmp_op_nodes_[i];
    if (!SetOpInputIdx(kernels[i], tensor_cache_ptr_, ms_node)) {
      return false;
    }
    std::unique_ptr<OpDefT> ms_node_tmp(ms_node);
    sub_ms_graph->nodes.emplace_back(std::move(ms_node_tmp));
  }
  if (!SetAllTensors(tensor_cache_ptr_, sub_ms_graph.get())) {
    return false;
  }
  if (!SetMemResue()) {
    return false;
  }
  sub_ms_graph_ = std::move(sub_ms_graph);
  sub_ms_graph_->name = "default_sub_graph";
  return true;
}

bool Kernel2Ms::CheckInputSizes(const std::vector<TensorPtr> &input_tensors,
                                const std::vector<uint32_t> &all_input_idxs) {
  if (input_tensors.size() != all_input_idxs.size()) {
    MS_LOG(EXCEPTION) << "real input tensors size:" << input_tensors.size()
                      << "not equal converted tesnors size:" << all_input_idxs.size() << "the graph has changed";
  }
  for (auto in : all_input_idxs) {
    if (in < sub_ms_graph_->allTensors.size()) {
      auto real_tensor = input_tensors[in];
      auto convert_dims = sub_ms_graph_->allTensors[in]->dims;
      auto real_dims = real_tensor->shape();
      if (real_dims.size() != convert_dims.size()) {
        return false;
      } else {
        for (size_t i = 0; i < convert_dims.size(); ++i) {
          if (convert_dims[i] != real_dims[i]) {
            return false;
          }
        }
      }
    } else {
      MS_LOG(EXCEPTION) << "index: " << in << "in all_input_idxs is valid";
    }
  }
  return true;
}

void Kernel2Ms::ReleaseContextRes() {
  tmp_op_nodes_.clear();
  node_indexs_.clear();
  index_nodes_.clear();
  tensor_cache_ptr_ = nullptr;
  all_output_tensors_.clear();
}

bool Kernel2Ms::KernelInput2MS(const std::vector<TensorPtr> &input_tensors) {
  const std::unordered_map<int, std::vector<ExTensorPtr>> &cache_tensors = tensor_cache_ptr_->GetCachedTensor();
  if (cache_tensors.empty()) {
    return false;
  }
  auto all_weights_idxs = GetAllInputWeightIdxs();
  auto all_input_idxs = GetAllInputIdxs();
  auto real_input_size = input_tensors.size();
  // check tensor size
  bool ret = CheckInputSizes(input_tensors, all_input_idxs);
  std::vector<uint32_t> match_to_rel_idxs;
  // indx order not matched,macth to it
  if (!ret) {
    for (auto idx : all_weights_idxs) {
      auto macth_idx = real_input_size - idx;
      match_to_rel_idxs.push_back(macth_idx);
    }
  } else {
    match_to_rel_idxs = all_weights_idxs;
  }
  if (match_to_rel_idxs.size() == all_weights_idxs.size()) {
    for (size_t j = 0; j < all_weights_idxs.size(); ++j) {
      auto cache_idx = all_weights_idxs[j];
      auto match_idx = match_to_rel_idxs[j];
      auto real_tensor = input_tensors[match_idx];
      auto real_size = LongToSize(real_tensor->data().nbytes());
      auto real_data = real_tensor->data_c();
      MS_EXCEPTION_IF_NULL(real_data);
      if (sub_ms_graph_->allTensors[cache_idx] != nullptr) {
        sub_ms_graph_->allTensors[cache_idx]->data.resize(real_size);
      }
      if (memcpy_s(sub_ms_graph_->allTensors[cache_idx]->data.data(), real_size, real_data, real_size) != 0) {
        MS_LOG(ERROR) << "KernelInput2MS memcpy_s failed";
        return false;
      }
    }
  }
  ReleaseContextRes();
  return true;
}

bool Kernel2Ms::SaveDeviceModel(const std::shared_ptr<GraphDefT> &new_ms_graph_ptr, const std::string &save_path_name) {
  MS_EXCEPTION_IF_NULL(new_ms_graph_ptr);
  return predict::utils::SaveDeviceModelUtil(new_ms_graph_ptr, save_path_name, sub_ms_graph_.release());
}
}  // namespace executor
}  // namespace mindspore
