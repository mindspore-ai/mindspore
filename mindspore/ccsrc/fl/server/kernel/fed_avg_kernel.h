/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FL_SERVER_KERNEL_FED_AVG_KERNEL_H_
#define MINDSPORE_CCSRC_FL_SERVER_KERNEL_FED_AVG_KERNEL_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <functional>
#include "backend/kernel_compiler/cpu/cpu_kernel.h"
#include "fl/server/common.h"
#include "fl/server/collective_ops_impl.h"
#include "fl/server/distributed_count_service.h"
#include "fl/server/local_meta_store.h"
#include "fl/server/kernel/aggregation_kernel.h"
#include "fl/server/kernel/aggregation_kernel_factory.h"

namespace mindspore {
namespace fl {
namespace server {
namespace kernel {
constexpr size_t kFedAvgInputsNum = 4;
// The implementation for the federated average. We do weighted average for the weights. The uploaded weights from
// FL-clients is already multiplied by its data size so only sum and division are done in this kernel.

// Pay attention that this kernel is the distributed version of federated average, which means each server node in the
// cluster in invalved in the aggragation process. So the DistributedCountService and CollectiveOpsImpl are called.
template <typename T, typename S>
class FedAvgKernel : public AggregationKernel {
 public:
  FedAvgKernel()
      : cnode_weight_idx_(0),
        weight_addr_(nullptr),
        data_size_addr_(nullptr),
        new_weight_addr_(nullptr),
        new_data_size_addr_(nullptr),
        participated_(false) {}
  ~FedAvgKernel() override = default;

  void InitKernel(const CNodePtr &kernel_node) override {
    MS_EXCEPTION_IF_NULL(kernel_node);
    std::string cnode_name = AnfAlgo::GetCNodeName(kernel_node);
    if (kNameToIdxMap.count(cnode_name) == 0 || kNameToIdxMap.at(cnode_name).count("inputs") == 0 ||
        kNameToIdxMap.at(cnode_name).at("inputs").count("weight") == 0) {
      MS_LOG(EXCEPTION) << "Can't find index info of weight for kernel " << cnode_name;
      return;
    }
    cnode_weight_idx_ = kNameToIdxMap.at(cnode_name).at("inputs").at("weight");
    std::vector<size_t> weight_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, cnode_weight_idx_);
    size_t weight_size =
      std::accumulate(weight_shape.begin(), weight_shape.end(), sizeof(T), std::multiplies<size_t>());
    size_t new_weight_size = weight_size;

    input_size_list_.push_back(weight_size);
    input_size_list_.push_back(sizeof(size_t));
    input_size_list_.push_back(new_weight_size);
    input_size_list_.push_back(sizeof(size_t));

    auto weight_node =
      AnfAlgo::VisitKernelWithReturnType(AnfAlgo::GetInputNode(kernel_node, cnode_weight_idx_), 0).first;
    MS_EXCEPTION_IF_NULL(weight_node);
    name_ = cnode_name + "." + weight_node->fullname_with_scope();
    first_cnt_handler_ = [&](std::shared_ptr<ps::core::MessageHandler>) {
      std::unique_lock<std::mutex> lock(weight_mutex_);
      if (!participated_) {
        ClearWeightAndDataSize();
      }
    };
    last_cnt_handler_ = [&](std::shared_ptr<ps::core::MessageHandler>) {
      MS_ERROR_IF_NULL_WO_RET_VAL(weight_addr_);
      MS_ERROR_IF_NULL_WO_RET_VAL(data_size_addr_);
      MS_ERROR_IF_NULL_WO_RET_VAL(weight_addr_->addr);
      MS_ERROR_IF_NULL_WO_RET_VAL(data_size_addr_->addr);
      T *weight_addr = reinterpret_cast<T *>(weight_addr_->addr);
      size_t weight_size = weight_addr_->size;
      S *data_size_addr = reinterpret_cast<S *>(data_size_addr_->addr);
      if (!CollectiveOpsImpl::GetInstance().AllReduce<T>(weight_addr, weight_addr, weight_size / sizeof(T))) {
        MS_LOG(ERROR) << "Federated average allreduce failed.";
        return;
      }
      if (!CollectiveOpsImpl::GetInstance().AllReduce<S>(data_size_addr, data_size_addr, 1)) {
        MS_LOG(ERROR) << "Federated average allreduce failed.";
        return;
      }
      if (data_size_addr[0] == 0) {
        MS_LOG(ERROR) << "After AllReduce, the data size is 0.";
        return;
      }
      LocalMetaStore::GetInstance().put_value(kCtxFedAvgTotalDataSize, data_size_addr[0]);
      for (size_t i = 0; i < weight_size / sizeof(T); i++) {
        weight_addr[i] /= data_size_addr[0];
      }
      done_ = true;
      return;
    };
    GenerateReuseKernelNodeInfo();
    DistributedCountService::GetInstance().RegisterCounter(name_, done_count_, {first_cnt_handler_, last_cnt_handler_});
    return;
  }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override {
    if (inputs.size() != kFedAvgInputsNum) {
      MS_LOG(ERROR) << "The inputs number of FedAvgKernel should be 4, but got " << inputs.size();
      return false;
    }
    MS_ERROR_IF_NULL_W_RET_VAL(inputs[0], false);
    MS_ERROR_IF_NULL_W_RET_VAL(inputs[1], false);
    MS_ERROR_IF_NULL_W_RET_VAL(inputs[2], false);
    MS_ERROR_IF_NULL_W_RET_VAL(inputs[3], false);
    MS_ERROR_IF_NULL_W_RET_VAL(inputs[0]->addr, false);
    MS_ERROR_IF_NULL_W_RET_VAL(inputs[1]->addr, false);
    MS_ERROR_IF_NULL_W_RET_VAL(inputs[2]->addr, false);
    MS_ERROR_IF_NULL_W_RET_VAL(inputs[3]->addr, false);

    std::unique_lock<std::mutex> lock(weight_mutex_);
    // The weight and new_weight values should be multiplied by clients already, so we don't need to do multiplication
    // again.
    T *weight_addr = reinterpret_cast<T *>(inputs[0]->addr);
    S *data_size_addr = reinterpret_cast<S *>(inputs[1]->addr);
    T *new_weight_addr = reinterpret_cast<T *>(inputs[2]->addr);
    S *new_data_size_addr = reinterpret_cast<S *>(inputs[3]->addr);
    if (accum_count_ == 0) {
      ClearWeightAndDataSize();
    }

    MS_LOG(DEBUG) << "Iteration: " << LocalMetaStore::GetInstance().curr_iter_num() << " launching FedAvgKernel for "
                  << name_ << " new data size is " << new_data_size_addr[0] << ", current total data size is "
                  << data_size_addr[0];
    for (size_t i = 0; i < inputs[2]->size / sizeof(T); i++) {
      weight_addr[i] += new_weight_addr[i];
    }
    data_size_addr[0] += new_data_size_addr[0];
    lock.unlock();

    accum_count_++;
    participated_ = true;
    return DistributedCountService::GetInstance().Count(
      name_, std::to_string(DistributedCountService::GetInstance().local_rank()) + "_" + std::to_string(accum_count_));
  }

  void Reset() override {
    accum_count_ = 0;
    done_ = false;
    participated_ = false;
    DistributedCountService::GetInstance().ResetCounter(name_);
    return;
  }

  bool IsAggregationDone() override { return done_; }

  void SetParameterAddress(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                           const std::vector<AddressPtr> &outputs) {
    weight_addr_ = inputs[0];
    data_size_addr_ = inputs[1];
    new_weight_addr_ = inputs[2];
    new_data_size_addr_ = inputs[3];
    return;
  }

  bool ReInitForScaling() override {
    DistributedCountService::GetInstance().RegisterCounter(name_, done_count_, {first_cnt_handler_, last_cnt_handler_});
    return true;
  }

  bool ReInitForUpdatingHyperParams(size_t aggr_threshold) override {
    done_count_ = aggr_threshold;
    if (!DistributedCountService::GetInstance().ReInitCounter(name_, done_count_)) {
      MS_LOG(ERROR) << "Reinitializing counter for " << name_ << " failed.";
      return false;
    }
    return true;
  }

 private:
  void GenerateReuseKernelNodeInfo() override {
    MS_LOG(INFO) << "FedAvg reuse 'weight' of the kernel node.";
    // Only the trainable parameter is reused for federated average.
    (void)reuse_kernel_node_inputs_info_.insert(std::make_pair(kWeight, cnode_weight_idx_));
    return;
  }

  // In some cases, the Launch method is not called and the weights involved in AllReduce should be set to 0.
  void ClearWeightAndDataSize() {
    MS_ERROR_IF_NULL_WO_RET_VAL(weight_addr_);
    MS_ERROR_IF_NULL_WO_RET_VAL(data_size_addr_);
    MS_ERROR_IF_NULL_WO_RET_VAL(weight_addr_->addr);
    MS_ERROR_IF_NULL_WO_RET_VAL(data_size_addr_->addr);
    int ret = memset_s(weight_addr_->addr, weight_addr_->size, 0x00, weight_addr_->size);
    if (ret != 0) {
      MS_LOG(ERROR) << "memset_s error, errorno(" << ret << ")";
      return;
    }
    ret = memset_s(data_size_addr_->addr, data_size_addr_->size, 0x00, data_size_addr_->size);
    if (ret != 0) {
      MS_LOG(ERROR) << "memset_s error, errorno(" << ret << ")";
      return;
    }
    return;
  }

  MessageCallback first_cnt_handler_;
  MessageCallback last_cnt_handler_;

  // The trainable parameter index of the kernel node which is parsed from the frontend func_graph.
  size_t cnode_weight_idx_;

  // The address pointer of the inputs.
  AddressPtr weight_addr_;
  AddressPtr data_size_addr_;
  AddressPtr new_weight_addr_;
  AddressPtr new_data_size_addr_;

  // Whether the kernel's Launch method is called.
  bool participated_;

  // The kernel could be called concurrently so we need lock to ensure threadsafe.
  std::mutex weight_mutex_;
};
}  // namespace kernel
}  // namespace server
}  // namespace fl
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FL_SERVER_KERNEL_FED_AVG_KERNEL_H_
