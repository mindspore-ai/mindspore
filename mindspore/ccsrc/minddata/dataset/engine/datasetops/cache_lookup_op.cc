/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/engine/datasetops/cache_lookup_op.h"
#include "minddata/dataset/core/config_manager.h"
#include "minddata/dataset/include/dataset/constants.h"
#include "minddata/dataset/core/global_context.h"
#include "minddata/dataset/engine/execution_tree.h"
#ifndef ENABLE_ANDROID
#include "utils/log_adapter.h"
#else
#include "mindspore/lite/src/common/log_adapter.h"
#endif

namespace mindspore {
namespace dataset {
Status CacheLookupOp::operator()() {
  if (!sampler_) {
    return Status(StatusCode::kMDUnexpectedError, __LINE__, __FILE__,
                  "Invalid parameter, CacheLookupOp requires a sampler before it can be executed, but got nullptr.");
  }
  RETURN_IF_NOT_OK(RegisterResources());
  // Kick off the workers
  RETURN_IF_NOT_OK(
    tree_->LaunchWorkers(num_workers_, std::bind(&CacheLookupOp::WorkerEntry, this, std::placeholders::_1), "", id()));
  // required task group sync after launching workers
  TaskManager::FindMe()->Post();
  // We have to wait until the leaf op has handshake with us.
  RETURN_IF_NOT_OK(leaf_op_wp_.Wait());
  RETURN_IF_NOT_OK(FetchSamplesToWorkers());
  return Status::OK();
}
Status CacheLookupOp::WorkerEntry(int32_t worker_id) {
  TaskManager::FindMe()->Post();
  RETURN_IF_NOT_OK(FetchFromCache(worker_id));
  return Status::OK();
}
Status CacheLookupOp::ResetSampler() { return Status::OK(); }
Status CacheLookupOp::HandshakeRandomAccessOp(const RandomAccessOp *op) {
  // We act like a sampler and as a dataset op. During handshake with leaf op,
  // We must wait until the leaf op has indexed everything.
  RETURN_IF_NOT_OK(sampler_->HandshakeRandomAccessOp(op));
  // Now we notify the main thread handshake has finished.
  leaf_op_wp_.Set();
  return Status::OK();
}
Status CacheLookupOp::InitSampler() { return SamplerRT::InitSampler(); }
void CacheLookupOp::Print(std::ostream &out, bool show_all) const { CacheBase::Print(out, show_all); }
void CacheLookupOp::SamplerPrint(std::ostream &out, bool show_all) const {
  out << "\nSampler: CacheLookupOp";
  if (show_all) {
    // Call the super class for displaying any common detailed info
    SamplerRT::SamplerPrint(out, show_all);
    // Then add our own info if any
  }
}
Status CacheLookupOp::GetNextSample(TensorRow *out) {
  std::vector<row_id_type> cache_miss;
  RETURN_IF_NOT_OK(keys_miss_->Pop(0, &cache_miss));
  // Ignore the case we have no cache miss, we can't return empty samples.
  while (cache_miss.empty()) {
    RETURN_IF_NOT_OK(keys_miss_->Pop(0, &cache_miss));
  }
  // Special code for eoe
  if (cache_miss.at(0) == eoe_row_id) {
    *out = std::move(TensorRow(TensorRow::kFlagEOE));
  } else {
    std::shared_ptr<Tensor> sample_ts;
    RETURN_IF_NOT_OK(CreateSamplerTensor(&sample_ts, cache_miss.size()));
    auto idPtr = sample_ts->begin<int64_t>();
    for (size_t i = 0; i < cache_miss.size(); ++i) {
      *idPtr = cache_miss.at(i);
      ++idPtr;
    }
    *out = {sample_ts};
  }
  return Status::OK();
}
Status CacheLookupOp::RegisterResources() {
  RETURN_IF_NOT_OK(CacheBase::RegisterResources());
  RETURN_IF_NOT_OK(leaf_op_wp_.Register(tree_->AllTasks()));
  return Status::OK();
}
Status CacheLookupOp::ComputeColMap() {
  // We don't know the column map at this point unless we contact the cache server
  // to fetch the schema but the cache server may not have it at this point either.
  // So we will just return OK and let MergeOp (our parent) to handle it.
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
