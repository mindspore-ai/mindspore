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

#include "minddata/dataset/engine/ir/datasetops/source/samplers/prebuilt_sampler_ir.h"
#include "minddata/dataset/engine/datasetops/source/sampler/sampler.h"
#include "minddata/dataset/core/config_manager.h"

#ifndef ENABLE_ANDROID
#include "minddata/dataset/util/random.h"
#include "minddata/mindrecord/include/shard_distributed_sample.h"
#include "minddata/mindrecord/include/shard_operator.h"
#include "minddata/mindrecord/include/shard_pk_sample.h"
#include "minddata/mindrecord/include/shard_sample.h"
#include "minddata/mindrecord/include/shard_sequential_sample.h"
#include "minddata/mindrecord/include/shard_shuffle.h"
#endif

namespace mindspore {
namespace dataset {
// Constructor
PreBuiltSamplerObj::PreBuiltSamplerObj(std::shared_ptr<SamplerRT> sampler) : sp_(std::move(sampler)) {}

// Destructor
PreBuiltSamplerObj::~PreBuiltSamplerObj() = default;

#ifndef ENABLE_ANDROID
PreBuiltSamplerObj::PreBuiltSamplerObj(std::shared_ptr<mindrecord::ShardOperator> sampler)
    : sp_minddataset_(std::move(sampler)) {}
#endif

Status PreBuiltSamplerObj::ValidateParams() { return Status::OK(); }

Status PreBuiltSamplerObj::SamplerBuild(std::shared_ptr<SamplerRT> *const sampler) {
  Status s = BuildChildren(&sp_);
  if (s.IsOk())
    *sampler = sp_;
  else
    *sampler = nullptr;
  return s;
}

#ifndef ENABLE_ANDROID
std::shared_ptr<mindrecord::ShardOperator> PreBuiltSamplerObj::BuildForMindDataset() { return sp_minddataset_; }
#endif

std::shared_ptr<SamplerObj> PreBuiltSamplerObj::SamplerCopy() {
#ifndef ENABLE_ANDROID
  if (sp_minddataset_ != nullptr) {
    auto sampler = std::make_shared<PreBuiltSamplerObj>(sp_minddataset_);
    for (const auto &child : children_) {
      Status rc = sampler->AddChildSampler(child);
      if (rc.IsError()) MS_LOG(ERROR) << "Error in copying the sampler. Message: " << rc;
    }
    return sampler;
  }
#endif
  auto sampler = std::make_shared<PreBuiltSamplerObj>(sp_);
  for (const auto &child : children_) {
    Status rc = sampler->AddChildSampler(child);
    if (rc.IsError()) MS_LOG(ERROR) << "Error in copying the sampler. Message: " << rc;
  }
  return sampler;
}

Status PreBuiltSamplerObj::to_json(nlohmann::json *const out_json) {
  RETURN_IF_NOT_OK(sp_->to_json(out_json));
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
