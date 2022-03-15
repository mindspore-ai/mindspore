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

#include "minddata/dataset/engine/ir/datasetops/source/samplers/skip_first_epoch_sampler_ir.h"
#include "minddata/dataset/engine/datasetops/source/sampler/skip_first_epoch_sampler.h"
#include "minddata/dataset/util/validators.h"

namespace mindspore {
namespace dataset {
// Constructor
SkipFirstEpochSamplerObj::SkipFirstEpochSamplerObj(int64_t start_index) : SequentialSamplerObj(start_index, 0) {}

// Destructor
SkipFirstEpochSamplerObj::~SkipFirstEpochSamplerObj() = default;

Status SkipFirstEpochSamplerObj::to_json(nlohmann::json *const out_json) {
  nlohmann::json args;
  RETURN_IF_NOT_OK(SamplerObj::to_json(&args));
  args["sampler_name"] = "SkipFirstEpochSampler";
  args["start_index"] = start_index_;
  *out_json = args;
  return Status::OK();
}

#ifndef ENABLE_ANDROID
Status SkipFirstEpochSamplerObj::from_json(nlohmann::json json_obj, std::shared_ptr<SamplerObj> *sampler) {
  RETURN_IF_NOT_OK(ValidateParamInJson(json_obj, "start_index", "SkipFirstEpochSampler"));
  int64_t start_index = json_obj["start_index"];
  *sampler = std::make_shared<SkipFirstEpochSamplerObj>(start_index);
  // Run common code in super class to add children samplers
  RETURN_IF_NOT_OK(SamplerObj::from_json(json_obj, sampler));
  return Status::OK();
}
#endif

Status SkipFirstEpochSamplerObj::SamplerBuild(std::shared_ptr<SamplerRT> *sampler) {
  // runtime sampler object
  *sampler = std::make_shared<dataset::SkipFirstEpochSamplerRT>(start_index_, 0);
  Status s = BuildChildren(sampler);
  sampler = s.IsOk() ? sampler : nullptr;
  return s;
}

std::shared_ptr<SamplerObj> SkipFirstEpochSamplerObj::SamplerCopy() {
  auto sampler = std::make_shared<SkipFirstEpochSamplerObj>(start_index_);
  for (const auto &child : children_) {
    Status rc = sampler->AddChildSampler(child);
    if (rc.IsError()) {
      MS_LOG(ERROR) << "[Internal ERROR] Error in copying the sampler. Message: " << rc;
    }
  }
  return sampler;
}
}  // namespace dataset
}  // namespace mindspore
