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

#include "minddata/dataset/engine/ir/datasetops/source/samplers/samplers_ir.h"
#include "minddata/dataset/engine/datasetops/source/sampler/sampler.h"
#ifndef ENABLE_ANDROID
#include "minddata/dataset/engine/serdes.h"
#endif

#include "minddata/dataset/core/config_manager.h"

namespace mindspore {
namespace dataset {

// Constructor
SamplerObj::SamplerObj() {}

// Destructor
SamplerObj::~SamplerObj() = default;

Status SamplerObj::BuildChildren(std::shared_ptr<SamplerRT> *const sampler) {
  for (auto child : children_) {
    std::shared_ptr<SamplerRT> sampler_rt = nullptr;
    RETURN_IF_NOT_OK(child->SamplerBuild(&sampler_rt));
    RETURN_IF_NOT_OK((*sampler)->AddChild(sampler_rt));
  }
  return Status::OK();
}

Status SamplerObj::AddChildSampler(std::shared_ptr<SamplerObj> child) {
  if (child == nullptr) {
    return Status::OK();
  }

  // Only samplers can be added, not any other DatasetOp.
  std::shared_ptr<SamplerObj> sampler = std::dynamic_pointer_cast<SamplerObj>(child);
  if (!sampler) {
    RETURN_STATUS_UNEXPECTED("Cannot add child, child is not a sampler object.");
  }

  // Samplers can have at most 1 child.
  if (!children_.empty()) {
    RETURN_STATUS_UNEXPECTED("Cannot add child sampler, this sampler already has a child.");
  }

  children_.push_back(child);

  return Status::OK();
}

Status SamplerObj::to_json(nlohmann::json *const out_json) {
  nlohmann::json args;
  if (!children_.empty()) {
    std::vector<nlohmann::json> children_args;
    for (auto child : children_) {
      nlohmann::json child_arg;
      RETURN_IF_NOT_OK(child->to_json(&child_arg));
      children_args.push_back(child_arg);
    }
    args["child_sampler"] = children_args;
  }
  *out_json = args;
  return Status::OK();
}

#ifndef ENABLE_ANDROID
Status SamplerObj::from_json(nlohmann::json json_obj, std::shared_ptr<SamplerObj> *parent_sampler) {
  for (nlohmann::json child : json_obj["child_sampler"]) {
    std::shared_ptr<SamplerObj> child_sampler;
    RETURN_IF_NOT_OK(Serdes::ConstructSampler(child, &child_sampler));
    (*parent_sampler)->AddChildSampler(child_sampler);
  }
  return Status::OK();
}
#endif
}  // namespace dataset
}  // namespace mindspore
