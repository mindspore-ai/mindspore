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

#include "minddata/dataset/engine/ir/datasetops/source/samplers/mindrecord_sampler_ir.h"

#include <memory>
#include <utility>

#ifndef ENABLE_ANDROID
#include "minddata/dataset/engine/datasetops/source/sampler/mind_record_sampler.h"
#include "minddata/mindrecord/include/shard_reader.h"
#endif

namespace mindspore {
namespace dataset {
#ifndef ENABLE_ANDROID
// This function not only creates a runtime sampler object, but also creates a ShardReader,
// which will also be needed to build a runtime MindRecordOp
// (cannot add another output parameter because it has to override base class's function)
Status MindRecordSamplerObj::SamplerBuild(std::shared_ptr<SamplerRT> *sampler) {
  shard_reader_ = std::make_unique<mindrecord::ShardReader>();
  *sampler = std::make_shared<MindRecordSamplerRT>(shard_reader_.get());
  return Status::OK();
}

std::shared_ptr<SamplerObj> MindRecordSamplerObj::SamplerCopy() {
  auto sampler = std::make_shared<MindRecordSamplerObj>();
  return sampler;
}

// Function to acquire the unique pointer of the newly created ShardReader object
// Note this function can only be called after SamplerBuild is finished, and can only be called once. Otherwise this
// function will return error status.
Status MindRecordSamplerObj::GetShardReader(std::unique_ptr<mindrecord::ShardReader> *shard_reader) {
  CHECK_FAIL_RETURN_UNEXPECTED(shard_reader_ != nullptr, "Internal error. Attempt to get an empty shard reader.");
  *shard_reader = std::move(shard_reader_);
  return Status::OK();
}
#endif
}  // namespace dataset
}  // namespace mindspore
