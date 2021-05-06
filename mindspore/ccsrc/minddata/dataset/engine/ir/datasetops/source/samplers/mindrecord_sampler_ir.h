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
#ifndef MINDSPORE_MINDRECORD_SAMPLER_IR_H
#define MINDSPORE_MINDRECORD_SAMPLER_IR_H

#include <memory>

#include "minddata/dataset/engine/ir/datasetops/source/samplers/samplers_ir.h"
#include "include/api/status.h"
#ifndef ENABLE_ANDROID
#include "minddata/mindrecord/include/shard_reader.h"
#endif

namespace mindspore {
namespace dataset {

#ifndef ENABLE_ANDROID
class MindRecordSamplerObj : public SamplerObj {
 public:
  /// \brief Constructor
  MindRecordSamplerObj() : shard_reader_(nullptr) {}

  /// \brief Destructor
  ~MindRecordSamplerObj() = default;

  /// \brief Convert a MindRecordSamplerObj into a runtime MindRecordSamplerRT object
  ///        Note that this function not only creates a runtime sampler object, but also creates a ShardReader,
  ///        which will also be needed to build a runtime MindRecordOp
  /// \param[out] sampler Shared pointer to the newly created runtime sampler
  /// \return The Status code of the function. It returns OK status if sampler is created successfully.
  Status SamplerBuild(std::shared_ptr<SamplerRT> *sampler) override;

  /// \brief Function to copy a MindRecordSamplerObj
  /// \return Shared pointer to the newly created SamplerObj
  std::shared_ptr<SamplerObj> SamplerCopy() override;

  /// \brief Function for parameter check. This class requires no input parameter.
  /// \return The Status code of the function. This function always return OK status.
  Status ValidateParams() override { return Status::OK(); }

  /// \brief Function to acquire the unique pointer of the newly created ShardReader object
  ///        Note that this function can only be called after SamplerBuild is called, and can only be called once
  /// \param shard_reader Unique pointer to the newly created ShardReader object
  /// \return The Status code of the function. It returns OK status if acquired a non-empty ShardReader object.
  Status GetShardReader(std::unique_ptr<mindrecord::ShardReader> *shard_reader);

 private:
  std::unique_ptr<mindrecord::ShardReader> shard_reader_;
};
#endif

}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_MINDRECORD_SAMPLER_IR_H
