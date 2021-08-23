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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_IR_DATASETOPS_SOURCE_SAMPLERS_SAMPLERS_IR_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_IR_DATASETOPS_SOURCE_SAMPLERS_SAMPLERS_IR_H_

#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <nlohmann/json.hpp>

#include "include/api/status.h"
#ifndef ENABLE_ANDROID
#include "minddata/mindrecord/include/shard_operator.h"
#endif

namespace mindspore {
namespace dataset {

// Internal Sampler class forward declaration
class SamplerRT;

class SamplerObj {
 public:
  /// \brief Constructor
  SamplerObj();

  /// \brief Destructor
  ~SamplerObj();

  /// \brief Pure virtual function for derived class to implement parameters validation
  /// \return The Status code of the function. It returns OK status if parameters are valid.
  virtual Status ValidateParams() = 0;

  /// \brief Pure virtual function to convert a SamplerObj class into a runtime sampler object
  /// \param[out] sampler Shared pointers to the newly created Sampler
  /// \return The Status code of the function. It returns OK status if sampler is created successfully.
  virtual Status SamplerBuild(std::shared_ptr<SamplerRT> *sampler) = 0;

  /// \brief Pure virtual function to copy a SamplerObj class
  /// \return Shared pointers to the newly copied SamplerObj
  virtual std::shared_ptr<SamplerObj> SamplerCopy() = 0;

  /// \brief Function for derived class to get the shard id of sampler
  /// \return The shard id of the derived sampler
  virtual int64_t ShardId() { return 0; }

  /// \brief Adds a child to the sampler
  /// \param[in] child The sampler to be added as child
  /// \return the Status code returned
  Status AddChildSampler(std::shared_ptr<SamplerObj> child);

  virtual Status to_json(nlohmann::json *const out_json);

#ifndef ENABLE_ANDROID
  /// \brief Function to construct children samplers
  /// \param[in] json_obj The JSON object to be deserialized
  /// \param[out] parent_sampler given parent sampler, output constructed parent sampler with children samplers added
  /// \return Status The status code returned
  static Status from_json(nlohmann::json json_obj, std::shared_ptr<SamplerObj> *parent_sampler);
#endif

  std::vector<std::shared_ptr<SamplerObj>> GetChild() { return children_; }

#ifndef ENABLE_ANDROID
  /// \brief Virtual function to convert a SamplerObj class into a runtime mindrecord sampler object,
  ///     only override by SubsetRandomSampler, PkSampler, RandomSampler, SequentialSampler, DistributedSampler
  /// \return Shared pointers to the newly created Sampler
  virtual std::shared_ptr<mindrecord::ShardOperator> BuildForMindDataset() { return nullptr; }
#endif

 protected:
  /// \brief A function that calls build on the children of this sampler
  /// \param[in] sampler The samplerRT object built from this sampler
  /// \return the Status code returned
  Status BuildChildren(std::shared_ptr<SamplerRT> *const sampler);

  std::vector<std::shared_ptr<SamplerObj>> children_;
};

}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_IR_DATASETOPS_SOURCE_SAMPLERS_SAMPLERS_IR_H_
