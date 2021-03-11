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
  ~SamplerObj() = default;

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

  virtual Status to_json(nlohmann::json *const out_json) { return Status::OK(); }

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

/* ####################################### Derived Sampler classes ################################# */
class DistributedSamplerObj : public SamplerObj {
 public:
  DistributedSamplerObj(int64_t num_shards, int64_t shard_id, bool shuffle, int64_t num_samples, uint32_t seed,
                        int64_t offset, bool even_dist);

  ~DistributedSamplerObj() = default;

  Status SamplerBuild(std::shared_ptr<SamplerRT> *sampler) override;

  std::shared_ptr<SamplerObj> SamplerCopy() override {
    auto sampler = std::make_shared<DistributedSamplerObj>(num_shards_, shard_id_, shuffle_, num_samples_, seed_,
                                                           offset_, even_dist_);
    for (auto child : children_) {
      sampler->AddChildSampler(child);
    }
    return sampler;
  }

#ifndef ENABLE_ANDROID
  std::shared_ptr<mindrecord::ShardOperator> BuildForMindDataset() override;
#endif

  /// \brief Get the arguments of node
  /// \param[out] out_json JSON string of all attributes
  /// \return Status of the function
  Status to_json(nlohmann::json *const out_json) override;

  Status ValidateParams() override;

  /// \brief Function to get the shard id of sampler
  /// \return The shard id of sampler
  int64_t ShardId() override { return shard_id_; }

 private:
  int64_t num_shards_;
  int64_t shard_id_;
  bool shuffle_;
  int64_t num_samples_;
  uint32_t seed_;
  int64_t offset_;
  bool even_dist_;
};

class PKSamplerObj : public SamplerObj {
 public:
  PKSamplerObj(int64_t num_val, bool shuffle, int64_t num_samples);

  ~PKSamplerObj() = default;

  Status SamplerBuild(std::shared_ptr<SamplerRT> *sampler) override;

  std::shared_ptr<SamplerObj> SamplerCopy() override {
    auto sampler = std::make_shared<PKSamplerObj>(num_val_, shuffle_, num_samples_);
    for (auto child : children_) {
      sampler->AddChildSampler(child);
    }
    return sampler;
  }

#ifndef ENABLE_ANDROID
  std::shared_ptr<mindrecord::ShardOperator> BuildForMindDataset() override;
#endif

  /// \brief Get the arguments of node
  /// \param[out] out_json JSON string of all attributes
  /// \return Status of the function
  Status to_json(nlohmann::json *const out_json) override;

  Status ValidateParams() override;

 private:
  int64_t num_val_;
  bool shuffle_;
  int64_t num_samples_;
};

class PreBuiltSamplerObj : public SamplerObj {
 public:
  explicit PreBuiltSamplerObj(std::shared_ptr<SamplerRT> sampler);
#ifndef ENABLE_ANDROID
  explicit PreBuiltSamplerObj(std::shared_ptr<mindrecord::ShardOperator> sampler);
#endif

  ~PreBuiltSamplerObj() = default;

  Status SamplerBuild(std::shared_ptr<SamplerRT> *const sampler) override;

#ifndef ENABLE_ANDROID
  std::shared_ptr<mindrecord::ShardOperator> BuildForMindDataset() override;
#endif

  std::shared_ptr<SamplerObj> SamplerCopy() override;

  Status ValidateParams() override;

  Status to_json(nlohmann::json *const out_json) override;

 private:
  std::shared_ptr<SamplerRT> sp_;
#ifndef ENABLE_ANDROID
  std::shared_ptr<mindrecord::ShardOperator> sp_minddataset_;
#endif
};

class RandomSamplerObj : public SamplerObj {
 public:
  RandomSamplerObj(bool replacement, int64_t num_samples, bool reshuffle_each_epoch = true);

  ~RandomSamplerObj() = default;

  Status SamplerBuild(std::shared_ptr<SamplerRT> *sampler) override;

  std::shared_ptr<SamplerObj> SamplerCopy() override {
    auto sampler = std::make_shared<RandomSamplerObj>(replacement_, num_samples_, reshuffle_each_epoch_);
    for (auto child : children_) {
      sampler->AddChildSampler(child);
    }
    return sampler;
  }

#ifndef ENABLE_ANDROID
  std::shared_ptr<mindrecord::ShardOperator> BuildForMindDataset() override;
#endif

  /// \brief Get the arguments of node
  /// \param[out] out_json JSON string of all attributes
  /// \return Status of the function
  Status to_json(nlohmann::json *const out_json) override;

  Status ValidateParams() override;

 private:
  bool replacement_;
  int64_t num_samples_;
  bool reshuffle_each_epoch_;
};

class SequentialSamplerObj : public SamplerObj {
 public:
  SequentialSamplerObj(int64_t start_index, int64_t num_samples);

  ~SequentialSamplerObj() = default;

  Status SamplerBuild(std::shared_ptr<SamplerRT> *sampler) override;

  std::shared_ptr<SamplerObj> SamplerCopy() override {
    auto sampler = std::make_shared<SequentialSamplerObj>(start_index_, num_samples_);
    for (auto child : children_) {
      sampler->AddChildSampler(child);
    }
    return sampler;
  }

#ifndef ENABLE_ANDROID
  std::shared_ptr<mindrecord::ShardOperator> BuildForMindDataset() override;
#endif

  /// \brief Get the arguments of node
  /// \param[out] out_json JSON string of all attributes
  /// \return Status of the function
  Status to_json(nlohmann::json *const out_json) override;

  Status ValidateParams() override;

 private:
  int64_t start_index_;
  int64_t num_samples_;
};

class SubsetSamplerObj : public SamplerObj {
 public:
  SubsetSamplerObj(std::vector<int64_t> indices, int64_t num_samples);

  ~SubsetSamplerObj() = default;

  Status SamplerBuild(std::shared_ptr<SamplerRT> *sampler) override;

  std::shared_ptr<SamplerObj> SamplerCopy() override {
    auto sampler = std::make_shared<SubsetSamplerObj>(indices_, num_samples_);
    for (auto child : children_) {
      sampler->AddChildSampler(child);
    }
    return sampler;
  }

#ifndef ENABLE_ANDROID
  std::shared_ptr<mindrecord::ShardOperator> BuildForMindDataset() override;
#endif

  /// \brief Get the arguments of node
  /// \param[out] out_json JSON string of all attributes
  /// \return Status of the function
  Status to_json(nlohmann::json *const out_json) override;

  Status ValidateParams() override;

 protected:
  const std::vector<int64_t> indices_;
  int64_t num_samples_;
};

class SubsetRandomSamplerObj : public SubsetSamplerObj {
 public:
  SubsetRandomSamplerObj(std::vector<int64_t> indices, int64_t num_samples);

  ~SubsetRandomSamplerObj() = default;

  Status to_json(nlohmann::json *const out_json) override;

  Status SamplerBuild(std::shared_ptr<SamplerRT> *sampler) override;

  std::shared_ptr<SamplerObj> SamplerCopy() override {
    auto sampler = std::make_shared<SubsetRandomSamplerObj>(indices_, num_samples_);
    for (auto child : children_) {
      sampler->AddChildSampler(child);
    }
    return sampler;
  }

#ifndef ENABLE_ANDROID
  std::shared_ptr<mindrecord::ShardOperator> BuildForMindDataset() override;
#endif

 private:
};

class WeightedRandomSamplerObj : public SamplerObj {
 public:
  explicit WeightedRandomSamplerObj(std::vector<double> weights, int64_t num_samples = 0, bool replacement = true);

  ~WeightedRandomSamplerObj() = default;

  Status SamplerBuild(std::shared_ptr<SamplerRT> *sampler) override;

  std::shared_ptr<SamplerObj> SamplerCopy() override {
    auto sampler = std::make_shared<WeightedRandomSamplerObj>(weights_, num_samples_, replacement_);
    for (auto child : children_) {
      sampler->AddChildSampler(child);
    }
    return sampler;
  }

  /// \brief Get the arguments of node
  /// \param[out] out_json JSON string of all attributes
  /// \return Status of the function
  Status to_json(nlohmann::json *const out_json) override;

  Status ValidateParams() override;

 private:
  const std::vector<double> weights_;
  int64_t num_samples_;
  bool replacement_;
};

}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_IR_DATASETOPS_SOURCE_SAMPLERS_SAMPLERS_IR_H_
