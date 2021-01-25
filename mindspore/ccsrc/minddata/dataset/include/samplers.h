/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_SAMPLERS_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_SAMPLERS_H_

#include <memory>
#include <string>
#include <vector>

#include "minddata/dataset/include/status.h"
#ifndef ENABLE_ANDROID
#include "minddata/mindrecord/include/shard_column.h"
#include "minddata/mindrecord/include/shard_error.h"
#include "minddata/mindrecord/include/shard_operator.h"
#include "minddata/mindrecord/include/shard_reader.h"
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
  /// \return Shared pointers to the newly created Sampler
  virtual std::shared_ptr<SamplerRT> SamplerBuild() = 0;

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

  virtual Status to_json(nlohmann::json *out_json) { return Status::OK(); }

#ifndef ENABLE_ANDROID
  /// \brief Virtual function to convert a SamplerObj class into a runtime mindrecord sampler object,
  ///     only override by SubsetRandomSampler, PkSampler, RandomSampler, SequentialSampler, DistributedSampler
  /// \return Shared pointers to the newly created Sampler
  virtual std::shared_ptr<mindrecord::ShardOperator> BuildForMindDataset() { return nullptr; }
#endif

 protected:
  /// \brief A function that calls build on the children of this sampler
  /// \param[in] sampler The samplerRT object built from this sampler
  void BuildChildren(std::shared_ptr<SamplerRT> sampler);

  std::vector<std::shared_ptr<SamplerObj>> children_;
};

class DistributedSamplerObj;
class PKSamplerObj;
class PreBuiltSamplerObj;
class RandomSamplerObj;
class SequentialSamplerObj;
class SubsetSamplerObj;
class SubsetRandomSamplerObj;
class WeightedRandomSamplerObj;

/// Function to create a Distributed Sampler.
/// \notes A Sampler that access a shard of the dataset.
/// \param[in] num_shards - Number of shards to divide the dataset into.
/// \param[in] shard_id - Shard ID of the current shard within num_shards.
/// \param[in] shuffle - If true, the indices are shuffled.
/// \param[in] num_samples - The number of samples to draw (default to all elements).
/// \param[in] seed - The seed in use when shuffle is true.
/// \param[in] offset - The starting position where access to elements in the dataset begins.
/// \param[in] even_dist - If true, each shard would return the same number of rows (default to true).
///     If false the total rows returned by all the shards would not have overlap.
/// \return Shared pointer to the current Sampler.
std::shared_ptr<DistributedSamplerObj> DistributedSampler(int64_t num_shards, int64_t shard_id, bool shuffle = true,
                                                          int64_t num_samples = 0, uint32_t seed = 1,
                                                          int64_t offset = -1, bool even_dist = true);

/// Function to create a PK Sampler.
/// \notes Samples K elements for each P class in the dataset.
///        This will sample all classes.
/// \param[in] num_val - Number of elements to sample for each class.
/// \param[in] shuffle - If true, the class IDs are shuffled.
/// \param[in] num_samples - The number of samples to draw (default to all elements).
/// \return Shared pointer to the current Sampler.
std::shared_ptr<PKSamplerObj> PKSampler(int64_t num_val, bool shuffle = false, int64_t num_samples = 0);

/// Function to create a Random Sampler.
/// \notes Samples the elements randomly.
/// \param[in] replacement - If true, put the sample ID back for the next draw.
/// \param[in] num_samples - The number of samples to draw (default to all elements).
/// \return Shared pointer to the current Sampler.
std::shared_ptr<RandomSamplerObj> RandomSampler(bool replacement = false, int64_t num_samples = 0);

/// Function to create a Sequential Sampler.
/// \notes Samples the dataset elements sequentially, same as not having a sampler.
/// \param[in] start_index - Index to start sampling at (default to start at first id).
/// \param[in] num_samples - The number of samples to draw (default to all elements).
/// \return Shared pointer to the current Sampler.
std::shared_ptr<SequentialSamplerObj> SequentialSampler(int64_t start_index = 0, int64_t num_samples = 0);

/// Function to create a Subset  Sampler.
/// \notes Samples the elements from a sequence of indices.
/// \param[in] indices - A vector sequence of indices.
/// \param[in] num_samples - The number of samples to draw (default to all elements).
/// \return Shared pointer to the current Sampler.
std::shared_ptr<SubsetSamplerObj> SubsetSampler(std::vector<int64_t> indices, int64_t num_samples = 0);

/// Function to create a Subset Random Sampler.
/// \notes Samples the elements randomly from a sequence of indices.
/// \param[in] indices - A vector sequence of indices.
/// \param[in] num_samples - The number of samples to draw (default to all elements).
/// \return Shared pointer to the current Sampler.
std::shared_ptr<SubsetRandomSamplerObj> SubsetRandomSampler(std::vector<int64_t> indices, int64_t num_samples = 0);

/// Function to create a Weighted Random Sampler.
/// \notes Samples the elements from [0, len(weights) - 1] randomly with the given
///        weights (probabilities).
/// \param[in] weights - A vector sequence of weights, not necessarily summing up to 1.
/// \param[in] num_samples - The number of samples to draw (default to all elements).
/// \param[in] replacement - If true, put the sample ID back for the next draw.
/// \return Shared pointer to the current Sampler.
std::shared_ptr<WeightedRandomSamplerObj> WeightedRandomSampler(std::vector<double> weights, int64_t num_samples = 0,
                                                                bool replacement = true);

/* ####################################### Derived Sampler classes ################################# */
class DistributedSamplerObj : public SamplerObj {
 public:
  DistributedSamplerObj(int64_t num_shards, int64_t shard_id, bool shuffle, int64_t num_samples, uint32_t seed,
                        int64_t offset, bool even_dist);

  ~DistributedSamplerObj() = default;

  std::shared_ptr<SamplerRT> SamplerBuild() override;

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

  std::shared_ptr<SamplerRT> SamplerBuild() override;

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

  std::shared_ptr<SamplerRT> SamplerBuild() override;

#ifndef ENABLE_ANDROID
  std::shared_ptr<mindrecord::ShardOperator> BuildForMindDataset() override;
#endif

  std::shared_ptr<SamplerObj> SamplerCopy() override;

  Status ValidateParams() override;

  Status to_json(nlohmann::json *out_json) override;

 private:
  std::shared_ptr<SamplerRT> sp_;
#ifndef ENABLE_ANDROID
  std::shared_ptr<mindrecord::ShardOperator> sp_minddataset_;
#endif
};

class RandomSamplerObj : public SamplerObj {
 public:
  RandomSamplerObj(bool replacement, int64_t num_samples);

  ~RandomSamplerObj() = default;

  std::shared_ptr<SamplerRT> SamplerBuild() override;

  std::shared_ptr<SamplerObj> SamplerCopy() override {
    auto sampler = std::make_shared<RandomSamplerObj>(replacement_, num_samples_);
    for (auto child : children_) {
      sampler->AddChildSampler(child);
    }
    return sampler;
  }

#ifndef ENABLE_ANDROID
  std::shared_ptr<mindrecord::ShardOperator> BuildForMindDataset() override;
#endif

  Status ValidateParams() override;

 private:
  bool replacement_;
  int64_t num_samples_;
};

class SequentialSamplerObj : public SamplerObj {
 public:
  SequentialSamplerObj(int64_t start_index, int64_t num_samples);

  ~SequentialSamplerObj() = default;

  std::shared_ptr<SamplerRT> SamplerBuild() override;

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

  Status ValidateParams() override;

 private:
  int64_t start_index_;
  int64_t num_samples_;
};

class SubsetSamplerObj : public SamplerObj {
 public:
  SubsetSamplerObj(std::vector<int64_t> indices, int64_t num_samples);

  ~SubsetSamplerObj() = default;

  std::shared_ptr<SamplerRT> SamplerBuild() override;

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

  Status ValidateParams() override;

 protected:
  const std::vector<int64_t> indices_;
  int64_t num_samples_;
};

class SubsetRandomSamplerObj : public SubsetSamplerObj {
 public:
  SubsetRandomSamplerObj(std::vector<int64_t> indices, int64_t num_samples);

  ~SubsetRandomSamplerObj() = default;

  std::shared_ptr<SamplerRT> SamplerBuild() override;

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

  std::shared_ptr<SamplerRT> SamplerBuild() override;

  std::shared_ptr<SamplerObj> SamplerCopy() override {
    auto sampler = std::make_shared<WeightedRandomSamplerObj>(weights_, num_samples_, replacement_);
    for (auto child : children_) {
      sampler->AddChildSampler(child);
    }
    return sampler;
  }

  Status ValidateParams() override;

 private:
  const std::vector<double> weights_;
  int64_t num_samples_;
  bool replacement_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_SAMPLERS_H_
