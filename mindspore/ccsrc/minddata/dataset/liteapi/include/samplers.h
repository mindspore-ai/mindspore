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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_SAMPLERS_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_SAMPLERS_H_

#include <memory>
#include <vector>

namespace mindspore {
namespace dataset {

// Forward declare
class SamplerObj;

// Abstract class to represent a sampler in the data pipeline.
/// \class Sampler samplers.h
/// \brief An abstract base class to represent a sampler in the data pipeline.
class Sampler : std::enable_shared_from_this<Sampler> {
  friend class AlbumDataset;
  friend class CelebADataset;
  friend class Cifar10Dataset;
  friend class Cifar100Dataset;
  friend class CLUEDataset;
  friend class CocoDataset;
  friend class CSVDataset;
  friend class ImageFolderDataset;
  friend class ManifestDataset;
  friend class MindDataDataset;
  friend class MnistDataset;
  friend class RandomDataDataset;
  friend class TextFileDataset;
  friend class TFRecordDataset;
  friend class VOCDataset;
  friend std::shared_ptr<SamplerObj> SelectSampler(int64_t, bool, int32_t, int32_t);

 public:
  /// \brief Constructor
  Sampler() {}

  /// \brief Destructor
  ~Sampler() = default;

  /// \brief A virtual function to add a child sampler.
  /// \param[in] child The child sampler to be added as a children of this sampler.
  virtual void AddChild(std::shared_ptr<Sampler> child) { children_.push_back(child); }

 protected:
  /// \brief Pure virtual function to convert a Sampler class into an IR Sampler object.
  /// \return shared pointer to the newly created TensorOperation.
  virtual std::shared_ptr<SamplerObj> Parse() const = 0;

  std::vector<std::shared_ptr<Sampler>> children_;
};

/// \brief A class to represent a Distributed Sampler in the data pipeline.
/// \notes A Sampler that accesses a shard of the dataset.
class DistributedSampler final : public Sampler {
  friend std::shared_ptr<SamplerObj> SelectSampler(int64_t, bool, int32_t, int32_t);

 public:
  /// \brief Constructor
  /// \param[in] num_shards - Number of shards to divide the dataset into.
  /// \param[in] shard_id - Shard ID of the current shard within num_shards.
  /// \param[in] shuffle - If true, the indices are shuffled.
  /// \param[in] num_samples - The number of samples to draw (default to all elements).
  /// \param[in] seed - The seed in use when shuffle is true.
  /// \param[in] offset - The starting position where access to elements in the dataset begins.
  /// \param[in] even_dist - If true, each shard would return the same number of rows (default to true).
  ///     If false the total rows returned by all the shards would not have overlap.
  explicit DistributedSampler(int64_t num_shards, int64_t shard_id, bool shuffle = true, int64_t num_samples = 0,
                              uint32_t seed = 1, int64_t offset = -1, bool even_dist = true);
  /// \brief Destructor.
  ~DistributedSampler() = default;

 protected:
  /// \brief Function to convert a Sampler into an IR SamplerObj.
  /// \return shared pointer to the newly created SamplerObj.
  std::shared_ptr<SamplerObj> Parse() const override;

 private:
  int64_t num_shards_;
  int64_t shard_id_;
  bool shuffle_;
  int64_t num_samples_;
  uint32_t seed_;
  int64_t offset_;
  bool even_dist_;
};

/// \brief A class to represent a PK Sampler in the data pipeline.
/// \notes Samples K elements for each P class in the dataset.
///        This will sample all classes.
class PKSampler final : public Sampler {
  friend std::shared_ptr<SamplerObj> SelectSampler(int64_t, bool, int32_t, int32_t);

 public:
  /// \brief Constructor
  /// \param[in] num_val - Number of elements to sample for each class.
  /// \param[in] shuffle - If true, the class IDs are shuffled.
  /// \param[in] num_samples - The number of samples to draw (default to all elements).
  explicit PKSampler(int64_t num_val, bool shuffle = false, int64_t num_samples = 0);

  /// \brief Destructor.
  ~PKSampler() = default;

 protected:
  /// \brief Function to convert a Sampler into an IR SamplerObj.
  /// \return shared pointer to the newly created SamplerObj.
  std::shared_ptr<SamplerObj> Parse() const override;

 private:
  int64_t num_val_;
  bool shuffle_;
  int64_t num_samples_;
};

/// \brief A class to represent a Random Sampler in the data pipeline.
/// \notes Samples the elements randomly.
class RandomSampler final : public Sampler {
  friend std::shared_ptr<SamplerObj> SelectSampler(int64_t, bool, int32_t, int32_t);

 public:
  /// \brief Constructor
  /// \param[in] replacement - If true, put the sample ID back for the next draw.
  /// \param[in] num_samples - The number of samples to draw (default to all elements).
  explicit RandomSampler(bool replacement = false, int64_t num_samples = 0);

  /// \brief Destructor.
  ~RandomSampler() = default;

 protected:
  /// \brief Function to convert a Sampler into an IR SamplerObj.
  /// \return shared pointer to the newly created SamplerObj.
  std::shared_ptr<SamplerObj> Parse() const override;

 private:
  bool replacement_;
  int64_t num_samples_;
};

/// \brief A class to represent a Sequential Sampler in the data pipeline.
/// \notes Samples the dataset elements sequentially, same as not having a sampler.
class SequentialSampler final : public Sampler {
  friend std::shared_ptr<SamplerObj> SelectSampler(int64_t, bool, int32_t, int32_t);

 public:
  /// \brief Constructor
  /// \param[in] start_index - Index to start sampling at (default to start at first id).
  /// \param[in] num_samples - The number of samples to draw (default to all elements).
  explicit SequentialSampler(int64_t start_index = 0, int64_t num_samples = 0);

  /// \brief Destructor.
  ~SequentialSampler() = default;

 protected:
  /// \brief Function to convert a Sampler into an IR SamplerObj.
  /// \return shared pointer to the newly created SamplerObj.
  std::shared_ptr<SamplerObj> Parse() const override;

 private:
  int64_t start_index_;
  int64_t num_samples_;
};

/// \brief A class to represent a Subset Sampler in the data pipeline.
/// \notes Samples the elements from a sequence of indices.
class SubsetSampler : public Sampler {
  friend std::shared_ptr<SamplerObj> SelectSampler(int64_t, bool, int32_t, int32_t);

 public:
  /// \brief Constructor
  /// \param[in] indices - A vector sequence of indices.
  /// \param[in] num_samples - The number of samples to draw (default to all elements).
  explicit SubsetSampler(std::vector<int64_t> indices, int64_t num_samples = 0);

  /// \brief Destructor.
  ~SubsetSampler() = default;

 protected:
  /// \brief Function to convert a Sampler into an IR SamplerObj.
  /// \return shared pointer to the newly created SamplerObj.
  std::shared_ptr<SamplerObj> Parse() const override;

  std::vector<int64_t> indices_;
  int64_t num_samples_;
};

/// \brief A class to represent a Subset Random Sampler in the data pipeline.
/// \notes Samples the elements randomly from a sequence of indices.
class SubsetRandomSampler final : public SubsetSampler {
  friend std::shared_ptr<SamplerObj> SelectSampler(int64_t, bool, int32_t, int32_t);

 public:
  /// \brief Constructor
  /// \param[in] indices - A vector sequence of indices.
  /// \param[in] num_samples - The number of samples to draw (default to all elements).
  explicit SubsetRandomSampler(std::vector<int64_t> indices, int64_t num_samples = 0);

  /// \brief Destructor.
  ~SubsetRandomSampler() = default;

 protected:
  /// \brief Function to convert a Sampler into an IR SamplerObj.
  /// \return shared pointer to the newly created SamplerObj.
  std::shared_ptr<SamplerObj> Parse() const override;
};

/// \brief A class to represent a Weighted Random Sampler in the data pipeline.
/// \notes Samples the elements from [0, len(weights) - 1] randomly with the given
///        weights (probabilities).
class WeightedRandomSampler final : public Sampler {
  friend std::shared_ptr<SamplerObj> SelectSampler(int64_t, bool, int32_t, int32_t);

 public:
  /// \brief Constructor
  /// \param[in] weights - A vector sequence of weights, not necessarily summing up to 1.
  /// \param[in] num_samples - The number of samples to draw (default to all elements).
  /// \param[in] replacement - If true, put the sample ID back for the next draw.
  explicit WeightedRandomSampler(std::vector<double> weights, int64_t num_samples = 0, bool replacement = true);

  /// \brief Destructor.
  ~WeightedRandomSampler() = default;

 protected:
  /// \brief Function to convert a Sampler into an IR SamplerObj.
  /// \return shared pointer to the newly created SamplerObj.
  std::shared_ptr<SamplerObj> Parse() const override;

 private:
  std::vector<double> weights_;
  int64_t num_samples_;
  bool replacement_;
};

}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_SAMPLERS_H_
