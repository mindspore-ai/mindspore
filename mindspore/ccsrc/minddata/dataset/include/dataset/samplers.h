/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_DATASET_SAMPLERS_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_DATASET_SAMPLERS_H_

#include <memory>
#include <vector>

#include "include/api/types.h"
#include "include/api/status.h"

namespace mindspore {
namespace dataset {
// Forward declare
class SamplerObj;

// Abstract class to represent a sampler in the data pipeline.
/// \class Sampler samplers.h
/// \brief An abstract base class to represent a sampler in the data pipeline.
class DATASET_API Sampler : std::enable_shared_from_this<Sampler> {
  friend class AlbumDataset;
  friend class Caltech256Dataset;
  friend class CelebADataset;
  friend class Cifar10Dataset;
  friend class Cifar100Dataset;
  friend class CityscapesDataset;
  friend class CLUEDataset;
  friend class CMUArcticDataset;
  friend class CocoDataset;
  friend class CSVDataset;
  friend class DIV2KDataset;
  friend class EMnistDataset;
  friend class FakeImageDataset;
  friend class FashionMnistDataset;
  friend class FlickrDataset;
  friend class Food101Dataset;
  friend class GTZANDataset;
  friend class ImageFolderDataset;
  friend class IMDBDataset;
  friend class KITTIDataset;
  friend class KMnistDataset;
  friend class LFWDataset;
  friend class LibriTTSDataset;
  friend class LJSpeechDataset;
  friend class LSUNDataset;
  friend class ManifestDataset;
  friend class MindDataDataset;
  friend class MnistDataset;
  friend class OmniglotDataset;
  friend class PhotoTourDataset;
  friend class Places365Dataset;
  friend class QMnistDataset;
  friend class RandomDataDataset;
  friend class RenderedSST2Dataset;
  friend class SBUDataset;
  friend class SemeionDataset;
  friend class SpeechCommandsDataset;
  friend class SST2Dataset;
  friend class STL10Dataset;
  friend class SUN397Dataset;
  friend class TedliumDataset;
  friend class TextFileDataset;
  friend class TFRecordDataset;
  friend class USPSDataset;
  friend class VOCDataset;
  friend class WIDERFaceDataset;
  friend class YesNoDataset;
  friend std::shared_ptr<SamplerObj> SelectSampler(int64_t num_samples, bool shuffle, int32_t num_shards,
                                                   int32_t shard_id);

 public:
  /// \brief Constructor
  Sampler() = default;

  /// \brief Destructor
  virtual ~Sampler() = default;

  /// \brief A virtual function to add a child sampler.
  /// \param[in] child The child sampler to be added as a children of this sampler.
  virtual void AddChild(const std::shared_ptr<Sampler> &child) { children_.push_back(child); }

 protected:
  /// \brief Pure virtual function to convert a Sampler class into an IR Sampler object.
  /// \return shared pointer to the newly created TensorOperation.
  virtual std::shared_ptr<SamplerObj> Parse() const = 0;

  /// \brief A function that calls Parse() on the children of this sampler
  /// \param[in] sampler The samplerIR object built from this sampler
  /// \return the Status code returned
  Status BuildChildren(std::shared_ptr<SamplerObj> *const sampler) const;

  std::vector<std::shared_ptr<Sampler>> children_;
};

/// \brief A class to represent a Distributed Sampler in the data pipeline.
/// \note A Sampler that accesses a shard of the dataset.
class DATASET_API DistributedSampler final : public Sampler {
  friend std::shared_ptr<SamplerObj> SelectSampler(int64_t num_samples, bool shuffle, int32_t num_shards,
                                                   int32_t shard_id);

 public:
  /// \brief Constructor
  /// \param[in] num_shards Number of shards to divide the dataset into.
  /// \param[in] shard_id Shard ID of the current shard within num_shards.
  /// \param[in] shuffle If true, the indices are shuffled (default=true).
  /// \param[in] num_samples The number of samples to draw (default=0, return all samples).
  /// \param[in] seed The seed in use when shuffle is true (default=1).
  /// \param[in] offset The starting position where access to elements in the dataset begins (default=-1).
  /// \param[in] even_dist If true, each shard would return the same number of rows (default=true).
  ///     If false the total rows returned by all the shards would not have overlap.
  /// \par Example
  /// \code
  ///      /* creates a distributed sampler with 2 shards in total. This shard is shard 0 */
  ///      std::string file_path = "/path/to/test.mindrecord";
  ///      std::shared_ptr<Dataset> ds = MindData(file_path, {}, std::make_shared<DistributedSampler>(2, 0, false));
  /// \endcode
  DistributedSampler(int64_t num_shards, int64_t shard_id, bool shuffle = true, int64_t num_samples = 0,
                     uint32_t seed = 1, int64_t offset = -1, bool even_dist = true);
  /// \brief Destructor.
  ~DistributedSampler() override = default;

 protected:
  /// \brief The function to convert a Sampler into an IR SamplerObj.
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
/// \note Samples K elements for each P class in the dataset.
///        This will sample all classes.
class DATASET_API PKSampler final : public Sampler {
  friend std::shared_ptr<SamplerObj> SelectSampler(int64_t num_samples, bool shuffle, int32_t num_shards,
                                                   int32_t shard_id);

 public:
  /// \brief Constructor
  /// \param[in] num_val Number of elements to sample for each class.
  /// \param[in] shuffle If true, the class IDs are shuffled (default=false).
  /// \param[in] num_samples The number of samples to draw (default=0, return all samples).
  /// \par Example
  /// \code
  ///      /* creates a PKSampler that will get 3 samples from every class. */
  ///      std::string folder_path = "/path/to/image/folder";
  ///      std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<PKSampler>(3));
  /// \endcode
  explicit PKSampler(int64_t num_val, bool shuffle = false, int64_t num_samples = 0);

  /// \brief Destructor.
  ~PKSampler() override = default;

 protected:
  /// \brief The function to convert a Sampler into an IR SamplerObj.
  /// \return shared pointer to the newly created SamplerObj.
  std::shared_ptr<SamplerObj> Parse() const override;

 private:
  int64_t num_val_;
  bool shuffle_;
  int64_t num_samples_;
};

/// \brief A class to represent a Random Sampler in the data pipeline.
/// \note Samples the elements randomly.
class DATASET_API RandomSampler final : public Sampler {
  friend std::shared_ptr<SamplerObj> SelectSampler(int64_t num_samples, bool shuffle, int32_t num_shards,
                                                   int32_t shard_id);

 public:
  /// \brief Constructor
  /// \param[in] replacement If true, put the sample ID back for the next draw (default=false).
  /// \param[in] num_samples The number of samples to draw (default=0, return all samples).
  /// \par Example
  /// \code
  ///      /* creates a RandomSampler that will get 10 samples randomly */
  ///      std::string folder_path = "/path/to/image/folder";
  ///      std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 10));
  /// \endcode
  explicit RandomSampler(bool replacement = false, int64_t num_samples = 0);

  /// \brief Destructor.
  ~RandomSampler() override = default;

 protected:
  /// \brief The function to convert a Sampler into an IR SamplerObj.
  /// \return shared pointer to the newly created SamplerObj.
  std::shared_ptr<SamplerObj> Parse() const override;

 private:
  bool replacement_;
  int64_t num_samples_;
};

/// \brief A class to represent a Sequential Sampler in the data pipeline.
/// \note Samples the dataset elements sequentially, same as not having a sampler.
class DATASET_API SequentialSampler final : public Sampler {
  friend std::shared_ptr<SamplerObj> SelectSampler(int64_t num_samples, bool shuffle, int32_t num_shards,
                                                   int32_t shard_id);

 public:
  /// \brief Constructor
  /// \param[in] start_index Index to start sampling at (default=0, start at first id).
  /// \param[in] num_samples The number of samples to draw (default=0, return all samples).
  /// \par Example
  /// \code
  ///      /* creates a SequentialSampler that will get 2 samples sequentially in original dataset */
  ///      std::string folder_path = "/path/to/image/folder";
  ///      std::shared_ptr<Dataset> ds = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>(0, 2));
  /// \endcode
  explicit SequentialSampler(int64_t start_index = 0, int64_t num_samples = 0);

  /// \brief Destructor.
  ~SequentialSampler() override = default;

 protected:
  /// \brief The function to convert a Sampler into an IR SamplerObj.
  /// \return shared pointer to the newly created SamplerObj.
  std::shared_ptr<SamplerObj> Parse() const override;

 private:
  int64_t start_index_;
  int64_t num_samples_;
};

/// \brief A class to represent a Subset Sampler in the data pipeline.
/// \note Samples the elements from a sequence of indices.
class DATASET_API SubsetSampler : public Sampler {
  friend std::shared_ptr<SamplerObj> SelectSampler(int64_t num_samples, bool shuffle, int32_t num_shards,
                                                   int32_t shard_id);

 public:
  /// \brief Constructor
  /// \param[in] indices A vector sequence of indices.
  /// \param[in] num_samples The number of samples to draw (default=0, return all samples).
  /// \par Example
  /// \code
  ///      /* creates a SubsetSampler, will sample from the provided indices */
  ///      std::string folder_path = "/path/to/image/folder";
  ///      std::shared_ptr<Dataset> ds = ImageFolder(folder_path, false, std::make_shared<SubsetSampler>({0, 2, 5}));
  /// \endcode
  explicit SubsetSampler(const std::vector<int64_t> &indices, int64_t num_samples = 0);

  /// \brief Destructor.
  ~SubsetSampler() override = default;

 protected:
  /// \brief The function to convert a Sampler into an IR SamplerObj.
  /// \return shared pointer to the newly created SamplerObj.
  std::shared_ptr<SamplerObj> Parse() const override;

  std::vector<int64_t> indices_;
  int64_t num_samples_;
};

/// \brief A class to represent a Subset Random Sampler in the data pipeline.
/// \note Samples the elements randomly from a sequence of indices.
class DATASET_API SubsetRandomSampler final : public SubsetSampler {
  friend std::shared_ptr<SamplerObj> SelectSampler(int64_t num_samples, bool shuffle, int32_t num_shards,
                                                   int32_t shard_id);

 public:
  /// \brief Constructor
  /// \param[in] indices A vector sequence of indices.
  /// \param[in] num_samples The number of samples to draw (default=0, return all samples).
  /// \par Example
  /// \code
  ///      /* create a SubsetRandomSampler, will sample from the provided indices */
  ///      std::string folder_path = "/path/to/image/folder";
  ///      std::shared_ptr<Dataset> ds = ImageFolder(folder_path, false, std::make_shared<SubsetRandomSampler>({2, 7}));
  /// \endcode
  explicit SubsetRandomSampler(const std::vector<int64_t> &indices, int64_t num_samples = 0);

  /// \brief Destructor.
  ~SubsetRandomSampler() override = default;

 protected:
  /// \brief The function to convert a Sampler into an IR SamplerObj.
  /// \return shared pointer to the newly created SamplerObj.
  std::shared_ptr<SamplerObj> Parse() const override;
};

/// \brief A class to represent a Weighted Random Sampler in the data pipeline.
/// \note Samples the elements from [0, len(weights) - 1] randomly with the given
///        weights (probabilities).
class DATASET_API WeightedRandomSampler final : public Sampler {
  friend std::shared_ptr<SamplerObj> SelectSampler(int64_t num_samples, bool shuffle, int32_t num_shards,
                                                   int32_t shard_id);

 public:
  /// \brief Constructor
  /// \param[in] weights A vector sequence of weights, not necessarily summing up to 1.
  /// \param[in] num_samples The number of samples to draw (default=0, return all samples).
  /// \param[in] replacement If true, put the sample ID back for the next draw (default=true).
  /// \par Example
  /// \code
  ///      /* creates a WeightedRandomSampler that will sample 4 elements without replacement */
  ///      std::vector<double> weights = {0.9, 0.8, 0.68, 0.7, 0.71, 0.6, 0.5, 0.4, 0.3, 0.5, 0.2, 0.1};
  ///      sampler = std::make_shared<WeightedRandomSampler>(weights, 4);
  ///      std::string folder_path = "/path/to/image/folder";
  ///      std::shared_ptr<Dataset> ds = ImageFolder(folder_path, false, sampler);
  /// \endcode
  explicit WeightedRandomSampler(const std::vector<double> &weights, int64_t num_samples = 0, bool replacement = true);

  /// \brief Destructor.
  ~WeightedRandomSampler() override = default;

 protected:
  /// \brief The function to convert a Sampler into an IR SamplerObj.
  /// \return shared pointer to the newly created SamplerObj.
  std::shared_ptr<SamplerObj> Parse() const override;

 private:
  std::vector<double> weights_;
  int64_t num_samples_;
  bool replacement_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_DATASET_SAMPLERS_H_
