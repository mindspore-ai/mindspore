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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_DATASETS_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_DATASETS_H_

#include <vector>
#include <memory>
#include <set>
#include <map>
#include <utility>
#include <string>
#include "minddata/dataset/include/tensor.h"
#include "minddata/dataset/include/iterator.h"
#include "minddata/dataset/include/samplers.h"

namespace mindspore {
namespace dataset {

// Forward declare
class DatasetOp;
class DataSchema;
class Tensor;
class TensorShape;

namespace api {

class TensorOperation;
class SamplerObj;
// Datasets classes (in alphabetical order)
class Cifar10Dataset;
class Cifar100Dataset;
class ImageFolderDataset;
class MnistDataset;
// Dataset Op classes (in alphabetical order)
class BatchDataset;
class MapDataset;
class ProjectDataset;
class RenameDataset;
class RepeatDataset;
class ShuffleDataset;
class SkipDataset;
class ZipDataset;

/// \brief Function to create a Cifar10 Dataset
/// \notes The generated dataset has two columns ['image', 'label']
/// \param[in] dataset_dir Path to the root directory that contains the dataset
/// \param[in] sampler Object used to choose samples from the dataset. If sampler is `nullptr`, A `RandomSampler`
///    will be used to randomly iterate the entire dataset
/// \return Shared pointer to the current Dataset
std::shared_ptr<Cifar10Dataset> Cifar10(const std::string &dataset_dir, std::shared_ptr<SamplerObj> sampler = nullptr);

/// \brief Function to create a Cifar100 Dataset
/// \notes The generated dataset has two columns ['image', 'coarse_label', 'fine_label']
/// \param[in] dataset_dir Path to the root directory that contains the dataset
/// \param[in] sampler Object used to choose samples from the dataset. If sampler is `nullptr`, A `RandomSampler`
///    will be used to randomly iterate the entire dataset
/// \return Shared pointer to the current Dataset
std::shared_ptr<Cifar100Dataset> Cifar100(const std::string &dataset_dir,
                                          std::shared_ptr<SamplerObj> sampler = nullptr);

/// \brief Function to create an ImageFolderDataset
/// \notes A source dataset that reads images from a tree of directories
///    All images within one folder have the same label
///    The generated dataset has two columns ['image', 'label']
/// \param[in] dataset_dir Path to the root directory that contains the dataset
/// \param[in] decode A flag to decode in ImageFolder
/// \param[in] sampler Object used to choose samples from the dataset. If sampler is `nullptr`,
///    A `RandomSampler` will be used to randomly iterate the entire dataset
/// \param[in] extensions File extensions to be read
/// \param[in] class_indexing a class name to label map
/// \return Shared pointer to the current ImageFolderDataset
std::shared_ptr<ImageFolderDataset> ImageFolder(std::string dataset_dir, bool decode = false,
                                                std::shared_ptr<SamplerObj> sampler = nullptr,
                                                std::set<std::string> extensions = {},
                                                std::map<std::string, int32_t> class_indexing = {});

/// \brief Function to create a MnistDataset
/// \notes The generated dataset has two columns ['image', 'label']
/// \param[in] dataset_dir Path to the root directory that contains the dataset
/// \param[in] sampler Object used to choose samples from the dataset. If sampler is `nullptr`,
///    A `RandomSampler` will be used to randomly iterate the entire dataset
/// \return Shared pointer to the current MnistDataset
std::shared_ptr<MnistDataset> Mnist(std::string dataset_dir, std::shared_ptr<SamplerObj> sampler = nullptr);

/// \class Dataset datasets.h
/// \brief A base class to represent a dataset in the data pipeline.
class Dataset : public std::enable_shared_from_this<Dataset> {
 public:
  friend class Iterator;

  /// \brief Constructor
  Dataset();

  /// \brief Destructor
  ~Dataset() = default;

  /// \brief Pure virtual function to convert a Dataset class into a runtime dataset object
  /// \return The list of shared pointers to the newly created DatasetOps
  virtual std::vector<std::shared_ptr<DatasetOp>> Build() = 0;

  /// \brief Pure virtual function for derived class to implement parameters validation
  /// \return bool True if all the params are valid
  virtual bool ValidateParams() = 0;

  /// \brief Setter function for runtime number of workers
  /// \param[in] num_workers The number of threads in this operator
  /// \return Shared pointer to the original object
  std::shared_ptr<Dataset> SetNumWorkers(int32_t num_workers) {
    num_workers_ = num_workers;
    return shared_from_this();
  }

  /// \brief Function to create an Iterator over the Dataset pipeline
  /// \return Shared pointer to the Iterator
  std::shared_ptr<Iterator> CreateIterator();

  /// \brief Function to create a BatchDataset
  /// \notes Combines batch_size number of consecutive rows into batches
  /// \param[in] batch_size Path to the root directory that contains the dataset
  /// \param[in] drop_remainder Determines whether or not to drop the last possibly incomplete
  ///    batch. If true, and if there are less than batch_size rows
  ///    available to make the last batch, then those rows will
  ///    be dropped and not propagated to the next node
  /// \return Shared pointer to the current BatchDataset
  std::shared_ptr<BatchDataset> Batch(int32_t batch_size, bool drop_remainder = false);

  /// \brief Function to create a MapDataset
  /// \notes Applies each operation in operations to this dataset
  /// \param[in] operations Vector of operations to be applied on the dataset. Operations are
  ///    applied in the order they appear in this list
  /// \param[in] input_columns Vector of the names of the columns that will be passed to the first
  ///    operation as input. The size of this list must match the number of
  ///    input columns expected by the first operator. The default input_columns
  ///    is the first column
  /// \param[in] output_columns Vector of names assigned to the columns outputted by the last operation
  ///    This parameter is mandatory if len(input_columns) != len(output_columns)
  ///    The size of this list must match the number of output columns of the
  ///    last operation. The default output_columns will have the same
  ///    name as the input columns, i.e., the columns will be replaced
  /// \param[in] project_columns A list of column names to project
  /// \return Shared pointer to the current MapDataset
  std::shared_ptr<MapDataset> Map(std::vector<std::shared_ptr<TensorOperation>> operations,
                                  std::vector<std::string> input_columns = {},
                                  std::vector<std::string> output_columns = {},
                                  const std::vector<std::string> &project_columns = {});

  /// \brief Function to create a Project Dataset
  /// \notes Applies project to the dataset
  /// \param[in] columns The name of columns to project
  /// \return Shared pointer to the current Dataset
  std::shared_ptr<ProjectDataset> Project(const std::vector<std::string> &columns);

  /// \brief Function to create a Rename Dataset
  /// \notes Renames the columns in the input dataset
  /// \param[in] input_columns List of the input columns to rename
  /// \param[in] output_columns List of the output columns
  /// \return Shared pointer to the current Dataset
  std::shared_ptr<RenameDataset> Rename(const std::vector<std::string> &input_columns,
                                        const std::vector<std::string> &output_columns);

  /// \brief Function to create a RepeatDataset
  /// \notes Repeats this dataset count times. Repeat indefinitely if count is -1
  /// \param[in] count Number of times the dataset should be repeated
  /// \return Shared pointer to the current Dataset
  /// \note Repeat will return shared pointer to `Dataset` instead of `RepeatDataset`
  ///    due to a limitation in the current implementation
  std::shared_ptr<Dataset> Repeat(int32_t count = -1);

  /// \brief Function to create a Shuffle Dataset
  /// \notes Randomly shuffles the rows of this dataset
  /// \param[in] buffer_size The size of the buffer (must be larger than 1) for shuffling
  /// \return Shared pointer to the current ShuffleDataset
  std::shared_ptr<ShuffleDataset> Shuffle(int32_t shuffle_size);

  /// \brief Function to create a SkipDataset
  /// \notes Skips count elements in this dataset.
  /// \param[in] count Number of elements the dataset to be skipped.
  /// \return Shared pointer to the current SkipDataset
  std::shared_ptr<SkipDataset> Skip(int32_t count);

  /// \brief Function to create a Zip Dataset
  /// \notes Applies zip to the dataset
  /// \param[in] datasets A list of shared pointer to the datasets that we want to zip
  /// \return Shared pointer to the current Dataset
  std::shared_ptr<ZipDataset> Zip(const std::vector<std::shared_ptr<Dataset>> &datasets);

 protected:
  std::vector<std::shared_ptr<Dataset>> children;
  std::shared_ptr<Dataset> parent;

  int32_t num_workers_;
  int32_t rows_per_buffer_;
  int32_t connector_que_size_;
};

/* ####################################### Derived Dataset classes ################################# */

class Cifar10Dataset : public Dataset {
 public:
  /// \brief Constructor
  Cifar10Dataset(const std::string &dataset_dir, std::shared_ptr<SamplerObj> sampler);

  /// \brief Destructor
  ~Cifar10Dataset() = default;

  /// \brief a base class override function to create the required runtime dataset op objects for this class
  /// \return The list of shared pointers to the newly created DatasetOps
  std::vector<std::shared_ptr<DatasetOp>> Build() override;

  /// \brief Parameters validation
  /// \return bool true if all the params are valid
  bool ValidateParams() override;

 private:
  std::string dataset_dir_;
  std::shared_ptr<SamplerObj> sampler_;
};

class Cifar100Dataset : public Dataset {
 public:
  /// \brief Constructor
  Cifar100Dataset(const std::string &dataset_dir, std::shared_ptr<SamplerObj> sampler);

  /// \brief Destructor
  ~Cifar100Dataset() = default;

  /// \brief a base class override function to create the required runtime dataset op objects for this class
  /// \return The list of shared pointers to the newly created DatasetOps
  std::vector<std::shared_ptr<DatasetOp>> Build() override;

  /// \brief Parameters validation
  /// \return bool true if all the params are valid
  bool ValidateParams() override;

 private:
  std::string dataset_dir_;
  std::shared_ptr<SamplerObj> sampler_;
};

/// \class ImageFolderDataset
/// \brief A Dataset derived class to represent ImageFolder dataset
class ImageFolderDataset : public Dataset {
 public:
  /// \brief Constructor
  ImageFolderDataset(std::string dataset_dir, bool decode, std::shared_ptr<SamplerObj> sampler, bool recursive,
                     std::set<std::string> extensions, std::map<std::string, int32_t> class_indexing);

  /// \brief Destructor
  ~ImageFolderDataset() = default;

  /// \brief a base class override function to create the required runtime dataset op objects for this class
  /// \return The list of shared pointers to the newly created DatasetOps
  std::vector<std::shared_ptr<DatasetOp>> Build() override;

  /// \brief Parameters validation
  /// \return bool true if all the params are valid
  bool ValidateParams() override;

 private:
  std::string dataset_dir_;
  bool decode_;
  bool recursive_;
  std::shared_ptr<SamplerObj> sampler_;
  std::map<std::string, int32_t> class_indexing_;
  std::set<std::string> exts_;
};

class MnistDataset : public Dataset {
 public:
  /// \brief Constructor
  MnistDataset(std::string dataset_dir, std::shared_ptr<SamplerObj> sampler);

  /// \brief Destructor
  ~MnistDataset() = default;

  /// \brief a base class override function to create the required runtime dataset op objects for this class
  /// \return The list of shared pointers to the newly created DatasetOps
  std::vector<std::shared_ptr<DatasetOp>> Build() override;

  /// \brief Parameters validation
  /// \return bool true if all the params are valid
  bool ValidateParams() override;

 private:
  std::string dataset_dir_;
  std::shared_ptr<SamplerObj> sampler_;
};

class BatchDataset : public Dataset {
 public:
  /// \brief Constructor
  BatchDataset(int32_t batch_size, bool drop_remainder, bool pad, std::vector<std::string> cols_to_map,
               std::map<std::string, std::pair<TensorShape, std::shared_ptr<Tensor>>> pad_map);

  /// \brief Destructor
  ~BatchDataset() = default;

  /// \brief a base class override function to create the required runtime dataset op objects for this class
  /// \return The list of shared pointers to the newly created DatasetOps
  std::vector<std::shared_ptr<DatasetOp>> Build() override;

  /// \brief Parameters validation
  /// \return bool true if all the params are valid
  bool ValidateParams() override;

 private:
  int32_t batch_size_;
  bool drop_remainder_;
  bool pad_;
  std::vector<std::string> cols_to_map_;
  std::map<std::string, std::pair<TensorShape, std::shared_ptr<Tensor>>> pad_map_;
};

class MapDataset : public Dataset {
 public:
  /// \brief Constructor
  MapDataset(std::vector<std::shared_ptr<TensorOperation>> operations, std::vector<std::string> input_columns = {},
             std::vector<std::string> output_columns = {}, const std::vector<std::string> &columns = {});

  /// \brief Destructor
  ~MapDataset() = default;

  /// \brief a base class override function to create the required runtime dataset op objects for this class
  /// \return The list of shared pointers to the newly created DatasetOps
  std::vector<std::shared_ptr<DatasetOp>> Build() override;

  /// \brief Parameters validation
  /// \return bool true if all the params are valid
  bool ValidateParams() override;

 private:
  std::vector<std::shared_ptr<TensorOperation>> operations_;
  std::vector<std::string> input_columns_;
  std::vector<std::string> output_columns_;
  std::vector<std::string> project_columns_;
};

class ProjectDataset : public Dataset {
 public:
  /// \brief Constructor
  explicit ProjectDataset(const std::vector<std::string> &columns);

  /// \brief Destructor
  ~ProjectDataset() = default;

  /// \brief a base class override function to create the required runtime dataset op objects for this class
  /// \return The list of shared pointers to the newly created DatasetOps
  std::vector<std::shared_ptr<DatasetOp>> Build() override;

  /// \brief Parameters validation
  /// \return bool true if all the params are valid
  bool ValidateParams() override;

 private:
  std::vector<std::string> columns_;
};

class RenameDataset : public Dataset {
 public:
  /// \brief Constructor
  explicit RenameDataset(const std::vector<std::string> &input_columns, const std::vector<std::string> &output_columns);

  /// \brief Destructor
  ~RenameDataset() = default;

  /// \brief a base class override function to create the required runtime dataset op objects for this class
  /// \return The list of shared pointers to the newly created DatasetOps
  std::vector<std::shared_ptr<DatasetOp>> Build() override;

  /// \brief Parameters validation
  /// \return bool true if all the params are valid
  bool ValidateParams() override;

 private:
  std::vector<std::string> input_columns_;
  std::vector<std::string> output_columns_;
};

class RepeatDataset : public Dataset {
 public:
  /// \brief Constructor
  explicit RepeatDataset(uint32_t count);

  /// \brief Destructor
  ~RepeatDataset() = default;

  /// \brief a base class override function to create the required runtime dataset op objects for this class
  /// \return The list of shared pointers to the newly created DatasetOps
  std::vector<std::shared_ptr<DatasetOp>> Build() override;

  /// \brief Parameters validation
  /// \return bool true if all the params are valid
  bool ValidateParams() override;

 private:
  uint32_t repeat_count_;
};

class ShuffleDataset : public Dataset {
 public:
  ShuffleDataset(int32_t shuffle_size, bool reset_every_epoch);

  ~ShuffleDataset() = default;

  std::vector<std::shared_ptr<DatasetOp>> Build() override;

  bool ValidateParams() override;

 private:
  int32_t shuffle_size_;
  uint32_t shuffle_seed_;
  bool reset_every_epoch_;
};

class SkipDataset : public Dataset {
 public:
  /// \brief Constructor
  explicit SkipDataset(int32_t count);

  /// \brief Destructor
  ~SkipDataset() = default;

  /// \brief a base class override function to create the required runtime dataset op objects for this class
  /// \return The list of shared pointers to the newly created DatasetOps
  std::vector<std::shared_ptr<DatasetOp>> Build() override;

  /// \brief Parameters validation
  /// \return bool true if all the params are valid
  bool ValidateParams() override;

 private:
  int32_t skip_count_;
};

class ZipDataset : public Dataset {
 public:
  /// \brief Constructor
  ZipDataset();

  /// \brief Destructor
  ~ZipDataset() = default;

  /// \brief a base class override function to create the required runtime dataset op objects for this class
  /// \return The list of shared pointers to the newly created DatasetOps
  std::vector<std::shared_ptr<DatasetOp>> Build() override;

  /// \brief Parameters validation
  /// \return bool true if all the params are valid
  bool ValidateParams() override;
};

}  // namespace api
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_DATASETS_H_
