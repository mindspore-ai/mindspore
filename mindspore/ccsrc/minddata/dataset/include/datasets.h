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

#include <unistd.h>
#include <vector>
#include <memory>
#include <set>
#include <map>
#include <utility>
#include <string>
#include "minddata/dataset/core/constants.h"
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
class CelebADataset;
class Cifar10Dataset;
class Cifar100Dataset;
class CLUEDataset;
class CocoDataset;
class ImageFolderDataset;
class MnistDataset;
class TextFileDataset;
class VOCDataset;
// Dataset Op classes (in alphabetical order)
class BatchDataset;
class ConcatDataset;
class MapDataset;
class ProjectDataset;
class RenameDataset;
class RepeatDataset;
class ShuffleDataset;
class SkipDataset;
class TakeDataset;
class ZipDataset;

/// \brief Function to create a CelebADataset
/// \notes The generated dataset has two columns ['image', 'attr'].
//     The type of the image tensor is uint8. The attr tensor is uint32 and one hot type.
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] dataset_type One of 'all', 'train', 'valid' or 'test'.
/// \param[in] decode Decode the images after reading (default=False).
/// \param[in] extensions List of file extensions to be included in the dataset (default=None).
/// \param[in] sampler Object used to choose samples from the dataset. If sampler is `nullptr`, A `RandomSampler`
///    will be used to randomly iterate the entire dataset
/// \return Shared pointer to the current Dataset
std::shared_ptr<CelebADataset> CelebA(const std::string &dataset_dir, const std::string &dataset_type = "all",
                                      const std::shared_ptr<SamplerObj> &sampler = nullptr, bool decode = false,
                                      const std::set<std::string> &extensions = {});

/// \brief Function to create a Cifar10 Dataset
/// \notes The generated dataset has two columns ['image', 'label']
/// \param[in] dataset_dir Path to the root directory that contains the dataset
/// \param[in] sampler Object used to choose samples from the dataset. If sampler is `nullptr`, A `RandomSampler`
///    will be used to randomly iterate the entire dataset
/// \return Shared pointer to the current Dataset
std::shared_ptr<Cifar10Dataset> Cifar10(const std::string &dataset_dir,
                                        const std::shared_ptr<SamplerObj> &sampler = nullptr);

/// \brief Function to create a Cifar100 Dataset
/// \notes The generated dataset has three columns ['image', 'coarse_label', 'fine_label']
/// \param[in] dataset_dir Path to the root directory that contains the dataset
/// \param[in] sampler Object used to choose samples from the dataset. If sampler is `nullptr`, A `RandomSampler`
///    will be used to randomly iterate the entire dataset
/// \return Shared pointer to the current Dataset
std::shared_ptr<Cifar100Dataset> Cifar100(const std::string &dataset_dir,
                                          const std::shared_ptr<SamplerObj> &sampler = nullptr);

/// \brief Function to create a CLUEDataset
/// \notes The generated dataset has a variable number of columns depending on the task and usage
/// \param[in] dataset_files List of files to be read to search for a pattern of files. The list
///    will be sorted in a lexicographical order.
/// \param[in] task The kind of task, one of "AFQMC", "TNEWS", "IFLYTEK", "CMNLI", "WSC" and "CSL" (default="AFQMC").
/// \param[in] usage Be used to "train", "test" or "eval" data (default="train").
/// \param[in] num_samples The number of samples to be included in the dataset.
///    (Default = 0 means all samples.)
/// \param[in] shuffle The mode for shuffling data every epoch. (Default=ShuffleMode.kGlobal)
///    Can be any of:
///    ShuffleMode.kFalse - No shuffling is performed.
///    ShuffleMode.kFiles - Shuffle files only.
///    ShuffleMode.kGlobal - Shuffle both the files and samples.
/// \param[in] num_shards Number of shards that the dataset should be divided into. (Default = 1)
/// \param[in] shard_id  The shard ID within num_shards. This argument should be
///    specified only when num_shards is also specified. (Default = 0)
/// \return Shared pointer to the current CLUEDataset
std::shared_ptr<CLUEDataset> CLUE(const std::vector<std::string> &dataset_files, const std::string &task = "AFQMC",
                                  const std::string &usage = "train", int64_t num_samples = 0,
                                  ShuffleMode shuffle = ShuffleMode::kGlobal, int32_t num_shards = 1,
                                  int32_t shard_id = 0);

/// \brief Function to create a CocoDataset
/// \notes The generated dataset has multi-columns :
///        - task='Detection', column: [['image', dtype=uint8], ['bbox', dtype=float32], ['category_id', dtype=uint32],
///                                     ['iscrowd', dtype=uint32]].
///        - task='Stuff', column: [['image', dtype=uint8], ['segmentation',dtype=float32], ['iscrowd', dtype=uint32]].
///        - task='Keypoint', column: [['image', dtype=uint8], ['keypoints', dtype=float32],
///                                    ['num_keypoints', dtype=uint32]].
///        - task='Panoptic', column: [['image', dtype=uint8], ['bbox', dtype=float32], ['category_id', dtype=uint32],
///                                    ['iscrowd', dtype=uint32], ['area', dtype=uitn32]].
/// \param[in] dataset_dir Path to the root directory that contains the dataset
/// \param[in] annotation_file Path to the annotation json
/// \param[in] task Set the task type of reading coco data, now support 'Detection'/'Stuff'/'Panoptic'/'Keypoint'
/// \param[in] decode Decode the images after reading
/// \param[in] sampler Object used to choose samples from the dataset. If sampler is `nullptr`, A `RandomSampler`
///    will be used to randomly iterate the entire dataset
/// \return Shared pointer to the current Dataset
std::shared_ptr<CocoDataset> Coco(const std::string &dataset_dir, const std::string &annotation_file,
                                  const std::string &task = "Detection", const bool &decode = false,
                                  const std::shared_ptr<SamplerObj> &sampler = nullptr);

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
std::shared_ptr<ImageFolderDataset> ImageFolder(const std::string &dataset_dir, bool decode = false,
                                                const std::shared_ptr<SamplerObj> &sampler = nullptr,
                                                const std::set<std::string> &extensions = {},
                                                const std::map<std::string, int32_t> &class_indexing = {});

/// \brief Function to create a MnistDataset
/// \notes The generated dataset has two columns ['image', 'label']
/// \param[in] dataset_dir Path to the root directory that contains the dataset
/// \param[in] sampler Object used to choose samples from the dataset. If sampler is `nullptr`,
///    A `RandomSampler` will be used to randomly iterate the entire dataset
/// \return Shared pointer to the current MnistDataset
std::shared_ptr<MnistDataset> Mnist(const std::string &dataset_dir,
                                    const std::shared_ptr<SamplerObj> &sampler = nullptr);

/// \brief Function to create a ConcatDataset
/// \notes Reload "+" operator to concat two datasets
/// \param[in] datasets1 Shared pointer to the first dataset to be concatenated
/// \param[in] datasets2 Shared pointer to the second dataset to be concatenated
/// \return Shared pointer to the current ConcatDataset
std::shared_ptr<ConcatDataset> operator+(const std::shared_ptr<Dataset> &datasets1,
                                         const std::shared_ptr<Dataset> &datasets2);

/// \brief Function to create a TextFileDataset
/// \notes The generated dataset has one column ['text']
/// \param[in] dataset_files List of files to be read to search for a pattern of files. The list
///    will be sorted in a lexicographical order.
/// \param[in] num_samples The number of samples to be included in the dataset.
///    (Default = 0 means all samples.)
/// \param[in] shuffle The mode for shuffling data every epoch. (Default=ShuffleMode.kGlobal)
///    Can be any of:
///    ShuffleMode.kFalse - No shuffling is performed.
///    ShuffleMode.kFiles - Shuffle files only.
///    ShuffleMode.kGlobal - Shuffle both the files and samples.
/// \param[in] num_shards Number of shards that the dataset should be divided into. (Default = 1)
/// \param[in] shard_id  The shard ID within num_shards. This argument should be
///    specified only when num_shards is also specified. (Default = 0)
/// \return Shared pointer to the current TextFileDataset
std::shared_ptr<TextFileDataset> TextFile(const std::vector<std::string> &dataset_files, int32_t num_samples = 0,
                                          ShuffleMode shuffle = ShuffleMode::kGlobal, int32_t num_shards = 1,
                                          int32_t shard_id = 0);

/// \brief Function to create a VOCDataset
/// \notes The generated dataset has multi-columns :
///        - task='Detection', column: [['image', dtype=uint8], ['bbox', dtype=float32], ['label', dtype=uint32],
///                                     ['difficult', dtype=uint32], ['truncate', dtype=uint32]].
///        - task='Segmentation', column: [['image', dtype=uint8], ['target',dtype=uint8]].
/// \param[in] dataset_dir Path to the root directory that contains the dataset
/// \param[in] task Set the task type of reading voc data, now only support "Segmentation" or "Detection"
/// \param[in] mode Set the data list txt file to be readed
/// \param[in] class_indexing A str-to-int mapping from label name to index
/// \param[in] decode Decode the images after reading
/// \param[in] sampler Object used to choose samples from the dataset. If sampler is `nullptr`, A `RandomSampler`
///    will be used to randomly iterate the entire dataset
/// \return Shared pointer to the current Dataset
std::shared_ptr<VOCDataset> VOC(const std::string &dataset_dir, const std::string &task = "Segmentation",
                                const std::string &mode = "train",
                                const std::map<std::string, int32_t> &class_indexing = {}, bool decode = false,
                                const std::shared_ptr<SamplerObj> &sampler = nullptr);

/// \brief Function to create a ZipDataset
/// \notes Applies zip to the dataset
/// \param[in] datasets List of shared pointers to the datasets that we want to zip
/// \return Shared pointer to the current Dataset
std::shared_ptr<ZipDataset> Zip(const std::vector<std::shared_ptr<Dataset>> &datasets);

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
  /// \param[in] columns List of columns to be used to specify the order of columns
  /// \return Shared pointer to the Iterator
  std::shared_ptr<Iterator> CreateIterator(std::vector<std::string> columns = {});

  /// \brief Function to create a BatchDataset
  /// \notes Combines batch_size number of consecutive rows into batches
  /// \param[in] batch_size Path to the root directory that contains the dataset
  /// \param[in] drop_remainder Determines whether or not to drop the last possibly incomplete
  ///    batch. If true, and if there are less than batch_size rows
  ///    available to make the last batch, then those rows will
  ///    be dropped and not propagated to the next node
  /// \return Shared pointer to the current BatchDataset
  std::shared_ptr<BatchDataset> Batch(int32_t batch_size, bool drop_remainder = false);

  /// \brief Function to create a ConcatDataset
  /// \notes Concat the datasets in the input
  /// \param[in] datasets List of shared pointers to the dataset that should be concatenated together
  /// \return Shared pointer to the current ConcatDataset
  std::shared_ptr<ConcatDataset> Concat(const std::vector<std::shared_ptr<Dataset>> &datasets);

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
  std::shared_ptr<ShuffleDataset> Shuffle(int32_t buffer_size);

  /// \brief Function to create a SkipDataset
  /// \notes Skips count elements in this dataset.
  /// \param[in] count Number of elements the dataset to be skipped.
  /// \return Shared pointer to the current SkipDataset
  std::shared_ptr<SkipDataset> Skip(int32_t count);

  /// \brief Function to create a TakeDataset
  /// \notes Takes count elements in this dataset.
  /// \param[in] count Number of elements the dataset to be taken.
  /// \return Shared pointer to the current Dataset
  std::shared_ptr<Dataset> Take(int32_t count = -1);

  /// \brief Function to create a Zip Dataset
  /// \notes Applies zip to the dataset
  /// \param[in] datasets A list of shared pointers to the datasets that we want to zip
  /// \return Shared pointer to the current Dataset
  std::shared_ptr<ZipDataset> Zip(const std::vector<std::shared_ptr<Dataset>> &datasets);

 protected:
  std::vector<std::shared_ptr<Dataset>> children;
  std::shared_ptr<Dataset> parent;

  int32_t num_workers_;
  int32_t rows_per_buffer_;
  int32_t connector_que_size_;
  int32_t worker_connector_size_;
};

/* ####################################### Derived Dataset classes ################################# */

// DERIVED DATASET CLASSES FOR LEAF-NODE DATASETS
// (In alphabetical order)

class CelebADataset : public Dataset {
 public:
  /// \brief Constructor
  CelebADataset(const std::string &dataset_dir, const std::string &dataset_type,
                const std::shared_ptr<SamplerObj> &sampler, const bool &decode,
                const std::set<std::string> &extensions);

  /// \brief Destructor
  ~CelebADataset() = default;

  /// \brief a base class override function to create the required runtime dataset op objects for this class
  /// \return shared pointer to the list of newly created DatasetOps
  std::vector<std::shared_ptr<DatasetOp>> Build() override;

  /// \brief Parameters validation
  /// \return bool true if all the params are valid
  bool ValidateParams() override;

 private:
  std::string dataset_dir_;
  std::string dataset_type_;
  bool decode_;
  std::set<std::string> extensions_;
  std::shared_ptr<SamplerObj> sampler_;
};
// DERIVED DATASET CLASSES FOR LEAF-NODE DATASETS
// (In alphabetical order)

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

/// \class CLUEDataset
/// \brief A Dataset derived class to represent CLUE dataset
class CLUEDataset : public Dataset {
 public:
  /// \brief Constructor
  CLUEDataset(const std::vector<std::string> dataset_files, std::string task, std::string usage, int64_t num_samples,
              ShuffleMode shuffle, int32_t num_shards, int32_t shard_id);

  /// \brief Destructor
  ~CLUEDataset() = default;

  /// \brief a base class override function to create the required runtime dataset op objects for this class
  /// \return The list of shared pointers to the newly created DatasetOps
  std::vector<std::shared_ptr<DatasetOp>> Build() override;

  /// \brief Parameters validation
  /// \return bool true if all the params are valid
  bool ValidateParams() override;

 private:
  /// \brief Split string based on a character delimiter
  /// \return A string vector
  std::vector<std::string> split(const std::string &s, char delim);

  std::vector<std::string> dataset_files_;
  std::string task_;
  std::string usage_;
  int64_t num_samples_;
  ShuffleMode shuffle_;
  int32_t num_shards_;
  int32_t shard_id_;
};

class CocoDataset : public Dataset {
 public:
  /// \brief Constructor
  CocoDataset(const std::string &dataset_dir, const std::string &annotation_file, const std::string &task,
              const bool &decode, const std::shared_ptr<SamplerObj> &sampler);

  /// \brief Destructor
  ~CocoDataset() = default;

  /// \brief a base class override function to create the required runtime dataset op objects for this class
  /// \return shared pointer to the list of newly created DatasetOps
  std::vector<std::shared_ptr<DatasetOp>> Build() override;

  /// \brief Parameters validation
  /// \return bool true if all the params are valid
  bool ValidateParams() override;

 private:
  std::string dataset_dir_;
  std::string annotation_file_;
  std::string task_;
  bool decode_;
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

/// \class TextFileDataset
/// \brief A Dataset derived class to represent TextFile dataset
class TextFileDataset : public Dataset {
 public:
  /// \brief Constructor
  TextFileDataset(std::vector<std::string> dataset_files, int32_t num_samples, ShuffleMode shuffle, int32_t num_shards,
                  int32_t shard_id);

  /// \brief Destructor
  ~TextFileDataset() = default;

  /// \brief a base class override function to create the required runtime dataset op objects for this class
  /// \return The list of shared pointers to the newly created DatasetOps
  std::vector<std::shared_ptr<DatasetOp>> Build() override;

  /// \brief Parameters validation
  /// \return bool true if all the params are valid
  bool ValidateParams() override;

 private:
  std::vector<std::string> dataset_files_;
  int32_t num_samples_;
  int32_t num_shards_;
  int32_t shard_id_;
  ShuffleMode shuffle_;
};

class VOCDataset : public Dataset {
 public:
  /// \brief Constructor
  VOCDataset(const std::string &dataset_dir, const std::string &task, const std::string &mode,
             const std::map<std::string, int32_t> &class_indexing, bool decode, std::shared_ptr<SamplerObj> sampler);

  /// \brief Destructor
  ~VOCDataset() = default;

  /// \brief a base class override function to create the required runtime dataset op objects for this class
  /// \return shared pointer to the list of newly created DatasetOps
  std::vector<std::shared_ptr<DatasetOp>> Build() override;

  /// \brief Parameters validation
  /// \return bool true if all the params are valid
  bool ValidateParams() override;

 private:
  const std::string kColumnImage = "image";
  const std::string kColumnTarget = "target";
  const std::string kColumnBbox = "bbox";
  const std::string kColumnLabel = "label";
  const std::string kColumnDifficult = "difficult";
  const std::string kColumnTruncate = "truncate";
  std::string dataset_dir_;
  std::string task_;
  std::string mode_;
  std::map<std::string, int32_t> class_index_;
  bool decode_;
  std::shared_ptr<SamplerObj> sampler_;
};

// DERIVED DATASET CLASSES FOR DATASET OPS
// (In alphabetical order)

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

class ConcatDataset : public Dataset {
 public:
  /// \brief Constructor
  explicit ConcatDataset(const std::vector<std::shared_ptr<Dataset>> &datasets);

  /// \brief Destructor
  ~ConcatDataset() = default;

  /// \brief a base class override function to create the required runtime dataset op objects for this class
  /// \return The list of shared pointers to the newly created DatasetOps
  std::vector<std::shared_ptr<DatasetOp>> Build() override;

  /// \brief Parameters validation
  /// \return bool true if all the params are valid
  bool ValidateParams() override;

 private:
  std::vector<std::shared_ptr<Dataset>> datasets_;
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
  explicit RepeatDataset(int32_t count);

  /// \brief Destructor
  ~RepeatDataset() = default;

  /// \brief a base class override function to create the required runtime dataset op objects for this class
  /// \return The list of shared pointers to the newly created DatasetOps
  std::vector<std::shared_ptr<DatasetOp>> Build() override;

  /// \brief Parameters validation
  /// \return bool true if all the params are valid
  bool ValidateParams() override;

 private:
  int32_t repeat_count_;
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

class TakeDataset : public Dataset {
 public:
  /// \brief Constructor
  explicit TakeDataset(int32_t count);

  /// \brief Destructor
  ~TakeDataset() = default;

  /// \brief a base class override function to create the required runtime dataset op objects for this class
  /// \return shared pointer to the list of newly created DatasetOps
  std::vector<std::shared_ptr<DatasetOp>> Build() override;

  /// \brief Parameters validation
  /// \return bool true if all the params are valid
  bool ValidateParams() override;

 private:
  int32_t take_count_;
};

class ZipDataset : public Dataset {
 public:
  /// \brief Constructor
  explicit ZipDataset(const std::vector<std::shared_ptr<Dataset>> &datasets);

  /// \brief Destructor
  ~ZipDataset() = default;

  /// \brief a base class override function to create the required runtime dataset op objects for this class
  /// \return The list of shared pointers to the newly created DatasetOps
  std::vector<std::shared_ptr<DatasetOp>> Build() override;

  /// \brief Parameters validation
  /// \return bool true if all the params are valid
  bool ValidateParams() override;

 private:
  std::vector<std::shared_ptr<Dataset>> datasets_;
};

}  // namespace api
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_DATASETS_H_
