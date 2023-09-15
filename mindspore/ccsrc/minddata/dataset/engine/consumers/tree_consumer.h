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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_CONSUMERS_TREE_CONSUMER_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_CONSUMERS_TREE_CONSUMER_H_

#include <memory>
#include <string>
#include <map>
#include <unordered_map>
#include <utility>
#include <vector>

#include "minddata/dataset/engine/tree_adapter.h"
#include "minddata/dataset/include/dataset/text.h"

namespace mindspore::dataset {
// Forward declare
class TreeAdapter;
class DatasetNode;
#ifndef ENABLE_SECURITY
class AutoTune;
class ProfilingManager;
#endif
/// A base class for tree consumers which would fetch rows from the tree pipeline
class TreeConsumer {
 public:
  /// Constructor that prepares an empty tree_adapter
  TreeConsumer();

  explicit TreeConsumer(int32_t num_epochs);

  /// \brief Destructor
  virtual ~TreeConsumer() = default;

  /// Initializes the consumer, this involves constructing and preparing the tree.
  /// \param root The dataset node that represent the root of the IR tree.
  /// \return Status code.
  virtual Status Init(const std::shared_ptr<DatasetNode> &root);

  /// Initializes the consumer, this involves constructing and preparing the tree.
  /// \param root The dataset node that represent the root of the IR tree.
  /// \param global_step The global step to initialize from.
  /// \param dataset_size The number of steps that one epoch has.
  /// \return Status code.
  virtual Status Init(const std::shared_ptr<DatasetNode> &root, int64_t global_step, int64_t dataset_size);

  /// Internal function to perform the termination
  /// \return Status error code
  virtual Status Terminate();

  /// Function for all consumers to get the offload JSON string.
  /// \return Offload JSON string.
  std::string GetOffload();

  /// Function to reset the current consumer to the provided step.
  /// The consumer will terminate the pipeline and create a new one with skip injected.
  /// \param step The step to reset the pipeline to.
  /// \param dataset_size The number of steps that one epoch has.
  /// \return Status error code
  virtual Status Reset(int64_t step, int64_t dataset_size);

  /// Function to stop the consumer.
  /// \return Status error code
  virtual Status Stop() { return Status::OK(); }

#ifndef ENABLE_SECURITY
  virtual Status RegisterProfilingManager();

  /// \brief Getter for profiling manager, no ownership
  ProfilingManager *GetProfilingManager() { return profiling_manager_.get(); }

  /// \brief Return profiling manager pointer
  std::shared_ptr<ProfilingManager> GetProfilingManagerPtr() { return profiling_manager_; }

  Status InitAutoTune();
#endif

  /// Returns the next row in a vector format
  /// \param[out] out std::vector of Tensors
  /// \return Status error code
  virtual Status GetNextAsVector(std::vector<TensorPtr> *const out) { return Status::OK(); }

  /// Returns the next row in as a map
  /// \param[out] out std::map of string to Tensor
  /// \return Status error code
  virtual Status GetNextAsMap(std::unordered_map<std::string, TensorPtr> *const out) { return Status::OK(); }

  /// Returns the next row in as a vector
  /// \param[out] out std::vector of pairs of string to Tensor
  /// \return Status error code
  virtual Status GetNextAsOrderedPair(std::vector<std::pair<std::string, std::shared_ptr<Tensor>>> *const vec) {
    return Status::OK();
  }

 protected:
  /// The class owns the tree_adapter that handles execution tree operations.
  std::unique_ptr<TreeAdapter> tree_adapter_;

#ifndef ENABLE_SECURITY
  /// Profiling Manager
  std::shared_ptr<ProfilingManager> profiling_manager_;
  std::shared_ptr<AutoTune> autotune_;
#endif

  /// Method to return the name of the consumer
  /// \return string
  virtual std::string Name() = 0;

  int32_t num_epochs_;
};

/// Consumer that iterates over the dataset and returns the rows one by one as a vector or a map
class IteratorConsumer : public TreeConsumer {
 public:
  /// Constructor which will call the base class default constructor.
  /// \param num_epochs number of epochs. Default to -1 (infinite epochs).
  explicit IteratorConsumer(int32_t num_epochs = -1) : TreeConsumer(num_epochs) {}

  ~IteratorConsumer() override = default;

  Status Init(const std::shared_ptr<DatasetNode> &root, int64_t global_step = 0, int64_t dataset_size = -1) override;

  /// Returns the next row in a vector format
  /// \param[out] out std::vector of Tensors
  /// \return Status error code
  Status GetNextAsVector(std::vector<TensorPtr> *out) override;

  /// Returns the next row in as a map
  /// \param[out] out std::map of string to Tensor
  /// \return Status error code
  Status GetNextAsMap(std::unordered_map<std::string, TensorPtr> *out) override;

  /// Returns the next row in as a vector
  /// \param[out] out std::vector of pairs of string to Tensor
  /// \return Status error code
  Status GetNextAsOrderedPair(std::vector<std::pair<std::string, std::shared_ptr<Tensor>>> *vec) override;

  Status RegisterProfilingManager() override;

 protected:
  /// Method to return the name of the consumer
  /// \return string
  std::string Name() override { return "IteratorConsumer"; }

 private:
  std::map<int32_t, std::string> column_order_;  // key: column id, val: column name
};

#ifndef ENABLE_ANDROID
/// Consumer that iterates over the dataset and writes it to disk
class SaveToDisk : public TreeConsumer {
 public:
  /// Constructor which will call the base class default constructor.
  /// \param dataset_path path the the dataset
  /// \param num_files number of files. Default to 1
  /// \param dataset_type The format of the dataset. Default to "mindrecod".
  explicit SaveToDisk(const std::string &dataset_path, int32_t num_files = 1,
                      const std::string &dataset_type = "mindrecord")
      : TreeConsumer(), dataset_path_(dataset_path), num_files_(num_files), dataset_type_(dataset_type) {}

  ~SaveToDisk() override = default;

  /// \brief Parameters validation
  /// \return Status Status::OK() if all the parameters are valid
  Status ValidateParams();

  /// Save the given dataset to MindRecord format on disk. This is a blocking method (i.e., after returning, all rows
  /// would be written to disk)
  /// \return  Status error code
  virtual Status Save();

 protected:
  /// Method to return the name of the consumer
  /// \return string
  std::string Name() override { return "SaveToDisk"; }

 private:
  template <typename T, typename S>
  Status TransformTensor(const unsigned char *src, const TensorShape &shape, int64_t num_of_elements,
                         std::unique_ptr<T> *data, std::unique_ptr<std::vector<uint8_t>> *data_ptr,
                         std::unique_ptr<S> *s, bool need_convert = false);

  Status FetchMetaFromTensorRow(const std::unordered_map<std::string, int32_t> &column_name_id_map,
                                const TensorRow &row, nlohmann::json *schema, std::vector<std::string> *index_fields);

  Status FetchDataFromTensorRow(const TensorRow &row,
                                const std::unordered_map<std::string, int32_t> &column_name_id_map,
                                nlohmann::json *row_raw_data,
                                std::map<std::string, std::unique_ptr<std::vector<uint8_t>>> *row_bin_data);

  Status FetchFloatData(std::shared_ptr<Tensor> tensor, std::string column_name, nlohmann::json *row_raw_data,
                        std::unique_ptr<std::vector<uint8_t>> *data_ptr);

  Status FetchItemData(std::shared_ptr<Tensor> tensor, std::string column_name, nlohmann::json *row_raw_data,
                       std::map<std::string, std::unique_ptr<std::vector<uint8_t>>> *row_bin_data);

  template <typename T>
  bool map_compare(T const &lhs, T const &rhs);

  Status CheckTensorRowShapes(const std::unordered_map<std::string, int32_t> &column_name_id_map, const TensorRow &row,
                              std::map<std::string, std::vector<int>> *PreTensorRowShapes_ptr);

  std::string dataset_path_;
  int32_t num_files_;
  std::string dataset_type_;
};
#endif

/// Consumer that iterates over the dataset and send it to a device
class ToDevice : public TreeConsumer {
 public:
  explicit ToDevice(int32_t num_epochs = -1) : TreeConsumer(num_epochs) {}

  ~ToDevice() override = default;

  Status Init(const std::shared_ptr<DatasetNode> &root, int64_t global_step = 0, int64_t dataset_size = -1) override;

  Status RegisterProfilingManager() override;

  Status Terminate() override;

  /// Send the data to device
  /// \return  Status error code
  virtual Status Send();

  /// Stop to send data to device
  /// \return  Status error code
  Status Stop() override;

  /// Continue to send data to device
  /// \return  Status error code
  virtual Status Continue();

  /// Get data info from TDT
  /// \return  Status error code
  virtual Status GetDataInfo(std::vector<DataType> *types, std::vector<TensorShape> *shapes);

  /// Get send info in sink mode
  /// \return  Status error code
  virtual Status GetSendInfo(std::vector<std::vector<double>> *send_info);

 protected:
  /// Method to return the name of the consumer
  /// \return string
  std::string Name() override { return "ToDevice"; }
};

/// Consumer that is used to get some pipeline information
class DatasetSizeGetter : public TreeConsumer, public std::enable_shared_from_this<DatasetSizeGetter> {
 public:
  DatasetSizeGetter() : dataset_size_(-1) {}

  ~DatasetSizeGetter() override = default;

  Status Init(const std::shared_ptr<DatasetNode> &root) override;

  Status Terminate() override;

  /// \brief Function to get the dataset size
  /// \param[in] estimate This is only supported by some of the ops and it's used to speed up the process of getting
  ///     dataset size at the expense of accuracy.
  /// \param[out] dataset_size the size of the dataset
  /// \return Status of the function
  Status GetDatasetSize(int64_t *size, bool estimate = false);

  virtual Status GetRow(const std::shared_ptr<TreeAdapter> &tree_adapter, TensorRow *row);

  std::string Name() override { return "DatasetSizeGetter"; }

  /// \brief Gets the dataset size by iterating over the entire dataset on a sub tree starting from ir_node
  /// param[in] ir_node The node that marks the top most of the sub tree on which we want to iterate
  /// \return Status - The status code return
  Status DryRun(const std::shared_ptr<DatasetNode> &ir_node, int64_t *dataset_size);

 private:
  std::shared_ptr<DatasetNode> root_;
  std::vector<std::shared_ptr<TreeAdapter>> tree_adapters_;  // this is vector to handle different branch of zip
  int64_t dataset_size_;
};

class BuildVocabConsumer : public TreeConsumer {
 public:
  /// BuildVocabConsumer Constructor which will call the base class default constructor.
  BuildVocabConsumer() = default;

  ~BuildVocabConsumer() override = default;

  Status Init(const std::shared_ptr<DatasetNode> &root) override;

  /// Start consuming
  /// \return  Status error code
  virtual Status Start();

 protected:
  /// Method to return the name of the consumer
  /// \return string
  std::string Name() override { return "BuildVocab"; }
};
}  // namespace mindspore::dataset
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_CONSUMERS_TREE_CONSUMER_H_
