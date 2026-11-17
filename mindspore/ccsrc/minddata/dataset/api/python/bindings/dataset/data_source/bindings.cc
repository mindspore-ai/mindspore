/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#include "pybind11/pybind11.h"

#include "minddata/dataset/api/python/pybind_conversion.h"
#include "minddata/dataset/api/python/pybind_register.h"
#include "minddata/dataset/api/python/python_mp.h"
#include "minddata/dataset/core/config_manager.h"
#include "minddata/dataset/core/data_type.h"
#include "minddata/dataset/include/dataset/constants.h"
#include "minddata/dataset/include/dataset/datasets.h"

// IR leaf nodes
#include "minddata/dataset/data_source/operation/csv_node.h"
#include "minddata/dataset/data_source/operation/generator_node.h"
#include "minddata/dataset/data_source/operation/image_folder_node.h"
#include "minddata/dataset/data_source/operation/random_node.h"
#include "minddata/dataset/data_source/operation/text_file_node.h"

// IR leaf nodes disabled for android
#include "minddata/dataset/data_source/operation/manifest_node.h"
#include "minddata/dataset/data_source/operation/minddata_node.h"
#include "minddata/dataset/data_source/operation/tf_record_node.h"

namespace mindspore {
namespace dataset {
// PYBIND FOR LEAF NODES
// (In alphabetical order)
PYBIND_REGISTER(CSVNode, 2, ([](const py::module *m) {
                  (void)py::class_<CSVNode, DatasetNode, std::shared_ptr<CSVNode>>(*m, "CSVNode", "to create a CSVNode")
                    .def(py::init([](const std::vector<std::string> &csv_files, char field_delim,
                                     const py::list &column_defaults, const std::vector<std::string> &column_names,
                                     int64_t num_samples, int32_t shuffle, int32_t num_shards, int32_t shard_id) {
                      auto csv =
                        std::make_shared<CSVNode>(csv_files, field_delim, toCSVBase(column_defaults), column_names,
                                                  num_samples, toShuffleMode(shuffle), num_shards, shard_id, nullptr);
                      THROW_IF_ERROR(csv->ValidateParams());
                      return csv;
                    }));
                }));

PYBIND_REGISTER(GeneratorNode, 2, ([](const py::module *m) {
                  (void)py::class_<GeneratorNode, DatasetNode, std::shared_ptr<GeneratorNode>>(
                    *m, "GeneratorNode", "to create a GeneratorNode")
                    .def(
                      py::init([](const py::function &generator_function, const std::vector<std::string> &column_names,
                                  const std::vector<DataType> &column_types, int64_t dataset_len,
                                  const py::handle &sampler, uint32_t num_parallel_workers,
                                  std::shared_ptr<PythonMultiprocessingRuntime> python_multiprocessing_runtime,
                                  bool has_batch_sampler) {
                        auto gen = std::make_shared<GeneratorNode>(
                          generator_function, column_names, column_types, dataset_len, toSamplerObj(sampler),
                          num_parallel_workers, std::move(python_multiprocessing_runtime), has_batch_sampler);
                        THROW_IF_ERROR(gen->ValidateParams());
                        return gen;
                      }))
                    .def(py::init([](const py::function &generator_function, const std::shared_ptr<SchemaObj> &schema,
                                     int64_t dataset_len, const py::handle &sampler, uint32_t num_parallel_workers,
                                     std::shared_ptr<PythonMultiprocessingRuntime> python_multiprocessing_runtime,
                                     bool has_batch_sampler) {
                      auto gen = std::make_shared<GeneratorNode>(
                        generator_function, schema, dataset_len, toSamplerObj(sampler), num_parallel_workers,
                        std::move(python_multiprocessing_runtime), has_batch_sampler);
                      THROW_IF_ERROR(gen->ValidateParams());
                      return gen;
                    }))
                    .def("Build", [](GeneratorNode &self) {
                      std::vector<std::shared_ptr<DatasetOp>> node_ops;
                      THROW_IF_ERROR(self.Build(&node_ops));
                      std::shared_ptr<GeneratorOp> generator_op = std::dynamic_pointer_cast<GeneratorOp>(node_ops[0]);
                      THROW_IF_ERROR(generator_op->CalculatedSampleId());
                      return generator_op;
                    });
                }));

PYBIND_REGISTER(GeneratorOp, 0, ([](const py::module *m) {
                  (void)py::class_<GeneratorOp, std::shared_ptr<GeneratorOp>>(*m, "GeneratorOp")
                    .def("GetMappedIndex", [](GeneratorOp &self, int64_t index) {
                      int64_t sample_id;
                      THROW_IF_ERROR(self.GetMappedIndex(index, &sample_id));
                      return sample_id;
                    });
                }));

PYBIND_REGISTER(ImageFolderNode, 2, ([](const py::module *m) {
                  (void)py::class_<ImageFolderNode, DatasetNode, std::shared_ptr<ImageFolderNode>>(
                    *m, "ImageFolderNode", "to create an ImageFolderNode")
                    .def(py::init([](const std::string &dataset_dir, bool decode, const py::handle &sampler,
                                     const py::list &extensions, const py::dict &class_indexing,
                                     const py::object &decrypt_obj) {
                      // Don't update recursive to true
                      bool recursive = false;  // Will be removed in future PR
                      py::function decrypt =
                        py::isinstance<py::function>(decrypt_obj) ? decrypt_obj.cast<py::function>() : py::function();
                      auto imagefolder = std::make_shared<ImageFolderNode>(
                        dataset_dir, decode, toSamplerObj(sampler), recursive, toStringSet(extensions),
                        toStringMap(class_indexing), nullptr, decrypt);
                      THROW_IF_ERROR(imagefolder->ValidateParams());
                      return imagefolder;
                    }));
                }));

PYBIND_REGISTER(ManifestNode, 2, ([](const py::module *m) {
                  (void)py::class_<ManifestNode, DatasetNode, std::shared_ptr<ManifestNode>>(*m, "ManifestNode",
                                                                                             "to create a ManifestNode")
                    .def(py::init([](const std::string &dataset_file, const std::string &usage,
                                     const py::handle &sampler, const py::dict &class_indexing, bool decode) {
                      auto manifest = std::make_shared<ManifestNode>(dataset_file, usage, toSamplerObj(sampler),
                                                                     toStringMap(class_indexing), decode, nullptr);
                      THROW_IF_ERROR(manifest->ValidateParams());
                      return manifest;
                    }));
                }));

PYBIND_REGISTER(
  MindDataNode, 2, ([](const py::module *m) {
    (void)py::class_<MindDataNode, DatasetNode, std::shared_ptr<MindDataNode>>(*m, "MindDataNode",
                                                                               "to create a MindDataNode")
      .def(py::init([](const std::string &dataset_file, const py::list &columns_list, const py::handle &sampler,
                       const py::dict &padded_sample, int64_t num_padded, ShuffleMode shuffle_mode) {
        nlohmann::json padded_sample_json;
        std::map<std::string, std::string> sample_bytes;
        THROW_IF_ERROR(ToJson(padded_sample, &padded_sample_json, &sample_bytes));
        auto minddata =
          std::make_shared<MindDataNode>(dataset_file, toStringVector(columns_list), toSamplerObj(sampler, true),
                                         padded_sample_json, num_padded, shuffle_mode, nullptr);
        minddata->SetSampleBytes(&sample_bytes);
        THROW_IF_ERROR(minddata->ValidateParams());
        return minddata;
      }))
      .def(py::init([](const py::list &dataset_file, const py::list &columns_list, const py::handle &sampler,
                       const py::dict &padded_sample, int64_t num_padded, ShuffleMode shuffle_mode) {
        nlohmann::json padded_sample_json;
        std::map<std::string, std::string> sample_bytes;
        THROW_IF_ERROR(ToJson(padded_sample, &padded_sample_json, &sample_bytes));
        auto minddata = std::make_shared<MindDataNode>(toStringVector(dataset_file), toStringVector(columns_list),
                                                       toSamplerObj(sampler, true), padded_sample_json, num_padded,
                                                       shuffle_mode, nullptr);
        minddata->SetSampleBytes(&sample_bytes);
        THROW_IF_ERROR(minddata->ValidateParams());
        return minddata;
      }))
      .def("Build", [](MindDataNode &self) {
        std::vector<std::shared_ptr<DatasetOp>> node_ops;
        THROW_IF_ERROR(self.Build(&node_ops));
        auto mindrecord_op = std::dynamic_pointer_cast<MindRecordOp>(node_ops[0]);
        THROW_IF_ERROR(mindrecord_op->ShardReaderLaunch());
        return mindrecord_op;
      });
  }));

PYBIND_REGISTER(MindRecordOp, 0, ([](const py::module *m) {
                  (void)py::class_<MindRecordOp, std::shared_ptr<MindRecordOp>>(*m, "MindRecordOp")
                    .def("__getitem__", [](MindRecordOp &self, size_t index) {
                      auto out = self[index];
                      std::vector<TensorPtr> row;
                      row = out.getRow();
                      {
                        py::gil_scoped_acquire gil_acquire;
                        py::list result;
                        for (const auto &el : row) {
                          result.append(el);
                        }
                        return result;
                      }
                    });
                }));

PYBIND_REGISTER(
  RandomNode, 2, ([](const py::module *m) {
    (void)py::class_<RandomNode, DatasetNode, std::shared_ptr<RandomNode>>(*m, "RandomNode", "to create a RandomNode")
      .def(py::init([](int32_t total_rows, const std::shared_ptr<SchemaObj> &schema, const py::list &columns_list) {
        auto random_node = std::make_shared<RandomNode>(total_rows, schema, toStringVector(columns_list), nullptr);
        THROW_IF_ERROR(random_node->ValidateParams());
        return random_node;
      }))
      .def(py::init([](int32_t total_rows, const std::string &schema, const py::list &columns_list) {
        auto random_node = std::make_shared<RandomNode>(total_rows, schema, toStringVector(columns_list), nullptr);
        THROW_IF_ERROR(random_node->ValidateParams());
        return random_node;
      }));
  }));

PYBIND_REGISTER(TextFileNode, 2, ([](const py::module *m) {
                  (void)py::class_<TextFileNode, DatasetNode, std::shared_ptr<TextFileNode>>(*m, "TextFileNode",
                                                                                             "to create a TextFileNode")
                    .def(py::init([](const py::list &dataset_files, int32_t num_samples, int32_t shuffle,
                                     int32_t num_shards, int32_t shard_id) {
                      std::shared_ptr<TextFileNode> textfile_node =
                        std::make_shared<TextFileNode>(toStringVector(dataset_files), num_samples,
                                                       toShuffleMode(shuffle), num_shards, shard_id, nullptr);
                      THROW_IF_ERROR(textfile_node->ValidateParams());
                      return textfile_node;
                    }));
                }));

PYBIND_REGISTER(
  TFRecordNode, 2, ([](const py::module *m) {
    (void)py::class_<TFRecordNode, DatasetNode, std::shared_ptr<TFRecordNode>>(*m, "TFRecordNode",
                                                                               "to create a TFRecordNode")
      .def(py::init([](const py::list &dataset_files, const std::shared_ptr<SchemaObj> &schema,
                       const py::list &columns_list, int64_t num_samples, int32_t shuffle, int32_t num_shards,
                       int32_t shard_id, bool shard_equal_rows, const std::string &compression_type) {
        std::shared_ptr<TFRecordNode> tfrecord = std::make_shared<TFRecordNode>(
          toStringVector(dataset_files), schema, toStringVector(columns_list), num_samples, toShuffleMode(shuffle),
          num_shards, shard_id, shard_equal_rows, nullptr, compression_type);
        THROW_IF_ERROR(tfrecord->ValidateParams());
        return tfrecord;
      }))
      .def(py::init([](const py::list &dataset_files, const std::string &schema, const py::list &columns_list,
                       int64_t num_samples, int32_t shuffle, int32_t num_shards, int32_t shard_id,
                       bool shard_equal_rows, const std::string &compression_type) {
        std::shared_ptr<TFRecordNode> tfrecord = std::make_shared<TFRecordNode>(
          toStringVector(dataset_files), schema, toStringVector(columns_list), num_samples, toShuffleMode(shuffle),
          num_shards, shard_id, shard_equal_rows, nullptr, compression_type);
        THROW_IF_ERROR(tfrecord->ValidateParams());
        return tfrecord;
      }));
  }));

}  // namespace dataset
}  // namespace mindspore
