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
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/stl_bind.h"

#include "minddata/dataset/api/python/pybind_conversion.h"
#include "minddata/dataset/api/python/pybind_register.h"
#include "minddata/dataset/callback/py_ds_callback.h"
#include "minddata/dataset/include/constants.h"
#include "minddata/dataset/core/global_context.h"
#include "minddata/dataset/engine/serdes.h"
#include "minddata/dataset/include/datasets.h"
#include "minddata/dataset/text/sentence_piece_vocab.h"

#include "minddata/dataset/core/config_manager.h"
#include "minddata/dataset/core/data_type.h"
#include "minddata/dataset/util/path.h"
#include "minddata/dataset/util/random.h"
#include "minddata/dataset/util/services.h"

// IR leaf nodes
#include "minddata/dataset/engine/ir/datasetops/source/album_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/celeba_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/cifar100_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/cifar10_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/clue_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/coco_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/csv_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/generator_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/image_folder_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/mnist_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/random_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/text_file_node.h"

// IR leaf nodes disabled for android
#ifndef ENABLE_ANDROID
#include "minddata/dataset/engine/ir/datasetops/source/manifest_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/minddata_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/tf_record_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/voc_node.h"
#endif

namespace mindspore {
namespace dataset {

// PYBIND FOR LEAF NODES
// (In alphabetical order)

PYBIND_REGISTER(CelebANode, 2, ([](const py::module *m) {
                  (void)py::class_<CelebANode, DatasetNode, std::shared_ptr<CelebANode>>(*m, "CelebANode",
                                                                                         "to create a CelebANode")
                    .def(py::init([](std::string dataset_dir, std::string usage, py::handle sampler, bool decode,
                                     py::list extensions) {
                      auto celebA = std::make_shared<CelebANode>(dataset_dir, usage, toSamplerObj(sampler), decode,
                                                                 toStringSet(extensions), nullptr);
                      THROW_IF_ERROR(celebA->ValidateParams());
                      return celebA;
                    }));
                }));

PYBIND_REGISTER(Cifar10Node, 2, ([](const py::module *m) {
                  (void)py::class_<Cifar10Node, DatasetNode, std::shared_ptr<Cifar10Node>>(*m, "Cifar10Node",
                                                                                           "to create a Cifar10Node")
                    .def(py::init([](std::string dataset_dir, std::string usage, py::handle sampler) {
                      auto cifar10 = std::make_shared<Cifar10Node>(dataset_dir, usage, toSamplerObj(sampler), nullptr);
                      THROW_IF_ERROR(cifar10->ValidateParams());
                      return cifar10;
                    }));
                }));

PYBIND_REGISTER(Cifar100Node, 2, ([](const py::module *m) {
                  (void)py::class_<Cifar100Node, DatasetNode, std::shared_ptr<Cifar100Node>>(*m, "Cifar100Node",
                                                                                             "to create a Cifar100Node")
                    .def(py::init([](std::string dataset_dir, std::string usage, py::handle sampler) {
                      auto cifar100 =
                        std::make_shared<Cifar100Node>(dataset_dir, usage, toSamplerObj(sampler), nullptr);
                      THROW_IF_ERROR(cifar100->ValidateParams());
                      return cifar100;
                    }));
                }));

PYBIND_REGISTER(CLUENode, 2, ([](const py::module *m) {
                  (void)py::class_<CLUENode, DatasetNode, std::shared_ptr<CLUENode>>(*m, "CLUENode",
                                                                                     "to create a CLUENode")
                    .def(py::init([](py::list files, std::string task, std::string usage, int64_t num_samples,
                                     int32_t shuffle, int32_t num_shards, int32_t shard_id) {
                      std::shared_ptr<CLUENode> clue_node =
                        std::make_shared<dataset::CLUENode>(toStringVector(files), task, usage, num_samples,
                                                            toShuffleMode(shuffle), num_shards, shard_id, nullptr);
                      THROW_IF_ERROR(clue_node->ValidateParams());
                      return clue_node;
                    }));
                }));

PYBIND_REGISTER(CocoNode, 2, ([](const py::module *m) {
                  (void)py::class_<CocoNode, DatasetNode, std::shared_ptr<CocoNode>>(*m, "CocoNode",
                                                                                     "to create a CocoNode")
                    .def(py::init([](std::string dataset_dir, std::string annotation_file, std::string task,
                                     bool decode, py::handle sampler) {
                      std::shared_ptr<CocoNode> coco = std::make_shared<CocoNode>(
                        dataset_dir, annotation_file, task, decode, toSamplerObj(sampler), nullptr);
                      THROW_IF_ERROR(coco->ValidateParams());
                      return coco;
                    }));
                }));

PYBIND_REGISTER(CSVNode, 2, ([](const py::module *m) {
                  (void)py::class_<CSVNode, DatasetNode, std::shared_ptr<CSVNode>>(*m, "CSVNode", "to create a CSVNode")
                    .def(py::init([](std::vector<std::string> csv_files, char field_delim, py::list column_defaults,
                                     std::vector<std::string> column_names, int64_t num_samples, int32_t shuffle,
                                     int32_t num_shards, int32_t shard_id) {
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
                      py::init([](py::function generator_function, const std::vector<std::string> &column_names,
                                  const std::vector<DataType> &column_types, int64_t dataset_len, py::handle sampler) {
                        auto gen = std::make_shared<GeneratorNode>(generator_function, column_names, column_types,
                                                                   dataset_len, toSamplerObj(sampler));
                        THROW_IF_ERROR(gen->ValidateParams());
                        return gen;
                      }))
                    .def(py::init([](py::function generator_function, const std::shared_ptr<SchemaObj> schema,
                                     int64_t dataset_len, py::handle sampler) {
                      auto gen =
                        std::make_shared<GeneratorNode>(generator_function, schema, dataset_len, toSamplerObj(sampler));
                      THROW_IF_ERROR(gen->ValidateParams());
                      return gen;
                    }));
                }));

PYBIND_REGISTER(ImageFolderNode, 2, ([](const py::module *m) {
                  (void)py::class_<ImageFolderNode, DatasetNode, std::shared_ptr<ImageFolderNode>>(
                    *m, "ImageFolderNode", "to create an ImageFolderNode")
                    .def(py::init([](std::string dataset_dir, bool decode, py::handle sampler, py::list extensions,
                                     py::dict class_indexing) {
                      // Don't update recursive to true
                      bool recursive = false;  // Will be removed in future PR
                      auto imagefolder = std::make_shared<ImageFolderNode>(dataset_dir, decode, toSamplerObj(sampler),
                                                                           recursive, toStringSet(extensions),
                                                                           toStringMap(class_indexing), nullptr);
                      THROW_IF_ERROR(imagefolder->ValidateParams());
                      return imagefolder;
                    }));
                }));

PYBIND_REGISTER(ManifestNode, 2, ([](const py::module *m) {
                  (void)py::class_<ManifestNode, DatasetNode, std::shared_ptr<ManifestNode>>(*m, "ManifestNode",
                                                                                             "to create a ManifestNode")
                    .def(py::init([](std::string dataset_file, std::string usage, py::handle sampler,
                                     py::dict class_indexing, bool decode) {
                      auto manifest = std::make_shared<ManifestNode>(dataset_file, usage, toSamplerObj(sampler),
                                                                     toStringMap(class_indexing), decode, nullptr);
                      THROW_IF_ERROR(manifest->ValidateParams());
                      return manifest;
                    }));
                }));

PYBIND_REGISTER(MindDataNode, 2, ([](const py::module *m) {
                  (void)py::class_<MindDataNode, DatasetNode, std::shared_ptr<MindDataNode>>(*m, "MindDataNode",
                                                                                             "to create a MindDataNode")
                    .def(py::init([](std::string dataset_file, py::list columns_list, py::handle sampler,
                                     py::dict padded_sample, int64_t num_padded) {
                      nlohmann::json padded_sample_json;
                      std::map<std::string, std::string> sample_bytes;
                      THROW_IF_ERROR(ToJson(padded_sample, &padded_sample_json, &sample_bytes));
                      auto minddata =
                        std::make_shared<MindDataNode>(dataset_file, toStringVector(columns_list),
                                                       toSamplerObj(sampler, true), padded_sample_json, num_padded);
                      minddata->SetSampleBytes(&sample_bytes);
                      THROW_IF_ERROR(minddata->ValidateParams());
                      return minddata;
                    }))
                    .def(py::init([](py::list dataset_file, py::list columns_list, py::handle sampler,
                                     py::dict padded_sample, int64_t num_padded) {
                      nlohmann::json padded_sample_json;
                      std::map<std::string, std::string> sample_bytes;
                      THROW_IF_ERROR(ToJson(padded_sample, &padded_sample_json, &sample_bytes));
                      auto minddata =
                        std::make_shared<MindDataNode>(toStringVector(dataset_file), toStringVector(columns_list),
                                                       toSamplerObj(sampler, true), padded_sample_json, num_padded);
                      minddata->SetSampleBytes(&sample_bytes);
                      THROW_IF_ERROR(minddata->ValidateParams());
                      return minddata;
                    }));
                }));

PYBIND_REGISTER(MnistNode, 2, ([](const py::module *m) {
                  (void)py::class_<MnistNode, DatasetNode, std::shared_ptr<MnistNode>>(*m, "MnistNode",
                                                                                       "to create an MnistNode")
                    .def(py::init([](std::string dataset_dir, std::string usage, py::handle sampler) {
                      auto mnist = std::make_shared<MnistNode>(dataset_dir, usage, toSamplerObj(sampler), nullptr);
                      THROW_IF_ERROR(mnist->ValidateParams());
                      return mnist;
                    }));
                }));

PYBIND_REGISTER(RandomNode, 2, ([](const py::module *m) {
                  (void)py::class_<RandomNode, DatasetNode, std::shared_ptr<RandomNode>>(*m, "RandomNode",
                                                                                         "to create a RandomNode")
                    .def(py::init([](int32_t total_rows, std::shared_ptr<SchemaObj> schema, py::list columns_list) {
                      auto random_node =
                        std::make_shared<RandomNode>(total_rows, schema, toStringVector(columns_list), nullptr);
                      THROW_IF_ERROR(random_node->ValidateParams());
                      return random_node;
                    }))
                    .def(py::init([](int32_t total_rows, std::string schema, py::list columns_list) {
                      auto random_node =
                        std::make_shared<RandomNode>(total_rows, schema, toStringVector(columns_list), nullptr);
                      THROW_IF_ERROR(random_node->ValidateParams());
                      return random_node;
                    }));
                }));

PYBIND_REGISTER(TextFileNode, 2, ([](const py::module *m) {
                  (void)py::class_<TextFileNode, DatasetNode, std::shared_ptr<TextFileNode>>(*m, "TextFileNode",
                                                                                             "to create a TextFileNode")
                    .def(py::init([](py::list dataset_files, int32_t num_samples, int32_t shuffle, int32_t num_shards,
                                     int32_t shard_id) {
                      std::shared_ptr<TextFileNode> textfile_node =
                        std::make_shared<TextFileNode>(toStringVector(dataset_files), num_samples,
                                                       toShuffleMode(shuffle), num_shards, shard_id, nullptr);
                      THROW_IF_ERROR(textfile_node->ValidateParams());
                      return textfile_node;
                    }));
                }));

PYBIND_REGISTER(TFRecordNode, 2, ([](const py::module *m) {
                  (void)py::class_<TFRecordNode, DatasetNode, std::shared_ptr<TFRecordNode>>(*m, "TFRecordNode",
                                                                                             "to create a TFRecordNode")
                    .def(py::init([](py::list dataset_files, std::shared_ptr<SchemaObj> schema, py::list columns_list,
                                     int64_t num_samples, int32_t shuffle, int32_t num_shards, int32_t shard_id,
                                     bool shard_equal_rows) {
                      std::shared_ptr<TFRecordNode> tfrecord = std::make_shared<TFRecordNode>(
                        toStringVector(dataset_files), schema, toStringVector(columns_list), num_samples,
                        toShuffleMode(shuffle), num_shards, shard_id, shard_equal_rows, nullptr);
                      THROW_IF_ERROR(tfrecord->ValidateParams());
                      return tfrecord;
                    }))
                    .def(py::init([](py::list dataset_files, std::string schema, py::list columns_list,
                                     int64_t num_samples, int32_t shuffle, int32_t num_shards, int32_t shard_id,
                                     bool shard_equal_rows) {
                      std::shared_ptr<TFRecordNode> tfrecord = std::make_shared<TFRecordNode>(
                        toStringVector(dataset_files), schema, toStringVector(columns_list), num_samples,
                        toShuffleMode(shuffle), num_shards, shard_id, shard_equal_rows, nullptr);
                      THROW_IF_ERROR(tfrecord->ValidateParams());
                      return tfrecord;
                    }));
                }));

PYBIND_REGISTER(VOCNode, 2, ([](const py::module *m) {
                  (void)py::class_<VOCNode, DatasetNode, std::shared_ptr<VOCNode>>(*m, "VOCNode", "to create a VOCNode")
                    .def(py::init([](std::string dataset_dir, std::string task, std::string usage,
                                     py::dict class_indexing, bool decode, py::handle sampler) {
                      std::shared_ptr<VOCNode> voc = std::make_shared<VOCNode>(
                        dataset_dir, task, usage, toStringMap(class_indexing), decode, toSamplerObj(sampler), nullptr);
                      THROW_IF_ERROR(voc->ValidateParams());
                      return voc;
                    }));
                }));

}  // namespace dataset
}  // namespace mindspore
