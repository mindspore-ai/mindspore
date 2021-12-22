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

#include "minddata/dataset/api/python/pybind_conversion.h"
#include "minddata/dataset/api/python/pybind_register.h"
#include "minddata/dataset/include/dataset/constants.h"
#include "minddata/dataset/include/dataset/datasets.h"

#include "minddata/dataset/core/config_manager.h"
#include "minddata/dataset/core/data_type.h"
#include "minddata/dataset/util/path.h"

// IR leaf nodes
#include "minddata/dataset/engine/ir/datasetops/source/ag_news_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/celeba_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/cifar100_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/cifar10_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/cityscapes_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/clue_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/coco_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/conll2000_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/csv_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/dbpedia_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/div2k_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/emnist_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/fake_image_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/fashion_mnist_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/flickr_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/generator_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/image_folder_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/iwslt2016_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/iwslt2017_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/kmnist_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/mnist_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/penn_treebank_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/random_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/speech_commands_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/stl10_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/tedlium_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/text_file_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/udpos_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/yahoo_answers_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/yelp_review_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/yes_no_node.h"

// IR leaf nodes disabled for android
#ifndef ENABLE_ANDROID
#include "minddata/dataset/engine/ir/datasetops/source/lj_speech_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/manifest_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/minddata_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/photo_tour_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/places365_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/qmnist_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/sbu_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/sogou_news_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/tf_record_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/usps_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/voc_node.h"
#endif

namespace mindspore {
namespace dataset {

// PYBIND FOR LEAF NODES
// (In alphabetical order)

PYBIND_REGISTER(AGNewsNode, 2, ([](const py::module *m) {
                  (void)py::class_<AGNewsNode, DatasetNode, std::shared_ptr<AGNewsNode>>(*m, "AGNewsNode",
                                                                                         "to create an AGNewsNode")
                    .def(py::init([](std::string dataset_dir, std::string usage, int64_t num_samples, int32_t shuffle,
                                     int32_t num_shards, int32_t shard_id) {
                      auto ag_news = std::make_shared<AGNewsNode>(dataset_dir, num_samples, toShuffleMode(shuffle),
                                                                  usage, num_shards, shard_id, nullptr);
                      THROW_IF_ERROR(ag_news->ValidateParams());
                      return ag_news;
                    }));
                }));

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

PYBIND_REGISTER(CityscapesNode, 2, ([](const py::module *m) {
                  (void)py::class_<CityscapesNode, DatasetNode, std::shared_ptr<CityscapesNode>>(
                    *m, "CityscapesNode", "to create a CityscapesNode")
                    .def(py::init([](std::string dataset_dir, std::string usage, std::string quality_mode,
                                     std::string task, bool decode, const py::handle &sampler) {
                      auto cityscapes = std::make_shared<CityscapesNode>(dataset_dir, usage, quality_mode, task, decode,
                                                                         toSamplerObj(sampler), nullptr);
                      THROW_IF_ERROR(cityscapes->ValidateParams());
                      return cityscapes;
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
                                     bool decode, const py::handle &sampler, bool extra_metadata) {
                      std::shared_ptr<CocoNode> coco = std::make_shared<CocoNode>(
                        dataset_dir, annotation_file, task, decode, toSamplerObj(sampler), nullptr, extra_metadata);
                      THROW_IF_ERROR(coco->ValidateParams());
                      return coco;
                    }));
                }));

PYBIND_REGISTER(CoNLL2000Node, 2, ([](const py::module *m) {
                  (void)py::class_<CoNLL2000Node, DatasetNode, std::shared_ptr<CoNLL2000Node>>(
                    *m, "CoNLL2000Node", "to create a CoNLL2000Node")
                    .def(py::init([](std::string dataset_dir, std::string usage, int64_t num_samples, int32_t shuffle,
                                     int32_t num_shards, int32_t shard_id) {
                      std::shared_ptr<CoNLL2000Node> conll2000 = std::make_shared<CoNLL2000Node>(
                        dataset_dir, usage, num_samples, toShuffleMode(shuffle), num_shards, shard_id, nullptr);
                      THROW_IF_ERROR(conll2000->ValidateParams());
                      return conll2000;
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

PYBIND_REGISTER(DBpediaNode, 2, ([](const py::module *m) {
                  (void)py::class_<DBpediaNode, DatasetNode, std::shared_ptr<DBpediaNode>>(*m, "DBpediaNode",
                                                                                           "to create a DBpediaNode")
                    .def(py::init([](std::string dataset_dir, std::string usage, int64_t num_samples, int32_t shuffle,
                                     int32_t num_shards, int32_t shard_id) {
                      auto dbpedia = std::make_shared<DBpediaNode>(
                        dataset_dir, usage, num_samples, toShuffleMode(shuffle), num_shards, shard_id, nullptr);
                      THROW_IF_ERROR(dbpedia->ValidateParams());
                      return dbpedia;
                    }));
                }));

PYBIND_REGISTER(DIV2KNode, 2, ([](const py::module *m) {
                  (void)py::class_<DIV2KNode, DatasetNode, std::shared_ptr<DIV2KNode>>(*m, "DIV2KNode",
                                                                                       "to create a DIV2KNode")
                    .def(py::init([](std::string dataset_dir, std::string usage, std::string downgrade, int32_t scale,
                                     bool decode, py::handle sampler) {
                      auto div2k = std::make_shared<DIV2KNode>(dataset_dir, usage, downgrade, scale, decode,
                                                               toSamplerObj(sampler), nullptr);
                      THROW_IF_ERROR(div2k->ValidateParams());
                      return div2k;
                    }));
                }));

PYBIND_REGISTER(EMnistNode, 2, ([](const py::module *m) {
                  (void)py::class_<EMnistNode, DatasetNode, std::shared_ptr<EMnistNode>>(*m, "EMnistNode",
                                                                                         "to create an EMnistNode")
                    .def(py::init([](std::string dataset_dir, std::string name, std::string usage, py::handle sampler) {
                      auto emnist =
                        std::make_shared<EMnistNode>(dataset_dir, name, usage, toSamplerObj(sampler), nullptr);
                      THROW_IF_ERROR(emnist->ValidateParams());
                      return emnist;
                    }));
                }));

PYBIND_REGISTER(FakeImageNode, 2, ([](const py::module *m) {
                  (void)py::class_<FakeImageNode, DatasetNode, std::shared_ptr<FakeImageNode>>(
                    *m, "FakeImageNode", "to create a FakeImageNode")
                    .def(py::init([](int32_t num_images, const std::vector<int32_t> image_size, int32_t num_classes,
                                     int32_t base_seed, py::handle sampler) {
                      auto fake_image = std::make_shared<FakeImageNode>(num_images, image_size, num_classes, base_seed,
                                                                        toSamplerObj(sampler), nullptr);
                      THROW_IF_ERROR(fake_image->ValidateParams());
                      return fake_image;
                    }));
                }));

PYBIND_REGISTER(FashionMnistNode, 2, ([](const py::module *m) {
                  (void)py::class_<FashionMnistNode, DatasetNode, std::shared_ptr<FashionMnistNode>>(
                    *m, "FashionMnistNode", "to create a FashionMnistNode")
                    .def(py::init([](std::string dataset_dir, std::string usage, py::handle sampler) {
                      auto fashion_mnist =
                        std::make_shared<FashionMnistNode>(dataset_dir, usage, toSamplerObj(sampler), nullptr);
                      THROW_IF_ERROR(fashion_mnist->ValidateParams());
                      return fashion_mnist;
                    }));
                }));

PYBIND_REGISTER(
  FlickrNode, 2, ([](const py::module *m) {
    (void)py::class_<FlickrNode, DatasetNode, std::shared_ptr<FlickrNode>>(*m, "FlickrNode", "to create a FlickrNode")
      .def(py::init([](std::string dataset_dir, std::string annotation_file, bool decode, const py::handle &sampler) {
        auto flickr =
          std::make_shared<FlickrNode>(dataset_dir, annotation_file, decode, toSamplerObj(sampler), nullptr);
        THROW_IF_ERROR(flickr->ValidateParams());
        return flickr;
      }));
  }));

PYBIND_REGISTER(GeneratorNode, 2, ([](const py::module *m) {
                  (void)py::class_<GeneratorNode, DatasetNode, std::shared_ptr<GeneratorNode>>(
                    *m, "GeneratorNode", "to create a GeneratorNode")
                    .def(py::init([](py::function generator_function, const std::vector<std::string> &column_names,
                                     const std::vector<DataType> &column_types, int64_t dataset_len, py::handle sampler,
                                     uint32_t num_parallel_workers) {
                      auto gen =
                        std::make_shared<GeneratorNode>(generator_function, column_names, column_types, dataset_len,
                                                        toSamplerObj(sampler), num_parallel_workers);
                      THROW_IF_ERROR(gen->ValidateParams());
                      return gen;
                    }))
                    .def(py::init([](py::function generator_function, const std::shared_ptr<SchemaObj> schema,
                                     int64_t dataset_len, py::handle sampler, uint32_t num_parallel_workers) {
                      auto gen = std::make_shared<GeneratorNode>(generator_function, schema, dataset_len,
                                                                 toSamplerObj(sampler), num_parallel_workers);
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

PYBIND_REGISTER(IWSLT2016Node, 2, ([](const py::module *m) {
                  (void)py::class_<IWSLT2016Node, DatasetNode, std::shared_ptr<IWSLT2016Node>>(
                    *m, "IWSLT2016Node", "to create an IWSLT2016Node")
                    .def(py::init([](std::string dataset_dir, std::string usage, std::vector<std::string> language_pair,
                                     std::string valid_set, std::string test_set, int64_t num_samples, int32_t shuffle,
                                     int32_t num_shards, int32_t shard_id) {
                      std::shared_ptr<IWSLT2016Node> iwslt2016 = std::make_shared<IWSLT2016Node>(
                        dataset_dir, usage, language_pair, valid_set, test_set, num_samples, toShuffleMode(shuffle),
                        num_shards, shard_id, nullptr);
                      THROW_IF_ERROR(iwslt2016->ValidateParams());
                      return iwslt2016;
                    }));
                }));

PYBIND_REGISTER(IWSLT2017Node, 2, ([](const py::module *m) {
                  (void)py::class_<IWSLT2017Node, DatasetNode, std::shared_ptr<IWSLT2017Node>>(
                    *m, "IWSLT2017Node", "to create an IWSLT2017Node")
                    .def(py::init([](std::string dataset_dir, std::string usage, std::vector<std::string> language_pair,
                                     int64_t num_samples, int32_t shuffle, int32_t num_shards, int32_t shard_id) {
                      std::shared_ptr<IWSLT2017Node> iwslt2017 =
                        std::make_shared<IWSLT2017Node>(dataset_dir, usage, language_pair, num_samples,
                                                        toShuffleMode(shuffle), num_shards, shard_id, nullptr);
                      THROW_IF_ERROR(iwslt2017->ValidateParams());
                      return iwslt2017;
                    }));
                }));

PYBIND_REGISTER(KMnistNode, 2, ([](const py::module *m) {
                  (void)py::class_<KMnistNode, DatasetNode, std::shared_ptr<KMnistNode>>(*m, "KMnistNode",
                                                                                         "to create a KMnistNode")
                    .def(py::init([](std::string dataset_dir, std::string usage, py::handle sampler) {
                      auto kmnist = std::make_shared<KMnistNode>(dataset_dir, usage, toSamplerObj(sampler), nullptr);
                      THROW_IF_ERROR(kmnist->ValidateParams());
                      return kmnist;
                    }));
                }));

PYBIND_REGISTER(LJSpeechNode, 2, ([](const py::module *m) {
                  (void)py::class_<LJSpeechNode, DatasetNode, std::shared_ptr<LJSpeechNode>>(*m, "LJSpeechNode",
                                                                                             "to create a LJSpeechNode")
                    .def(py::init([](std::string dataset_dir, py::handle sampler) {
                      auto lj_speech = std::make_shared<LJSpeechNode>(dataset_dir, toSamplerObj(sampler), nullptr);
                      THROW_IF_ERROR(lj_speech->ValidateParams());
                      return lj_speech;
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
                                     const py::dict &padded_sample, int64_t num_padded, ShuffleMode shuffle_mode) {
                      nlohmann::json padded_sample_json;
                      std::map<std::string, std::string> sample_bytes;
                      THROW_IF_ERROR(ToJson(padded_sample, &padded_sample_json, &sample_bytes));
                      auto minddata = std::make_shared<MindDataNode>(dataset_file, toStringVector(columns_list),
                                                                     toSamplerObj(sampler, true), padded_sample_json,
                                                                     num_padded, shuffle_mode, nullptr);
                      minddata->SetSampleBytes(&sample_bytes);
                      THROW_IF_ERROR(minddata->ValidateParams());
                      return minddata;
                    }))
                    .def(py::init([](py::list dataset_file, py::list columns_list, py::handle sampler,
                                     const py::dict &padded_sample, int64_t num_padded, ShuffleMode shuffle_mode) {
                      nlohmann::json padded_sample_json;
                      std::map<std::string, std::string> sample_bytes;
                      THROW_IF_ERROR(ToJson(padded_sample, &padded_sample_json, &sample_bytes));
                      auto minddata = std::make_shared<MindDataNode>(
                        toStringVector(dataset_file), toStringVector(columns_list), toSamplerObj(sampler, true),
                        padded_sample_json, num_padded, shuffle_mode, nullptr);
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

PYBIND_REGISTER(PennTreebankNode, 2, ([](const py::module *m) {
                  (void)py::class_<PennTreebankNode, DatasetNode, std::shared_ptr<PennTreebankNode>>(
                    *m, "PennTreebankNode", "to create a PennTreebankNode")
                    .def(py::init([](std::string dataset_dir, std::string usage, int32_t num_samples, int32_t shuffle,
                                     int32_t num_shards, int32_t shard_id) {
                      auto penn_treebank = std::make_shared<PennTreebankNode>(
                        dataset_dir, usage, num_samples, toShuffleMode(shuffle), num_shards, shard_id, nullptr);
                      THROW_IF_ERROR(penn_treebank->ValidateParams());
                      return penn_treebank;
                    }));
                }));

PYBIND_REGISTER(PhotoTourNode, 2, ([](const py::module *m) {
                  (void)py::class_<PhotoTourNode, DatasetNode, std::shared_ptr<PhotoTourNode>>(
                    *m, "PhotoTourNode", "to create a PhotoTourNode")
                    .def(py::init([](std::string dataset_dir, std::string name, std::string usage, py::handle sampler) {
                      auto photo_tour =
                        std::make_shared<PhotoTourNode>(dataset_dir, name, usage, toSamplerObj(sampler), nullptr);
                      THROW_IF_ERROR(photo_tour->ValidateParams());
                      return photo_tour;
                    }));
                }));

PYBIND_REGISTER(Places365Node, 2, ([](const py::module *m) {
                  (void)py::class_<Places365Node, DatasetNode, std::shared_ptr<Places365Node>>(
                    *m, "Places365Node", "to create a Places365Node")
                    .def(py::init(
                      [](std::string dataset_dir, std::string usage, bool small, bool decode, py::handle sampler) {
                        auto places365 = std::make_shared<Places365Node>(dataset_dir, usage, small, decode,
                                                                         toSamplerObj(sampler), nullptr);
                        THROW_IF_ERROR(places365->ValidateParams());
                        return places365;
                      }));
                }));

PYBIND_REGISTER(QMnistNode, 2, ([](const py::module *m) {
                  (void)py::class_<QMnistNode, DatasetNode, std::shared_ptr<QMnistNode>>(*m, "QMnistNode",
                                                                                         "to create a QMnistNode")
                    .def(py::init([](std::string dataset_dir, std::string usage, bool compat, py::handle sampler) {
                      auto qmnist =
                        std::make_shared<QMnistNode>(dataset_dir, usage, compat, toSamplerObj(sampler), nullptr);
                      THROW_IF_ERROR(qmnist->ValidateParams());
                      return qmnist;
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

PYBIND_REGISTER(SBUNode, 2, ([](const py::module *m) {
                  (void)py::class_<SBUNode, DatasetNode, std::shared_ptr<SBUNode>>(*m, "SBUNode",
                                                                                   "to create an SBUNode")
                    .def(py::init([](std::string dataset_dir, bool decode, const py::handle &sampler) {
                      auto sbu = std::make_shared<SBUNode>(dataset_dir, decode, toSamplerObj(sampler), nullptr);
                      THROW_IF_ERROR(sbu->ValidateParams());
                      return sbu;
                    }));
                }));

PYBIND_REGISTER(SogouNewsNode, 2, ([](const py::module *m) {
                  (void)py::class_<SogouNewsNode, DatasetNode, std::shared_ptr<SogouNewsNode>>(
                    *m, "SogouNewsNode", "to create a SogouNewsNode")
                    .def(py::init([](std::string dataset_dir, std::string usage, int64_t num_samples, int32_t shuffle,
                                     int32_t num_shards, int32_t shard_id) {
                      auto sogou_news = std::make_shared<SogouNewsNode>(
                        dataset_dir, usage, num_samples, toShuffleMode(shuffle), num_shards, shard_id, nullptr);
                      THROW_IF_ERROR(sogou_news->ValidateParams());
                      return sogou_news;
                    }));
                }));

PYBIND_REGISTER(SpeechCommandsNode, 2, ([](const py::module *m) {
                  (void)py::class_<SpeechCommandsNode, DatasetNode, std::shared_ptr<SpeechCommandsNode>>(
                    *m, "SpeechCommandsNode", "to create a SpeechCommandsNode")
                    .def(py::init([](std::string dataset_dir, std::string usage, py::handle sampler) {
                      auto speech_commands =
                        std::make_shared<SpeechCommandsNode>(dataset_dir, usage, toSamplerObj(sampler), nullptr);
                      THROW_IF_ERROR(speech_commands->ValidateParams());
                      return speech_commands;
                    }));
                }));

PYBIND_REGISTER(STL10Node, 2, ([](const py::module *m) {
                  (void)py::class_<STL10Node, DatasetNode, std::shared_ptr<STL10Node>>(*m, "STL10Node",
                                                                                       "to create a STL10Node")
                    .def(py::init([](std::string dataset_dir, std::string usage, py::handle sampler) {
                      auto stl10 = std::make_shared<STL10Node>(dataset_dir, usage, toSamplerObj(sampler), nullptr);
                      THROW_IF_ERROR(stl10->ValidateParams());
                      return stl10;
                    }));
                }));

PYBIND_REGISTER(TedliumNode, 2, ([](const py::module *m) {
                  (void)py::class_<TedliumNode, DatasetNode, std::shared_ptr<TedliumNode>>(*m, "TedliumNode",
                                                                                           "to create a TedliumNode")
                    .def(py::init([](std::string dataset_dir, std::string release, std::string usage,
                                     std::string extensions, py::handle sampler) {
                      auto tedlium = std::make_shared<TedliumNode>(dataset_dir, release, usage, extensions,
                                                                   toSamplerObj(sampler), nullptr);
                      THROW_IF_ERROR(tedlium->ValidateParams());
                      return tedlium;
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
                    .def(py::init([](const py::list dataset_files, std::shared_ptr<SchemaObj> schema,
                                     const py::list columns_list, int64_t num_samples, int32_t shuffle,
                                     int32_t num_shards, int32_t shard_id, bool shard_equal_rows) {
                      std::shared_ptr<TFRecordNode> tfrecord = std::make_shared<TFRecordNode>(
                        toStringVector(dataset_files), schema, toStringVector(columns_list), num_samples,
                        toShuffleMode(shuffle), num_shards, shard_id, shard_equal_rows, nullptr);
                      THROW_IF_ERROR(tfrecord->ValidateParams());
                      return tfrecord;
                    }))
                    .def(py::init([](const py::list dataset_files, std::string schema, const py::list columns_list,
                                     int64_t num_samples, int32_t shuffle, int32_t num_shards, int32_t shard_id,
                                     bool shard_equal_rows) {
                      std::shared_ptr<TFRecordNode> tfrecord = std::make_shared<TFRecordNode>(
                        toStringVector(dataset_files), schema, toStringVector(columns_list), num_samples,
                        toShuffleMode(shuffle), num_shards, shard_id, shard_equal_rows, nullptr);
                      THROW_IF_ERROR(tfrecord->ValidateParams());
                      return tfrecord;
                    }));
                }));

PYBIND_REGISTER(UDPOSNode, 2, ([](const py::module *m) {
                  (void)py::class_<UDPOSNode, DatasetNode, std::shared_ptr<UDPOSNode>>(*m, "UDPOSNode",
                                                                                       "to create an UDPOSNode")
                    .def(py::init([](std::string dataset_dir, std::string usage, int64_t num_samples, int32_t shuffle,
                                     int32_t num_shards, int32_t shard_id) {
                      std::shared_ptr<UDPOSNode> udpos = std::make_shared<UDPOSNode>(
                        dataset_dir, usage, num_samples, toShuffleMode(shuffle), num_shards, shard_id, nullptr);
                      THROW_IF_ERROR(udpos->ValidateParams());
                      return udpos;
                    }));
                }));

PYBIND_REGISTER(USPSNode, 2, ([](const py::module *m) {
                  (void)py::class_<USPSNode, DatasetNode, std::shared_ptr<USPSNode>>(*m, "USPSNode",
                                                                                     "to create an USPSNode")
                    .def(py::init([](std::string dataset_dir, std::string usage, int32_t num_samples, int32_t shuffle,
                                     int32_t num_shards, int32_t shard_id) {
                      auto usps = std::make_shared<USPSNode>(dataset_dir, usage, num_samples, toShuffleMode(shuffle),
                                                             num_shards, shard_id, nullptr);
                      THROW_IF_ERROR(usps->ValidateParams());
                      return usps;
                    }));
                }));

PYBIND_REGISTER(VOCNode, 2, ([](const py::module *m) {
                  (void)py::class_<VOCNode, DatasetNode, std::shared_ptr<VOCNode>>(*m, "VOCNode", "to create a VOCNode")
                    .def(py::init([](std::string dataset_dir, std::string task, std::string usage,
                                     const py::dict &class_indexing, bool decode, const py::handle &sampler,
                                     bool extra_metadata) {
                      std::shared_ptr<VOCNode> voc =
                        std::make_shared<VOCNode>(dataset_dir, task, usage, toStringMap(class_indexing), decode,
                                                  toSamplerObj(sampler), nullptr, extra_metadata);
                      THROW_IF_ERROR(voc->ValidateParams());
                      return voc;
                    }));
                }));

PYBIND_REGISTER(YahooAnswersNode, 2, ([](const py::module *m) {
                  (void)py::class_<YahooAnswersNode, DatasetNode, std::shared_ptr<YahooAnswersNode>>(
                    *m, "YahooAnswersNode", "to create a YahooAnswersNode")
                    .def(py::init([](std::string dataset_dir, std::string usage, int64_t num_samples, int32_t shuffle,
                                     int32_t num_shards, int32_t shard_id) {
                      auto yahoo_answers = std::make_shared<YahooAnswersNode>(
                        dataset_dir, usage, num_samples, toShuffleMode(shuffle), num_shards, shard_id, nullptr);
                      THROW_IF_ERROR(yahoo_answers->ValidateParams());
                      return yahoo_answers;
                    }));
                }));

PYBIND_REGISTER(YelpReviewNode, 2, ([](const py::module *m) {
                  (void)py::class_<YelpReviewNode, DatasetNode, std::shared_ptr<YelpReviewNode>>(
                    *m, "YelpReviewNode", "to create a YelpReviewNode")
                    .def(py::init([](std::string dataset_dir, std::string usage, int64_t num_samples, int32_t shuffle,
                                     int32_t num_shards, int32_t shard_id) {
                      std::shared_ptr<YelpReviewNode> yelp_review = std::make_shared<YelpReviewNode>(
                        dataset_dir, usage, num_samples, toShuffleMode(shuffle), num_shards, shard_id, nullptr);
                      THROW_IF_ERROR(yelp_review->ValidateParams());
                      return yelp_review;
                    }));
                }));

PYBIND_REGISTER(YesNoNode, 2, ([](const py::module *m) {
                  (void)py::class_<YesNoNode, DatasetNode, std::shared_ptr<YesNoNode>>(*m, "YesNoNode",
                                                                                       "to create a YesNoNode")
                    .def(py::init([](std::string dataset_dir, py::handle sampler) {
                      auto yes_no = std::make_shared<YesNoNode>(dataset_dir, toSamplerObj(sampler), nullptr);
                      THROW_IF_ERROR(yes_no->ValidateParams());
                      return yes_no;
                    }));
                }));

}  // namespace dataset
}  // namespace mindspore
