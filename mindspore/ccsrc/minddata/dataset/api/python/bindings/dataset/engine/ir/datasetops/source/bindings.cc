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
#include "minddata/dataset/util/path.h"

// IR leaf nodes
#include "minddata/dataset/engine/ir/datasetops/source/ag_news_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/amazon_review_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/caltech256_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/celeba_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/cifar10_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/cifar100_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/cityscapes_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/clue_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/cmu_arctic_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/coco_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/conll2000_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/csv_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/dbpedia_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/div2k_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/emnist_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/en_wik9_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/fake_image_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/fashion_mnist_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/flickr_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/food101_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/generator_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/gtzan_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/image_folder_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/imdb_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/iwslt2016_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/iwslt2017_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/kmnist_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/lfw_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/libri_tts_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/mnist_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/penn_treebank_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/random_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/rendered_sst2_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/semeion_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/speech_commands_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/squad_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/sst2_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/stl10_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/sun397_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/tedlium_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/text_file_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/udpos_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/wiki_text_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/yahoo_answers_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/yelp_review_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/yes_no_node.h"

// IR leaf nodes disabled for android
#ifndef ENABLE_ANDROID
#include "minddata/dataset/engine/ir/datasetops/source/kitti_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/lj_speech_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/lsun_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/manifest_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/minddata_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/multi30k_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/omniglot_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/photo_tour_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/places365_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/qmnist_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/sbu_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/sogou_news_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/tf_record_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/usps_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/voc_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/wider_face_node.h"
#endif

namespace mindspore {
namespace dataset {
// PYBIND FOR LEAF NODES
// (In alphabetical order)
PYBIND_REGISTER(AGNewsNode, 2, ([](const py::module *m) {
                  (void)py::class_<AGNewsNode, DatasetNode, std::shared_ptr<AGNewsNode>>(*m, "AGNewsNode",
                                                                                         "to create an AGNewsNode")
                    .def(py::init([](const std::string &dataset_dir, const std::string &usage, int64_t num_samples,
                                     int32_t shuffle, int32_t num_shards, int32_t shard_id) {
                      auto ag_news = std::make_shared<AGNewsNode>(dataset_dir, num_samples, toShuffleMode(shuffle),
                                                                  usage, num_shards, shard_id, nullptr);
                      THROW_IF_ERROR(ag_news->ValidateParams());
                      return ag_news;
                    }));
                }));

PYBIND_REGISTER(AmazonReviewNode, 2, ([](const py::module *m) {
                  (void)py::class_<AmazonReviewNode, DatasetNode, std::shared_ptr<AmazonReviewNode>>(
                    *m, "AmazonReviewNode", "to create an AmazonReviewNode")
                    .def(py::init([](const std::string &dataset_dir, const std::string &usage, int64_t num_samples,
                                     int32_t shuffle, int32_t num_shards, int32_t shard_id) {
                      std::shared_ptr<AmazonReviewNode> amazon_review = std::make_shared<AmazonReviewNode>(
                        dataset_dir, usage, num_samples, toShuffleMode(shuffle), num_shards, shard_id, nullptr);
                      THROW_IF_ERROR(amazon_review->ValidateParams());
                      return amazon_review;
                    }));
                }));

PYBIND_REGISTER(Caltech256Node, 2, ([](const py::module *m) {
                  (void)py::class_<Caltech256Node, DatasetNode, std::shared_ptr<Caltech256Node>>(
                    *m, "Caltech256Node", "to create a Caltech256Node")
                    .def(py::init([](const std::string &dataset_dir, bool decode, const py::handle &sampler) {
                      auto caltech256 =
                        std::make_shared<Caltech256Node>(dataset_dir, decode, toSamplerObj(sampler), nullptr);
                      THROW_IF_ERROR(caltech256->ValidateParams());
                      return caltech256;
                    }));
                }));

PYBIND_REGISTER(CelebANode, 2, ([](const py::module *m) {
                  (void)py::class_<CelebANode, DatasetNode, std::shared_ptr<CelebANode>>(*m, "CelebANode",
                                                                                         "to create a CelebANode")
                    .def(
                      py::init([](const std::string &dataset_dir, const std::string &usage, const py::handle &sampler,
                                  bool decode, const py::list &extensions, const py::object &decrypt_obj) {
                        py::function decrypt =
                          py::isinstance<py::function>(decrypt_obj) ? decrypt_obj.cast<py::function>() : py::function();
                        auto celebA = std::make_shared<CelebANode>(dataset_dir, usage, toSamplerObj(sampler), decode,
                                                                   toStringSet(extensions), nullptr, decrypt);
                        THROW_IF_ERROR(celebA->ValidateParams());
                        return celebA;
                      }));
                }));

PYBIND_REGISTER(Cifar10Node, 2, ([](const py::module *m) {
                  (void)py::class_<Cifar10Node, DatasetNode, std::shared_ptr<Cifar10Node>>(*m, "Cifar10Node",
                                                                                           "to create a Cifar10Node")
                    .def(py::init([](const std::string &dataset_dir, const std::string &usage,
                                     const py::handle &sampler) {
                      auto cifar10 = std::make_shared<Cifar10Node>(dataset_dir, usage, toSamplerObj(sampler), nullptr);
                      THROW_IF_ERROR(cifar10->ValidateParams());
                      return cifar10;
                    }));
                }));

PYBIND_REGISTER(Cifar100Node, 2, ([](const py::module *m) {
                  (void)py::class_<Cifar100Node, DatasetNode, std::shared_ptr<Cifar100Node>>(*m, "Cifar100Node",
                                                                                             "to create a Cifar100Node")
                    .def(
                      py::init([](const std::string &dataset_dir, const std::string &usage, const py::handle &sampler) {
                        auto cifar100 =
                          std::make_shared<Cifar100Node>(dataset_dir, usage, toSamplerObj(sampler), nullptr);
                        THROW_IF_ERROR(cifar100->ValidateParams());
                        return cifar100;
                      }));
                }));

PYBIND_REGISTER(CityscapesNode, 2, ([](const py::module *m) {
                  (void)py::class_<CityscapesNode, DatasetNode, std::shared_ptr<CityscapesNode>>(
                    *m, "CityscapesNode", "to create a CityscapesNode")
                    .def(py::init([](const std::string &dataset_dir, const std::string &usage,
                                     const std::string &quality_mode, const std::string &task, bool decode,
                                     const py::handle &sampler) {
                      auto cityscapes = std::make_shared<CityscapesNode>(dataset_dir, usage, quality_mode, task, decode,
                                                                         toSamplerObj(sampler), nullptr);
                      THROW_IF_ERROR(cityscapes->ValidateParams());
                      return cityscapes;
                    }));
                }));

PYBIND_REGISTER(CLUENode, 2, ([](const py::module *m) {
                  (void)py::class_<CLUENode, DatasetNode, std::shared_ptr<CLUENode>>(*m, "CLUENode",
                                                                                     "to create a CLUENode")
                    .def(py::init([](const py::list &files, const std::string &task, const std::string &usage,
                                     int64_t num_samples, int32_t shuffle, int32_t num_shards, int32_t shard_id) {
                      std::shared_ptr<CLUENode> clue_node =
                        std::make_shared<dataset::CLUENode>(toStringVector(files), task, usage, num_samples,
                                                            toShuffleMode(shuffle), num_shards, shard_id, nullptr);
                      THROW_IF_ERROR(clue_node->ValidateParams());
                      return clue_node;
                    }));
                }));

PYBIND_REGISTER(CMUArcticNode, 2, ([](const py::module *m) {
                  (void)py::class_<CMUArcticNode, DatasetNode, std::shared_ptr<CMUArcticNode>>(
                    *m, "CMUArcticNode", "to create a CMUArcticNode")
                    .def(
                      py::init([](const std::string &dataset_dir, const std::string &name, const py::handle &sampler) {
                        std::shared_ptr<CMUArcticNode> cmu_arctic =
                          std::make_shared<CMUArcticNode>(dataset_dir, name, toSamplerObj(sampler), nullptr);
                        THROW_IF_ERROR(cmu_arctic->ValidateParams());
                        return cmu_arctic;
                      }));
                }));

PYBIND_REGISTER(
  CocoNode, 2, ([](const py::module *m) {
    (void)py::class_<CocoNode, DatasetNode, std::shared_ptr<CocoNode>>(*m, "CocoNode", "to create a CocoNode")
      .def(py::init([](const std::string &dataset_dir, const std::string &annotation_file, const std::string &task,
                       bool decode, const py::handle &sampler, bool extra_metadata, const py::object &decrypt_obj) {
        py::function decrypt =
          py::isinstance<py::function>(decrypt_obj) ? decrypt_obj.cast<py::function>() : py::function();
        std::shared_ptr<CocoNode> coco = std::make_shared<CocoNode>(
          dataset_dir, annotation_file, task, decode, toSamplerObj(sampler), nullptr, extra_metadata, decrypt);
        THROW_IF_ERROR(coco->ValidateParams());
        return coco;
      }));
  }));

PYBIND_REGISTER(CoNLL2000Node, 2, ([](const py::module *m) {
                  (void)py::class_<CoNLL2000Node, DatasetNode, std::shared_ptr<CoNLL2000Node>>(
                    *m, "CoNLL2000Node", "to create a CoNLL2000Node")
                    .def(py::init([](const std::string &dataset_dir, const std::string &usage, int64_t num_samples,
                                     int32_t shuffle, int32_t num_shards, int32_t shard_id) {
                      std::shared_ptr<CoNLL2000Node> conll2000 = std::make_shared<CoNLL2000Node>(
                        dataset_dir, usage, num_samples, toShuffleMode(shuffle), num_shards, shard_id, nullptr);
                      THROW_IF_ERROR(conll2000->ValidateParams());
                      return conll2000;
                    }));
                }));

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

PYBIND_REGISTER(DBpediaNode, 2, ([](const py::module *m) {
                  (void)py::class_<DBpediaNode, DatasetNode, std::shared_ptr<DBpediaNode>>(*m, "DBpediaNode",
                                                                                           "to create a DBpediaNode")
                    .def(py::init([](const std::string &dataset_dir, const std::string &usage, int64_t num_samples,
                                     int32_t shuffle, int32_t num_shards, int32_t shard_id) {
                      auto dbpedia = std::make_shared<DBpediaNode>(
                        dataset_dir, usage, num_samples, toShuffleMode(shuffle), num_shards, shard_id, nullptr);
                      THROW_IF_ERROR(dbpedia->ValidateParams());
                      return dbpedia;
                    }));
                }));

PYBIND_REGISTER(DIV2KNode, 2, ([](const py::module *m) {
                  (void)py::class_<DIV2KNode, DatasetNode, std::shared_ptr<DIV2KNode>>(*m, "DIV2KNode",
                                                                                       "to create a DIV2KNode")
                    .def(
                      py::init([](const std::string &dataset_dir, const std::string &usage,
                                  const std::string &downgrade, int32_t scale, bool decode, const py::handle &sampler) {
                        auto div2k = std::make_shared<DIV2KNode>(dataset_dir, usage, downgrade, scale, decode,
                                                                 toSamplerObj(sampler), nullptr);
                        THROW_IF_ERROR(div2k->ValidateParams());
                        return div2k;
                      }));
                }));

PYBIND_REGISTER(EMnistNode, 2, ([](const py::module *m) {
                  (void)py::class_<EMnistNode, DatasetNode, std::shared_ptr<EMnistNode>>(*m, "EMnistNode",
                                                                                         "to create an EMnistNode")
                    .def(py::init([](const std::string &dataset_dir, const std::string &name, const std::string &usage,
                                     const py::handle &sampler) {
                      auto emnist =
                        std::make_shared<EMnistNode>(dataset_dir, name, usage, toSamplerObj(sampler), nullptr);
                      THROW_IF_ERROR(emnist->ValidateParams());
                      return emnist;
                    }));
                }));

PYBIND_REGISTER(EnWik9Node, 2, ([](const py::module *m) {
                  (void)py::class_<EnWik9Node, DatasetNode, std::shared_ptr<EnWik9Node>>(*m, "EnWik9Node",
                                                                                         "to create an EnWik9Node")
                    .def(py::init([](const std::string &dataset_dir, int32_t num_samples, int32_t shuffle,
                                     int32_t num_shards, int32_t shard_id) {
                      std::shared_ptr<EnWik9Node> en_wik9 = std::make_shared<EnWik9Node>(
                        dataset_dir, num_samples, toShuffleMode(shuffle), num_shards, shard_id, nullptr);
                      THROW_IF_ERROR(en_wik9->ValidateParams());
                      return en_wik9;
                    }));
                }));

PYBIND_REGISTER(FakeImageNode, 2, ([](const py::module *m) {
                  (void)py::class_<FakeImageNode, DatasetNode, std::shared_ptr<FakeImageNode>>(
                    *m, "FakeImageNode", "to create a FakeImageNode")
                    .def(py::init([](int32_t num_images, const std::vector<int32_t> &image_size, int32_t num_classes,
                                     int32_t base_seed, const py::handle &sampler) {
                      auto fake_image = std::make_shared<FakeImageNode>(num_images, image_size, num_classes, base_seed,
                                                                        toSamplerObj(sampler), nullptr);
                      THROW_IF_ERROR(fake_image->ValidateParams());
                      return fake_image;
                    }));
                }));

PYBIND_REGISTER(FashionMnistNode, 2, ([](const py::module *m) {
                  (void)py::class_<FashionMnistNode, DatasetNode, std::shared_ptr<FashionMnistNode>>(
                    *m, "FashionMnistNode", "to create a FashionMnistNode")
                    .def(
                      py::init([](const std::string &dataset_dir, const std::string &usage, const py::handle &sampler) {
                        auto fashion_mnist =
                          std::make_shared<FashionMnistNode>(dataset_dir, usage, toSamplerObj(sampler), nullptr);
                        THROW_IF_ERROR(fashion_mnist->ValidateParams());
                        return fashion_mnist;
                      }));
                }));

PYBIND_REGISTER(FlickrNode, 2, ([](const py::module *m) {
                  (void)py::class_<FlickrNode, DatasetNode, std::shared_ptr<FlickrNode>>(*m, "FlickrNode",
                                                                                         "to create a FlickrNode")
                    .def(py::init([](const std::string &dataset_dir, const std::string &annotation_file, bool decode,
                                     const py::handle &sampler) {
                      auto flickr = std::make_shared<FlickrNode>(dataset_dir, annotation_file, decode,
                                                                 toSamplerObj(sampler), nullptr);
                      THROW_IF_ERROR(flickr->ValidateParams());
                      return flickr;
                    }));
                }));

PYBIND_REGISTER(Food101Node, 2, ([](const py::module *m) {
                  (void)py::class_<Food101Node, DatasetNode, std::shared_ptr<Food101Node>>(*m, "Food101Node",
                                                                                           "to create a Food101Node")
                    .def(py::init([](const std::string &dataset_dir, const std::string &usage, bool decode,
                                     const py::handle &sampler) {
                      auto food101 =
                        std::make_shared<Food101Node>(dataset_dir, usage, decode, toSamplerObj(sampler), nullptr);
                      THROW_IF_ERROR(food101->ValidateParams());
                      return food101;
                    }));
                }));

PYBIND_REGISTER(
  GeneratorNode, 2, ([](const py::module *m) {
    (void)py::class_<GeneratorNode, DatasetNode, std::shared_ptr<GeneratorNode>>(*m, "GeneratorNode",
                                                                                 "to create a GeneratorNode")
      .def(py::init([](const py::function &generator_function, const std::vector<std::string> &column_names,
                       const std::vector<DataType> &column_types, int64_t dataset_len, const py::handle &sampler,
                       uint32_t num_parallel_workers, const std::shared_ptr<PythonMultiprocessingRuntime> &python_mp) {
        auto gen = std::make_shared<GeneratorNode>(generator_function, column_names, column_types, dataset_len,
                                                   toSamplerObj(sampler), num_parallel_workers, python_mp);
        THROW_IF_ERROR(gen->ValidateParams());
        return gen;
      }))
      .def(py::init([](const py::function &generator_function, const std::shared_ptr<SchemaObj> &schema,
                       int64_t dataset_len, const py::handle &sampler, uint32_t num_parallel_workers,
                       const std::shared_ptr<PythonMultiprocessingRuntime> &python_mp) {
        auto gen = std::make_shared<GeneratorNode>(generator_function, schema, dataset_len, toSamplerObj(sampler),
                                                   num_parallel_workers, python_mp);
        THROW_IF_ERROR(gen->ValidateParams());
        return gen;
      }));
  }));

PYBIND_REGISTER(GTZANNode, 2, ([](const py::module *m) {
                  (void)py::class_<GTZANNode, DatasetNode, std::shared_ptr<GTZANNode>>(*m, "GTZANNode",
                                                                                       "to create a GTZANNode")
                    .def(
                      py::init([](const std::string &dataset_dir, const std::string &usage, const py::handle &sampler) {
                        auto gtzan = std::make_shared<GTZANNode>(dataset_dir, usage, toSamplerObj(sampler), nullptr);
                        THROW_IF_ERROR(gtzan->ValidateParams());
                        return gtzan;
                      }));
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

PYBIND_REGISTER(IMDBNode, 2, ([](const py::module *m) {
                  (void)py::class_<IMDBNode, DatasetNode, std::shared_ptr<IMDBNode>>(*m, "IMDBNode",
                                                                                     "to create an IMDBNode")
                    .def(
                      py::init([](const std::string &dataset_dir, const std::string &usage, const py::handle &sampler) {
                        auto imdb = std::make_shared<IMDBNode>(dataset_dir, usage, toSamplerObj(sampler), nullptr);
                        THROW_IF_ERROR(imdb->ValidateParams());
                        return imdb;
                      }));
                }));

PYBIND_REGISTER(IWSLT2016Node, 2, ([](const py::module *m) {
                  (void)py::class_<IWSLT2016Node, DatasetNode, std::shared_ptr<IWSLT2016Node>>(
                    *m, "IWSLT2016Node", "to create an IWSLT2016Node")
                    .def(py::init([](const std::string &dataset_dir, const std::string &usage,
                                     const std::vector<std::string> &language_pair, const std::string &valid_set,
                                     const std::string &test_set, int64_t num_samples, int32_t shuffle,
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
                    .def(py::init([](const std::string &dataset_dir, const std::string &usage,
                                     const std::vector<std::string> &language_pair, int64_t num_samples,
                                     int32_t shuffle, int32_t num_shards, int32_t shard_id) {
                      std::shared_ptr<IWSLT2017Node> iwslt2017 =
                        std::make_shared<IWSLT2017Node>(dataset_dir, usage, language_pair, num_samples,
                                                        toShuffleMode(shuffle), num_shards, shard_id, nullptr);
                      THROW_IF_ERROR(iwslt2017->ValidateParams());
                      return iwslt2017;
                    }));
                }));

PYBIND_REGISTER(KITTINode, 2, ([](const py::module *m) {
                  (void)py::class_<KITTINode, DatasetNode, std::shared_ptr<KITTINode>>(*m, "KITTINode",
                                                                                       "to create a KITTINode")
                    .def(py::init([](const std::string &dataset_dir, const std::string &usage, bool decode,
                                     const py::handle &sampler) {
                      std::shared_ptr<KITTINode> kitti =
                        std::make_shared<KITTINode>(dataset_dir, usage, decode, toSamplerObj(sampler), nullptr);
                      THROW_IF_ERROR(kitti->ValidateParams());
                      return kitti;
                    }));
                }));

PYBIND_REGISTER(KMnistNode, 2, ([](const py::module *m) {
                  (void)py::class_<KMnistNode, DatasetNode, std::shared_ptr<KMnistNode>>(*m, "KMnistNode",
                                                                                         "to create a KMnistNode")
                    .def(
                      py::init([](const std::string &dataset_dir, const std::string &usage, const py::handle &sampler) {
                        auto kmnist = std::make_shared<KMnistNode>(dataset_dir, usage, toSamplerObj(sampler), nullptr);
                        THROW_IF_ERROR(kmnist->ValidateParams());
                        return kmnist;
                      }));
                }));

PYBIND_REGISTER(LFWNode, 2, ([](const py::module *m) {
                  (void)py::class_<LFWNode, DatasetNode, std::shared_ptr<LFWNode>>(*m, "LFWNode", "to create a LFWNode")
                    .def(py::init([](const std::string &dataset_dir, const std::string &task, const std::string &usage,
                                     const std::string &image_set, bool decode, const py::handle &sampler) {
                      auto lfw = std::make_shared<LFWNode>(dataset_dir, task, usage, image_set, decode,
                                                           toSamplerObj(sampler), nullptr);
                      THROW_IF_ERROR(lfw->ValidateParams());
                      return lfw;
                    }));
                }));

PYBIND_REGISTER(LibriTTSNode, 2, ([](const py::module *m) {
                  (void)py::class_<LibriTTSNode, DatasetNode, std::shared_ptr<LibriTTSNode>>(*m, "LibriTTSNode",
                                                                                             "to create a LibriTTSNode")
                    .def(
                      py::init([](const std::string &dataset_dir, const std::string &usage, const py::handle &sampler) {
                        std::shared_ptr<LibriTTSNode> libri_tts =
                          std::make_shared<LibriTTSNode>(dataset_dir, usage, toSamplerObj(sampler), nullptr);
                        THROW_IF_ERROR(libri_tts->ValidateParams());
                        return libri_tts;
                      }));
                }));

PYBIND_REGISTER(LJSpeechNode, 2, ([](const py::module *m) {
                  (void)py::class_<LJSpeechNode, DatasetNode, std::shared_ptr<LJSpeechNode>>(*m, "LJSpeechNode",
                                                                                             "to create a LJSpeechNode")
                    .def(py::init([](const std::string &dataset_dir, const py::handle &sampler) {
                      auto lj_speech = std::make_shared<LJSpeechNode>(dataset_dir, toSamplerObj(sampler), nullptr);
                      THROW_IF_ERROR(lj_speech->ValidateParams());
                      return lj_speech;
                    }));
                }));

PYBIND_REGISTER(LSUNNode, 2, ([](const py::module *m) {
                  (void)py::class_<LSUNNode, DatasetNode, std::shared_ptr<LSUNNode>>(*m, "LSUNNode",
                                                                                     "to create a LSUNNode")
                    .def(py::init([](const std::string &dataset_dir, const std::string &usage,
                                     const std::vector<std::string> &classes, bool decode, const py::handle &sampler) {
                      auto lsun =
                        std::make_shared<LSUNNode>(dataset_dir, usage, classes, decode, toSamplerObj(sampler), nullptr);
                      THROW_IF_ERROR(lsun->ValidateParams());
                      return lsun;
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
      }));
  }));

PYBIND_REGISTER(MnistNode, 2, ([](const py::module *m) {
                  (void)py::class_<MnistNode, DatasetNode, std::shared_ptr<MnistNode>>(*m, "MnistNode",
                                                                                       "to create an MnistNode")
                    .def(
                      py::init([](const std::string &dataset_dir, const std::string &usage, const py::handle &sampler) {
                        auto mnist = std::make_shared<MnistNode>(dataset_dir, usage, toSamplerObj(sampler), nullptr);
                        THROW_IF_ERROR(mnist->ValidateParams());
                        return mnist;
                      }));
                }));

PYBIND_REGISTER(Multi30kNode, 2, ([](const py::module *m) {
                  (void)py::class_<Multi30kNode, DatasetNode, std::shared_ptr<Multi30kNode>>(*m, "Multi30kNode",
                                                                                             "to create a Multi30kNode")
                    .def(py::init([](const std::string &dataset_dir, const std::string &usage,
                                     const std::vector<std::string> &language_pair, int64_t num_samples,
                                     int32_t shuffle, int32_t num_shards, int32_t shard_id) {
                      std::shared_ptr<Multi30kNode> multi30k =
                        std::make_shared<Multi30kNode>(dataset_dir, usage, language_pair, num_samples,
                                                       toShuffleMode(shuffle), num_shards, shard_id, nullptr);
                      THROW_IF_ERROR(multi30k->ValidateParams());
                      return multi30k;
                    }));
                }));

PYBIND_REGISTER(OmniglotNode, 2, ([](const py::module *m) {
                  (void)py::class_<OmniglotNode, DatasetNode, std::shared_ptr<OmniglotNode>>(
                    *m, "OmniglotNode", "to create an OmniglotNode")
                    .def(py::init([](const std::string &dataset_dir, bool background, bool decode,
                                     const py::handle &sampler) {
                      auto omniglot =
                        std::make_shared<OmniglotNode>(dataset_dir, background, decode, toSamplerObj(sampler), nullptr);
                      THROW_IF_ERROR(omniglot->ValidateParams());
                      return omniglot;
                    }));
                }));

PYBIND_REGISTER(PennTreebankNode, 2, ([](const py::module *m) {
                  (void)py::class_<PennTreebankNode, DatasetNode, std::shared_ptr<PennTreebankNode>>(
                    *m, "PennTreebankNode", "to create a PennTreebankNode")
                    .def(py::init([](const std::string &dataset_dir, const std::string &usage, int32_t num_samples,
                                     int32_t shuffle, int32_t num_shards, int32_t shard_id) {
                      auto penn_treebank = std::make_shared<PennTreebankNode>(
                        dataset_dir, usage, num_samples, toShuffleMode(shuffle), num_shards, shard_id, nullptr);
                      THROW_IF_ERROR(penn_treebank->ValidateParams());
                      return penn_treebank;
                    }));
                }));

PYBIND_REGISTER(PhotoTourNode, 2, ([](const py::module *m) {
                  (void)py::class_<PhotoTourNode, DatasetNode, std::shared_ptr<PhotoTourNode>>(
                    *m, "PhotoTourNode", "to create a PhotoTourNode")
                    .def(py::init([](const std::string &dataset_dir, const std::string &name, const std::string &usage,
                                     const py::handle &sampler) {
                      auto photo_tour =
                        std::make_shared<PhotoTourNode>(dataset_dir, name, usage, toSamplerObj(sampler), nullptr);
                      THROW_IF_ERROR(photo_tour->ValidateParams());
                      return photo_tour;
                    }));
                }));

PYBIND_REGISTER(Places365Node, 2, ([](const py::module *m) {
                  (void)py::class_<Places365Node, DatasetNode, std::shared_ptr<Places365Node>>(
                    *m, "Places365Node", "to create a Places365Node")
                    .def(py::init([](const std::string &dataset_dir, const std::string &usage, bool small, bool decode,
                                     const py::handle &sampler) {
                      auto places365 = std::make_shared<Places365Node>(dataset_dir, usage, small, decode,
                                                                       toSamplerObj(sampler), nullptr);
                      THROW_IF_ERROR(places365->ValidateParams());
                      return places365;
                    }));
                }));

PYBIND_REGISTER(QMnistNode, 2, ([](const py::module *m) {
                  (void)py::class_<QMnistNode, DatasetNode, std::shared_ptr<QMnistNode>>(*m, "QMnistNode",
                                                                                         "to create a QMnistNode")
                    .def(py::init([](const std::string &dataset_dir, const std::string &usage, bool compat,
                                     const py::handle &sampler) {
                      auto qmnist =
                        std::make_shared<QMnistNode>(dataset_dir, usage, compat, toSamplerObj(sampler), nullptr);
                      THROW_IF_ERROR(qmnist->ValidateParams());
                      return qmnist;
                    }));
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

PYBIND_REGISTER(RenderedSST2Node, 2, ([](const py::module *m) {
                  (void)py::class_<RenderedSST2Node, DatasetNode, std::shared_ptr<RenderedSST2Node>>(
                    *m, "RenderedSST2Node", "to create a RenderedSST2Node")
                    .def(py::init([](const std::string &dataset_dir, const std::string &usage, bool decode,
                                     const py::handle &sampler) {
                      auto rendered_sst2 =
                        std::make_shared<RenderedSST2Node>(dataset_dir, usage, decode, toSamplerObj(sampler), nullptr);
                      THROW_IF_ERROR(rendered_sst2->ValidateParams());
                      return rendered_sst2;
                    }));
                }));

PYBIND_REGISTER(SBUNode, 2, ([](const py::module *m) {
                  (void)py::class_<SBUNode, DatasetNode, std::shared_ptr<SBUNode>>(*m, "SBUNode",
                                                                                   "to create an SBUNode")
                    .def(py::init([](const std::string &dataset_dir, bool decode, const py::handle &sampler) {
                      auto sbu = std::make_shared<SBUNode>(dataset_dir, decode, toSamplerObj(sampler), nullptr);
                      THROW_IF_ERROR(sbu->ValidateParams());
                      return sbu;
                    }));
                }));

PYBIND_REGISTER(SemeionNode, 2, ([](const py::module *m) {
                  (void)py::class_<SemeionNode, DatasetNode, std::shared_ptr<SemeionNode>>(*m, "SemeionNode",
                                                                                           "to create a SemeionNode")
                    .def(py::init([](const std::string &dataset_dir, const py::handle &sampler) {
                      auto semeion = std::make_shared<SemeionNode>(dataset_dir, toSamplerObj(sampler), nullptr);
                      THROW_IF_ERROR(semeion->ValidateParams());
                      return semeion;
                    }));
                }));

PYBIND_REGISTER(SogouNewsNode, 2, ([](const py::module *m) {
                  (void)py::class_<SogouNewsNode, DatasetNode, std::shared_ptr<SogouNewsNode>>(
                    *m, "SogouNewsNode", "to create a SogouNewsNode")
                    .def(py::init([](const std::string &dataset_dir, const std::string &usage, int64_t num_samples,
                                     int32_t shuffle, int32_t num_shards, int32_t shard_id) {
                      auto sogou_news = std::make_shared<SogouNewsNode>(
                        dataset_dir, usage, num_samples, toShuffleMode(shuffle), num_shards, shard_id, nullptr);
                      THROW_IF_ERROR(sogou_news->ValidateParams());
                      return sogou_news;
                    }));
                }));

PYBIND_REGISTER(SpeechCommandsNode, 2, ([](const py::module *m) {
                  (void)py::class_<SpeechCommandsNode, DatasetNode, std::shared_ptr<SpeechCommandsNode>>(
                    *m, "SpeechCommandsNode", "to create a SpeechCommandsNode")
                    .def(
                      py::init([](const std::string &dataset_dir, const std::string &usage, const py::handle &sampler) {
                        auto speech_commands =
                          std::make_shared<SpeechCommandsNode>(dataset_dir, usage, toSamplerObj(sampler), nullptr);
                        THROW_IF_ERROR(speech_commands->ValidateParams());
                        return speech_commands;
                      }));
                }));

PYBIND_REGISTER(SQuADNode, 2, ([](const py::module *m) {
                  (void)py::class_<SQuADNode, DatasetNode, std::shared_ptr<SQuADNode>>(*m, "SQuADNode",
                                                                                       "to create a SQuADNode")
                    .def(py::init([](const std::string &dataset_dir, const std::string &usage, int64_t num_samples,
                                     int32_t shuffle, int32_t num_shards, int32_t shard_id) {
                      std::shared_ptr<SQuADNode> squad = std::make_shared<dataset::SQuADNode>(
                        dataset_dir, usage, num_samples, toShuffleMode(shuffle), num_shards, shard_id, nullptr);
                      THROW_IF_ERROR(squad->ValidateParams());
                      return squad;
                    }));
                }));

PYBIND_REGISTER(SST2Node, 2, ([](const py::module *m) {
                  (void)py::class_<SST2Node, DatasetNode, std::shared_ptr<SST2Node>>(*m, "SST2Node",
                                                                                     "to create a SST2Node")
                    .def(py::init([](const std::string &dataset_dir, const std::string &usage, int64_t num_samples,
                                     int32_t shuffle, int32_t num_shards, int32_t shard_id) {
                      auto sst2 = std::make_shared<SST2Node>(dataset_dir, usage, num_samples, toShuffleMode(shuffle),
                                                             num_shards, shard_id, nullptr);
                      THROW_IF_ERROR(sst2->ValidateParams());
                      return sst2;
                    }));
                }));

PYBIND_REGISTER(STL10Node, 2, ([](const py::module *m) {
                  (void)py::class_<STL10Node, DatasetNode, std::shared_ptr<STL10Node>>(*m, "STL10Node",
                                                                                       "to create a STL10Node")
                    .def(
                      py::init([](const std::string &dataset_dir, const std::string &usage, const py::handle &sampler) {
                        auto stl10 = std::make_shared<STL10Node>(dataset_dir, usage, toSamplerObj(sampler), nullptr);
                        THROW_IF_ERROR(stl10->ValidateParams());
                        return stl10;
                      }));
                }));

PYBIND_REGISTER(SUN397Node, 2, ([](const py::module *m) {
                  (void)py::class_<SUN397Node, DatasetNode, std::shared_ptr<SUN397Node>>(*m, "SUN397Node",
                                                                                         "to create a SUN397Node")
                    .def(py::init([](const std::string &dataset_dir, bool decode, const py::handle &sampler) {
                      auto sun397 = std::make_shared<SUN397Node>(dataset_dir, decode, toSamplerObj(sampler), nullptr);
                      THROW_IF_ERROR(sun397->ValidateParams());
                      return sun397;
                    }));
                }));

PYBIND_REGISTER(TedliumNode, 2, ([](const py::module *m) {
                  (void)py::class_<TedliumNode, DatasetNode, std::shared_ptr<TedliumNode>>(*m, "TedliumNode",
                                                                                           "to create a TedliumNode")
                    .def(
                      py::init([](const std::string &dataset_dir, const std::string &release, const std::string &usage,
                                  const std::string &extensions, const py::handle &sampler) {
                        auto tedlium = std::make_shared<TedliumNode>(dataset_dir, release, usage, extensions,
                                                                     toSamplerObj(sampler), nullptr);
                        THROW_IF_ERROR(tedlium->ValidateParams());
                        return tedlium;
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

PYBIND_REGISTER(UDPOSNode, 2, ([](const py::module *m) {
                  (void)py::class_<UDPOSNode, DatasetNode, std::shared_ptr<UDPOSNode>>(*m, "UDPOSNode",
                                                                                       "to create an UDPOSNode")
                    .def(py::init([](const std::string &dataset_dir, const std::string &usage, int64_t num_samples,
                                     int32_t shuffle, int32_t num_shards, int32_t shard_id) {
                      std::shared_ptr<UDPOSNode> udpos = std::make_shared<UDPOSNode>(
                        dataset_dir, usage, num_samples, toShuffleMode(shuffle), num_shards, shard_id, nullptr);
                      THROW_IF_ERROR(udpos->ValidateParams());
                      return udpos;
                    }));
                }));

PYBIND_REGISTER(USPSNode, 2, ([](const py::module *m) {
                  (void)py::class_<USPSNode, DatasetNode, std::shared_ptr<USPSNode>>(*m, "USPSNode",
                                                                                     "to create an USPSNode")
                    .def(py::init([](const std::string &dataset_dir, const std::string &usage, int32_t num_samples,
                                     int32_t shuffle, int32_t num_shards, int32_t shard_id) {
                      auto usps = std::make_shared<USPSNode>(dataset_dir, usage, num_samples, toShuffleMode(shuffle),
                                                             num_shards, shard_id, nullptr);
                      THROW_IF_ERROR(usps->ValidateParams());
                      return usps;
                    }));
                }));

PYBIND_REGISTER(VOCNode, 2, ([](const py::module *m) {
                  (void)py::class_<VOCNode, DatasetNode, std::shared_ptr<VOCNode>>(*m, "VOCNode", "to create a VOCNode")
                    .def(py::init([](const std::string &dataset_dir, const std::string &task, const std::string &usage,
                                     const py::dict &class_indexing, bool decode, const py::handle &sampler,
                                     bool extra_metadata, const py::object &decrypt_obj) {
                      py::function decrypt =
                        py::isinstance<py::function>(decrypt_obj) ? decrypt_obj.cast<py::function>() : py::function();
                      std::shared_ptr<VOCNode> voc =
                        std::make_shared<VOCNode>(dataset_dir, task, usage, toStringMap(class_indexing), decode,
                                                  toSamplerObj(sampler), nullptr, extra_metadata, decrypt);
                      THROW_IF_ERROR(voc->ValidateParams());
                      return voc;
                    }));
                }));

PYBIND_REGISTER(WIDERFaceNode, 2, ([](const py::module *m) {
                  (void)py::class_<WIDERFaceNode, DatasetNode, std::shared_ptr<WIDERFaceNode>>(
                    *m, "WIDERFaceNode", "to create a WIDERFaceNode")
                    .def(py::init([](const std::string &dataset_dir, const std::string &usage, bool decode,
                                     const py::handle &sampler) {
                      auto wider_face =
                        std::make_shared<WIDERFaceNode>(dataset_dir, usage, decode, toSamplerObj(sampler), nullptr);
                      THROW_IF_ERROR(wider_face->ValidateParams());
                      return wider_face;
                    }));
                }));

PYBIND_REGISTER(WikiTextNode, 2, ([](const py::module *m) {
                  (void)py::class_<WikiTextNode, DatasetNode, std::shared_ptr<WikiTextNode>>(*m, "WikiTextNode",
                                                                                             "to create a WikiTextNode")
                    .def(py::init([](const std::string &dataset_dir, const std::string &usage, int32_t num_samples,
                                     int32_t shuffle, int32_t num_shards, int32_t shard_id) {
                      auto wiki_text = std::make_shared<WikiTextNode>(
                        dataset_dir, usage, num_samples, toShuffleMode(shuffle), num_shards, shard_id, nullptr);
                      THROW_IF_ERROR(wiki_text->ValidateParams());
                      return wiki_text;
                    }));
                }));

PYBIND_REGISTER(YahooAnswersNode, 2, ([](const py::module *m) {
                  (void)py::class_<YahooAnswersNode, DatasetNode, std::shared_ptr<YahooAnswersNode>>(
                    *m, "YahooAnswersNode", "to create a YahooAnswersNode")
                    .def(py::init([](const std::string &dataset_dir, const std::string &usage, int64_t num_samples,
                                     int32_t shuffle, int32_t num_shards, int32_t shard_id) {
                      auto yahoo_answers = std::make_shared<YahooAnswersNode>(
                        dataset_dir, usage, num_samples, toShuffleMode(shuffle), num_shards, shard_id, nullptr);
                      THROW_IF_ERROR(yahoo_answers->ValidateParams());
                      return yahoo_answers;
                    }));
                }));

PYBIND_REGISTER(YelpReviewNode, 2, ([](const py::module *m) {
                  (void)py::class_<YelpReviewNode, DatasetNode, std::shared_ptr<YelpReviewNode>>(
                    *m, "YelpReviewNode", "to create a YelpReviewNode")
                    .def(py::init([](const std::string &dataset_dir, const std::string &usage, int64_t num_samples,
                                     int32_t shuffle, int32_t num_shards, int32_t shard_id) {
                      std::shared_ptr<YelpReviewNode> yelp_review = std::make_shared<YelpReviewNode>(
                        dataset_dir, usage, num_samples, toShuffleMode(shuffle), num_shards, shard_id, nullptr);
                      THROW_IF_ERROR(yelp_review->ValidateParams());
                      return yelp_review;
                    }));
                }));

PYBIND_REGISTER(YesNoNode, 2, ([](const py::module *m) {
                  (void)py::class_<YesNoNode, DatasetNode, std::shared_ptr<YesNoNode>>(*m, "YesNoNode",
                                                                                       "to create a YesNoNode")
                    .def(py::init([](const std::string &dataset_dir, const py::handle &sampler) {
                      auto yes_no = std::make_shared<YesNoNode>(dataset_dir, toSamplerObj(sampler), nullptr);
                      THROW_IF_ERROR(yes_no->ValidateParams());
                      return yes_no;
                    }));
                }));
}  // namespace dataset
}  // namespace mindspore
