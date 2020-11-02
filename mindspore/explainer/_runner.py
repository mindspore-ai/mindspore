# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Runner."""
from time import time
from typing import Tuple, List, Optional

import numpy as np

from mindspore.train._utils import check_value_type
from mindspore.train.summary_pb2 import Explain
import mindspore as ms
import mindspore.dataset as ds
from mindspore import log
from mindspore.ops.operations import ExpandDims
from mindspore.train.summary._summary_adapter import _convert_image_format, _make_image
from mindspore.train.summary.summary_record import SummaryRecord
from .benchmark import Localization
from .benchmark._attribution.metric import AttributionMetric
from .explanation._attribution._attribution import Attribution

_EXPAND_DIMS = ExpandDims()
_CMAP_0 = np.reshape(np.array([55, 25, 86, 255]), [1, 1, 4]) / 255
_CMAP_1 = np.reshape(np.array([255, 255, 0, 255]), [1, 1, 4]) / 255


def _normalize(img_np):
    """Normalize the image in the numpy array to be in [0, 255]. """
    max_ = img_np.max()
    min_ = img_np.min()
    normed = (img_np - min_) / (max_ - min_).clip(min=1e-10)
    return (normed * 255).astype(np.uint8)


def _make_rgba(saliency):
    """Make rgba image for saliency map."""
    saliency = saliency.asnumpy().squeeze()
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min()).clip(1e-10)
    rgba = np.empty((saliency.shape[0], saliency.shape[1], 4))
    rgba[:, :, :] = np.expand_dims(saliency, 2)
    rgba = rgba * _CMAP_1 + (1 - rgba) * _CMAP_0
    rgba[:, :, -1] = saliency * 1
    return rgba


class ExplainRunner:
    """
    A high-level API for users to generate and store results of the explanation methods and the evaluation methods.

    After generating results with the explanation methods and the evaluation methods, the results will be written into
    a specified file with `mindspore.summary.SummaryRecord`. The stored content can be viewed using MindInsight.

    Args:
        summary_dir (str, optional): The directory path to save the summary files which store the generated results.
            Default: "./"

    Examples:
        >>> from mindspore.explainer import ExplainRunner
        >>> # init a runner with a specified directory
        >>> summary_dir = "summary_dir"
        >>> runner = ExplainRunner(summary_dir)
    """

    def __init__(self, summary_dir: Optional[str] = "./"):
        check_value_type("summary_dir", summary_dir, str)
        self._summary_dir = summary_dir
        self._count = 0
        self._classes = None
        self._model = None

    def run(self,
            dataset: Tuple,
            explainers: List,
            benchmarkers: Optional[List] = None):
        """
        Genereates results and writes results into the summary files in `summary_dir` specified during the object
        initialization.

        Args:
            dataset (tuple): A tuple that contains `mindspore.dataset` object for iteration and its labels.

                - dataset[0]: A `mindspore.dataset` object to provide data to explain.
                - dataset[1]: A list of string that specifies the label names of the dataset.

            explainers (list[Explanation]): A list of explanation objects to generate attribution results. Explanation
                object is an instance initialized with the explanation methods in module
                `mindspore.explainer.explanation`.
            benchmarkers (list[Benchmark], optional): A list of benchmark objects to generate evaluation results.
                Default: None

        Examples:
            >>> from mindspore.explainer.explanation import GuidedBackprop, Gradient
            >>> # obtain dataset object
            >>> dataset = get_dataset()
            >>> classes = ["cat", "dog", ...]
            >>> # load checkpoint to a network, e.g. resnet50
            >>> param_dict = load_checkpoint("checkpoint.ckpt")
            >>> net = resnet50(len(classes))
            >>> load_parama_into_net(net, param_dict)
            >>> # bind net with its output activation
            >>> model = nn.SequentialCell([net, nn.Sigmoid()])
            >>> gbp = GuidedBackprop(model)
            >>> gradient = Gradient(model)
            >>> runner = ExplainRunner("./")
            >>> explainers = [gbp, gradient]
            >>> runner.run((dataset, classes), explainers)
        """

        check_value_type("dataset", dataset, tuple)
        if len(dataset) != 2:
            raise ValueError("Argument `dataset` should be a tuple with length = 2.")

        dataset, classes = dataset
        self._verify_data_form(dataset, benchmarkers)
        self._classes = classes

        check_value_type("explainers", explainers, list)
        if not explainers:
            raise ValueError("Argument `explainers` must be a non-empty list")

        for exp in explainers:
            if not isinstance(exp, Attribution):
                raise TypeError("Argument explainers should be a list of objects of classes in "
                                "`mindspore.explainer.explanation`.")
        if benchmarkers is not None:
            check_value_type("benchmarkers", benchmarkers, list)
            for bench in benchmarkers:
                if not isinstance(bench, AttributionMetric):
                    raise TypeError("Argument benchmarkers should be a list of objects of classes in explanation"
                                    "`mindspore.explainer.benchmark`.")

        self._model = explainers[0].model
        next_element = dataset.create_tuple_iterator().get_next()
        inputs, _, _ = self._unpack_next_element(next_element)
        prop_test = self._model(inputs)
        check_value_type("output of model im explainer", prop_test, ms.Tensor)
        if prop_test.shape[1] > len(self._classes):
            raise ValueError("The dimension of model output should not exceed the length of dataset classes. Please "
                             "check dataset classes or the black-box model in the explainer again.")

        with SummaryRecord(self._summary_dir) as summary:
            print("Start running and writing......")
            begin = time()
            print("Start writing metadata.")

            explain = Explain()
            explain.metadata.label.extend(classes)
            exp_names = [exp.__class__.__name__ for exp in explainers]
            explain.metadata.explain_method.extend(exp_names)
            if benchmarkers is not None:
                bench_names = [bench.__class__.__name__ for bench in benchmarkers]
                explain.metadata.benchmark_method.extend(bench_names)

            summary.add_value("explainer", "metadata", explain)
            summary.record(1)

            print("Finish writing metadata.")

            now = time()
            print("Start running and writing inference data......")
            imageid_labels = self._run_inference(dataset, summary)
            print("Finish running and writing inference data. Time elapsed: {}s".format(time() - now))

            if benchmarkers is None or not benchmarkers:
                for exp in explainers:
                    start = time()
                    print("Start running and writing explanation data for {}......".format(exp.__class__.__name__))
                    self._count = 0
                    ds.config.set_seed(58)
                    for idx, next_element in enumerate(dataset):
                        now = time()
                        self._run_exp_step(next_element, exp, imageid_labels, summary)
                        print("Finish writing {}-th explanation data. Time elapsed: {}".format(
                            idx, time() - now))
                    print("Finish running and writing explanation data for {}. Time elapsed: {}".format(
                        exp.__class__.__name__, time() - start))
            else:
                for exp in explainers:
                    explain = Explain()
                    for bench in benchmarkers:
                        bench.reset()
                    print(f"Start running and writing explanation and benchmark data for {exp.__class__.__name__}.")
                    self._count = 0
                    start = time()
                    ds.config.set_seed(58)
                    for idx, next_element in enumerate(dataset):
                        now = time()
                        saliency_dict_lst = self._run_exp_step(next_element, exp, imageid_labels, summary)
                        print("Finish writing {}-th batch explanation data. Time elapsed: {}s".format(
                            idx, time() - now))
                        for bench in benchmarkers:
                            now = time()
                            self._run_exp_benchmark_step(next_element, exp, bench, saliency_dict_lst)
                            print("Finish running {}-th batch benchmark data for {}. Time elapsed: {}s".format(
                                idx, bench.__class__.__name__, time() - now))

                    for bench in benchmarkers:
                        benchmark = explain.benchmark.add()
                        benchmark.explain_method = exp.__class__.__name__
                        benchmark.benchmark_method = bench.__class__.__name__

                        benchmark.total_score = bench.performance
                        benchmark.label_score.extend(bench.class_performances)

                    print("Finish running and writing explanation and benchmark data for {}. "
                          "Time elapsed: {}s".format(exp.__class__.__name__, time() - start))
                    summary.add_value('explainer', 'benchmark', explain)
                    summary.record(1)
            print("Finish running and writing. Total time elapsed: {}s".format(time() - begin))

    @staticmethod
    def _verify_data_form(dataset, benchmarkers):
        """
        Verify the validity of dataset.

        Args:
            dataset (`ds`): the user parsed dataset.
            benchmarkers (list[`AttributionMetric`]): the user parsed benchmarkers.
        """
        next_element = dataset.create_tuple_iterator().get_next()

        if len(next_element) not in [1, 2, 3]:
            raise ValueError("The dataset should provide [images] or [images, labels], [images, labels, bboxes]"
                             " as columns.")

        if len(next_element) == 3:
            inputs, labels, bboxes = next_element
            if bboxes.shape[-1] != 4:
                raise ValueError("The third element of dataset should be bounding boxes with shape of "
                                 "[batch_size, num_ground_truth, 4].")
        else:
            if True in [isinstance(bench, Localization) for bench in benchmarkers]:
                raise ValueError("The dataset must provide bboxes if Localization is to be computed.")

            if len(next_element) == 2:
                inputs, labels = next_element
            if len(next_element) == 1:
                inputs = next_element[0]

        if len(inputs.shape) > 4 or len(inputs.shape) < 3 or inputs.shape[-3] not in [1, 3, 4]:
            raise ValueError(
                "Image shape {} is unrecognizable: the dimension of image can only be CHW or NCHW.".format(
                    inputs.shape))
        if len(inputs.shape) == 3:
            log.warning(
                "Image shape {} is 3-dimensional. All the data will be automatically unsqueezed at the 0-th"
                " dimension as batch data.".format(inputs.shape))

        if len(next_element) > 1:
            if len(labels.shape) > 2 and (np.array(labels.shape[1:]) > 1).sum() > 1:
                raise ValueError(
                    "Labels shape {} is unrecognizable: labels should not have more than two dimensions"
                    " with length greater than 1.".format(labels.shape))

    def _transform_data(self, inputs, labels, bboxes, ifbbox):
        """
        Transform the data from one iteration of dataset to a unifying form for the follow-up operations.

        Args:
            inputs (Tensor): the image data
            labels (Tensor): the labels
            bboxes (Tensor): the boudnding boxes data
            ifbbox (bool): whether to preprocess bboxes. If True, a dictionary that indicates bounding boxes w.r.t label
             id will be returned. If False, the returned bboxes is the the parsed bboxes.

        Returns:
            inputs (Tensor): the image data, unified to a 4D Tensor.
            labels (List[List[int]]): the ground truth labels.
            bboxes (Union[List[Dict], None, Tensor]): the bounding boxes
        """
        inputs = ms.Tensor(inputs, ms.float32)
        if len(inputs.shape) == 3:
            inputs = _EXPAND_DIMS(inputs, 0)
            if isinstance(labels, ms.Tensor):
                labels = ms.Tensor(labels, ms.int32)
                labels = _EXPAND_DIMS(labels, 0)
            if isinstance(bboxes, ms.Tensor):
                bboxes = ms.Tensor(bboxes, ms.int32)
                bboxes = _EXPAND_DIMS(bboxes, 0)

        input_len = len(inputs)
        if bboxes is not None and ifbbox:
            bboxes = ms.Tensor(bboxes, ms.int32)
            masks_lst = []
            labels = labels.asnumpy().reshape([input_len, -1])
            bboxes = bboxes.asnumpy().reshape([input_len, -1, 4])
            for idx, label in enumerate(labels):
                height, width = inputs[idx].shape[-2], inputs[idx].shape[-1]
                masks = {}
                for j, label_item in enumerate(label):
                    target = int(label_item)
                    if -1 < target < len(self._classes):
                        if target not in masks:
                            mask = np.zeros((1, 1, height, width))
                        else:
                            mask = masks[target]
                        x_min, y_min, x_len, y_len = bboxes[idx][j].astype(int)
                        mask[:, :, x_min:x_min + x_len, y_min:y_min + y_len] = 1
                        masks[target] = mask

                masks_lst.append(masks)
            bboxes = masks_lst

        labels = ms.Tensor(labels, ms.int32)
        if len(labels.shape) == 1:
            labels_lst = [[int(i)] for i in labels.asnumpy()]
        else:
            labels = labels.asnumpy().reshape([input_len, -1])
            labels_lst = []
            for item in labels:
                labels_lst.append(list(set(int(i) for i in item if -1 < int(i) < len(self._classes))))
        labels = labels_lst
        return inputs, labels, bboxes

    def _unpack_next_element(self, next_element, ifbbox=False):
        """
        Unpack a single iteration of dataset.

        Args:
            next_element (Tuple): a single element iterated from dataset object.
            ifbbox (bool): whether to preprocess bboxes in self._transform_data.

        Returns:
            Tuple, a unified Tuple contains image_data, labels, and bounding boxes.
        """
        if len(next_element) == 3:
            inputs, labels, bboxes = next_element
        elif len(next_element) == 2:
            inputs, labels = next_element
            bboxes = None
        else:
            inputs = next_element[0]
            labels = [[] for x in inputs]
            bboxes = None
        inputs, labels, bboxes = self._transform_data(inputs, labels, bboxes, ifbbox)
        return inputs, labels, bboxes

    @staticmethod
    def _make_label_batch(labels):
        """
        Unify a List of List of labels to be a 2D Tensor with shape (b, m), where b = len(labels) and m is the max
        length of all the rows in labels.

        Args:
            labels (List[List]): the union labels of a data batch.

        Returns:
            2D Tensor.
        """

        max_len = max([len(l) for l in labels])
        batch_labels = np.zeros((len(labels), max_len))

        for idx, _ in enumerate(batch_labels):
            length = len(labels[idx])
            batch_labels[idx, :length] = np.array(labels[idx])

        return ms.Tensor(batch_labels, ms.int32)

    def _run_inference(self, dataset, summary, threshod=0.5):
        """
        Run inference for the dataset and write the inference related data into summary.

        Args:
            dataset (`ds`): the parsed dataset
            summary (`SummaryRecord`): the summary object to store the data
            threshold (float): the threshold for prediction.

        Returns:
            imageid_labels (dict): a dict that maps image_id and the union of its ground truth and predicted labels.
        """
        imageid_labels = {}
        ds.config.set_seed(58)
        self._count = 0
        for j, next_element in enumerate(dataset):
            now = time()
            inputs, labels, _ = self._unpack_next_element(next_element)
            prob = self._model(inputs).asnumpy()
            for idx, inp in enumerate(inputs):
                gt_labels = labels[idx]
                gt_probs = [float(prob[idx][i]) for i in gt_labels]

                data_np = _convert_image_format(np.expand_dims(inp.asnumpy(), 0), 'NCHW')
                _, _, _, image_string = _make_image(_normalize(data_np))

                predicted_labels = [int(i) for i in (prob[idx] > threshod).nonzero()[0]]
                predicted_probs = [float(prob[idx][i]) for i in predicted_labels]

                union_labs = list(set(gt_labels + predicted_labels))
                imageid_labels[str(self._count)] = union_labs

                explain = Explain()
                explain.image_id = str(self._count)
                explain.image_data = image_string
                summary.add_value("explainer", "image", explain)

                explain = Explain()
                explain.image_id = str(self._count)
                explain.ground_truth_label.extend(gt_labels)
                explain.inference.ground_truth_prob.extend(gt_probs)
                explain.inference.predicted_label.extend(predicted_labels)
                explain.inference.predicted_prob.extend(predicted_probs)
                summary.add_value("explainer", "inference", explain)

                summary.record(1)

                self._count += 1
            print("Finish running and writing {}-th batch inference data. Time elapsed: {}s".format(j, time() - now))
        return imageid_labels

    def _run_exp_step(self, next_element, explainer, imageid_labels, summary):
        """
        Run the explanation for each step and write explanation results into summary.

        Args:
            next_element (Tuple): data of one step
            explainer (_Attribution): an Attribution object to generate saliency maps.
            imageid_labels (dict): a dict that maps the image_id and its union labels.
            summary (SummaryRecord): the summary object to store the data

        Returns:
            List of dict that maps label to its corresponding saliency map.
        """
        inputs, labels, _ = self._unpack_next_element(next_element)
        count = self._count
        unions = []
        for _ in range(len(labels)):
            unions_labels = imageid_labels[str(count)]
            unions.append(unions_labels)
            count += 1

        batch_unions = self._make_label_batch(unions)
        saliency_dict_lst = []

        batch_saliency_full = []
        for i in range(len(batch_unions[0])):
            batch_saliency = explainer(inputs, batch_unions[:, i])
            batch_saliency_full.append(batch_saliency)

        for idx, union in enumerate(unions):
            saliency_dict = {}
            explain = Explain()
            explain.image_id = str(self._count)
            for k, lab in enumerate(union):
                saliency = batch_saliency_full[k][idx:idx + 1]

                saliency_dict[lab] = saliency

                saliency_np = _make_rgba(saliency)
                _, _, _, saliency_string = _make_image(_normalize(saliency_np))

                explanation = explain.explanation.add()
                explanation.explain_method = explainer.__class__.__name__

                explanation.label = lab
                explanation.heatmap = saliency_string

            summary.add_value("explainer", "explanation", explain)
            summary.record(1)

            self._count += 1
            saliency_dict_lst.append(saliency_dict)
        return saliency_dict_lst

    def _run_exp_benchmark_step(self, next_element, explainer, benchmarker, saliency_dict_lst):
        """
        Run the explanation and evaluation for each step and write explanation results into summary.

        Args:
            next_element (Tuple): Data of one step
            explainer (`_Attribution`): An Attribution object to generate saliency maps.
            imageid_labels (dict): A dict that maps the image_id and its union labels.
        """
        inputs, labels, _ = self._unpack_next_element(next_element)
        for idx, inp in enumerate(inputs):
            inp = _EXPAND_DIMS(inp, 0)
            saliency_dict = saliency_dict_lst[idx]
            for label, saliency in saliency_dict.items():
                if isinstance(benchmarker, Localization):
                    _, _, bboxes = self._unpack_next_element(next_element, True)
                    if label in labels[idx]:
                        res = benchmarker.evaluate(explainer, inp, targets=label, mask=bboxes[idx][label],
                                                   saliency=saliency)
                        benchmarker.aggregate(res, label)
                else:
                    res = benchmarker.evaluate(explainer, inp, targets=label, saliency=saliency)
                    benchmarker.aggregate(res, label)
