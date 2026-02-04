# Copyright 2025 Huawei Technologies Co., Ltd
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
# ==============================================================================
"""Fetcher module."""

from .collate import default_collate, default_convert


class Fetcher:
    """
    Fetcher to fetch data from dataset.
    """

    def __init__(self, dataset, auto_collation, collate_fn):
        self.dataset = dataset
        self.auto_collation = auto_collation
        self.collate_fn = collate_fn
        if collate_fn is None:
            if self.auto_collation:
                self.collate_fn = default_collate
            else:
                self.collate_fn = default_convert

    def fetch(self, indices):
        """
        Fetch data from dataset.
        """

        raise NotImplementedError(f"{self.__class__.__name__} should implement fetch method.")

    def reset(self):
        """
        Reset the dataset.
        """


class MapDatasetFetcher(Fetcher):
    """
    Fetcher for MapDataset.
    """

    def fetch(self, indices):
        """
        Fetch data from dataset.
        """
        if self.auto_collation:
            data = [self.dataset[index] for index in indices]
        else:
            data = self.dataset[indices]
        return self.collate_fn(data)


class IterableDatasetFetcher(Fetcher):
    """
    Fetcher for IterableDataset.
    """

    def __init__(self, dataset, auto_collation, collate_fn, drop_last):
        """
        Initialize the fetcher.
        """

        super().__init__(dataset, auto_collation, collate_fn)
        self.drop_last = drop_last
        self.reset()

    def fetch(self, indices):
        """
        Fetch data from dataset.
        """

        if self.ended:
            raise StopIteration

        if self.auto_collation:
            data = []
            for _ in indices:
                try:
                    data.append(next(self.dataset_iterator))
                except StopIteration:
                    if data:
                        self.ended = True
                        break
            # once get none from dataset_iterator, iter stops
            # or received data size less than indices size, iter stops
            if not data or self.drop_last and len(data) < len(indices):
                raise StopIteration
        else:
            data = next(self.dataset_iterator)
        return self.collate_fn(data)

    def reset(self):
        """
        Reset the fetcher.
        """

        self.dataset_iterator = iter(self.dataset)
        self.ended = False
