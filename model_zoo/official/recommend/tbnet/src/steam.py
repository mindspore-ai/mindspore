# Copyright 2021 Huawei Technologies Co., Ltd
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
"""STEAM dataset explainer."""

import json

from src.aggregator import Recommendation


class TextExplainer:
    """Text explainer for STEAM game recommendations."""
    SAME_RELATION_TPL = 'User played the game <%s> before, which has the same %s '\
                        '<%s> as the recommend game.'
    DIFF_RELATION_TPL = 'User played the game <%s> before, which has the %s <%s> '\
                        'while <%s> is the %s of the recommended game.'

    def __init__(self, translate_path: str):
        """Construct from the translate json file."""
        with open(translate_path) as file:
            self._translator = json.load(file)

    def explain(self, path: Recommendation.Path) -> str:
        """Explain the path."""
        rel1_str = self.translate_relation(path.relation1)
        entity_str = self.translate_entity(path.entity)
        hist_item_str = self.translate_item(path.hist_item)
        if path.relation1 == path.relation2:
            return self.SAME_RELATION_TPL % (hist_item_str, rel1_str, entity_str)
        rel2_str = self.translate_relation(path.relation2)
        return self.DIFF_RELATION_TPL % (hist_item_str, rel2_str, entity_str, entity_str, rel1_str)

    def translate_item(self, item: int) -> str:
        """Translate an item."""
        return self._translate('item', item)

    def translate_entity(self, entity: int) -> str:
        """Translate an entity."""
        return self._translate('entity', entity)

    def translate_relation(self, relation: int) -> str:
        """Translate a relation."""
        return self._translate('relation', relation)

    def _translate(self, obj_type, obj_id):
        """Translate an object."""
        try:
            return self._translator[obj_type][str(obj_id)]
        except KeyError:
            return f'[{obj_type}:{obj_id}]'
