# Copyright 2024 Huawei Technologies Co., Ltd
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
"""generation config."""
import copy
import logging
from typing import Any, Dict

logger = logging.getLogger()

__all__ = ["GenerationConfig"]


class GenerationConfig:
    """Class that holds a configuration for a generation task."""

    def __init__(self, **kwargs):
        # max generate length
        self.max_length = kwargs.pop("max_decode_length", 20)
        self.max_length = kwargs.pop("max_length", self.max_length)
        self.max_new_tokens = kwargs.pop("max_new_tokens", None)
        self.min_length = kwargs.pop("min_length", 0)
        self.min_new_tokens = kwargs.pop("min_new_tokens", None)

        # number of beams
        self.num_beams = kwargs.pop("num_beams", 1)
        # do sample or not
        self.do_sample = kwargs.pop("do_sample", False)
        # incremental infer
        self.use_past = kwargs.pop("use_past", False)
        # logits processors
        self.temperature = kwargs.pop("temperature", 1.0)
        self.top_k = kwargs.pop("top_k", 50)
        self.top_p = kwargs.pop("top_p", 1.0)
        self.repetition_penalty = kwargs.pop("repetition_penalty", 1.0)
        self.encoder_repetition_penalty = kwargs.pop("encoder_repetition_penalty", 1.0)
        self.renormalize_logits = kwargs.pop("renormalize_logits", False)

        # Special tokens that can be used at generation time
        self.pad_token_id = kwargs.pop("pad_token_id", None)
        self.bos_token_id = kwargs.pop("bos_token_id", None)
        self.eos_token_id = kwargs.pop("eos_token_id", None)

        # interface.
        self._from_model_config = kwargs.pop("_from_model_config", False)
        # Additional attributes without default values
        if not self._from_model_config:
            # we don't want to copy values from the model config
            # if we're initializing a `GenerationConfig` from a
            # model's default configuration file
            for key, value in kwargs.items():
                try:
                    setattr(self, key, value)
                except AttributeError as err:
                    logger.error("Can't set %s with value %s for %s", key, value, self)
                    raise err

    def __str__(self) -> str:
        return str(self.__dict__)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], **kwargs):
        """
        Instantiates a [`GenerationConfig`] from a Python dictionary of parameters.

        Args:
            config_dict (`Dict[str, Any]`):
                Dictionary that will be used to instantiate the configuration object.
            kwargs (`Dict[str, Any]`):
                Additional parameters from which to initialize the configuration object.

        Returns:
            [`GenerationConfig`]: The configuration object instantiated from those parameters.
        """
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)

        config = cls(**{**config_dict, **kwargs})
        unused_kwargs = config.update(**kwargs)
        logger.debug("Generate config %s", config)
        if return_unused_kwargs:
            return config, unused_kwargs
        return config

    @classmethod
    def from_model_config(cls, model_config) -> "GenerationConfig":
        """
        Instantiates a [`GenerationConfig`] from a [`PretrainedConfig`].
        This function is useful to convert legacy [`PretrainedConfig`] objects,
        which may contain generation parameters, into a stand-alone [`GenerationConfig`].

        Args:
            model_config (`PretrainedConfig`):
                The model config that will be used to instantiate the generation config.

        Returns:
            [`GenerationConfig`]: The configuration object instantiated from those parameters.
        """
        config_dict = model_config.to_dict()
        config_dict.pop("_from_model_config", None)
        config = cls.from_dict(config_dict, return_unused_kwargs=False, _from_model_config=True)

        return config

    def update(self, **kwargs):
        """
        Updates attributes of this class instance with attributes from `kwargs`
        if they match existing atributtes, returning all the unused kwargs.

        Args:
            kwargs (`Dict[str, Any]`):
                Dictionary of attributes to tentatively update this class.

        Returns:
            `Dict[str, Any]`: Dictionary containing all the key-value pairs
            that were not used to update the instance.
        """
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                to_remove.append(key)
        unused_kwargs = {key: value for key, value in kwargs.items() if key not in to_remove}
        return unused_kwargs

    def to_dict(self) -> Dict[str, Any]:
        """to dict convert function."""
        output = copy.deepcopy(self.__dict__)
        return output
