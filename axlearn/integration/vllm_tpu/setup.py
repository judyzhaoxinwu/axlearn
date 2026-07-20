# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""vLLM TPU Plugin Setup for AxLearn Models."""

from setuptools import find_packages, setup

setup(
    name="vllm-axlearn-plugin",
    version="0.1.0",
    description="vLLM TPU plugin adapter for AxLearn models",
    packages=find_packages(),
    entry_points={"vllm.general_plugins": ["register_axlearn_model = vllm_axlearn:register"]},
)
