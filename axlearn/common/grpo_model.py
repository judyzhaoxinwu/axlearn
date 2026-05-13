# Copyright © 2026 Apple Inc.

"""Encapsulates composite model architectures wrapping Actor and Reference models."""

from typing import Optional, Tuple

from axlearn.common.base_layer import Module, ParameterSpec
from axlearn.common.base_model import BaseModel
from axlearn.common.config import REQUIRED, Required, config_class
from axlearn.common.module import NestedTensor, Tensor
from axlearn.common.utils import Nested


class GrpoModel(BaseModel):
    """Composite container model orchestrating Actor, Reference, and Sampler configurations."""

    @config_class
    class Config(BaseModel.Config):
        """Configures GrpoModel."""

        actor: Required[BaseModel.Config] = REQUIRED
        reference: Required[BaseModel.Config] = REQUIRED
        sampler: Optional[BaseModel.Config] = None

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)

        # Instantiate children submodules natively inside initialized context
        self._add_child("actor", cfg.actor)
        self._add_child("reference", cfg.reference)
        if cfg.sampler is not None:
            self._add_child("sampler", cfg.sampler)

    def create_parameter_specs_recursively(self) -> Nested[ParameterSpec]:
        specs = {
            "actor": self.actor.create_parameter_specs_recursively(),
            "reference": self.reference.create_parameter_specs_recursively(),
        }
        if "sampler" in self.children:
            specs["sampler"] = self.sampler.create_parameter_specs_recursively()
        return specs

    def forward(self, input_batch: NestedTensor) -> Tuple[Tensor, NestedTensor]:
        """Executes parallel evaluation transformations.

        Returns:
            A tuple of (loss, aux_outputs).
        """
        # Pass data to internal Actor model directly
        actor_outputs, _ = self.actor(input_batch)

        # Reference model forward operations
        reference_outputs, _ = self.reference(input_batch)

        outputs = {
            "actor": actor_outputs,
            "reference": reference_outputs,
        }

        # Dummy placeholder return; functional training logic runs in separate learner steps
        return Tensor(0.0), outputs

    def predict(self, input_batch: NestedTensor) -> NestedTensor:
        """Executes simple forward predictions over task completions."""
        return self.actor.predict(input_batch)
