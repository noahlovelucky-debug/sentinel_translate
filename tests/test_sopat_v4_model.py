from __future__ import annotations

import inspect

import pytest
import torch

from sentinel_v3.sensors import SENTINEL1, SENTINEL2, SensorSpec
from sentinel_v4.model import SOPAT, SOPATConfig


def _tiny_model() -> SOPAT:
    torch.manual_seed(7)
    return SOPAT(
        SOPATConfig(
            width=8,
            hidden=32,
            encoder_depth=1,
            heads=4,
            adapter_rank=8,
            transport_heads=4,
            anchor_window_size=2,
        )
    )


def _inputs(
    source_sensor: SensorSpec = SENTINEL1,
    target_sensor: SensorSpec = SENTINEL2,
    *,
    batch: int = 1,
    frames: int = 2,
    changed: bool = True,
) -> dict[str, object]:
    torch.manual_seed(13)
    height = width = 16
    source_anchor = torch.empty(batch, len(source_sensor.channels), height, width).uniform_(-0.5, 0.5)
    target_anchor = torch.empty(batch, len(target_sensor.channels), height, width).uniform_(-0.5, 0.5)
    observations = source_anchor[:, None].expand(-1, frames, -1, -1, -1).clone()
    if changed:
        observations[:, 0] = (observations[:, 0] + 0.1).clamp(-1.0, 1.0)
        if frames > 1:
            observations[:, 1] = (observations[:, 1] - 0.06).clamp(-1.0, 1.0)
    return {
        "observations": observations,
        "observation_valid": torch.ones(batch, frames, 1, height, width),
        "observation_days": -torch.arange(1, frames + 1, dtype=torch.float32).expand(batch, -1),
        "observation_present": torch.ones(batch, frames, dtype=torch.bool),
        "source_anchor": source_anchor,
        "source_anchor_valid": torch.ones(batch, 1, height, width),
        "target_anchor": target_anchor,
        "target_anchor_valid": torch.ones(batch, 1, height, width),
        "source_anchor_days": torch.full((batch,), -4.0),
        "target_anchor_days": torch.full((batch,), -3.0),
        "source_sensor": source_sensor,
        "target_sensor": target_sensor,
    }


@pytest.mark.parametrize(
    ("source_sensor", "target_sensor"),
    ((SENTINEL1, SENTINEL2), (SENTINEL2, SENTINEL1)),
)
def test_one_model_supports_both_directions_and_initially_copies_anchor(
    source_sensor: SensorSpec, target_sensor: SensorSpec
) -> None:
    model = _tiny_model().eval()
    inputs = _inputs(source_sensor, target_sensor)

    output = model(**inputs)  # type: ignore[arg-type]

    batch, _, height, width = inputs["target_anchor"].shape  # type: ignore[union-attr]
    assert output.physical.shape == (batch, len(target_sensor.channels), height, width)
    assert output.log_variance.shape == (batch, 1, height, width)
    assert [feature.shape[1] for feature in output.transported_change] == [8, 16, 32, 32]
    assert output.source_anchor_cross.shape == inputs["source_anchor"].shape
    assert output.target_anchor_cross.shape == inputs["target_anchor"].shape
    assert torch.equal(output.physical, inputs["target_anchor"])
    assert torch.equal(output.raw_delta, torch.zeros_like(output.raw_delta))


def test_forward_contract_never_accepts_target_label() -> None:
    parameters = inspect.signature(SOPAT.forward).parameters

    assert set(parameters) == {
        "self",
        "observations",
        "observation_valid",
        "observation_days",
        "observation_present",
        "source_anchor",
        "source_anchor_valid",
        "target_anchor",
        "target_anchor_valid",
        "source_anchor_days",
        "target_anchor_days",
        "source_sensor",
        "target_sensor",
    }
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for name, parameter in parameters.items()
        if name != "self"
    )
    assert not {"target", "target_label", "target_values", "label"} & set(parameters)


def test_activation_checkpointing_delegates_to_shared_encoder() -> None:
    model = _tiny_model()

    model.set_activation_checkpointing(True)
    assert model.encoder.activation_checkpointing is True
    model.set_activation_checkpointing(False)
    assert model.encoder.activation_checkpointing is False
    with pytest.raises(TypeError, match="bool"):
        model.set_activation_checkpointing(1)  # type: ignore[arg-type]


def test_target_renderers_keep_independent_channel_updates() -> None:
    model = _tiny_model().eval()
    inputs = _inputs()
    renderer = model.renderers["optical"]
    with torch.no_grad():
        renderer.delta.bias.copy_(torch.linspace(0.1, 1.0, renderer.delta.out_channels))

    output = model(**inputs)  # type: ignore[arg-type]

    assert renderer.delta.out_channels == 10
    assert model.renderers["sar"].delta.out_channels == 2
    assert not torch.allclose(output.raw_delta[:, 0], output.raw_delta[:, 1])
    assert not torch.allclose(output.raw_delta[:, 1], output.raw_delta[:, -1])


def test_set_transport_is_permutation_invariant() -> None:
    model = _tiny_model().eval()
    inputs = _inputs(frames=3)
    with torch.no_grad():
        model.renderers["optical"].delta.weight.normal_(std=0.02)
        model.renderers["optical"].delta.bias.fill_(0.03)
    ordering = torch.tensor([2, 0, 1])
    reordered = dict(inputs)
    for name in ("observations", "observation_valid", "observation_days", "observation_present"):
        reordered[name] = inputs[name][:, ordering]  # type: ignore[index]

    original_output = model(**inputs)  # type: ignore[arg-type]
    reordered_output = model(**reordered)  # type: ignore[arg-type]

    torch.testing.assert_close(original_output.physical, reordered_output.physical, atol=1e-5, rtol=1e-5)
    for original, reordered_feature in zip(
        original_output.transported_change, reordered_output.transported_change, strict=True
    ):
        torch.testing.assert_close(original, reordered_feature, atol=1e-5, rtol=1e-5)


def test_absent_padding_is_inert_even_after_renderer_bias() -> None:
    model = _tiny_model().eval()
    base = _inputs(frames=1)
    with torch.no_grad():
        model.renderers["optical"].delta.bias.fill_(0.25)
    padded = _inputs(frames=2)
    for name in (
        "source_anchor",
        "source_anchor_valid",
        "target_anchor",
        "target_anchor_valid",
        "source_anchor_days",
        "target_anchor_days",
    ):
        padded[name] = base[name]
    padded["observations"][:, 0] = base["observations"][:, 0]  # type: ignore[index]
    padded["observation_valid"][:, 0] = base["observation_valid"][:, 0]  # type: ignore[index]
    padded["observation_days"][:, 0] = base["observation_days"][:, 0]  # type: ignore[index]
    padded["observation_present"][:, 0] = True  # type: ignore[index]
    padded["observations"][:, 1].fill_(0.91)  # type: ignore[index]
    padded["observation_valid"][:, 1].uniform_(0.0, 1.0)  # type: ignore[index]
    padded["observation_days"][:, 1] = 12345.0  # type: ignore[index]
    padded["observation_present"][:, 1] = False  # type: ignore[index]

    base_output = model(**base)  # type: ignore[arg-type]
    padded_output = model(**padded)  # type: ignore[arg-type]

    torch.testing.assert_close(base_output.physical, padded_output.physical, atol=1e-6, rtol=0.0)
    torch.testing.assert_close(base_output.raw_delta, padded_output.raw_delta, atol=1e-6, rtol=0.0)
    for base_change, padded_change in zip(
        base_output.transported_change, padded_output.transported_change, strict=True
    ):
        torch.testing.assert_close(base_change, padded_change, atol=1e-6, rtol=0.0)


def test_factorizer_anchor_only_route_has_nonzero_loss_and_gradients() -> None:
    model = _tiny_model()
    model.set_training_stage("factorizer")
    inputs = _inputs()

    output = model.factorize_anchors(
        source_anchor=inputs["source_anchor"],  # type: ignore[arg-type]
        source_anchor_valid=inputs["source_anchor_valid"],  # type: ignore[arg-type]
        target_anchor=inputs["target_anchor"],  # type: ignore[arg-type]
        target_anchor_valid=inputs["target_anchor_valid"],  # type: ignore[arg-type]
        source_sensor=inputs["source_sensor"],  # type: ignore[arg-type]
        target_sensor=inputs["target_sensor"],  # type: ignore[arg-type]
    )
    loss = (
        (output.source_anchor_reconstruction - inputs["source_anchor"]).square().mean()  # type: ignore[operator]
        + (output.target_anchor_reconstruction - inputs["target_anchor"]).square().mean()  # type: ignore[operator]
    )
    assert loss.item() > 0.0
    loss.backward()

    assert model.encoder.projector.spatial.weight.grad is not None
    assert model.encoder.projector.spatial.weight.grad.abs().sum().item() > 0.0
    assert model.factorizer.common[0].weight.grad is not None
    assert model.factorizer.common[0].weight.grad.abs().sum().item() > 0.0
    assert model.anchor_reconstructors["sar"].weight.grad is not None
    assert model.anchor_reconstructors["optical"].weight.grad is not None
    assert not any(parameter.requires_grad for parameter in model.transport.parameters())
    assert not any(parameter.requires_grad for parameter in model.renderers.parameters())


def test_physical_gradients_reach_trunk_then_null_change_stays_exact_after_training() -> None:
    model = _tiny_model()
    model.set_training_stage("physical")
    inputs = _inputs(frames=1)
    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad], lr=0.05
    )

    first = model(**inputs)  # type: ignore[arg-type]
    first.physical.square().mean().backward()
    assert model.renderers["optical"].delta.bias.grad is not None
    assert model.renderers["optical"].delta.bias.grad.abs().sum().item() > 0.0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    second = model(**inputs)  # type: ignore[arg-type]
    second.physical.square().mean().backward()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum().item() > 0.0
        for parameter in model.transport.parameters()
    )
    assert model.renderers["optical"].delta.bias.detach().abs().sum().item() > 0.0

    null_inputs = dict(inputs)
    null_inputs["observations"] = inputs["source_anchor"][:, None].clone()  # type: ignore[index]
    null_output = model(**null_inputs)  # type: ignore[arg-type]
    assert torch.equal(null_output.raw_delta, torch.zeros_like(null_output.raw_delta))
    assert torch.equal(null_output.physical, inputs["target_anchor"])
