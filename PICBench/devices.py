import jax
import jax.example_libraries.optimizers as opt
import json
import jax.numpy as jnp
import matplotlib.pyplot as plt  # plotting
import sax
from tqdm.notebook import trange


def straight(wl=1.55, wl0=1.55, neff=2.34, ng=3.4, length=10.0, loss=0.0) -> sax.SDict:
    dwl = wl - wl0
    dneff_dwl = (ng - neff) / wl0
    neff = neff - dwl * dneff_dwl
    phase = 2 * jnp.pi * neff * length / wl
    transmission = 10 ** (-loss * length / 20) * jnp.exp(1j * phase)
    sdict = sax.reciprocal(
        {
            ("I1", "O1"): transmission,
        }
    )
    return sdict


def coupler_single(coupling=0.5) -> sax.SDict:
    kappa = coupling ** 0.5
    tau = (1 - coupling) ** 0.5
    coupler_dict = sax.reciprocal(
        {
            ("I1", "O1"): tau,
            ("I1", "O2"): 1j * kappa,
            ("I2", "O1"): 1j * kappa,
            ("I2", "O2"): tau,
        }
    )
    return coupler_dict


def mmi1x2_single(coupling=0.5) -> sax.SDict:
    kappa = coupling ** 0.5
    tau = (1 - coupling) ** 0.5
    mmi_dict = sax.reciprocal(
        {
            ("I1", "O1"): kappa,
            ("I1", "O2"): tau,
        }
    )
    return mmi_dict


def straight_heat_metal(wl=1.55, wl0=1.55, neff=2.34, ng=3.4, length=10.0, loss=0.0, phase_shift=90) -> sax.SDict:
    phase_shift_rad = phase_shift * jnp.pi / 180
    dwl = wl - wl0
    dneff_dwl = (ng - neff) / wl0
    neff = neff - dwl * dneff_dwl
    phase = 2 * jnp.pi * neff * length / wl + phase_shift_rad
    transmission = 10 ** (-loss * length / 20) * jnp.exp(1j * phase)
    sdict = sax.reciprocal(
        {
            ("I1", "O1"): transmission,
        }
    )
    return sdict


mmi1x2, _ = sax.circuit(
    netlist={
        "instances": {
            'mmi': 'mmi',
            "waveguide1": {'component': 'waveguide', 'settings': {'length': 0.0}},
            "waveguide2": {'component': 'waveguide', 'settings': {'length': 0.0}},
        },
        "connections": {
            "mmi,O1": "waveguide1,I1",
            "mmi,O2": "waveguide2,I1",
        },
        "ports": {
            "I1": "mmi,I1",
            "O1": "waveguide1,O1",
            "O2": "waveguide2,O1",
        },
    },
    models={
        "mmi": mmi1x2_single,
        "waveguide": straight,
    },
)

coupler, _ = sax.circuit(
    netlist={
        "instances": {
            'coupler': 'coupler',
            "waveguide1": {'component': 'waveguide', 'settings': {'length': 0.0}},
            "waveguide2": {'component': 'waveguide', 'settings': {'length': 0.0}},
        },
        "connections": {
            "coupler,O1": "waveguide1,I1",
            "coupler,O2": "waveguide2,I1",
        },
        "ports": {
            "I1": "coupler,I1",
            "I2": "coupler,I2",
            "O1": "waveguide1,O1",
            "O2": "waveguide2,O1",
        },
    },
    models={
        "coupler": coupler_single,
        "waveguide": straight,
    },
)


mzi, _ = sax.circuit(
    netlist={
        "instances": {
            "lft": "mmi",
            "top": "waveguide",
            "btm": "waveguide",
            "rgt": "mmi",
        },
        "connections": {
            "lft,O1": "top,I1",
            "top,O1": "rgt,O1",
            "lft,O2": "btm,I1",
            "btm,O1": "rgt,O2",
        },
        "ports": {
            "I1": "lft,I1",
            "O1": "rgt,I1",
        },
    },
    models={
        "mmi": mmi1x2,
        "waveguide": straight,
    },
)

mzi_2x2, _ = sax.circuit(
    netlist={
        "instances": {
            "lft": "coupler",
            "top": "waveguide",
            "btm": "waveguide",
            "rgt": "coupler",
        },
        "connections": {
            "lft,O1": "top,I1",
            "top,O1": "rgt,I1",
            "lft,O2": "btm,I1",
            "btm,O1": "rgt,I2",
        },
        "ports": {
            "I1": "lft,I1",
            "I2": "lft,I2",
            "O1": "rgt,O1",
            "O2": "rgt,O2",
        },
    },
    models={
        "coupler": coupler,
        "waveguide": straight,
    },
)

OSU, _ = sax.circuit(
        netlist={
            "instances": {
                "coupler1": "coupler",
                "coupler2": "coupler",
                "phase_shifter2": "phase_shifter",
                "phase_shifter1": {"component": "phase_shifter", "settings": {"phase_shift": 270}},
            },
            "connections": {
                "coupler1, O1": "phase_shifter1,I1",
                "phase_shifter1,O1": "coupler2,I1",
                "coupler1, O2": "phase_shifter2,I1",
                "phase_shifter2,O1": "coupler2,I2",
            },
            "ports": {
                "I1": "coupler1,I1",
                "I2": "coupler1,I2",
                "O1": "coupler2,O1",
                "O2": "coupler2,O2",
            },
        },
        models={
            "coupler": coupler,
            "phase_shifter": straight_heat_metal,
        },
    )


mzi_ps, _ = sax.circuit(
    netlist={
        "instances": {
            "coupler1": "coupler",
            "coupler2": "coupler",
            "ps1": "phase_shifter",
            "ps2": "phase_shifter",
            "waveguide": "waveguide",
        },
        "connections": {
            "coupler1,O1": "ps1,I1",
            "coupler1,O2": "waveguide,I1",
            "ps1,O1": "coupler2,I1",
            "waveguide,O1": "coupler2,I2",
            "coupler2,O1": "ps2,I1",
        },
        "ports": {
            "I1": "coupler1,I1",
            "I2": "coupler1,I2",
            "O1": "ps2,O1",
            "O2": "coupler2,O2",
        },
    },
    models={
        "coupler": coupler,
        "phase_shifter": straight_heat_metal,
        "waveguide": straight,
    },
)

mzm, _ = sax.circuit(
    netlist={
        "instances": {
          "splitter": "mmi",
          "combiner": "mmi",
          "phase_shifter1": "phase_shifter",
          "phase_shifter2": "waveguide",
        },
        "connections": {
          "splitter,O1": "phase_shifter1,I1",
          "splitter,O2": "phase_shifter2,I1",
          "phase_shifter1,O1": "combiner,O1",
          "phase_shifter2,O1": "combiner,O2"
        },
        "ports": {
          "I1": "splitter,I1",
          "O1": "combiner,I1",
        },
    },
    models={
        "mmi": mmi1x2,
        "phase_shifter": straight_heat_metal,
        "waveguide": straight,
    },
)

mzm_dual, _ = sax.circuit(
    netlist={
        "instances": {
          "splitter": "mmi",
          "combiner": "mmi",
          "phase_shifter1": "phase_shifter",
          "phase_shifter2": {"component": "phase_shifter", "settings": {"phase_shift": 0}},
        },
        "connections": {
          "splitter,O1": "phase_shifter1,I1",
          "splitter,O2": "phase_shifter2,I1",
          "phase_shifter1,O1": "combiner,O1",
          "phase_shifter2,O1": "combiner,O2",
        },
        "ports": {
          "I1": "splitter,I1",
          "O1": "combiner,I1",
        },
    },
    models={
        "mmi": mmi1x2,
        "phase_shifter": straight_heat_metal,
    },
)


def mrr(wl=1.55, kappa=0.3, neff=2.34, alpha=0.99, cwl=1.55) -> sax.SDict:
    map_dict = {1.543: 9.74, 1.546: 9.78, 1.547: 9.68, 1.55: 9.7, 1.552: 9.5, 1.554: 9.83, 1.556: 9.95}
    r = map_dict[cwl]
    t = (1 - kappa**2) ** 0.5
    beta = neff * (2 * jnp.pi / wl)
    theta = 2 * jnp.pi * r * beta
    ring_dict = sax.reciprocal(
        {
            ("I1", "O1"): jnp.abs((t - alpha * jnp.exp(-1j * theta)) / (1 - t * alpha * jnp.exp(-1j * theta))) ** 2,
            ("I1", "O2"): jnp.abs((1j * kappa * alpha**2 * jnp.exp(-1j * theta /2)) / (1 - t * alpha * jnp.exp(-1j * theta)))**2,
            ("I1", "O3"): 1 - jnp.abs((t - alpha * jnp.exp(-1j * theta)) / (1 - t * alpha * jnp.exp(-1j * theta))) ** 2 - jnp.abs((1j * kappa * alpha**2 * jnp.exp(-1j * theta /2)) / (1 - t * alpha * jnp.exp(-1j * theta)))**2,
        }
    )
    return ring_dict