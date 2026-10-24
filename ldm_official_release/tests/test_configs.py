"""Config checks, no model built and no data read.

For every configs/*.yaml:
  1. it parses;
  2. every `target:` imports to a real class;
  3. every key under that node's `params:` is a keyword the class accepts, and no
     required keyword is missing. A misspelt switch otherwise either crashes ten
     minutes into a queued job or -- when the class takes **kwargs -- is swallowed
     and the run trains the DEFAULT recipe under the wrong name.
"""
import glob
import importlib
import inspect
import os
import unittest

from _env import ROOT, rel

from omegaconf import OmegaConf

CONFIGS = sorted(glob.glob(os.path.join(ROOT, "configs", "*.yaml")))
_P = inspect.Parameter
OWN_PACKAGES = {"ldm_ftmse", "ldm", "taming", "main"}


def target_nodes(node, path=""):
    """Yield (path, target, params-dict) for every {target: ...} mapping in a config."""
    if isinstance(node, dict):
        if isinstance(node.get("target"), str):
            yield path, node["target"], node.get("params") or {}
        for k, v in node.items():
            yield from target_nodes(v, "{}.{}".format(path, k) if path else str(k))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from target_nodes(v, "{}[{}]".format(path, i))


def resolve(target):
    module, name = target.rsplit(".", 1)
    return getattr(importlib.import_module(module), name)


def accepted_keywords(cls):
    """(names, open_ended, required) for cls(**params).

    Walks the MRO while each __init__ takes **kwargs, which in this codebase always
    means super().__init__(*args, **kwargs). That convention is only assumed for the
    repo's own packages: a third-party **kwargs (WandbLogger's goes to wandb.init) is
    treated as open-ended, where any key is legal and only `required` can be checked.
    """
    names, required = set(), set()
    for i, klass in enumerate(inspect.getmro(cls)):
        init = klass.__dict__.get("__init__")
        if init is None:
            continue
        try:
            sig = inspect.signature(init)
        except (TypeError, ValueError):
            return names, True, required
        ps = list(sig.parameters.values())[1:]
        forwards = any(p.kind is _P.VAR_KEYWORD for p in ps)
        for p in ps:
            if p.kind in (_P.POSITIONAL_OR_KEYWORD, _P.KEYWORD_ONLY):
                names.add(p.name)
                if i == 0 and p.default is _P.empty:
                    required.add(p.name)
        if not forwards:
            return names, False, required
        if klass.__module__.split(".")[0] not in OWN_PACKAGES:
            return names, True, required
    return names, True, required


class TestConfigs(unittest.TestCase):
    def test_configs_exist(self):
        self.assertTrue(CONFIGS, "no configs/*.yaml under " + ROOT)

    def test_every_config_parses(self):
        bad = []
        for f in CONFIGS:
            try:
                cfg = OmegaConf.load(f)
                # _*.yaml are fragments merged onto a full config with a second --base.
                if os.path.basename(f).startswith("_"):
                    continue
                if "model" not in cfg or "data" not in cfg:
                    bad.append("{}: no model/data section".format(rel(f)))
            except Exception as e:                                  # noqa: BLE001
                bad.append("{}: {}".format(rel(f), e))
        self.assertEqual(bad, [], "\n" + "\n".join(bad))

    def test_targets_import_and_accept_their_params(self):
        bad, cache = [], {}
        for f in CONFIGS:
            cfg = OmegaConf.to_container(OmegaConf.load(f), resolve=False)
            for path, target, params in target_nodes(cfg):
                if target not in cache:
                    try:
                        cache[target] = accepted_keywords(resolve(target))
                    except Exception as e:                          # noqa: BLE001
                        cache[target] = e
                got = cache[target]
                if isinstance(got, Exception):
                    bad.append("{} [{}]: cannot import {}: {}".format(rel(f), path, target, got))
                    continue
                names, open_ended, required = got
                unknown = sorted(set(params) - names) if not open_ended else []
                missing = sorted(required - set(params))
                if unknown:
                    bad.append("{} [{}]: {} does not accept {}".format(rel(f), path, target, unknown))
                if missing:
                    bad.append("{} [{}]: {} needs {}".format(rel(f), path, target, missing))
        self.assertEqual(bad, [], "\n" + "\n".join(bad))

    def test_band_weight_keys_are_valid_band_reprs(self):
        """Steerable band weights are keyed by repr(tuple); a typo is only caught at
        the first training step, after the queue wait."""
        import re
        ok = re.compile(r"^\('band', \d+, \d+\)$|^\('residual_(high|low)pass',\)$")
        bad = []
        for f in CONFIGS:
            cfg = OmegaConf.to_container(OmegaConf.load(f), resolve=False)
            for path, _, params in target_nodes(cfg):
                for key, val in params.items():
                    if "band_weights" in key and isinstance(val, dict):
                        bad += ["{} [{}.{}]: {!r}".format(rel(f), path, key, k)
                                for k in val if not ok.match(str(k))]
        self.assertEqual(bad, [], "\n" + "\n".join(bad))


if __name__ == "__main__":
    unittest.main()
