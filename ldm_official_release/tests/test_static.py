"""Static checks: every own .py parses, every .sh passes `bash -n`, and the scripts
and configs a job script names actually exist."""
import ast
import os
import re
import subprocess
import unittest

from _env import ROOT, own_files, rel


class TestPythonSyntax(unittest.TestCase):
    def test_every_py_file_parses(self):
        bad = []
        for f in own_files(".py"):
            try:
                with open(f, encoding="utf-8") as fh:
                    ast.parse(fh.read(), f)
            except SyntaxError as e:
                bad.append("{}:{}: {}".format(rel(f), e.lineno, e.msg))
        self.assertEqual(bad, [], "\n" + "\n".join(bad))

    def test_no_leftover_conflict_markers(self):
        bad = []
        for f in own_files(".py", ".sh", ".yaml"):
            with open(f, encoding="utf-8", errors="replace") as fh:
                for i, line in enumerate(fh, 1):
                    if line.startswith(("<<<<<<< ", ">>>>>>> ")):
                        bad.append("{}:{}".format(rel(f), i))
        self.assertEqual(bad, [])


class TestShellScripts(unittest.TestCase):
    def test_every_sh_file_passes_bash_n(self):
        bad = []
        for f in own_files(".sh"):
            r = subprocess.run(["bash", "-n", f], capture_output=True, text=True)
            if r.returncode != 0:
                bad.append("{}: {}".format(rel(f), r.stderr.strip()))
        self.assertEqual(bad, [], "\n" + "\n".join(bad))

    def test_referenced_scripts_and_configs_exist(self):
        """A job script that names scripts/foo.py or configs/bar.yaml must find it.

        Only literal names are checked; anything built from a shell variable is
        skipped, as are comment lines.
        """
        pat = re.compile(r"(?<![\w.$}-])((?:scripts|configs)/[A-Za-z0-9_.\-]+\.(?:py|sh|yaml))")
        own_prefixes = ("LDM_ROOT}/", "LDM_ROOT/", "$L/", "${L}/", os.path.basename(ROOT) + "/")
        bad = []
        for f in own_files(".sh"):
            with open(f, encoding="utf-8", errors="replace") as fh:
                for i, line in enumerate(fh, 1):
                    if line.lstrip().startswith("#"):
                        continue
                    code = line.split(" #")[0]
                    for m in pat.finditer(code):
                        # "$ROOT/distillation/scripts/x.sh" is another repo's scripts/.
                        before = code[:m.start()]
                        if before.endswith("/") and not before.endswith(own_prefixes):
                            continue
                        if not os.path.exists(os.path.join(ROOT, m.group(1))):
                            bad.append("{}:{}: {}".format(rel(f), i, m.group(1)))
        self.assertEqual(bad, [], "\n" + "\n".join(bad))


if __name__ == "__main__":
    unittest.main()
