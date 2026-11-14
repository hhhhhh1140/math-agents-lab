import os
import shutil

import pytest

from math_agents.execution import DockerExecutor
from math_agents.backend import DemoBackend
from math_agents.workflow import Runner
from math_agents.optimized import OptimizedRunner


@pytest.mark.skipif(os.getenv("MATH_TEST_DOCKER") != "1" or not shutil.which("docker"),
                    reason="Opt in with MATH_TEST_DOCKER=1 after building the Docker image")
def test_docker_sympy_and_timeout():
    executor = DockerExecutor(timeout=15)
    executor.check()
    result = executor.run("import sympy as s\nx=s.Symbol('x')\nprint(s.solve(2*x+1-7, x)[0])")
    assert result == {"ok": True, "output": "3\n"}
    isolation = executor.run("import os\nfrom pathlib import Path\nassert 'MATH_API_KEY' not in os.environ\nassert not Path('/var/run/docker.sock').exists()\nassert not Path('/app').exists()\nprint('isolated')")
    assert isolation == {"ok": True, "output": "isolated\n"}
    state = Runner(DemoBackend(), executor, 1).run("multi", "Solve 2*x + 1 = 7 for x.")
    assert state["answer"] == "3" and state["accepted"]
    normalized = OptimizedRunner(DemoBackend(), executor).execute("solver", "from sympy import sqrt\nprint(sqrt(2))")
    assert normalized == {"ok": True, "output": "\\sqrt{2}\n"}
    assert not executor.run("raise ValueError('failed check')")["ok"]
    executor.timeout = 2
    assert executor.run("while True: pass")["ok"] is False
