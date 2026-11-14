import subprocess
import threading
import uuid


class DockerExecutor:
    def __init__(self, image="math-agent-python:local", timeout=15):
        self.image, self.timeout = image, timeout

    def check(self):
        subprocess.run(["docker", "image", "inspect", self.image], check=True, capture_output=True, timeout=15)

    def run(self, code):
        if len(code.encode()) > 32000:
            return {"ok": False, "output": "Code exceeds 32 KB limit"}
        name = "math-agent-" + uuid.uuid4().hex
        # No host mounts, API credentials, network, privileges, or writable root.
        command = ["docker", "run", "--rm", "-i", "--name", name,
                   "--network=none", "--read-only", "--cap-drop=ALL",
                   "--security-opt=no-new-privileges", "--memory=256m", "--cpus=1",
                   "--pids-limit=32", "--ulimit", "fsize=1048576:1048576",
                   "--tmpfs", "/tmp:rw,noexec,nosuid,size=16m", self.image]
        # Drain output continuously, retaining at most 16 KB in host memory.
        with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT) as process:
            output = bytearray()
            def drain():
                while chunk := process.stdout.read(4096):
                    output.extend(chunk[:max(0, 16000 - len(output))])
            reader = threading.Thread(target=drain, daemon=True)
            reader.start()
            def feed():
                try:
                    process.stdin.write(code.encode())
                    process.stdin.close()
                except (BrokenPipeError, OSError):
                    pass
            writer = threading.Thread(target=feed, daemon=True)
            writer.start()
            try:
                process.wait(timeout=self.timeout)
                writer.join(timeout=2)
                reader.join(timeout=2)
                return {"ok": process.returncode == 0, "output": output.decode(errors="replace")}
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
                return {"ok": False, "output": "Execution timed out"}
            finally:
                subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=15)


class DemoExecutor:
    """Does NOT execute code. Maps exact offline fixture strings to outputs."""
    def run(self, code):
        known = {"print(12)": "12", "print(3)": "3", "print('PASS')": "PASS"}
        return {"ok": code in known, "output": known.get(code, "Unsupported demo code")}
