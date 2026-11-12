"""
Talk to the finger-lock app on the UNO Q from the laptop.

    python3 tools/latch.py mode count    # live finger count on the LED matrix (testing)
    python3 tools/latch.py mode lock     # the real safe
    python3 tools/latch.py lock          # servo to locked
    python3 tools/latch.py unlock        # servo to upright
    python3 tools/latch.py angle 120     # servo to any angle, for tuning
    python3 tools/latch.py snap          # copy what the board's camera sees to snaps/ on the laptop
    python3 tools/latch.py record 3      # board camera saves training pictures of 3 (~6 a second)
    python3 tools/latch.py record stop
    python3 tools/latch.py pull          # copy the board's pictures into data/, then clear them on the board
    python3 tools/latch.py status        # last lines of the app's log
    python3 tools/latch.py watch         # keep printing the log (Ctrl+C to stop)
"""
import base64
import io
import subprocess
import tarfile
import sys
import time
from pathlib import Path

APP = "/home/arduino/ArduinoApps/finger-lock"
TOOLS = Path(__file__).resolve().parent
DEPLOY = TOOLS / "deploy.py"


def board(cmd, raw=False):
    out = subprocess.run([sys.executable, str(DEPLOY), "--cmd", cmd], capture_output=True, text=True)
    if raw:
        return out.stdout
    lines = [l for l in out.stdout.splitlines() if not l.startswith(("Trying", "$ ssh"))]
    return "\n".join(lines + ([out.stderr.strip()] if out.returncode else []))


def send(command):
    board(f"echo '{command}' > {APP}/cmd.txt")
    time.sleep(1.0)


def snap():
    send("snap")
    snaps = TOOLS.parent / "snaps"
    snaps.mkdir(exist_ok=True)
    stamp = time.strftime("%H%M%S")
    for name in ("snap_frame", "snap_crop"):
        out = board(f"echo ==B64==; base64 -w0 {APP}/{name}.jpg; echo; echo ==END==", raw=True)
        body = out.split("\n==B64==\n")[-1].split("\n==END==")[0].strip()
        path = snaps / f"{name}_{stamp}.jpg"
        path.write_bytes(base64.b64decode(body))
        print(f"saved {path}")


def pull():
    out = board(f"cd {APP} && [ -d record ] && echo ==B64== && tar czf - record | base64 -w0; echo; echo ==END==",
                raw=True)
    if "\n==B64==\n" not in out:
        sys.exit("No pictures recorded on the board yet.")
    body = out.split("\n==B64==\n")[-1].split("\n==END==")[0].strip()
    data = TOOLS.parent / "data"
    got = {}
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(body)), mode="r:gz") as tar:
        for m in tar.getmembers():
            parts = Path(m.name).parts          # record/<label>/<file>.jpg
            if m.isfile() and len(parts) == 3 and parts[1] in "012345" and parts[2].endswith(".jpg"):
                dest = data / parts[1] / parts[2]
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(tar.extractfile(m).read())
                got[parts[1]] = got.get(parts[1], 0) + 1
    total = sum(got.values())
    on_board = int(board(f"find {APP}/record -name '*.jpg' | wc -l", raw=True).split()[-1])
    print(f"copied {total} pictures into data/: {dict(sorted(got.items()))}")
    if total == on_board:
        board(f"rm -rf {APP}/record")
        print("cleared them off the board")
    else:
        print(f"board has {on_board}, copied {total}: left them on the board, run pull again")


def main():
    a = sys.argv[1:]
    if a in (["lock"], ["unlock"], ["mode", "count"], ["mode", "lock"]) or \
            (len(a) == 2 and a[0] == "angle" and a[1].isdigit()):
        send(" ".join(a))
        a = ["status"]
    if len(a) == 2 and a[0] == "record" and (a[1] == "stop" or a[1] in list("012345")):
        send(" ".join(a))
        a = ["status"]
    if a == ["pull"]:
        pull()
    elif a == ["snap"]:
        snap()
    elif a == ["status"]:
        print(board(f"arduino-app-cli app logs {APP} --tail 6"))
    elif a == ["watch"]:
        subprocess.run([sys.executable, str(DEPLOY), "--cmd", f"arduino-app-cli app logs {APP} --follow"])
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
