import contextlib
import json
import os
import sys
import traceback

def emit(payload):
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
    sys.stdout.flush()

try:
    with contextlib.redirect_stdout(sys.stderr):
        import laya
        device = os.environ.get("LMT_LAYA_DEVICE", "cuda").strip() or "cuda"
        agent = laya.load(
            "convaiinnovations/laya",
            device=device,
            subfolder="typed-decisions",
        )

    emit({
        "ready": True,
        "version": getattr(laya, "__version__", "unknown"),
        "model": "convaiinnovations/laya/typed-decisions",
        "device": str(agent.device),
    })
except Exception as exc:
    emit({
        "ready": False,
        "error": f"{type(exc).__name__}: {exc}",
        "trace": traceback.format_exc(limit=6),
    })
    sys.exit(2)

for raw in sys.stdin:
    raw = raw.strip()
    if not raw:
        continue

    request_id = None
    try:
        req = json.loads(raw)
        request_id = req.get("id")
        state = req.get("state", {})
        questions = req.get("questions", {})

        with contextlib.redirect_stdout(sys.stderr):
            result = agent.predict(state, questions)

        emit({
            "id": request_id,
            "ok": True,
            "result": result,
            "device": str(agent.device),
        })
    except Exception as exc:
        emit({
            "id": request_id,
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "trace": traceback.format_exc(limit=6),
        })
