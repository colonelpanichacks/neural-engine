"""ANE Training Dashboard — Real-time web UI for Neural Engine transformer training."""

import atexit
import os
import re
import signal
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from threading import Thread, Lock

from flask import Flask, render_template
from flask_socketio import SocketIO, emit

# ═══════════════════════════════════════════════════════════════════
# App setup
# ═══════════════════════════════════════════════════════════════════

app = Flask(__name__)
app.config['SECRET_KEY'] = 'ane-trainer-2026'
socketio = SocketIO(app, async_mode='threading', cors_allowed_origins='*')

TRAIN_DIR = Path(__file__).resolve().parent.parent  # ane-training/training/
MODELS = {
    'qwen3_06b': {'name': 'Qwen3-0.6B', 'dim': 1024, 'hidden': 3072, 'heads': 16,
                   'kv_heads': 8, 'hd': 128, 'seq': 256, 'vocab': 151936, 'layers': 28},
    'stories110m': {'name': 'Stories 110M', 'dim': 768, 'hidden': 2048, 'heads': 12,
                    'kv_heads': 12, 'hd': 64, 'seq': 256, 'vocab': 32000, 'layers': 12},
    'qwen3_06b_seq512': {'name': 'Qwen3-0.6B (SEQ=512)', 'dim': 1024, 'hidden': 3072,
                          'heads': 16, 'kv_heads': 8, 'hd': 128, 'seq': 512,
                          'vocab': 151936, 'layers': 28},
}

# ═══════════════════════════════════════════════════════════════════
# Regex parsers — match train.m printf format strings exactly
# ═══════════════════════════════════════════════════════════════════

RE_STEP = re.compile(
    r'step\s+(\d+)\s+loss=([\d.]+)\s+lr=([\d.e+-]+)\s+([\d.]+)ms/step\s+'
    r'x\[([-\d.]+),([-\d.]+)\]\s+dy\[([-\d.e+-]+),([-\d.e+-]+)\]'
)
RE_TIMING = re.compile(
    r'timing: ane_fwd=([\d.]+) io_fwd=([\d.]+) rms=([\d.]+) '
    r'ane_bwd=([\d.]+) io_bwd=([\d.]+) rms_bwd=([\d.]+) '
    r'cls=([\d.]+) dw=([\d.]+) dwait=([\d.]+) resid=([\d.]+) '
    r'gqa=([\d.]+) rope=([\d.]+) \[gap=([\d.]+)\]'
)
RE_IO_BWD = re.compile(
    r'io_bwd: ffn_w=([\d.]+) ffn_r=([\d.]+) wot_w=([\d.]+) '
    r's1=([\d.]+) s2=([\d.]+) s2r=([\d.]+) qkv_w=([\d.]+) qkv_r=([\d.]+)'
)
RE_GRAD_NORM = re.compile(
    r'grad_norm=([\d.]+)\s+attn=([\d.]+)\s+ffn=([\d.]+)\s+embed=([\d.]+)'
)
RE_ADAM = re.compile(r'\[adam\] step (\d+): ([\d.]+)ms')
RE_CKPT = re.compile(r'\[ckpt saved, best_loss=([\d.]+)\]')
RE_MODEL = re.compile(r'ANE Dynamic Training: (.+?) \((\d+) layers')
RE_CONFIG = re.compile(r'dim=(\d+) q_dim=(\d+) kv_dim=(\d+) hd=(\d+) hidden=(\d+) seq=(\d+) vocab=(\d+)')
RE_PARAMS = re.compile(r'Params: ([\d.]+)M')
RE_RESUMED = re.compile(r'\[RESUMED step (\d+), loss=([\d.]+)\]')
RE_COMPILED = re.compile(r'Compiled (\d+) kernels in (\d+)ms')
RE_FLOPS = re.compile(r'FLOPs/step:.*?total=([\d.]+)M')

# ═══════════════════════════════════════════════════════════════════
# Training state
# ═══════════════════════════════════════════════════════════════════

@dataclass
class TrainingState:
    status: str = 'stopped'  # stopped, compiling, running
    model: str = ''
    model_name: str = ''
    pid: int = 0
    step: int = 0
    loss: float = 0.0
    best_loss: float = 999.0
    lr: float = 0.0
    ms_step: float = 0.0
    seq: int = 256
    params_m: float = 0.0
    total_flops_m: float = 0.0
    kernels: int = 0
    compile_ms: float = 0.0

    # Timing breakdown
    timing: dict = field(default_factory=dict)
    io_bwd: dict = field(default_factory=dict)
    grad_norm: dict = field(default_factory=dict)
    adam_ms: float = 0.0

    # History buffers (capped)
    loss_history: list = field(default_factory=list)
    timing_history: list = field(default_factory=list)
    grad_history: list = field(default_factory=list)
    step_history: list = field(default_factory=list)

    # Log buffer
    log_lines: list = field(default_factory=list)

    MAX_HISTORY: int = 2000
    MAX_LOG: int = 500

    def add_step(self, step, loss, lr, ms_step):
        self.step = step
        self.loss = loss
        self.lr = lr
        self.ms_step = ms_step
        if loss < self.best_loss:
            self.best_loss = loss
        tok_sec = self.seq / (ms_step / 1000) if ms_step > 0 else 0
        entry = {'step': step, 'loss': loss, 'lr': lr, 'ms_step': ms_step, 'tok_sec': tok_sec}
        self.loss_history.append(entry)
        self.step_history.append(entry)
        if len(self.loss_history) > self.MAX_HISTORY:
            self.loss_history = self.loss_history[-self.MAX_HISTORY:]
        if len(self.step_history) > self.MAX_HISTORY:
            self.step_history = self.step_history[-self.MAX_HISTORY:]

    def add_timing(self, t):
        self.timing = t
        self.timing_history.append(t)
        if len(self.timing_history) > 100:
            self.timing_history = self.timing_history[-100:]

    def add_grad(self, g):
        self.grad_norm = g
        self.grad_history.append(g)
        if len(self.grad_history) > self.MAX_HISTORY:
            self.grad_history = self.grad_history[-self.MAX_HISTORY:]

    def add_log(self, line, line_type='info'):
        self.log_lines.append({'line': line.rstrip(), 'type': line_type})
        if len(self.log_lines) > self.MAX_LOG:
            self.log_lines = self.log_lines[-self.MAX_LOG:]

    def to_init_dict(self):
        tok_sec = self.seq / (self.ms_step / 1000) if self.ms_step > 0 else 0
        ane_pct = 0
        if self.timing and self.ms_step > 0:
            ane_pct = (self.timing.get('ane_fwd', 0) + self.timing.get('ane_bwd', 0)) / self.ms_step * 100
        return {
            'status': self.status, 'model': self.model, 'model_name': self.model_name,
            'step': self.step, 'loss': self.loss, 'best_loss': self.best_loss,
            'lr': self.lr, 'ms_step': self.ms_step, 'tok_sec': tok_sec,
            'ane_pct': ane_pct, 'seq': self.seq,
            'params_m': self.params_m, 'total_flops_m': self.total_flops_m,
            'kernels': self.kernels, 'compile_ms': self.compile_ms,
            'timing': self.timing, 'io_bwd': self.io_bwd, 'grad_norm': self.grad_norm,
            'adam_ms': self.adam_ms,
            'loss_history': self.loss_history[-self.MAX_HISTORY:],
            'timing_history': self.timing_history[-100:],
            'grad_history': self.grad_history[-self.MAX_HISTORY:],
            'log_lines': self.log_lines[-self.MAX_LOG:],
        }


state = TrainingState()
state_lock = Lock()
train_proc = None

# ═══════════════════════════════════════════════════════════════════
# Line parser
# ═══════════════════════════════════════════════════════════════════

def parse_line(line):
    """Parse a single line from train.m stdout. Emit WebSocket events for matched patterns."""
    line_type = 'info'

    # Step metrics
    m = RE_STEP.search(line)
    if m:
        step, loss, lr, ms_step = int(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4))
        x_min, x_max = float(m.group(5)), float(m.group(6))
        dy_min, dy_max = float(m.group(7)), float(m.group(8))
        tok_sec = state.seq / (ms_step / 1000) if ms_step > 0 else 0
        with state_lock:
            state.add_step(step, loss, lr, ms_step)
            state.status = 'running'
        socketio.emit('step', {
            'step': step, 'loss': loss, 'lr': lr, 'ms_step': ms_step,
            'x_min': x_min, 'x_max': x_max, 'dy_min': dy_min, 'dy_max': dy_max,
            'tok_sec': tok_sec
        })
        line_type = 'step'

    # Timing breakdown
    m = RE_TIMING.search(line)
    if m:
        keys = ['ane_fwd', 'io_fwd', 'rms', 'ane_bwd', 'io_bwd', 'rms_bwd',
                'cls', 'dw', 'dwait', 'resid', 'gqa', 'rope', 'gap']
        t = {k: float(m.group(i+1)) for i, k in enumerate(keys)}
        with state_lock:
            state.add_timing(t)
        socketio.emit('timing', t)
        line_type = 'timing'

    # IO backward breakdown
    m = RE_IO_BWD.search(line)
    if m:
        keys = ['ffn_w', 'ffn_r', 'wot_w', 's1', 's2', 's2r', 'qkv_w', 'qkv_r']
        d = {k: float(m.group(i+1)) for i, k in enumerate(keys)}
        with state_lock:
            state.io_bwd = d
        socketio.emit('io_bwd', d)

    # Gradient norms
    m = RE_GRAD_NORM.search(line)
    if m:
        g = {'total': float(m.group(1)), 'attn': float(m.group(2)),
             'ffn': float(m.group(3)), 'embed': float(m.group(4))}
        with state_lock:
            state.add_grad(g)
        socketio.emit('grad_norm', g)
        line_type = 'grad'

    # Adam step
    m = RE_ADAM.search(line)
    if m:
        adam_step, adam_ms = int(m.group(1)), float(m.group(2))
        with state_lock:
            state.adam_ms = adam_ms
        socketio.emit('adam', {'step': adam_step, 'ms': adam_ms})
        line_type = 'adam'

    # Checkpoint saved
    m = RE_CKPT.search(line)
    if m:
        best = float(m.group(1))
        with state_lock:
            state.best_loss = best
        socketio.emit('checkpoint', {'best_loss': best})
        line_type = 'ckpt'

    # Model info
    m = RE_MODEL.search(line)
    if m:
        with state_lock:
            state.model_name = m.group(1)

    # Params
    m = RE_PARAMS.search(line)
    if m:
        with state_lock:
            state.params_m = float(m.group(1))

    # Compiled kernels
    m = RE_COMPILED.search(line)
    if m:
        with state_lock:
            state.kernels = int(m.group(1))
            state.compile_ms = float(m.group(2))
            state.status = 'running'
        socketio.emit('status', {'state': 'running', 'pid': state.pid, 'model': state.model})

    # FLOPs
    m = RE_FLOPS.search(line)
    if m:
        with state_lock:
            state.total_flops_m = float(m.group(1))

    # Resumed
    m = RE_RESUMED.search(line)
    if m:
        with state_lock:
            state.step = int(m.group(1))
            state.loss = float(m.group(2))
            state.status = 'compiling'
        socketio.emit('status', {'state': 'compiling', 'pid': state.pid, 'model': state.model})

    # Error detection
    if 'error' in line.lower() or 'FAIL' in line or 'fault' in line.lower():
        line_type = 'error'

    # Add to log
    with state_lock:
        state.add_log(line, line_type)
    socketio.emit('log', {'line': line.rstrip(), 'type': line_type})

# ═══════════════════════════════════════════════════════════════════
# Process management
# ═══════════════════════════════════════════════════════════════════

def stdout_reader(proc):
    """Read stdout from training process line by line."""
    try:
        for raw_line in iter(proc.stdout.readline, b''):
            line = raw_line.decode('utf-8', errors='replace')
            if line:
                parse_line(line)
    except Exception as e:
        socketio.emit('log', {'line': f'[dashboard] reader error: {e}', 'type': 'error'})
    finally:
        proc.wait()
        with state_lock:
            state.status = 'stopped'
        socketio.emit('status', {'state': 'stopped', 'pid': 0, 'model': state.model})
        socketio.emit('log', {'line': f'[dashboard] training process exited (code {proc.returncode})', 'type': 'info'})


def spawn_training(model, resume=True, steps=None, lr=None, accum=None, warmup=None, clip=None, data=None):
    """Spawn the training subprocess."""
    global train_proc

    if train_proc and train_proc.poll() is None:
        return False, 'Training already running'

    # Build command
    model_arg = f' MODEL={model}' if model != 'qwen3_06b' else ' MODEL=qwen3_06b'
    cmd = f'cd training_dynamic && make{model_arg} 2>&1 && ./train'
    cmd += ' --resume' if resume else ' --scratch'
    if steps: cmd += f' --steps {steps}'
    if lr: cmd += f' --lr {lr}'
    if accum: cmd += f' --accum {accum}'
    if warmup: cmd += f' --warmup {warmup}'
    if clip: cmd += f' --clip {clip}'
    if data: cmd += f' --data {data}'

    try:
        train_proc = subprocess.Popen(
            ['bash', '-c', cmd],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            cwd=str(TRAIN_DIR),
            preexec_fn=os.setsid  # process group for clean kill
        )
        with state_lock:
            state.status = 'compiling'
            state.model = model
            state.pid = train_proc.pid
            if model in MODELS:
                state.seq = MODELS[model]['seq']
                state.model_name = MODELS[model]['name']

        # Start reader thread
        reader = Thread(target=stdout_reader, args=(train_proc,), daemon=True)
        reader.start()

        socketio.emit('status', {'state': 'compiling', 'pid': train_proc.pid, 'model': model})
        socketio.emit('log', {'line': f'[dashboard] started: {cmd}', 'type': 'info'})
        return True, 'Training started'

    except Exception as e:
        return False, str(e)


def kill_training():
    """Kill the training subprocess."""
    global train_proc
    if train_proc and train_proc.poll() is None:
        try:
            os.killpg(os.getpgid(train_proc.pid), signal.SIGTERM)
            try:
                train_proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(train_proc.pid), signal.SIGKILL)
                train_proc.wait(timeout=2)
        except ProcessLookupError:
            pass
        with state_lock:
            state.status = 'stopped'
        socketio.emit('status', {'state': 'stopped', 'pid': 0, 'model': state.model})
        socketio.emit('log', {'line': '[dashboard] training stopped by user', 'type': 'info'})
        return True
    return False


def cleanup():
    """Kill training on server shutdown."""
    kill_training()

atexit.register(cleanup)

# ═══════════════════════════════════════════════════════════════════
# Routes + WebSocket events
# ═══════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return render_template('index.html', models=MODELS)


@socketio.on('connect')
def on_connect():
    """Send current state + history to newly connected client."""
    with state_lock:
        emit('init', state.to_init_dict())


@socketio.on('start')
def on_start(data):
    ok, msg = spawn_training(
        model=data.get('model', 'qwen3_06b'),
        resume=data.get('resume', True),
        steps=data.get('steps'),
        lr=data.get('lr'),
        accum=data.get('accum'),
        warmup=data.get('warmup'),
        clip=data.get('clip'),
        data=data.get('data'),
    )
    if not ok:
        emit('log', {'line': f'[dashboard] start failed: {msg}', 'type': 'error'})


@socketio.on('stop')
def on_stop():
    kill_training()


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5055))
    print(f'ANE Training Dashboard: http://localhost:{port}')
    socketio.run(app, host='0.0.0.0', port=port, debug=False, allow_unsafe_werkzeug=True)
