# app.py
import streamlit as st
from collections import defaultdict, deque

# --------------------------
# Exceptions & Utilities
# --------------------------
class VMException(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code

def safe_int(x, default=0):
    try:
        return int(x)
    except:
        return default

# --------------------------
# Assembly-style Virtual CPU
# --------------------------
class VirtualCPU:
    def __init__(self, mem_size=256, optimize_threshold=3):
        self.registers = {"R0": 0, "R1": 0, "R2": 0, "R3": 0}
        self.memory = [0] * mem_size
        self.pc = 0
        self.running = True
        self.instructions = []
        self.code_cache = {}         # pc -> function
        self.execution_count = defaultdict(int)
        self.flags = {}
        self.last_exception = None
        self.io_log = []
        self.input_queue = deque()
        self.trace_counts = defaultdict(int)
        self.trace_cache = {}        # trace_key -> function
        self.optimize_threshold = optimize_threshold
        self.trace_history = []      # last few traces for UI

    def load_program(self, program_lines):
        self.instructions = [ln.strip() for ln in program_lines if ln.strip()]
        self.pc = 0
        self.running = True
        self.code_cache.clear()
        self.execution_count.clear()
        self.last_exception = None
        self.trace_counts.clear()
        self.trace_cache.clear()
        self.trace_history.clear()

    def execute_step(self):
        """Execute a single step (may execute an optimized trace if present)."""
        if not self.running or self.pc >= len(self.instructions):
            return

        try:
            # If this PC starts a cached trace, execute it.
            trace_key = self._trace_key_starting_at(self.pc)
            if trace_key and trace_key in self.trace_cache:
                # Execute optimized trace function
                self.trace_cache[trace_key]()
                self.trace_counts[trace_key] += 1
                self.trace_history.append((trace_key, self.trace_counts[trace_key]))
                # trimmed history
                if len(self.trace_history) > 20:
                    self.trace_history.pop(0)
                return

            instr = self.instructions[self.pc]
            # Normal execution path (or per-pc code cache)
            self.execution_count[self.pc] += 1

            if self.pc in self.code_cache:
                self.code_cache[self.pc]()  # optimized per-instruction
            else:
                self._run_instruction(instr)

            # After running, maybe optimize per-instruction
            if self.execution_count[self.pc] > self.optimize_threshold and self.pc not in self.code_cache:
                self._optimize_block(self.pc)

        except VMException as e:
            self.running = False
            self.last_exception = {"code": e.code, "msg": str(e), "pc": self.pc}

    def run_all(self, step_limit=10000):
        steps = 0
        while self.running and self.pc < len(self.instructions) and steps < step_limit:
            self.execute_step()
            steps += 1
        if steps >= step_limit:
            self.last_exception = {"code": "STEP_LIMIT", "msg": "Step limit reached", "pc": self.pc}
            self.running = False

    def _run_instruction(self, instr):
        parts = instr.split()
        if not parts:
            return
        cmd = parts[0].upper()

        if cmd == "MOV":
            # MOV Rn value  OR MOV Rn Rm
            dst = parts[1]
            src = parts[2]
            if src in self.registers:
                self.registers[dst] = self.registers[src]
            else:
                self.registers[dst] = safe_int(src)
        elif cmd == "ADD":
            a, b = parts[1], parts[2]
            self.registers[a] += (self.registers[b] if b in self.registers else safe_int(b))
        elif cmd == "SUB":
            a, b = parts[1], parts[2]
            self.registers[a] -= (self.registers[b] if b in self.registers else safe_int(b))
        elif cmd == "MUL":
            a, b = parts[1], parts[2]
            self.registers[a] *= (self.registers[b] if b in self.registers else safe_int(b))
        elif cmd == "DIV":
            a, b = parts[1], parts[2]
            divisor = self.registers[b] if b in self.registers else safe_int(b)
            if divisor == 0:
                raise VMException("DIV_BY_ZERO", f"Attempted DIV by zero at PC={self.pc}")
            self.registers[a] //= divisor
        elif cmd == "JMP":
            addr = safe_int(parts[1])
            if not (0 <= addr < len(self.instructions)):
                raise VMException("BAD_JUMP", f"Jump target out of range: {addr}")
            self.pc = addr - 1  # -1 because execute_step increments pc after execution
        elif cmd == "CMP":
            a, b = parts[1], parts[2]
            aval = self.registers[a] if a in self.registers else safe_int(a)
            bval = self.registers[b] if b in self.registers else safe_int(b)
            self.flags['EQ'] = (aval == bval)
            self.flags['GT'] = (aval > bval)
            self.flags['LT'] = (aval < bval)
        elif cmd == "JE":  # jump if equal
            addr = safe_int(parts[1])
            if self.flags.get('EQ'):
                self.pc = addr - 1
        elif cmd == "PATCH":
            # PATCH target_index new instruction...
            target = safe_int(parts[1])
            new_instr = " ".join(parts[2:])
            if 0 <= target < len(self.instructions):
                self.instructions[target] = new_instr
            else:
                raise VMException("PATCH_ERR", "Invalid patch target")
        elif cmd == "OUT":
            # OUT Rn or OUT immediate
            src = parts[1]
            val = self.registers[src] if src in self.registers else safe_int(src)
            self.io_log.append(("OUT", val))
        elif cmd == "IN":
            # IN Rn -> read from input_queue if available, else 0
            dst = parts[1]
            if self.input_queue:
                self.registers[dst] = self.input_queue.popleft()
            else:
                self.registers[dst] = 0
        elif cmd == "HLT":
            self.running = False
        else:
            # Unknown instruction
            raise VMException("ILLEGAL_INSTR", f"Unknown instruction '{cmd}' at PC={self.pc}")

        # advance program counter by default
        self.pc += 1

    # ----------------
    # Per-instruction optimization (simple)
    def _optimize_block(self, pc):
        """Create a specialized python function for this single instruction"""
        instr = self.instructions[pc]
        parts = instr.split()
        cmd = parts[0].upper()

        def make_fn(parts_local):
            def fn():
                # This is intentionally lightweight and mirrors logic in _run_instruction
                # but avoids heavy parsing each time.
                if parts_local[0] == "MOV":
                    dst = parts_local[1]
                    src = parts_local[2]
                    if src in self.registers:
                        self.registers[dst] = self.registers[src]
                    else:
                        self.registers[dst] = safe_int(src)
                elif parts_local[0] == "ADD":
                    a, b = parts_local[1], parts_local[2]
                    self.registers[a] += (self.registers[b] if b in self.registers else safe_int(b))
                elif parts_local[0] == "SUB":
                    a, b = parts_local[1], parts_local[2]
                    self.registers[a] -= (self.registers[b] if b in self.registers else safe_int(b))
                elif parts_local[0] == "OUT":
                    src = parts_local[1]
                    val = self.registers[src] if src in self.registers else safe_int(src)
                    self.io_log.append(("OUT", val))
                else:
                    # fall back for complex ops
                    self._run_instruction(" ".join(parts_local))
            return fn

        self.code_cache[pc] = make_fn(parts)

    # ----------------
    # Trace recording & optimization
    def _trace_key_starting_at(self, start_pc, max_len=8):
        """
        Identify a potential trace starting at start_pc by walking linear sequence
        until branch/halt or max_len reached. Returns tuple of PCs.
        """
        if start_pc >= len(self.instructions):
            return None
        trace = []
        pc = start_pc
        steps = 0
        while 0 <= pc < len(self.instructions) and steps < max_len:
            trace.append(pc)
            instr = self.instructions[pc].split()[0].upper()
            steps += 1
            # break on branch-like instruction
            if instr in ("JMP", "JE", "HLT", "CMP", "DIV", "JE", "PATCH"):
                break
            pc += 1
        return tuple(trace)

    def record_and_maybe_optimize_trace(self, start_pc):
        trace_key = self._trace_key_starting_at(start_pc)
        if not trace_key:
            return
        self.trace_counts[trace_key] += 1
        if self.trace_counts[trace_key] > self.optimize_threshold and trace_key not in self.trace_cache:
            self._optimize_trace(trace_key)

    def _optimize_trace(self, trace_key):
        # Build a fast Python function executing sequence of instructions for the trace.
        instrs = [self.instructions[pc] for pc in trace_key]

        def trace_fn():
            # execute sequence without reparsing heavily
            for i, instr in enumerate(instrs):
                parts = instr.split()
                cmd = parts[0].upper()
                # very limited optimized ops
                if cmd == "MOV":
                    dst = parts[1]; src = parts[2]
                    if src in self.registers:
                        self.registers[dst] = self.registers[src]
                    else:
                        self.registers[dst] = safe_int(src)
                elif cmd == "ADD":
                    a, b = parts[1], parts[2]
                    self.registers[a] += (self.registers[b] if b in self.registers else safe_int(b))
                elif cmd == "SUB":
                    a, b = parts[1], parts[2]
                    self.registers[a] -= (self.registers[b] if b in self.registers else safe_int(b))
                elif cmd == "OUT":
                    src = parts[1]
                    val = self.registers[src] if src in self.registers else safe_int(src)
                    self.io_log.append(("OUT", val))
                else:
                    # fallback for complex instruction
                    self._run_instruction(instr)
            # After executing trace, set PC to end+1 (simulate linear execution)
            last_pc = trace_key[-1]
            self.pc = last_pc + 1

        self.trace_cache[trace_key] = trace_fn
        # also add to trace history for UI
        self.trace_history.append((trace_key, 0))


# --------------------------
# Bytecode Virtual Machine (stack-based, heap + GC)
# --------------------------
class BytecodeVM:
    def __init__(self):
        self.stack = []
        self.heap = {}         # obj_id -> dict(fields)
        self.pc = 0
        self.code = []
        self.running = True
        self.next_obj_id = 1
        self.frames = []       # not heavily used, but placeholder for CALL/RET
        self.last_exception = None

    def load(self, code_lines):
        self.code = [ln.strip() for ln in code_lines if ln.strip()]
        self.pc = 0
        self.running = True
        self.stack.clear()
        self.heap.clear()
        self.next_obj_id = 1
        self.frames.clear()
        self.last_exception = None

    def step(self):
        if not self.running or self.pc >= len(self.code):
            return
        instr = self.code[self.pc]
        parts = instr.split()
        op = parts[0].upper()
        try:
            if op == "PUSH":
                self.stack.append(safe_int(parts[1]))
            elif op == "POP":
                if self.stack: self.stack.pop()
            elif op == "ADD":
                b = self.stack.pop(); a = self.stack.pop(); self.stack.append(a + b)
            elif op == "SUB":
                b = self.stack.pop(); a = self.stack.pop(); self.stack.append(a - b)
            elif op == "MUL":
                b = self.stack.pop(); a = self.stack.pop(); self.stack.append(a * b)
            elif op == "DIV":
                b = self.stack.pop(); a = self.stack.pop()
                if b == 0:
                    raise VMException("DIV_BY_ZERO", f"Bytecode DIV by zero at PC={self.pc}")
                self.stack.append(a // b)
            elif op == "NEW":
                oid = self.next_obj_id; self.next_obj_id += 1
                self.heap[oid] = {"__refcount": 0}
                self.stack.append(oid)
            elif op == "PUTFIELD":
                field = parts[1]
                val = self.stack.pop()
                obj = self.stack.pop()
                if obj not in self.heap:
                    raise VMException("NO_SUCH_OBJ", f"Invalid object id {obj}")
                self.heap[obj][field] = val
            elif op == "GETFIELD":
                field = parts[1]
                obj = self.stack.pop()
                if obj not in self.heap:
                    raise VMException("NO_SUCH_OBJ", f"Invalid object id {obj}")
                self.stack.append(self.heap[obj].get(field, 0))
            elif op == "GC":
                self.simple_gc()
            elif op == "PRINT":
                # debug helper: push top-of-stack value into last_exception field for UI display
                val = self.stack[-1] if self.stack else None
                self.last_exception = {"code": "PRINT", "msg": f"TOP={val}"}
            elif op == "HLT":
                self.running = False
            else:
                raise VMException("ILLEGAL_BYTECODE", f"Unknown bytecode '{op}' at PC={self.pc}")
        except VMException as e:
            self.running = False
            self.last_exception = {"code": e.code, "msg": str(e), "pc": self.pc}
        finally:
            self.pc += 1

    def run_all(self, step_limit=10000):
        steps = 0
        while self.running and self.pc < len(self.code) and steps < step_limit:
            self.step()
            steps += 1
        if steps >= step_limit:
            self.last_exception = {"code": "STEP_LIMIT", "msg": "Step limit reached", "pc": self.pc}
            self.running = False

    def simple_gc(self):
        # Mark reachable from stack
        reachable = set()
        to_visit = [x for x in self.stack if isinstance(x, int) and x in self.heap]
        while to_visit:
            oid = to_visit.pop()
            if oid in reachable: continue
            reachable.add(oid)
            for v in self.heap[oid].values():
                if isinstance(v, int) and v in self.heap and v not in reachable:
                    to_visit.append(v)
        # Sweep
        removed = []
        for oid in list(self.heap.keys()):
            if oid not in reachable:
                del self.heap[oid]
                removed.append(oid)
        return removed

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="Virtual Machines — Interactive Demo", layout="wide")
st.title("🎛️ Virtual Machines — Interactive Simulator (Assembly + Bytecode + GC + Traces)")

# Initialize session-state singletons
if "asm_cpu" not in st.session_state:
    st.session_state.asm_cpu = VirtualCPU()

if "bc_vm" not in st.session_state:
    st.session_state.bc_vm = BytecodeVM()

# Demo programs (assembly + bytecode)
DEMO_ASM = {
    "Hot Loop (optimize demo)": [
        "MOV R0 0",
        "MOV R1 1",
        "MOV R2 1000",
        "ADD R0 R1",
        "SUB R2 R1",
        "CMP R2 R1",
        "JE 8",
        "JMP 3",
        "HLT"
    ],
    "DIV by zero (exception)": [
        "MOV R0 10",
        "MOV R1 0",
        "DIV R0 R1",
        "HLT"
    ],
    "Self-modifying PATCH demo": [
        "MOV R0 1",
        "PATCH 2 MOV R0 99",  # modify instruction at index 2 (0-based)
        "MOV R1 2",
        "HLT"
    ],
    "I/O demo (OUT/IN)": [
        "MOV R0 42",
        "OUT R0",
        "IN R2",
        "OUT R2",
        "HLT"
    ],
}

DEMO_BC = {
    "Object + GC demo": [
        "NEW",
        "PUSH 10",
        "PUTFIELD val",
        "NEW",
        "PUSH 20",
        "PUTFIELD val",
        "GC",
        "HLT"
    ],
    "Math + PRINT": [
        "PUSH 15",
        "PUSH 27",
        "ADD",
        "PRINT",
        "HLT"
    ],
    "DIV by zero (bytecode)": [
        "PUSH 10",
        "PUSH 0",
        "DIV",
        "HLT"
    ]
}

tabs = st.tabs(["Assembly VM", "Bytecode VM", "Traces", "GC", "I/O", "Demo Programs", "README"])

# --------------------------
# Assembly VM Tab
# --------------------------
with tabs[0]:
    st.header("Assembly-style Virtual CPU")
    asm_col1, asm_col2 = st.columns([2, 1])
    with asm_col1:
        asm_program_text = st.text_area("Assembly Program (one instruction per line):",
                                        value="\n".join(DEMO_ASM["Hot Loop (optimize demo)"]),
                                        height=240)
        asm_lines = asm_program_text.splitlines()
        if st.button("Load Assembly Program"):
            st.session_state.asm_cpu = VirtualCPU()
            st.session_state.asm_cpu.load_program(asm_lines)
            st.success("Assembly program loaded.")
        run_step = st.button("Step (Assembly)")
        run_all = st.button("Run All (Assembly)")
        reset_asm = st.button("Reset Assembly VM")

    with asm_col2:
        st.subheader("Registers")
        st.table(st.session_state.asm_cpu.registers.items())
        st.subheader("PC")
        st.write(st.session_state.asm_cpu.pc)
        st.subheader("Flags")
        st.write(st.session_state.asm_cpu.flags)
        st.subheader("Last Exception")
        st.write(st.session_state.asm_cpu.last_exception)

    if run_step:
        st.session_state.asm_cpu.execute_step()
    if run_all:
        st.session_state.asm_cpu.run_all()
    if reset_asm:
        st.session_state.asm_cpu = VirtualCPU()
        st.success("Assembly VM reset.")

    st.subheader("First 32 bytes of Memory")
    mem_preview = {i: st.session_state.asm_cpu.memory[i] for i in range(32)}
    st.table(mem_preview.items())

    st.subheader("Code Cache (Optimized per-PC keys)")
    st.write(list(st.session_state.asm_cpu.code_cache.keys()))

    st.subheader("Trace Cache Keys (optimized traces)")
    st.write([tuple(k) for k in st.session_state.asm_cpu.trace_cache.keys()])

    st.subheader("I/O Log")
    st.write(st.session_state.asm_cpu.io_log)

# --------------------------
# Bytecode VM Tab
# --------------------------
with tabs[1]:
    st.header("Bytecode VM (Stack + Heap + GC)")
    bc_col1, bc_col2 = st.columns([2, 1])
    with bc_col1:
        bc_program_text = st.text_area("Bytecode Program (PUSH/NEW/PUTFIELD/GETFIELD/GC/HLT):",
                                       value="\n".join(DEMO_BC["Object + GC demo"]),
                                       height=240)
        bc_lines = bc_program_text.splitlines()
        if st.button("Load Bytecode Program"):
            st.session_state.bc_vm = BytecodeVM()
            st.session_state.bc_vm.load(bc_lines)
            st.success("Bytecode program loaded.")
        if st.button("Step (Bytecode)"):
            st.session_state.bc_vm.step()
        if st.button("Run All (Bytecode)"):
            st.session_state.bc_vm.run_all()
        if st.button("Reset Bytecode VM"):
            st.session_state.bc_vm = BytecodeVM()
            st.success("Bytecode VM reset.")

    with bc_col2:
        st.subheader("Stack (top at bottom)")
        st.write(list(st.session_state.bc_vm.stack))
        st.subheader("Heap (objects)")
        st.write(st.session_state.bc_vm.heap)
        st.subheader("PC")
        st.write(st.session_state.bc_vm.pc)
        st.subheader("Last Exception/HINT")
        st.write(getattr(st.session_state.bc_vm, "last_exception", None))

# --------------------------
# Traces Tab
# --------------------------
with tabs[2]:
    st.header("Trace Recording & Optimization")
    st.write("Traces are short linear sequences of PCs detected at runtime. You can optimize a trace to produce a fast path.")
    trace_display = []
    for tk, count in st.session_state.asm_cpu.trace_counts.items():
        trace_display.append({"trace": str(tk), "count": count, "optimized": tk in st.session_state.asm_cpu.trace_cache})
    st.table(trace_display)

    st.subheader("Trace History (recent)")
    history = [{"trace": str(t[0]), "occurrence": t[1]} for t in st.session_state.asm_cpu.trace_history]
    st.table(history)

    with st.expander("Manually optimize a trace (paste tuple like (0,1,2))"):
        manual = st.text_input("Trace key")
        if st.button("Optimize Manual Trace"):
            try:
                parsed = tuple(int(x.strip()) for x in manual.strip("() ").split(",") if x.strip())
                if parsed not in st.session_state.asm_cpu.trace_cache:
                    st.session_state.asm_cpu._optimize_trace(parsed)
                    st.success(f"Optimized trace {parsed}")
                else:
                    st.info("Trace already optimized.")
            except Exception as e:
                st.error("Could not parse trace key. Use format: (0,1,2)")

# --------------------------
# GC Tab (for Bytecode VM)
# --------------------------
with tabs[3]:
    st.header("Garbage Collection (Bytecode VM)")
    st.write("Trigger the Bytecode VM's simple mark-and-sweep GC and observe heap shrinkage.")
    if st.button("Trigger GC (Bytecode)"):
        removed = st.session_state.bc_vm.simple_gc()
        st.info(f"Removed objects: {removed}")
    st.subheader("Heap")
    st.write(st.session_state.bc_vm.heap)

# --------------------------
# I/O Tab
# --------------------------
with tabs[4]:
    st.header("I/O Simulation (Assembly VM)")
    st.write("Use IN to read from input queue; OUT logs values. Input queue is an explicit list you control.")
    with st.form("io_form"):
        add_val = st.text_input("Value to enqueue (integer)", value="123")
        submitted = st.form_submit_button("Enqueue Input")
        if submitted:
            try:
                val = int(add_val)
                st.session_state.asm_cpu.input_queue.append(val)
                st.success(f"Enqueued {val}")
            except:
                st.error("Please enter an integer.")

    st.subheader("Input Queue (left to right pop order)")
    st.write(list(st.session_state.asm_cpu.input_queue))
    st.subheader("I/O Log (most recent last)")
    st.write(st.session_state.asm_cpu.io_log)

# --------------------------
# Demo Programs Tab
# --------------------------
with tabs[5]:
    st.header("Demo Programs")
    st.write("Select a demo and load it into the VM(s) to showcase features.")
    demo_category = st.selectbox("Category", ["Assembly", "Bytecode"])
    if demo_category == "Assembly":
        demo_name = st.selectbox("Assembly Demos", list(DEMO_ASM.keys()))
        if st.button("Load Assembly Demo"):
            st.session_state.asm_cpu = VirtualCPU()
            st.session_state.asm_cpu.load_program(DEMO_ASM[demo_name])
            st.success(f"Loaded assembly demo: {demo_name}")
    else:
        demo_name = st.selectbox("Bytecode Demos", list(DEMO_BC.keys()))
        if st.button("Load Bytecode Demo"):
            st.session_state.bc_vm = BytecodeVM()
            st.session_state.bc_vm.load(DEMO_BC[demo_name])
            st.success(f"Loaded bytecode demo: {demo_name}")

    st.markdown("""
    **Suggested demo route to impress your professor**
    1. Load **Hot Loop (optimize demo)** and show trace/cache growth while running `Run Step` repeatedly or `Run All`.  
    2. Load **DIV by zero** and demonstrate precise trap handling (show `Last Exception`).  
    3. Load **Self-modifying PATCH demo** and show the instruction list before/after.  
    4. Switch to Bytecode, load **Object + GC demo**, allocate objects, show heap, trigger `GC`.  
    5. Show I/O demo: enqueue input, `IN` reads it, `OUT` logs it.
    """)
