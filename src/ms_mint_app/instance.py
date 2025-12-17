import dash
from dash import dcc, html
import os
import sys
import signal
import atexit
import contextlib
import psutil
from pathlib import Path
import platform
import threading
import logging

class SingleInstance:
    """Ensures only one instance of the app is running"""

    def __init__(self, name="dash_app", temp_dir=None, debug=False):
        self.name = name
        self.debug = debug
        self.pid_file = self._get_pid_file(temp_dir)
        self._cleaned = False
        self.logger = logging.getLogger(f"SingleInstance.{name}")
        
        if self.debug:
            self.logger.info("Debug mode enabled - instance checking and cleanup will be skipped")
            return

        # Register automatic cleanup
        atexit.register(self._cleanup)

        # Capture termination signals
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

    def _get_pid_file(self, temp_dir):
        """Gets PID file path based on OS"""
        base = Path(temp_dir) or Path(os.environ.get('TEMP', '.')) if platform.system() == 'Windows' else Path('/tmp')
        return base / f"{self.name}.pid"

    def _handle_signal(self, signum, frame):
        """
        Handles termination signals.
        Does NOT cleanup here - just triggers normal Python shutdown.
        """
        self.logger.info(f"Signal {signum} received. Initiating shutdown...")
        # sys.exit() will trigger all atexit handlers including Dash cleanup
        sys.exit(0)

    def _cleanup(self):
        """
        Cleans up subprocesses, threads, and PID file.
        Automatically executed at the end of Python shutdown.
        """
        if self._cleaned or self.debug:
            return

        self._cleaned = True
        self.logger.info("Starting final cleanup...")

        try:
            current_proc = psutil.Process(os.getpid())

            if children := current_proc.children(recursive=True):
                self.logger.info(f"Found {len(children)} child process(es)")
                for child in children:
                    with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
                        self.logger.info(f"Terminating: PID {child.pid} ({child.name()})")
                        child.terminate()
                # Wait for graceful termination
                gone, alive = psutil.wait_procs(children, timeout=3)

                # Force kill if they didn't respond
                if alive:
                    self.logger.warning(f"{len(alive)} process(es) didn't respond, forcing kill...")
                    for p in alive:
                        with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
                            p.kill()
            # Info about threads (Python handles them automatically)
            active_threads = threading.enumerate()
            if len(active_threads) > 1:
                self.logger.info(f"Active threads: {len(active_threads)}")
                for thread in active_threads:
                    if thread != threading.main_thread():
                        self.logger.debug(f"  - {thread.name} (daemon={thread.daemon})")

        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            self.logger.warning(f"Error listing processes: {e}")

        # Clean up PID file
        try:
            if self.pid_file.exists():
                self.pid_file.unlink()
                self.logger.info("PID file removed")
        except Exception as e:
            self.logger.warning(f"Error removing PID file: {e}")

        self.logger.info("Cleanup completed")

    def _is_running(self):
        """Checks if an instance is already running"""
        if not self.pid_file.exists():
            return False, None

        with contextlib.suppress(ValueError, psutil.NoSuchProcess, psutil.AccessDenied):
            with open(self.pid_file, 'r') as f:
                pid = int(f.read().strip())

            if psutil.pid_exists(pid):
                proc = psutil.Process(pid)
                if 'python' in proc.name().lower():
                    return True, pid
        self.pid_file.unlink()
        return False, None

    def _kill_existing(self, pid):
        """Kills an existing instance and its children"""
        try:
            self.logger.info(f"Killing previous instance (PID: {pid})")
            proc = psutil.Process(pid)

            # Terminate children first
            children = proc.children(recursive=True)
            for child in children:
                with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
                    child.terminate()
            psutil.wait_procs(children, timeout=3)

            # Terminate main process
            proc.terminate()
            proc.wait(timeout=5)

            if self.pid_file.exists():
                self.pid_file.unlink()

            self.logger.info("Previous instance terminated")
            return True
        except Exception as e:
            self.logger.error(f"Could not kill existing instance: {e}")
            return False

    def ensure_single(self, force=False):
        """
        Ensures only one instance exists

        Args:
            force: If True, kills existing instance. If False, exits with error.
        """
        if self.debug:
            self.logger.info("Debug mode - skipping instance checking")
            return
            
        running, pid = self._is_running()

        if running:
            if force:
                if not self._kill_existing(pid):
                    sys.exit(1)
            else:
                self.logger.error(f"Instance already running (PID: {pid})")
                self.logger.info(f"Use force=True or execute: kill {pid}")
                sys.exit(1)

        # Save current PID
        with open(self.pid_file, 'w') as f:
            f.write(str(os.getpid()))

        self.logger.info(f"{self.name} started (PID: {os.getpid()})")
        self.logger.info(f"System: {platform.system()}")

if __name__ == "__main__":
    # Create logger for main
    logger = logging.getLogger("Main")

    # 1. Create single instance manager
    single = SingleInstance(name="mi_dash_app", port=8050, debug=True)

    # 2. Ensure single instance (will be skipped in debug mode)
    single.ensure_single(force=True)

    # 4. Create Dash app
    app = dash.Dash(__name__)

    app.layout = html.Div([
        html.H1("🔒 Single Instance - Clean Shutdown"),
        html.Div([
            html.H3("Shutdown Flow:"),
            html.Ol([
                html.Li("SIGINT/SIGTERM signal received"),
                html.Li("Signal handler calls sys.exit()"),
                html.Li("Dash closes its resources (server, etc)"),
                html.Li("Python executes atexit handlers"),
                html.Li("SingleInstance._cleanup() executes"),
                html.Li("Subprocesses terminated, PID cleaned"),
            ]),
        ], style={'padding': '20px', 'background': '#e8f5e9', 'borderRadius': '5px'}),
        html.Div([
            html.P(f"PID: {os.getpid()}"),
            html.P(f"System: {platform.system()}"),
        ], style={'padding': '10px', 'background': '#f0f0f0', 'marginTop': '10px'}),
        dcc.Interval(id='interval', interval=1000, n_intervals=0),
        html.Div(id='output')
    ])


    @app.callback(
        dash.Output('output', 'children'),
        dash.Input('interval', 'n_intervals')
    )
    def update(n):
        proc = psutil.Process(os.getpid())
        children = len(proc.children(recursive=True))
        threads = len(threading.enumerate())

        return html.Div([
            html.P(f"⏱️ Uptime: {n}s"),
            html.P(f"👶 Child processes: {children}"),
            html.P(f"🧵 Threads: {threads}"),
        ], style={'padding': '10px', 'background': '#fff3e0', 'marginTop': '10px'})


    try:
        logger.info("Server started. Press Ctrl+C for clean shutdown.")
        # Dash will do its own cleanup before atexit executes
        app.run(debug=False, host='0.0.0.0', port=single.port, use_reloader=False)
    except KeyboardInterrupt:
        # This catches Ctrl+C but sys.exit() was already called by signal handler
        logger.info("KeyboardInterrupt caught")
    except SystemExit:
        # Normal - sys.exit() was called
        logger.info("SystemExit - shutting down normally")

    # We arrive here after shutdown
    logger.info("App terminated, atexit handlers will execute now...")