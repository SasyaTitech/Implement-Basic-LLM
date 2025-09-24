import time
from collections import defaultdict
from typing import Self
from rich.console import Console
from rich.table import Table


class RegionTimer:
    def __init__(self):
        self.t0 = {}
        self.total = defaultdict(float)
        self.count = defaultdict(int)

    def start(self, name):
        self.t0[name] = time.perf_counter()

    def stop(self, name):
        dt = time.perf_counter() - self.t0.pop(name)
        self.total[name] += dt
        self.count[name] += 1

    def merge(self, other: Self):
        for name, t in other.total.items():
            self.total[name] += t
            self.count[name] += other.count[name]

    def report(self, log=print):
        # If using default print, use rich table for better formatting
        if log == print:
            console = Console()
            table = Table(title="⏱️  Region Timer Report", title_style="bold magenta")

            table.add_column("Region", justify="left", style="cyan", no_wrap=True)
            table.add_column("Total Time", justify="right", style="green")
            table.add_column("Avg Time", justify="right", style="yellow")
            table.add_column("Count", justify="right", style="blue")
            table.add_column("Percentage", justify="right", style="red")

            # Calculate total time for percentage calculation
            total_time = sum(self.total.values())

            for name, tot in sorted(self.total.items(), key=lambda x: -x[1]):
                c = self.count[name]
                percentage = (tot / total_time) * 100 if total_time > 0 else 0
                table.add_row(name, f"{tot*1000:.2f} ms", f"{tot/c*1000:.2f} ms", f"{c}x", f"{percentage:.1f}%")

            console.print(table)
        else:
            # Fallback to original format for custom log functions
            for name, tot in sorted(self.total.items(), key=lambda x: -x[1]):
                c = self.count[name]
                log(f"{name}: {tot*1000:.2f} ms total | {tot/c*1000:.2f} ms avg over {c}x")


class ContextTimer:
    def __init__(self, timer: RegionTimer, name: str, effect: bool) -> None:
        self.timer = timer
        self.name = name
        self.effect = effect

    def __enter__(self):
        if self.effect:
            self.timer.start(self.name)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.effect:
            self.timer.stop(self.name)
