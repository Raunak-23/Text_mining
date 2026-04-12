"""Rich terminal helpers used across the CLI."""
from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.theme import Theme

THEME = Theme(
    {
        "info": "cyan",
        "success": "bold green",
        "warning": "yellow",
        "error": "bold red",
        "highlight": "bold magenta",
        "muted": "dim",
    }
)

console = Console(theme=THEME)


def print_header(title: str, subtitle: str = "") -> None:
    body = f"[bold]{title}[/bold]"
    if subtitle:
        body += f"\n[muted]{subtitle}[/muted]"
    console.print(Panel(body, expand=False, border_style="cyan"))


def print_success(msg: str) -> None:
    console.print(f"[success]OK[/success] {msg}")


def print_warning(msg: str) -> None:
    console.print(f"[warning]WARN[/warning] {msg}")


def print_error(msg: str) -> None:
    console.print(f"[error]ERR[/error] {msg}")


def print_info(msg: str) -> None:
    console.print(f"[info]>>[/info] {msg}")


def make_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    )
