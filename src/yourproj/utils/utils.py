# src/yourproj/utils/notebook_helpers.py
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from html import escape

from IPython.display import Code, HTML, Javascript, display


# -------------------------------------------------------------------
# Project discovery & notebook setup
# -------------------------------------------------------------------

@dataclass(frozen=True)
class Paths:
    """Convenience container for important project paths."""
    ROOT: Path
    SRC: Path


def _find_project_root(
    start: Path | None = None,
    markers: Iterable[str] = ("pyproject.toml", ".git", "README.md"),
) -> Path:
    """
    Walk upward from `start` (or CWD) until any marker is found and
    return that directory as the project root.
    """
    here = Path(start or Path.cwd()).resolve()
    while True:
        if any((here / m).exists() for m in markers):
            return here
        if here.parent == here:
            # we reached the filesystem root
            raise RuntimeError(
                f"Could not find project root from {start or Path.cwd()}. "
                f"Tried markers: {', '.join(markers)}"
            )
        here = here.parent


def setup_notebook(markers: Iterable[str] = ("pyproject.toml", ".git", "README.md")) -> Paths:
    """
    - Discover project root
    - Ensure `src` is on sys.path (at the front)
    - chdir into project root so relative paths in the notebook are stable

    Returns:
        Paths(ROOT=..., SRC=...)
    """
    root = _find_project_root(markers=markers)
    src = root / "src"

    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    if Path.cwd().resolve() != root:
        os.chdir(root)

    return Paths(ROOT=root, SRC=src)


# -------------------------------------------------------------------
# Presentation helpers (CSS, headers, code includes)
# -------------------------------------------------------------------

_CSS_INJECTED = False
_TAG_CSS_INJECTED = False


def _inject_css_once() -> None:
    """Inject minimal CSS once per kernel to style headers and details."""
    global _CSS_INJECTED
    if _CSS_INJECTED:
        return
    display(
        HTML(
            """
<style>
/* ---- Theme tokens (easy to tweak) ---- */
:root {
  --acc-strong: #2563eb;   /* blue-600 */
  --acc-medium: #3b82f6;   /* blue-500 */
  --acc-soft: #93c5fd;     /* blue-300 */
  --bg-card: #ffffff;
  --bg-subtle: #f5f7fb;    /* soft bluish */
  --text-strong: #111827;  /* gray-900 */
  --text-normal: #1f2937;  /* gray-800 */
  --text-muted: #6b7280;   /* gray-500 */
  --ring: rgba(37,99,235,.16);
}

/* --- Document Title Banner (kept) --- */
.nb-doc-title {
  background: linear-gradient(135deg, #1a73e8, #673ab7);
  color: #fff;
  padding: 30px 24px;
  border-radius: 16px;
  text-align: center;
  box-shadow: 0 8px 24px rgba(0,0,0,0.12);
  margin: 12px 0 28px 0;
}
.nb-doc-title h1 {
  margin: 0 0 10px 0;
  font-size: 2.4rem;
  line-height: 1.2;
  font-weight: 700;
}
.nb-doc-title .subtitle {
  margin: 0 auto;
  max-width: 900px;
  font-size: 1.06rem;
  opacity: .95;
}

/* --- Section (clearly larger, card with accent) --- */
.nb-section {
  background: var(--bg-card);
  border: 1px solid #e5e7eb;
  border-left: 8px solid var(--acc-strong);
  padding: 18px 20px;
  border-radius: 12px;
  margin: 26px 0 14px 0;
  box-shadow: 0 4px 18px var(--ring);
}
.nb-section h2 {
  margin: 0 0 6px 0;
  font-size: 1.55rem;
  font-weight: 650;
  color: var(--text-strong);
}
.nb-section .subtitle {
  margin: 0;
  color: var(--text-muted);
  font-size: 0.98rem;
}

/* --- Subsection (smaller, subtler background) --- */
.nb-subsection {
  background: linear-gradient(180deg, var(--bg-subtle), #fff 85%);
  border: 1px solid #e8edf7;
  border-left: 4px solid var(--acc-soft);
  padding: 12px 14px;
  border-radius: 10px;
  margin: 16px 0 10px 0;
}
.nb-subsection h3 {
  margin: 0 0 4px 0;
  font-size: 1.2rem;
  font-weight: 560;
  color: var(--text-normal);
}
.nb-subsection .subtitle {
  margin: 0;
  color: var(--text-muted);
  font-size: 0.95rem;
}

/* Existing styles (for legacy header & code blocks) */
div.custom-header {
  padding: 18px; border: 2px solid #e8eaed; border-radius: 12px; background: #fcfcff;
}
div.custom-header h1 { margin: 0; color: #1a73e8; }
div.custom-header p  { margin: 6px 0 0 0; color: #5f6368; }
details.show-src summary { cursor: pointer; font-weight: 600; }
details.show-src summary::marker { color: #1a73e8; }
pre { margin: 10px 0; }
</style>
"""
        )
    )
    _CSS_INJECTED = True


def doc_title(title: str, subtitle: str | None = None) -> None:
    _inject_css_once()
    sub_html = f'<div class="subtitle">{escape(subtitle or "")}</div>' if subtitle else ""
    html = f"""
    <div class="nb-doc-title" role="banner" aria-label="Document title">
      <h1>{escape(title)}</h1>
      {sub_html}
    </div>
    """
    display(HTML(html))


def section(title: str, subtitle: str | None = None, accent: str = "#2563eb") -> None:
    _inject_css_once()
    sub_html = f'<div class="subtitle">{escape(subtitle or "")}</div>' if subtitle else ""
    html = f"""
    <div class="nb-section" style="border-left-color:{accent}" role="region" aria-label="{escape(title)} section">
      <h2>{escape(title)}</h2>
      {sub_html}
    </div>
    """
    display(HTML(html))


def subsection(title: str, subtitle: str | None = None, accent: str = "#93c5fd") -> None:
    _inject_css_once()
    sub_html = f'<div class="subtitle">{escape(subtitle or "")}</div>' if subtitle else ""
    html = f"""
    <div class="nb-subsection" style="border-left-color:{accent}" role="region" aria-label="{escape(title)} subsection">
      <h3>{escape(title)}</h3>
      {sub_html}
    </div>
    """
    display(HTML(html))

def _resolve_source_path(paths: Paths, ref: str | Path) -> Path:
    """
    Accepts:
      - 'src/yourproj/file.py'
      - 'yourproj/file.py'
      - Path objects

    Returns an absolute path under project ROOT.
    """
    p = Path(ref)
    if not p.is_absolute():
        # allow either 'src/...' or module-like 'yourproj/...'
        if str(p).startswith("src/"):
            p = paths.ROOT / p
        else:
            p = paths.ROOT / "src" / p
    return p.resolve()


def show_code(ref: str | Path, language: str = "python") -> None:
    """
    Render a source file nicely in the output area (not in the input cell).

    Usage:
        show_code("yourproj/config.py")
        show_code("src/yourproj/pipeline/train.py")
    """
    paths = setup_notebook()  # idempotent
    file_path = _resolve_source_path(paths, ref)
    if not file_path.exists():
        raise FileNotFoundError(file_path)
    display(Code(filename=str(file_path), language=language))


def show_code_collapsible(
    ref: str | Path,
    title: str = "Source",
    language: str = "python",
    open: bool = False,
) -> None:
    """
    Render a collapsible <details> block containing the file's source.
    Great for readable notebooks and Quarto exports.
    """
    _inject_css_once()
    paths = setup_notebook()
    file_path = _resolve_source_path(paths, ref)
    if not file_path.exists():
        raise FileNotFoundError(file_path)
    code_html = escape(file_path.read_text())
    open_attr = " open" if open else ""
    html = f"""
<details class="show-src"{open_attr}>
  <summary>{escape(title)} — {escape(str(file_path.relative_to(paths.ROOT)))}</summary>
  <pre><code class="language-{escape(language)}">{code_html}</code></pre>
</details>
"""
    display(HTML(html))


# -------------------------------------------------------------------
# Live hide helpers (for JupyterLab UI only)
# -------------------------------------------------------------------

def hide_input() -> None:
    """
    Hide the input (code) of the currently active cell.

    Notes:
    - Works in JupyterLab (v3/v4) and classic Notebook by probing common DOM structures.
    - The notebook must be trusted for JavaScript to execute.
    - Run this at the END of the cell you want to hide.
    """
    js = """
    (function(){
      try {
        // Candidate selector sets for different frontends
        const candidates = [
          { root: '.jp-Notebook', active: '.jp-Cell.jp-mod-active, .jp-Cell.jp-mod-selected', input: '.jp-Cell-inputWrapper' },
          { root: 'body',         active: '.cell.selected',                                      input: '.input' }
        ];

        let active = null;
        let inputSel = null;
        for (const c of candidates) {
          const rootEl = document.querySelector(c.root);
          if (!rootEl) continue;
          const a = rootEl.querySelector(c.active);
          if (a) { active = a; inputSel = c.input; break; }
        }
        if (!active) return;
        const input = active.querySelector(inputSel);
        if (input) input.style.display = 'none';
      } catch (e) {
        console.warn('hide_input error', e);
      }
    })();
    """
    display(Javascript(js))


def hide_output() -> None:
    """
    Hide the output of the currently active cell.

    Notes:
    - Works in JupyterLab (v3/v4) and classic Notebook by probing common DOM structures.
    - The notebook must be trusted for JavaScript to execute.
    - Run this at the END of the cell you want to hide.
    """
    js = """
    (function(){
      try {
        const candidates = [
          { root: '.jp-Notebook', active: '.jp-Cell.jp-mod-active, .jp-Cell.jp-mod-selected', output: '.jp-Cell-outputWrapper' },
          { root: 'body',         active: '.cell.selected',                                      output: '.output_wrapper, .output' }
        ];

        let active = null;
        let outSel = null;
        for (const c of candidates) {
          const rootEl = document.querySelector(c.root);
          if (!rootEl) continue;
          const a = rootEl.querySelector(c.active);
          if (a) { active = a; outSel = c.output; break; }
        }
        if (!active) return;
        const output = active.querySelector(outSel);
        if (output) output.style.display = 'none';
      } catch (e) {
        console.warn('hide_output error', e);
      }
    })();
    """
    display(Javascript(js))


def enable_tag_hiding_css() -> None:
    """
    Inject CSS that hides inputs/outputs for cells tagged 'hide-input' or
    'hide-output' in live notebooks. For exports, prefer Quarto/nbconvert
    options (e.g., echo: false, or tag-based removers).
    """
    global _TAG_CSS_INJECTED
    if _TAG_CSS_INJECTED:
        return
    css = """
    <style>
      /* JupyterLab: cells reflect tags in data-tags attribute */
      .jp-Cell[data-tags*="hide-input"]  .jp-Cell-inputWrapper  { display: none !important; }
      .jp-Cell[data-tags*="hide-output"] .jp-Cell-outputWrapper { display: none !important; }

      /* Classic Notebook: tag extension adds .tag_<name> on .cell */
      .cell.tag_hide-input  .input  { display: none !important; }
      .cell.tag_hide-output .output,
      .cell.tag_hide-output .output_wrapper { display: none !important; }
    </style>
    """
    display(HTML(css))
    _TAG_CSS_INJECTED = True


__all__ = [
    "Paths",
    "setup_notebook",
    "section",
    "doc_title",
    "subsection",
    "show_code",
    "show_code_collapsible",
    "hide_input",
    "hide_output",
    "enable_tag_hiding_css",
]


# -------------------------------------------------------------------
# VS Code helper: collapse cells by tags by editing notebook JSON
# -------------------------------------------------------------------

def collapse_vscode_notebook_by_tags(
    nb_path: str | Path,
    hide_input_tags: set[str] | None = None,
    hide_output_tags: set[str] | None = None,
) -> int:
    """
    Collapse inputs/outputs in a Jupyter .ipynb for VS Code based on tags.

    VS Code does not honor our injected CSS/JS. Instead, it respects certain
    cell metadata fields. This helper modifies the notebook file on disk:

    - If a cell has any tag in `hide_input_tags`, set both:
        cell.metadata.inputCollapsed = true
        cell.metadata.jupyter.source_hidden = true
    - If a cell has any tag in `hide_output_tags`, set both:
        cell.metadata.outputCollapsed = true
        cell.metadata.jupyter.outputs_hidden = true

    Returns the number of cells modified.
    """
    import json

    hide_input_tags = hide_input_tags or {"hide-input"}
    hide_output_tags = hide_output_tags or {"hide-output"}

    p = Path(nb_path)
    data = json.loads(p.read_text())
    modified = 0

    for cell in data.get("cells", []):
        if cell.get("cell_type") not in {"code", "markdown"}:
            continue
        md = cell.setdefault("metadata", {})
        tags = set(md.get("tags", []))
        jup = md.setdefault("jupyter", {})

        did = False
        if tags & hide_input_tags:
            if md.get("inputCollapsed") is not True:
                md["inputCollapsed"] = True
                did = True
            if jup.get("source_hidden") is not True:
                jup["source_hidden"] = True
                did = True
        if tags & hide_output_tags:
            if md.get("outputCollapsed") is not True:
                md["outputCollapsed"] = True
                did = True
            if jup.get("outputs_hidden") is not True:
                jup["outputs_hidden"] = True
                did = True
        if did:
            modified += 1

    # Write back only if changes happened
    if modified:
        p.write_text(json.dumps(data, ensure_ascii=False, indent=1))

    return modified


# -------------------------------------------------------------------
# Filesystem / JSON helpers used by pipeline code
# -------------------------------------------------------------------

def ensure_dir(path: Path | str) -> None:
    """Create directory `path` if it doesn't already exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)


def save_json(obj, path: Path | str) -> None:
    """Save `obj` as pretty JSON to `path` (creating parent dirs)."""
    import json

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True))
