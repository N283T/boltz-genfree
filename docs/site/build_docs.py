# Copyright (c) 2024 Jeremy Wohlwend, Gabriele Corso, Saro Passaro (see LICENSE)
# /// script
# requires-python = ">=3.12"
# dependencies = ["markdown-it-py"]
# ///
"""Build the Docs section of the project website from the repository's Markdown.

    uv run docs/site/build_docs.py <outdir>          # write the pages into <outdir>
    uv run docs/site/build_docs.py --check <outdir>  # verify links of built pages

Pages are served at /docs/<page>.html; shared site files live one level up
(../site.css, ../logo-*.svg, ../index.html, ../how-it-works.html, ../results.html,
../webgpu/web/screen.html).
The output is deterministic: no timestamps, stable slugs, sorted iteration.
"""

from __future__ import annotations

import html
import re
import shutil
import sys
from html.parser import HTMLParser
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import unquote, urldefrag, urlparse

from markdown_it import MarkdownIt

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from markdown_it.token import Token
    from markdown_it.utils import EnvType, OptionsDict

REPO = Path(__file__).resolve().parents[2]
GITHUB = "https://github.com/N283T/boltz-genfree"
# Published site; Markdown links to it become relative links on the built pages.
SITE_URL = "https://n283t.github.io/boltz-genfree/"
BRANCH = "main"
H2_LEVEL = 2
CHECK_ARGC = 3
BUILD_ARGC = 2

# (output file, source file, sidebar label, page heading override,
#  take the first paragraph as lede)
PAGES = [
    ("index.html", "docs/guide/getting-started.md", "Getting started", None, True),
    ("configuration.html", "docs/guide/configuration.md", "Configuration", None, True),
    ("browser-tool.html", "docs/guide/browser-tool.md", "Browser tool", None, True),
]
SIDEBAR = [
    ("Start", ["index.html"]),
    ("Guide", ["configuration.html"]),
    ("Browser", ["browser-tool.html"]),
]
# 24px line icons (2px square-capped stroke), one per page
ICONS = {
    "index.html": '<path d="M5 3l14 9-14 9z"/>',
    "configuration.html": (
        '<path d="M4 6h10M18 6h2M4 12h4M12 12h8M4 18h12M20 18h0"/>'
        '<path d="M14 4v4M8 10v4M16 16v4"/>'
    ),
    "browser-tool.html": (
        '<path d="M3 4h18v16H3zM3 9h18"/><path d="M10 12.5v5l4.5-2.5z"/>'
    ),
}
# Pages of the deployed site that docs pages may link to (relative to the site root).
SITE_FILES = {
    "site.css",
    "logo-on-dark.svg",
    "logo-on-light.svg",
    "index.html",
    "how-it-works.html",
    "results.html",
    "webgpu/web/screen.html",
}
TOOL_PATH = "webgpu/web/screen.html"
TOOL_HREF = f"../{TOOL_PATH}"
SOURCE_TO_PAGE = {src: out for out, src, *_ in PAGES}

COPY_SCRIPT = """<script>
document.querySelectorAll('.code .copy').forEach(function (b) {
  b.addEventListener('click', function () {
    var t = b.closest('.code').querySelector('pre').innerText;
    (navigator.clipboard ? navigator.clipboard.writeText(t) : Promise.reject())\
.then(function () {
      b.textContent = 'Copied'; \
setTimeout(function () { b.textContent = 'Copy'; }, 1500);
    }, function () {});
  });
});
if (matchMedia('(max-width:759px)').matches) \
document.querySelector('.dside details').removeAttribute('open');
</script>"""

warnings: list[str] = []


def esc(s: str) -> str:
    """Escape text for HTML (quotes included)."""
    return html.escape(s, quote=True)


# ---------------------------------------------------------------------- links


def repo_url(rel: str) -> str:
    """Return the GitHub blob/tree URL of a repository path."""
    path = REPO / rel
    kind = "tree" if path.is_dir() else "blob"
    if not path.exists():
        warnings.append(f"link target does not exist in the repository: {rel}")
    return f"{GITHUB}/{kind}/{BRANCH}/{rel.rstrip('/')}"


def resolve_repo_path(href_path: str, source: str) -> str:
    """Resolve a link path found in `source` to a repository-relative path."""
    base = (REPO / source).parent
    target = (base / unquote(href_path)).resolve()
    if not target.exists() and (REPO / unquote(href_path)).exists():
        # repo-root-relative link written inside docs/
        target = (REPO / unquote(href_path)).resolve()
    try:
        return target.relative_to(REPO).as_posix()
    except ValueError:
        warnings.append(f"{source}: link escapes the repository: {href_path}")
        return href_path


def rewrite_link(href: str, source: str) -> str:
    """Map a Markdown link target to a generated page, GitHub URL or itself."""
    if href.startswith(SITE_URL):
        path, _, frag = href[len(SITE_URL) :].partition("#")
        if path not in SITE_FILES:
            warnings.append(f"{source}: link to a page the site does not have: {href}")
            return href
        return f"../{path}" + (f"#{frag}" if frag else "")
    parsed = urlparse(href)
    if parsed.scheme or parsed.netloc or href.startswith(("#", "mailto:")):
        return href
    rel = resolve_repo_path(parsed.path, source)
    frag = f"#{parsed.fragment}" if parsed.fragment else ""
    if rel in SOURCE_TO_PAGE:
        return SOURCE_TO_PAGE[rel] + frag
    if rel == TOOL_PATH:
        return TOOL_HREF + frag
    return repo_url(rel) + frag


def rewrite_image(src: str, source: str, images: dict[str, Path]) -> str:
    """Register a referenced image for copying and return its new src."""
    parsed = urlparse(src)
    if parsed.scheme or parsed.netloc:
        return src
    rel = resolve_repo_path(parsed.path, source)
    path = REPO / rel
    if not path.is_file():
        warnings.append(f"{source}: image not found: {rel}")
        return src
    if images.get(path.name, path) != path:
        warnings.append(f"image name collision: {path.name}")
    images[path.name] = path
    return f"images/{path.name}"


# ---------------------------------------------------------------------- rendering


def _static_rule(markup: str) -> Callable[..., str]:
    """Return a renderer rule that always emits `markup`."""

    def rule(*_args: object) -> str:
        return markup

    return rule


def slugify(text: str) -> str:
    """Return a GitHub-like heading slug."""
    s = re.sub(r"[^\w\s-]", "", text.lower(), flags=re.ASCII)
    s = re.sub(r"[\s_]+", "-", s.strip()).strip("-")
    return s or "section"


def inline_text(tok: Token) -> str:
    """Return the plain text of an inline token."""
    out = []
    for ch in tok.children or []:
        if ch.type in ("text", "code_inline"):
            out.append(ch.content)
        elif ch.type in ("softbreak", "hardbreak"):
            out.append(" ")
    return "".join(out)


COMMENT_RE = re.compile(r"(^|\s)(#.*)$")


def highlight(code: str, lang: str) -> str:
    """Escape the code; mark trailing `#` comments in shell-like blocks."""
    shellish = lang in ("bash", "sh", "shell", "zsh", "yaml", "yml", "console")
    lines = []
    for line in code.split("\n"):
        m = COMMENT_RE.search(line) if shellish else None
        if m:
            head = line[: m.start(2)]
            lines.append(esc(head) + f'<span class="c">{esc(m.group(2))}</span>')
        else:
            lines.append(esc(line))
    return "\n".join(lines)


class Renderer:
    """Turns one Markdown source into the article HTML plus the outline."""

    def __init__(self, source: str, images: dict[str, Path]) -> None:
        """Set up a Markdown parser whose output rules emit the docs markup."""
        self.source = source
        self.images = images
        self.md = MarkdownIt("commonmark", {"html": False}).enable("table")
        self.md.renderer.rules["fence"] = self.fence
        self.md.renderer.rules["code_block"] = self.fence
        self.md.renderer.rules["table_open"] = _static_rule(
            '<div class="tscroll"><table>\n'
        )
        self.md.renderer.rules["table_close"] = _static_rule("</table></div>\n")
        self.md.renderer.rules["heading_open"] = self.heading_open
        self.md.renderer.rules["heading_close"] = self.heading_close
        self.md.renderer.rules["blockquote_open"] = self.blockquote_open
        self.md.renderer.rules["blockquote_close"] = _static_rule("</div></div>\n")
        self.md.renderer.rules["image"] = self.image
        self.md.renderer.rules["link_open"] = self.link_open
        self.outline: list[tuple[int, str, str, str]] = []  # (level, id, text, number)
        self.title = ""
        self.lede_tokens: list[Token] = []

    # block rules
    def fence(
        self,
        tokens: Sequence[Token],
        idx: int,
        _options: OptionsDict,
        _env: EnvType,
    ) -> str:
        """Render a fenced code block with a label bar."""
        tok = tokens[idx]
        lang = (tok.info.strip().split() or [""])[0].lower()
        label = lang or "text"
        file_label = (
            '<span class="file">terminal</span>'
            if lang in ("bash", "sh", "shell", "console")
            else ""
        )
        return (
            f'<div class="code">\n<div class="ch">'
            f'<span class="lang">{esc(label)}</span>'
            f"{file_label}"
            '<button class="copy" type="button">Copy</button></div>\n'
            f"<pre><code>{highlight(tok.content.rstrip(chr(10)), lang)}</code></pre>"
            "\n</div>\n"
        )

    def heading_open(
        self,
        tokens: Sequence[Token],
        idx: int,
        _options: OptionsDict,
        _env: EnvType,
    ) -> str:
        """Open a heading with its id and section number."""
        tok = tokens[idx]
        hid, number = tok.meta["id"], tok.meta.get("number", "")
        hn = f'<span class="hn">{number}</span>' if number else ""
        return f'<{tok.tag} id="{esc(hid)}">{hn}<span class="ht">'

    def heading_close(
        self,
        tokens: Sequence[Token],
        idx: int,
        _options: OptionsDict,
        _env: EnvType,
    ) -> str:
        """Close a heading and add its anchor link."""
        tok = tokens[idx]
        hid = tokens[idx - 2].meta["id"]
        return (
            f'</span><a class="anchor" href="#{esc(hid)}" '
            'aria-label="Link to this section">#</a>'
            f"</{tok.tag}>\n"
        )

    def blockquote_open(
        self,
        _tokens: Sequence[Token],
        _idx: int,
        _options: OptionsDict,
        _env: EnvType,
    ) -> str:
        """Render a blockquote as a callout."""
        return '<div class="call warn"><span class="ct">Note</span><div class="cb">'

    # inline rules
    def image(
        self,
        tokens: Sequence[Token],
        idx: int,
        options: OptionsDict,
        env: EnvType,
    ) -> str:
        """Render an image, copying its file and rewriting src."""
        tok = tokens[idx]
        src = rewrite_image(tok.attrGet("src") or "", self.source, self.images)
        alt = self.md.renderer.renderInlineAsText(tok.children, options, env)
        return f'<img src="{esc(src)}" alt="{esc(alt)}" loading="lazy">'

    def link_open(
        self,
        tokens: Sequence[Token],
        idx: int,
        _options: OptionsDict,
        _env: EnvType,
    ) -> str:
        """Open a link, rewriting repository-relative targets."""
        tok = tokens[idx]
        href = rewrite_link(tok.attrGet("href") or "", self.source)
        return f'<a href="{esc(href)}">'

    # driver
    def render(self, text: str, take_lede: bool) -> tuple[str, str, str]:
        """Return (body HTML, lede HTML, title) and fill `self.outline`."""
        tokens = self.md.parse(text)
        # Drop the first h1 (becomes the page title) and optionally take the
        # first paragraph as lede.
        out: list[Token] = []
        i = 0
        lede = ""
        while i < len(tokens):
            t = tokens[i]
            if t.type == "heading_open" and t.tag == "h1" and not self.title:
                self.title = inline_text(tokens[i + 1])
                i += 3
                if take_lede and tokens[i].type == "paragraph_open":
                    lede = self.md.renderer.renderInline(
                        tokens[i + 1].children, self.md.options, {}
                    )
                    i += 3
                continue
            out.append(t)
            i += 1
        used: set[str] = set()
        for k, t in enumerate(out):
            if t.type != "heading_open":
                continue
            text_ = inline_text(out[k + 1])
            base = slugify(text_)
            hid, c = base, 1
            while hid in used:
                c += 1
                hid = f"{base}-{c}"
            used.add(hid)
            number = ""
            t.meta = {"id": hid, "number": number}
            if t.tag in ("h2", "h3"):
                self.outline.append((int(t.tag[1]), hid, text_, number))
        body = self.md.renderer.render(out, self.md.options, {})
        return body, lede, self.title


# ---------------------------------------------------------------------- page chrome


EXTRA_CSS = (
    "<style>"
    ".darticle h2 .ht,.darticle h3 .ht{min-width:0;overflow-wrap:anywhere}"
    ".dtoc li.sub a{padding-left:26px}"
    ".darticle img{margin-top:20px;border:1px solid var(--line-2)}"
    ".darticle a,.darticle p code,.darticle li code{overflow-wrap:anywhere}"
    ".darticle td:first-child{white-space:normal}"
    "</style>"
)


def chrome_head(title: str, description: str) -> str:
    """Return the document head and the shared site bar."""
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)} &middot; Boltz-GenFree Docs</title>
<meta name="description" content="{esc(description)}">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wdth,wght@12..96,75..100,200..800&family=Atkinson+Hyperlegible:ital,wght@0,400;0,700;1,400;1,700&family=Red+Hat+Mono:wght@400..700&display=swap">
<link rel="stylesheet" href="../site.css">
{EXTRA_CSS}
</head>
<body>
<a class="skip" href="#main">Skip to content</a>

<header class="bar">
  <div class="wrap">
    <a class="brand" href="../index.html">\
<img src="../logo-on-dark.svg" alt="Boltz-GenFree home"></a>
    <nav class="site-nav" aria-label="Site">
      <ol>
        <li><a href="../index.html"><span class="nn">01</span>Home</a></li>
        <li><a href="index.html" aria-current="page">\
<span class="nn">02</span>Docs</a></li>
        <li><a href="../how-it-works.html">\
<span class="nn">03</span>How it works</a></li>
        <li><a href="../results.html"><span class="nn">04</span>Results</a></li>
        <li><a class="wgpu" href="../webgpu/web/screen.html">\
<span class="nn">05</span>WebGPU</a></li>
        <li class="gh-li"><a class="gh" href="{GITHUB}">GitHub</a></li>
      </ol>
    </nav>
  </div>
</header>
"""


def chrome_foot() -> str:
    """Return the shared footer and the page script."""
    return f"""
<footer class="foot">
  <div class="rule"></div>
  <div class="wrap">
    <div class="cols">
      <div class="fbrand">
        <img src="../logo-on-light.svg" alt="Boltz-GenFree">
        <p>Structure-generation-free ranking for first-stage affinity screening.</p>
      </div>
      <div>
        <h3>Site</h3>
        <ul>
          <li><a href="../index.html">Home</a></li>
          <li><a href="index.html" aria-current="page">Docs</a></li>
          <li><a href="../how-it-works.html">How it works</a></li>
          <li><a href="../results.html">Results</a></li>
          <li><a href="../webgpu/web/screen.html">WebGPU</a></li>
        </ul>
      </div>
      <div>
        <h3>Project</h3>
        <ul>
          <li><a href="{GITHUB}">Source code</a></li>
          <li><a href="{GITHUB}/releases">Releases</a></li>
          <li><a href="{GITHUB}/issues">Issues</a></li>
        </ul>
      </div>
      <div>
        <h3>Cite</h3>
        <ul>
          <li><a href="https://doi.org/10.26434/chemrxiv.15012606/v1">Preprint</a></li>
          <li><a href="../results.html#cite">How to cite</a></li>
          <li><a href="https://doi.org/10.5281/zenodo.23949585">Data on Zenodo</a></li>
        </ul>
      </div>
    </div>
  </div>
  <div class="legal"><span>MIT License &middot; Boltz-2 weights under their own \
terms</span><span>Not affiliated with the Boltz authors</span></div>
</footer>
{COPY_SCRIPT}
</body>
</html>
"""


def sidebar(current: str) -> str:
    """Return the docs menu with the current page marked."""
    labels = {out: label for out, _s, label, _h, _l in PAGES}
    groups = []
    for name, pages in SIDEBAR:
        items = []
        for p in pages:
            cur = ' aria-current="page"' if p == current else ""
            icon = (
                '<svg class="ic" viewBox="0 0 24 24" aria-hidden="true">'
                f"{ICONS[p]}</svg>"
            )
            items.append(
                f'          <li><a href="{p}"{cur}>{icon}{esc(labels[p])}</a></li>'
            )
        groups.append(
            f'      <div class="dgroup">\n        <h4>{esc(name)}</h4>\n'
            "        <ul>\n" + "\n".join(items) + "\n        </ul>\n      </div>"
        )
    return (
        '<aside class="dside" aria-label="Documentation">\n    <details open>\n'
        "      <summary>Docs menu</summary>\n"
        + "\n".join(groups)
        + "\n    </details>\n  </aside>"
    )


def toc(outline: list[tuple[int, str, str, str]]) -> str:
    """Return the "On this page" list (section headings only)."""
    items = [
        f'      <li><a href="#{esc(hid)}">{esc(text)}</a></li>'
        for level, hid, text, _number in outline
        if level == H2_LEVEL
    ]
    return (
        '<aside class="dtoc" aria-label="On this page">\n'
        '    <span class="lab">On this page</span>\n'
        "    <ol>\n" + "\n".join(items) + "\n    </ol>\n  </aside>"
    )


def pager(idx: int) -> str:
    """Return the previous/next navigation."""
    prev = (
        f'<a class="pv" href="{PAGES[idx - 1][0]}"><span class="lab">Previous</span>'
        f"<b>{esc(PAGES[idx - 1][2])}</b></a>"
        if idx > 0
        else "<span></span>"
    )
    nxt = (
        f'<a class="nx" href="{PAGES[idx + 1][0]}"><span class="lab">Next</span>'
        f"<b>{esc(PAGES[idx + 1][2])}</b></a>"
        if idx < len(PAGES) - 1
        else ""
    )
    return (
        '<nav class="grid pn" aria-label="Previous and next">\n'
        f"      {prev}\n      {nxt}\n    </nav>"
    )


def plain(s: str) -> str:
    """Strip tags from an HTML fragment."""
    return re.sub(r"<[^>]+>", "", html.unescape(s))


def build(outdir: Path) -> None:
    """Write every page and its images into `outdir`."""
    outdir.mkdir(parents=True, exist_ok=True)
    images: dict[str, Path] = {}
    for idx, (out, src, label, h1, want_lede) in enumerate(PAGES):
        r = Renderer(src, images)
        body, lede, title = r.render(
            (REPO / src).read_text(encoding="utf-8"), want_lede
        )
        heading = h1 or title
        desc = plain(lede) if lede else f"{heading}: Boltz-GenFree documentation."
        edit = f"{GITHUB}/edit/{BRANCH}/{src}"
        lede_html = f"\n      <p>{lede}</p>" if lede else ""
        head = (
            '<header class="pagehead dhead">\n    <div class="wrap">\n'
            f'      <p class="crumb">Docs <b>/</b> {esc(label)}</p>\n'
            f"      <h1>{esc(heading)}</h1>{lede_html}\n"
            "    </div>\n  </header>"
        )
        page = (
            chrome_head(heading, desc)
            + "\n"
            + '<main id="main">\n'
            + head
            + '\n<div class="docs">\n  '
            + sidebar(out)
            + '\n\n  <article class="darticle">\n'
            + "\n".join(
                "    " + ln if ln else ln for ln in body.rstrip("\n").split("\n")
            )
            + "\n\n    "
            + pager(idx)
            + '\n    <p class="edit">Found a mistake? '
            + f'<a href="{edit}">Edit this page on GitHub</a></p>\n  </article>\n\n  '
            + toc(r.outline)
            + "\n</div>\n</main>\n"
            + chrome_foot()
        )
        (outdir / out).write_text(page, encoding="utf-8")
    img_dir = outdir / "images"
    img_dir.mkdir(exist_ok=True)
    for name in sorted(images):
        shutil.copyfile(images[name], img_dir / name)
    for w in sorted(set(warnings)):
        sys.stderr.write(f"warning: {w}\n")
    sys.stdout.write(f"wrote {len(PAGES)} pages and {len(images)} images to {outdir}\n")


# ---------------------------------------------------------------------- link checker


class RefCollector(HTMLParser):
    """Collect href/src references and element ids of one page."""

    def __init__(self) -> None:
        """Start with no references and no ids."""
        super().__init__()
        self.refs: list[str] = []
        self.ids: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Record ids and link targets."""
        del tag  # every element is inspected
        a = dict(attrs)
        if element_id := a.get("id"):
            self.ids.add(element_id)
        for key in ("href", "src"):
            if target := a.get(key):
                self.refs.append(target)


def check(outdir: Path) -> int:
    """Verify relative references of built pages; return the exit status."""
    parsed: dict[str, RefCollector] = {}
    for f in sorted(outdir.glob("*.html")):
        c = RefCollector()
        c.feed(f.read_text(encoding="utf-8"))
        parsed[f.name] = c
    bad = 0
    total = 0
    for name, c in parsed.items():
        for ref in c.refs:
            if urlparse(ref).scheme:
                continue
            total += 1
            path, frag = urldefrag(ref)
            if not path:
                ok = frag in c.ids
            elif path.startswith("../"):
                ok = path[3:] in SITE_FILES
            else:
                ok = (outdir / path).is_file()
                if ok and frag and path in parsed:
                    ok = frag in parsed[path].ids
            if not ok:
                bad += 1
                sys.stdout.write(f"BROKEN {name}: {ref}\n")
    sys.stdout.write(
        f"checked {total} relative references in {len(parsed)} pages: {bad} broken\n"
    )
    return 1 if bad else 0


def main(argv: list[str]) -> int:
    """Command-line entry point."""
    if len(argv) == CHECK_ARGC and argv[1] == "--check":
        return check(Path(argv[2]))
    if len(argv) != BUILD_ARGC or argv[1].startswith("-"):
        sys.stderr.write(f"{__doc__}\n")
        return 2
    build(Path(argv[1]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
