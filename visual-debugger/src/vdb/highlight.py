"""Python syntax highlighting as HTML, one string per source line."""

import builtins
import html
import io
import keyword
import token
import tokenize

BUILTIN_NAMES = frozenset(dir(builtins))
STRING_TOKENS = {token.STRING} | {
    getattr(token, n) for n in (
        "FSTRING_START", "FSTRING_MIDDLE", "FSTRING_END",
        "TSTRING_START", "TSTRING_MIDDLE", "TSTRING_END",
    ) if hasattr(token, n)
}
SKIPPED = {tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.COMMENT}


def css_class(tok, prev):
    if tok.type == tokenize.COMMENT:
        return "c"
    if tok.type in STRING_TOKENS:
        return "s"
    if tok.type == tokenize.NUMBER:
        return "n"
    if tok.type == tokenize.NAME:
        if keyword.iskeyword(tok.string):
            return "k"
        if prev is not None and prev.string in ("def", "class"):
            return "f"
        if tok.string in BUILTIN_NAMES:
            return "b"
    return None


def render_line(line, spans):
    out, cursor = [], 0
    for start, end, cls in sorted(spans):
        if start < cursor or start >= end:
            continue
        out.append(html.escape(line[cursor:start]))
        out.append(f'<span class="t{cls}">{html.escape(line[start:end])}</span>')
        cursor = end
    out.append(html.escape(line[cursor:]))
    return "".join(out)


def highlight_lines(source):
    lines = source.splitlines()
    spans = [[] for _ in lines]
    prev = None
    try:
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            cls = css_class(tok, prev)
            if tok.type not in SKIPPED:
                prev = tok
            if not cls:
                continue
            (start_row, start_col), (end_row, end_col) = tok.start, tok.end
            for row in range(start_row, min(end_row, len(lines)) + 1):
                start = start_col if row == start_row else 0
                end = end_col if row == end_row else len(lines[row - 1])
                spans[row - 1].append((start, end, cls))
    except (tokenize.TokenError, SyntaxError):
        pass
    return [render_line(line, line_spans) for line, line_spans in zip(lines, spans)]
