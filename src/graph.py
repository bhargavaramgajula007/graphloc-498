"""Build a competition-schema code graph (NetworkX node-link style) for a repo at a commit,
and map a gold patch onto the graph nodes it edits.

Nodes: module / class / function (methods + nested functions), id = dotted symbol path
       (e.g. requests.models.Response.json), with source text + file/line span.
Edges: contains (parent->child), calls, imports, inherits.
"""
import ast, re, subprocess
from collections import defaultdict

TEST_RE = re.compile(r"(^|/)(tests?|testing)/|(^|/)test_[^/]*\.py$|_test\.py$|conftest\.py$")
SKIP_PREFIX = ("docs/", "docs_src/", "scripts/", "examples/", "bench", "tools/", ".github/")


def git(clone, *a, inp=None):
    return subprocess.run(["git", "-C", clone, *a], capture_output=True, input=inp,
                          ).stdout


def list_src_files(clone, commit):
    out = git(clone, "ls-tree", "-r", "--name-only", commit).decode("utf-8", "replace")
    return [p for p in out.splitlines() if p.endswith(".py") and not TEST_RE.search(p)
            and not p.startswith(SKIP_PREFIX) and p != "setup.py" and "/." not in p]


def read_files(clone, commit, paths):
    """Fast bulk read via git cat-file --batch."""
    inp = "".join(f"{commit}:{p}\n" for p in paths).encode()
    raw = git(clone, "cat-file", "--batch", inp=inp)
    out, i = {}, 0
    for p in paths:
        nl = raw.index(b"\n", i)
        header = raw[i:nl].split()
        if header[-1] == b"missing":
            i = nl + 1
            continue
        size = int(header[2])
        out[p] = raw[nl + 1: nl + 1 + size].decode("utf-8", "replace")
        i = nl + 1 + size + 1
    return out


def module_name(path):
    p = path[:-3]
    for pre in ("src/", "lib/"):
        if p.startswith(pre):
            p = p[len(pre):]
    parts = p.split("/")
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


class _Collector(ast.NodeVisitor):
    def __init__(self, mod, path, lines, G):
        self.mod, self.path, self.lines, self.G = mod, path, lines, G
        self.stack = [mod]
        self.cls_stack = []

    def _add(self, node, kind):
        parent = self.stack[-1]
        nid = f"{parent}.{node.name}"
        start = min([d.lineno for d in node.decorator_list] + [node.lineno])
        self.G["nodes"][nid] = dict(id=nid, name=nid, kind=kind, file=self.path, start=start,
                                    end=node.end_lineno, module=self.mod,
                                    text="\n".join(self.lines[start - 1: node.end_lineno]))
        self.G["edges"].append((parent, nid, "contains"))
        return nid

    def visit_ClassDef(self, node):
        nid = self._add(node, "class")
        self.G["class_bases"][nid] = [ast.unparse(b) for b in node.bases]
        self.G["ast"][nid] = node
        self.stack.append(nid); self.cls_stack.append(nid)
        self.generic_visit(node)
        self.stack.pop(); self.cls_stack.pop()

    def visit_FunctionDef(self, node):
        nid = self._add(node, "function")
        self.G["ast"][nid] = node
        self.G["owner_class"][nid] = self.cls_stack[-1] if self.cls_stack and \
            self.stack[-1] == self.cls_stack[-1] else None
        self.stack.append(nid)
        self.generic_visit(node)
        self.stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef


def _imports(tree, mod, is_pkg):
    """alias -> fully qualified target."""
    table = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                table[a.asname or a.name.split(".")[0]] = a.name if a.asname else a.name.split(".")[0]
        elif isinstance(n, ast.ImportFrom):
            if n.level:
                base = mod.split(".") if is_pkg else mod.split(".")[:-1]
                base = base[: len(base) - (n.level - 1)] if n.level > 1 else base
                src = ".".join(base + ([n.module] if n.module else []))
            else:
                src = n.module or ""
            for a in n.names:
                table[a.asname or a.name] = f"{src}.{a.name}"
    return table


def build_graph(clone, commit):
    G = dict(nodes={}, edges=[], class_bases={}, ast={}, owner_class={})
    paths = list_src_files(clone, commit)
    files = read_files(clone, commit, paths)
    mod_info = {}
    for p, src in files.items():
        mod = module_name(p)
        lines = src.splitlines()
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        G["nodes"][mod] = dict(id=mod, name=mod, kind="module", file=p, start=1,
                               end=len(lines), module=mod, text=src[:20000])
        _Collector(mod, p, lines, G).visit(tree)
        mod_info[mod] = (tree, _imports(tree, mod, p.endswith("__init__.py")))

    nodes = G["nodes"]
    by_short = defaultdict(list)
    for nid in nodes:
        by_short[nid.rsplit(".", 1)[-1]].append(nid)

    def resolve_fq(fq):
        # follow package re-exports: a.b.X where a.b re-imports X from a.b.c
        for _ in range(4):
            if fq in nodes:
                return fq
            head, _, tail = fq.rpartition(".")
            if head in mod_info and tail in mod_info[head][1]:
                fq = mod_info[head][1][tail]
            else:
                return None
        return fq if fq in nodes else None

    def class_methods(cls, name, depth=0):
        if f"{cls}.{name}" in nodes:
            return f"{cls}.{name}"
        if depth > 4:
            return None
        for b in G["class_bases"].get(cls, []):
            bid = resolve_name(b, nodes[cls]["module"], None)
            if bid and nodes[bid]["kind"] == "class":
                r = class_methods(bid, name, depth + 1)
                if r:
                    return r
        return None

    def resolve_name(expr, mod, scope):
        head, _, rest = expr.partition(".")
        cands = []
        if scope:  # enclosing scopes (nested defs, class members)
            s = scope
            while s and s != mod:
                cands.append(f"{s}.{head}")
                s = s.rpartition(".")[0]
        cands.append(f"{mod}.{head}")
        imp = mod_info.get(mod, (None, {}))[1]
        if head in imp:
            cands.append(imp[head])
        for c in cands:
            fq = resolve_fq(c + ("." + rest if rest else "")) or (resolve_fq(c) if not rest else None)
            if fq:
                return fq
        return None

    # inherits edges
    for cls, bases in G["class_bases"].items():
        for b in bases:
            bid = resolve_name(b, nodes[cls]["module"], cls.rpartition(".")[0])
            if bid and bid != cls:
                G["edges"].append((cls, bid, "inherits"))
    # imports edges
    for mod, (_, imp) in mod_info.items():
        for alias, fq in imp.items():
            tgt = resolve_fq(fq)
            if tgt and tgt != mod:
                G["edges"].append((mod, tgt, "imports"))
    # calls edges
    for fid, fnode in list(G["ast"].items()):
        if nodes[fid]["kind"] != "function":
            continue
        mod = nodes[fid]["module"]
        owner = G["owner_class"].get(fid)
        seen = set()
        for n in ast.walk(fnode):
            if not isinstance(n, ast.Call):
                continue
            f, tgt = n.func, None
            if isinstance(f, ast.Name):
                tgt = resolve_name(f.id, mod, fid)
            elif isinstance(f, ast.Attribute):
                if isinstance(f.value, ast.Name) and f.value.id in ("self", "cls") and owner:
                    tgt = class_methods(owner, f.attr)
                elif isinstance(f.value, ast.Call) and isinstance(f.value.func, ast.Name) \
                        and f.value.func.id == "super" and owner:
                    for b in G["class_bases"].get(owner, []):
                        bid = resolve_name(b, mod, None)
                        if bid:
                            tgt = class_methods(bid, f.attr)
                            if tgt:
                                break
                else:
                    try:
                        tgt = resolve_name(ast.unparse(f), mod, fid)
                    except Exception:
                        tgt = None
                    if tgt is None and len(by_short.get(f.attr, [])) == 1 and not f.attr.startswith("__"):
                        tgt = by_short[f.attr][0]  # unique-name fallback
            if tgt and tgt != fid and tgt not in seen:
                seen.add(tgt)
                G["edges"].append((fid, tgt, "calls"))
    del G["ast"]
    return G


def to_nx(G):
    import networkx as nx
    g = nx.MultiDiGraph()
    for nid, d in G["nodes"].items():
        g.add_node(nid, **{k: v for k, v in d.items() if k != "text"})
    for s, t, ty in G["edges"]:
        g.add_edge(s, t, type=ty)
    return g


def gold_nodes(G, patch):
    """Map edited old-side lines of a source patch to the innermost enclosing graph node."""
    from unidiff import PatchSet
    by_file = defaultdict(list)
    for nid, d in G["nodes"].items():
        by_file[d["file"]].append(d)
    gold, gold_files = set(), set()
    for pf in PatchSet(patch):
        path = pf.source_file[2:] if pf.source_file.startswith("a/") else pf.source_file
        if pf.is_added_file:
            continue
        gold_files.add(path)
        touched = set()
        for h in pf:
            prev_old = h.source_start
            for line in h:
                if line.is_removed:
                    touched.add(line.source_line_no)
                elif line.is_added:
                    touched.add(prev_old)  # insertion point
                if line.source_line_no:
                    prev_old = line.source_line_no
        cands = by_file.get(path, [])
        for ln in touched:
            inner = [d for d in cands if d["start"] <= ln <= d["end"] and d["kind"] != "module"]
            if inner:
                gold.add(min(inner, key=lambda d: d["end"] - d["start"])["id"])
    return sorted(gold), sorted(gold_files)
