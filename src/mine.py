"""Mine issue-linked bug-fix tasks from a git repo.

A task = (base_commit, problem_statement from a *linked GitHub issue*, patch on non-test .py
files, test_patch on test files). Mirrors the SWE-bench / competition curation recipe, but
keeps only tasks whose prompt is an issue written *before* the fix (no PR-description leakage).

usage: python src/mine.py <owner/repo> <local_clone_dir> <out.jsonl> [max_tasks]
"""
import json, os, re, subprocess, sys, time

OWNER_REPO, CLONE, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
MAX = int(sys.argv[4]) if len(sys.argv) > 4 else 80
OWNER, NAME = OWNER_REPO.split("/")


def git(*a):
    return subprocess.run(["git", "-C", CLONE, *a], capture_output=True, text=True,
                          encoding="utf-8", errors="replace").stdout


def gh_graphql(query, **vars):
    args = ["gh", "api", "graphql", "-f", f"query={query}"]
    for k, v in vars.items():
        args += ["-F", f"{k}={v}"]
    for attempt in range(4):
        r = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode == 0:
            return json.loads(r.stdout)
        if "rate limit" in r.stderr.lower():
            time.sleep(60)
        else:
            time.sleep(2)
    return None


Q_PR = """query($o:String!,$n:String!,$num:Int!){repository(owner:$o,name:$n){
 pullRequest(number:$num){title body createdAt
  closingIssuesReferences(first:5){nodes{number title body createdAt}}}}}"""
Q_ISSUE = """query($o:String!,$n:String!,$num:Int!){repository(owner:$o,name:$n){
 issue(number:$num){number title body createdAt}}}"""


def is_test(path):
    p = path.lower()
    return p.startswith(("tests/", "test/", "testing/")) or "/tests/" in p or \
        p.split("/")[-1].startswith("test_") or p.endswith("_test.py") or p.endswith("conftest.py")


def is_src_py(path):
    return path.endswith(".py") and not is_test(path) and not path.startswith(
        ("docs/", "docs_src/", "scripts/", "examples/", "bench", "tools/"))


def split_patch(diff):
    """Split a unified diff into per-file chunks -> (src_patch, test_patch, src_files, test_files)."""
    chunks = re.split(r"(?m)^(?=diff --git )", diff)
    src, tst, sf, tf = [], [], [], []
    for c in chunks:
        m = re.match(r"diff --git a/(\S+) b/(\S+)", c)
        if not m:
            continue
        path = m.group(2)
        if is_src_py(path):
            src.append(c); sf.append(path)
        elif is_test(path) and path.endswith(".py"):
            tst.append(c); tf.append(path)
    return "".join(src), "".join(tst), sf, tf


def main():
    branch = git("rev-parse", "--abbrev-ref", "origin/HEAD").strip() or "HEAD"
    log = git("log", branch, "--first-parent", "--format=%H%x1f%P%x1f%s%x1f%cI")
    tasks, seen_issues = [], set()
    n_cand = 0
    for line in log.splitlines():
        if len(tasks) >= MAX:
            break
        h, parents, subj, date = line.split("\x1f")
        parents = parents.split()
        m = re.search(r"\(#(\d+)\)\s*$", subj) or re.search(r"Merge pull request #(\d+)", subj)
        if not m or date < os.environ.get("MIN_DATE", "2018"):
            continue
        pr = int(m.group(1))
        base = parents[0]
        diff = git("diff", base, h, "--", "*.py")
        src, tst, sf, tf = split_patch(diff)
        if not sf or not tf or len(sf) > 3:
            continue
        nchg = sum(1 for l in src.splitlines() if l[:1] in "+-" and l[:3] not in ("+++", "---"))
        if nchg < 2 or nchg > 120:
            continue
        n_cand += 1
        d = gh_graphql(Q_PR, o=OWNER, n=NAME, num=pr)
        try:
            prd = d["data"]["repository"]["pullRequest"]
        except (TypeError, KeyError):
            continue
        if prd is None:
            continue
        issues = prd["closingIssuesReferences"]["nodes"]
        if not issues:  # fallback: "fixes #N" / "closes #N" style refs in PR body
            refs = re.findall(r"(?i)(?:fix(?:es|ed)?|close[sd]?|resolve[sd]?)\s*:?\s*#(\d+)",
                              (prd.get("body") or "") + " " + subj)
            for r_ in refs[:2]:
                di = gh_graphql(Q_ISSUE, o=OWNER, n=NAME, num=int(r_))
                try:
                    iss = di["data"]["repository"]["issue"]
                except (TypeError, KeyError):
                    iss = None
                if iss:
                    issues.append(iss)
        issues = [i for i in issues if i and i["createdAt"] <= prd["createdAt"]]
        if not issues or issues[0]["number"] in seen_issues:
            continue
        iss = issues[0]
        body = (iss.get("body") or "").strip()
        if len(body) < 40:
            continue
        seen_issues.add(iss["number"])
        tasks.append(dict(
            instance_id=f"{NAME}_{pr}", repo=OWNER_REPO, base_commit=base, fix_commit=h,
            pr_number=pr, issue_number=iss["number"],
            problem_statement=f"{iss['title']}\n\n{body}",
            patch=src, test_patch=tst, src_files=sf, test_files=tf,
            n_changed_lines=nchg, created_at=iss["createdAt"]))
        print(f"[{NAME}] {len(tasks)}/{n_cand} pr#{pr} issue#{iss['number']} files={sf}", flush=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")
    print(f"[{NAME}] DONE {len(tasks)} tasks from {n_cand} candidates", flush=True)


if __name__ == "__main__":
    main()
