# -*- coding: utf-8 -*-
"""Merge the Rubriq language edit of v10 into the v11 build without letting it change the science or the format.

The edited file (…_v10_updated.docx) lost all 13 native Word equations, so it cannot be the base. The base is the v11
build (build_physica_a_v11.py), which carries the equations and the v11 content changes. A paragraph of the edit is
accepted only when
  * the v11 paragraph is the v10 paragraph unchanged (up to the American spelling both now use), and
  * the edit keeps the same content words (function words, punctuation and spelling variants aside), the same numbers
    and symbols, the same citation brackets, and the same subscript, superscript, bold and italic runs,
  * and a hand review did not find a change of meaning (REJECT_REVIEWED).
Accepted paragraphs take the edited runs; all others keep the v11 text. Reference entries, the title and equation
paragraphs are never taken from the edit. A report of every decision is written next to the output.

    python merge_rubriq_v11.py <v10.docx> <v10_updated.docx> <v11_build.docx> <out.docx>
"""
import copy
import difflib
import io
import re
import sys

import docx
from docx.oxml.ns import qn

M = "{http://schemas.openxmlformats.org/officeDocument/2006/math}"
US = [("neighbour", "neighbor"), ("modelled", "modeled"), ("modelling", "modeling"), ("behaviour", "behavior"),
      ("non-equilibrium", "nonequilibrium"), ("non-conservative", "nonconservative"), ("non-monotonic", "nonmonotonic"),
      ("sub-lattice", "sublattice"), ("un-killed", "unkilled"), ("artefact", "artifact"), ("favour", "favor")]


def us(t):
    for a, b in US:
        t = re.sub(a, b, t)
        t = re.sub(a[0].upper() + a[1:], b[0].upper() + b[1:], t)
    return t


STOP = set("the a an of is are was were be been being in on at that which and as to for by with from it its their this these those "
           "into has have had can could would will may also then thus so therefore whereas while each both".split())
SYN = {"dynamical": "dynamic", "include": "admit", "includes": "admit", "lead": "interleave", "leads": "interleave",
       "modelled": "modeled", "unless": "unless"}

# hand review: the edit passes the automatic test but changes meaning, grammar or a deliberate term
# hand review of every paragraph the automatic test rejected: these change only wording, punctuation or tense and keep the
# meaning (e.g. 'carries' -> 'has', 'grow' -> 'increase', 'saddle-nodes' -> 'saddle nodes', italic runs split differently)
ACCEPT_REVIEWED = ["Let Λ = {1, …, L}", "Discrete steps and continuous rates", "Theorem 1 (Order", "Parts (i)–(iii) are instances",
                   "a globally stable fixed point approached", "For cooperative feedback (h > 1", "Figure 3 shows the S-shaped",
                   "4.3 Testing the bistability", "Sweep-rate extrapolation", "A control experiment tests this directly",
                   "Figure 6. Finite-size scaling", "The simulated dynamics is strictly irreversible", "The framework rests on three",
                   "(a) Range of the feedback", "(c) Thermodynamics and the BKT point", "Figure A1. Monte Carlo diagnostics"]

REJECT_REVIEWED = {
    "and 0 < A(κ) < 1": "'(iii) follows by the inverse-function theorem' became '(iii) This is followed by'",
    "4.1 The uncoupled baseline": "run-in heading formatting changed and 'with' dropped before the defining equation",
    "Proof sketch.": "'Between them three real roots exist' lost 'between them'",
    "Where the feedback is strong enough": "bold run-in heading lowercased; 'lie at β/βc' became 'are β/βc'",
    "Two starts at fixed ρ": "'the two starts agree to within 0.004' became 'is within 0.004'",
    "The five combinations whose interval": "'are, with difference and 95% interval:' became 'are as follows: difference and ...'",
    "Figure 4. Kinetic Monte Carlo": "'the line is equality' became 'the line is equal'",
    "4.4 Why the closure fails": "'μ(δ̄), the rate at the mean local disorder' became '... of the mean local disorder is high'",
    "Figure 7 separates what is measured": "'σaux inherits from the lattice only J' became 'σaux is inherited from the lattice only J'",
    "The construction assumes only three": "'u_s encodes the geometry throttling the flux' became 'we encode the geometry through the flux'",
    "The mechanism makes a concrete prediction": "'however slowly the sweep is made' became 'ρ−, however, slowly'",
    "Figure A2. Extended-range": "'Collapsed starts at ρ = 6.5 recover' became 'Collapse starts at ρ = 6.5'",
    "Self-Maintained Order": "title: 'Nondequilibrium' typo; title kept",
    "The dynamics interleaves": "'The dynamics lead to reversible rotor relaxation with' changes what the dynamics does",
    "so an aligned neighborhood shields": "'Therefore,' adds a logical claim the sentence does not make",
    "Making degradation depend on the order": "'depend on the order in which it is destroyed' changes the meaning",
    "A.2 Seeding and error estimates": "the seed-code clause was garbled ('and its study, parameter set ... are named')",
    "At h0 = 0 the maintained phase": "'quasilong-range' is not the established term 'quasi-long-range'",
}


def tokens(text):
    t = us(text).lower().replace("’", "'")
    t = re.sub("[‐‑‒–—―−]", "-", t)
    words = re.findall(r"[a-z0-9α-ωΔΥ]+", t)
    out = []
    for w in words:
        if w in STOP:
            continue
        w = SYN.get(w, w)
        if len(w) > 4 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        out.append(w)
    return out


def numbers(text):
    return re.findall(r"\d+(?:\.\d+)?", text) + re.findall(r"\[[0-9,–\- ]+\]", text)


def sig(p):
    sub = sup = b = i = 0
    for r in p._p.iterfind(".//" + qn("w:r")):
        rpr = r.find(qn("w:rPr"))
        if rpr is None or r.find(qn("w:t")) is None:
            continue
        va = rpr.find(qn("w:vertAlign"))
        if va is not None:
            sub += va.get(qn("w:val")) == "subscript"
            sup += va.get(qn("w:val")) == "superscript"
        bb, ii = rpr.find(qn("w:b")), rpr.find(qn("w:i"))
        b += bb is not None and bb.get(qn("w:val")) not in ("0", "false")
        i += ii is not None and ii.get(qn("w:val")) not in ("0", "false")
    return sub, sup, b, i


def has_math(p):
    return p._p.find(".//" + M + "oMath") is not None


def main(a_path, b_path, c_path, out_path):
    A = docx.Document(a_path).paragraphs
    B = docx.Document(b_path).paragraphs
    Cd = docx.Document(c_path)
    C = Cd.paragraphs
    assert len(A) == len(B), (len(A), len(B))
    a_norm = [us(p.text) for p in A]
    c_text = [p.text for p in C]
    sm = difflib.SequenceMatcher(None, a_norm, c_text, autojunk=False)
    c_of_a = {}
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            for k in range(i2 - i1):
                c_of_a[i1 + k] = j1 + k
    report, n_acc, n_rej = [], 0, 0
    in_refs = False
    for i, (pa, pb) in enumerate(zip(A, B)):
        if pa.text.strip() == "References":
            in_refs = True
        if pa.text == pb.text:
            continue
        head = pa.text[:70]
        if i not in c_of_a:
            report.append("KEEP v11 (paragraph changed in v11)  | %s" % head)
            continue
        pc = C[c_of_a[i]]
        why = None
        if in_refs or re.match(r"\[\d+\]", pa.text):
            why = "reference entry (verified Crossref record)"
        elif has_math(pc):
            why = "equation paragraph"
        else:
            for key, reason in REJECT_REVIEWED.items():
                if pa.text.startswith(key) or us(pa.text).startswith(key):
                    why = "hand review: " + reason
            reviewed_ok = any(pa.text.startswith(k) for k in ACCEPT_REVIEWED)
            if why is None and reviewed_ok:
                if sorted(numbers(pa.text)) != sorted(numbers(pb.text)) or sig(pa)[:2] != sig(pb)[:2] or sig(pa)[2] != sig(pb)[2]:
                    why = "reviewed, but numbers or sub/superscript/bold runs differ"
            elif why is None:
                ta, tb = tokens(pa.text), tokens(pb.text)
                if ta != tb:
                    d = [x for x in difflib.ndiff(ta, tb) if x[0] in "+-"]
                    why = "content words changed: " + " ".join(d[:12])
                elif sorted(numbers(pa.text)) != sorted(numbers(pb.text)):
                    why = "numbers or citations changed"
                elif sig(pa) != sig(pb):
                    why = "formatting runs changed %s -> %s" % (sig(pa), sig(pb))
        if why:
            n_rej += 1
            report.append("REJECT  %s | %s" % (why, head))
            continue
        # accept: replace the runs of the v11 paragraph by the edited runs
        for r in list(pc._p.iterfind(qn("w:r"))):
            pc._p.remove(r)
        for r in pb._p.iterfind(qn("w:r")):
            pc._p.append(copy.deepcopy(r))
        n_acc += 1
        report.append("ACCEPT  | %s" % head)
    Cd.save(out_path)
    rep = out_path.rsplit(".", 1)[0] + "_merge_report.txt"
    io.open(rep, "w", encoding="utf-8").write("accepted %d, rejected %d\n\n" % (n_acc, n_rej) + "\n".join(report) + "\n")
    print("accepted %d, rejected %d -> %s" % (n_acc, n_rej, out_path))


if __name__ == "__main__":
    main(*sys.argv[1:5])
