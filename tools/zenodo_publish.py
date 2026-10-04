# -*- coding: utf-8 -*-
"""Publish the two Zenodo records for manuscript v9.

  software    the draft reserved by zenodo_reserve.py (DOI already written into the manuscript): the tagged
              GitHub release as a zip, MIT
  preprint    a new version of the existing preprint concept 10.5281/zenodo.21210708: manuscript v9 (docx + pdf),
              highlights and graphical abstract, CC BY 4.0

Lessons carried over from earlier deposits: Zenodo's write schema is supplied in full; a new version inherits
the previous version's files, which are deleted from the draft first; a new version branches from the latest
record of the concept. The token is read from ZENODO_TOKEN and never written to disk.

    python zenodo_publish.py software|preprint [--dry] [--draft=ID to finish an interrupted preprint draft]
"""
import json
import os
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request

TOKEN = os.environ.get("ZENODO_TOKEN")
if not TOKEN:
    raise SystemExit("ZENODO_TOKEN is not set in the environment")
API = "https://zenodo.org/api"
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
STATE = os.path.join("C:" + os.sep, "YouTube", "_lattice_zenodo_state.json")
TAG, VERSION = "v1.0.0", "1.0.0"
PREPRINT_CONCEPT = "21210708"
DRY = "--dry" in sys.argv
GITHUB = "https://github.com/sandlerleon/rotational-lattice-maintenance"
CREATORS = [{"name": "Sandler, Leon", "affiliation": "Independent Researcher", "orcid": "0009-0007-4584-808X"}]
TITLE_PAPER = "Self-Maintained Order and Hysteretic Collapse in a Non-Equilibrium Rotational Lattice"
TITLE_CODE = ("Rotational-lattice maintenance: kinetic Monte Carlo test of mean-field bistability, sensing range "
              "and BKT finite-size scaling")
KEYWORDS = ["non-equilibrium statistical mechanics", "kinetic Monte Carlo", "XY model", "bistability and hysteresis",
            "mean-field closure", "dynamic hysteresis", "Berezinskii-Kosterlitz-Thouless", "finite-size scaling",
            "helicity modulus", "entropy production"]

R = json.load(open(os.path.join(REPO, "results", "mc_results.json"), encoding="utf-8"))
F = R["fss"]
S = R["sensing"]["sweep"]

FINDINGS = """<p><strong>What the Monte Carlo shows.</strong> The annealed mean-field reduction of the model is bistable
(window 3.92 &lt; &rho; &lt; 11.30 at h = 6, K = 0.5, &beta; = 20; threshold &beta;<sub>c</sub> = 2.42 at h = 6, and no
bistability for h &le; 1). On the lattice with nearest-neighbour feedback it is not. The loop area of &rho; sweeps
(L = 16&ndash;64, nine rates, eight runs) peaks at 2.5 for fast sweeps and is zero within error for dwells of 64 steps
or more at every size and in a colder field-free variant; the mean-field static area is 4.30. Ordered and collapsed
starts inside the window converge to the same state (within 0.004 in all 40 cases), in 15&ndash;115 steps independent
of N. The cause is the closure: the lattice applies the average of the steep feedback over a broad distribution of
local disorder, which flattens it. With the kill rate sensing wider blocks, the slow-sweep loop area is %.2f, %.2f,
%.2f, %.2f and %.2f for 4, 8, 24, 80 sites and global sensing, and the two starts split at 80 sites and globally.
Bistability is realized when degradation senses order over an extended range. Quenched-dilution finite-size scaling
(L = 16&ndash;96, eight realizations) places the field-free loss of stiffness at f<sub>KT</sub> = %.3f (1/ln<sup>2</sup>L
extrapolation, 95%% interval %.3f&ndash;%.3f) and %.3f (Weber&ndash;Minnhagen, %.3f&ndash;%.3f), far below site
percolation (0.407).</p>""" % (S["nn"]["256"]["area"], S["r1"]["256"]["area"], S["r2"]["256"]["area"], S["r4"]["256"]["area"],
                               S["global"]["256"]["area"], F["extrap_lnL2"]["f_inf"], F["extrap_lnL2"]["f_inf_ci95"][0],
                               F["extrap_lnL2"]["f_inf_ci95"][1], F["weber_minnhagen"]["f_KT"],
                               F["weber_minnhagen"]["f_KT_ci95"][0], F["weber_minnhagen"]["f_KT_ci95"][1])

DESC_CODE = """<p>Kinetic Monte Carlo study behind version 9 of <em>%s</em> (submitted to Physica A; preprint
<a href="https://doi.org/10.5281/zenodo.21210708">10.5281/zenodo.21210708</a>). It contains the engine (verified bit for bit
against the MyUncle framework, <a href="https://doi.org/10.5281/zenodo.21223569">10.5281/zenodo.21223569</a>), its test suite,
six studies (sweep-rate hysteresis, two-start branches, autocorrelation and current balance, quenched-dilution
finite-size scaling, sensing range, closure test), the analysis that turns raw runs into every number and figure of
Sections 4.3, 4.4, 5 and Appendix A, the mean-field threshold scan, all raw run outputs and the manuscript.</p>%s
<p>Every run is seeded from SeedSequence([20261003, code]) and can be repeated alone; intervals are bootstrap
intervals over independent runs or disorder realizations.</p>""" % (TITLE_PAPER, FINDINGS)

DESC_PAPER = """<p>Version 9, revised for Physica A. A two-dimensional rotor lattice whose ordering couplings are removed by an
order-dependent flux and restored by repair. The mean-field reduction is bistable above a computed feedback threshold;
version 9 adds a kinetic Monte Carlo study designed to separate static from rate-dependent hysteresis, and finds that
the bistability survives on the lattice only when degradation senses order over an extended region.</p>%s
<p>Also: an exact order&ndash;entropy bridge (dS/dU = &minus;&kappa; for the von Mises law) and the Schnakenberg entropy
production of the maintenance cycle. Code, raw data and analysis: <a href="%s">%s</a>, archived at
<a href="https://doi.org/{SW}">{SW}</a>. Version 9 also corrects the discrete-time statement of the &beta; = 0 baseline
and the positivity condition of the entropy production (&lambda;R &gt; &epsilon;<sup>2</sup>).</p>""" % (FINDINGS, GITHUB, GITHUB)


def req(method, url, data=None, headers=None, raw=None):
    h = {"Authorization": "Bearer " + TOKEN}
    if headers:
        h.update(headers)
    body = raw if raw is not None else (json.dumps(data).encode() if data is not None else None)
    if data is not None and raw is None:
        h["Content-Type"] = "application/json"
    r = urllib.request.Request(url, data=body, headers=h, method=method)
    try:
        with urllib.request.urlopen(r, timeout=600) as resp:
            t = resp.read()
            return json.loads(t) if t else {}
    except urllib.error.HTTPError as e:
        raise SystemExit("%s %s -> %s\n%s" % (method, url, e.code, e.read().decode()[:800]))


def upload(bucket, path, name):
    with open(path, "rb") as fh:
        req("PUT", "%s/%s" % (bucket, urllib.parse.quote(name)), raw=fh.read(), headers={"Content-Type": "application/octet-stream"})
    print("   uploaded %-58s %9.1f kB" % (name, os.path.getsize(path) / 1024.0))


def finish(did, meta):
    req("PUT", "%s/deposit/depositions/%s" % (API, did), data={"metadata": meta})
    print("   metadata written")
    if DRY:
        print("   DRY RUN - draft %s left unpublished" % did)
        return None
    pub = req("POST", "%s/deposit/depositions/%s/actions/publish" % (API, did))
    rec = req("GET", "%s/records/%s" % (API, pub["id"]))
    print("   PUBLISHED  DOI %s  concept %s" % (rec.get("doi"), rec.get("conceptdoi")))
    return rec


def software():
    st = json.load(open(STATE))["software"]
    tmp = os.path.join(os.environ.get("TEMP", "."), "rotational-lattice-maintenance-%s.zip" % VERSION)
    subprocess.check_call(["git", "-C", REPO, "archive", "--format=zip", "--prefix=rotational-lattice-maintenance-%s/" % VERSION,
                           "-o", tmp, TAG])
    print("=== software draft %s (reserved DOI %s)" % (st["id"], st["doi"]))
    upload(st["bucket"], tmp, os.path.basename(tmp))
    meta = {"title": TITLE_CODE, "upload_type": "software", "description": DESC_CODE, "creators": CREATORS,
            "keywords": KEYWORDS, "access_right": "open", "license": "mit-license", "version": VERSION, "language": "eng",
            "prereserve_doi": {"doi": st["doi"]},
            "related_identifiers": [
                {"identifier": GITHUB + "/tree/" + TAG, "relation": "isSupplementTo", "scheme": "url"},
                {"identifier": "10.5281/zenodo.21210708", "relation": "isSupplementTo", "scheme": "doi"},
                {"identifier": "10.5281/zenodo.21223569", "relation": "isDerivedFrom", "scheme": "doi"}]}
    finish(st["id"], meta)


def preprint():
    sw = json.load(open(STATE))["software"]["doi"]
    names = ("Self-Maintained_Order_Hysteretic_Collapse_v9.pdf", "Self-Maintained_Order_Hysteretic_Collapse_v9.docx",
             "Highlights_v9.docx", "Graphical_Abstract_v9.png")
    resume = [a.split("=", 1)[1] for a in sys.argv if a.startswith("--draft=")]
    if resume:                                   # a new-version draft already holding the v9 files
        draft = req("GET", "%s/deposit/depositions/%s" % (API, resume[0]))
        did = draft["id"]
        have = sorted(f["filename"] for f in draft.get("files", []))
        if have != sorted(names):
            raise SystemExit("draft %s holds %s" % (did, have))
        print("=== resuming preprint draft %s with its %d files" % (did, len(have)))
    else:
        latest = req("GET", "%s/records/%s/versions/latest" % (API, PREPRINT_CONCEPT))
        print("=== preprint concept %s -> latest record %s" % (PREPRINT_CONCEPT, latest["id"]))
        dep = req("POST", "%s/deposit/depositions/%s/actions/newversion" % (API, latest["id"]))
        draft = req("GET", dep["links"]["latest_draft"])
        did = draft["id"]
        print("   draft %s" % did)
        for f in draft.get("files", []):
            req("DELETE", "%s/deposit/depositions/%s/files/%s" % (API, did, f["id"]))
        print("   inherited files removed: %d" % len(draft.get("files", [])))
        ms = os.path.join(REPO, "manuscript")
        for name in names:
            upload(draft["links"]["bucket"], os.path.join(ms, name), name)
    meta = {"title": TITLE_PAPER, "upload_type": "publication", "publication_type": "preprint",
            "description": DESC_PAPER.replace("{SW}", sw), "creators": CREATORS, "keywords": KEYWORDS, "access_right": "open",
            "license": "cc-by-4.0", "version": "9", "language": "eng",
            "related_identifiers": [
                {"identifier": sw, "relation": "isSupplementedBy", "scheme": "doi"},
                {"identifier": GITHUB, "relation": "isSupplementedBy", "scheme": "url"},
                {"identifier": "10.5281/zenodo.21223569", "relation": "references", "scheme": "doi"}]}
    finish(did, meta)


if __name__ == "__main__":
    what = [a for a in sys.argv[1:] if not a.startswith("--")]
    if what == ["software"]:
        software()
    elif what == ["preprint"]:
        preprint()
    else:
        print(__doc__)
