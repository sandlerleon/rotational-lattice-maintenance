# -*- coding: utf-8 -*-
"""Open a new-version draft of the software record and record its pre-reserved DOI, so the DOI can be written into the
manuscript before anything is published. Adds key `software_<version>` to C:\YouTube\_lattice_zenodo_state.json.
The token is read from ZENODO_TOKEN and never written to disk.

    python zenodo_reserve_newversion.py 1.1.0
"""
import json
import os
import sys
import urllib.request

TOKEN = os.environ["ZENODO_TOKEN"]
STATE = os.path.join("C:" + os.sep, "YouTube", "_lattice_zenodo_state.json")
API = "https://zenodo.org/api"
version = sys.argv[1]
st = json.load(open(STATE))
key = "software_" + version
if key in st:
    raise SystemExit("already reserved: %s" % st[key])


def req(method, url):
    r = urllib.request.Request(url, method=method, headers={"Authorization": "Bearer " + TOKEN})
    with urllib.request.urlopen(r, timeout=60) as resp:
        return json.load(resp)


latest = req("GET", "%s/records/%s/versions/latest" % (API, st["software"]["concept_rec_id"]))
dep = req("POST", "%s/deposit/depositions/%s/actions/newversion" % (API, latest["id"]))
draft = req("GET", dep["links"]["latest_draft"])
st[key] = {"id": draft["id"], "doi": draft["metadata"]["prereserve_doi"]["doi"], "bucket": draft["links"]["bucket"],
           "inherited_files": [f["id"] for f in draft.get("files", [])], "parent": latest["id"]}
json.dump(st, open(STATE, "w"), indent=1)
print(json.dumps(st[key], indent=1))
