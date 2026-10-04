# -*- coding: utf-8 -*-
"""Create the Zenodo draft for the software record and pre-reserve its DOI, so the DOI can be written into the
manuscript before anything is published. State goes to C:\YouTube\_lattice_zenodo_state.json (no token in it).
The token is read from ZENODO_TOKEN and never written to disk."""
import json
import os
import urllib.request

TOKEN = os.environ["ZENODO_TOKEN"]
STATE = os.path.join("C:" + os.sep, "YouTube", "_lattice_zenodo_state.json")
if os.path.exists(STATE):
    raise SystemExit("already reserved: " + open(STATE).read())
req = urllib.request.Request("https://zenodo.org/api/deposit/depositions", method="POST",
                             data=json.dumps({"metadata": {"upload_type": "software", "prereserve_doi": True}}).encode(),
                             headers={"Authorization": "Bearer " + TOKEN, "Content-Type": "application/json"})
dep = json.load(urllib.request.urlopen(req, timeout=60))
st = {"software": {"id": dep["id"], "doi": dep["metadata"]["prereserve_doi"]["doi"], "bucket": dep["links"]["bucket"],
                   "concept_rec_id": dep.get("conceptrecid"), "concept_doi": "10.5281/zenodo.%s" % dep.get("conceptrecid")}}
json.dump(st, open(STATE, "w"), indent=1)
print(json.dumps(st, indent=1))
