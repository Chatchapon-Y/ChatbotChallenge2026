"""Fact sheets: counts and totals worked out once, at build time. Lv4.

A Lv4 answer is not written anywhere. "How many 3D printers does the
Centre run?" (7) needs the FDM, SLA and ultra-high resolution printers on
innowings and the two older models on innoacademy, added up. Retrieval
fetches 10 chunks and the model counts whatever those happen to hold, so
it answered 14. Here the items are listed and counted once, by hand-
checked rules, and saved as short passages the bot can retrieve.

Every fact names the phrases it was read from. A phrase that is no
longer in data/pages_clean.json (the site changed, or a typo here) is
reported, so a stale number is never shipped silently.

Run from submission_repo (no network, no API):

    python -m build.facts

Writes data/facts.json, and data/facts_lv5.json from the team's notes.txt
files in data/lv5_raw/.
bot/lv4.py reads both. Nothing goes into Chroma, so data/chroma is
untouched and nobody has to re-index.
"""
import json
import re
from pathlib import Path

PAGES = Path("data/pages_clean.json")
POSTER_COUNTS = Path("data/poster_counts.json")
# The notes are committed under data/lv5_raw/ (the photos are not);
# data/lv5_photos/ is the same survey on the machine that took it.
LV5_DIRS = [Path("data/lv5_raw"), Path("data/lv5_photos")]
OUT = Path("data/facts.json")
OUT_LV5 = Path("data/facts_lv5.json")

IW = "https://innowings.engg.hku.hk"
IA = "https://innoacademy.engg.hku.hk"


def fact(title: str, text: str, sources: list[str], evidence: list[str]) -> dict:
    return {"title": title, "text": text, "sources": sources, "evidence": evidence}


def pitch_facts() -> list[dict]:
    shortlisted = {
        "2023": ["Breathing in the Virtual World", "Carbon Dragon",
                 "E-HealK Telemedicine Platform",
                 "High-fidelity Stereo Imaging Depth Estimation for Metaverse Devices",
                 "Real-Time 3D Neural Holography for Near-eye Display",
                 "Smart PV Panel Cleaning Robot", "StoryGPT"],
        "2024": ["AR Glasses for Archaeologists", "Caretaker Connect App",
                 "Float: Message in a Bottle to Connect with Native Speakers Worldwide",
                 "Get your period", "NeuroHarbor", "Sixth Sense", "VRemind"],
    }
    stated = {"2025": "7 proposals are shortlisted", "2026": "6 proposals are shortlisted"}
    out = []
    for year, names in shortlisted.items():
        out.append(fact(
            f"Pitch New Tech Ideas {year}: shortlisted proposals (finalists)",
            f"Pitch New Tech Ideas {year} shortlisted {len(names)} proposals "
            f"(finalists): {'; '.join(names)}. The {year} page also still shows an "
            f"old block '5 proposals are shortlisted ... March 2, 2022': that is "
            f"about the 2022 contest, not {year}.",
            [f"{IA}/pitching_{year}/"], ["Shortlisted Projects"] + names))
    for year, phrase in stated.items():
        out.append(fact(
            f"Pitch New Tech Ideas {year}: shortlisted proposals (finalists)",
            f"Pitch New Tech Ideas {year} shortlisted {phrase.split()[0]} proposals "
            f"(finalists).",
            [f"{IA}/pitching_{year}/"], [phrase]))
    out.append(fact(
        "Pitch New Tech Ideas: finalists per year compared",
        "Shortlisted proposals (finalists) in Pitch New Tech Ideas: 2023: 7, "
        "2024: 7, 2025: 7, 2026: 6. 2024 had 1 more than 2026; 2025 had 1 more "
        "than 2026.",
        [f"{IA}/pitching_{y}/" for y in ("2023", "2024", "2025", "2026")], []))
    out.append(fact(
        "Pitch New Tech Ideas: prize money per year",
        "Every year from 2023 to 2026 Pitch New Tech Ideas selects a maximum of 3 "
        "winners, each awarded a cash prize of HK$1,200: prize money paid to the "
        "winners is at most 3 x HK$1,200 = HK$3,600. 'The Best Presenter' is a "
        "separate award, a cash prize of HK$1,000; only a question that counts "
        "the Best Presenter award too gets HK$4,600. Winning proposals can also "
        "get up to HK$100K of funding support (not a cash prize).",
        [f"{IA}/pitching_{y}/" for y in ("2023", "2024", "2025", "2026")],
        ["A maximum of 3 proposals will be selected as winners and awarded a cash prize of HK$1,200",
         "Best Presenter” will be awarded a cash prize of HK$1,000"]))
    return out


def printer_facts() -> list[dict]:
    return [fact(
        "3D printers run by the Innovation Wing: all models",
        "The Innovation Wing (the Centre) runs 7 3D printer models in total, "
        "listed on its two equipment pages: "
        "innowings equipment page: Bambu Lab P1S, Bambu Lab X1C and Bambu Lab "
        "H2D (3 FDM printers, PLA), Formlabs Form4 (SLA, resin) and MicroArch "
        "S240 (ultra-high resolution, 10 um); innoacademy equipment page: "
        "Ultimaker 3 Extended and Raise3D Pro2+. "
        "That is 3 Bambu Lab models, 5 FDM models in all (3 Bambu Lab, "
        "Ultimaker, Raise3D), 1 SLA printer, 1 ultra-high resolution printer. "
        "Largest build volume among the Bambu Lab printers: H2D, 350 x 320 x "
        "325 mm (P1S and X1C: 256 x 256 x 256 mm).",
        [f"{IW}/innowing1/equipment/", f"{IA}/equipment/"],
        ["Bambu Lab P1S", "Bambu Lab X1C", "Bambu Lab H2D", "Formlabs Form4",
         "MicroArch S240", "Ultimaker 3 extended", "Raise 3D pro 2+",
         "Bambu Lab H2D : 350 x 320 x 325"])]


def venue_facts() -> list[dict]:
    return [fact(
        "Innovation Wing venues: floor area and seats (venue page)",
        "Venues for events, net usable area: Makerspace assembling area 1: 153 "
        "square metres, 10 workbenches with 6 seats each (60 seats). Makerspace "
        "assembling area 2: 130 square metres, 5 workbenches with 6 seats each "
        "(30 seats). Areas 1 and 2 together: 283 square metres, 15 workbenches, "
        "90 seats. Open event area (LG/F): 66 square metres, 6 free tables by "
        "default. Event Hall A: 93 square metres; Event Hall B: 93 square "
        "metres; together 186 square metres; each hall seats at most 60 in a "
        "seminar setting (120 for both). Digital Learning Studio: 55 square "
        "metres, 29 computers with CAD software and 1 teacher desk with "
        "computer. Largest venue: Makerspace assembling area 1 (153); smallest: "
        "Digital Learning Studio (55). The open event area is 11 square metres "
        "larger than the Digital Learning Studio. Note: the innowings "
        "equipment page instead says the Digital Learning Studio has 20 high "
        "performance computers including the instructor's.",
        [f"{IA}/venue/", f"{IW}/innowing1/equipment/"],
        ["153 square meter 10 workbenches, each workbench has 6 seats",
         "130 square meter (Assembling area 2) 5 workbenches",
         "66 square meter", "93 square meter (Hall A) 93 square meter (Hall B)",
         "at most 60 seats", "55 square meter", "29 computers with CAD software",
         "20 high performance computers (including the instructor"])]


def opening_facts() -> list[dict]:
    return [fact(
        "Innovation Wing opening hours per week, and staff on duty",
        "Opening hours for members (innowings membership page): Monday to "
        "Friday 9:00am to 9:00pm (12 hours a day, 60 hours), Saturday 9:00am "
        "to 1:00pm (4 hours): 64 hours a week; closed on Sunday and "
        "university/public holidays. The innoacademy membership page gives "
        "older hours: Monday to Friday 9:30am to 9:00pm, Saturday 9:30am to "
        "5:00pm (65 hours a week). In most opening hours 5 tutors and 4 "
        "technicians are on duty: 9 staff in total.",
        [f"{IW}/innowing1/membership/", f"{IA}/membership/", f"{IW}/tutor/", f"{IW}/staff/"],
        ["Monday to Friday: 9:00am – 9:00pm", "Saturday: 9:00am – 1:00pm",
         "Monday to Friday: 9:30 am – 9:00 pm", "Saturday 9:30am – 5:00pm",
         "you can find 5 tutors on duty", "you can find 4 technicians on duty"])]


def poster_facts() -> list[dict]:
    if not POSTER_COUNTS.exists():
        print(f"skip poster counts: {POSTER_COUNTS} missing (python -m build.counts)")
        return []
    counts = {y: int(n) for y, n in json.loads(POSTER_COUNTS.read_text(encoding="utf-8")).items()}
    years = sorted(counts)
    most = max(years, key=counts.get)
    least = min(years, key=counts.get)
    per_year = ", ".join(f"{y}: {counts[y]}" for y in reversed(years))
    diffs = "; ".join(f"{b} vs {a}: {counts[b] - counts[a]:+d}"
                      for a, b in zip(years, years[1:]))
    return [fact(
        "Workshop posters per year on the Innovation Academy website",
        f"Number of workshop posters on the Innovation Academy workshop page "
        f"(innoacademy.engg.hku.hk/workshop/), counted per 'Year' heading: "
        f"{per_year}. Total {years[0]}-{years[-1]}: {sum(counts.values())}. Most "
        f"posters: {most} ({counts[most]}); fewest: {least} ({counts[least]}). "
        f"Change from the year before: {diffs}.",
        [f"{IA}/workshop/"], [])]


def lv5_facts() -> list[dict]:
    """Our own photo notes: the counts we made in the building (Lv5)."""
    out = []
    root = next((d for d in LV5_DIRS if any(d.glob("**/notes.txt"))), None)
    for notes in sorted(root.glob("**/notes.txt")) if root else []:
        text = notes.read_text(encoding="utf-8").strip()
        if not text:
            continue
        where = notes.parent.relative_to(root).as_posix().replace("_", " ").replace("/", ", ")
        out.append(fact(f"In-building notes: {where}",
                        # The marker answer.SYSTEM_PROMPT trusts for counts.
                        f"Notes written on site by our team, counted by hand, "
                        f"{where}: {text}", [], []))
    return out


def check(facts: list[dict], pages: dict[str, str]) -> int:
    """Print every evidence phrase missing from its sources. Returns the count."""
    norm = lambda s: re.sub(r"\s+", " ", s).lower()
    missing = 0
    for f in facts:
        hay = norm(" ".join(pages.get(u, "") for u in f["sources"]))
        for phrase in f["evidence"]:
            if norm(phrase) not in hay:
                missing += 1
                print(f"  MISSING in {f['sources']}: {phrase!r}  ({f['title']})")
    return missing


if __name__ == "__main__":
    pages = {p["url"]: p["text"] for p in json.loads(PAGES.read_text(encoding="utf-8"))}
    facts = pitch_facts() + printer_facts() + venue_facts() + opening_facts() + poster_facts()
    missing = check(facts, pages)
    OUT.write_text(json.dumps(facts, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{len(facts)} facts -> {OUT}   ({missing} evidence phrases missing)")

    lv5 = lv5_facts()
    OUT_LV5.write_text(json.dumps(lv5, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{len(lv5)} in-building notes -> {OUT_LV5}")
