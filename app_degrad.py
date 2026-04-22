import io
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

# =============================
# Optional RDKit imports
# =============================
RDKit_AVAILABLE = True
RDKit_IMPORT_ERROR = None
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Draw, rdMolDescriptors
    from rdkit.Chem.rdChemReactions import ReactionFromSmarts
except Exception as e:
    RDKit_AVAILABLE = False
    RDKit_IMPORT_ERROR = str(e)


# =============================
# Page config
# =============================
st.set_page_config(
    page_title="Degradation & Nitrosamine Predictor",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================
# Constants and knowledge base
# =============================
APP_TITLE = "Degradation & Nitrosamine Predictor"
APP_SUBTITLE = (
    "MVP for prediction of pharmaceutical degradation products, API–excipient alerts, "
    "and nitrosamine risk triage."
)

EXCIPIENT_DB = [
    {
        "Excipient": "Lactose",
        "Category": "Reducing sugar",
        "Risk": "Maillard / carbonyl reactivity",
        "Typical concern": "Can react with primary/secondary amines under heat/humidity.",
        "Severity": 4,
    },
    {
        "Excipient": "Mannitol",
        "Category": "Polyol",
        "Risk": "Generally lower carbonyl reactivity",
        "Typical concern": "Usually lower direct reactivity than lactose.",
        "Severity": 1,
    },
    {
        "Excipient": "Microcrystalline cellulose",
        "Category": "Filler",
        "Risk": "Low direct chemical reactivity",
        "Typical concern": "Usually inert in many solid formulations.",
        "Severity": 1,
    },
    {
        "Excipient": "PEG",
        "Category": "Polyether",
        "Risk": "Peroxide-related oxidation",
        "Typical concern": "May contain trace peroxides that promote oxidation.",
        "Severity": 4,
    },
    {
        "Excipient": "Povidone",
        "Category": "Polymer",
        "Risk": "Peroxide-related oxidation",
        "Typical concern": "Can contribute to oxidative stress depending on grade/storage.",
        "Severity": 3,
    },
    {
        "Excipient": "Magnesium stearate",
        "Category": "Lubricant",
        "Risk": "Basic microenvironment / hydrolysis modulation",
        "Typical concern": "Can affect local pH and hydrolysis indirectly.",
        "Severity": 2,
    },
    {
        "Excipient": "Sodium starch glycolate",
        "Category": "Disintegrant",
        "Risk": "Moisture contribution",
        "Typical concern": "Can influence local water availability.",
        "Severity": 2,
    },
    {
        "Excipient": "Sodium nitrite",
        "Category": "Nitrosating agent",
        "Risk": "Direct nitrosation risk",
        "Typical concern": "Strong nitrosamine risk driver in acidic conditions.",
        "Severity": 5,
    },
    {
        "Excipient": "Crospovidone",
        "Category": "Disintegrant",
        "Risk": "Potential oxidative impurities depending on source",
        "Typical concern": "May contribute to oxidative stress in some systems.",
        "Severity": 2,
    },
]

NITROSATING_AGENTS = {
    "Sodium nitrite",
    "Nitrite",
    "Nitric oxide donors",
    "Nitrous acid",
    "Nitrate/nitrite contaminated excipient",
}

DEFAULT_EXCIPIENTS = [
    "Lactose",
    "Mannitol",
    "Microcrystalline cellulose",
    "PEG",
    "Povidone",
    "Magnesium stearate",
]

CONDITION_PRESETS = {
    "Hydrolytic (acidic)": {
        "pH": 2.0,
        "temperature_c": 40,
        "water_present": True,
        "oxidative_stress": False,
        "light_exposure": False,
        "metal_trace": False,
        "humidity_high": True,
    },
    "Hydrolytic (basic)": {
        "pH": 10.0,
        "temperature_c": 40,
        "water_present": True,
        "oxidative_stress": False,
        "light_exposure": False,
        "metal_trace": False,
        "humidity_high": True,
    },
    "Oxidative": {
        "pH": 7.0,
        "temperature_c": 40,
        "water_present": True,
        "oxidative_stress": True,
        "light_exposure": False,
        "metal_trace": True,
        "humidity_high": False,
    },
    "Photolytic": {
        "pH": 7.0,
        "temperature_c": 25,
        "water_present": False,
        "oxidative_stress": False,
        "light_exposure": True,
        "metal_trace": False,
        "humidity_high": False,
    },
    "Thermal / humid": {
        "pH": 7.0,
        "temperature_c": 60,
        "water_present": True,
        "oxidative_stress": False,
        "light_exposure": False,
        "metal_trace": False,
        "humidity_high": True,
    },
}


@dataclass
class DegradationRule:
    name: str
    family: str
    smirks: str
    trigger_notes: str
    base_score: int
    conditions: Dict[str, object]
    max_products: int = 5


DEGRADATION_RULES = [
    DegradationRule(
        name="Ester hydrolysis",
        family="Hydrolysis",
        smirks="[C:1](=[O:2])[O:3][C:4]>>[C:1](=[O:2])[O].[O][C:4]",
        trigger_notes="Favored by water, acid/base, and heat.",
        base_score=80,
        conditions={"needs_water": True, "pH_zone": "extreme"},
    ),
    DegradationRule(
        name="Carbamate hydrolysis",
        family="Hydrolysis",
        smirks="[N:1][C:2](=[O:3])[O:4][C:5]>>[N:1][C:2](=[O:3])O.[O][C:5]",
        trigger_notes="Favored by water and acidic/basic stress.",
        base_score=72,
        conditions={"needs_water": True, "pH_zone": "extreme"},
    ),
    DegradationRule(
        name="Amide hydrolysis (simplified)",
        family="Hydrolysis",
        smirks="[C:1](=[O:2])[N:3][C:4]>>[C:1](=[O:2])O.[N:3][C:4]",
        trigger_notes="Usually slower; favored by harsh pH and heat.",
        base_score=48,
        conditions={"needs_water": True, "pH_zone": "extreme", "heat_bonus": True},
    ),
    DegradationRule(
        name="Thioether oxidation to sulfoxide",
        family="Oxidation",
        smirks="[C:1][S:2][C:3]>>[C:1][S:2](=O)[C:3]",
        trigger_notes="Favored by peroxide, oxygen, or oxidative excipients.",
        base_score=86,
        conditions={"needs_oxidation": True},
    ),
    DegradationRule(
        name="Sulfoxide oxidation to sulfone",
        family="Oxidation",
        smirks="[C:1][S:2](=O)[C:3]>>[C:1][S:2](=O)(=O)[C:3]",
        trigger_notes="Further oxidation under stronger oxidative stress.",
        base_score=70,
        conditions={"needs_oxidation": True},
    ),
    DegradationRule(
        name="Tertiary amine oxidation to N-oxide",
        family="Oxidation",
        smirks="[N+0:1]([C:2])([C:3])[C:4]>>[N+:1]([C:2])([C:3])([C:4])[O-]",
        trigger_notes="Common under peroxide-related stress.",
        base_score=82,
        conditions={"needs_oxidation": True},
    ),
    DegradationRule(
        name="Phenol oxidation to quinone-like product (simplified)",
        family="Oxidation",
        smirks="[cH:1]1[cH:2][c:3]([OH:4])[cH:5][cH:6][cH:7]1>>[O:4]=[c:3]1[cH:2][cH:1][cH:7][cH:6][cH:5]1",
        trigger_notes="Simplified aromatic oxidation alert for phenolic systems.",
        base_score=55,
        conditions={"needs_oxidation": True, "light_bonus": True},
    ),
    DegradationRule(
        name="Benzylic alcohol oxidation to aldehyde",
        family="Oxidation",
        smirks="[c:1][CH2:2][OH:3]>>[c:1][CH:2]=O",
        trigger_notes="Favored by oxidants and oxygen/metal catalysis.",
        base_score=66,
        conditions={"needs_oxidation": True},
    ),
]


# =============================
# Utility functions
# =============================
def init_session_state() -> None:
    defaults = {
        "deg_results": [],
        "nitro_results": {},
        "selected_preset": "Custom",
        "api_smiles": "",
        "api_name": "",
        "selected_excipients": DEFAULT_EXCIPIENTS,
        "custom_excipients": "",
        "pH": 7.0,
        "temperature_c": 25,
        "water_present": True,
        "oxidative_stress": False,
        "light_exposure": False,
        "metal_trace": False,
        "humidity_high": False,
        "formulation_type": "Solid oral",
        "nitrosating_source_present": False,
        "storage_time_months": 6,
        "max_products_per_rule": 5,
        "max_total_products": 50,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def safe_mol_from_smiles(smiles: str):
    if not RDKit_AVAILABLE:
        return None
    if not smiles or not smiles.strip():
        return None
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            return None
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def canonicalize_smiles(smiles: str) -> Optional[str]:
    mol = safe_mol_from_smiles(smiles)
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def mol_to_formula_and_mass(mol) -> Tuple[str, float]:
    if mol is None:
        return "", 0.0
    try:
        formula = rdMolDescriptors.CalcMolFormula(mol)
        exact_mass = rdMolDescriptors.CalcExactMolWt(mol)
        return formula, float(exact_mass)
    except Exception:
        return "", 0.0


def count_functional_groups(mol) -> Dict[str, int]:
    if mol is None or not RDKit_AVAILABLE:
        return {}

    patterns = {
        "ester": "[CX3](=O)[OX2][#6]",
        "amide": "[CX3](=O)[NX3][#6]",
        "carbamate": "[NX3][CX3](=O)[OX2][#6]",
        "lactam_like": "[NX3][CX3](=O)",
        "thioether": "[#6][SX2][#6]",
        "sulfoxide": "[#6]S(=O)[#6]",
        "tertiary_amine": "[NX3+0]([#6])([#6])[#6]",
        "secondary_amine": "[NX3;H1]([#6])[#6]",
        "primary_amine": "[NX3;H2][#6]",
        "phenol": "c[OX2H]",
        "aniline_like": "c[NX3;H1,H2,+0]",
        "benzylic_alcohol": "c[CH2][OX2H]",
    }

    out = {}
    for name, smarts in patterns.items():
        try:
            patt = Chem.MolFromSmarts(smarts)
            out[name] = len(mol.GetSubstructMatches(patt)) if patt is not None else 0
        except Exception:
            out[name] = 0
    return out


def get_molecular_summary(smiles: str) -> Dict[str, object]:
    mol = safe_mol_from_smiles(smiles)
    if mol is None:
        return {"valid": False}

    formula, exact_mass = mol_to_formula_and_mass(mol)
    fg = count_functional_groups(mol)
    try:
        mw = Descriptors.MolWt(mol)
        hba = Descriptors.NumHAcceptors(mol)
        hbd = Descriptors.NumHDonors(mol)
        logp = Descriptors.MolLogP(mol)
        tpsa = rdMolDescriptors.CalcTPSA(mol)
    except Exception:
        mw, hba, hbd, logp, tpsa = 0.0, 0, 0, 0.0, 0.0

    return {
        "valid": True,
        "canonical_smiles": Chem.MolToSmiles(mol, canonical=True),
        "formula": formula,
        "exact_mass": exact_mass,
        "mol_wt": mw,
        "hba": hba,
        "hbd": hbd,
        "logp": logp,
        "tpsa": tpsa,
        "functional_groups": fg,
    }


def parse_custom_excipients(text: str) -> List[str]:
    if not text.strip():
        return []
    raw = [x.strip() for x in text.replace(";", ",").split(",")]
    return [x for x in raw if x]


def get_all_excipients(selected_excipients: List[str], custom_excipients: str) -> List[str]:
    excipients = list(selected_excipients)
    excipients.extend(parse_custom_excipients(custom_excipients))
    # Deduplicate preserving order
    seen = set()
    ordered = []
    for item in excipients:
        if item.lower() not in seen:
            ordered.append(item)
            seen.add(item.lower())
    return ordered


def excipient_risk_summary(excipients: List[str]) -> Tuple[List[Dict[str, object]], int, List[str]]:
    db_map = {row["Excipient"].lower(): row for row in EXCIPIENT_DB}
    matched_rows = []
    severity_sum = 0
    alerts = []

    for exc in excipients:
        rec = db_map.get(exc.lower())
        if rec:
            matched_rows.append(rec)
            severity_sum += int(rec["Severity"])
            alerts.append(f"{exc}: {rec['Risk']}")
        else:
            matched_rows.append(
                {
                    "Excipient": exc,
                    "Category": "Custom",
                    "Risk": "Unknown / not curated",
                    "Typical concern": "No curated rule yet.",
                    "Severity": 1,
                }
            )
            severity_sum += 1
    return matched_rows, severity_sum, alerts


def condition_context() -> Dict[str, object]:
    return {
        "pH": float(st.session_state.pH),
        "temperature_c": int(st.session_state.temperature_c),
        "water_present": bool(st.session_state.water_present),
        "oxidative_stress": bool(st.session_state.oxidative_stress),
        "light_exposure": bool(st.session_state.light_exposure),
        "metal_trace": bool(st.session_state.metal_trace),
        "humidity_high": bool(st.session_state.humidity_high),
        "formulation_type": st.session_state.formulation_type,
        "nitrosating_source_present": bool(st.session_state.nitrosating_source_present),
        "storage_time_months": int(st.session_state.storage_time_months),
    }


def condition_penalty_or_bonus(rule: DegradationRule, context: Dict[str, object]) -> int:
    score = 0
    rule_conditions = rule.conditions
    pH = float(context["pH"])

    if rule_conditions.get("needs_water"):
        score += 12 if context["water_present"] else -35

    if rule_conditions.get("needs_oxidation"):
        score += 18 if context["oxidative_stress"] else -30
        if context["metal_trace"]:
            score += 8

    pH_zone = rule_conditions.get("pH_zone")
    if pH_zone == "extreme":
        if pH <= 3 or pH >= 9:
            score += 18
        elif 4 <= pH <= 8:
            score -= 12

    if rule_conditions.get("heat_bonus") and context["temperature_c"] >= 40:
        score += 8

    if rule_conditions.get("light_bonus") and context["light_exposure"]:
        score += 8

    if context["humidity_high"]:
        score += 4

    return score


def build_rule_reaction(rule: DegradationRule):
    if not RDKit_AVAILABLE:
        return None
    try:
        rxn = ReactionFromSmarts(rule.smirks)
        return rxn
    except Exception:
        return None


def apply_rule_to_molecule(parent_smiles: str, rule: DegradationRule, max_products: int = 5) -> List[Dict[str, object]]:
    if not RDKit_AVAILABLE:
        return []

    mol = safe_mol_from_smiles(parent_smiles)
    if mol is None:
        return []

    rxn = build_rule_reaction(rule)
    if rxn is None:
        return []

    products_out = []
    seen = set()
    try:
        outcomes = rxn.RunReactants((mol,))
    except Exception:
        return []

    for outcome in outcomes:
        for product_mol in outcome:
            if product_mol is None:
                continue
            try:
                Chem.SanitizeMol(product_mol)
                smi = Chem.MolToSmiles(product_mol, canonical=True)
                if not smi or smi == parent_smiles or smi in seen:
                    continue
                seen.add(smi)
                formula, exact_mass = mol_to_formula_and_mass(product_mol)
                products_out.append(
                    {
                        "product_smiles": smi,
                        "formula": formula,
                        "exact_mass": exact_mass,
                    }
                )
                if len(products_out) >= max_products:
                    return products_out
            except Exception:
                continue
    return products_out


def degradation_rules_screen(parent_smiles: str, context: Dict[str, object], excipients: List[str]) -> List[Dict[str, object]]:
    mol = safe_mol_from_smiles(parent_smiles)
    if mol is None:
        return []

    fg = count_functional_groups(mol)
    exc_rows, exc_severity_sum, exc_alerts = excipient_risk_summary(excipients)
    oxidative_excipient_present = any(
        row["Risk"].lower().startswith("peroxide") or "oxidation" in row["Risk"].lower() for row in exc_rows
    )
    carbonyl_reactive_excipient_present = any(
        "maillard" in row["Risk"].lower() or "carbonyl" in row["Risk"].lower() for row in exc_rows
    )

    if oxidative_excipient_present and not context["oxidative_stress"]:
        context = dict(context)
        context["oxidative_stress"] = True

    results = []
    for rule in DEGRADATION_RULES:
        # Pre-filter by functional groups
        applicable = False
        if rule.name == "Ester hydrolysis" and fg.get("ester", 0) > 0:
            applicable = True
        elif rule.name == "Carbamate hydrolysis" and fg.get("carbamate", 0) > 0:
            applicable = True
        elif rule.name == "Amide hydrolysis (simplified)" and fg.get("amide", 0) > 0:
            applicable = True
        elif rule.name == "Thioether oxidation to sulfoxide" and fg.get("thioether", 0) > 0:
            applicable = True
        elif rule.name == "Sulfoxide oxidation to sulfone" and fg.get("sulfoxide", 0) > 0:
            applicable = True
        elif rule.name == "Tertiary amine oxidation to N-oxide" and fg.get("tertiary_amine", 0) > 0:
            applicable = True
        elif rule.name == "Phenol oxidation to quinone-like product (simplified)" and fg.get("phenol", 0) > 0:
            applicable = True
        elif rule.name == "Benzylic alcohol oxidation to aldehyde" and fg.get("benzylic_alcohol", 0) > 0:
            applicable = True

        if not applicable:
            continue

        score = rule.base_score + condition_penalty_or_bonus(rule, context)

        if carbonyl_reactive_excipient_present and fg.get("primary_amine", 0) + fg.get("secondary_amine", 0) > 0:
            score += 8

        products = apply_rule_to_molecule(
            parent_smiles,
            rule,
            max_products=min(int(st.session_state.max_products_per_rule), rule.max_products),
        )

        if not products:
            # Keep the alert even if product enumeration fails.
            results.append(
                {
                    "rule_name": rule.name,
                    "family": rule.family,
                    "score": max(0, min(100, score)),
                    "trigger_notes": rule.trigger_notes,
                    "products_found": 0,
                    "product_smiles": None,
                    "formula": None,
                    "exact_mass": None,
                    "excipient_alerts": " | ".join(exc_alerts) if exc_alerts else "",
                    "enumeration_status": "No explicit product enumerated",
                }
            )
            continue

        for prod in products:
            results.append(
                {
                    "rule_name": rule.name,
                    "family": rule.family,
                    "score": max(0, min(100, score)),
                    "trigger_notes": rule.trigger_notes,
                    "products_found": len(products),
                    "product_smiles": prod["product_smiles"],
                    "formula": prod["formula"],
                    "exact_mass": round(float(prod["exact_mass"]), 5),
                    "excipient_alerts": " | ".join(exc_alerts) if exc_alerts else "",
                    "enumeration_status": "Enumerated",
                }
            )

    results = sorted(results, key=lambda x: (x["score"], x["family"]), reverse=True)
    return results[: int(st.session_state.max_total_products)]


def identify_nitrosatable_centers(smiles: str) -> Dict[str, int]:
    mol = safe_mol_from_smiles(smiles)
    if mol is None:
        return {}
    fg = count_functional_groups(mol)
    return {
        "secondary_amine": fg.get("secondary_amine", 0),
        "tertiary_amine": fg.get("tertiary_amine", 0),
        "primary_amine": fg.get("primary_amine", 0),
        "aniline_like": fg.get("aniline_like", 0),
    }


def predict_simple_nitrosamines(smiles: str) -> List[Dict[str, object]]:
    """
    Simple heuristic candidates. This is deliberately conservative and educational.
    It provides candidate alerts rather than guaranteed products.
    """
    mol = safe_mol_from_smiles(smiles)
    if mol is None:
        return []

    candidates = []
    centers = identify_nitrosatable_centers(smiles)

    if centers.get("secondary_amine", 0) > 0:
        candidates.append(
            {
                "candidate_type": "N-nitrosamine from secondary amine",
                "likelihood": "High structural susceptibility",
                "rationale": "Secondary amines are classical nitrosamine precursors under nitrosating conditions.",
            }
        )

    if centers.get("tertiary_amine", 0) > 0:
        candidates.append(
            {
                "candidate_type": "Possible NDSRI / nitrosation-related tertiary amine pathway",
                "likelihood": "Context-dependent",
                "rationale": "Tertiary amines may participate through dealkylation, rearrangement, or related pathways depending on conditions.",
            }
        )

    if centers.get("primary_amine", 0) > 0:
        candidates.append(
            {
                "candidate_type": "Primary amine nitrosation alert",
                "likelihood": "Lower nitrosamine persistence; other nitrosation chemistry possible",
                "rationale": "Primary amines are less typical persistent nitrosamines but still relevant for nitrosation screening.",
            }
        )

    if centers.get("aniline_like", 0) > 0:
        candidates.append(
            {
                "candidate_type": "Aniline-like nitrosation alert",
                "likelihood": "Aromatic amine context-dependent",
                "rationale": "Aromatic amino groups can undergo nitrosation-related chemistry depending on formulation context.",
            }
        )

    return candidates


def nitrosamine_risk_score(smiles: str, context: Dict[str, object], excipients: List[str]) -> Dict[str, object]:
    centers = identify_nitrosatable_centers(smiles)
    exc_rows, exc_severity_sum, exc_alerts = excipient_risk_summary(excipients)

    nitrosating_excipient_hits = [
        row["Excipient"]
        for row in exc_rows
        if row["Excipient"] in NITROSATING_AGENTS or "nitros" in row["Risk"].lower() or "nitrite" in row["Risk"].lower()
    ]

    score = 0
    reasons = []

    if centers.get("secondary_amine", 0) > 0:
        score += 45
        reasons.append("Secondary amine present.")

    if centers.get("tertiary_amine", 0) > 0:
        score += 28
        reasons.append("Tertiary amine present.")

    if centers.get("primary_amine", 0) > 0:
        score += 10
        reasons.append("Primary amine present (screening relevance).")

    if context["nitrosating_source_present"]:
        score += 30
        reasons.append("Nitrosating source declared in formulation/process.")

    if nitrosating_excipient_hits:
        score += 25
        reasons.append(f"Nitrosating excipient alerts: {', '.join(nitrosating_excipient_hits)}.")

    if float(context["pH"]) <= 5.5:
        score += 18
        reasons.append("Acidic environment can favor nitrosation chemistry.")

    if context["humidity_high"]:
        score += 8
        reasons.append("High humidity can facilitate reactive microenvironments.")

    if int(context["temperature_c"]) >= 40:
        score += 8
        reasons.append("Elevated temperature may accelerate degradation/nitrosation pathways.")

    if int(context["storage_time_months"]) >= 12:
        score += 8
        reasons.append("Longer storage time increases cumulative risk exposure.")

    if context["formulation_type"] in ["Solid oral", "Capsule", "Tablet"]:
        score += 4
        reasons.append("Solid dosage form may support local microenvironment effects.")

    score = max(0, min(100, score))

    if score >= 75:
        band = "High"
    elif score >= 40:
        band = "Moderate"
    else:
        band = "Low"

    candidates = predict_simple_nitrosamines(st.session_state.api_smiles)

    return {
        "score": score,
        "band": band,
        "reasons": reasons,
        "centers": centers,
        "candidate_alerts": candidates,
        "excipient_alerts": exc_alerts,
    }


def results_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def json_bytes(data: object) -> bytes:
    return json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")


def render_molecule(smiles: str, width: int = 500):
    if not RDKit_AVAILABLE:
        st.warning("RDKit is not installed, so molecular rendering is disabled.")
        return
    mol = safe_mol_from_smiles(smiles)
    if mol is None:
        st.warning("Invalid SMILES. Unable to render the molecule.")
        return
    img = Draw.MolToImage(mol, size=(width, int(width * 0.6)))
    st.image(img, use_container_width=False)


def apply_preset(preset_name: str) -> None:
    if preset_name not in CONDITION_PRESETS:
        return
    preset = CONDITION_PRESETS[preset_name]
    for key, value in preset.items():
        st.session_state[key] = value
    st.session_state.selected_preset = preset_name


# =============================
# Session state init
# =============================
init_session_state()


# =============================
# Sidebar
# =============================
st.sidebar.title("Navigation & Inputs")
st.sidebar.caption("All processing is triggered only by buttons.")

st.sidebar.subheader("Molecule")
st.sidebar.text_input("API / compound name", key="api_name")
st.sidebar.text_area(
    "API SMILES",
    key="api_smiles",
    height=120,
    placeholder="Paste a valid SMILES here...",
)

preset_options = ["Custom"] + list(CONDITION_PRESETS.keys())
selected_preset = st.sidebar.selectbox(
    "Condition preset",
    options=preset_options,
    index=preset_options.index(st.session_state.selected_preset) if st.session_state.selected_preset in preset_options else 0,
)

if st.sidebar.button("Load preset conditions"):
    if selected_preset != "Custom":
        apply_preset(selected_preset)
        st.sidebar.success(f"Loaded preset: {selected_preset}")
    else:
        st.sidebar.info("Custom selected. No preset values applied.")

st.sidebar.subheader("Stress context")
st.sidebar.slider("pH", min_value=0.0, max_value=14.0, value=float(st.session_state.pH), step=0.5, key="pH")
st.sidebar.slider("Temperature (°C)", min_value=0, max_value=100, value=int(st.session_state.temperature_c), step=1, key="temperature_c")
st.sidebar.checkbox("Water present", key="water_present")
st.sidebar.checkbox("Oxidative stress", key="oxidative_stress")
st.sidebar.checkbox("Light exposure", key="light_exposure")
st.sidebar.checkbox("Metal trace / catalytic metals", key="metal_trace")
st.sidebar.checkbox("High humidity", key="humidity_high")
st.sidebar.selectbox(
    "Formulation type",
    options=["Solid oral", "Tablet", "Capsule", "Solution", "Suspension", "Topical", "Injectable", "Other"],
    key="formulation_type",
)
st.sidebar.slider("Storage time (months)", min_value=0, max_value=60, value=int(st.session_state.storage_time_months), step=1, key="storage_time_months")

st.sidebar.subheader("Excipients")
excipient_options = [row["Excipient"] for row in EXCIPIENT_DB]
st.sidebar.multiselect(
    "Curated excipients",
    options=excipient_options,
    default=st.session_state.selected_excipients,
    key="selected_excipients",
)
st.sidebar.text_area(
    "Custom excipients (comma-separated)",
    key="custom_excipients",
    height=100,
    placeholder="e.g. Sodium nitrite, Citric acid",
)
st.sidebar.checkbox("Explicit nitrosating source present", key="nitrosating_source_present")

st.sidebar.subheader("Prediction limits")
st.sidebar.slider("Max products per rule", min_value=1, max_value=10, value=int(st.session_state.max_products_per_rule), step=1, key="max_products_per_rule")
st.sidebar.slider("Max total products", min_value=10, max_value=200, value=int(st.session_state.max_total_products), step=10, key="max_total_products")

run_deg_sidebar = st.sidebar.button("Run degradation prediction", type="primary")
run_nitro_sidebar = st.sidebar.button("Run nitrosamine triage", type="primary")
run_all_sidebar = st.sidebar.button("Run full analysis", type="primary")
clear_all_sidebar = st.sidebar.button("Clear results")

if clear_all_sidebar:
    st.session_state.deg_results = []
    st.session_state.nitro_results = {}
    st.sidebar.success("Results cleared.")


# =============================
# Header
# =============================
st.title(APP_TITLE)
st.caption(APP_SUBTITLE)

if not RDKit_AVAILABLE:
    st.error(
        "RDKit is not available in this environment. Install RDKit to enable structure parsing, reactions, and molecular rendering. "
        f"Import error: {RDKit_IMPORT_ERROR}"
    )


# =============================
# Top level tabs
# =============================
tab_overview, tab_deg, tab_nitro, tab_excip, tab_kb, tab_results = st.tabs(
    [
        "Overview",
        "Degradation Predictor",
        "Nitrosamine Triage",
        "Excipients",
        "Knowledge Base",
        "Results & Export",
    ]
)


# =============================
# Shared values
# =============================
all_excipients = get_all_excipients(st.session_state.selected_excipients, st.session_state.custom_excipients)
context = condition_context()
summary = get_molecular_summary(st.session_state.api_smiles)


# =============================
# Trigger logic from sidebar buttons
# =============================
def run_degradation_workflow() -> None:
    if not RDKit_AVAILABLE:
        st.warning("RDKit is required for degradation prediction.")
        return
    if not st.session_state.api_smiles.strip():
        st.warning("Please provide an API SMILES before running the degradation predictor.")
        return
    if not summary.get("valid", False):
        st.warning("Invalid SMILES. Please correct it before running the prediction.")
        return
    st.session_state.deg_results = degradation_rules_screen(st.session_state.api_smiles, context, all_excipients)


def run_nitrosamine_workflow() -> None:
    if not RDKit_AVAILABLE:
        st.warning("RDKit is required for nitrosamine triage.")
        return
    if not st.session_state.api_smiles.strip():
        st.warning("Please provide an API SMILES before running nitrosamine triage.")
        return
    if not summary.get("valid", False):
        st.warning("Invalid SMILES. Please correct it before running the analysis.")
        return
    st.session_state.nitro_results = nitrosamine_risk_score(st.session_state.api_smiles, context, all_excipients)


if run_deg_sidebar:
    run_degradation_workflow()

if run_nitro_sidebar:
    run_nitrosamine_workflow()

if run_all_sidebar:
    run_degradation_workflow()
    run_nitrosamine_workflow()


# =============================
# Tab: Overview
# =============================
with tab_overview:
    col1, col2 = st.columns([1.1, 1.4])

    with col1:
        st.subheader("Current input")
        st.write(f"**Compound name:** {st.session_state.api_name or 'Not provided'}")
        st.write(f"**SMILES:** `{st.session_state.api_smiles or 'Not provided'}`")
        st.write(f"**Formulation type:** {st.session_state.formulation_type}")
        st.write(f"**Excipients:** {', '.join(all_excipients) if all_excipients else 'None'}")

        st.subheader("Condition summary")
        st.json(context)

    with col2:
        st.subheader("Molecule preview")
        if st.session_state.api_smiles.strip() and summary.get("valid", False):
            render_molecule(st.session_state.api_smiles, width=420)
        else:
            st.info("Provide a valid SMILES to display the structure.")

    st.divider()

    if summary.get("valid", False):
        st.subheader("Molecular summary")
        fg = summary.get("functional_groups", {})
        summary_df = pd.DataFrame(
            {
                "Property": [
                    "Canonical SMILES",
                    "Formula",
                    "Exact mass",
                    "Molecular weight",
                    "HBA",
                    "HBD",
                    "cLogP",
                    "TPSA",
                ],
                "Value": [
                    summary["canonical_smiles"],
                    summary["formula"],
                    round(summary["exact_mass"], 5),
                    round(summary["mol_wt"], 3),
                    summary["hba"],
                    summary["hbd"],
                    round(summary["logp"], 3),
                    round(summary["tpsa"], 3),
                ],
            }
        )
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        fg_df = pd.DataFrame(
            [{"Functional group": k, "Count": v} for k, v in fg.items()]
        )
        st.subheader("Detected functional groups")
        st.dataframe(fg_df, use_container_width=True, hide_index=True)
    else:
        st.warning("Invalid or missing SMILES. The app will not run the chemistry modules until this is corrected.")

    st.divider()
    col_run1, col_run2, col_run3 = st.columns(3)
    with col_run1:
        if st.button("Run degradation predictor", key="run_deg_main"):
            run_degradation_workflow()
    with col_run2:
        if st.button("Run nitrosamine triage", key="run_nitro_main"):
            run_nitrosamine_workflow()
    with col_run3:
        if st.button("Run everything", key="run_all_main"):
            run_degradation_workflow()
            run_nitrosamine_workflow()


# =============================
# Tab: Degradation Predictor
# =============================
with tab_deg:
    st.subheader("Degradation product predictor")
    st.caption("This MVP combines structure alerts, simplified reaction transforms, and condition-based prioritization.")

    info_col1, info_col2 = st.columns([1, 1])
    with info_col1:
        st.markdown(
            "**What it does**\n"
            "- Screens for hydrolysis and oxidation liabilities\n"
            "- Applies simplified rule-based reaction transforms\n"
            "- Ranks products by condition plausibility\n"
            "- Incorporates excipient-related alert modifiers"
        )
    with info_col2:
        st.markdown(
            "**Important limitation**\n"
            "This is a first-pass prioritization engine, not a validated mechanistic oracle. "
            "The results should guide LC-MS method planning and expert review."
        )

    if st.button("Run degradation prediction now", key="run_deg_tab", type="primary"):
        run_degradation_workflow()

    deg_results = st.session_state.deg_results
    if deg_results:
        deg_df = pd.DataFrame(deg_results)
        st.success(f"Generated {len(deg_df)} degradation alerts/products.")

        metric1, metric2, metric3 = st.columns(3)
        metric1.metric("Total alerts/products", len(deg_df))
        metric2.metric("Enumerated products", int((deg_df["enumeration_status"] == "Enumerated").sum()))
        metric3.metric("Top score", int(deg_df["score"].max()))

        st.dataframe(deg_df, use_container_width=True, hide_index=True)

        enumerated = deg_df[deg_df["product_smiles"].notna()].copy()
        if not enumerated.empty:
            st.subheader("Preview selected predicted product")
            option_labels = [
                f"{i+1}. {row['rule_name']} | score {row['score']} | {row['product_smiles']}"
                for i, (_, row) in enumerate(enumerated.head(25).iterrows())
            ]
            selected = st.selectbox("Choose a predicted product", option_labels, key="selected_deg_preview")
            idx = option_labels.index(selected)
            product_row = enumerated.head(25).iloc[idx]
            colA, colB = st.columns([1.1, 1])
            with colA:
                render_molecule(product_row["product_smiles"], width=420)
            with colB:
                st.write(f"**Rule:** {product_row['rule_name']}")
                st.write(f"**Family:** {product_row['family']}")
                st.write(f"**Score:** {product_row['score']}")
                st.write(f"**Formula:** {product_row['formula']}")
                st.write(f"**Exact mass:** {product_row['exact_mass']}")
                st.write(f"**Notes:** {product_row['trigger_notes']}")
                if product_row["excipient_alerts"]:
                    st.write(f"**Excipient alerts:** {product_row['excipient_alerts']}")

        st.download_button(
            "Download degradation results CSV",
            data=results_to_csv_bytes(deg_df),
            file_name="degradation_results.csv",
            mime="text/csv",
            key="download_deg_csv",
        )
        st.download_button(
            "Download degradation results JSON",
            data=json_bytes(deg_results),
            file_name="degradation_results.json",
            mime="application/json",
            key="download_deg_json",
        )
    else:
        st.info("No degradation results yet. Use the button above or the sidebar to run the predictor.")


# =============================
# Tab: Nitrosamine Triage
# =============================
with tab_nitro:
    st.subheader("Nitrosamine risk triage")
    st.caption("This module is intended for structural risk screening and contextual prioritization, not definitive confirmation.")

    st.markdown(
        "**Current logic includes:**\n"
        "- Detection of nitrosatable amine motifs\n"
        "- Contextual scoring for pH, humidity, temperature, storage time, and excipients\n"
        "- Candidate alert generation for classical nitrosamine susceptibility and NDSRI-like pathways"
    )

    if st.button("Run nitrosamine triage now", key="run_nitro_tab", type="primary"):
        run_nitrosamine_workflow()

    nitro = st.session_state.nitro_results
    if nitro:
        coln1, coln2, coln3 = st.columns(3)
        coln1.metric("Nitrosamine risk score", nitro["score"])
        coln2.metric("Risk band", nitro["band"])
        coln3.metric("Candidate alerts", len(nitro.get("candidate_alerts", [])))

        st.subheader("Detected nitrosatable centers")
        centers_df = pd.DataFrame(
            [{"Center": k, "Count": v} for k, v in nitro.get("centers", {}).items()]
        )
        st.dataframe(centers_df, use_container_width=True, hide_index=True)

        st.subheader("Risk rationale")
        for reason in nitro.get("reasons", []):
            st.write(f"- {reason}")

        st.subheader("Candidate nitrosation alerts")
        alerts = nitro.get("candidate_alerts", [])
        if alerts:
            alerts_df = pd.DataFrame(alerts)
            st.dataframe(alerts_df, use_container_width=True, hide_index=True)
        else:
            st.info("No obvious nitrosamine structural alert detected by the current heuristic rules.")

        if nitro.get("excipient_alerts"):
            st.subheader("Excipient-driven alerts")
            for alert in nitro["excipient_alerts"]:
                st.write(f"- {alert}")

        st.download_button(
            "Download nitrosamine triage JSON",
            data=json_bytes(nitro),
            file_name="nitrosamine_triage.json",
            mime="application/json",
            key="download_nitro_json",
        )
    else:
        st.info("No nitrosamine result yet. Run the triage with the button above or in the sidebar.")


# =============================
# Tab: Excipients
# =============================
with tab_excip:
    st.subheader("Excipient database and formulation context")
    db_df = pd.DataFrame(EXCIPIENT_DB)
    st.dataframe(db_df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Current formulation excipient summary")
    if all_excipients:
        matched_rows, severity_sum, alerts = excipient_risk_summary(all_excipients)
        st.dataframe(pd.DataFrame(matched_rows), use_container_width=True, hide_index=True)

        colx1, colx2 = st.columns(2)
        colx1.metric("Total excipient severity", severity_sum)
        colx2.metric("Number of excipients", len(all_excipients))

        if alerts:
            st.markdown("**Active alerts**")
            for alert in alerts:
                st.write(f"- {alert}")
    else:
        st.info("No excipients selected yet.")


# =============================
# Tab: Knowledge Base
# =============================
with tab_kb:
    st.subheader("Implemented degradation rules")
    rules_df = pd.DataFrame(
        [
            {
                "Rule": r.name,
                "Family": r.family,
                "Base score": r.base_score,
                "Trigger notes": r.trigger_notes,
                "SMIRKS": r.smirks,
                "Conditions": json.dumps(r.conditions),
            }
            for r in DEGRADATION_RULES
        ]
    )
    st.dataframe(rules_df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Roadmap suggestions for next versions")
    st.markdown(
        "- Add more reaction rules for photolysis, dealkylation, dehalogenation, and rearrangements.\n"
        "- Expand API–excipient interaction logic.\n"
        "- Add user-editable knowledge base from CSV.\n"
        "- Add batch mode from uploaded CSV with Name and SMILES columns.\n"
        "- Add LC-MS prioritization based on adducts, ion mode, and expected polarity.\n"
        "- Add nitrosamine-specific precursor decomposition and context trees."
    )


# =============================
# Tab: Results & Export
# =============================
with tab_results:
    st.subheader("Combined results")

    deg_results = st.session_state.deg_results
    nitro_results = st.session_state.nitro_results

    if deg_results:
        st.markdown("**Degradation predictor output**")
        deg_df = pd.DataFrame(deg_results)
        st.dataframe(deg_df, use_container_width=True, hide_index=True)
    else:
        st.info("No degradation output stored.")

    if nitro_results:
        st.markdown("**Nitrosamine triage output**")
        nitro_summary = {
            "score": nitro_results.get("score"),
            "band": nitro_results.get("band"),
            "reasons": nitro_results.get("reasons"),
            "centers": nitro_results.get("centers"),
            "candidate_alerts": nitro_results.get("candidate_alerts"),
        }
        st.json(nitro_summary)
    else:
        st.info("No nitrosamine output stored.")

    st.divider()
    st.subheader("Export package")
    combined_export = {
        "app": APP_TITLE,
        "compound_name": st.session_state.api_name,
        "smiles": st.session_state.api_smiles,
        "context": context,
        "excipients": all_excipients,
        "molecular_summary": summary,
        "degradation_results": deg_results,
        "nitrosamine_results": nitro_results,
    }

    st.download_button(
        "Download full analysis JSON",
        data=json_bytes(combined_export),
        file_name="full_analysis.json",
        mime="application/json",
        key="download_full_json",
    )

    if deg_results:
        st.download_button(
            "Download combined degradation CSV",
            data=results_to_csv_bytes(pd.DataFrame(deg_results)),
            file_name="combined_degradation.csv",
            mime="text/csv",
            key="download_combined_deg_csv",
        )

    report_lines = [
        f"App: {APP_TITLE}",
        f"Compound: {st.session_state.api_name or 'N/A'}",
        f"SMILES: {st.session_state.api_smiles or 'N/A'}",
        f"Formulation type: {st.session_state.formulation_type}",
        f"Excipients: {', '.join(all_excipients) if all_excipients else 'None'}",
        f"Degradation results: {len(deg_results) if deg_results else 0}",
    ]
    if nitro_results:
        report_lines.extend(
            [
                f"Nitrosamine risk score: {nitro_results.get('score')}",
                f"Nitrosamine band: {nitro_results.get('band')}",
            ]
        )

    st.download_button(
        "Download simple TXT report",
        data="\n".join(report_lines).encode("utf-8"),
        file_name="analysis_report.txt",
        mime="text/plain",
        key="download_txt_report",
    )


# =============================
# Footer
# =============================
st.divider()
st.caption(
    "This MVP is intended for prioritization, hypothesis generation, and LC-MS planning support. "
    "It does not replace experimental forced degradation studies, impurity identification, or expert review."
)
