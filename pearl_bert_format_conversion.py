###################################################################
# purpose: covert the structured longitudinal data into bert format
# one record per study_id
# input dataset: simulated_sample.csv
# output dataset: simulated_sample_bert.csv
###################################################################

# import packages
###################################################################
import pandas as pd
import numpy as np
###################################################################

# read study dataset, replace with your study dataset
df = pd.read_csv("study/data/simulated_sample.csv")

# convert date format
df["visit_date"] = pd.to_datetime(df["visit_date"], errors="coerce")
df["index_date"] = pd.to_datetime(df["index_date"], errors="coerce")

# sort datasets by study_id, index_date, visit_date (descending)
df = df.sort_values(by=["study_id", "index_date", "visit_date"], ascending=[True, True, False])

# align demo features and the features of all visit dates to a single reord of bert tokens format per study_id

# define functions for conversion
def _is_missing(x) -> bool:
    return x is None or pd.isna(x)

def _sas_int(x):
    if _is_missing(x):
        return None
    return int(np.trunc(x))

def build_bert_format(g: pd.DataFrame, doc_id_start: int) -> dict:
    g = g.copy()
    first = g.iloc[0]

    doc_id = doc_id_start + 1

    gender = "" if _is_missing(first.get("gender")) else str(first.get("gender")).lower()
    race_eth = "" if _is_missing(first.get("race_eth")) else str(first.get("race_eth"))
    race_first = race_eth.split()[0].lower() if race_eth.split() else ""
    smoking = "" if _is_missing(first.get("smoking")) else str(first.get("smoking")).lower().strip()
    text = f"{gender} {race_first} {smoking} end".strip()

    for _, r in g.iterrows():

        days = (r["index_date"] - r["visit_date"]).days if (pd.notna(r["index_date"]) and pd.notna(r["visit_date"])) else None
        age = r.get("age")
        text = (
            text.strip()
            + " t" + ("" if days is None else str(days))
            + " a" + ("" if _is_missing(age) else str(age))
        )

        bmi = r.get("bmi")
        if not _is_missing(bmi):
            if 0 <= bmi < 18.4:
                text += " blow"
            elif bmi > 53.9:
                text += " bhigh"
            elif 18.4 <= bmi <= 53.9:
                text += " b" + str(_sas_int(bmi))

        if r.get("acp") == 1:
            text += " acp"

        eos = r.get("eos_ord_value")
        if not _is_missing(eos):
            if 0 <= eos < 100:
                text += " eoslow"
            elif eos > 800:
                text += " eoshigh"
            elif 100 <= eos <= 800:
                text += " eos" + str(_sas_int(eos))

        dx_map = [
            ("AllergicRhinitus", "d1"),
            ("Anemia", "d2"),
            ("Anxiety", "d3"),
            ("Anxiety_AnxietyDisorders", "d4"),
            ("Anxiety_OCD", "d5"),
            ("Anxiety_PhobicDisorders", "d6"),
            ("Asthma", "d7"),
            ("AtopicDermatitis", "d8"),
            ("CVD", "d9"),
            ("ChronicRhinitus", "d10"),
            ("ChronicSinusitus", "d11"),
            ("Dementia", "d12"),
            ("Depression", "d13"),
            ("Diabetes", "d14"),
            ("GERD", "d15"),
            ("HeartDisease", "d16"),
            ("Hyperlipidemia", "d17"),
            ("Hypertension", "d18"),
            ("NasalPolyp", "d19"),
            ("PneuALRI", "d20"),
            ("PostNasalDrip", "d21"),
            ("SleepDisorders", "d22"),
            ("SleepDisorders_Insomnia", "d23"),
            ("SleepDisorders_Others", "d24"),
            ("SleepDisorders_SleepApnea", "d25"),
        ]
        for col, tok in dx_map:
            if r.get(col) == 1:
                text += f" {tok}"

        rx_map = [
            ("ANTIBACT_ANTIMICRO_AGENTS", "r1"),
            ("Antacids", "r2"),
            ("AntianginalAgents", "r3"),
            ("AntianxietyAgents", "r4"),
            ("Antiarrythmics", "r5"),
            ("Anticoagulants", "r6"),
            ("Antidepressants", "r7"),
            ("Antidiabetics", "r8"),
            ("AntifungalAgents", "r9"),
            ("Antihyperlipidemics", "r10"),
            ("AntihypertensiveAgents", "r11"),
            ("AntiobesityAgents", "r12"),
            ("Antipsychotics", "r13"),
            ("Antiseptics_Disinfectants", "r14"),
            ("AntiviralAgents", "r15"),
            ("BetaBlockers", "r16"),
            ("CalciumChannelBlockers", "r17"),
            ("Diuretics", "r18"),
            ("Glimepiride", "r19"),
            ("Glipizide", "r20"),
            ("Glyburide", "r21"),
            ("Hyponotics", "r22"),
            ("ICS", "r23"),
            ("ICS_LABA", "r24"),
            ("Insulin", "r25"),
            ("LABA", "r26"),
            ("LABA_LAMA", "r27"),
            ("LAMA", "r28"),
            ("Metformin", "r29"),
            ("SABA", "r30"),
            ("SABA_SAMA", "r31"),
            ("SAMA", "r32"),
            ("SCS", "r33"),
            ("SomatostaticAgents", "r34"),
            ("Sulfonylureas", "r35"),
            ("UlcerDrugs", "r36"),
            ("asthmaAgents", "r37"),
            ("biologic", "r38"),
            ("leukotriene", "r39"),
            ("theophylline", "r40"),
        ]
        for col, tok in rx_map:
            if r.get(col) == 1:
                text += f" {tok}"

        if r.get("aeroallergentests") == 1:
            text += " alltest"

        if r.get("aeroallergentests_pos") == 1:
            text += " allpos"

        if r.get("hpylori") == 1:
            text += " hpylori"

        if r.get("influenza") == 1:
            text += " influenza"

        evs = r.get("evs_minperweek")
        if not _is_missing(evs):
            if evs >= 0:
                text += " evs" + str(_sas_int(evs))

    text = text.strip() + " end"

    return {
        "study_id": first.get("study_id"),
        "index_date": first.get("index_date"),
        "text": text,
        "labels": first.get("labels"),
        "doc_id": doc_id,
    }

# initiatizing list
out_rows = []
doc_id_running = 0

#consolidate multiple visit_date records to single record per study_id
for (study_id, index_date), g in df.groupby(["study_id", "index_date"], sort=False):
    out_rows.append(build_bert_format(g, doc_id_running))
    doc_id_running += 1

# convert list to dataframe and retain neccessary columns
cohort_bert = pd.DataFrame(out_rows)
cohort_bert = cohort_bert[["study_id", "index_date", "text", "labels", "doc_id"]]

# save the converted formated into csv file
cohort_bert.to_csv("study/data/simulated_sample_bert.csv", index=False)
