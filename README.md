# Predikcia platu pomocou neurónovej siete

**Predmet:** Princípy a aplikácie neurónových sietí
**Úloha:** Regresia — predpovedať ročný plat (USD) na základe pracovných charakteristík
**Notebook:** [salary_nn.ipynb](salary_nn.ipynb)

## Setup

```bash
pip install -r req.txt
```

Dataset sa stiahne automaticky cez `kagglehub` (`nalisha/job-salary-prediction-dataset`).

---

## 1. Príprava a analýza dát

**Dataset:** `job_salary_prediction_dataset.csv`, **250 000 vzoriek**, 10 stĺpcov, žiadne chýbajúce hodnoty.

| Typ | Stĺpce |
|---|---|
| Numerické (3) | `experience_years` (0–20), `skills_count` (1–19), `certifications` (0–5) |
| Kategorické (6) | `job_title` (12), `education_level` (5), `industry` (10), `company_size` (5), `location` (10), `remote_work` (3) |
| Target | `salary` (31 867 – 333 046 USD; μ = 145 718, σ = 37 408) |

**Predspracovanie:**

1. **One-Hot Encoding** kategorických stĺpcov s `drop_first=True` → **42 vstupných čŕt**
   - `drop_first=True` redukuje multikolinearitu (k kategórií → k−1 stĺpcov; chýbajúca kategória je referenčná)
2. **Split 70 / 15 / 15** (train / val / test), `random_state=67` pre reprodukovateľnosť
   - Train: 175 000 | Val: 37 500 | Test: 37 500
3. **Štandardizácia featúr** (`(x − μ) / σ`, +1e−6 v menovateli proti deleniu nulou)
   - μ a σ fitnuté **iba na trénovacej množine** — val/test simulujú nasadenie na nevidené dáta
4. **PyTorch DataLoader**, batch size 64; `shuffle=True` pre train, `False` pre val/test

---

## 2. Architektúra modelu a zdôvodnenie

**`SalaryMLP`** — Multi-Layer Perceptron pre regresiu.

```
Input (42) → Linear(128) → ReLU → Linear(64) → ReLU → Linear(1)
```

- **13 825 parametrov**
- Výstupná vrstva: **1 neurón, bez aktivácie** (regresia musí vedieť predpovedať ľubovoľnú reálnu hodnotu)
- Lievikovitá redukcia šírky (128 → 64) — bežný vzor pre tabuľkové dáta
- **ReLU** aktivácia: rýchla, neutrpí vanishing gradientom

**Zdôvodnenie voľby:**
- Pri tabuľkových dátach s mixom numerických a one-hot featúr je MLP štandardnou baseline.
- Architektúra je zámerne **malá** — najprv overujeme, či je úloha vôbec riešiteľná feed-forward sieťou; až potom pridávame komplexitu.

---

## 3. Trénovací proces

| Parameter | Hodnota |
|---|---|
| Loss | `nn.MSELoss()` (mean reduction) |
| Optimizer | Adam, `lr = 1e-3` |
| Batch size | 64 |
| Max epochs | 200 |
| Early stopping | patience = 20 epôch na `val_loss` |
| Device | CUDA (GPU) |

**Prečo MSE:** hladká, derivovateľná, gradient čisto definovaný (`2(ŷ − y)`). Štandard pre regresiu.
**Prečo Adam:** adaptívny learning rate per-parameter, robustný default, rýchla konvergencia.
**Prečo early stopping:** zabráni pretrénovaniu, automaticky obnoví váhy modelu s najnižšou `val_loss`.

**Sledované metriky:**
- **MSE** — minimalizovaná loss (jednotky USD², trénovacia veličina)
- **MAE** — interpretovateľná metrika v USD (priemerná absolútna chyba predikcie)

---

## 4. Evaluácia a interpretácia (baseline pred zmenou architektúry)

> *Snapshot po ~70 epochách. Trénovanie ešte beží, finálne čísla po dobehnutí (alebo early stopping) budú mierne lepšie.*

| Metrika | Train | Val |
|---|---|---|
| MSE | 25 686 635 | 27 766 961 |
| MAE | **4 042 USD** | **4 212 USD** |
| RMSE (= √MSE) | ~5 068 USD | ~5 269 USD |

### Porovnanie s naivným baseline („predpovedaj vždy priemer")

| | Baseline | Náš model | Zlepšenie |
|---|---|---|---|
| MAE | ~30 000 USD | 4 212 USD | **86 %** |
| MSE | ~1.4 mld | 27.8M | **98 %** |

### Sanity-check rozloženia chýb

`RMSE / MAE = 5 269 / 4 212 ≈ **1.25**` → presne to, čo predpovedá teória pre normálne rozdelené reziduály (1.2533). Žiadne extrémne outliers, distribúcia chýb je „zdravá".

### Generalization gap

Train MAE 4 042 vs Val MAE 4 212 → rozdiel **~4 %**. Model **negeneralizuje zle** — pri 13k parametroch a 175k vzorkách žiadny náznak pretrénovania.

### Relatívna chyba

`4 212 / 145 718 ≈ **2.9 %** priemerného platu`.

### Vizualizácie v notebooku

- Krivky train/val MSE a MAE počas trénovania (kontrola konvergencie)
- Scatter plot predikcia vs. skutočnosť (body blízko diagonály = dobrá predikcia)
- Histogram reziduálov (symetrický okolo nuly = nezaujatý model)

---

## 5. Diskusia

**Čo funguje:**
- Model sa demonštrovateľne naučil úlohu (98 % redukcia MSE oproti baseline).
- Žiadne pretrénovanie napriek tomu, že nepoužívame regularizáciu (dropout, weight decay).
- Tréning je stabilný, loss konzistentne klesá.

**Kontext a limitácie:**
- Dataset je s vysokou pravdepodobnosťou **syntetický** (rovnomerné kategórie, čisté rozsahy, žiadne nuly). MAE ~4 200 USD je pravdepodobne blízko **irreducible error floor** — t.j. šumu v dátach, pod ktorý sa žiadny model nedostane.
- V reálnom svete by bola predikcia platov výrazne ťažšia (typicky 10–20 % MAE) kvôli skrytým faktorom (firma, manažér, vyjednávanie, benefity).
- Model neuvažuje **interakcie medzi featúrami** explicitne — feed-forward sieť ich vie aproximovať, ale nie tak efektívne ako napr. gradient boosting alebo wide-and-deep architektúry.

**Vyskúšané experimenty:** detailný záznam v [EXPERIMENTS.md](EXPERIMENTS.md). Stručne:

| # | Variant | Konvergencia | Val MAE |
|---|---|---|---|
| 0 | Baseline | ~150 epôch | 4 174 USD |
| 1 | + štandardizácia `y` | **~20 epôch** | 4 116 USD |

Štandardizácia targetu **~7.5× zrýchlila konvergenciu** pri rovnakej finálnej kvalite — silné nepriame potvrdenie, že baseline bola limitovaná **šumom dát, nie optimalizáciou**.

**Smery, ktoré ešte ideme skúšať:**
- Hlbšia/širšia sieť (`42 → 256 → 128 → 64 → 1`) — kapacitný test
- Feature importance (zero-out analýza) — interpretácia, ktoré featúry sú dominantné

---

## Štruktúra projektu

```
NeuralNetwork/
├── salary_nn.ipynb     # hlavný notebook (dáta → model → tréning → evaluácia)
├── data/               # dataset (sťahovaný automaticky)
├── req.txt             # Python závislosti
└── README.md
```
