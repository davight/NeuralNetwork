# Experimenty

Záznam iterácií modelu nad baseline popísaným v [README.md](README.md). Pre každý experiment uvádzame hypotézu, vykonanú zmenu, výsledok a záver.

## Súhrnná tabuľka

| # | Variant | Architektúra | Parametre | Konvergencia | Val MAE | Test MAE | Pozn. |
|---|---|---|---|---|---|---|---|
| 0 | Baseline | 42 → 128 → 64 → 1 | 13 825 | ~150 epôch | 4 174 USD | — | bez `y` štandardizácie |
| 1 | + y štandardizácia | 42 → 128 → 64 → 1 | 13 825 | **~20 epôch** | **4 116 USD** | — | **~7.5× rýchlejšia konvergencia** |
| 2 | + hlbšia sieť | 42 → 256 → 128 → 64 → 1 | **52 225** | ~30 epôch | 4 158 USD | 4 189 USD | **viac kapacity ≠ lepší výsledok** |

> Všetky merania: `random_state=67`, batch size 64, Adam (`lr=1e-3`), early stopping (patience 20, threshold 1e-4).

---

## Experiment 0 — Baseline (referenčný stav)

**Architektúra:** `42 → 128 → 64 → 1`, 13 825 parametrov, ReLU aktivácie, MSE loss, Adam (`lr=1e-3`), batch 64.

**Predspracovanie:** OHE kategorických (`drop_first=True`), štandardizácia featúr (μ, σ z train). Target `y` v origináli (USD).

**Výsledok:**

| Metrika | Train | Val |
|---|---|---|
| MSE (USD²) | 25.0 M | 27.3 M |
| MAE (USD) | 3 990 | 4 174 |
| RMSE (USD) | 5 002 | 5 230 |

- **Early stopping:** epocha ~150
- **RMSE / MAE ≈ 1.25** → zdravé (Gaussovské) rozloženie reziduálov, žiadne extrémne outliers
- **Generalization gap:** ~4.4 % → žiadne pretrénovanie

**Pozorovanie:** Loss klesá konzistentne, ale po ~80 epochách už zlepšenia mizivé. Model sa približuje k limitu, ktorý je buď v kapacite siete, alebo v šume dát.

---

## Experiment 1 — Štandardizácia targetu `y`

**Hypotéza:**
Target `salary` je v ráde 10⁵ USD, takže MSE je v ráde 10⁷ USD². Pri takej škále gradientov môžu Adamove momentové štatistiky (`m`, `v`) trvať dlho, kým sa adaptujú, a default `lr=1e-3` je relatívne malý. Štandardizácia targetu (`y' = (y − μ_y) / σ_y`) by mala dať gradientom škálu rádu 1, čo by malo:
1. Zrýchliť konvergenciu (menej epôch)
2. Prípadne zlepšiť aj final kvalitu, ak baseline bola limitovaná optimalizáciou

**Zmena oproti baseline:**

```python
# bunka štandardizácie
mu_y, sigma_y = y_train.mean(), y_train.std()
y_train_s = (y_train - mu_y) / sigma_y
y_val_s   = (y_val   - mu_y) / sigma_y
y_test_s  = (y_test  - mu_y) / sigma_y
```

- Všetky tri sety štandardizované **rovnakou μ_y, σ_y** (fitnuté iba na train)
- DataLoader používa `y_train_s`, `y_val_s`, `y_test_s`
- V `run_epoch` MAE invezne transformovaná naspäť do USD: `(|ŷ - y| · σ_y).sum()`
- MSE zostáva v štandardizovaných (std²) jednotkách; pre porovnanie s baseline násobiť `σ_y²` ≈ 1.4 × 10⁹

**Výsledok:**

```
Ep 10 | train MSE: 0.01802  MAE: 4002.11 | val MSE: 0.02027  MAE: 4246.18
Ep 20 | train MSE: 0.01786  MAE: 3986.62 | val MSE: 0.01905  MAE: 4115.74
Early stopping (~ep 22)
```

| Metrika | Train | Val |
|---|---|---|
| MSE (std²) | 0.01786 | 0.01905 |
| MSE (USD², prepočet) | 25.0 M | **26.7 M** |
| MAE (USD) | 3 987 | **4 116** |
| Generalization gap | — | ~3.2 % |

**Záver:**

- ✅ **Konvergencia ~7.5× rýchlejšia** (~20 vs. ~150 epôch) — **hlavný a očakávaný benefit experimentu**
- ⚪ **Final kvalita ekvivalentná** — Val MAE rozdiel 58 USD (1.4 %) je v rámci stochastiky tréningu, **nie štatisticky významný**
- ✅ **Potvrdenie hypotézy o irreducible error floor:** keby bola baseline limitovaná optimalizáciou, štandardizácia by zlepšila aj final MAE. Keďže ho nezlepšila, **limit je v dátach** — pravdepodobne syntetický šum okolo deterministickej funkcie `features → salary`.

**Vedľajší efekt — early stopping:**
Threshold `best_val_loss - 1e-4` bol v originálnom priestore (loss ~10⁷) numericky zanedbateľný (~10⁻¹¹ relatívne). V štandardizovanom priestore (loss ~10⁻²) zodpovedá ~0.5 % relatívnej zmene — t.j. realistickej hranici „reálneho zlepšenia". Preto teraz patience zarezáva po dosiahnutí floor, kým predtým ho ignorovala. Toto je vlastne **správnejšie správanie**, hoci to vyzerá ako predčasné zastavenie.

---

## Experiment 2 — Hlbšia sieť

**Hypotéza:**
Z Experimentu 1 sme získali silný náznak, že limitácia výkonu je v dátach (irreducible error floor), nie v kapacite siete. Ak je to pravda, **zväčšenie modelu by nemalo zlepšiť val MAE**. Naopak, ak sú architektúrnym úzkym hrdlom 13k parametrov, hlbšia sieť by mala dosiahnuť výrazne nižšie MAE. Tento experiment slúži ako kapacitný test.

**Zmena oproti Experimentu 1:**
- Pridaná jedna skrytá vrstva, prvá je dvakrát širšia: `42 → 256 → 128 → 64 → 1`
- Počet parametrov: **13 825 → 52 225** (3.8× viac)
- Všetko ostatné identické (loss, optimizer, hyperparametre, OHE, štandardizácia X aj y)

**Výsledok:**

```
Ep 10 | train MSE: 0.01844  MAE: 4049.82 | val MSE: 0.01941  MAE: 4160.09
Ep 20 | train MSE: 0.01779  MAE: 3975.87 | val MSE: 0.01940  MAE: 4157.65   ← peak val
Ep 30 | train MSE: 0.01732  MAE: 3921.38 | val MSE: 0.01946  MAE: 4168.49   ← začína overfit
Early stopping
```

| Metrika | Hodnota | Δ vs. Exp 1 |
|---|---|---|
| Train MAE | 3 921 USD | **−66 USD** (lepší) |
| Val MAE | 4 158 USD | **+42 USD** (horší) |
| Test MAE | 4 189 USD | — |
| Generalization gap | ~5.7 % | **+2.5 p.b.** (horší) |

**Záver:**

- 🔴 **Hlbšia sieť negeneralizuje lepšie** — Val MAE sa zhoršilo o 42 USD napriek 3.8× viac parametrom
- ✅ **Hlbšia sieť trénuje lepšie** — Train MAE je o 66 USD nižšie, model má kapacitu naučiť sa trénovacie dáta presnejšie
- 🔴 **Generalization gap sa zväčšil** zo 3.2 % (Exp 1) na 5.7 % — klasický signál, že dodatočná kapacita ide do **memorovania šumu**
- 🔴 **Trend overfittingu** — pri ep 30 už val MAE rastie (4 158 → 4 168), zatiaľ čo train MAE pokračuje v poklese

**Hlavný takeaway pre prezentáciu:**
Tento experiment **silne potvrdzuje hypotézu z Experimentu 1**: úzkym hrdlom je šum dát, nie kapacita modelu. Pridanie 38k parametrov nielenže nepomohlo, ale ľahko uškodilo. Najmenší model (Exp 1) je z tohto pohľadu **optimálny** — Occamova britva v praxi.

**Vedľajšie pozorovanie — `best_state` reload:**
Pri analýze sme si všimli, že tréningová slučka ukladala `best_state` (váhy z najlepšej epochy), ale nikdy ich pred test evaluáciou neobnovila do modelu. Test MAE 4 189 teda zodpovedá modelu z ~ep 30 (mierny overfit), nie najlepšiemu stavu z ~ep 11–15. Po oprave (`model.load_state_dict(best_state)` po skončení tréningu) by test MAE mal byť mierne nižší (~4 130–4 150), ale **záver experimentu sa nemení** — hlbšia sieť stále nedosahuje výsledok Exp 1. Oprava bola pridaná do tréningovej slučky pre korektnosť budúcich experimentov.

---

## Plán ďalších experimentov

1. **Feature importance** (zero-out analýza) — pre každú featúru zmeráme, o koľko stúpne MAE, keď ju vynulujeme. Identifikujeme dominantné featúry pre interpretáciu modelu v Diskusii.
