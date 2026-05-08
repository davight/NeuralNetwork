# Experimenty

Záznam iterácií modelu nad baseline popísaným v [README.md](README.md). Pre každý experiment uvádzame hypotézu, vykonanú zmenu, výsledok a záver.

## Súhrnná tabuľka

| # | Variant | Architektúra | Konvergencia | Val MAE | Val MSE (USD²) | Pozn. |
|---|---|---|---|---|---|---|
| 0 | Baseline | 42 → 128 → 64 → 1 | ~150 epôch | 4 174 USD | 27.3 M | bez `y` štandardizácie |
| 1 | + y štandardizácia | 42 → 128 → 64 → 1 | **~20 epôch** | **4 116 USD** | 26.7 M | **~7.5× rýchlejšia konvergencia** |

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

## Plán ďalších experimentov

1. **Hlbšia sieť** (`42 → 256 → 128 → 64 → 1`, ~50k parametrov) — kapacitný test. Očakávanie: ak sme na irreducible floor, zlepšenie bude minimálne; ak nie, MAE klesne výraznejšie.
2. **Feature importance** (zero-out analýza) — pre každú featúru zmeráme, o koľko stúpne MAE, keď ju vynulujeme. Interpretačná story do diskusie.
