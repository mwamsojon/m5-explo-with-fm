# Chronos-2 sur M5 — Rapport de Progression Expérimentale

**Modèle :** `autogluon/chronos-2-small`  
**Tâche :** Prévision de la demande Walmart M5 — 30 490 séries, horizon 28 jours  
**Métrique :** WRMSSE (moyenne géométrique sur 3 points de coupe, sauf mention contraire) ; l'analyse cold-start utilise le RMSSE par série non pondéré  
**Points de coupe :** Printemps 2016-04-24 · Hiver 2016-01-03 · Automne 2015-10-04  
**Date :** Juin 2026

---

## 1. Cadre Expérimental

### 1.1 La compétition M5

M5 est une compétition de prévision de la demande portant sur 30 490 séries temporelles de ventes Walmart, couvrant 3 États (CA, TX, WI), 10 magasins, 3 catégories (FOODS, HOBBIES, HOUSEHOLD) et 7 départements. L'horizon de prévision est de 28 jours.

La métrique d'évaluation est le **WRMSSE** (Weighted Root Mean Scaled Squared Error), calculé sur 12 niveaux d'agrégation — de la SKU individuelle jusqu'au total des ventes. Le poids de chaque SKU est proportionnel à ses ventes historiques : les séries FOODS à fort volume dominent le score (~57 % du WRMSSE total). Une amélioration sur les FOODS se traduit directement par une amélioration globale ; une amélioration sur les séries à faible volume peut être invisible dans la métrique.

### 1.2 Caractéristiques des séries

Les propriétés suivantes conditionnent directement les choix de modélisation :

- **Haute intermittence** : de nombreuses SKUs ont des jours à ventes nulles. Le WRMSSE est dominé par les articles à fort volume, dont les patterns saisonniers sont plus nets.
- **Saisonnalité hebdomadaire marquée** : les jours de distribution des aides SNAP (vendredi en CA, TX, WI) génèrent des pics de ventes systématiques.
- **Effets promotionnels et événementiels** : promotions de prix, jours fériés, événements sportifs et culturels.
- **Longueur de contexte hétérogène** : les articles à fort volume bénéficient d'un historique long ; les articles à faible volume sont mieux capturés avec un contexte court — l'historique distant est du bruit pour les séries intermittentes.

### 1.3 Méthodologie d'évaluation

Toutes les expériences à partir de v9 utilisent la **moyenne géométrique WRMSSE sur 3 points de coupe** (printemps, hiver, automne). Cette approche offre deux avantages : (i) robustesse aux patterns saisonniers spécifiques d'une seule fenêtre, (ii) réduction du sur-apprentissage sur un seul cutoff. Les expériences antérieures (v2–v8) n'utilisaient que le cutoff printemps (2016-04-24) et ne sont **pas directement comparables** aux résultats ultérieurs. Les résultats mono-cutoff sont explicitement signalés.

**Note métrique pour l'analyse cold-start (§11) :** L'analyse cold-start utilise le RMSSE par série non pondéré, pas le WRMSSE. Les poids de revenus dans le WRMSSE suppriment précisément le signal nouveaux produits que l'on cherche à mesurer. Les deux métriques ne sont **pas comparables** — ne pas les mélanger dans un même classement.

---

## 2. Référence Naïve

Une prévision naïve saisonnière (répétition cyclique des 7 dernières valeurs journalières) sert d'ancre pour l'interprétation des améliorations.

| Point de coupe | WRMSSE |
|---|---|
| Printemps 2016-04-24 | 1.4639 |
| Hiver 2016-01-03 | 1.4216 |
| Automne 2015-10-04 | 2.0848 |
| **Géo-mean 3 cutoffs** | **1.6310** |

Le cutoff automne est nettement plus difficile : le stockage pré-fêtes perturbe le cycle hebdomadaire sur lequel la naïve repose. Tous les résultats Chronos-2 dépassent confortablement cette référence. Le ratio d'amélioration entre la meilleure méthode Chronos (0.7969) et la naïve (1.6310) est de l'ordre de 51 %.

---

## 3. Phase 1 — Exploration Zero-Shot (Cutoff Printemps Uniquement)

L'objectif initial était de caractériser la capacité zero-shot brute de Chronos-2 avant tout fine-tuning.

### 3.1 Sensibilité à la Longueur de Contexte (CL)

| Configuration | WRMSSE (printemps) |
|---|---|
| ZS, CL=2, sans covariables | 1.243 |
| ZS, CL=8, + prix | 1.255 |
| ZS, CL=8, + événements | 1.267 |
| ZS, CL=16 | 1.350 |
| ZS, CL=32 | 1.520 |
| ZS, CL=128+ | 1.8–2.1 |

**Résultat :** La longueur de contexte est le paramètre le plus impactant en ZS. Un CL court (2–8) surpasse largement un CL long. Ce contre-intuitif s'explique par la haute intermittence des séries M5 : un historique long introduit du bruit irrélevant sur les périodes de ventes nulles passées. Les covariables (prix, événements) n'apportent pas d'amélioration significative en batches globaux mélangés.

**Note :** CL=1 est numériquement identique à la baseline naïve saisonnière (1.6310). Avec une seule observation, Chronos produit le même repeat hebdomadaire en dernière valeur. Tout run utile nécessite au minimum CL=7.

### 3.2 Interaction Covariables × Cross-Learning (Batches Globaux Mélangés)

En batches globaux (30 490 séries mélangées), l'interaction entre covariables et `cross_learning` est contre-intuitive :

|  | xcl=True | xcl=False |
|--|---|---|
| **Avec covariables** | 1.1765 | **1.0776** |
| **Sans covariables** | **1.2504** | 1.2716 |

Les covariables et le cross-learning sont **substituables, pas complémentaires** en batch global. Quand les deux sont actifs, la cross-attention mélange les représentations de covariables *entre* séries incompatibles (ex. série SNAP Californie qui attend sur une série Texas lors d'une période SNAP CA uniquement), injectant du bruit. Sans covariables, la cross-attention capte un signal utile de similarité de niveau de demande. **Conclusion :** avec des données exogènes riches, désactiver le cross-learning globalement ; l'activer uniquement en batches segmentés par catégorie (voir §6).

### 3.3 HPO ZS par Segment de Demande

Segmenter les 30 490 séries en groupes homogènes et optimiser le CL par groupe produit un gain majeur sans entraînement GPU.

**Segmentation par lissage** (classification ADI/CV²) :

| Rang | CL_Smooth | CL_Erratic | CL_Interm. | CL_Lumpy | Covariables | WRMSSE |
|---|---|---|---|---|---|---|
| 1 | 16 | 4 | 1 | 8 | weekend + prix + snap | **0.9418** |
| 2 | 16 | 2 | 1 | 8 | weekend + prix + snap | 0.9729 |

**Segmentation par quartile de poids de ventes** (Low/Med-Low/Med-High/High) :

| Rang | CL_Low | CL_MedLow | CL_MedHigh | CL_High | Covariables | WRMSSE |
|---|---|---|---|---|---|---|
| 1 | 16 | 4 | 1 | 16 | weekend + prix + snap | **0.9082** |
| 2 | 8 | 4 | 1 | 256 | weekend + prix + snap | 0.9137 |

**Résultat :** L'attribution de longueurs de contexte différentes par segment améliore le WRMSSE de ~27 % (1.243 → 0.908) sans aucun entraînement. Les articles à fort poids bénéficient d'un contexte long (CL=16–256) ; les articles à poids moyen-fort n'ont besoin que d'un contexte minimal (CL=1). Ce résultat établit le **ZS sensible aux segments** comme base pour toutes les expériences suivantes.

---

## 4. Phase 2 — Fine-Tuning par Segment de Poids (v2–v11)

> **Note :** Tous les résultats de fine-tuning dans cette phase utilisent le cutoff printemps sauf mention contraire. Les runs LoRA des tiers High/Med-High (v9+) peuvent être affectés par le bug PeftModel identifié ultérieurement (voir §7.1). Le fine-tuning en mode full (tier Med-Low dans v9) n'est pas concerné.

### 4.1 Fine-Tuning Uniforme (v2) — Échec Initial

Premier essai de fine-tuning avec des hyperparamètres identiques pour tous les segments.

**WRMSSE (printemps) : 1.017** — inférieur au baseline ZS par segment (0.908).

**Explication :** Un fine-tuning uniforme détériore les performances. Le même lr/steps appliqué à toutes les séries sur-apprend les articles à faible fréquence (peu d'exemples, signal pauvre) tout en sous-adaptant les articles à fort volume. Le fine-tuning doit être **spécifique au segment**.

### 4.2 Fine-Tuning par Segment (v6) — Meilleur Résultat Mono-Cutoff

HPO complet par segment (CL, ft_steps, mode, lr, bs, covariables) sur 216 essais Optuna.

| Segment | CL | Steps | Mode | LR | Covariables |
|---|---|---|---|---|---|
| Low | 8 | ZS | — | — | weekend + événements |
| Med-Low | 1 | 200 | full | 5e-5 | événements + snap |
| Med-High | 4 | 200 | lora | 1e-4 | weekend |
| High | 16 | 100 | lora | 1e-2 | weekend + prix + snap |

**WRMSSE (printemps) : 0.8364**

**Résultats clés :**
- Le tier Low reste en ZS — le fine-tuning n'apporte rien aux séries éparses et bruitées.
- Le tier High nécessite un fine-tuning agressif (LR élevé 1e-2, covariables prix + snap + weekend).
- Les 4 meilleurs essais se valent à 0.8364, ne différant que par le choix de covariables pour Med-Low — sensibilité faible.

*Note : La partie LoRA (High/Med-High) peut être affectée par le bug PeftModel décrit en §7.1, ce qui fragilise l'interprétation de ce résultat.*

### 4.3 Réencodage des Covariables (v7/v8) — Régression

Passage des chaînes de covariables unifiées à des flags booléens individuels par covariable et par segment ; extension des steps à 2 000.

**Résultat (v8, 2 cutoffs Optuna) :** geo = **0.9082** — régression significative par rapport à v6.

**Explication :** L'explosion de l'espace de recherche (16 paramètres booléens supplémentaires) a ralenti la convergence du TPE. La chaîne unifiée de v6 constituait un biais inductif plus fort. Davantage de steps de fine-tuning n'a pas compensé.

### 4.4 HPO Tier — Segmentation par Poids de Ventes (v9–v11)

HPO complet (CL, mode/lr/steps/bs de FT, portée du fine-tuning — quels tiers s'entraînent ensemble) sur 3 cutoffs.

| Version | Essais | WRMSSE geo 3-cutoffs |
|---|---|---|
| v9 | 362 | 0.8947 |
| v10 | 150 | 0.8536 |
| v11 | 100 | **0.8544** |

**Configuration v11, essai #11 :** utilisée comme *ensemble de référence par tier* dans toutes les fusions ultérieures.

**Ensemble tier seul (sans fusion Chronos) :**

| Cutoff | WRMSSE |
|---|---|
| Printemps 2016-04-24 | 0.8467 |
| Hiver 2016-01-03 | 0.8712 |
| Automne 2015-10-04 | 0.7420 |
| **geo** | **0.8180** |

**Résultat :** La progression v9→v11 (0.8947→0.8544) illustre l'importance de l'affinement progressif des hyperparamètres. L'ensemble v11 devient le signal de référence dominant pour toute l'étude — sa suppression dégrade fortement les résultats (voir §6.3).

---

## 5. Phase 3 — HPO ZS par Département (v12–v14, 3 Cutoffs)

### 5.1 Changement d'Axe de Segmentation : du Tier au Département

L'hypothèse centrale est que les 7 **départements produits** (FOODS_1/2/3, HOBBIES_1/2, HOUSEHOLD_1/2) constituent un axe de segmentation plus naturel que les quartiles de volume pour la prévision univariée. Les séries d'un même département partagent la même catégorie produit, la même saisonnalité et la même sensibilité aux prix. Tous les runs de cette phase utilisent `cross_learning=False`.

### 5.2 HPO CL par Département (v12/v13)

| Version | Essais | CL optimaux | EW geo |
|---|---|---|---|
| v12 | 100 | F1=28, F2=7, F3=28, HB1=1, HB2=14, HH1=28, HH2=1 | 0.8702 |
| v13 | 150 | F1=28, F2=7, F3=28, HB1=1, HB2=7, HH1=28, HH2=7 | **0.8672** |

**Schéma des CL optimaux :** La longueur de contexte optimale reflète la régularité de la demande — les FOODS à fort volume (F1=28, F3=28) et HOUSEHOLD_1 (HH1=28) bénéficient d'un historique long, tandis que HOBBIES_1 (HB1=1) — très intermittent — est mieux prédit avec uniquement le dernier jour de contexte.

Fusion 50% ensemble tier + 50% ZS département avec poids OLS :

| Méthode | geo-mean |
|---|---|
| EW (50% tier + 50% ZS dept) | 0.8672 |
| OLS global (α unique) | 0.8128 |
| **OLS par département** | **0.8066** |

**Poids OLS par département (v13)** — α élevé = davantage tier, α faible = davantage Chronos ZS :

| Département | α | Part Chronos |
|---|---|---|
| FOODS_1 | 0.988 | 1.2 % |
| FOODS_2 | 1.000 | 0.0 % |
| FOODS_3 | 0.939 | 6.1 % |
| HOBBIES_1 | 0.779 | 22.1 % |
| HOBBIES_2 | 1.000 | 0.0 % |
| HOUSEHOLD_1 | 0.759 | 24.1 % |
| HOUSEHOLD_2 | 0.995 | 0.5 % |

**Résultat :** L'OLS par département améliore le passage de 0.8672 à 0.8066. Les FOODS sont quasi-entièrement portés par l'ensemble tier (α≈1) ; HOUSEHOLD_1 et HOBBIES_1 tirent le plus de valeur du signal Chronos ZS.

### 5.3 Chronos-2 en Mode Autonome — Base Model vs Small (Gap 2 / v17)

Deux expériences isolent la performance de Chronos sans aucun ensemble tier, pour quantifier sa contribution intrinsèque :

| Modèle | xcl | EW geo | OLS geo |
|---|---|---|---|
| chronos-2-**small**, CLs v13 (Gap 2) | False | 1.0739 | 0.8105 |
| chronos-2-**BASE**, CLs v13 (v17) | False | **1.1963** ⚠️ | — |

⚠️ **Correction (17 juin) :** La version précédente indiquait 0.9123 EW pour BASE standalone — c'était une erreur. 0.9123 était le résultat *fusionné* (50% tier v11 + 50% BASE ZS), pas le standalone. La colonne OLS du tableau ci-dessus (0.8105) correspondait également à un résultat fusionné-tier, pas standalone.

**Observation corrigée :** Le modèle small xcl=True (v20d) est le **meilleur ZS standalone** avec 1.0325 EW / 0.9829 OLS. Le modèle BASE xcl=False est *pire* en standalone (1.1963 EW). L'avantage du BASE n'apparaît que lorsqu'il est fusionné avec l'ensemble tier.

### 5.4 HPO ZS par Magasin (v14)

10 modèles séparés (CA_1–4, TX_1–3, WI_1–3), 3 cutoffs.

| Méthode | geo-mean |
|---|---|
| v14 EW (50% tier + 50% ZS magasin) | 0.8667 |
| v14 OLS par magasin | 0.8132 |

**Résultat :** La segmentation par magasin (0.8132) est inférieure à la segmentation par département (0.8066). La géographie n'encode pas la dynamique de la demande — CA, TX et WI vendent les mêmes catégories de produits avec des patterns similaires.

---

## 6. Phase 4 — Découverte du Cross-Learning (v20)

### 6.1 La Découverte Centrale et l'Effet de la Composition des Batches

Chronos-2 propose un paramètre `cross_learning=True` dans `predict_quantiles()`, qui active le **mécanisme d'attention de groupe** (*group attention*). Toutes les expériences précédentes (v12/v13) avaient implicitement utilisé `cross_learning=False`, laissant ce mécanisme inactivé.

L'effet du cross-learning dépend **critiquement** de la composition des batches :

| xcl | Composition du batch | Résultat standalone ZS |
|---|---|---|
| True | Tous départements mélangés (global) | 1.1680 — se dégrade |
| False | Tous départements mélangés (global) | **1.0776** — neutre |
| False | Batches par département | 1.0739 — neutre |
| **True** | **Batches par département** | **1.0325 — meilleur ZS** |

Le mécanisme : dans un batch de département FOODS_1, 100 séries partagent la même éligibilité SNAP, les mêmes événements promotionnels et des patterns de demande hebdomadaires similaires. La cross-attention extrait un signal de groupe genuinement utile. En batch global mélangé, la cross-attention voit des patterns incompatibles entre catégories et introduit du bruit. **La cohérence sémantique des batches est un prérequis pour que le cross-learning aide.**

### 6.2 HPO CL avec Cross-Learning (v20)

Même structure que v13, avec `cross_learning=True` et batch_size=100. 150 essais Optuna.

**CL optimaux par département (v20) :**

| F1 | F2 | F3 | HB1 | HB2 | HH1 | HH2 |
|---|---|---|---|---|---|---|
| 28 | 7 | 28 | 1 | 7 | 28 | **1** |

À noter : HH2 passe de CL=7 (v13) à CL=1 (v20). Le CL optimal peut changer lorsque le cross-learning modifie la façon dont le modèle utilise le contexte historique.

| Méthode | geo-mean |
|---|---|
| v13 EW (xcl=False) | 0.8672 |
| **v20 EW (xcl=True)** | **0.8417** |
| Δ | −0.0255 |

**L'activation seule du cross_learning apporte +0.025 geo sans aucun autre changement.** C'est le gain unitaire le plus important de toute l'étude.

### 6.3 Fusion OLS v20

Application du même cadre OLS par département (α × tier + (1−α) × ZS dept) aux prévisions v20 :

| Méthode | Printemps | Hiver | Automne | geo-mean |
|---|---|---|---|---|
| v20 EW | — | — | — | 0.8417 |
| **v20 OLS** | **0.8272** | **0.8726** | **0.7012** | **0.7969** |

**Poids OLS par département (v20) :**

| Département | α | Part Chronos |
|---|---|---|
| FOODS_1 | 0.961 | 3.9 % |
| FOODS_2 | 1.000 | 0.0 % |
| FOODS_3 | 0.900 | 10.0 % |
| HOBBIES_1 | 0.740 | 26.0 % |
| HOBBIES_2 | 1.000 | 0.0 % |
| HOUSEHOLD_1 | 0.644 | **35.6 %** |
| HOUSEHOLD_2 | 0.853 | 14.7 % |

**Le v20 OLS (geo = 0.7969) est le meilleur résultat Chronos, franchissant la barre des 0.80.**

| Comparaison | geo-mean |
|---|---|
| v13 OLS (xcl=False) | 0.8066 |
| **v20 OLS (xcl=True)** | **0.7969** |
| Δ vs v13 OLS | −0.0097 |
| Δ vs v20 EW | −0.0448 |

### 6.4 Décomposition de la Valeur Ajoutée de l'Ensemble Tier

Pour quantifier la contribution respective du cross-learning, de l'OLS et de l'ensemble tier :

| Méthode | geo-mean |
|---|---|
| ZS dept EW (v20 CLs, xcl=True, sans tier) | 1.0325 |
| ZS dept OLS (mise à l'échelle par dept, sans tier) | 0.9829 |
| **v20 OLS (avec ensemble tier)** | **0.7969** |
| Δ tier (0.9829→0.7969) | **−0.186** |

**L'ensemble tier est le signal porteur dominant : sa suppression dégrade le résultat de ~0.19 geo.**

Poids de mise à l'échelle Chronos-only (sans tier) :

| FOODS_1 | FOODS_2 | FOODS_3 | HOBBIES_1 | HOBBIES_2 | HH_1 | HH_2 |
|---|---|---|---|---|---|---|
| **1.593** | 1.181 | 1.162 | 0.919 | 1.116 | 1.127 | 0.972 |

**Biais de sous-estimation systématique des FOODS :** Chronos sous-estime les ventes FOODS lorsque le cross-learning est limité aux pairs du même département — tous les pairs partagent le même biais. Un facteur de 1.59× est nécessaire pour FOODS_1. L'ensemble tier, calibré indépendamment, corrige ce biais naturellement — ce qui explique pourquoi il domine pour tous les départements FOODS (α≈1.0).

---

## 7. Phase 5 — Investigation du Fine-Tuning (v15–v24)

### 7.1 Bug PeftModel — Cause Racine et Correction

Tous les runs LoRA de v15 à v22 utilisaient `Chronos2Pipeline.fit()` suivi d'une inférence en mémoire. Un bug a été identifié : `fit()` retourne un objet `PeftModel` dont la méthode `forward()` est incompatible avec le chemin d'inférence quantile de Chronos-2, produisant des sorties incorrectes.

**Correction (appliquée dans pipeline.py, lignes 351–361) :** Appel à `model.merge_and_unload()` avant de retourner le pipeline en mémoire.

```python
# Chemin brisé — poids de l'adaptateur ignorés silencieusement
pipeline = MeanScaleUniformBins(model)

# Chemin correct — adaptateur fusionné avant wrapping
pipeline = MeanScaleUniformBins(model.merge_and_unload())
```

**Périmètre affecté :**
- v15–v22 : tous les runs LoRA FT (résultats invalides)
- v9 Med-Low (full FT, non LoRA) : non affecté
- v11 High (LoRA, 50 steps) : antérieur à la période concernée, probablement non affecté

Ce bug a rendu les mois d'expérimentation FT entre v15 et v22 non interprétables, et a motivé la réévaluation complète avec le pipeline corrigé.

### 7.2 Courbe WRMSSE vs Nombre de Steps (FT Global)

Avant de relancer des HPO coûteux, une courbe de steps a été calculée sur le cutoff printemps avec un FT global (toutes les 30 490 séries, sans segmentation) :

| Steps | WRMSSE brut | Fusion EW (50% tier + 50% FT) |
|---|---|---|
| 0 (ZS) | 1.0630 | **0.8623** |
| 5 | 1.0372 | **0.8581** ✓ |
| 10 | 1.0491 | 0.8642 |
| 25 | 1.0810 | 0.8833 |
| 50 | 1.1644 | 0.9259 |
| 100 | 1.3446 | 1.0116 |
| 200 | 1.5492 | 1.1029 |
| 500 | 1.6281 | 1.1321 |
| 1 000 | 1.6398 | 1.1277 |

**Résultat : Oubli catastrophique confirmé.** Le FT améliore marginalement à 5 steps (EW 0.8623 → 0.8581, Δ=+0.004), puis se dégrade de façon monotone. À 10 steps, la performance est déjà inférieure au ZS. À 100 steps, le WRMSSE brut a presque doublé. Les mises à jour gradient écrasent la représentation cross-learning pré-entraînée plus vite que l'adaptation spécifique à la tâche ne se construit.

*Limite : La courbe de steps a été calculée sur le cutoff printemps uniquement. L'amélioration marginale à 5 steps n'a pas été validée sur les cutoffs hiver et automne.*

### 7.3 FT Brut Chronos après Correction du Bug (v23, v24a)

Après correction du bug PeftModel, le FT a été relancé en utilisant directement l'API Chronos :

| Run | Configuration | geo fusion EW |
|---|---|---|
| v23 | FT par département, CLs v20, 200 steps | 0.9509 |
| v24a | FT global (30k séries), 200 steps | 0.9461 |

Les deux résultats sont nettement inférieurs au baseline ZS EW (0.8417), cohérents avec l'oubli catastrophique observé dans la courbe de steps.

**Conclusion : Le fine-tuning n'est pas une voie viable d'amélioration dans l'espace d'hyperparamètres exploré. La capacité ZS pré-entraînée du modèle, combinée au cross-learning, surpasse systématiquement les tentatives de fine-tuning.**

---

## 8. Phase 6 — Mélange Raisonné : ZS Tier vs ZS Département (v21)

### 8.1 Motivation

Le résultat v20 OLS (0.7969) repose sur l'ensemble tier v9/v11 comme signal dominant — un ensemble calibré extérieurement, construit avec un FT minimal. On peut légitimement se demander : est-il possible de remplacer ce signal externe par un signal Chronos ZS pur, en changeant simplement l'axe de groupement pour le cross-learning ?

**Hypothèse :** Le ZS par département capture la similarité produit (cross-learning intra-FOODS, intra-HOBBIES, etc.). Le ZS par tier de volume capture la similarité de niveau de demande (cross-learning entre séries de tous départements ayant la même intensité de ventes). Ces deux axes sont complémentaires : une série FOODS à fort volume pourrait bénéficier de peers non-FOODS à fort volume pour corriger son biais de sous-estimation.

### 8.2 Résultats du ZS par Tier

Inférence ZS sur les groupes de quartiles de poids (Low/Med-Low/Med-High/High, ~7 620 séries chacun) avec `cross_learning=True` :

**Sélection du CL par tier** (CL ∈ {7, 14, 28}) :

| Tier | CL=7 | CL=14 | CL=28 | Meilleur CL |
|---|---|---|---|---|
| Low | 1.1784 | 1.1657 | 1.1572 | **28** |
| Med-Low | 1.1572 | 1.1487 | 1.1562 | **14** |
| Med-High | 1.1487 | 1.1580 | 1.1652 | **7** |
| High | 1.1487 | 1.1299 | 1.0610 | **28** |

| ZS Tier EW (meilleurs CLs) | Printemps | Hiver | Automne | geo |
|---|---|---|---|---|
| | 1.0247 | 1.2800 | 0.8784 | **1.0483** |

Le ZS par tier (1.0483) est moins performant que le ZS par département (1.0325 EW). Le groupement par département encode une structure cross-série plus utile prédictiblement.

### 8.3 Fusion OLS : ZS Dept + ZS Tier (par paire dept × tier)

Chaque série appartient à exactement un département ET un tier → 28 combinaisons (dept, tier). L'OLS trouve α ∈ [0,1] par paire : `prévision = α × tier_ZS + (1−α) × dept_ZS`.

**Résultat final :**

| Printemps | Hiver | Automne | geo-mean |
|---|---|---|---|
| 0.9139 | 1.1180 | 0.8065 | **0.9375** |

**Matrice des α par paire (dept, tier)** [0 = tout dept_ZS ; 1 = tout tier_ZS] :

| Département | Low | Med-Low | Med-High | High |
|---|---|---|---|---|
| FOODS_1 | 0.33 | **1.00** | **1.00** | **1.00** |
| FOODS_2 | 0.00 | 0.00 | 0.67 | 0.54 |
| FOODS_3 | 0.33 | **1.00** | **1.00** | **1.00** |
| HOBBIES_1 | 0.00 | 0.11 | 0.09 | 0.29 |
| HOBBIES_2 | 0.00 | 0.00 | **1.00** | 0.32 |
| HOUSEHOLD_1 | 0.00 | **1.00** | **1.00** | 0.67 |
| HOUSEHOLD_2 | 0.00 | 0.00 | 0.13 | 0.11 |

**Patterns observés :**
- FOODS et HOUSEHOLD préfèrent le tier ZS à partir de Med-Low : leurs peers de même volume (non-FOODS) fournissent une meilleure calibration d'échelle.
- HOBBIES préfère fortement le dept ZS quel que soit le tier : la similarité catégorie-produit est plus importante que la similarité de volume pour les articles à demande sporadique.
- Les séries Low préfèrent systématiquement le dept ZS (α≈0) : les articles peu vendus sont mieux capturés par leurs pairs de même catégorie.

| Méthode | geo-mean |
|---|---|
| ZS Dept EW (v20, sans tier) | 1.0325 |
| ZS Dept OLS (mise à l'échelle) | 0.9829 |
| ZS Tier EW (v21) | 1.0483 |
| **ZS Tier + Dept OLS (v21)** | **0.9375** |
| v20 OLS (ensemble tier ext. + ZS dept) | **0.7969** |

**Conclusion :** La fusion tier ZS + dept ZS améliore modestement le résultat dept-seul (0.9829 → 0.9375, Δ=0.045). Cependant, l'écart avec le v20 OLS reste important (~0.14 geo). L'ensemble tier v9/v11 ne peut pas être remplacé par un groupement ZS pur : son avantage réside dans la **calibration absolue d'échelle** résultant du FT minimal à 50–200 steps.

---

## 9. Phase 7 — Benchmark LightGBM Global

Pour établir un benchmark alternatif rigoureux, un modèle LightGBM global a été entraîné sur les mêmes 30 490 séries M5.

### 9.1 Ingénierie des Features

**37 features au total :**

- **Lags (6) :** `lag_28`, `lag_29`, `lag_30`, `lag_31`, `lag_35`, `lag_42` — capture les cycles de demande hebdomadaires et bi-hebdomadaires. Note : `lag_28` est structurellement indisponible pour les séries avec moins de 28 jours d'historique actif (voir §11).
- **Statistiques rolling (10) :** moyenne et écart-type sur fenêtres de 7, 14, 30, 60 et 180 jours, toutes ancrées au lag-28 pour éviter la fuite d'information.
- **Features prix (3) :** prix de vente, variation de prix, prix normalisé.
- **Indicateurs SNAP (3) :** CA, TX, WI — les jours SNAP génèrent des pics systématiques le vendredi.
- **Features calendaires (4) :** indicateurs week-end, fins de mois, type d'événement, encodage des événements.
- **Moyennes target-encodées (3) :** par magasin, par département, par état — calculées sur l'historique d'entraînement avant le fit du modèle.

### 9.2 Entraînement et HPO

- **Objectif :** perte tweedie (puissance de variance optimisée via Optuna) pour les données de comptage intermittentes
- **HPO :** 19 trials Optuna (échantillonneur TPE) sur `num_leaves`, `min_data_in_leaf`, `learning_rate`, `feature_fraction`, `bagging_fraction`, `lambda_l1`, `lambda_l2`, `tweedie_variance_power`
- **Entraînement :** gradient boosting avec early stopping à 800 rounds ; validé croisement sur 3 cutoffs de calibration

### 9.3 Résultats

| Cutoff | WRMSSE LightGBM |
|---|---|
| Printemps 2016-04-24 | 0.6178 |
| Hiver 2016-01-03 | 0.6406 |
| Automne 2015-10-04 | 0.6016 |
| **Géo-mean 3 cutoffs** | **0.6198** |

**LightGBM bat Chronos v20 OLS (0.7969) de −22 %.** C'est un écart substantiel, pas du bruit de mesure.

**Mise en contexte :** Le pipeline à 37 features a requis une expertise domaine (quelles fenêtres de lag sont pertinentes pour les cycles retail hebdomadaires), une infrastructure de feature store, un HPO sur 19 trials et un débogage itératif. Le pipeline Chronos a requis un sweep de 4 longueurs de contexte, l'activation d'un booléen (`cross_learning=True`) et l'ajustement de 7 régressions OLS. Les deux atteignent une précision de niveau production avec des profils d'investissement très différents.

---

## 10. Phase 8 — Blend OLS LightGBM × Chronos

### 10.1 Dispositif du Blend

Avec deux modèles complémentaires (LGBM et Chronos v20 OLS), la question naturelle est de savoir si leur combinaison capture une diversité additionnelle. Un blend OLS scalaire LOO-3 a été ajusté : `prévision = β × LGBM + (1−β) × Chronos`, optimisé par leave-one-cutoff-out par département.

### 10.2 Résultats

| Méthode | Printemps | Hiver | Automne | geo-mean |
|---|---|---|---|---|
| LightGBM standalone | 0.6178 | 0.6406 | 0.6016 | **0.6198** |
| Chronos v20 OLS | 0.8272 | 0.8726 | 0.7012 | 0.7969 |
| **Blend LGBM × Chronos OLS** | — | — | — | **0.6254** |

**Le blend (0.6254) est *pire* que LightGBM standalone (0.6198).** Le poids optimal LOO-3 à l'Automne converge vers β=1.0 (LGBM pur) ; à l'Hiver et au Printemps, β≈0.85 (15% de Chronos dilue les performances).

**Résultat :** Sur les séries à historique complet, Chronos n'apporte aucune diversité orthogonale à LightGBM sur ces trois cutoffs. Là où Chronos est bon, LGBM est déjà meilleur. Le blend nuit en moyenne.

**Implication :** La valeur propre de Chronos-2 face à LightGBM n'est pas sur les séries à historique complet — elle est sur les nouveaux produits avec un historique insuffisant pour que `lag_28` existe (voir §11).

---

## 11. Phase 9 — Segmentation Naturelle Cold-Start

### 11.1 Motivation et Métrique

Les phases précédentes utilisaient le WRMSSE, qui pondère les séries par contribution au chiffre d'affaires. Ce mécanisme supprime précisément le signal nouveaux produits : un SKU lancé avec 10 jours d'historique a un poids quasi-nul dans le WRMSSE et est invisible dans la métrique agrégée.

Pour mesurer la performance cold-start directement, nous sommes passés au **RMSSE par série non pondéré** et avons groupé les séries par **longueur d'historique actif** (jours depuis la première vente non nulle à chaque cutoff).

Signal Chronos utilisé : ZS pur dept xcl=True (sans ensemble tier, sans OLS) — exactement ce qu'un praticien déploierait sur un nouveau produit sans aucune donnée d'entraînement disponible.

### 11.2 Conception des Buckets d'Historique

Les seuils des buckets sont ancrés aux prérequis des features LGBM :
- `lag_28` : nécessite ≥28 jours (feature la plus prédictive)
- `lag_42` : nécessite ≥42 jours
- `roll_mean_180_l28` : nécessite ≥208 jours
- Un cycle saisonnier annuel complet : ≥365 jours

### 11.3 Résultats — Agrégés sur 3 Cutoffs (géo-mean des médianes par cutoff)

| Historique Actif | n (Automne) | RMSSE LGBM | RMSSE Chronos | Gagnant | Chronos gagne % |
|---|---|---|---|---|---|
| **< 28 jours** | 32 | 1.059 | **1.019** | **Chronos** | **59 %** |
| 28 – 90 jours | 57 | **0.812** | 0.847 | LGBM | 40 % |
| 90 – 365 jours | 1 274 | **0.735** | 0.871 | LGBM | 20 % |
| 365 – 730 jours | 3 642 | **0.712** | 0.836 | LGBM | 20 % |
| 730+ jours | 25 418 | **0.664** | 0.772 | LGBM | 23 % |

*Métrique : géo-mean des RMSSE médians par série et par cutoff (non pondéré). Non comparable aux colonnes WRMSSE des sections précédentes.*

### 11.4 Détail par Cutoff pour le Bucket <28j

| Cutoff | n | LGBM médiane | Chronos médiane | Chronos gagne % |
|---|---|---|---|---|
| Automne 2015-10-04 | 32 | 1.285 | **1.189** | **72 %** |
| Hiver 2016-01-03 | 11 | 0.873 | 0.874 | 45 % (égalité) |
| Printemps 2016-04-24 | 0 | — | — | — |

L'automne (octobre, pré-fêtes) capture le plus de lancements de produits. Chronos gagne 72 % des séries avec <28j d'historique à ce cutoff. L'hiver est une quasi-égalité (11 séries). Le printemps n'a aucune série dans ce bucket en avril 2016 — tous les produits lancés avant ce cutoff ont franchi le seuil des 28 jours.

### 11.5 Le Seuil des 28 Jours — Explication Structurelle (Argument Principal)

Le seuil à 28 jours n'est pas arbitraire. Il coïncide exactement avec l'historique minimum requis pour que `lag_28` soit disponible — la feature de lag la plus prédictive de LGBM, qui capture la demande du même jour de la semaine quatre semaines auparavant. En dessous de 28 jours :
- `lag_28` est un NaN **structurel** — pas une estimation bruitée, pas une valeur manquante imputable, mais une feature qui n'existe tout simplement pas
- LGBM route ces séries vers sa branche NaN, entraînée pour gérer des ruptures de stock temporaires et des gaps d'intermittence — pas de véritables nouveaux produits
- Chronos-2 n'a pas un tel minimum : avec 7 observations (une semaine), il capture le cycle hebdomadaire ; avec 1 observation, il produit déjà une prévision

Cet argument est **indépendant de la taille de l'échantillon** et tient par construction. C'est le principal fondement de la règle des 28 jours — les données empiriques (§11.6) l'illustrent mais n'en sont pas la preuve fondatrice.

À 28 jours et au-delà, `lag_28` devient disponible, l'avantage de LGBM croît de façon monotone, et le taux de victoire de Chronos tombe à 20–40 %.

### 11.6 Solidité de l'Évidence Empirique — Forces et Limites

Les résultats RMSSE par bucket confirment l'argument structurel de façon directionnelle, mais la base empirique pour le bucket <28j est mince :

| Cutoff | n (<28j) | Taux de victoire Chronos | Interprétation |
|---|---|---|---|
| Automne 2015-10-04 | 32 | **72 %** | Directionnel — Chronos gagne |
| Hiver 2016-01-03 | 11 | 45 % | **Égalité — ne confirme pas l'Automne** |
| Printemps 2016-04-24 | **0** | — | Aucune série cold-start dans ce bucket |

Le résultat cold-start repose effectivement sur **un seul cutoff** (Automne) avec **32 séries**. L'Hiver (n=11) montre une quasi-égalité — il contredit plutôt qu'il ne confirme le résultat d'Automne. Le Printemps ne contribue rien au bucket <28j : en avril 2016, tous les produits M5 avaient dépassé le seuil des 28 jours. Avec n=32, l'intervalle de confiance sur "59–72 % de taux de victoire Chronos" est suffisamment large pour inclure la quasi-parité.

**Lacunes de couverture supplémentaires :**
- Les trois cutoffs se situent entre Automne 2015 et Printemps 2016. Aucun point d'évaluation estival (juillet–août) n'existe — une limite connue de la conception du benchmark M5.
- Les poids OLS par département α sont ajustés sur 2 observations d'entraînement par fold LOO-3. FOODS_2 α=1.000 (zéro contribution Chronos à chaque cutoff) peut refléter un signal réel ou un artefact d'ajustement sur ces folds aussi minces.

**Conséquence pour le cadrage de l'article :** Commencer par l'argument structurel `lag_28`. Présenter le résultat empirique Automne (n=32, 72 % CW) comme illustration. Ne pas mettre en avant le taux de victoire en première ligne — il provient d'un seul cutoff et l'égalité à l'Hiver affaiblit la claim si citée sans contexte.

### 11.7 Règle de Déploiement Pratique

**Déployer Chronos-2 ZS pour tout SKU avec moins de 28 jours d'historique de ventes actif. Passer à LightGBM (ou au modèle de production) dès que `lag_28` est disponible.**

Cette règle est mécanistiquement justifiée par l'argument de NaN structurel. La confirmation empirique est directionnelle (n=43 sur 2 cutoffs) ; les praticiens souhaitant l'adopter devraient la valider sur leurs propres données de lancement de produits avant de retenir 28 jours comme seuil fixe — il peut varier selon la catégorie ou le contexte retail.

---

## 12. Tableau Récapitulatif — Tous les Résultats Clés

| Expérience | Méthode | Éval | geo-mean |
|---|---|---|---|
| Référence naïve | Naïve saisonnière | 3-cutoffs | 1.6310 |
| ZS v1 (CL=2, sans cov) | Chronos ZS global | Printemps seul | 1.243 |
| ZS HPO par poids de ventes | 4 tiers, CL/cov par segment | Printemps seul | 0.908 |
| FT v6 par segment | 4 tiers, CL+FT par segment | Printemps seul | 0.836 † |
| v9 HPO tier | ZS+FT par tier | 3-cutoffs | 0.8947 |
| v10 HPO tier | ZS+FT par tier | 3-cutoffs | 0.8536 |
| v11 HPO tier | ZS+FT par tier | 3-cutoffs | 0.8544 |
| Ensemble tier seul | Matrices W v11 | 3-cutoffs | 0.8180 |
| v12 ZS dept EW | 7 depts, xcl=False | 3-cutoffs | 0.8702 |
| **v13 ZS dept + tier OLS** | 7 depts, xcl=False | 3-cutoffs | **0.8066** |
| v14 ZS magasin OLS | 10 magasins, xcl=False | 3-cutoffs | 0.8132 |
| v13 small seul (sans tier) | ZS dept, xcl=False | 3-cutoffs | 1.0739 |
| **v17 BASE seul (sans tier)** | ZS dept, xcl=False | 3-cutoffs | **1.1963** ⚠️ |
| v20 ZS dept EW | xcl=True, CLs optimaux | 3-cutoffs | 0.8417 |
| **v20 ZS dept + tier OLS** | xcl=True, α par dept | 3-cutoffs | **0.7969** ★ |
| Chronos OLS seul (sans tier) | ZS dept, xcl=True | 3-cutoffs | 0.9829 |
| v23 FT (bug corrigé) | LoRA par dept, 200 steps | 3-cutoffs | 0.9509 |
| v24a FT global | LoRA toutes séries, 200 steps | 3-cutoffs | 0.9461 |
| v21 ZS tier EW | Groupes tier, xcl=True | 3-cutoffs | 1.0483 |
| v21 ZS tier+dept OLS | α par (dept, tier) | 3-cutoffs | 0.9375 |
| **LightGBM global** | **37 features, HPO tweedie** | **3-cutoffs** | **0.6198** |
| Blend LGBM × Chronos OLS | LOO-3 scalaire β par dept | 3-cutoffs | 0.6254 |

*† Printemps uniquement ; composante LoRA potentiellement affectée par le bug PeftModel*  
*★ Meilleur résultat Chronos*

**Résultats cold-start (RMSSE par série — métrique séparée, non comparable au WRMSSE ci-dessus) :**

| Bucket d'historique actif | LGBM médiane RMSSE | Chronos médiane RMSSE | Gagnant |
|---|---|---|---|
| < 28 jours | 1.059 | **1.019** | **Chronos** |
| 28 – 90 jours | **0.812** | 0.847 | LGBM |
| 730+ jours | **0.664** | 0.772 | LGBM |

---

## 13. Synthèse des Résultats

**R1 — Le cross-learning est le levier principal (+0.025 geo EW)**

L'activation de `cross_learning=True` dans `predict_quantiles()` seule améliore le EW de 0.8672 à 0.8417 — le gain unitaire le plus important de toute l'étude. Ce mécanisme active l'attention de groupe de Chronos-2. Toutes les expériences antérieures (v12/v13) avaient laissé ce mécanisme inactif. Nécessite des batches segmentés par département — les batches globaux avec covariables dégradent les performances (1.0776 → 1.1765 avec xcl=True global).

**R2 — L'OLS par département ajoute +0.045 sur l'EW**

Le passage d'une pondération 50/50 fixe à un α optimisé par LOO-3 par département améliore 0.8417 → 0.7969. L'OLS corrige deux biais structurels : (i) les FOODS bénéficient davantage de l'ensemble tier (signal externement calibré) ; (ii) HOUSEHOLD_1 et HOBBIES_1 tirent davantage de valeur du signal Chronos ZS.

**R3 — L'ensemble tier est le signal dominant (~0.19 geo de contribution)**

La suppression de l'ensemble v9/v11 (ZS dept OLS seul, sans tier) donne 0.9829 contre 0.7969 avec tier. L'ensemble fournit une calibration absolue d'échelle — résultat d'un FT minimal (50–200 steps) — que le ZS seul ne peut pas reproduire par des choix de groupement.

**R4 — Biais de sous-estimation systématique des FOODS par Chronos**

Les poids de mise à l'échelle Chronos-only : FOODS_1=1.59×, FOODS_2=1.18×, FOODS_3=1.16×. Chronos sous-estime les ventes FOODS lorsque le cross-learning est limité aux pairs du même département — tous les pairs partagent le même biais. L'ensemble tier corrige ce biais nativement.

**R5 — L'oubli catastrophique rend le fine-tuning non viable au-delà de 5 steps**

La courbe de steps (FT global, cutoff printemps) montre : 5 steps donne une amélioration marginale (EW 0.8623→0.8581, Δ=+0.004), puis dégradation monotone. À 10 steps, la performance est déjà inférieure au ZS. À 100 steps, le WRMSSE brut a presque doublé.

**R6 — La segmentation par département surpasse la segmentation géographique**

OLS par département (0.8066) > OLS par magasin (0.8132) > par état (estimé encore moins bon). La catégorie produit encode la dynamique de la demande mieux que la géographie.

**R7 — Le modèle small xcl=True est le meilleur ZS standalone ; BASE est compétitif uniquement en fusion tier**

⚠️ *Correction (17 juin) — le chiffre précédent 0.9123 attribué à BASE standalone était une erreur (c'était un résultat blended).*

En ZS standalone confirmé : chronos-2-**small** xcl=True (v20d) = **1.0325 EW / 0.9829 OLS** — meilleur résultat ZS sans tier. chronos-2-BASE xcl=False (v17d) = **1.1963 EW** — pire que small, pas meilleur.

**R8 — LightGBM domine sur les séries à historique complet (−22 % vs meilleur Chronos)**

Un LightGBM global à 37 features avec HPO Optuna atteint geo=0.6198, battant Chronos v20 OLS (0.7969) de 22 %. Le blend des deux modèles (OLS LOO-3) ne récupère pas de diversité — le blend (0.6254) est pire que LGBM seul. Sur les séries à historique complet, Chronos n'apporte aucun signal orthogonal au-delà de ce que LGBM capture déjà.

**R9 — Seuil cold-start à 28 jours : Chronos gagne en dessous de la disponibilité de `lag_28`**

En dessous de 28 jours d'historique de ventes actif, Chronos-2 ZS bat LGBM sur 59–72 % des séries individuelles (médiane RMSSE géo-mean : 1.019 vs 1.059). À partir de 28 jours, LGBM gagne par une marge croissante. Le seuil à 28 jours est mécanistiquement expliqué par la disponibilité de `lag_28` — la feature la plus prédictive de LGBM est structurellement indéfinie pour les séries avec moins de 28 jours de ventes. **Règle pratique : utiliser Chronos-2 durant les 28 premiers jours après le lancement d'une SKU ; passer à LGBM dès que `lag_28` est disponible.**

*Note de validation : l'argument structurel (lag_28 est un NaN dur en dessous de 28j) est indépendant de la taille de l'échantillon et constitue le support principal. L'évidence empirique est directionnelle : Automne n=32 (72 % CW) est le seul cutoff fortement directionnel ; Hiver n=11 (45 %) est une égalité ; le Printemps contribue 0 série. Ne pas citer les taux de victoire empiriques sans ce contexte.*

---

## 14. Résultats Négatifs

| Approche | Résultat | Raison |
|---|---|---|
| FT uniforme (v2) | 1.017 (printemps) | Sur-apprentissage des séries peu fréquentes ; sous-adaptation des séries à fort volume |
| Segmentation par magasin | 0.8132 OLS | La géographie n'encode pas la dynamique de la demande |
| HPO état × poids (12 segments) | ~1.024 | Explosion de l'espace de paramètres sans signal supplémentaire |
| FT LoRA (v15–v22) | Invalide | Bug PeftModel : `merge_and_unload()` manquant |
| FT brut après correction (v23/v24a) | 0.9461–0.9509 | Oubli catastrophique même à 200 steps |
| Ensemble multi-seeds | 0.8554 | Variance des seeds FT trop faible pour créer de la diversité ; pire que le seed unique (0.8544) |
| Réoptimisation W (400 FEV sur v11) | 0.8180 | Borne inférieure — l'hiver ne bénéficie pas de davantage de steps FT |
| ZS tier Chronos (v21) | 1.0483 EW | Le cross-learning intra-volume est plus faible que le cross-learning intra-catégorie |
| xcl=True en batches globaux mélangés | 1.1680 | Représentations de covariables contaminées entre séries incompatibles |
| Blend LGBM × Chronos OLS | 0.6254 | Aucune diversité orthogonale sur les séries à historique complet ; pire que LGBM seul |

---

## 15. Questions Ouvertes

**Q1 — Validation des 5 steps de FT sur 3 cutoffs** *(priorité faible)*  
La courbe de steps n'a été calculée que sur le cutoff printemps. L'amélioration marginale à 5 steps (EW 0.8581 vs ZS 0.8623) mérite validation sur hiver et automne. Compte tenu de la vitesse de l'oubli catastrophique, l'amélioration pourrait être spécifique au pattern saisonnier du printemps et ne pas justifier la surcharge d'infrastructure d'un pipeline FT de production.

**Q2 — Un remplacement Chronos-natif de l'ensemble tier est-il possible ?**  
v21 a montré que le ZS tier seul ne remplace pas l'ensemble externe. La valeur du tier réside dans la calibration absolue d'échelle (issue du FT minimal). Une approche Chronos-native pourrait combiner des facteurs de mise à l'échelle par département (w=1.59 pour FOODS_1) avec le signal ZS cross-learning — potentiellement via un post-traitement de calibration plutôt que du FT.

**Q3 — Que se passe-t-il dans la zone de transition 28–90 jours ?**  
Le poids de blend β optimal converge vers 1.0 (LGBM pur) à l'Automne. Au Printemps, β≈0.85, ce qui signifie que 15% de Chronos dilue les performances. Un sweep plus fin dans la plage 28–56 jours (quand lag_28 vient de devenir disponible mais que lag_42 manque encore) permettrait de déterminer si une transition progressive est préférable à un switch binaire à 28 jours.

**Q4 — Validation externe de la règle cold-start des 28 jours** *(priorité haute pour la crédibilité de l'article)*  
L'évidence empirique actuelle repose sur n=32 séries d'un seul cutoff effectif (Automne 2015). L'Hiver (n=11) montre une égalité, pas une confirmation. Une validation sur un dataset de lancement de nouveaux produits indépendant — même synthétique — apporterait le second point de confirmation que le résultat cold-start n'a pas encore. Sans cela, l'argument structurel doit porter la narration.

**Q5 — Lacune d'évaluation estivale**  
Les trois cutoffs M5 se situent entre Septembre 2015 et Avril 2016. Aucun point d'évaluation estival (juillet–août) n'est disponible — une limite connue de la conception du benchmark M5. La demande estivale en retail (produits saisonniers, vacances, rentrée scolaire) peut constituer un régime genuinement différent. La généralisabilité des résultats Chronos vs LGBM à l'été n'est pas testée.

---

## 16. Évaluation de la Couverture de Validation

Synthèse de la solidité de l'évidence par narration, à des fins de cadrage de l'article.

| Narration | Cutoffs | Qualité de l'évidence | Support principal | Verdict |
|---|---|---|---|---|
| Workflow Chronos (v20 OLS = 0.7969) | 3 (standard M5) | Solide — géo-mean sur 3 régimes saisonniers, OLS LOO-3 | Empirique | **Suffisant** |
| Benchmark LightGBM (0.6198, −22 %) | 3 (mêmes) | Solide — écart trop large pour être renversé par la variance 3-cutoffs | Empirique | **Suffisant** |
| Interaction xcl×covariables | 3 + ablation global/dept | Solide — mécanisme cohérent sur plusieurs variantes d'expérience | Mécanistique + empirique | **Suffisant** |
| Oubli catastrophique (FT dégrade <10 steps) | 1 (printemps seul) | Adéquat — l'amplitude de la dégradation (×2 à 200 steps) est trop large pour être cutoff-spécifique | Empirique (1 cutoff) | **Adéquat** |
| Règle cold-start 28 jours | 1 effectif (Automne) | Mince — n=32, Hiver en égalité, Printemps=0 ; l'argument structurel porte la narration | **Structurel (principal)** | **Conditionnel** |
| Poids OLS α par département | 3 (LOO-3, 2 obs/fold) | Directionnel — directions fiables, magnitudes fragiles | Empirique (folds minces) | **Directionnel** |

**Conditionnel** = citer l'argument structurel en premier ; les données empiriques illustrent mais ne prouvent pas de façon autonome.  
**Directionnel** = le pattern qualitatif (quels depts préfèrent quel modèle) est fiable ; les valeurs α spécifiques ne doivent pas être sur-interprétées.

**Lacune saisonnière :** Automne / Hiver / Printemps sont couverts. L'Été (juillet–août) est absent des données de test M5 — cette limite s'applique à tous les résultats également et doit être mentionnée une fois dans l'article plutôt que répétée par résultat.
