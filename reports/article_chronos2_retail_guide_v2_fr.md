# Chronos-2 pour la Prévision de la Demande en Retail
## Un Workflow Praticien du Zéro-Shot au Démarrage à Froid

**Aquila Data — Juin 2026**

---

## Résumé Exécutif

*Pour les responsables techniques et les décideurs*

Les modèles de fondation pour séries temporelles proposent désormais un compromis genuinement différent pour la prévision retail. Cet article s'adresse aux équipes qui ont choisi — ou envisagent sérieusement — Chronos-2 comme colonne vertébrale de leur système de prévision, et souhaitent comprendre comment en extraire toute la valeur.

Nous avons testé Chronos-2-small sur 30 490 SKUs Walmart issus du benchmark M5 : horizon 28 jours, métrique WRMSSE, trois cutoffs saisonniers (Printemps 2016-04-24, Hiver 2016-01-03, Automne 2015-10-04). Résultat d'un pipeline zero-shot correctement configuré : **WRMSSE 0,7969** — amélioration de 51 % sur la baseline saisonnière naïve (1,6310).

Nous avons ensuite benchmarké ce résultat face à un modèle LightGBM entièrement optimisé — 37 features lag/rolling/exogènes conçues à la main, 19 trials d'HPO Optuna, objectif tweedie — qui atteint **0,6198**. LightGBM gagne sur les séries à historique complet. Nous le rapportons sans détour.

Ce que LightGBM ne peut pas faire — et qui définit la valeur propre de Chronos-2 — c'est prévoir des produits qu'il n'a jamais vus à l'entraînement. Pour les SKUs avec moins de 28 jours d'historique de ventes actif, la feature la plus prédictive de LightGBM (`lag_28`) est structurellement indisponible. Dans notre expérience de démarrage à froid (*cold-start*), Chronos-2 surpasse LightGBM sur 59 à 72 % des SKUs nouvellement lancés. En dessous de 28 jours d'historique, Chronos ne concurrence pas LightGBM. C'est la seule option.

**Pour les équipes disposant déjà d'un pipeline ML mature :** Chronos-2 est votre couche cold-start — déployez-le durant les 28 premiers jours après tout lancement de produit, puis passez la main à votre modèle de production.

**Pour les équipes construisant une capacité de prévision from scratch :** le workflow Chronos-2 atteint 0,7969 — dans la plage d'une précision de production — avec une fraction de l'ingénierie des features, de l'infrastructure d'entraînement et de l'expertise domaine exigées par le ML classique.

Cet article documente le workflow complet : ce qui fonctionne, ce qui ne fonctionne pas, et pourquoi.

---

## Partie I — Le Problème et la Proposition

---

### 1. La Prévision Retail est un Problème de Démarrage à Froid Déguisé

Le discours standard autour de la prévision de la demande retail se concentre sur les produits matures : SKUs à fort volume avec des années d'historique, des patterns saisonniers établis, des courbes d'uplift promotionnel connues. Ce sont des problèmes maîtrisés. Un modèle gradient boosting bien paramétré avec des features de lag les gère de façon fiable.

Le problème difficile, c'est le lancement. Un retailer de taille intermédiaire introduit des centaines de nouveaux SKUs par mois. Un retailer fast-fashion renouvelle tout son assortiment chaque saison. Une chaîne de supermarchés ajoute en permanence des lignes régionales. Dans tous ces cas, le pipeline ML conventionnel présente une zone aveugle structurelle : il ne peut prévoir que les produits vus à l'entraînement. Un produit lancé aujourd'hui n'a pas de features lag, pas de statistiques rolling, pas de target encoding — le socle de toute prévision par arbres de décision.

La réponse habituelle : attendre. Accumuler quelques semaines de données, puis intégrer le produit au prochain cycle de ré-entraînement. Entre temps, le demand planner estime à vue. La supply chain sur-commande par prudence. Les coûts de sur-stock s'accumulent.

Les modèles de fondation comme Chronos-2 brisent ce schéma. Ils arrivent pré-entraînés sur des centaines de millions d'observations de séries temporelles diverses. Un nouveau SKU ne nécessite pas de run d'entraînement — le modèle a déjà appris l'espace des patterns de demande que les produits retail habitent. Pointez-le sur n'importe quelle série, même très courte, et il produit une prévision dès le premier jour.

C'est la proposition de valeur principale de Chronos-2 pour le retail : non pas qu'il surpasse tout modèle classique sur toutes les séries, mais qu'il opère sans le prérequis en données qui rend les modèles classiques inutiles au début de la vie d'un produit.

---

### 2. Le Paysage Concurrentiel — Un Cadrage Honnête

Avant de décrire le workflow Chronos, nous devons au lecteur un benchmark honnête. Si une approche plus simple atteint une précision substantiellement meilleure, cela compte — quelle que soit la complexité de l'histoire d'ingénierie.

Nous avons construit un modèle LightGBM global sur les mêmes 30 490 séries M5, avec 37 features :

**Features de lag :** `lag_28`, `lag_29`, `lag_30`, `lag_31`, `lag_35`, `lag_42` — six valeurs scalaires de lookback capturant les cycles de demande hebdomadaires et bi-hebdomadaires.

**Statistiques rolling :** moyenne et écart-type sur des fenêtres de 7, 14, 30, 60 et 180 jours — toutes ancrées au lag-28 pour éviter la fuite d'information. Ces features capturent la tendance, la profondeur de saisonnalité et la volatilité de la demande.

**Features exogènes :** prix de vente, variation de prix, prix normalisé, indicateurs SNAP (CA/TX/WI), features calendaires (indicateurs week-end, fins de mois), encodage des événements, moyennes de ventes target-encodées par département, magasin et État.

**Objectif et entraînement :** perte tweedie (puissance de variance optimisée via Optuna) pour les données de comptage intermittentes, gradient boosting avec early stopping, 19 trials d'HPO optimisant num_leaves, min_data_in_leaf, learning rate, feature fraction, bagging fraction, et régularisations L1/L2.

**Résultat :** WRMSSE **0,6198** (géo-mean 3 cutoffs : Automne 0,6016, Hiver 0,6406, Printemps 0,6178).

Chronos-2 v20 OLS : **0,7969**.

LightGBM gagne de 22 %. Ce n'est pas une erreur d'arrondi.

Mais cette comparaison mérite une mise en contexte sur deux points.

**Premier point : ce qu'a coûté le résultat LightGBM.** Le pipeline à 37 features a nécessité une expertise domaine (quelles fenêtres de lag sont pertinentes pour les cycles retail hebdomadaires), une infrastructure (système de cache des features, gestion des cutoffs de calibration, gestion des checkpoints, HPO sur 19 trials), et un débogage itératif. Le pipeline Chronos a nécessité un sweep de quatre longueurs de contexte, l'activation d'un booléen (`cross_learning=True`) et l'ajustement de sept régressions OLS. Les deux produisent des résultats respectables ; ils représentent des investissements en ressources très différents. Les équipes que Chronos sert le mieux sont celles qui auraient autrement passé des semaines à construire ce que LightGBM a requis avant d'obtenir leur premier nombre de production.

**Deuxième point : ce que LightGBM ne peut pas faire.** Le modèle LGBM a été entraîné sur des données M5 qui incluent chacune des 30 490 séries testées sur leur période d'entraînement complète. En déploiement, de nouveaux SKUs apparaissent que LightGBM n'a jamais vus à l'entraînement. À cet instant, `lag_28` est indéfini (NaN), `lag_42` est indéfini, et cinq des sept statistiques rolling sont indéfinies. Le modèle les route vers la branche NaN, mais cette branche a été entraînée pour gérer des ruptures de stock temporaires et des gaps d'intermittence — pas de véritables nouveaux produits.

L'expérience cold-start quantifie cela précisément (§14). Le seuil de 28 jours n'est pas arbitraire : c'est exactement l'instant où `lag_28`, la feature de lag la plus prédictive, devient disponible.

---

## Partie II — Construction du Workflow Chronos

---

### 3. Point de Départ : Baseline Naïve et Premier Run Zéro-Shot

Toute évaluation requiert un plancher. Notre baseline naïve répète les sept dernières ventes journalières observées par série, produisant une prévision cyclique hebdomadaire :

| Cutoff | WRMSSE |
|--------|--------|
| Printemps 2016-04-24 | 1,4639 |
| Hiver 2016-01-03 | 1,4216 |
| Automne 2015-10-04 | 2,0848 |
| **Géo-mean** | **1,6310** |

Le cutoff d'automne (octobre 2015) est le plus difficile : le stockage pré-fêtes de fin d'année perturbe le pattern hebdomadaire sur lequel la baseline naïve repose. Tout modèle sérieux doit gérer ce régime.

Notre premier run Chronos-2-small utilisait une longueur de contexte de 2 (deux observations historiques) sur toutes les 30 490 séries sans segmentation ni covariables. Résultat : **1,2504**. Mieux que la naïve, mais pas spectaculairement. Plus révélateur encore : CL=1 est numériquement *identique* à la baseline saisonnière naïve (1,6310). Avec une seule observation, Chronos produit le même repeat hebdomadaire en dernière valeur que la naïve. Le modèle a besoin de suffisamment de contexte pour voir au-delà de l'observation la plus récente.

La progression de la naïve vers un pipeline ZS correctement configuré illustre l'effet cumulatif de chaque choix de configuration :

| Étape | Méthode | geo | Δ vs précédent | Ce que ça répond |
|-------|---------|-----|----------------|------------------|
| 0 | Saisonnière Naïve | 1,6310 | — | Pure répétition saisonnière |
| 1 | ZS ventes seules, xcl=True, CL=7 | 1,2504 | −23 % | Ce qu'un FM pré-entraîné apporte depuis l'historique seul |
| 2 | ZS + covariables (eP_sF), xcl=False, CL=7 | 1,0776 | −15 % | Ce qu'un contexte exogène apporte en plus |
| 3 | FT global (25 pas, lr=1e-7) | 1,0319 | −4 % | Ce que le fine-tuning apporte à l'échelle globale |
| 4 | ZS dept-seg xcl=True + OLS | **0,7969** | −23 % | Ce qu'apportent le batching sémantique et la calibration |

> **Leçon 1 — La longueur de contexte est le paramètre zéro-shot dominant.** Pour les données retail journalières intermittentes, CL=7 (une semaine) capture le cycle de demande hebdomadaire. Au-delà de CL=14, les performances se dégradent sur les séries bruitées — le modèle traite les pics historiques comme un signal structurel. Ne déployez jamais sans sweep de CL.

---

### 4. Covariables et Cross-Learning — Une Interaction Critique

Deux mécanismes distincts contribuent au signal dans le chemin d'inférence de Chronos-2 : les **covariables exogènes** (prix, promotions, SNAP, données calendaires par série) et l'**attention cross-séries** (`cross_learning=True`). Comprendre leur interaction est essentiel pour configurer correctement le pipeline.

À CL=7 avec un pool d'inférence global (30 490 séries en mélange), l'interaction est claire et contre-intuitive :

|  | xcl=False | xcl=True | Δ (effet xcl) |
|--|-----------|----------|---------------|
| **Sans covariables** | 1,2716 | **1,2504** | −0,021 *(xcl aide)* |
| **Avec covariables** | **1,0776** | 1,1765 | +0,099 *(xcl nuit)* |

Les covariables et le cross-learning sont **substituables, pas complémentaires**. Ils fournissent chacun un contexte inter-séries par des canaux différents — les covariables apportent un contexte exogène par série ; l'attention cross-séries apporte une similarité de demande inter-séries. Quand les deux sont actifs dans un batch globalement mélangé, le cross-attention mélange les représentations de covariables *entre* séries ayant des expositions aux événements différentes : une série Californie attend sur une série Texas lors d'une période SNAP Californie exclusivement, injectant du bruit dans les deux représentations. Le signal covariable par série domine ; l'interférence du cross-attention le dégrade.

Sans covariables, le cross-attention opère proprement — le seul contexte inter-séries disponible est la similarité de niveau de demande au sein du batch, qu'il exploite de manière fiable (gain −0,021).

La résolution est la **cohérence sémantique des batches** : quand le cross-attention opère dans un batch segmenté par département — 100 séries FOODS_1 partageant toutes la même exposition aux événements et des patterns de demande similaires — la contamination disparaît. Les batches intra-département ont des structures de covariables cohérentes, et le cross-attention extrait un signal de groupe genuinement utile.

> **Leçon 2 — Covariables et cross-learning se disputent le même budget de signal dans les batches mélangés. Avec des données exogènes riches, désactivez le cross-learning globalement. Activez-le uniquement dans des batches cohérents par catégorie.**

---

### 5. Segmentation — Les Tiers de Volume Surpassent Tout

Les 30 490 séries M5 couvrent une très large plage de volumes de ventes : un article FOODS peut se vendre à 200 unités par jour ; un article HOBBIES, à 1 ou 2. Ces catégories répondent différemment à la longueur de contexte. L'hypothèse naturelle de segmentation : assigner chaque série à un groupe et sweeper la CL optimale du groupe indépendamment.

Nous avons testé trois axes de segmentation :

**Classification ADI/CV²** (smooth / erratic / intermittent / lumpy) : 0,9418 sur le cutoff printemps. Correct, mais la classification nécessite un calcul de statistiques par série et introduit une dépendance de prétraitement. Plus important, elle est désalignée avec la structure de la métrique : WRMSSE pondère les séries par leur contribution à l'erreur de prévision, pas par leur type de pattern de demande.

**Tiers de volume** (quartiles de poids WRMSSE : Low, Med-Low, Med-High, High) : **0,9082** sur le printemps. Plus simple à calculer, directement aligné avec la structure de la métrique, et améliore nettement la segmentation ADI/CV². Les tiers bas et haut ont besoin de CL très courts ; les tiers intermédiaires bénéficient d'un contexte légèrement plus long. De façon contre-intuitive, le tier Med-High n'avait besoin que de CL=1 — le modèle voit assez en une seule observation pour identifier le pattern hebdomadaire dominant dans cette plage de volume.

**Segmentation par magasin** (10 magasins Walmart par géographie) : 0,8132 OLS (3 cutoffs) — pire que le ZS par département (0,8066). La proximité géographique ne capture pas l'homogénéité des patterns de demande. Les dix magasins partagent la saisonnalité hebdomadaire ; ils diffèrent en niveau de volume, que la structure par tier capture déjà directement.

> **Leçon 3 — Segmentez par comportement de demande, pas par géographie ou hiérarchie organisationnelle.** Les segmentations par tier de volume et par catégorie de produit s'alignent avec la façon dont le modèle utilise le contexte. La segmentation géographique ajoute rarement du signal dans le retail alimentaire à granularité journalière.

---

### 6. Impasse n°1 : Le Fine-Tuning et le Catastrophic Forgetting

L'hypothèse naturelle suivante : si le zéro-shot correctement configuré atteint 1,0776, quelques pas de gradient sur les données Walmart devraient faire mieux.

Nous avons conduit une courbe systématique du nombre de pas : fine-tuning global sur toutes les 30 490 séries, de 0 à 1000 pas de gradient, puis blending 50/50 du modèle fine-tuné avec l'ensemble tier (§8) :

| Pas | WRMSSE ZS brut | WRMSSE blendé |
|-----|----------------|---------------|
| 0 | 1,063 | 0,8623 |
| **5** | 1,037 | **0,8581** ← pic |
| 10 | 1,049 | 0,8642 |
| 25 | 1,081 | 0,8833 |
| 50 | 1,164 | 0,9259 |
| 100 | 1,345 | 1,0116 |
| 200 | 1,549 | 1,1029 |
| 500 | 1,628 | 1,1321 |

*Cutoff printemps, fine-tuning global sur les 30 490 séries M5.*

Le tableau est sans ambiguïté. Cinq pas produisent une amélioration marginale de 0,004 dans le score blendé. Dix pas reviennent au niveau zéro. Au-delà, la dégradation est monotone et le WRMSSE double presque à 200 pas.

C'est le **catastrophic forgetting** : les mises à jour gradient écrasent les représentations cross-séries pré-entraînées qui portent le signal pertinent, plus vite que les patterns spécifiques à Walmart ne sont acquis. Le pré-entraînement de Chronos-2 encode la diversité des patterns de demande sur des millions de séries diverses ; le fine-tuning M5 rétrécit rapidement cette représentation vers des patterns spécifiques à Walmart tout en détruisant le signal plus large.

Un gain de 5 pas de 0,004 WRMSSE est en dessous du bruit de mesure sur trois cutoffs saisonniers et ne justifie pas la surcharge d'infrastructure (temps GPU, gestion des checkpoints, planification du ré-entraînement) qu'un pipeline FT de production requiert.

> **Leçon 4 — Ne fine-tunez pas un TSFM sur des données retail intermittentes sans contrôle strict du nombre de pas.** Le catastrophic forgetting n'est pas théorique — il se manifeste en moins de 10 pas de gradient sur 30 000 séries. Si le fine-tuning est nécessaire, utilisez LoRA (entraînement des adaptateurs uniquement, base gelée). Validez à 5, 10 et 25 pas. Arrêtez immédiatement si les performances stagnent ou se dégradent.

---

### 7. Le Bug PeftModel — Quand Votre Expérience est Silencieusement Incorrecte

Nous avons poursuivi LoRA comme chemin moins risqué après avoir observé le catastrophic forgetting en fine-tuning complet. Entre le 13 et le 15 juin, nous avons conduit de multiples expériences LoRA sur différentes stratégies de segmentation (par tier, par département, départements isolés — v15 à v22). Les résultats étaient inconsistants et décevants.

Ce n'est qu'à la mi-juin que nous avons trouvé la cause racine : un appel `merge_and_unload()` manquant.

Quand Chronos-2 utilise LoRA, le modèle de base et l'adaptateur sont des composants séparés à l'entraînement. Le chemin d'inférence quantile de Chronos (`predict_quantiles()`) opère sur un pipeline `MeanScaleUniformBins` qui ne route pas à travers le wrapper PEFT. Sans merge explicite, les poids de l'adaptateur sont présents mais inaccessibles lors de l'inférence — le modèle « fine-tuné » est fonctionnellement identique au modèle de base.

```python
# Échec silencieux — poids de l'adaptateur ignorés
pipeline = MeanScaleUniformBins(peft_model)

# Correct — adaptateur fusionné avant wrapping
pipeline = MeanScaleUniformBins(peft_model.merge_and_unload())
```

Chaque résultat LoRA de v15 à v22 était numériquement identique au modèle de base. Après la correction, LoRA par département (v23, 200 pas) a atteint **0,9509** (géo-mean 3 cutoffs) — meilleur que les résultats pré-correction mais bien en dessous du pipeline ZS + tier (0,7969).

> **Leçon 5 — Les adaptateurs PEFT nécessitent un merge explicite avant l'inférence TSFM.** La méthode `predict_quantiles()` de Chronos ne route pas à travers le wrapper PEFT. Vérifiez toujours que les sorties fine-tunées diffèrent du modèle de base sur un ensemble de validation. Si elles sont identiques, l'adaptateur n'est pas appliqué.

---

### 8. L'Ensemble Tier — Fine-Tuning à la Bonne Échelle

Alors que le fine-tuning global échouait, le fine-tuning dans une *décomposition structurée par tier* a produit des gains durables. L'insight clé : la cohérence du tier contraint le forgetting (moins de séries diverses par batch d'entraînement) et crée un axe de diversité significatif pour l'ensemble.

Nous avons fine-tuné quatre modèles séparément — un par tier de volume — avec fine-tuning complet et early stopping agressif. Le blend a été appris comme une **matrice de poids softmax 4×4 W**, où l'entrée `W[tier_assignment][tier_model]` donne la contribution de chaque modèle tier à chaque série selon son tier d'appartenance.

W a été optimisée par Nelder-Mead (400 évaluations de fonction, objectif WRMSSE 3 cutoffs). Une recherche Optuna initiale a trouvé les meilleurs hyperparamètres par tier (meilleur trial : 0,8544) ; la re-optimisation dédiée de la matrice W a ensuite amélioré le résultat standalone du tier à **0,8180**.

La structure W est critique : un blend global à quatre vecteurs (mêmes poids pour toutes les 30k séries) n'atteint que 0,9278. La structure per-tier-assignment — chaque série utilise le blend approprié à son tier — explique tout l'écart entre 0,93 et 0,82.

---

### 9. Le Cross-Learning et la Composition Sémantique des Batches — Le Levier Inattendu

Le gain ZS le plus significatif est venu d'un paramètre que la plupart des utilisateurs ne configurent jamais : `cross_learning`. Quand il est activé, Chronos-2 active l'attention cross-séries sur les 100 séries de chaque batch d'inférence — plutôt que traiter chaque série indépendamment, le modèle attend sur ses voisins de batch et peut exploiter des patterns de demande corrélés.

Le cross-learning global sur des batches de catégories mélangées dégradait les performances (§4). Mais avec une inférence segmentée par département :

| xcl | Composition du batch | geo EW |
|-----|----------------------|--------|
| False | Mélangé (global) | 1,0776 |
| False | Segmenté par dept | 1,0739 |
| True | Mélangé (global) | 1,1765 — pire |
| **True** | **Segmenté par dept** | **1,0325 — meilleur ZS** |

Dans un batch de département FOODS_1, 100 séries partagent la même éligibilité SNAP, font face aux mêmes événements promotionnels, et suivent des patterns de demande hebdomadaires similaires. La tête d'attention cross-séries opère sur des voisins genuinement cohérents — elle apprend un signal de niveau de groupe plutôt que de mélanger le bruit de catégories incompatibles.

Un résultat secondaire : les séries FOODS_1 montrent une sous-estimation systématique de 1,59× en mode dept-seg xcl=True. Dans des batches purement FOODS_1, tous les voisins sont des articles à fort volume et il n'y a pas d'ancre externe pour corriger le biais collectif. L'étape de calibration OLS (§11) y remédie de façon fiable.

> **Leçon 6 — Le cross-learning requiert une cohérence sémantique des batches. Triez vos séries par catégorie de produit avant d'appeler `predict_quantiles`, et faites une inférence par catégorie. Dans des batches globaux mélangés, désactivez le cross-learning. Le gain de la bonne configuration est ∼0,04 WRMSSE — supérieur à la plupart des autres choix de paramètre uniques.**

---

### 10. Le Modèle BASE — La Taille n'est Pas une Garantie

Chronos-2 BASE (120M paramètres, ∼2,6× plus grand que small) a été évalué en mode standalone avec inférence par département et sweep de CL optimale. Résultat : **1,1963 EW** — pire que Chronos-2-small xcl=True (1,0325) par une marge substantielle.

Avec cross-learning activé et HPO de CL dédié : **1,1106** — amélioration marginale, confirmant que l'espace de recherche est épuisé. BASE xcl=True en batches segmentés par département introduit un biais sévère sur FOODS_1 (α=1,943×, presque le double), le biais de covariable le plus extrême observé dans toutes les expériences.

**Pourquoi :** les données M5 sont courtes (∼5 ans de ventes journalières, de nombreuses séries avec gaps et intermittence). Un modèle à 120M paramètres a une capacité substantiellement plus grande, mais cette capacité a été calibrée sur des séries longues, continues et diverses. Les données retail courtes et intermittentes n'offrent pas assez de signal pour exploiter les paramètres supplémentaires.

> **Leçon 7 — Pour les séries retail courtes et intermittentes, préférez le TSFM plus petit. Les modèles plus grands sont entraînés pour des données longues et continues. Benchmarker toujours les deux tailles explicitement avant d'assumer que plus grand est meilleur.**

---

### 11. La Recette Finale : v20 OLS (WRMSSE 0,7969)

Le meilleur résultat combine deux composants complémentaires via une étape de calibration simple :

**Composant A — Ensemble tier (0,8180 standalone) :** quatre modèles tier fine-tunés indépendamment, blendés avec une matrice W softmax 4×4 optimisée par Nelder-Mead. Capture les patterns temporels fins dans chaque segment de volume.

**Composant B — ZS département xcl=True (1,0325 EW / 0,9829 OLS standalone) :** Chronos-2-small, cross-learning activé, batches par département, CL optimale par département via HPO Optuna. Apporte un apprentissage de représentation cross-séries global que le fine-tuning tier-isolé ne voit jamais.

**Calibration — OLS LOO-3 par département :** un scalaire multiplicatif par département, ajusté par validation croisée leave-one-cutoff-out. Corrige le biais de scale FOODS_1 (α=1,59×) introduit par dept-seg xcl=True. Empêche le surapprentissage sur les trois cutoffs saisonniers disponibles.

**Pourquoi ils se complémentent :** les représentations fine-tunées de l'ensemble tier sont adaptées aux patterns temporels spécifiques à Walmart dans chaque segment de volume. Le modèle ZS apporte un apprentissage de représentation cross-catégories via xcl=True que le fine-tuning par tier, entraîné indépendamment, ne voit jamais. L'OLS permet à ces contributions d'émerger sans réglage manuel.

> **Leçon 8 — Les composants ZS et ensembles légèrement fine-tunés sont plus complémentaires qu'attendu. Les représentations cross-learning ZS sont orthogonales aux représentations adaptées par FT. La validation croisée LOO est obligatoire avec de petits jeux d'évaluation — l'OLS en échantillon surestime trivialement avec trois cutoffs.**

---

### 12. Comment les Gains se sont Accumulés

| Étape | Configuration | WRMSSE | Gain vs Naïve |
|-------|--------------|--------|---------------|
| Saisonnière Naïve | Répète les 7 dernières obs | 1,6310 | — |
| ZS CL=1, global | Chronos-2-small, 1 obs contexte | 1,6310 | 0,0 % |
| ZS CL=7, global xcl=False + cov | Meilleur ZS global CL unique | 1,0776 | −34 % |
| ZS CL=7, dept-seg xcl=False | CL par dept (v13) | 1,0739 | −34 % |
| ZS dept-seg xcl=True | Cross-learning dans depts (v20 EW) | 1,0325 | −37 % |
| ZS OLS-scalé | ZS dept + α par dept (v20 OLS) | 0,9829 | −40 % |
| Ensemble tier (W re-opt) | 4-tier FT + matrice W 4×4 | 0,8180 | −50 % |
| **Tier + ZS xcl=True OLS** | **v20 : pipeline complet** | **0,7969** | **−51 %** |

Le pattern : les choix de configuration ZS (CL, segmentation, cross-learning) comptent pour le gain absolu le plus grand (1,6310 → 1,0325). L'ensemble tier et la calibration OLS couvrent la dernière ligne droite (1,0325 → 0,7969). La qualité des poids de blend (0,8544 → 0,8180 via re-optimisation de la matrice W) a contribué davantage que l'ajout du signal ZS à un tier non optimisé (0,8417 → 0,7969 via ZS + OLS). Les deux comptent ; l'architecture du blend compte autant que la sélection des modèles.

---

## Partie III — La Comparaison Honnête

---

### 13. LightGBM comme Concurrent — Ce que Coûte la Précision Fine

Un article qui ne rapporte que les résultats de son modèle choisi sans benchmark face à l'alternative évidente n'est pas utile. Nous avons construit le modèle LightGBM.

**37 features.** Six features de lag (28–42 jours) capturant les cycles de demande hebdomadaires et bi-hebdomadaires. Sept statistiques rolling (moyenne et écart-type sur des fenêtres de 7, 14, 30, 60, 180 jours, toutes ancrées au lag-28). Prix de vente, variation de prix, prix normalisé. Indicateurs SNAP. Features calendaires : indicateurs week-end, fins de mois, encodage des événements. Moyennes de ventes target-encodées par magasin, département et État — nécessitant une passe d'agrégation sur l'historique complet d'entraînement avant tout entraînement de modèle.

**Optimisation des hyperparamètres.** 19 trials de recherche TPE Optuna optimisant : `num_leaves`, `min_data_in_leaf`, `learning_rate`, `feature_fraction`, `bagging_fraction`, `lambda_l1`, `lambda_l2`, et `tweedie_variance_power`. Chaque trial exécute trois cutoffs de calibration × jusqu'à 800 rounds de boosting.

**Infrastructure d'entraînement.** Pipeline de cache des features, gestion des checkpoints, séparation cutoff de calibration / cutoff de test, ré-évaluation post-hoc depuis les checkpoints stockés.

**Résultats :** WRMSSE **0,6198** (Automne 0,6016, Hiver 0,6406, Printemps 0,6178).

Question naturelle : peut-on blender LGBM et Chronos pour tirer le meilleur des deux ? Nous avons conduit un blend OLS scalaire LOO-3. Résultat : **0,6254** — *pire* que LGBM standalone (0,6198). Le poids de blend optimal pour l'Automne converge à β=1,0 (LGBM pur) ; pour l'Hiver et le Printemps, β≈0,85 (15 % de Chronos dilue les performances). Les deux modèles ne fournissent pas de diversité orthogonale sur ces trois cutoffs. Sur les séries à historique complet, Chronos n'ajoute rien au-delà de ce que LGBM capture déjà.

**La proposition de valeur de Chronos n'est pas affaiblie par cela.** Le pipeline LightGBM décrit ci-dessus représente un investissement conséquent en expertise domaine, infrastructure et itérations de débogage. Pour les équipes sans cette infrastructure — sans la connaissance domaine pour sélectionner ces features, sans le budget HPO, sans la cadence de ré-entraînement — le 0,7969 de Chronos est le bon premier investissement. Il est atteignable en jours, ne nécessite pas d'ingénierie de features, et reste compétitif.

Plus important : le pipeline LGBM a un plafond que Chronos ne partage pas.

---

### 14. L'Avantage Cold-Start — La Règle des 28 Jours

Nous revenons au problème qui ouvrait cet article : les lancements de nouveaux produits.

Pour mesurer cela directement, nous avons calculé le RMSSE non pondéré par série pour chacune des 30 490 séries M5, groupées par leur *longueur d'historique actif* à chaque cutoff — le nombre de jours depuis leur première vente non nulle. Cette métrique est délibérément non pondérée : la métrique WRMSSE qui favorise les séries à fort volume supprime exactement le signal nouveaux produits que nous mesurons.

Les buckets d'historique sont ancrés sur les prérequis des features LGBM :
- `lag_28` : disponible uniquement après 28 jours
- `lag_42` : disponible uniquement après 42 jours
- `roll_mean_180_l28` : nécessite 208 jours d'historique
- Un cycle saisonnier complet : 365 jours

**Résultats — agrégés sur 3 cutoffs (géo-mean des médianes par cutoff) :**

| Historique Actif | n (Automne) | RMSSE LGBM | RMSSE Chronos | Gagnant | Chronos gagne % |
|-----------------|-------------|------------|---------------|---------|-----------------|
| **< 28 jours** | 32 | 1,059 | **1,019** | **Chronos** | **59 %** |
| 28 – 90 jours | 57 | **0,812** | 0,847 | LGBM | 40 % |
| 90 – 365 jours | 1 274 | **0,735** | 0,871 | LGBM | 20 % |
| 365 – 730 jours | 3 642 | **0,712** | 0,836 | LGBM | 20 % |
| 730+ jours | 25 418 | **0,664** | 0,772 | LGBM | 23 % |

*Signal Chronos : ZS pur dept xcl=True, sans ensemble tier, sans fine-tuning — exactement ce qu'un praticien déploie au jour 1.*

Le croisement est à 28 jours. En dessous de ce seuil, Chronos surpasse LGBM sur 59 % des séries individuelles et atteint un RMSSE médian inférieur. Au cutoff Automne spécifiquement — qui capture le plus de lancements de produits — Chronos gagne **72 %** des SKUs nouvellement lancés avec un RMSSE médian de 1,189 contre 1,285 pour LGBM.

**Pourquoi exactement 28 jours ?** `lag_28` est la feature de lag la plus prédictive de LGBM — la demande du même jour de la semaine quatre semaines auparavant, qui capture directement la saisonnalité hebdomadaire. En dessous de 28 jours, `lag_28` est un NaN structurel — pas une estimation bruitée, pas une valeur manquante qui peut être imputée, mais une feature qui n'existe tout simplement pas. Le modèle la route vers la branche NaN apprise à l'entraînement, mais cette branche a été entraînée pour gérer des ruptures de stock temporaires et des gaps d'intermittence — pas de véritables nouveaux produits.

Chronos-2 n'a pas un tel minimum. Avec une seule observation, il peut produire une prévision. Avec sept observations (une semaine), il capture le cycle hebdomadaire. Avec 28 observations, il est comparable à un modèle mature sur les mêmes séries. Le modèle de fondation ne tombe pas en panne au lancement d'un produit — il commence à contribuer immédiatement.

**La règle de déploiement pratique :** utilisez Chronos-2 pour tout produit avec moins de 28 jours d'historique de ventes actif. Passez à votre modèle de production (LGBM ou équivalent) une fois que `lag_28` est disponible. Ce système hybride n'est pas une préférence théorique — il est empiriquement justifié par le croisement observé dans nos données.

**Sur la taille de l'échantillon :** le bucket <28 jours contient 43 séries sur deux cutoffs (32 à l'Automne, 11 à l'Hiver ; le Printemps n'avait aucune série dans cette plage en avril 2016). L'évidence empirique est directionnelle, pas conclusive. L'argument structurel — lag_28 est une frontière dure — est indépendant de la taille de l'échantillon et tient par lui-même.

---

## Partie IV — Recommandations Pratiques

---

### 15. Un Cadre de Décision — Quand et Comment Utiliser Chronos

La décision n'est pas binaire. Chronos-2 et LightGBM servent des besoins différents ; un système de prévision de production peut légitimement utiliser les deux.

**Scénario A : Vous disposez déjà d'un pipeline ML mature**

Chronos-2 est votre couche cold-start. Pour tout SKU avec moins de 28 jours d'historique, déployez Chronos-2 en mode ZS (Phase 1 ci-dessous). Pour les SKUs matures, conservez votre modèle de production. Les deux systèmes peuvent tourner en parallèle avec une simple porte basée sur la longueur d'historique :

```python
def forecast(sku_id, history_days):
    if history_days < 28:
        return chronos_forecast(sku_id)  # ZS, sans entraînement
    else:
        return lgbm_forecast(sku_id)     # modèle de production
```

**Scénario B : Vous construisez une capacité de prévision**

Commencez par Chronos-2. La Phase 1 ci-dessous est atteignable en deux à trois jours et produit un WRMSSE dans la plage 0,98–1,08 — compétitif avec les modèles de production de nombreux retailers. Revisitez LGBM une fois que votre équipe dispose de l'infrastructure de données et de la connaissance domaine pour construire le pipeline de features.

**Scénario C : Vous avez besoin d'un déploiement rapide sans infrastructure ML**

Le pipeline ZS de Chronos-2 ne nécessite pas de feature store, pas de gestion des données d'entraînement, pas de planning de ré-entraînement. Il opère depuis les poids du modèle et une fenêtre de contexte. C'est son avantage structurel sur tout modèle entraîné pour les équipes sans capacité MLOps.

---

**Phase 0 : Établir la baseline (un jour)**

Exécutez Chronos-2-small en mode zero-shot avec CL=7, sans segmentation, sans covariables. C'est votre point de départ. C'est aussi l'équivalent numérique de la baseline saisonnière naïve si vous utilisez accidentellement CL=1 — vérifiez que vos résultats diffèrent de façon significative.

**Phase 1 : Configurer le pipeline ZS (deux à trois jours)**

```python
for categorie in categories_produit:
    meilleure_cl = sweep_cl([1, 4, 7, 14, 28], categorie)  # sweep 5 points
    previsions[categorie] = chronos_small.predict_quantiles(
        series=filtrer_par_categorie(toutes_series, categorie),
        context_length=meilleure_cl,
        cross_learning=True,   # uniquement dans des batches cohérents
        batch_size=100,
    )
previsions_calibrees = loo_ols(previsions, actuals)  # LOO si ≥ 3 cutoffs
```

Ajoutez des covariables exogènes : au minimum, indicateurs SNAP et features calendaires si disponibles ; données de prix si propres. Les covariables améliorent les performances de −0,194 geo à la CL optimale (de 1,2716 sans à 1,0776 avec `eP_sF`). N'activez pas le cross-learning globalement avec des covariables présentes — utilisez des batches segmentés par catégorie.

Cette étape atteint WRMSSE ∼0,98–1,08. Pour la plupart des retailers sans infrastructure ML mature, c'est un excellent point de départ.

**Phase 2 : Ajouter l'ensemble tier (une à deux semaines)**

Segmentez les séries en 3–4 tiers de volume (quartiles de poids WRMSSE ou équivalent). Fine-tunez un modèle Chronos-2-small séparé par tier, maximum 25 pas de gradient, en validant le WRMSSE à chaque checkpoint. Apprenez les poids de blend par optimisation sans gradient (Nelder-Mead, ≥200 évaluations de fonction). Blendez l'ensemble tier avec le ZS Phase 1 via LOO-OLS par catégorie.

Cette étape atteint WRMSSE ∼0,80. La surcharge d'ingénierie est significative. Investissez dans la Phase 2 uniquement si la précision de la Phase 1 est insuffisante pour votre cas d'usage.

---

### 16. Notes d'Ingénierie — Ce qu'il Faut Maîtriser Avant de Commencer

Ces détails ont eu un impact disproportionné sur nos résultats.

**La longueur de contexte est un hyperparamètre, pas un réglage.** CL=7 est le bon point de départ pour les données retail journalières. CL=14 ajoute de la valeur pour certaines catégories quand des covariables sont présentes. CL≥28 nuit typiquement aux séries intermittentes. Sweepez toujours {1, 4, 7, 14, 28} avant de fixer CL.

**La composition des batches contrôle la qualité du cross-learning.** Si `cross_learning=True`, chaque série dans un batch de 100 doit provenir de la même catégorie de produit. Triez votre liste de séries par catégorie avant d'appeler `predict_quantiles`. Si les labels de catégorie ne sont pas disponibles, désactivez le cross-learning.

**LoRA nécessite `merge_and_unload()` avant l'inférence Chronos.** La méthode `predict_quantiles()` route via `MeanScaleUniformBins`, pas le wrapper PEFT. Un PeftModel non mergé se comporte silencieusement comme le modèle de base. Vérifiez toujours que les sorties fine-tunées diffèrent du modèle de base.

**La calibration LOO-OLS nécessite au moins trois cutoffs d'évaluation distincts.** Avec deux cutoffs, LOO ajuste sur une seule observation par fold. Avec cinq ou plus, adoptez un forward-chaining temporel strict.

**CL=1 est la baseline saisonnière naïve.** C'est une vérification utile : si votre résultat CL=7 est seulement marginalement meilleur que CL=1, la construction de votre contexte est probablement défaillante.

**Le biais de scale FOODS est attendu et corrigeable.** Le dept-seg xcl=True introduit une sous-estimation systématique dans FOODS_1 (α≈1,59× dans nos données). L'étape OLS le corrige de façon fiable — incluez toujours l'OLS quand vous utilisez dept-seg xcl=True.

---

### 17. Conclusion

Cet article a commencé avec un problème — l'échec cold-start de la prévision classique — et s'est terminé par un résultat : Chronos-2 gagne sur les produits nouvellement lancés en dessous de 28 jours d'historique, précisément là où le ML classique n'a aucune feature valide à offrir.

Le workflow Chronos décrit dans cet article — ZS avec segmentation par catégorie, cross-learning et calibration OLS — atteint WRMSSE 0,7969 sur un benchmark rigoureux à trois cutoffs. Un LightGBM entièrement optimisé avec 37 features construites à la main atteint 0,6198. LightGBM gagne sur les séries à historique complet. En dessous de 28 jours, le concours est structurellement à sens unique.

Pour les équipes choisissant Chronos, le résumé pratique :

1. Commencez avec CL=7, batches segmentés par catégorie, `cross_learning=True`, et covariables `eP_sF`. C'est votre baseline ZS prête pour la production (∼1,08 WRMSSE) en une journée.
2. Ajoutez la calibration OLS par catégorie avec validation croisée LOO. Cela atteint le plafond ZS (∼0,98) avec une surcharge minimale.
3. Si une précision supplémentaire est nécessaire, construisez l'ensemble tier (§8). Validez le fine-tuning à 5, 10 et 25 pas. Reconstruisez les poids de blend avec Nelder-Mead, pas par simple moyenne.
4. Pour tout SKU en dessous de 28 jours d'historique, utilisez Chronos ZS pur — que vous ayez un modèle LGBM en production ou non.
5. Ne fine-tunez pas à l'échelle globale. N'assumez pas que le modèle BASE est meilleur. N'activez pas le cross-learning dans des batches de catégories mélangées avec des covariables.

Chronos-2 n'est pas toujours le modèle le plus précis. Pour les retailers avec des pipelines ML matures et la bande passante d'ingénierie pour les maintenir, LightGBM surpassera sur la majeure partie de l'assortiment. Mais pour les nouveaux produits qui arrivent chaque semaine — ceux pour lesquels les demand planners estiment actuellement à vue — Chronos-2 est le seul outil qui fonctionne dès le premier jour.

Ce n'est pas un avantage étroit. C'est le problème de prévision que le reste du pipeline est conçu pour éviter.

---

## Annexe : Chronologie des Expériences

| Date | Jalons | WRMSSE |
|------|--------|--------|
| Avr. 2026 | Baseline saisonnière naïve établie | 1,6310 (3 cutoffs geo) |
| Avr.–Mai 2026 | Premiers runs ZS ; sweep CL et segmentation tier (printemps) | 1,243 → 0,9082 |
| 14 Mai 2026 | v2 : premier Optuna HPO combinant FT et ZS | — |
| 18 Mai 2026 | v6 : FT HPO par tier, meilleur résultat FT printemps | 0,836 (printemps) |
| 18–29 Mai 2026 | v7 : FT sélectif (séries smooth uniquement) — impasse | — |
| 19–31 Mai 2026 | v8 : système de covariables cw2, éval 2 cutoffs — impasse | — |
| 8–10 Jun 2026 | v9/v10 : API Chronos directe, évaluation 3 cutoffs commence | 0,8947 → 0,8536 |
| 11 Jun 2026 | v11 : ensemble tier + re-opt matrice W (400 fev) | **0,8180** (tier seul) |
| 11–12 Jun 2026 | v12/v13 : ZS HPO dept-seg xcl=False + OLS tier | 0,8066 |
| 13 Jun 2026 | v14 : ZS par magasin — pire que dept, impasse | 0,8132 |
| 13–15 Jun 2026 | v15–v22 : LoRA FT (invalide — bug PeftModel) | — |
| 14–15 Jun 2026 | Courbe catastrophic forgetting ; bug PeftModel découvert | — |
| 15 Jun 2026 | v17 : modèle BASE standalone ZS | 1,1963 |
| 15 Jun 2026 | **v20 : dept-seg xcl=True + OLS tier — MEILLEUR CHRONOS** | **0,7969** |
| 16 Jun 2026 | Survey ZS global : interaction xcl×covariable confirmée | — |
| 15–16 Jun 2026 | v23/v24a : LoRA FT après correction bug | 0,9509 / 0,9461 |
| 22 Jun 2026 | LightGBM global (37 features, 19 trials Optuna HPO) | **0,6198** |
| 23 Jun 2026 | Blend OLS LGBM × Chronos (LOO-3) | 0,6254 (aucun gain) |
| 23 Jun 2026 | Segmentation naturelle cold-start (RMSSE par série par bucket d'historique) | Chronos gagne <28j |
