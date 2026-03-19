# From Pixels to Personas: Tracking the Evolution of Anime Characters

This repository contains the official implementation for the paper **"From Pixels to Personas: Tracking the Evolution of Anime Characters"**.

**Authors:** Rongze Liu, Jiaxin Pei, Jian Zhu

## Abstract

Anime, originated from Japan, is one of the most influential cultural products in modern society and is especially popular among younger generations. The popularity of anime reflects important cultural evolutions in our society. Despite existing research on anime as a cultural phenomenon, we still have a limited understanding of how anime really evolves over the years. In this study, using a large-scale multimodal dataset of anime characters from an anime review site, we applied computational methods that integrate textual, visual, and production features of anime characters with online popularity traces. By combining LLM-extracted personality features with avatar features, we identify recurring personality archetypes and visual tropes with their temporal evolution over the past decades. We found that the target audience of anime has undergone a systematic shift from children to a maturing audience of teenagers and young adults over time. Character design has been undergoing moe-ification, with softer or sexualized female traits becoming increasingly prominent since the 2000s. Some personality archetypes are often visually predictable, yet audiences also tend to prefer less conventionalized characters. Finally, we reveal that visual signals play a more dominant role than personality traits in shaping audience preferences, with features such as moe-style faces and mechanical designs contributing greatly to popularity. These findings offer insights into the broader dynamics of anime's cultural and creative practices.

## Project Structure

The code is organized into modular scripts for each stage of the research pipeline:

- data collection and preprocessing
- LLM annotation pipeline
- visual and personality feature extraction
- dimensionality reduction
- community detection
- temporal classification
- personality archetype prediction
- popularity modeling
- figure plotting

```text
├── scripts/
    ├── data_collection/                             <- Data scraping and parsing
    │   ├── avatar_tagging.py
    │   ├── crawl_char_ind.py
    │   ├── crawl_characters.py
    │   ├── crawl_high_res_image.py
    │   ├── parse_mal.py
    │   ├── parse_mal_char.py
    │   ├── parse_ap.py
    │   ├── parse_ap_characters.py
    │   ├── extract_bio.py
    │   └── character_bio_extraction.py
    ├── llm_annotation/
    │   ├── personality_keywords_annotation.py       <- LLM extraction of personality keywords
    │   └── reannotate_failures.py                   <- Re-run LLM on failed parses
    ├── embedding/
    │   ├── extract_avatar_embeddings.py             <- Extract DINOv2 embeddings for avatars
    │   └── extract_personality_embeddings.py        <- Extract embedding for personality keywords
    ├── modeling/
    │   ├── pca_avatar_embeddings.py                 <- PCA on normalized visual embeddings
    │   ├── pca_personality_embeddings.py            <- PCA on normalized personality embeddings
    │   ├── umap_personality_embeddings.py           <- UMAP reduction for personality embeddings
    │   ├── personality_archetype_community_detection.py <- Leiden communities from personality UMAP coords
    │   ├── predict_popularity_from_pcs_metadata.py  <- Popularity modeling using PCs + metadata
    │   ├── feature_importance_shap_popularity.py    <- SHAP analysis for fitted popularity models
    │   ├── predict_archetype_from_visual_pcs__baseline.py <- Random/majority baselines for archetype prediction
    │   ├── predict_archetype_from_visual_pcs__xgb.py <- XGBoost archetype classifier from visual PCs
    │   ├── predict_archetype_from_visual_pcs__rf.py  <- Random Forest archetype classifier from visual PCs
    │   ├── finetune_dinov2_period.py                <- Fine-tune DINOv2 for period classification
    │   ├── finetune_dinov2_archetype.py             <- Fine-tune DINOv2 for archetype classification
    │   └── evaluate_frozen_dinov2_period.py         <- Frozen DINOv2 baseline for period classification
    ├── utils/
    │   ├── create_year_labeled_dataset.py           <- Build year-labeled CSV from metadata + images
    │   ├── standardize_anime_air_dates.py           <- Standardize year/period features
    │   ├── anime_character_mapping.py               <- Build anime-character mapping table
    │   ├── character_info_aggregation.py            <- Merge bio JSON + LLM keywords JSONL
    │   ├── compute_annotation_overlap.py            <- Overlap metrics between human vs model keyword sets
    │   ├── compute_inter_annotator_agreement.py     <- Inter-annotator agreement metrics
    │   ├── prepare_period_classification_dataset.py <- Create train/val/test sets for period classification task
    │   └── prepare_community_classification_dataset.py <- Create train/val/test sets for archetype classification task
    └── plotting/                                    <- Plotting scripts
        ├── anime_per_period.py
        ├── avg_characters_per_anime_by_role.py
        ├── anime_rating_source_trends.py
        ├── personality_temporal_trends.py
        ├── personality_communities_scatter.py
        ├── community_vs_anime_metadata.py
        ├── character_popularity_plots.py
        ├── pca_means_by_period_grid.py
        ├── pc_image_examples.py
        └── confusion_matrices.py
├── requirements.txt
└── README.md
```
