# SEAGLMI benchmark: shortcut-controlled evaluation of lncRNA–miRNA interaction prediction

This repository accompanies the article *Reassessing lncRNA–miRNA interaction prediction under shortcut-controlled evaluation* (Hou, Kone and Wang). It provides the benchmark partitions, both negative-sampling protocols, the source ledger of the positives, all evaluation results, the SEAGLMI model definition and the code that regenerates the reported tables and figures.

## Benchmark at a glance

- **Positives**: 1,525 lncRNA–miRNA pairs (454 lncRNAs, 535 miRNAs) with low-throughput database support (NPInter v3–v5, ENCORI) that pass bibliographic and identity screening. miRNAs follow miRBase 22.1; each lncRNA is represented by its GENCODE v50 Ensembl_canonical transcript.
- **Settings**: warm, blind-lncRNA (lncRNA clusters at 50% identity), blind-miRNA (seed families) and blind-both, each with five outer folds and an inner validation fold.
- **Negative-sampling protocols**
  - *standard*: uniformly sampled lncRNA–miRNA pairs;
  - *strict*: each negative keeps one endpoint of a positive and redraws the other in proportion to positive endpoint frequencies, so that popular RNAs appear about as often among negatives as among positives.
- **Source-held-out evaluation**: positives are grouped into citation components (PubMed identifiers linked through shared pairs) and whole components are held out; five fixed negative draws per fold and ratio.
- **Ratios**: training uses 1:2 negatives, inner validation 1:1, outer tests 1:1 and 1:10.

## Repository layout

```
benchmark/
  positives_ledger.tsv            one row per positive: miRNA, lncRNA, canonical transcript, PubMed sources,
                                  citation component, source-held-out fold and outer test fold per setting
  lncRNA_transcripts.tsv          canonical transcript of each of the 454 lncRNAs
  citation_component_summary.json component sizes and counts
  candidate_pool_summary.tsv      rejection rates, realized ratios and endpoint balance of the test pools
  upstream_methods.json           repositories and commits of the evaluated published methods
  splits/
    main/{strict,standard}/{warm,blind_lncRNA,blind_miRNA,blind_both}/fold{0-4}/
        train.tsv  val.tsv  test_1to1.tsv  test_1to10.tsv
    source_held_out/{strict,standard}/fold{0-4}/
        train.tsv  val.tsv  test_1to1_draw{0-4}.tsv  test_1to10_draw{0-4}.tsv
results/                          verified evaluation results (see below)
  corrected_tests/                400 Nadeau–Bengio corrected tests with Holm adjustment
src/
  seaglmi_model.py                SEAGLMI network definition (PyTorch)
  deterministic_pool.py           deterministic adaptive max pooling used for reproducible training
  negative_sampling.py            standard and strict negative sampling
scripts/
  rebuild_tables_figures.py       regenerates result tables and figures into generated/
generated/                        output of rebuild_tables_figures.py
```

Every split file is tab-separated with columns `mirna_node` (miRBase accession), `ensg` (Ensembl gene identifier), `label` (1 positive, 0 negative) and `arm_inferred`.

## Results

File prefixes denote the three evaluations: `A_` main benchmark, `B_` matched-sampling controls (training-negative choices), `C1_` source-held-out evaluation.

| File | Content |
|---|---|
| `A_metrics_by_fold.tsv` | AUROC, AP, MCC, precision, recall and F1 for every method, protocol, setting, fold and ratio, with realized ratios and selected thresholds |
| `A_summary.tsv` | five-fold means and sample SDs |
| `B_*` | matched-control results and the 640 prespecified paired contrasts |
| `C1_*` | source-held-out results and the 2,000-replicate source-bootstrap intervals |
| `all980_outer_unit_selected_configuration_checkpoint_threshold.tsv` | selected configuration, checkpoint and threshold of every outer unit |
| `fold_training_validation_outer_sizes.tsv`, `outer_fold_draw_sample_sizes.tsv` | realized sample sizes |
| `corrected_tests/` | analysis plan, paired fold differences and all 400 corrected tests |

Method keys: `seaglmi` (SEAGLMI), `sitegbdt` (SEAGLMI-GBDT), `pairfeat_gbdt` (PairFeat-GBDT), `seed_sites` (seed rule), `degree_prior`, `kmer_lr`, `deeplmi`, `spgnn`, `hetgat_lmi_v2` (HetGAT-LMI), `mpgk_lmi`, `tec_lncmir`, `premli_keras224_v2` (preMLI), `slnpm_correct` (SLNPM), `ndalma`.

## Reproducing the tables and figures

```bash
pip install -r requirements.txt
python scripts/rebuild_tables_figures.py
```

The script reads `results/` and writes LaTeX tables and figures to `generated/`; it performs no training or resampling. The figures use Times New Roman, which must be installed.

## Using the benchmark

Train on `train.tsv`, select configurations, epochs and thresholds on `val.tsv` only, and report results on the test files of both protocols. We recommend reporting a degree-only reference and a seed-site reference alongside any new method, and reading AP on sampled pools as a relative measure rather than an expected experimental hit rate.

## Third-party data and code

Interaction records, RNA sequences, expression data, pretrained embeddings (RNA-FM, rna2vec) and the code of the evaluated published methods remain under their original terms and are not redistributed here. They can be obtained from NPInter, ENCORI, miRBase, GENCODE, CCLE and the repositories listed in `benchmark/upstream_methods.json`.

## Citation

Please cite the accompanying article (citation details to be added on publication).

## License

To be added by the authors.
