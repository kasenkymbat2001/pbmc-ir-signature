# Running on real data

Nothing here is downloaded automatically. Put raw files under `data/` (it is git-ignored) and check every dataset's licence and metadata before use.

## 1. Training cohort: GSE87005 (bulk PBMC, IR vs insulin-sensitive)

1. From the GEO page of GSE87005, download `GSE87005_series_matrix.txt.gz` and the platform annotation table (the GPL linked on the page, *Download full table*).
2. Look at the platform table header to find the probe-ID and gene-symbol column names.
3. Convert:

```bash
irsig prepare-geo --series-matrix data/GSE87005_series_matrix.txt.gz \
  --platform-table data/GPLxxxx.txt --id-col ID --symbol-col GENE_SYMBOL \
  --dataset GSE87005 --out data/processed/GSE87005
```

4. The command prints the metadata columns parsed from `!Sample_characteristics_ch1`. Open `metadata.csv` and find the column that encodes IR vs insulin-sensitive status, and any BMI/age/sex columns.

**Second cohort for LODO.** Search GEO for other PBMC or whole-blood cohorts with per-sample HOMA-IR or clamp-based labels. Process each the same way with its own `--dataset` name. For a continuous HOMA-IR, define classes in a pre-registered way (e.g. cohort-specific top vs bottom tertile) and note this in the results. With only one cohort, leave out `--lodo`, and the pipeline uses repeated donor-grouped K-fold.

## 2. Target cohort: PD PBMC single-cell

Option A, Seurat `.rds` (e.g. Zenodo 14372434, ~14 GB; needs a machine with plenty of RAM):

```r
# inspect first: which columns hold donor, diagnosis, cell type, age, sex?
obj <- readRDS("data/scPD_AllCells_Annotated.rds"); head(obj@meta.data)
```

```bash
Rscript scripts/seurat_pseudobulk.R rds=data/scPD_AllCells_Annotated.rds \
  donor_col=<donor column> celltype_col=<cell-type column> out=data/processed/pd_pseudobulk
```

Option B, `.h5ad` with raw counts:

```bash
irsig pseudobulk --h5ad data/pd_pbmc.h5ad --donor-col donor --celltype-col cell_type \
  --layer counts --out data/processed/pd_pseudobulk
```

Keep only PD and control donors in the association step (`--case/--control`). MSA/PSP donors can be analysed separately.

## 3. Train, restricted to genes detected in the PD data

```bash
irsig train \
  --expr data/processed/GSE87005/expression.csv \
  --meta data/processed/GSE87005/metadata.csv \
  --label-col <ir column> --positive <IR value> --negative <IS value> \
  --restrict-genes data/processed/pd_pseudobulk/detected_genes.txt \
  --covariates <bmi column> \
  --outer-folds 5 --repeats 20 --n-perm 500 --k-grid 10,25,50 --c-grid 0.05,0.2,1 \
  --out results/train_GSE87005
```

With 20 donors, keep the grids small. Report the primary AUROC together with its CI and the permutation p, not just the point estimate.

## 4. Apply to PD vs controls

```bash
irsig apply --model results/train_GSE87005/model.joblib \
  --expr data/processed/pd_pseudobulk/pseudobulk_counts.csv \
  --meta data/processed/pd_pseudobulk/pseudobulk_metadata.csv \
  --proportions data/processed/pd_pseudobulk/cell_type_proportions.csv \
  --group-col <diagnosis column> --case <PD value> --control <control value> \
  --covariates <age column> <sex column> prop_<monocyte label> \
  --out results/apply_PD
```

`association_tests.csv` gives, per cell type, the adjusted PD-vs-control difference in SD units with a 95 % CI, the OLS and Mann-Whitney p-values, and the BH-FDR. Add `--severity-col` for a within-PD Spearman correlation with e.g. UPDRS-III or MoCA, if available.
