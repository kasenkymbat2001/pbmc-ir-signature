#!/usr/bin/env Rscript
# Pseudobulk a Seurat / SingleCellExperiment .rds directly in R, writing the same files as
# `irsig pseudobulk`. Avoids converting very large objects (e.g. multi-GB Seurat .rds) to h5ad.
#
# Usage:
#   Rscript scripts/seurat_pseudobulk.R rds=data/scPD.rds donor_col=donor celltype_col=cell_type \
#       out=results/pd_pseudobulk [min_cells=20] [detect_frac=0.5]
#
# Needs: Matrix; plus SeuratObject (for Seurat input) or SummarizedExperiment (for SCE input).
# A plain list(counts = <genes x cells dgCMatrix>, meta = <data.frame>) also works.

suppressPackageStartupMessages(library(Matrix))

args <- commandArgs(trailingOnly = TRUE)
kv <- setNames(sub("^[^=]+=", "", args), sub("=.*$", "", args))
need <- c("rds", "donor_col", "celltype_col", "out")
miss <- setdiff(need, names(kv))
if (length(miss)) stop("Missing arguments: ", paste(miss, collapse = ", "))
min_cells <- as.integer(ifelse(is.na(kv["min_cells"]), 20, kv["min_cells"]))
detect_frac <- as.numeric(ifelse(is.na(kv["detect_frac"]), 0.5, kv["detect_frac"]))
dir.create(kv[["out"]], recursive = TRUE, showWarnings = FALSE)

message("Reading ", kv[["rds"]], " (large objects need a lot of RAM)...")
obj <- readRDS(kv[["rds"]])

if (inherits(obj, "Seurat")) {
  suppressPackageStartupMessages(library(SeuratObject))
  assay <- DefaultAssay(obj)
  layers <- tryCatch(Layers(obj[[assay]]), error = function(e) character(0))
  if (sum(grepl("^counts", layers)) > 1) obj[[assay]] <- JoinLayers(obj[[assay]])  # Seurat v5 split layers
  counts <- tryCatch(LayerData(obj, assay = assay, layer = "counts"),
                     error = function(e) GetAssayData(obj, assay = assay, slot = "counts"))
  meta <- obj@meta.data
} else if (inherits(obj, "SingleCellExperiment")) {
  counts <- SummarizedExperiment::assay(obj, "counts")
  meta <- as.data.frame(SummarizedExperiment::colData(obj))
} else if (is.list(obj) && all(c("counts", "meta") %in% names(obj))) {
  counts <- obj$counts
  meta <- obj$meta
} else {
  stop("Unsupported object class: ", paste(class(obj), collapse = "/"))
}
rm(obj); invisible(gc())
counts <- as(counts, "CsparseMatrix")
if (any(abs(counts@x[seq_len(min(1e4, length(counts@x)))] - round(counts@x[seq_len(min(1e4, length(counts@x)))])) > 1e-6))
  stop("counts are not integers - pass raw UMI counts")

donor <- as.character(meta[[kv[["donor_col"]]]])
ct <- as.character(meta[[kv[["celltype_col"]]]])
if (is.null(donor) || is.null(ct) || length(donor) != ncol(counts)) stop("donor/cell-type columns not found")

aggregate_by <- function(key) {
  f <- factor(key)
  ind <- sparseMatrix(i = seq_along(f), j = as.integer(f), x = 1,
                      dims = c(length(f), nlevels(f)), dimnames = list(NULL, levels(f)))
  list(sum = counts %*% ind, n = as.vector(table(f)[levels(f)]))
}
by_ct <- aggregate_by(paste(donor, ct, sep = "|"))
by_all <- aggregate_by(donor)
colnames(by_all$sum) <- paste(colnames(by_all$sum), "ALL_PBMC", sep = "|")

pb <- cbind(by_ct$sum, by_all$sum)
n_cells <- c(by_ct$n, by_all$n)
keep <- n_cells >= min_cells
pb <- pb[, keep, drop = FALSE]
parts <- do.call(rbind, strsplit(colnames(pb), "|", fixed = TRUE))
pb_meta <- data.frame(sample_id = colnames(pb), donor = parts[, 1], cell_type = parts[, 2],
                      n_cells = n_cells[keep], stringsAsFactors = FALSE)

# donor-level metadata = columns constant within every donor
other <- setdiff(colnames(meta), c(kv[["donor_col"]], kv[["celltype_col"]]))
const <- other[vapply(other, function(cn) all(tapply(meta[[cn]], donor, function(v) length(unique(v)) <= 1)), logical(1))]
if (length(const)) {
  dm <- meta[!duplicated(donor), const, drop = FALSE]
  dm$donor <- donor[!duplicated(donor)]
  pb_meta <- merge(pb_meta, dm, by = "donor", all.x = TRUE, sort = FALSE)
  pb_meta <- pb_meta[match(colnames(pb), pb_meta$sample_id), c("sample_id", setdiff(names(pb_meta), "sample_id"))]
}

out <- kv[["out"]]
dense <- as.matrix(pb)
write.csv(data.frame(gene = rownames(dense), dense, check.names = FALSE),
          file.path(out, "pseudobulk_counts.csv"), row.names = FALSE, quote = FALSE)
write.csv(pb_meta, file.path(out, "pseudobulk_metadata.csv"), row.names = FALSE)
props <- prop.table(table(donor = donor, cell_type = ct), 1)
pw <- as.data.frame.matrix(props)
write.csv(data.frame(donor = rownames(pw), pw, check.names = FALSE),
          file.path(out, "cell_type_proportions.csv"), row.names = FALSE)
all_cols <- grepl("\\|ALL_PBMC$", colnames(pb))
detected <- sort(rownames(pb)[rowMeans(pb[, all_cols, drop = FALSE] >= 1) >= detect_frac])
writeLines(detected, file.path(out, "detected_genes.txt"))
message(ncol(pb), " pseudobulk samples, ", length(detected), " detected genes -> ", out)
